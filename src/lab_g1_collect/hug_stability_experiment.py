"""Batch HUG repeatability experiments on simulator RGB-D captures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


T_MANO_WRIST_INSPIRE_BASE_TRANSLATION = torch.tensor(
    [0.0285, 0.0027, 0.0063], dtype=torch.float32
)


def _select_captures(root: Path, limit: int) -> list[tuple[str, Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for metadata_path in sorted(root.glob("episode_*/input_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset = metadata_path.parent / "dataset"
        if (dataset / "sim_capture.pkl").is_file():
            grouped[str(metadata.get("object_name", "unknown"))].append(dataset)
    selected: list[tuple[str, Path]] = []
    # Round-robin selection prevents the numerous cylinder captures from
    # crowding out the rarer beaker and box inputs.
    for sample_index in range(max((len(paths) for paths in grouped.values()), default=0)):
        for object_name in sorted(grouped):
            if sample_index < len(grouped[object_name]):
                selected.append((object_name, grouped[object_name][sample_index]))
                if len(selected) == limit:
                    return selected
    return selected


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_m": float(values.mean()),
        "std_m": float(values.std()),
        "min_m": float(values.min()),
        "p05_m": float(np.percentile(values, 5)),
        "p50_m": float(np.percentile(values, 50)),
        "p95_m": float(np.percentile(values, 95)),
        "max_m": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=Path("outputs/hug_runtime"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hug/hug_full.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hug_stability"))
    parser.add_argument("--captures", type=int, default=8)
    parser.add_argument("--samples-per-setting", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    from hug.dataloader.grasp_dataset import GraspDataset
    from hug.inference import load_model
    from hug.utils.transform_utils import six_d_to_rotation_matrix

    args.output.mkdir(parents=True, exist_ok=True)
    captures = _select_captures(args.capture_root, args.captures)
    if not captures:
        raise FileNotFoundError(f"未在 {args.capture_root} 找到 HUG simulator captures")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, use_ema=True, device=device)
    offset = T_MANO_WRIST_INSPIRE_BASE_TRANSLATION.to(device)
    rows: list[dict[str, float | int | str]] = []

    for capture_index, (object_name, dataset_path) in enumerate(captures):
        dataset = GraspDataset(
            str(dataset_path), split="val", use_rgb=model.use_rgb,
            use_depth=model.use_depth, pcl_crop_radius=model.pcl_crop_radius,
        )
        item = dataset[0]
        point_uv = item["point_uv"].unsqueeze(0).to(device)
        camera_K = item["camera_K"].unsqueeze(0).to(device)
        rgb = item.get("rgb")
        pcl_xyz = item.get("pcl_xyz")
        pcl_rgb = item.get("pcl_rgb")
        rgb = rgb.unsqueeze(0).to(device) if rgb is not None else None
        pcl_xyz = pcl_xyz.unsqueeze(0).to(device) if pcl_xyz is not None else None
        pcl_rgb = pcl_rgb.unsqueeze(0).to(device) if pcl_rgb is not None else None
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            condition = model.encode_scene(
                point_uv, camera_K, rgb=rgb, pcl_xyz=pcl_xyz, pcl_rgb=pcl_rgb
            )
        object_xyz = model._backproject(point_uv, camera_K)[0]

        for steps in args.sampling_steps:
            generated = 0
            while generated < args.samples_per_setting:
                count = min(args.batch_size, args.samples_per_setting - generated)
                # A stable per-batch seed makes the complete experiment exactly
                # reproducible without forcing every candidate to be identical.
                torch.manual_seed(args.seed + capture_index * 100_000 + steps * 1_000 + generated)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        args.seed + capture_index * 100_000 + steps * 1_000 + generated
                    )
                cond_batch = condition.expand(count, *condition.shape[1:])
                with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
                ):
                    samples = model.flow.sample(cond_batch, steps=steps)
                    translation, wrist_6d, _ = model.mano.decode_mano_params(samples)
                    wrist_rotation = six_d_to_rotation_matrix(wrist_6d)
                    base_xyz = translation + torch.matmul(
                        wrist_rotation, offset.view(1, 3, 1)
                    ).squeeze(-1)
                    distances = torch.linalg.vector_norm(base_xyz - object_xyz, dim=1)
                for local_index in range(count):
                    xyz = base_xyz[local_index].float().cpu().numpy()
                    rows.append({
                        "capture_index": capture_index,
                        "object_name": object_name,
                        "dataset": str(dataset_path),
                        "sampling_steps": steps,
                        "sample_index": generated + local_index,
                        "base_x_m": float(xyz[0]),
                        "base_y_m": float(xyz[1]),
                        "base_z_m": float(xyz[2]),
                        "base_object_distance_m": float(distances[local_index]),
                    })
                generated += count
            print(
                f"[{capture_index + 1}/{len(captures)}] {object_name} steps={steps} "
                f"samples={args.samples_per_setting}", flush=True,
            )

    csv_path = args.output / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped_rows: dict[tuple[str, int], list[float]] = defaultdict(list)
    capture_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[(str(row["object_name"]), int(row["sampling_steps"]))].append(
            float(row["base_object_distance_m"])
        )
        capture_rows[(int(row["capture_index"]), int(row["sampling_steps"]))].append(row)
    summary = {
        "experiment": {
            "seed": args.seed,
            "device": device,
            "captures": len(captures),
            "samples": len(rows),
            "samples_per_setting": args.samples_per_setting,
            "sampling_steps": args.sampling_steps,
            "point_cloud_policy": "one random 4096-point sample fixed per capture",
        },
        "by_object_and_steps": {
            f"{name}/steps_{steps}": _summary(np.asarray(values))
            for (name, steps), values in sorted(grouped_rows.items())
        },
        "same_input_spread": {},
    }
    for (capture_index, steps), group in sorted(capture_rows.items()):
        xyz = np.asarray([[row["base_x_m"], row["base_y_m"], row["base_z_m"]] for row in group])
        centroid = xyz.mean(axis=0)
        spread = np.linalg.norm(xyz - centroid, axis=1)
        summary["same_input_spread"][f"capture_{capture_index}/steps_{steps}"] = {
            **_summary(spread),
            "axis_std_m": xyz.std(axis=0).tolist(),
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["experiment"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
