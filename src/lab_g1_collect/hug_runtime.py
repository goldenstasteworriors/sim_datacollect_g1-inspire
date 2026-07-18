"""Run the official HUG model for one simulator RGB-D capture."""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

from .hug_condition import add_condition_point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-base-distance", type=float, default=0.12)
    args = parser.parse_args()

    from hug.inference import main as infer
    from hug.prepare_inputs import prepare_pkl
    from hug.utils.pcl_utils import pixel_to_xyz
    from lab_g1_collect.retarget import T_MANO_WRIST_INSPIRE_BASE, load_hug_geometry

    capture = np.load(args.capture)
    object_name = str(capture["object_name"]) if "object_name" in capture else "beaker"
    pkl_path = prepare_pkl(
        capture["rgb"].astype(np.uint8),
        capture["depth_mm"].astype(np.uint16),
        capture["K"].astype(np.float64),
        "sim_capture",
        args.dataset,
        object_name=object_name,
    )
    add_condition_point(pkl_path, float(capture["u_224"]), float(capture["v_224"]), radius=14)
    candidate_count = max(1, args.candidates)
    candidate_names = []
    for index in range(candidate_count):
        name = f"candidate_{index:03d}"
        candidate_path = args.dataset / f"{name}.pkl"
        shutil.copyfile(pkl_path, candidate_path)
        candidate_names.append(name)
    candidate_list = args.dataset / "candidate_samples.txt"
    candidate_list.write_text("\n".join(candidate_names) + "\n", encoding="utf-8")
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    infer(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        sample_name=str(candidate_list),
        num_samples=None,
        batch_size=candidate_count,
        sampling_steps=args.sampling_steps,
    )

    with pkl_path.open("rb") as handle:
        prepared = pickle.load(handle)
    depth_array = cv2.imdecode(np.frombuffer(prepared["depth"], np.uint8), cv2.IMREAD_UNCHANGED)
    u, v = (float(value) for value in prepared["condition_point"])
    ui = int(np.clip(round(u), 0, depth_array.shape[1] - 1))
    vi = int(np.clip(round(v), 0, depth_array.shape[0] - 1))
    depth_m = float(depth_array[vi, ui]) / 1000.0
    object_xyz = pixel_to_xyz(u, v, depth_m, np.asarray(prepared["camera"]["K"]))
    scored = []
    for name in candidate_names:
        prediction = args.dataset / "grasp_pred" / f"{name}.pkl"
        wrist, _ = load_hug_geometry(prediction)
        base_xyz = (wrist @ T_MANO_WRIST_INSPIRE_BASE)[:3, 3]
        distance = float(np.linalg.norm(base_xyz - object_xyz))
        scored.append((abs(distance - args.target_base_distance), distance, prediction))
    _, selected_distance, selected_path = min(scored, key=lambda item: item[0])
    shutil.copyfile(selected_path, args.dataset / "grasp_pred" / "sim_capture.pkl")
    print(
        f"[HUG candidates] selected={selected_path.stem} count={candidate_count} "
        f"base_object_distance_m={selected_distance:.4f}", flush=True,
    )


if __name__ == "__main__":
    main()
