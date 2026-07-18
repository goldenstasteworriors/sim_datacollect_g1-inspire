"""Filesystem bridge between Isaac's environment and the official HUG environment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .retarget import load_hug_geometry, load_hug_prediction


HUG_MAX_DEPTH_MM = 3000


def _save_hug_input_debug(
    run_dir: Path, rgb: np.ndarray, depth_mm: np.ndarray, K: np.ndarray,
    point_uv_224: tuple[float, float], object_name: str,
) -> None:
    """保存 HUG 原始输入、224 输入尺度预览和便于检查的元数据。"""
    height, width = rgb.shape[:2]
    square = min(height, width)
    x_offset = (width - square) // 2
    y_offset = (height - square) // 2
    rgb_224 = Image.fromarray(rgb).crop(
        (x_offset, y_offset, x_offset + square, y_offset + square)
    ).resize((224, 224), Image.Resampling.LANCZOS)

    valid_mask = (depth_mm > 0) & (depth_mm < HUG_MAX_DEPTH_MM)
    valid_depth = depth_mm[valid_mask]
    if valid_depth.size:
        low, high = np.percentile(valid_depth, (2, 98))
        high = max(float(high), float(low) + 1.0)
        normalized = np.clip((depth_mm.astype(np.float32) - low) / (high - low), 0, 1)
    else:
        low, high = 0.0, 1.0
        normalized = np.zeros_like(depth_mm, dtype=np.float32)
    depth_rgb = np.stack(
        [255 * normalized, 255 * (1.0 - np.abs(2.0 * normalized - 1.0)),
         255 * (1.0 - normalized)], axis=-1,
    ).astype(np.uint8)
    depth_rgb[~valid_mask] = 0
    depth_224 = Image.fromarray(depth_rgb).crop(
        (x_offset, y_offset, x_offset + square, y_offset + square)
    ).resize((224, 224), Image.Resampling.NEAREST)

    conditioned = rgb_224.copy()
    draw = ImageDraw.Draw(conditioned)
    u, v = point_uv_224
    radius = 14
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline="red", width=3)
    draw.line((u - 20, v, u + 20, v), fill="yellow", width=2)
    draw.line((u, v - 20, u, v + 20), fill="yellow", width=2)

    preview = Image.new("RGB", (224 * 3, 252), "white")
    preview.paste(rgb_224, (0, 28))
    preview.paste(depth_224, (224, 28))
    preview.paste(conditioned, (448, 28))
    labels = ImageDraw.Draw(preview)
    labels.text((6, 7), "HUG RGB 224x224", fill="black")
    labels.text((230, 7), f"Depth <3m ({low:.0f}-{high:.0f} mm)", fill="black")
    labels.text((454, 7), f"{object_name} point ({u:.1f}, {v:.1f})", fill="black")

    Image.fromarray(rgb).save(run_dir / "rgb.png")
    Image.fromarray(depth_mm, mode="I;16").save(run_dir / "depth_mm.png")
    preview.save(run_dir / "hug_input_preview.png")
    metadata = {
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth_mm.shape),
        "depth_unit": "millimeter",
        "visualized_depth_range_mm": [1, HUG_MAX_DEPTH_MM],
        "K_original": np.asarray(K, dtype=float).tolist(),
        "center_crop": {"x_offset": x_offset, "y_offset": y_offset, "size": square},
        "hug_size": [224, 224],
        "condition_point_uv_224": [float(u), float(v)],
        "condition_radius_px": radius,
        "object_name": object_name,
    }
    (run_dir / "input_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _save_hug_output_debug(
    run_dir: Path, rgb: np.ndarray, K: np.ndarray, wrist: np.ndarray,
    landmarks: np.ndarray,
) -> None:
    """将 HUG 原始相机系 MANO 手骨架和腕部 XYZ 位姿投影到 224 RGB。"""
    height, width = rgb.shape[:2]
    square = min(height, width)
    x_offset = (width - square) // 2
    y_offset = (height - square) // 2
    scale = 224.0 / square
    K_224 = np.asarray(K, dtype=np.float64).copy()
    K_224[0, 2] -= x_offset
    K_224[1, 2] -= y_offset
    K_224[:2, :] *= scale
    canvas = Image.fromarray(rgb).crop(
        (x_offset, y_offset, x_offset + square, y_offset + square)
    ).resize((224, 224), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)

    def project(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        pixels_h = (K_224 @ points.T).T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        pixels[points[:, 2] <= 0] = np.nan
        return pixels

    hand_uv = project(landmarks)
    finger_chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                     (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    for chain in finger_chains:
        points = [tuple(hand_uv[index]) for index in chain if np.isfinite(hand_uv[index]).all()]
        if len(points) >= 2:
            draw.line(points, fill="white", width=2)
    for u, v in hand_uv:
        if np.isfinite((u, v)).all():
            draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill="white")

    axis_length_m = 0.08
    origin = wrist[:3, 3]
    axis_points = np.vstack((origin, origin + wrist[:3, :3].T * axis_length_m))
    axis_uv = project(axis_points)
    axis_colors = ("red", "lime", "blue")
    for endpoint, color in zip(axis_uv[1:], axis_colors):
        if np.isfinite(axis_uv[0]).all() and np.isfinite(endpoint).all():
            draw.line((tuple(axis_uv[0]), tuple(endpoint)), fill=color, width=4)
    if np.isfinite(axis_uv[0]).all():
        u, v = axis_uv[0]
        draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline="yellow", width=2)

    labeled = Image.new("RGB", (224, 248), "white")
    labeled.paste(canvas, (0, 24))
    ImageDraw.Draw(labeled).text((5, 5), "HUG wrist: X red / Y green / Z blue", fill="black")
    labeled.save(run_dir / "hug_output_wrist_pose.png")
    output_metadata = {
        "T_camera_wrist": np.asarray(wrist, dtype=float).tolist(),
        "wrist_origin_uv_224": axis_uv[0].tolist(),
        "axis_length_m": axis_length_m,
        "landmarks_3d_camera": np.asarray(landmarks, dtype=float).tolist(),
    }
    (run_dir / "hug_output_metadata.json").write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_hug_capture(
    *, project: Path, episode_index: int, rgb: np.ndarray, depth_m: np.ndarray,
    K: np.ndarray, point_uv_224: tuple[float, float], object_name: str = "beaker",
    sampling_steps: int = 5, candidates: int = 8, debug_stride: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    run_dir = project / "outputs" / "hug_runtime" / f"episode_{episode_index:06d}"
    dataset = run_dir / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    capture = run_dir / "capture.npz"
    depth_m = np.asarray(depth_m)
    if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
        depth_m = depth_m[..., 0]
    if depth_m.ndim != 2:
        raise ValueError(f"HUG depth 应为 HxW 或 HxWx1，实际为 {depth_m.shape}")
    depth_mm = np.clip(np.nan_to_num(depth_m) * 1000.0, 0, 65534).astype(np.uint16)
    rgb = np.asarray(rgb, dtype=np.uint8)
    K = np.asarray(K, dtype=np.float64)
    np.savez_compressed(
        capture, rgb=rgb, depth_mm=depth_mm,
        K=K, u_224=np.float32(point_uv_224[0]),
        v_224=np.float32(point_uv_224[1]), object_name=np.asarray(object_name),
    )
    save_debug = debug_stride > 0 and episode_index % debug_stride == 0
    if save_debug:
        _save_hug_input_debug(run_dir, rgb, depth_mm, K, point_uv_224, object_name)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{project / 'src'}:{project / 'third_party/hug'}",
        "HF_HOME": str(project / ".cache/huggingface"),
        "TMPDIR": str(project / ".cache/tmp"),
        "HTTP_PROXY": "http://127.0.0.1:7897",
        "HTTPS_PROXY": "http://127.0.0.1:7897",
        "http_proxy": "http://127.0.0.1:7897",
        "https_proxy": "http://127.0.0.1:7897",
    })
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    command = [
        "/home/ykj/miniconda3/bin/conda", "run", "--no-capture-output", "-n", "hug",
        "python", "-m", "lab_g1_collect.hug_runtime", str(capture), str(dataset),
        str(project / "checkpoints/hug/hug_full.safetensors"),
        "--sampling-steps", str(sampling_steps),
        "--candidates", str(candidates),
    ]
    subprocess.run(command, cwd=project, env=env, check=True, timeout=180)
    prediction_path = dataset / "grasp_pred" / "sim_capture.pkl"
    wrist, landmarks = load_hug_geometry(prediction_path)
    if save_debug:
        _save_hug_output_debug(run_dir, rgb, K, wrist, landmarks)
    return load_hug_prediction(prediction_path)
