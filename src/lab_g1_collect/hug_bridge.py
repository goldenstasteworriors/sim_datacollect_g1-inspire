"""Filesystem bridge between Isaac's environment and the official HUG environment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from .retarget import load_hug_prediction


def run_hug_capture(
    *, project: Path, episode_index: int, rgb: np.ndarray, depth_m: np.ndarray,
    K: np.ndarray, point_uv_224: tuple[float, float], sampling_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    run_dir = project / "outputs" / "hug_runtime" / f"episode_{episode_index:06d}"
    dataset = run_dir / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    capture = run_dir / "capture.npz"
    depth_mm = np.clip(np.nan_to_num(depth_m) * 1000.0, 0, 65534).astype(np.uint16)
    np.savez_compressed(
        capture, rgb=np.asarray(rgb, dtype=np.uint8), depth_mm=depth_mm,
        K=np.asarray(K, dtype=np.float64), u_224=np.float32(point_uv_224[0]),
        v_224=np.float32(point_uv_224[1]),
    )
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
    ]
    subprocess.run(command, cwd=project, env=env, check=True, timeout=180)
    return load_hug_prediction(dataset / "grasp_pred" / "sim_capture.pkl")
