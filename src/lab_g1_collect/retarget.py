from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """Load NumPy 2 pickles in Isaac environments that still pin NumPy 1.x."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _chain_flexion(points: np.ndarray, chain: tuple[int, ...]) -> float:
    segments = np.diff(points[list(chain)], axis=0)
    norms = np.linalg.norm(segments, axis=1, keepdims=True)
    segments = segments / np.maximum(norms, 1e-8)
    dots = np.sum(segments[:-1] * segments[1:], axis=1)
    bends = np.arccos(np.clip(dots, -1.0, 1.0))
    return float(np.clip(bends.sum() / (0.85 * np.pi), 0.0, 1.0))


def mano_landmarks_to_inspire(landmarks_3d: np.ndarray) -> np.ndarray:
    """将 HUG/MANO 21 点右手映射为 RH56DFTP 的 6 路归一化命令。

    输出顺序与 Unitree 官方 DFX 服务一致：小指、无名指、中指、食指、
    拇指弯曲、拇指旋转。0 为张开，1 为闭合。
    """
    points = np.asarray(landmarks_3d, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError(f"landmarks_3d 应为有限值 (21, 3)，实际为 {points.shape}")
    flex = {name: _chain_flexion(points, chain) for name, chain in FINGER_CHAINS.items()}

    palm_axis = points[5] - points[17]
    thumb_axis = points[4] - points[2]
    palm_axis /= max(np.linalg.norm(palm_axis), 1e-8)
    thumb_axis /= max(np.linalg.norm(thumb_axis), 1e-8)
    # 对掌程度越高，拇指与掌横轴越接近反向。
    thumb_rotation = float(np.clip((1.0 - np.dot(palm_axis, thumb_axis)) * 0.5, 0.0, 1.0))
    return np.array(
        [flex["pinky"], flex["ring"], flex["middle"], flex["index"], flex["thumb"], thumb_rotation],
        dtype=np.float32,
    )


def load_hug_prediction(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 HUG 保存的 grasp_pred pkl，返回腕部位姿和 Inspire 命令。"""
    with Path(path).open("rb") as handle:
        data = _NumpyCompatibleUnpickler(handle).load()
    grasp = data["grasp"]
    if isinstance(grasp, dict):
        landmarks = np.asarray(grasp["landmarks_3d"])
        wrist = np.asarray(grasp["T_camera_wrist"])
    else:
        landmarks = np.asarray(grasp.landmarks_3d)
        wrist = np.asarray(grasp.T_camera_wrist)
    if wrist.shape != (4, 4):
        raise ValueError(f"T_camera_wrist 应为 (4, 4)，实际为 {wrist.shape}")
    return wrist.astype(np.float32), mano_landmarks_to_inspire(landmarks)
