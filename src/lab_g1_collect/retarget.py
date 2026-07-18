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

# MANO wrist frame -> Inspire right-hand mounting frame.  Keep this calibration
# explicit: HUG predicts a human wrist frame, while arm IK controls the robot's
# right_hand_base_link.  Values are wxyz and metres and can be replaced after a
# physical hand-eye calibration without changing the inference/control code.
MANO_WRIST_TO_INSPIRE_TRANSLATION_M = np.array([0.0, 0.0, 0.025], dtype=np.float32)
MANO_WRIST_TO_INSPIRE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
INSPIRE_TO_MANO_PALM_LENGTH_RATIO = 1.45


def _matrix_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def mano_wrist_to_inspire_pose(wrist: np.ndarray, landmarks_3d: np.ndarray | None = None) -> np.ndarray:
    """将 HUG/MANO 腕位姿变换为 Inspire 手掌安装基座位姿。

    有21点骨架时，直接用掌面几何重建朝向，避免把 MANO 内部坐标轴误当作
    Inspire 安装轴。Inspire 四指沿本体 -Y 伸展，食指到小指方向对应 -Z。
    """
    wrist = np.asarray(wrist, dtype=np.float32)
    if wrist.shape != (4, 4) or not np.isfinite(wrist).all():
        raise ValueError(f"T_camera_wrist 应为有限值 (4, 4)，实际为 {wrist.shape}")
    if landmarks_3d is not None:
        points = np.asarray(landmarks_3d, dtype=np.float32)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError(f"landmarks_3d 应为有限值 (21, 3)，实际为 {points.shape}")
        origin = points[0]
        mcp_center = points[[5, 9, 13, 17]].mean(axis=0)
        finger_forward = mcp_center - origin
        finger_forward /= max(np.linalg.norm(finger_forward), 1e-8)
        index_to_pinky = points[17] - points[5]
        index_to_pinky -= finger_forward * np.dot(index_to_pinky, finger_forward)
        index_to_pinky /= max(np.linalg.norm(index_to_pinky), 1e-8)
        x_axis = np.cross(index_to_pinky, finger_forward)
        x_axis /= max(np.linalg.norm(x_axis), 1e-8)
        y_axis = -finger_forward
        z_axis = index_to_pinky
        result = np.eye(4, dtype=np.float32)
        result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
        mano_palm_vector = mcp_center - origin
        result[:3, 3] = mcp_center - INSPIRE_TO_MANO_PALM_LENGTH_RATIO * mano_palm_vector
        return result

    wrist_to_inspire = np.eye(4, dtype=np.float32)
    wrist_to_inspire[:3, :3] = _matrix_from_quat_wxyz(MANO_WRIST_TO_INSPIRE_QUAT_WXYZ)
    wrist_to_inspire[:3, 3] = MANO_WRIST_TO_INSPIRE_TRANSLATION_M
    return wrist @ wrist_to_inspire


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
    wrist, landmarks = load_hug_geometry(path)
    return mano_wrist_to_inspire_pose(wrist, landmarks), mano_landmarks_to_inspire(landmarks)


def load_hug_geometry(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 HUG 原始相机系腕部位姿和 MANO 21 点，不进行机器人重定向。"""
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
    return wrist, landmarks
