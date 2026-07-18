from __future__ import annotations

import pickle
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


# MANO wrist frame -> Inspire right-hand mounting frame.  Keep this calibration
# explicit: HUG predicts a human wrist frame, while arm IK controls the robot's
# right_hand_base_link.  Values are wxyz and metres and can be replaced after a
# physical hand-eye calibration without changing the inference/control code.


@lru_cache(maxsize=1)
def _mano_wrist_to_inspire_base() -> np.ndarray:
    path = Path(__file__).resolve().parents[2] / "configs/mano_inspire_calibration.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    transform = np.asarray(data["T_mano_wrist_inspire_base"], dtype=np.float32)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"MANO-Inspire 标定矩阵无效: {path}")
    return transform

# Official Unitree xr_teleoperate Inspire configuration. HUG/MANO uses 21
# landmarks, while Unitree's XR stream uses 25 slots with fingertips at
# 4/9/14/19/24. Only wrist and fingertips are consumed by DexPilot.
MANO_TIPS = (4, 8, 12, 16, 20)
XR_TIPS = (4, 9, 14, 19, 24)
INSPIRE_API_JOINT_NAMES = (
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
)


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
    return wrist @ _mano_wrist_to_inspire_base()


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """Load NumPy 2 pickles in Isaac environments that still pin NumPy 1.x."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


@lru_cache(maxsize=1)
def _official_inspire_retargeter():
    """Build Unitree's pinned DexPilot solver and hardware-order index map."""
    import yaml
    from dex_retargeting import RetargetingConfig

    project = Path(__file__).resolve().parents[2]
    assets = project / "third_party/xr_teleoperate/assets"
    config_path = assets / "inspire_hand/inspire_hand.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少 Unitree Inspire 重定向配置: {config_path}")
    RetargetingConfig.set_default_urdf_dir(assets)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["right"]
    retargeter = RetargetingConfig.from_dict(config).build()
    output_indices = np.array(
        [retargeter.joint_names.index(name) for name in INSPIRE_API_JOINT_NAMES], dtype=np.int64
    )
    return retargeter, output_indices


def mano_landmarks_to_inspire(landmarks_3d: np.ndarray) -> np.ndarray:
    """用 Unitree 官方 DexPilot 将 MANO 21 点映射为 Inspire 6 路弧度命令。

    输出顺序与 Unitree 官方 DFX/FTP 接口一致：小指、无名指、中指、食指、
    拇指弯曲、拇指旋转。输出已经是 URDF 关节弧度，不是 0~1 归一化值。
    """
    points = np.asarray(landmarks_3d, dtype=np.float32)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError(f"landmarks_3d 应为有限值 (21, 3)，实际为 {points.shape}")
    xr_points = np.repeat(points[0:1], 25, axis=0)
    xr_points[0] = points[0]
    xr_points[list(XR_TIPS)] = points[list(MANO_TIPS)]
    retargeter, output_indices = _official_inspire_retargeter()
    human_indices = retargeter.optimizer.target_link_human_indices
    reference_vectors = xr_points[human_indices[1]] - xr_points[human_indices[0]]
    # Each HUG grasp is independent; do not leak the low-pass/filter state from
    # a previous episode into the next episode's grasp solution.
    retargeter.reset()
    robot_qpos = retargeter.retarget(reference_vectors)
    command = np.asarray(robot_qpos[output_indices], dtype=np.float32)
    lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1], dtype=np.float32)
    upper = np.array([1.7, 1.7, 1.7, 1.7, 0.5, 1.3], dtype=np.float32)
    return np.clip(command, lower, upper)


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
