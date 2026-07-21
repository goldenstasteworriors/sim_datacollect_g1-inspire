"""Shared G1 initialization pose derived from SONICMJ/GEAR-SONIC."""

from __future__ import annotations

import numpy as np


SONICMJ_INITIAL_POSE_NAME = "sonicmj"
SONICMJ_INITIAL_POSE_SOURCE = (
    "/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/sonic_mj/assets.py:"
    "SONIC_G1_DEFAULT_JOINT_POS"
)

# Unitree G1 29-DoF hardware order. This is also SONIC's canonical MuJoCo order.
G1_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

SONICMJ_INITIAL_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_hip_pitch_joint": -0.312,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 0.6,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 0.6,
}

# The official Unitree Arm SDK example publishes arms first and waist last.
G1_ARM_SDK_JOINT_NAMES = (
    *G1_BODY_JOINT_NAMES[15:22],
    *G1_BODY_JOINT_NAMES[22:29],
    *G1_BODY_JOINT_NAMES[12:15],
)


def sonicmj_initial_q(joint_names: tuple[str, ...] = G1_BODY_JOINT_NAMES) -> np.ndarray:
    """Return the SONICMJ pose in the requested named joint order."""
    unknown = sorted(set(joint_names) - set(G1_BODY_JOINT_NAMES))
    if unknown:
        raise KeyError(f"SONICMJ 初始化姿态不包含关节: {unknown}")
    return np.asarray(
        [SONICMJ_INITIAL_JOINT_POS.get(name, 0.0) for name in joint_names],
        dtype=np.float64,
    )


def smooth_initialization_trajectory(
    current_q: np.ndarray,
    target_q: np.ndarray,
    *,
    fps: int,
    duration_s: float,
    max_speed_rad_s: float,
) -> np.ndarray:
    """Generate a smoothstep joint trajectory and reject an unsafe duration."""
    current = np.asarray(current_q, dtype=np.float64).reshape(-1)
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if current.shape != target.shape or not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("初始化起点/终点必须是同维有限关节向量")
    if fps <= 0 or duration_s <= 0 or max_speed_rad_s <= 0:
        raise ValueError("fps、duration_s 和 max_speed_rad_s 必须为正")
    # max derivative of 3t^2-2t^3 is 1.5 at t=0.5.
    required_duration = 1.5 * float(np.max(np.abs(target - current))) / max_speed_rad_s
    if duration_s + 1e-9 < required_duration:
        raise ValueError(
            f"初始化时长 {duration_s:.3f}s 过短；按 {max_speed_rad_s:.3f}rad/s "
            f"上限至少需要 {required_duration:.3f}s"
        )
    count = max(2, int(round(duration_s * fps)) + 1)
    phase = np.linspace(0.0, 1.0, count, dtype=np.float64)
    alpha = phase * phase * (3.0 - 2.0 * phase)
    return (current[None] + alpha[:, None] * (target - current)[None]).astype(np.float32)
