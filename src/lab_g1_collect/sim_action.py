from __future__ import annotations

import numpy as np


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

RIGHT_INSPIRE_COUPLING = {
    "R_pinky_proximal_joint": (0, 1.0), "R_pinky_intermediate_joint": (0, 1.0),
    "R_ring_proximal_joint": (1, 1.0), "R_ring_intermediate_joint": (1, 1.0),
    "R_middle_proximal_joint": (2, 1.0), "R_middle_intermediate_joint": (2, 1.0),
    "R_index_proximal_joint": (3, 1.0), "R_index_intermediate_joint": (3, 1.0),
    "R_thumb_proximal_pitch_joint": (4, 1.0), "R_thumb_intermediate_joint": (4, 1.5),
    "R_thumb_distal_joint": (4, 2.4), "R_thumb_proximal_yaw_joint": (5, 1.0),
}


def compact_to_isaac_action(compact: np.ndarray, joint_names: list[str], default_joint_pos: np.ndarray) -> np.ndarray:
    """将 13 维策略输出嵌入 Unitree 官方 53 维 Isaac articulation action。

    未控制的腰、腿、左臂和左手保持默认偏移 0；右手耦合系数完全复用官方
    `action_provider_dds.py`。
    """
    compact = np.asarray(compact, dtype=np.float32)
    defaults = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
    if compact.shape != (13,) or defaults.shape != (len(joint_names),):
        raise ValueError("compact/default_joint_pos 维度不匹配")
    index = {name: i for i, name in enumerate(joint_names)}
    required = set(RIGHT_ARM_JOINTS) | set(RIGHT_INSPIRE_COUPLING)
    missing = required - index.keys()
    if missing:
        raise KeyError(f"Isaac articulation 缺少关节: {sorted(missing)}")
    action = np.zeros(len(joint_names), dtype=np.float32)
    for source, name in enumerate(RIGHT_ARM_JOINTS):
        action[index[name]] = compact[source] - defaults[index[name]]
    hand = compact[7:]
    for name, (source, scale) in RIGHT_INSPIRE_COUPLING.items():
        action[index[name]] = hand[source] * scale - defaults[index[name]]
    return action

