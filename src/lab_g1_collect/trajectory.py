from __future__ import annotations

import numpy as np


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate_grasp_trajectory(
    start_arm: np.ndarray,
    grasp_arm: np.ndarray,
    lift_arm: np.ndarray,
    hand_target: np.ndarray,
    *,
    fps: int,
    seconds: float,
    phase_ratios: tuple[float, float, float] = (0.35, 0.25, 0.40),
) -> np.ndarray:
    """生成 [右臂 7DoF, Inspire 右手 6DoF] 的平滑动作轨迹。"""
    start_arm = np.asarray(start_arm, dtype=np.float32)
    grasp_arm = np.asarray(grasp_arm, dtype=np.float32)
    lift_arm = np.asarray(lift_arm, dtype=np.float32)
    hand_target = np.asarray(hand_target, dtype=np.float32)
    if any(x.shape != (7,) for x in (start_arm, grasp_arm, lift_arm)):
        raise ValueError("右臂关节向量必须为 7 维")
    if hand_target.shape != (6,):
        raise ValueError("Inspire 命令必须为 6 维")
    ratios = np.asarray(phase_ratios, dtype=float)
    if not np.isclose(ratios.sum(), 1.0) or np.any(ratios <= 0):
        raise ValueError("阶段比例必须为正且总和为 1")
    count = max(3, int(round(fps * seconds)))
    cuts = np.cumsum(np.r_[0, np.maximum(1, np.floor(count * ratios).astype(int))])
    cuts[-1] = count
    action = np.zeros((count, 13), dtype=np.float32)

    def blend(begin: int, end: int, a: np.ndarray, b: np.ndarray) -> None:
        alpha = smoothstep(np.linspace(0, 1, max(end - begin, 1), dtype=np.float32))[:, None]
        action[begin:end, :7] = a + alpha * (b - a)

    blend(cuts[0], cuts[1], start_arm, grasp_arm)
    action[cuts[0]:cuts[1], 7:] = 0
    action[cuts[1]:cuts[2], :7] = grasp_arm
    close_alpha = smoothstep(np.linspace(0, 1, cuts[2] - cuts[1], dtype=np.float32))[:, None]
    action[cuts[1]:cuts[2], 7:] = close_alpha * hand_target
    blend(cuts[2], cuts[3], grasp_arm, lift_arm)
    action[cuts[2]:cuts[3], 7:] = hand_target
    return action

