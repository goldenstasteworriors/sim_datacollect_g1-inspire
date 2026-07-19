"""Bounded multi-start 6D IK feasibility checks for the G1 right arm."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .sim_action import RIGHT_ARM_JOINTS


@lru_cache(maxsize=1)
def _g1_model():
    import pinocchio as pin

    project = Path(__file__).resolve().parents[2]
    urdf = project / "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    q_indices = np.array([model.joints[model.getJointId(name)].idx_q for name in RIGHT_ARM_JOINTS])
    lower = model.lowerPositionLimit[q_indices].copy()
    upper = model.upperPositionLimit[q_indices].copy()
    frame_id = model.getFrameId("right_hand_palm_link")
    return pin, model, data, q_indices, lower, upper, frame_id


def solve_right_arm_ik(
    target_position_base: np.ndarray,
    target_rotation_base: np.ndarray,
    current_arm: np.ndarray,
    *,
    seed: int = 42,
    starts: int = 6,
    position_tolerance_m: float = 0.02,
    rotation_tolerance_rad: float = 0.30,
) -> dict:
    """Solve bounded 6D IK and return feasibility plus residuals and q solution."""
    pin, model, data, q_indices, lower, upper, frame_id = _g1_model()
    target_position = np.asarray(target_position_base, dtype=np.float64).reshape(3)
    target_rotation = np.asarray(target_rotation_base, dtype=np.float64).reshape(3, 3)
    current = np.clip(np.asarray(current_arm, dtype=np.float64).reshape(7), lower, upper)
    q_full = pin.neutral(model)

    def errors(arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_full[q_indices] = arm
        pin.forwardKinematics(model, data, q_full)
        pin.updateFramePlacements(model, data)
        pose = data.oMf[frame_id]
        position_error = pose.translation - target_position
        rotation_error = pin.log3(pose.rotation.T @ target_rotation)
        return position_error, rotation_error

    def residual(arm: np.ndarray) -> np.ndarray:
        position_error, rotation_error = errors(arm)
        # Metres and radians have different practical tolerances. This weight
        # keeps translation primary without allowing arbitrary wrist rotation.
        return np.r_[position_error, 0.08 * rotation_error]

    rng = np.random.default_rng(seed)
    initial = [current, np.clip(np.zeros(7), lower, upper)]
    for _ in range(max(0, starts - len(initial))):
        initial.append(rng.uniform(lower, upper))
    best = None
    for start in initial:
        result = least_squares(
            residual, start, bounds=(lower, upper), method="trf",
            max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-9,
        )
        position_error, rotation_error = errors(result.x)
        position_norm = float(np.linalg.norm(position_error))
        rotation_norm = float(np.linalg.norm(rotation_error))
        joint_motion = float(np.linalg.norm(result.x - current))
        score = position_norm / position_tolerance_m + rotation_norm / rotation_tolerance_rad
        candidate = (score, position_norm, rotation_norm, joint_motion, result.x.copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {
        "reachable": bool(best[1] <= position_tolerance_m and best[2] <= rotation_tolerance_rad),
        "position_error_m": best[1],
        "rotation_error_rad": best[2],
        "joint_motion_rad": best[3],
        "joint_positions": best[4].astype(np.float32),
    }
