"""Official G1 head-camera geometry from NVIDIA GR00T-WholeBodyControl."""

from __future__ import annotations

import numpy as np


OFFICIAL_G1_MJCF_URL = (
    "https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/"
    "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
)


def _translation(xyz: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = xyz
    return transform


def _rotation(axis: str, angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    if axis == "x":
        matrix = np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    elif axis == "y":
        matrix = np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    elif axis == "z":
        matrix = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    else:
        raise ValueError(f"unsupported rotation axis: {axis}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = matrix
    return transform


def official_g1_T_pelvis_camera_optical(waist_q: np.ndarray) -> np.ndarray:
    """Return optical-camera pose in pelvis coordinates for yaw/roll/pitch.

    The MJCF camera frame looks along local -Z with +Y up. RealSense RGB-D
    points use the conventional optical frame (+X right, +Y down, +Z forward),
    so the final fixed rotation flips camera Y and Z.
    """
    yaw, roll, pitch = np.asarray(waist_q, dtype=np.float64).reshape(3)

    # Official MJCF body chain:
    # pelvis -> waist_yaw -> waist_roll(pos) -> torso(pos) -> head_camera(pos/euler)
    transform = _rotation("z", float(yaw))
    transform @= _translation((-0.0039635, 0.0, 0.035))
    transform @= _rotation("x", float(roll))
    transform @= _translation((0.0, 0.0, 0.019))
    transform @= _rotation("y", float(pitch))
    transform @= _translation((0.06, 0.0, 0.45))

    # MJCF compiler defaults to intrinsic eulerseq="xyz". The official camera
    # has euler="0 -0.8 -1.57", hence R = Rx(0) Ry(-0.8) Rz(-1.57).
    transform @= _rotation("x", 0.0)
    transform @= _rotation("y", -0.8)
    transform @= _rotation("z", -1.57)
    optical_from_mjcf = np.eye(4, dtype=np.float64)
    optical_from_mjcf[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform @= optical_from_mjcf
    return transform
