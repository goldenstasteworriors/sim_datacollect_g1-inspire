#!/usr/bin/env python3
"""Visualize Cartesian tracking and Pinocchio/Isaac FK consistency for an episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from lab_g1_collect.arm_ik import _g1_model


def _pose_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        np.r_[pose_wxyz[4:7], pose_wxyz[3]]
    ).as_matrix()
    matrix[:3, 3] = pose_wxyz[:3]
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-m", type=float, default=0.04)
    args = parser.parse_args()

    data = np.load(args.episode / "episode.npz")
    actual = data["observation.ee_pose_w"]
    target = data["target.ee_pose_w"]
    arm = data["observation.state"][:, :7]
    tracking_error = np.linalg.norm(actual[:, :3] - target[:, :3], axis=1)

    pin, model, model_data, q_indices, _lower, _upper, frame_id = _g1_model()
    q_full = pin.neutral(model)
    pin_poses = []
    for joints in arm:
        q_full[q_indices] = joints
        pin.forwardKinematics(model, model_data, q_full)
        pin.updateFramePlacements(model, model_data)
        pin_poses.append(model_data.oMf[frame_id].homogeneous.copy())

    # Remove the constant frame/base offset at frame zero. Any remaining error
    # varying with q is a kinematic-model mismatch, not a wrist-frame offset.
    correction = np.linalg.inv(pin_poses[0]) @ _pose_matrix(actual[0])
    pin_positions = np.asarray([(pose @ correction)[:3, 3] for pose in pin_poses])
    fk_error = np.linalg.norm(pin_positions - actual[:, :3], axis=1)
    exceeded = np.flatnonzero(tracking_error > args.tolerance_m)

    args.output.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(13, 5.5))
    axis = figure.add_subplot(121, projection="3d")
    axis.plot(*target[:, :3].T, label="target path", linewidth=2.5)
    axis.plot(*actual[:, :3].T, label="Isaac actual", linewidth=2.0)
    axis.scatter(*target[-1, :3], marker="*", s=160, label="HUG/phase endpoint")
    if exceeded.size:
        first = int(exceeded[0])
        axis.scatter(*actual[first, :3], color="red", s=70, label=f"first > {args.tolerance_m:.2f} m")
    axis.set_xlabel("world X / m")
    axis.set_ylabel("world Y / m")
    axis.set_zlabel("world Z / m")
    axis.set_title("Target path vs Isaac end effector")
    axis.legend()

    error_axis = figure.add_subplot(122)
    frames = np.arange(len(actual))
    error_axis.plot(frames, tracking_error * 1000, label="tracking error")
    error_axis.plot(frames, fk_error * 1000, label="Pinocchio–Isaac FK mismatch")
    error_axis.axhline(args.tolerance_m * 1000, color="red", linestyle="--", label="online gate")
    error_axis.set_xlabel("recorded frame (10 simulation steps)")
    error_axis.set_ylabel("position error / mm")
    error_axis.set_title("Position error")
    error_axis.grid(alpha=0.3)
    error_axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / "ik_reachability.png", dpi=180)
    plt.close(figure)

    report = {
        "episode": str(args.episode),
        "recorded_frames": int(len(actual)),
        "tracking_error_m": {
            "final": float(tracking_error[-1]),
            "maximum": float(tracking_error.max()),
            "first_over_tolerance_frame": int(exceeded[0]) if exceeded.size else None,
        },
        "pinocchio_isaac_fk_error_after_frame0_alignment_m": {
            "final": float(fk_error[-1]),
            "median": float(np.median(fk_error)),
            "maximum": float(fk_error.max()),
        },
        "interpretation": (
            "The current offline IK model cannot establish physical reachability when "
            "its FK diverges from Isaac after a frame-zero rigid alignment."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
