from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin


ROBOT_MCP_JOINTS = (
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
)
MANO_MCP_INDICES = (5, 9, 13, 17)


def palm_basis(wrist: np.ndarray, mcps: np.ndarray) -> np.ndarray:
    forward = mcps.mean(axis=0) - wrist
    forward /= np.linalg.norm(forward)
    lateral = mcps[-1] - mcps[0]
    lateral -= forward * np.dot(lateral, forward)
    lateral /= np.linalg.norm(lateral)
    normal = np.cross(lateral, forward)
    normal /= np.linalg.norm(normal)
    return np.column_stack((normal, -forward, lateral))


def calibrate(mano_landmarks: np.ndarray, urdf_path: Path) -> dict:
    mano_wrist = mano_landmarks[0]
    mano_mcps = mano_landmarks[list(MANO_MCP_INDICES)]
    mano_basis = palm_basis(mano_wrist, mano_mcps)

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    pin.forwardKinematics(model, data, pin.neutral(model))
    robot_wrist = np.zeros(3)
    robot_mcps = np.stack(
        [data.oMi[model.getJointId(name)].translation for name in ROBOT_MCP_JOINTS]
    )
    robot_basis = palm_basis(robot_wrist, robot_mcps)

    # Pose of Inspire base B expressed in MANO wrist frame W:
    # R_WB maps robot-base vectors into MANO coordinates. Translation aligns
    # the two four-MCP centers, which is the grasp-relevant palm anchor.
    rotation_wrist_base = mano_basis @ robot_basis.T
    translation_wrist_base = (
        mano_mcps.mean(axis=0) - rotation_wrist_base @ robot_mcps.mean(axis=0)
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_wrist_base
    transform[:3, 3] = translation_wrist_base
    robot_mcps_in_mano = (rotation_wrist_base @ robot_mcps.T).T + translation_wrist_base
    residuals = np.linalg.norm(robot_mcps_in_mano - mano_mcps, axis=1)
    return {
        "T_mano_wrist_inspire_base": transform.tolist(),
        "mano_palm_length_m": float(np.linalg.norm(mano_mcps.mean(axis=0) - mano_wrist)),
        "inspire_palm_length_m": float(np.linalg.norm(robot_mcps.mean(axis=0) - robot_wrist)),
        "inspire_to_mano_palm_length_ratio": float(
            np.linalg.norm(robot_mcps.mean(axis=0) - robot_wrist)
            / np.linalg.norm(mano_mcps.mean(axis=0) - mano_wrist)
        ),
        "mcp_alignment_error_m": residuals.tolist(),
        "mcp_alignment_rmse_m": float(np.sqrt(np.mean(residuals**2))),
        "mano_mcps_m": mano_mcps.tolist(),
        "inspire_mcps_in_mano_m": robot_mcps_in_mano.tolist(),
        "method": "canonical palm-frame orientation plus four-MCP-center alignment",
    }


def plot_calibration(result: dict, output: Path) -> None:
    mano = np.asarray(result["mano_mcps_m"])
    robot = np.asarray(result["inspire_mcps_in_mano_m"])
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*mano.T, s=65, label="MANO MCP", color="#168aad")
    ax.scatter(*robot.T, s=55, marker="x", label="Inspire MCP aligned", color="#e85d04")
    for index, (human, robot_point) in enumerate(zip(mano, robot)):
        ax.plot(*np.stack((human, robot_point)).T, color="#666666", linewidth=1)
        ax.text(*human, ("index", "middle", "ring", "pinky")[index])
    transform = np.asarray(result["T_mano_wrist_inspire_base"])
    origin = transform[:3, 3]
    for axis, color, label in zip(transform[:3, :3].T, "rgb", "xyz"):
        ax.quiver(*origin, *axis, length=0.035, color=color, label=f"Inspire {label}")
    ax.scatter(0, 0, 0, marker="o", s=80, color="black", label="MANO wrist")
    ax.scatter(*origin, marker="^", s=80, color="#6a040f", label="Inspire base")
    ax.set_xlabel("MANO x / m")
    ax.set_ylabel("MANO y / m")
    ax.set_zlabel("MANO z / m")
    ax.set_title(f"MANO–Inspire calibration, MCP RMSE={result['mcp_alignment_rmse_m']*1000:.1f} mm")
    ax.legend(fontsize=8)
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mano", type=Path, default=project / "assets/calibration/mano_right_hug_canonical.npz"
    )
    parser.add_argument(
        "--urdf", type=Path,
        default=project / "third_party/xr_teleoperate/assets/inspire_hand/inspire_hand_right.urdf",
    )
    parser.add_argument(
        "--output", type=Path, default=project / "configs/mano_inspire_calibration.json"
    )
    parser.add_argument(
        "--plot", type=Path,
        default=project / "assets/calibration/mano_inspire_alignment.png",
    )
    args = parser.parse_args()
    landmarks = np.load(args.mano)["landmarks"]
    result = calibrate(landmarks, args.urdf.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_calibration(result, args.plot)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
