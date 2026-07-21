"""Create an arms-and-waist SONICMJ plan without generating leg commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .initial_pose import (
    G1_ARM_SDK_JOINT_NAMES,
    G1_BODY_JOINT_NAMES,
    SONICMJ_INITIAL_POSE_NAME,
    SONICMJ_INITIAL_POSE_SOURCE,
    smooth_initialization_trajectory,
    sonicmj_initial_q,
)
from .real_state import read_g1_arm_state


CONTROL_OUTPUT_ENABLED = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real_robot_dry_run.yaml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/real_robot_dry_run/sonicmj_initialization"),
    )
    args = parser.parse_args()
    if CONTROL_OUTPUT_ENABLED:
        raise RuntimeError("安全断言失败：初始化规划器不得启用控制输出")

    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if config["safety"].get("mode") != "dry_run" or config["safety"].get("allow_control") is not False:
        raise ValueError("真机配置必须固定为 mode=dry_run 且 allow_control=false")
    init_cfg = config["initialization"]
    if init_cfg.get("profile") != SONICMJ_INITIAL_POSE_NAME:
        raise ValueError("initialization.profile 必须为 sonicmj")

    state = read_g1_arm_state(config["robot_state"])
    if state.body_q is None or state.body_dq is None:
        raise RuntimeError(
            "机器人上的 g1_read_lowstate 尚未包含完整 LowState；"
            "需要它只读提取双臂和腰部状态，请先重新编译部署只读工具"
        )
    body_index = {name: index for index, name in enumerate(G1_BODY_JOINT_NAMES)}
    arm_sdk_indices = np.asarray(
        [body_index[name] for name in G1_ARM_SDK_JOINT_NAMES], dtype=np.int64
    )
    if np.any(arm_sdk_indices < 12):
        raise RuntimeError("安全断言失败：Arm SDK 初始化集合包含腿部关节索引")
    current_q = state.body_q[arm_sdk_indices]
    current_dq = state.body_dq[arm_sdk_indices]
    max_measured_speed = float(np.max(np.abs(current_dq)))
    stationary_limit = float(init_cfg["max_measured_speed_rad_s"])
    if max_measured_speed > stationary_limit:
        raise RuntimeError(
            f"机器人双臂/腰部仍在运动，17DoF |dq|max={max_measured_speed:.3f}rad/s，"
            f"超过规划上限 {stationary_limit:.3f}rad/s"
        )

    target_q = sonicmj_initial_q(G1_ARM_SDK_JOINT_NAMES)
    max_joint_speed = float(init_cfg["max_joint_speed_rad_s"])
    configured_min_duration = float(init_cfg["duration_s"])
    # The maximum derivative of smoothstep(3t^2 - 2t^3) is 1.5.  Match the
    # hardware initializer: treat the configured duration as a lower bound and
    # extend it whenever the measured starting pose needs more time.
    required_duration = (
        1.5 * float(np.max(np.abs(target_q - current_q))) / max_joint_speed
    )
    duration = max(configured_min_duration, required_duration)
    trajectory = smooth_initialization_trajectory(
        current_q,
        target_q,
        fps=int(init_cfg["fps"]),
        duration_s=duration,
        max_speed_rad_s=max_joint_speed,
    )
    measured_max_speed = float(
        np.max(np.abs(np.diff(trajectory, axis=0))) * int(init_cfg["fps"])
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "initialization_plan.npz",
        trajectory=trajectory,
        arm_sdk_trajectory=trajectory,
        current_arm_sdk_q=current_q.astype(np.float32),
        target_arm_sdk_q=target_q.astype(np.float32),
        arm_sdk_joint_names=np.asarray(G1_ARM_SDK_JOINT_NAMES),
        arm_sdk_motor_indices=arm_sdk_indices,
    )
    metadata = {
        "mode": "dry_run",
        "control_output_enabled": CONTROL_OUTPUT_ENABLED,
        "profile": SONICMJ_INITIAL_POSE_NAME,
        "source": SONICMJ_INITIAL_POSE_SOURCE,
        "lowstate_source": state.source,
        "mode_machine": state.mode_machine,
        "fps": int(init_cfg["fps"]),
        "duration_s": duration,
        "configured_min_duration_s": configured_min_duration,
        "required_duration_s": required_duration,
        "max_measured_arm_sdk_speed_rad_s": max_measured_speed,
        "planned_max_joint_speed_rad_s": measured_max_speed,
        "configured_max_joint_speed_rad_s": max_joint_speed,
        "commanded_joint_names": list(G1_ARM_SDK_JOINT_NAMES),
        "commanded_motor_indices": arm_sdk_indices.tolist(),
        "current_arm_sdk_q": current_q.tolist(),
        "target_arm_sdk_q": target_q.tolist(),
        "leg_command_indices": [],
        "leg_commands_generated": False,
        "real_execution_scope": "waist_and_dual_arms_via_rt/arm_sdk; legs_remain_under_SONIC_WBC",
    }
    report = output / "initialization_plan.json"
    report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"已保存初始化 dry-run 计划: {report}")


if __name__ == "__main__":
    main()
