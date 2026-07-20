from __future__ import annotations

import argparse
import json
import os
import random
import select
import sys
import termios
import traceback
import tty
from pathlib import Path


class TerminalKeyReader:
    """Non-blocking single-key input while preserving the terminal on exit."""

    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdin.isatty()
        self.fd = sys.stdin.fileno() if self.enabled else None
        self.previous = None

    def __enter__(self):
        if self.enabled:
            self.previous = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def read(self) -> str | None:
        if self.enabled and select.select([self.fd], [], [], 0.0)[0]:
            return os.read(self.fd, 1).decode(errors="ignore").lower()
        return None

    def __exit__(self, _type, _value, _traceback):
        if self.previous is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动并单步验证 LabUtopia 烧杯 G1+Inspire 场景")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-collect", action="store_true")
    parser.add_argument("--use-hug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object-shape", choices=("beaker", "box", "cylinder", "sphere"), default="beaker")
    parser.add_argument("--hand-closure-scale", type=float, default=1.0)
    parser.add_argument("--hug-sampling-steps", type=int, default=5)
    parser.add_argument("--hug-candidates", type=int, default=8)
    parser.add_argument(
        "--pregrasp-offset-m", type=float, default=0.03,
        help="沿物体中心到HUG grasp腕位置方向向外生成pre-grasp的距离，默认0.03 m",
    )
    parser.add_argument("--debug-ik", action="store_true")
    parser.add_argument(
        "--arm-ik", choices=("xr_teleoperate", "dls"), default="xr_teleoperate",
        help="手臂 IK：宇树 xr_teleoperate 官方 G1_29 CasADi IK，或旧的本地 DLS 对照",
    )
    parser.add_argument(
        "--xr-ik-profile", choices=("autonomous", "teleop"), default="autonomous",
        help="XR IK权衡：自主抓取位置优先并对齐Isaac软限位，或上游遥操作原始权重",
    )
    parser.add_argument(
        "--ik-rotation-weight", type=float, default=0.25,
        help="pre-grasp/grasp 阶段 DLS 姿态误差权重；0 用于仅验证位置可达性",
    )
    parser.add_argument(
        "--ik-command-integration", type=float, default=0.95,
        help="将当前 DLS 修正累积到上次位置命令的比例（0 表示不累积）",
    )
    parser.add_argument(
        "--ik-max-joint-step", type=float, default=0.02,
        help="每个控制步允许 DLS 位置目标领先当前关节的最大弧度",
    )
    parser.add_argument(
        "--ik-max-command-lead", type=float, default=0.15,
        help="位置命令相对实测关节的最大领先弧度，用于抑制累积超调",
    )
    parser.add_argument(
        "--joint-limit-warning-rad", type=float, default=0.08,
        help="GUI中将接近Isaac软限位的右臂关节用红球标出的余量阈值",
    )
    parser.add_argument(
        "--waypoint-tolerance-m", type=float, default=0.04,
        help="pre-grasp/grasp 动态到达门槛；默认严格值0.04 m",
    )
    parser.add_argument("--output", default="outputs/gui_collect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object-randomization-m", type=float, default=0.04)
    parser.add_argument(
        "--table-top-height-m", type=float, default=0.70,
        help="碰撞桌板上表面的世界坐标高度，默认与真实桌面一致为0.70 m",
    )
    parser.add_argument(
        "--replay-real-plan", type=Path,
        help="回放 real_grasp 生成的 dry_run_plan.npz；只在 Isaac 仿真内执行",
    )
    parser.add_argument(
        "--object-fixed-xyz", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="用世界坐标固定物体位置并关闭XY随机化，例如 -0.06 0.38 0.86",
    )
    parser.add_argument(
        "--object-grid-xy", type=int, default=None,
        help="以--object-fixed-xyz为中心生成N×N个XY位置（建议奇数）",
    )
    parser.add_argument(
        "--object-grid-step-m", type=float, default=0.02,
        help="物体XY网格相邻位置间距，默认0.02 m",
    )
    parser.add_argument(
        "--object-grid-step-xy", type=float, nargs=2, default=None,
        metavar=("DX", "DY"), help="分别指定物体网格X/Y间距，覆盖--object-grid-step-m",
    )
    parser.add_argument(
        "--nearby-ik-test", action="store_true",
        help="将圆柱固定在右手附近，并用初始腕位姿的小偏移验证手臂 IK（绕过 HUG）",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--keep-success-visuals", type=int, default=5)
    parser.add_argument("--keep-failure-visuals", type=int, default=5)
    args = parser.parse_args()
    if not 0.2 <= args.table_top_height_m <= 1.5:
        parser.error("--table-top-height-m 必须在 0.2~1.5 m")
    if not 0.01 <= args.waypoint_tolerance_m <= 0.06:
        parser.error("--waypoint-tolerance-m 必须在 0.01~0.06 m")
    replay_plan = None
    if args.replay_real_plan is not None:
        import numpy as np
        plan_path = args.replay_real_plan.expanduser().resolve()
        replay_plan = {key: value.copy() for key, value in np.load(plan_path).items()}
        required = {
            "T_base_pregrasp", "T_base_grasp", "T_base_lift",
            "object_xyz_base", "hand_target_rad",
        }
        missing = sorted(required - replay_plan.keys())
        if missing:
            parser.error(f"real plan 缺少字段: {missing}")
    episode_rng = random.Random(args.seed)
    object_grid = None
    if args.object_grid_xy is not None:
        if args.object_fixed_xyz is None:
            parser.error("--object-grid-xy requires --object-fixed-xyz X Y Z")
        grid_size = max(1, args.object_grid_xy)
        half = (grid_size - 1) / 2.0
        if args.object_grid_step_xy is None:
            step_x = step_y = max(0.0, args.object_grid_step_m)
        else:
            step_x, step_y = (max(0.0, value) for value in args.object_grid_step_xy)
        center = args.object_fixed_xyz
        object_grid = [
            [center[0] + (ix - half) * step_x, center[1] + (iy - half) * step_y, center[2]]
            for ix in range(grid_size) for iy in range(grid_size)
        ]
    reset_index = 0
    os.environ["LAB_OBJECT_SHAPE"] = args.object_shape
    os.environ["LAB_TABLE_TOP_HEIGHT_M"] = str(args.table_top_height_m)

    project = Path(__file__).resolve().parents[2]
    unitree = project / "third_party/unitree_sim_isaaclab"
    # Unitree's scene modules resolve their checked-in USD assets from this
    # variable at import time. Set a local default so launching this module does
    # not depend on the caller exporting PROJECT_ROOT first.
    os.environ.setdefault("PROJECT_ROOT", str(unitree))
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(unitree))

    # Match Unitree's official launcher order so Pinocchio resolves its bundled
    # Assimp symbols before Isaac Kit loads another Assimp build.
    import pinocchio as pin

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = launcher.app
    try:
        import gymnasium as gym
        import omni.usd
        import torch
        import tasks  # noqa: F401
        import sim_tasks.lab_beaker_g1_inspire  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from isaaclab.utils.math import matrix_from_quat

        task_id = "Isaac-PickPlace-LabBeaker-G129-Inspire-Right"
        # Isaac Sim 5.0's GUI path passes an Ar.ResolvedPath to a Boost binding
        # that accepts only str. The checked-in LabUtopia asset is a validated
        # USD crate, so bypass only this broken preflight predicate.
        omni.usd.is_usd_crate_file_version_supported = lambda _path: True
        print("[sim-smoke] registered", file=sys.stderr, flush=True)
        cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        cfg.scene.robot.spawn.activate_contact_sensors = True
        cfg.viewer.eye = (2.2, 2.2, 1.8)
        cfg.viewer.lookat = (-0.25, 0.45, 1.05)
        print("[sim-smoke] config parsed", file=sys.stderr, flush=True)
        env = gym.make(task_id, cfg=cfg).unwrapped
        print("[sim-smoke] environment created", file=sys.stderr, flush=True)
        zero_action = torch.zeros(env.action_space.shape, device=env.device)

        def reset_scene():
            nonlocal reset_index
            observation, _ = env.reset()
            obj = env.scene["object"]
            default_state = obj.data.default_root_state.clone()
            default_state[:, :3] += env.scene.env_origins
            if object_grid is not None:
                grid_position = object_grid[reset_index % len(object_grid)]
                reset_index += 1
                default_state[:, :3] = torch.tensor(grid_position, device=env.device)
            elif args.object_fixed_xyz is not None:
                default_state[:, :3] = torch.tensor(
                    args.object_fixed_xyz, device=env.device
                )
            elif replay_plan is not None:
                root_pose = env.scene["robot"].data.root_pose_w[0]
                root_rotation = matrix_from_quat(root_pose[3:7].unsqueeze(0))[0]
                object_base = torch.tensor(
                    replay_plan["object_xyz_base"], device=env.device, dtype=root_pose.dtype
                )
                default_state[:, :3] = root_pose[:3] + root_rotation @ object_base
            elif args.nearby_ik_test:
                default_state[:, :3] = torch.tensor(
                    [-0.040, 0.380, 0.860], device=env.device
                )
            else:
                randomization = max(0.0, args.object_randomization_m)
                default_state[:, 0] += episode_rng.uniform(-randomization, randomization)
                default_state[:, 1] += episode_rng.uniform(-randomization, randomization)
            obj.write_root_pose_to_sim(default_state[:, :7])
            obj.write_root_velocity_to_sim(torch.zeros_like(default_state[:, 7:13]))
            env.scene.write_data_to_sim()
            # Actions are relative to the configured default joint pose. Let the
            # object settle and refresh camera tensors before planning from RGB-D.
            for _ in range(20):
                observation, *_ = env.step(zero_action)
            # Settling can still move a previously grasped object through residual
            # contacts. Restore it once more after the robot has reached reset pose.
            obj.write_root_pose_to_sim(default_state[:, :7])
            obj.write_root_velocity_to_sim(torch.zeros_like(default_state[:, 7:13]))
            env.scene.write_data_to_sim()
            observation, *_ = env.step(zero_action)
            position = obj.data.root_pos_w[0].cpu().tolist()
            print(f"[sim-collect] object reset xyz={position}", file=sys.stderr, flush=True)
            return observation

        observation = reset_scene()
        print("[sim-smoke] environment reset", file=sys.stderr, flush=True)
        print(
            "[sim-smoke] beaker xyz",
            env.scene["object"].data.root_pos_w[0].cpu().tolist(),
            file=sys.stderr,
            flush=True,
        )
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
        box = bound.ComputeWorldBound(stage.GetPrimAtPath("/World/envs/env_0/Object")).ComputeAlignedBox()
        print("[sim-smoke] beaker bbox", box.GetMin(), box.GetMax(), file=sys.stderr, flush=True)
        robot = env.scene["robot"]
        for body_index, body_name in enumerate(robot.body_names):
            lowered = body_name.lower()
            if "hand" in lowered or "wrist" in lowered or body_name.startswith("R_"):
                xyz = robot.data.body_pos_w[0, body_index].cpu().tolist()
                print(f"[sim-smoke] body {body_name} xyz {xyz}", file=sys.stderr, flush=True)
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG
        from isaaclab.utils.math import (
            compute_pose_error, quat_from_matrix, quat_inv,
            quat_slerp, subtract_frame_transforms,
        )
        from .sim_action import RIGHT_ARM_JOINTS, compact_to_isaac_action

        joint_names = list(env.scene["robot"].joint_names)
        default_pos = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
        import numpy as np

        arm_default = np.array([default_pos[joint_names.index(name)] for name in RIGHT_ARM_JOINTS], dtype=np.float32)
        pregrasp_end = 150
        grasp_end = 300
        close_end = 400
        lift_end = 550
        cycle_steps = 600
        arm_joint_ids = [joint_names.index(name) for name in RIGHT_ARM_JOINTS]
        left_arm_joint_names = [name.replace("right_", "left_", 1) for name in RIGHT_ARM_JOINTS]
        left_arm_joint_ids = [joint_names.index(name) for name in left_arm_joint_names]
        right_hand_index = env.scene["robot"].body_names.index("right_hand_base_link")
        right_finger_indices = [
            index for index, name in enumerate(env.scene["robot"].body_names) if name.startswith("R_")
        ]
        jacobian_index = right_hand_index - 1 if robot.is_fixed_base else right_hand_index
        hug_marker_cfg = FRAME_MARKER_CFG.copy()
        hug_marker_cfg.markers.pop("connecting_line", None)
        hug_marker_cfg.markers["frame"].scale = (0.18, 0.18, 0.18)
        hug_marker = VisualizationMarkers(hug_marker_cfg.replace(prim_path="/Visuals/HUGGraspPose"))
        pregrasp_marker_cfg = FRAME_MARKER_CFG.copy()
        pregrasp_marker_cfg.markers.pop("connecting_line", None)
        pregrasp_marker_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
        pregrasp_marker = VisualizationMarkers(
            pregrasp_marker_cfg.replace(prim_path="/Visuals/HUGPregraspPose")
        )
        reach_path_cfg = SPHERE_MARKER_CFG.copy()
        reach_path_cfg.markers["sphere"].radius = 0.009
        reach_path_cfg.markers["sphere"].visual_material.diffuse_color = (0.10, 0.35, 1.00)
        reach_path_marker = VisualizationMarkers(
            reach_path_cfg.replace(prim_path="/Visuals/RealPlanReachPath")
        )
        approach_path_cfg = SPHERE_MARKER_CFG.copy()
        approach_path_cfg.markers["sphere"].radius = 0.010
        approach_path_cfg.markers["sphere"].visual_material.diffuse_color = (1.00, 0.35, 0.05)
        approach_path_marker = VisualizationMarkers(
            approach_path_cfg.replace(prim_path="/Visuals/RealPlanApproachPath")
        )
        lift_path_cfg = SPHERE_MARKER_CFG.copy()
        lift_path_cfg.markers["sphere"].radius = 0.010
        lift_path_cfg.markers["sphere"].visual_material.diffuse_color = (0.10, 0.90, 0.25)
        lift_path_marker = VisualizationMarkers(
            lift_path_cfg.replace(prim_path="/Visuals/RealPlanLiftPath")
        )
        limit_marker_cfg = SPHERE_MARKER_CFG.copy()
        limit_marker_cfg.markers["sphere"].radius = 0.04
        limit_marker = VisualizationMarkers(
            limit_marker_cfg.replace(prim_path="/Visuals/RightArmJointLimitWarning")
        )
        hidden_limit_marker_pos = torch.tensor([[0.0, 0.0, -10.0]], device=env.device)
        limit_marker.visualize(hidden_limit_marker_pos)
        arm_joint_body_ids = [
            robot.body_names.index(name.replace("_joint", "_link")) for name in RIGHT_ARM_JOINTS
        ]
        last_limit_warning_joint = None

        xr_arm_ik = None
        xr_probe_ik = None
        xr_right_frame_correction = None
        if args.arm_ik == "xr_teleoperate":
            from .xr_teleoperate_ik import XrTeleoperateArmIK
            xr_arm_ik = XrTeleoperateArmIK(project, profile=args.xr_ik_profile)
            xr_probe_ik = XrTeleoperateArmIK(project, profile=args.xr_ik_profile)
            dual_q = np.r_[
                robot.data.joint_pos[0, left_arm_joint_ids].cpu().numpy(),
                robot.data.joint_pos[0, arm_joint_ids].cpu().numpy(),
            ]
            left_limits = robot.data.soft_joint_pos_limits[0, left_arm_joint_ids].cpu().numpy()
            right_limits = robot.data.soft_joint_pos_limits[0, arm_joint_ids].cpu().numpy()
            dual_lower = np.r_[left_limits[:, 0], right_limits[:, 0]]
            dual_upper = np.r_[left_limits[:, 1], right_limits[:, 1]]
            model_right = np.asarray(
                xr_arm_ik.initialize(dual_q, dual_lower, dual_upper)["right"]
            )
            xr_probe_ik.initialize(dual_q, dual_lower, dual_upper)
            root_pose = robot.data.root_pose_w
            actual_right = robot.data.body_pose_w[:, right_hand_index]
            actual_pos_b, actual_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7], actual_right[:, :3], actual_right[:, 3:7]
            )
            actual_right_matrix = np.eye(4)
            actual_right_matrix[:3, :3] = matrix_from_quat(actual_quat_b)[0].cpu().numpy()
            actual_right_matrix[:3, 3] = actual_pos_b[0].cpu().numpy()
            xr_right_frame_correction = np.linalg.inv(model_right) @ actual_right_matrix
            print(
                "[arm-ik] using xr_teleoperate G1_29_ArmIK (CasADi/IPOPT)",
                file=sys.stderr, flush=True,
            )

        def xr_target_matrix(target_pos_w, target_quat_w):
            root_pose = robot.data.root_pose_w
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7],
                target_pos_w.unsqueeze(0), target_quat_w.unsqueeze(0),
            )
            actual_target = np.eye(4)
            actual_target[:3, :3] = matrix_from_quat(target_quat_b)[0].cpu().numpy()
            actual_target[:3, 3] = target_pos_b[0].cpu().numpy()
            return actual_target @ np.linalg.inv(xr_right_frame_correction), actual_target

        def probe_xr_endpoint(name, target_pos_w, target_quat_w, dual_q):
            xr_target, actual_target = xr_target_matrix(target_pos_w, target_quat_w)
            result = xr_probe_ik.probe(xr_target, dual_q)
            predicted_actual = result["right"] @ xr_right_frame_correction
            position_error = float(np.linalg.norm(
                predicted_actual[:3, 3] - actual_target[:3, 3]
            ))
            rotation_delta = predicted_actual[:3, :3] @ actual_target[:3, :3].T
            rotation_error = float(np.arccos(np.clip(
                (np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0
            )))
            q = result["q"]
            margins = np.minimum(q - dual_lower, dual_upper - q)
            endpoint = {
                "name": name,
                "solver_success": result["solver_success"],
                "position_error_m": position_error,
                "rotation_error_deg": float(np.degrees(rotation_error)),
                "minimum_joint_margin_rad": float(margins.min()),
                "within_10mm_15deg": bool(
                    result["solver_success"] and position_error <= 0.01
                    and rotation_error <= np.deg2rad(15.0)
                ),
            }
            print(f"[IK endpoint probe] {endpoint}", file=sys.stderr, flush=True)
            return endpoint, q

        def plan_episode(index: int):
            front = env.scene["front_camera"].data
            object_pos = env.scene["object"].data.root_pos_w[0]
            # LabUtopia Beaker_01's rigid root is at the bottom, not at the
            # visible/collision center. HUG must be conditioned on the body.
            center_offset_z = 0.036 if args.object_shape == "beaker" else 0.0
            object_center = object_pos + torch.tensor([0.0, 0.0, center_offset_z], device=env.device)
            camera_pos = front.pos_w[0]
            camera_rot = matrix_from_quat(front.quat_w_ros[0])
            point_camera = camera_rot.T @ (object_center - camera_pos)
            K_tensor = front.intrinsic_matrices[0]
            projected = K_tensor @ point_camera
            u = float(projected[0] / projected[2])
            v = float(projected[1] / projected[2])
            height, width = front.output["rgb"].shape[1:3]
            square = min(height, width)
            u_224 = (u - (width - square) / 2.0) * 224.0 / square
            v_224 = (v - (height - square) / 2.0) * 224.0 / square
            if not (0 <= u_224 < 224 and 0 <= v_224 < 224):
                raise RuntimeError(f"烧杯条件点不在 HUG 图像内: {(u_224, v_224)}")
            start_pos = robot.data.body_pos_w[0, right_hand_index].clone()
            start_quat = robot.data.body_quat_w[0, right_hand_index].clone()
            hug_candidate_stats = []
            hug_sampling_stats = None
            selected_hug_candidate = None
            if args.nearby_ik_test:
                # This target is only 67 mm from the reset wrist pose and keeps
                # the exact reset orientation. It isolates arm IK from HUG and
                # places the cylinder about 96 mm in front of the hand base.
                grasp_pos = start_pos + torch.tensor(
                    [-0.040, 0.050, 0.020], device=env.device
                )
                grasp_quat = start_quat.clone()
                hand_target = args.hand_closure_scale * np.array(
                    [1.5, 1.5, 1.5, 1.5, 0.45, 0.6], dtype=np.float32
                )
                hug_distance = float(torch.linalg.vector_norm(grasp_pos - object_center))
                planning_valid = True
                planning_failure = None
                print(
                    f"[nearby-ik-test] start={start_pos.cpu().tolist()} "
                    f"grasp={grasp_pos.cpu().tolist()} object={object_center.cpu().tolist()}",
                    file=sys.stderr, flush=True,
                )
            elif replay_plan is not None:
                root_pose = robot.data.root_pose_w[0]
                root_rotation = matrix_from_quat(root_pose[3:7].unsqueeze(0))[0]

                def replay_pose(name):
                    pose_base = torch.tensor(
                        replay_plan[name], device=env.device, dtype=root_pose.dtype
                    )
                    position = root_pose[:3] + root_rotation @ pose_base[:3, 3]
                    rotation = root_rotation @ pose_base[:3, :3]
                    return position, quat_from_matrix(rotation.unsqueeze(0))[0]

                pregrasp_pos, grasp_quat = replay_pose("T_base_pregrasp")
                grasp_pos, grasp_quat = replay_pose("T_base_grasp")
                lift_pos, _ = replay_pose("T_base_lift")
                hand_target = args.hand_closure_scale * np.asarray(
                    replay_plan["hand_target_rad"], dtype=np.float32
                )
                hug_distance = float(torch.linalg.vector_norm(grasp_pos - object_center))
                planning_valid = True
                planning_failure = None
                selected_hug_candidate = "real_rgbd_replay"
                print(
                    f"[real-plan replay] object={object_center.cpu().tolist()} "
                    f"pregrasp={pregrasp_pos.cpu().tolist()} grasp={grasp_pos.cpu().tolist()} "
                    f"lift={lift_pos.cpu().tolist()}",
                    file=sys.stderr, flush=True,
                )
            elif args.use_hug and args.auto_collect:
                from .hug_bridge import run_hug_capture
                hug_candidates = run_hug_capture(
                    project=project, episode_index=index,
                    rgb=front.output["rgb"][0, ..., :3].cpu().numpy(),
                    depth_m=front.output["distance_to_image_plane"][0].cpu().numpy(),
                    K=K_tensor.cpu().numpy(), point_uv_224=(u_224, v_224),
                    object_name=args.object_shape,
                    sampling_steps=args.hug_sampling_steps,
                    candidates=args.hug_candidates,
                    return_candidates=True,
                )
                evaluated = []
                for name, wrist_camera, candidate_hand in hug_candidates:
                    wrist_camera_t = torch.tensor(wrist_camera, device=env.device)
                    candidate_pos = camera_pos + camera_rot @ wrist_camera_t[:3, 3]
                    candidate_rot = camera_rot @ wrist_camera_t[:3, :3]
                    candidate_quat = quat_from_matrix(candidate_rot.unsqueeze(0))[0]
                    direction = candidate_pos - object_center
                    distance = float(torch.linalg.vector_norm(direction))
                    direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=1e-6)
                    candidate_pregrasp = candidate_pos + max(0.0, args.pregrasp_offset_m) * direction

                    in_distance_range = 0.10 <= distance <= 0.22
                    # Do not infer Isaac reachability from the separate Pinocchio
                    # URDF. Prefer a geometrically plausible HUG wrist distance;
                    # the Isaac-native online IK is solely responsible for motion.
                    score = abs(distance - 0.14) + (0.0 if in_distance_range else 10.0)
                    evaluated.append({
                        "name": name, "score": score, "in_distance_range": in_distance_range,
                        "grasp_pos": candidate_pos, "pregrasp_pos": candidate_pregrasp,
                        "grasp_quat": candidate_quat, "hand": candidate_hand,
                        "distance": distance,
                    })
                    hug_candidate_stats.append({
                        "name": name,
                        "wrist_xyz": candidate_pos.cpu().tolist(),
                        "wrist_quat_wxyz": candidate_quat.cpu().tolist(),
                        "object_distance_m": distance,
                    })
                    print(
                        f"[HUG candidate] {name} distance={distance:.3f} "
                        f"in_distance_range={in_distance_range}",
                        file=sys.stderr, flush=True,
                    )
                selected = min(evaluated, key=lambda item: item["score"])
                selected_hug_candidate = selected["name"]
                grasp_pos = selected["grasp_pos"]
                grasp_quat = selected["grasp_quat"]
                hand_target = selected["hand"]
                hug_distance = selected["distance"]
                planning_valid = True
                planning_failure = None
                candidate_positions = torch.stack([item["grasp_pos"] for item in evaluated])
                candidate_quats = torch.stack([item["grasp_quat"] for item in evaluated])
                pairwise_translation = torch.cdist(candidate_positions, candidate_positions)
                pairwise_quat_dot = torch.clamp(
                    torch.abs(candidate_quats @ candidate_quats.T), max=1.0
                )
                pairwise_rotation = 2.0 * torch.acos(pairwise_quat_dot)
                upper = torch.triu_indices(
                    len(evaluated), len(evaluated), offset=1, device=env.device
                )
                translation_values = pairwise_translation[upper[0], upper[1]]
                rotation_values = pairwise_rotation[upper[0], upper[1]]
                if len(evaluated) == 1:
                    translation_values = torch.zeros(1, device=env.device)
                    rotation_values = torch.zeros(1, device=env.device)
                object_distances = torch.tensor(
                    [item["distance"] for item in evaluated], device=env.device
                )
                hug_sampling_stats = {
                    "candidate_count": len(evaluated),
                    "object_distance_mean_m": float(object_distances.mean()),
                    "object_distance_std_m": float(object_distances.std(unbiased=False)),
                    "object_distance_min_m": float(object_distances.min()),
                    "object_distance_max_m": float(object_distances.max()),
                    "pairwise_translation_mean_m": float(translation_values.mean()),
                    "pairwise_translation_max_m": float(translation_values.max()),
                    "pairwise_rotation_mean_deg": float(torch.rad2deg(rotation_values).mean()),
                    "pairwise_rotation_max_deg": float(torch.rad2deg(rotation_values).max()),
                }
                print(
                    f"[HUG candidate] selected={selected['name']} "
                    f"distance={selected['distance']:.3f}",
                    file=sys.stderr, flush=True,
                )
                print(
                    "[HUG sampling spread] "
                    f"translation_mean_m={hug_sampling_stats['pairwise_translation_mean_m']:.3f} "
                    f"translation_max_m={hug_sampling_stats['pairwise_translation_max_m']:.3f} "
                    f"rotation_mean_deg={hug_sampling_stats['pairwise_rotation_mean_deg']:.1f} "
                    f"rotation_max_deg={hug_sampling_stats['pairwise_rotation_max_deg']:.1f}",
                    file=sys.stderr, flush=True,
                )
            else:
                hand_target = args.hand_closure_scale * np.array(
                    [1.5, 1.5, 1.5, 1.5, 0.45, 0.6], dtype=np.float32
                )
                # Thumb is ~4 cm to the hand's -X side. Shift the hand base so
                # the object sits between thumb and four fingers.
                # Keep the palm body clear of the tabletop. A lower target can
                # visually surround the object but pins the closed hand against
                # the table, preventing the arm from executing the lift.
                grasp_pos = object_center + torch.tensor([-0.080, -0.115, 0.005], device=env.device)
                grasp_quat = robot.data.body_quat_w[0, right_hand_index].clone()
                planning_valid = True
                planning_failure = None
            # Retreat radially away from the object rather than towards the
            # hand's initial position.  The latter can put the pre-grasp on the
            # object side of a bad/noisy grasp and make the open hand collide
            # with the object before executing the final approach.
            if replay_plan is not None:
                pass
            elif args.nearby_ik_test:
                pregrasp_pos = torch.lerp(start_pos, grasp_pos, 0.5)
            else:
                retreat_direction = grasp_pos - object_center
                retreat_norm = torch.linalg.vector_norm(retreat_direction)
                if float(retreat_norm) < 1e-6:
                    # A grasp exactly at the object center has no radial direction;
                    # use the current hand side as a deterministic fallback.
                    retreat_direction = start_pos - object_center
                    retreat_norm = torch.linalg.vector_norm(retreat_direction)
                retreat_direction = retreat_direction / torch.clamp(retreat_norm, min=1e-6)
                pregrasp_pos = grasp_pos + max(0.0, args.pregrasp_offset_m) * retreat_direction
            if replay_plan is None:
                lift_pos = grasp_pos + torch.tensor([0.0, 0.0, 0.12], device=env.device)
            endpoint_probes = None
            if xr_probe_ik is not None:
                probe_q = np.r_[
                    robot.data.joint_pos[0, left_arm_joint_ids].cpu().numpy(),
                    robot.data.joint_pos[0, arm_joint_ids].cpu().numpy(),
                ]
                pregrasp_probe, probe_q = probe_xr_endpoint(
                    "pregrasp", pregrasp_pos, grasp_quat, probe_q
                )
                grasp_probe, _probe_q = probe_xr_endpoint(
                    "grasp", grasp_pos, grasp_quat, probe_q
                )
                endpoint_probes = {"pregrasp": pregrasp_probe, "grasp": grasp_probe}
            within_coarse_workspace = (
                hug_distance <= 0.35 if (args.use_hug and args.auto_collect) or replay_plan is not None
                else True
            )
            print(
                "[HUG target] "
                f"episode={index:06d} xyz={grasp_pos.cpu().tolist()} "
                f"quat_wxyz={grasp_quat.cpu().tolist()} "
                f"object_distance_m={float(torch.linalg.vector_norm(grasp_pos - object_center)):.3f} "
                f"pregrasp_object_distance_m={float(torch.linalg.vector_norm(pregrasp_pos - object_center)):.3f} "
                f"within_coarse_workspace={within_coarse_workspace}",
                file=sys.stderr, flush=True,
            )
            if not planning_valid:
                print(
                    f"[sim-collect] episode {index:06d} rejected before motion: {planning_failure}",
                    file=sys.stderr, flush=True,
                )
            return {
                "start_pos": start_pos, "start_quat": start_quat,
                "pregrasp_pos": pregrasp_pos, "grasp_pos": grasp_pos,
                "lift_pos": lift_pos, "grasp_quat": grasp_quat,
                "hand_target": np.asarray(hand_target, dtype=np.float32),
                "uv_224": (u_224, v_224),
                "planning_valid": planning_valid,
                "planning_failure": planning_failure,
                "selected_hug_candidate": selected_hug_candidate,
                "hug_candidate_stats": hug_candidate_stats,
                "hug_sampling_stats": hug_sampling_stats,
                "pregrasp_reached": False,
                "grasp_reached": False,
                "failure_category": None,
                "endpoint_ik_probes": endpoint_probes,
            }

        def target_for_phase(plan: dict, phase: int):
            if not plan["planning_valid"]:
                return plan["start_pos"], plan["start_quat"], np.zeros(6, dtype=np.float32)
            if phase < pregrasp_end:
                alpha = phase / float(pregrasp_end - 1)
                pos = torch.lerp(plan["start_pos"], plan["pregrasp_pos"], alpha)
                quat = quat_slerp(plan["start_quat"], plan["grasp_quat"], alpha)
                hand = np.zeros(6, dtype=np.float32)
            elif phase < grasp_end:
                alpha = (phase - pregrasp_end) / float(grasp_end - pregrasp_end - 1)
                pos = torch.lerp(plan["pregrasp_pos"], plan["grasp_pos"], alpha)
                quat = plan["grasp_quat"]
                hand = np.zeros(6, dtype=np.float32)
            elif phase < close_end:
                alpha = (phase - grasp_end) / float(close_end - grasp_end - 1)
                pos, quat = plan["grasp_pos"], plan["grasp_quat"]
                hand = plan["hand_target"] * (alpha * alpha * (3.0 - 2.0 * alpha))
            elif phase < lift_end:
                alpha = (phase - close_end) / float(lift_end - close_end - 1)
                pos = torch.lerp(plan["grasp_pos"], plan["lift_pos"], alpha)
                quat, hand = plan["grasp_quat"], plan["hand_target"]
            else:
                pos, quat, hand = plan["lift_pos"], plan["grasp_quat"], plan["hand_target"]
            return pos, quat, hand

        def visualize_plan(plan: dict) -> None:
            hug_marker.visualize(plan["grasp_pos"].unsqueeze(0), plan["grasp_quat"].unsqueeze(0))
            pregrasp_marker.visualize(
                plan["pregrasp_pos"].unsqueeze(0), plan["grasp_quat"].unsqueeze(0)
            )
            alpha_reach = torch.linspace(0.0, 1.0, 24, device=env.device).unsqueeze(1)
            alpha_short = torch.linspace(0.0, 1.0, 12, device=env.device).unsqueeze(1)
            reach_path_marker.visualize(
                plan["start_pos"].unsqueeze(0)
                + alpha_reach * (plan["pregrasp_pos"] - plan["start_pos"]).unsqueeze(0)
            )
            approach_path_marker.visualize(
                plan["pregrasp_pos"].unsqueeze(0)
                + alpha_short * (plan["grasp_pos"] - plan["pregrasp_pos"]).unsqueeze(0)
            )
            lift_path_marker.visualize(
                plan["grasp_pos"].unsqueeze(0)
                + alpha_short * (plan["lift_pos"] - plan["grasp_pos"]).unsqueeze(0)
            )
        expanded = compact_to_isaac_action(np.r_[arm_default, np.zeros(6, dtype=np.float32)], joint_names, default_pos)
        writer = None
        capture_stride = 10
        root = Path(args.output)
        episode_index = 0
        completed_episodes = 0
        visual_counts = {"success": 0, "failed": 0}
        def next_writer():
            nonlocal episode_index
            while any((root / f"episode_{episode_index:06d}{suffix}").exists()
                      for suffix in ("", ".incomplete", ".failed")):
                episode_index += 1
            current_index = episode_index
            result = EpisodeWriter(root, current_index, {
                "instruction": f"用右手抓起{args.object_shape}并抬升",
                "robot_type": "unitree_g1_right_inspire_rh56dftp",
                "source": (
                    "real_rgbd_hug_replayed_in_isaac"
                    if replay_plan is not None else
                    "isaac_rgbd_hug_mano_diff_ik" if args.use_hug else
                    "isaac_object_pose_diff_ik"
                ),
                "fps": 30,
                "table_top_height_m": args.table_top_height_m,
                "waypoint_tolerance_m": args.waypoint_tolerance_m,
                "real_plan": str(args.replay_real_plan.resolve()) if args.replay_real_plan else None,
            })
            episode_index += 1
            return result, current_index
        if args.auto_collect:
            from .dataset import EpisodeWriter
            writer, current_episode_index = next_writer()
        else:
            current_episode_index = 0
        plan = plan_episode(current_episode_index)
        visualize_plan(plan)
        print(f"[sim-collect] episode {current_episode_index:06d} planned", file=sys.stderr, flush=True)
        import contextlib
        with TerminalKeyReader(enabled=not args.headless) as keys, open("/dev/null", "w") as sink:
          print("[sim-collect] terminal shortcut: press r to reset", file=sys.stderr, flush=True)
          initial_beaker_z = float(env.scene["object"].data.root_pos_w[0, 2])
          initial_object_xyz = env.scene["object"].data.root_pos_w[0].cpu().tolist()
          initial_hand_z = float(robot.data.body_pos_w[0, right_hand_index, 2])
          stability_samples = []
          clearance_samples = []
          right_arm_contact_force_max_n = 0.0
          episode_start_step = 0
          motion_phase = 0 if plan["planning_valid"] else cycle_steps - 1
          waypoint_wait_steps = 0
          approach_recorded_phases = set()
          approach_diagnostics = {
              "sample_count": 0,
              "max_target_tracking_error_m": 0.0,
              "max_cross_track_error_m": 0.0,
              "max_backward_progress": 0.0,
              "first_tracking_error_over_0.04_alpha": None,
              "final_progress": None,
          }
          previous_approach_progress = None
          arm_command = torch.tensor(arm_default, device=env.device)
          for step in range(args.steps):
            phase = motion_phase
            target_pos_w, target_quat_w, hand_command = target_for_phase(plan, phase)
            root_pose = robot.data.root_pose_w
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7],
                target_pos_w.unsqueeze(0), target_quat_w.unsqueeze(0),
            )
            ee_pose_w = robot.data.body_pose_w[:, right_hand_index]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose[:, :3], root_pose[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7],
            )
            arm_current = robot.data.joint_pos[:, arm_joint_ids]
            pos_error, rot_error = compute_pose_error(
                ee_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle"
            )
            predicted_delta = None
            if xr_arm_ik is not None:
                dual_q = np.r_[
                    robot.data.joint_pos[0, left_arm_joint_ids].cpu().numpy(),
                    arm_current[0].cpu().numpy(),
                ]
                dual_dq = np.r_[
                    robot.data.joint_vel[0, left_arm_joint_ids].cpu().numpy(),
                    robot.data.joint_vel[0, arm_joint_ids].cpu().numpy(),
                ]
                right_target, _actual_target = xr_target_matrix(target_pos_w, target_quat_w)
                solution_q = xr_arm_ik.solve(right_target, dual_q, dual_dq)
                arm_desired = torch.tensor(solution_q[-7:], device=env.device, dtype=arm_current.dtype)
            else:
                jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_index, :, arm_joint_ids]
                world_to_base = matrix_from_quat(quat_inv(root_pose[:, 3:7]))
                jacobian[:, :3, :] = torch.bmm(world_to_base, jacobian[:, :3, :])
                jacobian[:, 3:, :] = torch.bmm(world_to_base, jacobian[:, 3:, :])
                damping = 0.05
                if phase < close_end:
                    rotation_weight = float(np.clip(args.ik_rotation_weight, 0.0, 1.0))
                    task_weights = torch.tensor(
                        [1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight],
                        device=env.device,
                    )
                    task_jacobian = jacobian * task_weights.view(1, 6, 1)
                    task_error = torch.cat((pos_error, rot_error), dim=-1) * task_weights
                else:
                    task_jacobian = jacobian[:, :3, :]
                    task_error = pos_error
                task_dim = task_jacobian.shape[1]
                normal = torch.bmm(task_jacobian, task_jacobian.transpose(1, 2))
                normal += (damping * damping) * torch.eye(task_dim, device=env.device).unsqueeze(0)
                delta_joint = torch.bmm(
                    task_jacobian.transpose(1, 2),
                    torch.linalg.solve(normal, task_error.unsqueeze(-1)),
                ).squeeze(-1)
                max_joint_step_rad = float(np.clip(args.ik_max_joint_step, 0.005, 0.15))
                if pregrasp_end <= phase < grasp_end:
                    max_joint_step_rad *= 0.5
                scale = torch.clamp(
                    max_joint_step_rad / torch.clamp(delta_joint[0].abs().max(), min=1e-9), max=1.0
                )
                integration = float(np.clip(args.ik_command_integration, 0.0, 1.0))
                arm_desired = (
                    arm_current[0] + delta_joint[0] * scale
                    + integration * (arm_command - arm_current[0])
                )
                predicted_delta = torch.mv(task_jacobian[0], arm_desired - arm_current[0])
            command_lead = float(np.clip(args.ik_max_command_lead, 0.01, 0.15))
            if pregrasp_end <= phase < grasp_end:
                command_lead *= 2.0 / 3.0
            arm_desired = torch.clamp(
                arm_desired, arm_current[0] - command_lead, arm_current[0] + command_lead
            )
            limits = robot.data.soft_joint_pos_limits[0, arm_joint_ids]
            arm_desired = torch.clamp(arm_desired, limits[:, 0], limits[:, 1])
            joint_margins = torch.minimum(
                arm_current[0] - limits[:, 0], limits[:, 1] - arm_current[0]
            )
            minimum_margin_index = int(torch.argmin(joint_margins))
            minimum_margin = float(joint_margins[minimum_margin_index])
            warning_threshold = max(0.0, float(args.joint_limit_warning_rad))
            warning_joint = RIGHT_ARM_JOINTS[minimum_margin_index]
            if minimum_margin <= warning_threshold:
                warning_body_id = arm_joint_body_ids[minimum_margin_index]
                limit_marker.visualize(robot.data.body_pos_w[:, warning_body_id])
                if warning_joint != last_limit_warning_joint:
                    current_angle = float(arm_current[0, minimum_margin_index])
                    print(
                        f"[joint-limit-warning] joint={warning_joint} "
                        f"q={current_angle:.4f} margin_rad={minimum_margin:.4f}",
                        file=sys.stderr, flush=True,
                    )
                last_limit_warning_joint = warning_joint
            else:
                limit_marker.visualize(hidden_limit_marker_pos)
                if last_limit_warning_joint is not None:
                    print("[joint-limit-warning] cleared", file=sys.stderr, flush=True)
                last_limit_warning_joint = None
            arm_command = arm_desired.detach()
            if args.debug_ik and phase in (
                0, pregrasp_end - 1, grasp_end - 1,
                close_end - 1, lift_end - 1, cycle_steps - 1,
            ):
                print(
                    f"[IK debug] phase={phase} ee={ee_pose_w[0, :3].cpu().tolist()} "
                    f"target={target_pos_w.cpu().tolist()} error_b={pos_error[0].cpu().tolist()} "
                    f"rotation_error_rad={float(torch.linalg.vector_norm(rot_error[0])):.4f} "
                    f"predicted_delta={predicted_delta[:3].cpu().tolist() if predicted_delta is not None else 'xr_teleoperate'} "
                    f"joint_limit_margin_rad={float(joint_margins[minimum_margin_index]):.4f} "
                    f"limiting_joint={RIGHT_ARM_JOINTS[minimum_margin_index]}",
                    file=sys.stderr, flush=True,
                )
            compact = np.r_[arm_desired.cpu().numpy(), hand_command].astype(np.float32)
            expanded = compact_to_isaac_action(compact, joint_names, default_pos)
            action = torch.tensor(expanded, device=env.device).unsqueeze(0)
            with contextlib.redirect_stdout(sink):
                observation, *_ = env.step(action)
            non_finger_right_ids = [right_hand_index, *arm_joint_body_ids]
            body_heights = robot.data.body_pos_w[0, non_finger_right_ids, 2]
            clearance_samples.append(float(body_heights.min()) - args.table_top_height_m)
            contact_forces = getattr(robot.data, "net_contact_forces_w", None)
            if contact_forces is not None:
                right_arm_contact_force_max_n = max(
                    right_arm_contact_force_max_n,
                    float(torch.linalg.vector_norm(
                        contact_forces[0, non_finger_right_ids], dim=1
                    ).max()),
                )
            if pregrasp_end <= phase < grasp_end and phase not in approach_recorded_phases:
                approach_recorded_phases.add(phase)
                actual_after = robot.data.body_pos_w[0, right_hand_index]
                line = plan["grasp_pos"] - plan["pregrasp_pos"]
                line_length_sq = torch.clamp(torch.dot(line, line), min=1e-9)
                progress = float(torch.dot(actual_after - plan["pregrasp_pos"], line) / line_length_sq)
                clamped_progress = float(np.clip(progress, 0.0, 1.0))
                closest = plan["pregrasp_pos"] + clamped_progress * line
                cross_track = float(torch.linalg.vector_norm(actual_after - closest))
                tracking_error = float(torch.linalg.vector_norm(actual_after - target_pos_w))
                commanded_alpha = (phase - pregrasp_end) / float(grasp_end - pregrasp_end - 1)
                approach_diagnostics["sample_count"] += 1
                approach_diagnostics["max_target_tracking_error_m"] = max(
                    approach_diagnostics["max_target_tracking_error_m"], tracking_error
                )
                approach_diagnostics["max_cross_track_error_m"] = max(
                    approach_diagnostics["max_cross_track_error_m"], cross_track
                )
                if previous_approach_progress is not None:
                    approach_diagnostics["max_backward_progress"] = max(
                        approach_diagnostics["max_backward_progress"],
                        previous_approach_progress - progress,
                    )
                previous_approach_progress = progress
                approach_diagnostics["final_progress"] = progress
                if (tracking_error > 0.04 and
                        approach_diagnostics["first_tracking_error_over_0.04_alpha"] is None):
                    approach_diagnostics["first_tracking_error_over_0.04_alpha"] = commanded_alpha
            if writer is not None and (phase % capture_stride == 0 or phase == cycle_steps - 1):
                robot_pos = env.scene["robot"].data.joint_pos[0].cpu().numpy()
                arm_state = np.array([robot_pos[joint_names.index(name)] for name in RIGHT_ARM_JOINTS])
                front = env.scene["front_camera"].data.output
                wrist = env.scene["right_wrist_camera"].data.output
                writer.add_frame(
                    timestamp=(step - episode_start_step) * 0.01,
                    state=np.r_[arm_state, compact[7:]], action=compact,
                    images={"front": front["rgb"][0, ..., :3].cpu().numpy(),
                            "right_wrist": wrist["rgb"][0, ..., :3].cpu().numpy()},
                    depth=front["distance_to_image_plane"][0].cpu().numpy(),
                    ee_actual=ee_pose_w[0].cpu().numpy(),
                    ee_target=np.r_[
                        target_pos_w.cpu().numpy(), target_quat_w.cpu().numpy()
                    ],
                )
            manual_reset = keys.read() == "r"
            if manual_reset:
                print("[sim-collect] manual reset requested", file=sys.stderr, flush=True)
            if phase >= cycle_steps - 30:
                object_data = env.scene["object"].data
                object_pos = object_data.root_pos_w[0]
                finger_positions = env.scene["robot"].data.body_pos_w[0, right_finger_indices]
                nearest_finger_distance = torch.linalg.vector_norm(
                    finger_positions - object_pos.unsqueeze(0), dim=1
                ).min()
                stability_samples.append((
                    float(object_pos[2]) - initial_beaker_z,
                    float(nearest_finger_distance),
                    float(torch.linalg.vector_norm(object_data.root_vel_w[0, :3])),
                    float(torch.linalg.vector_norm(object_data.root_vel_w[0, 3:])),
                    float(robot.data.body_pos_w[0, right_hand_index, 2]) - initial_hand_z,
                    float(torch.linalg.vector_norm(target_pos_w - robot.data.body_pos_w[0, right_hand_index])),
                ))
            if writer is not None and (phase == cycle_steps - 1 or manual_reset):
                if manual_reset:
                    metrics = {"manual_reset": True, "phase": phase}
                    checks = {"manual_reset": False}
                else:
                    samples = np.asarray(stability_samples, dtype=np.float64)
                    metrics = {
                        "min_lift_m": float(samples[:, 0].min()),
                        "max_finger_distance_m": float(samples[:, 1].max()),
                        "max_linear_speed_mps": float(samples[:, 2].max()),
                        "max_angular_speed_radps": float(samples[:, 3].max()),
                        "min_hand_lift_m": float(samples[:, 4].min()),
                        "max_hand_target_error_m": float(samples[:, 5].max()),
                        "object_initial_xyz": initial_object_xyz,
                        "grasp_target_xyz": plan["grasp_pos"].cpu().tolist(),
                        "pregrasp_target_xyz": plan["pregrasp_pos"].cpu().tolist(),
                        "pregrasp_offset_m": max(0.0, args.pregrasp_offset_m),
                        "planning_failure": plan["planning_failure"],
                        "selected_hug_candidate": plan["selected_hug_candidate"],
                        "hug_candidate_stats": plan["hug_candidate_stats"],
                        "hug_sampling_stats": plan["hug_sampling_stats"],
                        "pregrasp_reached": plan["pregrasp_reached"],
                        "grasp_reached": plan["grasp_reached"],
                        "failure_category": plan["failure_category"],
                        "approach_diagnostics": approach_diagnostics,
                        "endpoint_ik_probes": plan["endpoint_ik_probes"],
                        "table_top_height_m": args.table_top_height_m,
                        "min_right_arm_body_origin_clearance_m": min(clearance_samples),
                        "max_nonfinger_right_arm_contact_force_n": right_arm_contact_force_max_n,
                        "waypoint_tolerance_m": args.waypoint_tolerance_m,
                    }
                    checks = {
                        "planning_valid": bool(plan["planning_valid"]),
                        "lift_ge_0.03m": metrics["min_lift_m"] >= 0.03,
                        "finger_distance_le_0.10m": metrics["max_finger_distance_m"] <= 0.10,
                        "linear_speed_le_0.15mps": metrics["max_linear_speed_mps"] <= 0.15,
                        "angular_speed_le_1.0radps": metrics["max_angular_speed_radps"] <= 1.0,
                        "nonfinger_contact_force_le_5n": (
                            metrics["max_nonfinger_right_arm_contact_force_n"] <= 5.0
                        ),
                    }
                metrics["checks"] = checks
                status = "success" if all(checks.values()) else "failed"
                visual_limit = (
                    args.keep_success_visuals if status == "success" else args.keep_failure_visuals
                )
                keep_visuals = visual_counts[status] < max(0, visual_limit)
                saved = writer.finish(
                    status=status, metrics=metrics, keep_visuals=keep_visuals
                )
                if keep_visuals:
                    visual_counts[status] += 1
                completed_episodes += 1
                print(f"[sim-collect] {status}: {saved} {metrics}", file=sys.stderr, flush=True)
                writer = None
                if args.episodes is not None and completed_episodes >= args.episodes:
                    print(
                        f"[sim-collect] requested episodes completed: {completed_episodes}",
                        file=sys.stderr, flush=True,
                    )
                    break
                # Do not create a trailing .incomplete episode in finite runs.
                enough_for_next = args.steps - (step + 1) >= cycle_steps
                if enough_for_next or (manual_reset and step + 1 < args.steps):
                    print("[sim-collect] resetting environment", file=sys.stderr, flush=True)
                    with contextlib.redirect_stdout(sink):
                        observation = reset_scene()
                    initial_beaker_z = float(env.scene["object"].data.root_pos_w[0, 2])
                    initial_object_xyz = env.scene["object"].data.root_pos_w[0].cpu().tolist()
                    initial_hand_z = float(robot.data.body_pos_w[0, right_hand_index, 2])
                    arm_command = torch.tensor(arm_default, device=env.device)
                    stability_samples.clear()
                    clearance_samples.clear()
                    right_arm_contact_force_max_n = 0.0
                    episode_start_step = step + 1
                    motion_phase = 0
                    waypoint_wait_steps = 0
                    approach_recorded_phases.clear()
                    approach_diagnostics = {
                        "sample_count": 0,
                        "max_target_tracking_error_m": 0.0,
                        "max_cross_track_error_m": 0.0,
                        "max_backward_progress": 0.0,
                        "first_tracking_error_over_0.04_alpha": None,
                        "final_progress": None,
                    }
                    previous_approach_progress = None
                    writer, current_episode_index = next_writer()
                    print(
                        f"[sim-collect] episode {current_episode_index:06d} planning with fresh RGB-D",
                        file=sys.stderr, flush=True,
                    )
                    plan = plan_episode(current_episode_index)
                    if not plan["planning_valid"]:
                        motion_phase = cycle_steps - 1
                    visualize_plan(plan)
                    print(f"[sim-collect] episode {current_episode_index:06d} running", file=sys.stderr, flush=True)
                    continue
            # Do not start the straight approach until the measured hand has
            # actually reached pre-grasp. Likewise, do not close before the
            # measured hand reaches grasp. A bounded wait still guarantees a
            # terminal outcome for unreachable targets.
            measured_error = float(torch.linalg.vector_norm(
                target_pos_w - robot.data.body_pos_w[0, right_hand_index]
            ))
            gate = None
            if phase == pregrasp_end - 1:
                gate = ("pre-grasp", args.waypoint_tolerance_m, 100)
            elif phase == grasp_end - 1:
                gate = ("grasp", args.waypoint_tolerance_m, 100)
            if gate is not None and measured_error > gate[1]:
                if waypoint_wait_steps < gate[2]:
                    waypoint_wait_steps += 1
                    if waypoint_wait_steps == 1 or waypoint_wait_steps % 50 == 0:
                        print(
                            f"[IK gate] waiting_at={gate[0]} error_m={measured_error:.4f} "
                            f"wait_steps={waypoint_wait_steps}", file=sys.stderr, flush=True,
                        )
                else:
                    failure = (
                        f"IK {gate[0]} tracking timeout: error {measured_error:.4f} m "
                        f"after {waypoint_wait_steps} wait steps"
                    )
                    plan["planning_valid"] = False
                    plan["planning_failure"] = failure
                    if gate[0] == "pre-grasp":
                        plan["failure_category"] = "pregrasp_target_unreached"
                    elif approach_diagnostics["max_cross_track_error_m"] > 0.04:
                        plan["failure_category"] = "straight_approach_deviation_and_grasp_unreached"
                    else:
                        plan["failure_category"] = "grasp_target_unreached"
                    motion_phase = cycle_steps - 1
                    print(f"[sim-collect] {failure}", file=sys.stderr, flush=True)
            else:
                if gate is not None and measured_error <= gate[1]:
                    if gate[0] == "pre-grasp":
                        plan["pregrasp_reached"] = True
                    else:
                        plan["grasp_reached"] = True
                        if approach_diagnostics["max_cross_track_error_m"] > 0.04:
                            plan["failure_category"] = "straight_approach_deviation_but_grasp_reached"
                        else:
                            plan["failure_category"] = "ik_path_completed"
                if gate is not None and waypoint_wait_steps:
                    print(
                        f"[IK gate] leaving={gate[0]} error_m={measured_error:.4f} "
                        f"wait_steps={waypoint_wait_steps}", file=sys.stderr, flush=True,
                    )
                waypoint_wait_steps = 0
                motion_phase += 1
        report = {
            "task": task_id,
            "action_shape": list(env.action_space.shape),
            "observation_groups": sorted(observation.keys()),
            "controlled_nonzero_at_default": int((expanded != 0).sum()),
            "steps": args.steps,
        }
        # Isaac Kit may terminate before conda-run flushes captured stdout; stderr is immediate.
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
        if xr_arm_ik is not None:
            xr_arm_ik.close()
        if xr_probe_ik is not None:
            xr_probe_ik.close()
        env.close()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
