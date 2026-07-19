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
    parser.add_argument("--debug-ik", action="store_true")
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
    parser.add_argument("--output", default="outputs/gui_collect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--object-randomization-m", type=float, default=0.04)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--keep-success-visuals", type=int, default=5)
    parser.add_argument("--keep-failure-visuals", type=int, default=5)
    args = parser.parse_args()
    episode_rng = random.Random(args.seed)
    os.environ["LAB_OBJECT_SHAPE"] = args.object_shape

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
    import pinocchio  # noqa: F401

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

        task_id = "Isaac-PickPlace-LabBeaker-G129-Inspire-Right"
        # Isaac Sim 5.0's GUI path passes an Ar.ResolvedPath to a Boost binding
        # that accepts only str. The checked-in LabUtopia asset is a validated
        # USD crate, so bypass only this broken preflight predicate.
        omni.usd.is_usd_crate_file_version_supported = lambda _path: True
        print("[sim-smoke] registered", file=sys.stderr, flush=True)
        cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        cfg.viewer.eye = (2.2, 2.2, 1.8)
        cfg.viewer.lookat = (-0.25, 0.45, 1.05)
        print("[sim-smoke] config parsed", file=sys.stderr, flush=True)
        env = gym.make(task_id, cfg=cfg).unwrapped
        print("[sim-smoke] environment created", file=sys.stderr, flush=True)
        zero_action = torch.zeros(env.action_space.shape, device=env.device)

        def reset_scene():
            observation, _ = env.reset()
            obj = env.scene["object"]
            default_state = obj.data.default_root_state.clone()
            default_state[:, :3] += env.scene.env_origins
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
        from isaaclab.markers.config import FRAME_MARKER_CFG
        from isaaclab.utils.math import (
            compute_pose_error, matrix_from_quat, quat_from_matrix, quat_inv,
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
            if args.use_hug and args.auto_collect:
                from .arm_ik import right_arm_frame_correction, solve_right_arm_ik
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
                root_pose = robot.data.root_pose_w
                current_arm = robot.data.joint_pos[0, arm_joint_ids].cpu().numpy()
                current_ee = robot.data.body_pose_w[:, right_hand_index]
                current_pos_b, current_quat_b = subtract_frame_transforms(
                    root_pose[:, :3], root_pose[:, 3:7],
                    current_ee[:, :3], current_ee[:, 3:7],
                )
                frame_correction = right_arm_frame_correction(
                    current_arm, current_pos_b[0].cpu().numpy(),
                    matrix_from_quat(current_quat_b)[0].cpu().numpy(),
                )
                evaluated = []
                for candidate_offset, (name, wrist_camera, candidate_hand) in enumerate(hug_candidates):
                    wrist_camera_t = torch.tensor(wrist_camera, device=env.device)
                    candidate_pos = camera_pos + camera_rot @ wrist_camera_t[:3, 3]
                    candidate_rot = camera_rot @ wrist_camera_t[:3, :3]
                    candidate_quat = quat_from_matrix(candidate_rot.unsqueeze(0))[0]
                    direction = candidate_pos - object_center
                    distance = float(torch.linalg.vector_norm(direction))
                    direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=1e-6)
                    candidate_pregrasp = candidate_pos + 0.10 * direction

                    grasp_pos_b, grasp_quat_b = subtract_frame_transforms(
                        root_pose[:, :3], root_pose[:, 3:7],
                        candidate_pos.unsqueeze(0), candidate_quat.unsqueeze(0),
                    )
                    pre_pos_b, pre_quat_b = subtract_frame_transforms(
                        root_pose[:, :3], root_pose[:, 3:7],
                        candidate_pregrasp.unsqueeze(0), candidate_quat.unsqueeze(0),
                    )
                    grasp_ik = solve_right_arm_ik(
                        grasp_pos_b[0].cpu().numpy(),
                        matrix_from_quat(grasp_quat_b)[0].cpu().numpy(), current_arm,
                        frame_correction=frame_correction,
                        seed=args.seed + index * 100 + candidate_offset,
                    )
                    pregrasp_ik = solve_right_arm_ik(
                        pre_pos_b[0].cpu().numpy(),
                        matrix_from_quat(pre_quat_b)[0].cpu().numpy(), current_arm,
                        frame_correction=frame_correction,
                        seed=args.seed + index * 100 + candidate_offset + 10_000,
                    )
                    feasible = bool(grasp_ik["reachable"] and pregrasp_ik["reachable"])
                    score = (
                        abs(distance - 0.14) * 10.0
                        + grasp_ik["position_error_m"] + pregrasp_ik["position_error_m"]
                        + 0.05 * (grasp_ik["rotation_error_rad"] + pregrasp_ik["rotation_error_rad"])
                        + 0.01 * grasp_ik["joint_motion_rad"]
                        + 0.002 / max(grasp_ik["joint_limit_margin_rad"], 1e-3)
                        + (0.0 if feasible else 100.0)
                    )
                    evaluated.append({
                        "name": name, "score": score, "feasible": feasible,
                        "grasp_pos": candidate_pos, "pregrasp_pos": candidate_pregrasp,
                        "grasp_quat": candidate_quat, "hand": candidate_hand,
                        "distance": distance, "grasp_ik": grasp_ik, "pregrasp_ik": pregrasp_ik,
                    })
                    print(
                        f"[IK candidate] {name} feasible={feasible} distance={distance:.3f} "
                        f"pre=({pregrasp_ik['position_error_m']:.3f}m,"
                        f"{pregrasp_ik['rotation_error_rad']:.3f}rad) "
                        f"grasp=({grasp_ik['position_error_m']:.3f}m,"
                        f"{grasp_ik['rotation_error_rad']:.3f}rad) "
                        f"motion={grasp_ik['joint_motion_rad']:.2f}rad "
                        f"margin={grasp_ik['joint_limit_margin_rad']:.2f}rad",
                        file=sys.stderr, flush=True,
                    )
                selected = min(evaluated, key=lambda item: item["score"])
                grasp_pos = selected["grasp_pos"]
                grasp_quat = selected["grasp_quat"]
                hand_target = selected["hand"]
                hug_distance = selected["distance"]
                planning_valid = bool(selected["feasible"] and hug_distance <= 0.35)
                planning_failure = None if planning_valid else "8个HUG候选均未通过G1右臂6D IK"
                pregrasp_arm_target = selected["pregrasp_ik"]["joint_positions"]
                grasp_arm_target = selected["grasp_ik"]["joint_positions"]
                print(
                    f"[IK candidate] selected={selected['name']} feasible={selected['feasible']}",
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
                pregrasp_arm_target = arm_default.copy()
                grasp_arm_target = arm_default.copy()
            # Retreat radially away from the object rather than towards the
            # hand's initial position.  The latter can put the pre-grasp on the
            # object side of a bad/noisy grasp and make the open hand collide
            # with the object before executing the final approach.
            retreat_direction = grasp_pos - object_center
            retreat_norm = torch.linalg.vector_norm(retreat_direction)
            if float(retreat_norm) < 1e-6:
                # A grasp exactly at the object center has no radial direction;
                # use the current hand side as a deterministic fallback.
                retreat_direction = start_pos - object_center
                retreat_norm = torch.linalg.vector_norm(retreat_direction)
            retreat_direction = retreat_direction / torch.clamp(retreat_norm, min=1e-6)
            pregrasp_pos = grasp_pos + 0.10 * retreat_direction
            lift_pos = grasp_pos + torch.tensor([0.0, 0.0, 0.12], device=env.device)
            reachable = hug_distance <= 0.35 if args.use_hug and args.auto_collect else True
            print(
                "[HUG target] "
                f"episode={index:06d} xyz={grasp_pos.cpu().tolist()} "
                f"quat_wxyz={grasp_quat.cpu().tolist()} "
                f"object_distance_m={float(torch.linalg.vector_norm(grasp_pos - object_center)):.3f} "
                f"pregrasp_object_distance_m={float(torch.linalg.vector_norm(pregrasp_pos - object_center)):.3f} "
                f"reachable={reachable}",
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
                "pregrasp_arm_target": np.asarray(pregrasp_arm_target, dtype=np.float32),
                "grasp_arm_target": np.asarray(grasp_arm_target, dtype=np.float32),
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
        expanded = compact_to_isaac_action(np.r_[arm_default, np.zeros(6, dtype=np.float32)], joint_names, default_pos)
        writer = None
        capture_stride = 3
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
                "source": "isaac_rgbd_hug_mano_diff_ik" if args.use_hug else "isaac_object_pose_diff_ik",
                "fps": 30,
            })
            episode_index += 1
            return result, current_index
        if args.auto_collect:
            from .dataset import EpisodeWriter
            writer, current_episode_index = next_writer()
        else:
            current_episode_index = 0
        plan = plan_episode(current_episode_index)
        hug_marker.visualize(plan["grasp_pos"].unsqueeze(0), plan["grasp_quat"].unsqueeze(0))
        pregrasp_marker.visualize(plan["pregrasp_pos"].unsqueeze(0), plan["grasp_quat"].unsqueeze(0))
        print(f"[sim-collect] episode {current_episode_index:06d} planned", file=sys.stderr, flush=True)
        import contextlib
        with TerminalKeyReader(enabled=not args.headless) as keys, open("/dev/null", "w") as sink:
          print("[sim-collect] terminal shortcut: press r to reset", file=sys.stderr, flush=True)
          initial_beaker_z = float(env.scene["object"].data.root_pos_w[0, 2])
          initial_hand_z = float(robot.data.body_pos_w[0, right_hand_index, 2])
          stability_samples = []
          episode_start_step = 0
          motion_phase = 0 if plan["planning_valid"] else cycle_steps - 1
          waypoint_wait_steps = 0
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
            jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_index, :, arm_joint_ids]
            world_to_base = matrix_from_quat(quat_inv(root_pose[:, 3:7]))
            jacobian[:, :3, :] = torch.bmm(world_to_base, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(world_to_base, jacobian[:, 3:, :])
            arm_current = robot.data.joint_pos[:, arm_joint_ids]
            pos_error, rot_error = compute_pose_error(
                ee_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle"
            )
            damping = 0.05
            if phase < close_end:
                task_weights = torch.tensor(
                    [1.0, 1.0, 1.0, 0.25, 0.25, 0.25], device=env.device
                )
                task_jacobian = jacobian * task_weights.view(1, 6, 1)
                task_error = torch.cat((pos_error, rot_error), dim=-1) * task_weights
            else:
                # Once contact is established, orientation is secondary. Solve an
                # explicit 3-D position task instead of padding a rank-deficient
                # 6-D task with three zero-weight rotation rows.
                task_jacobian = jacobian[:, :3, :]
                task_error = pos_error
            task_dim = task_jacobian.shape[1]
            normal = torch.bmm(task_jacobian, task_jacobian.transpose(1, 2))
            normal += (damping * damping) * torch.eye(task_dim, device=env.device).unsqueeze(0)
            delta_joint = torch.bmm(
                task_jacobian.transpose(1, 2),
                torch.linalg.solve(normal, task_error.unsqueeze(-1)),
            ).squeeze(-1)
            # Preserve the DLS direction when limiting velocity. Independent
            # per-joint clipping distorts the Cartesian direction near singularities.
            # The final Cartesian approach is deliberately slower than free-
            # space motion so the implicit actuators track the straight target
            # without overshooting and correcting backwards near the object.
            max_joint_step_rad = float(np.clip(args.ik_max_joint_step, 0.005, 0.15))
            if pregrasp_end <= phase < grasp_end:
                max_joint_step_rad *= 0.5
            scale = torch.clamp(
                max_joint_step_rad / torch.clamp(delta_joint[0].abs().max(), min=1e-9), max=1.0
            )
            # A pure q_current + delta_q command is too soft for this implicit
            # position actuator, while full accumulation repeatedly integrates
            # tracking error and overshoots. Blend a bounded fraction of the
            # DLS correction into the previous command as actuator feed-forward.
            integration = float(np.clip(args.ik_command_integration, 0.0, 1.0))
            if phase < pregrasp_end:
                posture_alpha = phase / float(pregrasp_end - 1)
                posture_target = torch.lerp(
                    torch.tensor(arm_default, device=env.device),
                    torch.tensor(plan["pregrasp_arm_target"], device=env.device),
                    posture_alpha,
                )
            elif phase < grasp_end:
                posture_alpha = (phase - pregrasp_end) / float(grasp_end - pregrasp_end - 1)
                posture_target = torch.lerp(
                    torch.tensor(plan["pregrasp_arm_target"], device=env.device),
                    torch.tensor(plan["grasp_arm_target"], device=env.device),
                    posture_alpha,
                )
            else:
                posture_target = torch.tensor(plan["grasp_arm_target"], device=env.device)
            arm_desired = (
                arm_current[0] + delta_joint[0] * scale
                + integration * (arm_command - arm_current[0])
                + 0.05 * (posture_target - arm_current[0])
            )
            command_lead = float(np.clip(args.ik_max_command_lead, 0.01, 0.15))
            if pregrasp_end <= phase < grasp_end:
                command_lead *= 2.0 / 3.0
            arm_desired = torch.clamp(
                arm_desired, arm_current[0] - command_lead, arm_current[0] + command_lead
            )
            limits = robot.data.soft_joint_pos_limits[0, arm_joint_ids]
            arm_desired = torch.clamp(arm_desired, limits[:, 0], limits[:, 1])
            arm_command = arm_desired.detach()
            if args.debug_ik and phase in (
                0, pregrasp_end - 1, grasp_end - 1,
                close_end - 1, lift_end - 1, cycle_steps - 1,
            ):
                predicted_delta = torch.mv(task_jacobian[0], arm_desired - arm_current[0])
                print(
                    f"[IK debug] phase={phase} ee={ee_pose_w[0, :3].cpu().tolist()} "
                    f"target={target_pos_w.cpu().tolist()} error_b={pos_error[0].cpu().tolist()} "
                    f"predicted_delta={predicted_delta[:3].cpu().tolist()}",
                    file=sys.stderr, flush=True,
                )
            compact = np.r_[arm_desired.cpu().numpy(), hand_command].astype(np.float32)
            expanded = compact_to_isaac_action(compact, joint_names, default_pos)
            action = torch.tensor(expanded, device=env.device).unsqueeze(0)
            with contextlib.redirect_stdout(sink):
                observation, *_ = env.step(action)
            if writer is not None and phase % capture_stride == 0:
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
                    }
                    checks = {
                        "planning_valid": bool(plan["planning_valid"]),
                        "lift_ge_0.03m": metrics["min_lift_m"] >= 0.03,
                        "finger_distance_le_0.10m": metrics["max_finger_distance_m"] <= 0.10,
                        "linear_speed_le_0.15mps": metrics["max_linear_speed_mps"] <= 0.15,
                        "angular_speed_le_1.0radps": metrics["max_angular_speed_radps"] <= 1.0,
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
                    initial_hand_z = float(robot.data.body_pos_w[0, right_hand_index, 2])
                    arm_command = torch.tensor(arm_default, device=env.device)
                    stability_samples.clear()
                    episode_start_step = step + 1
                    motion_phase = 0
                    waypoint_wait_steps = 0
                    writer, current_episode_index = next_writer()
                    print(
                        f"[sim-collect] episode {current_episode_index:06d} planning with fresh RGB-D",
                        file=sys.stderr, flush=True,
                    )
                    plan = plan_episode(current_episode_index)
                    if not plan["planning_valid"]:
                        motion_phase = cycle_steps - 1
                    hug_marker.visualize(plan["grasp_pos"].unsqueeze(0), plan["grasp_quat"].unsqueeze(0))
                    pregrasp_marker.visualize(
                        plan["pregrasp_pos"].unsqueeze(0), plan["grasp_quat"].unsqueeze(0)
                    )
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
                gate = ("pre-grasp", 0.04, 100)
            elif phase == grasp_end - 1:
                gate = ("grasp", 0.04, 100)
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
                        f"IK {gate[0]} unreachable: error {measured_error:.4f} m "
                        f"after {waypoint_wait_steps} wait steps"
                    )
                    plan["planning_valid"] = False
                    plan["planning_failure"] = failure
                    motion_phase = cycle_steps - 1
                    print(f"[sim-collect] {failure}", file=sys.stderr, flush=True)
            else:
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
        env.close()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
