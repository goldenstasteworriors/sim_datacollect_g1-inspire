from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="启动并单步验证 LabUtopia 烧杯 G1+Inspire 场景")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-collect", action="store_true")
    parser.add_argument("--use-hug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="outputs/gui_collect")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    unitree = project / "third_party/unitree_sim_isaaclab"
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
        observation, _ = env.reset()
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
            if "hand" in lowered or "wrist" in lowered:
                xyz = robot.data.body_pos_w[0, body_index].cpu().tolist()
                print(f"[sim-smoke] body {body_name} xyz {xyz}", file=sys.stderr, flush=True)
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG
        from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, quat_inv, quat_slerp, subtract_frame_transforms
        from .sim_action import RIGHT_ARM_JOINTS, compact_to_isaac_action

        joint_names = list(env.scene["robot"].joint_names)
        default_pos = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
        import numpy as np

        arm_default = np.array([default_pos[joint_names.index(name)] for name in RIGHT_ARM_JOINTS], dtype=np.float32)
        cycle_steps = 450
        arm_joint_ids = [joint_names.index(name) for name in RIGHT_ARM_JOINTS]
        right_hand_index = env.scene["robot"].body_names.index("right_hand_base_link")
        jacobian_index = right_hand_index - 1 if robot.is_fixed_base else right_hand_index
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1, device=env.device,
        )
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
            camera_pos = front.pos_w[0]
            camera_rot = matrix_from_quat(front.quat_w_ros[0])
            point_camera = camera_rot.T @ (object_pos - camera_pos)
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
            if args.use_hug and args.auto_collect:
                from .hug_bridge import run_hug_capture
                wrist_camera, hand_target = run_hug_capture(
                    project=project, episode_index=index,
                    rgb=front.output["rgb"][0, ..., :3].cpu().numpy(),
                    depth_m=front.output["distance_to_image_plane"][0].cpu().numpy(),
                    K=K_tensor.cpu().numpy(), point_uv_224=(u_224, v_224),
                )
                wrist_camera_t = torch.tensor(wrist_camera, device=env.device)
                grasp_pos = camera_pos + camera_rot @ wrist_camera_t[:3, 3]
                grasp_rot = camera_rot @ wrist_camera_t[:3, :3]
                grasp_quat = quat_from_matrix(grasp_rot.unsqueeze(0))[0]
                hug_distance = float(torch.linalg.vector_norm(grasp_pos - object_pos))
                planning_valid = hug_distance <= 0.25
                planning_failure = None if planning_valid else f"HUG wrist distance {hug_distance:.3f} m > 0.25 m"
            else:
                hand_target = np.array([0.85, 0.85, 0.8, 0.75, 0.7, 0.55], dtype=np.float32)
                grasp_pos = object_pos + torch.tensor([0.0, -0.075, 0.035], device=env.device)
                grasp_quat = robot.data.body_quat_w[0, right_hand_index].clone()
                planning_valid = True
                planning_failure = None
            start_pos = robot.data.body_pos_w[0, right_hand_index].clone()
            start_quat = robot.data.body_quat_w[0, right_hand_index].clone()
            direction = start_pos - grasp_pos
            direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=1e-6)
            pregrasp_pos = grasp_pos + 0.10 * direction
            lift_pos = grasp_pos + torch.tensor([0.0, 0.0, 0.12], device=env.device)
            reachable = hug_distance <= 0.25 if args.use_hug and args.auto_collect else True
            print(
                "[HUG target] "
                f"episode={index:06d} xyz={grasp_pos.cpu().tolist()} "
                f"quat_wxyz={grasp_quat.cpu().tolist()} "
                f"object_distance_m={float(torch.linalg.vector_norm(grasp_pos - object_pos)):.3f} "
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
            }

        def target_for_phase(plan: dict, phase: int):
            if not plan["planning_valid"]:
                return plan["start_pos"], plan["start_quat"], np.zeros(6, dtype=np.float32)
            if phase < 100:
                alpha = phase / 99.0
                pos = torch.lerp(plan["start_pos"], plan["pregrasp_pos"], alpha)
                quat = quat_slerp(plan["start_quat"], plan["grasp_quat"], alpha)
                hand = np.zeros(6, dtype=np.float32)
            elif phase < 180:
                alpha = (phase - 100) / 79.0
                pos = torch.lerp(plan["pregrasp_pos"], plan["grasp_pos"], alpha)
                quat = plan["grasp_quat"]
                hand = np.zeros(6, dtype=np.float32)
            elif phase < 280:
                alpha = (phase - 180) / 99.0
                pos, quat = plan["grasp_pos"], plan["grasp_quat"]
                hand = plan["hand_target"] * (alpha * alpha * (3.0 - 2.0 * alpha))
            elif phase < 400:
                alpha = (phase - 280) / 119.0
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
        def next_writer():
            nonlocal episode_index
            while any((root / f"episode_{episode_index:06d}{suffix}").exists()
                      for suffix in ("", ".incomplete", ".failed")):
                episode_index += 1
            current_index = episode_index
            result = EpisodeWriter(root, current_index, {
                "instruction": "用右手抓起烧杯并抬升",
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
        ik.reset()
        import contextlib
        with open("/dev/null", "w") as sink:
          initial_beaker_z = float(env.scene["object"].data.root_pos_w[0, 2])
          stability_samples = []
          for step in range(args.steps):
            phase = step % cycle_steps
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
            ik.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))
            jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_index, :, arm_joint_ids]
            world_to_base = matrix_from_quat(quat_inv(root_pose[:, 3:7]))
            jacobian[:, :3, :] = torch.bmm(world_to_base, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(world_to_base, jacobian[:, 3:, :])
            arm_current = robot.data.joint_pos[:, arm_joint_ids]
            arm_desired = ik.compute(ee_pos_b, ee_quat_b, jacobian, arm_current)[0]
            limits = robot.data.soft_joint_pos_limits[0, arm_joint_ids]
            max_joint_step_rad = 0.02
            arm_desired = torch.clamp(
                arm_desired,
                arm_current[0] - max_joint_step_rad,
                arm_current[0] + max_joint_step_rad,
            )
            arm_desired = torch.clamp(arm_desired, limits[:, 0], limits[:, 1])
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
                    timestamp=phase * 0.01,
                    state=np.r_[arm_state, compact[7:]], action=compact,
                    images={"front": front["rgb"][0, ..., :3].cpu().numpy(),
                            "right_wrist": wrist["rgb"][0, ..., :3].cpu().numpy()},
                    depth=front["distance_to_image_plane"][0].cpu().numpy(),
                )
            if phase >= cycle_steps - 30:
                object_data = env.scene["object"].data
                object_pos = object_data.root_pos_w[0]
                hand_pos = env.scene["robot"].data.body_pos_w[0, right_hand_index]
                stability_samples.append((
                    float(object_pos[2]) - initial_beaker_z,
                    float(torch.linalg.vector_norm(object_pos - hand_pos)),
                    float(torch.linalg.vector_norm(object_data.root_vel_w[0, :3])),
                    float(torch.linalg.vector_norm(object_data.root_vel_w[0, 3:])),
                ))
            if writer is not None and phase == cycle_steps - 1:
                samples = np.asarray(stability_samples, dtype=np.float64)
                metrics = {
                    "min_lift_m": float(samples[:, 0].min()),
                    "max_hand_distance_m": float(samples[:, 1].max()),
                    "max_linear_speed_mps": float(samples[:, 2].max()),
                    "max_angular_speed_radps": float(samples[:, 3].max()),
                }
                checks = {
                    "planning_valid": bool(plan["planning_valid"]),
                    "lift_ge_0.03m": metrics["min_lift_m"] >= 0.03,
                    "hand_distance_le_0.12m": metrics["max_hand_distance_m"] <= 0.12,
                    "linear_speed_le_0.15mps": metrics["max_linear_speed_mps"] <= 0.15,
                    "angular_speed_le_1.0radps": metrics["max_angular_speed_radps"] <= 1.0,
                }
                metrics["checks"] = checks
                status = "success" if all(checks.values()) else "failed"
                saved = writer.finish(status=status, metrics=metrics)
                print(f"[sim-collect] {status}: {saved} {metrics}", file=sys.stderr, flush=True)
                writer = None
                # Do not create a trailing .incomplete episode in finite runs.
                if args.steps - (step + 1) >= cycle_steps:
                    print("[sim-collect] resetting environment", file=sys.stderr, flush=True)
                    with contextlib.redirect_stdout(sink):
                        observation, _ = env.reset()
                        # reset() restores physics state, but camera tensors may still
                        # contain the previous episode's final render. Apply one default
                        # control step so RGB-D and body transforms describe the reset scene.
                        default_action = torch.tensor(default_pos, device=env.device).unsqueeze(0)
                        observation, *_ = env.step(default_action)
                    initial_beaker_z = float(env.scene["object"].data.root_pos_w[0, 2])
                    stability_samples.clear()
                    writer, current_episode_index = next_writer()
                    print(
                        f"[sim-collect] episode {current_episode_index:06d} planning with fresh RGB-D",
                        file=sys.stderr, flush=True,
                    )
                    plan = plan_episode(current_episode_index)
                    hug_marker.visualize(plan["grasp_pos"].unsqueeze(0), plan["grasp_quat"].unsqueeze(0))
                    pregrasp_marker.visualize(
                        plan["pregrasp_pos"].unsqueeze(0), plan["grasp_quat"].unsqueeze(0)
                    )
                    ik.reset()
                    print(f"[sim-collect] episode {current_episode_index:06d} running", file=sys.stderr, flush=True)
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
