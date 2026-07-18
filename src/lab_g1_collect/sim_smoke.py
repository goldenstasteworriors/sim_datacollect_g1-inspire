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
        from .sim_action import RIGHT_ARM_JOINTS, compact_to_isaac_action
        from .trajectory import generate_grasp_trajectory

        joint_names = list(env.scene["robot"].joint_names)
        default_pos = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
        import numpy as np

        arm_default = np.array([default_pos[joint_names.index(name)] for name in RIGHT_ARM_JOINTS], dtype=np.float32)
        grasp_arm = arm_default.copy()
        lift_arm = grasp_arm + np.array([0.22, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        trajectory = generate_grasp_trajectory(
            arm_default, grasp_arm, lift_arm,
            np.array([0.85, 0.85, 0.8, 0.75, 0.7, 0.55], dtype=np.float32),
            fps=100, seconds=4.0,
        )
        trajectory = np.concatenate([trajectory, np.repeat(trajectory[-1:], 50, axis=0)])
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
            result = EpisodeWriter(root, episode_index, {
                "instruction": "用右手抓起烧杯并抬升",
                "robot_type": "unitree_g1_right_inspire_rh56dftp",
                "source": "isaac_gui_scripted",
                "fps": 30,
            })
            episode_index += 1
            return result
        if args.auto_collect:
            from .dataset import EpisodeWriter
            writer = next_writer()
        import contextlib
        with open("/dev/null", "w") as sink:
          initial_beaker_z = float(env.scene["object"].data.root_pos_w[0, 2])
          right_hand_index = env.scene["robot"].body_names.index("right_hand_base_link")
          stability_samples = []
          for step in range(args.steps):
            phase = step % len(trajectory)
            compact = trajectory[phase]
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
            if phase >= len(trajectory) - 30:
                object_data = env.scene["object"].data
                object_pos = object_data.root_pos_w[0]
                hand_pos = env.scene["robot"].data.body_pos_w[0, right_hand_index]
                stability_samples.append((
                    float(object_pos[2]) - initial_beaker_z,
                    float(torch.linalg.vector_norm(object_pos - hand_pos)),
                    float(torch.linalg.vector_norm(object_data.root_vel_w[0, :3])),
                    float(torch.linalg.vector_norm(object_data.root_vel_w[0, 3:])),
                ))
            if writer is not None and phase == len(trajectory) - 1:
                samples = np.asarray(stability_samples, dtype=np.float64)
                metrics = {
                    "min_lift_m": float(samples[:, 0].min()),
                    "max_hand_distance_m": float(samples[:, 1].max()),
                    "max_linear_speed_mps": float(samples[:, 2].max()),
                    "max_angular_speed_radps": float(samples[:, 3].max()),
                }
                checks = {
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
                if step + 1 < args.steps:
                    with contextlib.redirect_stdout(sink):
                        observation, _ = env.reset()
                    initial_beaker_z = float(env.scene["object"].data.root_pos_w[0, 2])
                    stability_samples.clear()
                    writer = next_writer()
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
