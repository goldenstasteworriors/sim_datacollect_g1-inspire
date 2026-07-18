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
        expanded = compact_to_isaac_action(np.r_[arm_default, np.zeros(6, dtype=np.float32)], joint_names, default_pos)
        writer = None
        capture_stride = 3
        if args.auto_collect:
            from .dataset import EpisodeWriter
            root = Path(args.output)
            episode_index = 0
            while ((root / f"episode_{episode_index:06d}").exists() or
                   (root / f"episode_{episode_index:06d}.incomplete").exists()):
                episode_index += 1
            writer = EpisodeWriter(root, episode_index, {
                "instruction": "用右手抓起烧杯并抬升",
                "robot_type": "unitree_g1_right_inspire_rh56dftp",
                "source": "isaac_gui_scripted",
                "fps": 30,
            })
        import contextlib
        with open("/dev/null", "w") as sink:
          initial_beaker_z = float(env.scene["object"].data.root_pos_w[0, 2])
          for step in range(args.steps):
            compact = trajectory[step % len(trajectory)]
            expanded = compact_to_isaac_action(compact, joint_names, default_pos)
            action = torch.tensor(expanded, device=env.device).unsqueeze(0)
            with contextlib.redirect_stdout(sink):
                observation, *_ = env.step(action)
            if writer is not None and step < len(trajectory) and step % capture_stride == 0:
                robot_pos = env.scene["robot"].data.joint_pos[0].cpu().numpy()
                arm_state = np.array([robot_pos[joint_names.index(name)] for name in RIGHT_ARM_JOINTS])
                front = env.scene["front_camera"].data.output
                wrist = env.scene["right_wrist_camera"].data.output
                writer.add_frame(
                    timestamp=step * 0.01,
                    state=np.r_[arm_state, compact[7:]], action=compact,
                    images={"front": front["rgb"][0, ..., :3].cpu().numpy(),
                            "right_wrist": wrist["rgb"][0, ..., :3].cpu().numpy()},
                    depth=front["distance_to_image_plane"][0].cpu().numpy(),
                )
            if writer is not None and step == len(trajectory) - 1:
                lifted = float(env.scene["object"].data.root_pos_w[0, 2]) - initial_beaker_z
                print(f"[sim-collect] beaker lift {lifted:.4f} m", file=sys.stderr, flush=True)
                print(f"[sim-collect] saved {writer.finish()}", file=sys.stderr, flush=True)
                writer = None
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
