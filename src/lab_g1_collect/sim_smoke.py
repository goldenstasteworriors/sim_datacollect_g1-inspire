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
        print("[sim-smoke] config parsed", file=sys.stderr, flush=True)
        env = gym.make(task_id, cfg=cfg).unwrapped
        print("[sim-smoke] environment created", file=sys.stderr, flush=True)
        observation, _ = env.reset()
        print("[sim-smoke] environment reset", file=sys.stderr, flush=True)
        from .sim_action import RIGHT_ARM_JOINTS, compact_to_isaac_action

        joint_names = list(env.scene["robot"].joint_names)
        default_pos = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
        import numpy as np

        arm_default = np.array([default_pos[joint_names.index(name)] for name in RIGHT_ARM_JOINTS], dtype=np.float32)
        expanded = compact_to_isaac_action(np.r_[arm_default, np.zeros(6, dtype=np.float32)], joint_names, default_pos)
        for _ in range(args.steps):
            action = torch.tensor(expanded, device=env.device).unsqueeze(0)
            observation, *_ = env.step(action)
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
