from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="启动并单步验证 LabUtopia 烧杯 G1+Inspire 场景")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    unitree = project / "third_party/unitree_sim_isaaclab"
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(unitree))

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless)
    simulation_app = launcher.app
    try:
        import gymnasium as gym
        import torch
        import tasks  # noqa: F401
        import sim_tasks.lab_beaker_g1_inspire  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        task_id = "Isaac-PickPlace-LabBeaker-G129-Inspire-Right"
        cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
        env = gym.make(task_id, cfg=cfg).unwrapped
        observation, _ = env.reset()
        for _ in range(args.steps):
            action = torch.zeros(env.action_space.shape, device=env.device)
            observation, *_ = env.step(action)
        report = {
            "task": task_id,
            "action_shape": list(env.action_space.shape),
            "observation_groups": sorted(observation.keys()),
            "steps": args.steps,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

