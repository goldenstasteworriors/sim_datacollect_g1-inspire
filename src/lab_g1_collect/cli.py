from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .dataset import EpisodeWriter, validate_episode
from .retarget import mano_landmarks_to_inspire
from .trajectory import generate_grasp_trajectory


def _open_hand_landmarks() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    bases = [(-0.035, 0), (-0.018, 0), (0, 0), (0.018, 0), (0.035, 0)]
    chains = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    for (x, _), chain in zip(bases, chains):
        for j, idx in enumerate(chain, 1):
            points[idx] = (x, j * 0.025, 0)
    return points


def cmd_doctor(config_path: str) -> None:
    cfg = load_config(config_path)
    checks = {
        "beaker_asset": cfg.path("task", "asset_path").is_file(),
        "mano_right": (cfg.path("hug", "mano_dir") / "MANO_RIGHT.pkl").is_file(),
        "hug_checkout": (cfg.root / "third_party/hug/src/inference.py").is_file(),
        "unitree_checkout": (cfg.root / "third_party/unitree_sim_isaaclab/sim_main.py").is_file(),
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


def cmd_mock_collect(config_path: str, output: str | None) -> None:
    cfg = load_config(config_path)
    hand = mano_landmarks_to_inspire(_open_hand_landmarks())
    start = np.array([0.20, -0.15, 0.05, 0.75, 0.0, 0.25, 0.0], dtype=np.float32)
    grasp = np.array([0.35, -0.30, 0.10, 1.05, 0.0, 0.42, 0.0], dtype=np.float32)
    lift = grasp + np.array([-0.12, 0, 0, -0.18, 0, 0, 0], dtype=np.float32)
    phases = cfg.raw["task"]["phases"]
    actions = generate_grasp_trajectory(
        start, grasp, lift, np.maximum(hand, np.array([0.75, 0.75, 0.7, 0.65, 0.65, 0.5])),
        fps=cfg.fps, seconds=float(cfg.raw["task"]["episode_seconds"]),
        phase_ratios=(phases["approach"], phases["close"], phases["lift"]),
    )
    root = Path(output).resolve() if output else cfg.path("dataset", "root")
    root.mkdir(parents=True, exist_ok=True)
    writer = EpisodeWriter(root, 0, {
        "instruction": cfg.raw["task"]["instruction"],
        "robot_type": cfg.raw["dataset"]["robot_type"],
        "source": "deterministic_mock",
        "fps": cfg.fps,
    })
    state = np.r_[start, np.zeros(6, dtype=np.float32)]
    height, width = cfg.raw["dataset"]["image_size"]
    for index, action in enumerate(actions):
        state = 0.7 * state + 0.3 * action
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[..., 1] = np.uint8(40 + 180 * index / max(len(actions) - 1, 1))
        base[height // 3: 2 * height // 3, width // 2 - 20: width // 2 + 20] = (180, 210, 230)
        writer.add_frame(timestamp=index / cfg.fps, state=state, action=action,
                         images={"front": base, "right_wrist": np.flip(base, axis=1)},
                         depth=np.full((height, width), 0.65, dtype=np.float32))
    episode = writer.finish()
    print(json.dumps(validate_episode(episode), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 右手实验器材自动数据采集")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    mock = sub.add_parser("mock-collect")
    mock.add_argument("--output")
    check = sub.add_parser("validate")
    check.add_argument("episode")
    export = sub.add_parser("export-lerobot")
    export.add_argument("episode")
    export.add_argument("target")
    export.add_argument("--repo-id", default="local/lab_g1_beaker")
    args = parser.parse_args()
    if args.command == "doctor":
        cmd_doctor(args.config)
    elif args.command == "mock-collect":
        cmd_mock_collect(args.config, args.output)
    elif args.command == "validate":
        print(json.dumps(validate_episode(args.episode), ensure_ascii=False, indent=2))
    else:
        from .lerobot_export import export_episode

        print(export_episode(args.episode, args.target, args.repo_id))


if __name__ == "__main__":
    main()
