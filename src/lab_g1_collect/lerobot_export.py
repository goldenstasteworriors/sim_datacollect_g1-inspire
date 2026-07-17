from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


JOINT_NAMES = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw", "right_pinky", "right_ring",
    "right_middle", "right_index", "right_thumb_bend", "right_thumb_rotation",
]


def export_episode(source: str | Path, target: str | Path, repo_id: str = "local/lab_g1_beaker") -> Path:
    """把内部原子 episode 转为 LeRobot v3/v4 数据集。需在安装 LeRobot 的环境运行。"""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("当前环境没有 LeRobot；请在 labvla/lerobot 环境执行") from exc

    source = Path(source)
    target = Path(target).resolve()
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (source / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    arrays = np.load(source / "episode.npz")
    first = np.asarray(Image.open(source / rows[0]["images"]["front"]).convert("RGB"))
    image_shape = tuple(first.shape)
    features = {
        "observation.state": {"dtype": "float32", "shape": (13,), "names": {"axes": JOINT_NAMES}},
        "action": {"dtype": "float32", "shape": (13,), "names": {"axes": JOINT_NAMES}},
        "observation.images.front": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channels"]},
        "observation.images.right_wrist": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channels"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=target,
        fps=int(meta["fps"]),
        robot_type=meta["robot_type"],
        features=features,
        use_videos=False,
    )
    for index, row in enumerate(rows):
        dataset.add_frame({
            "observation.state": arrays["observation.state"][index],
            "action": arrays["action"][index],
            "observation.images.front": np.asarray(Image.open(source / row["images"]["front"]).convert("RGB")),
            "observation.images.right_wrist": np.asarray(Image.open(source / row["images"]["right_wrist"]).convert("RGB")),
            "task": meta["instruction"],
        })
    dataset.save_episode()
    dataset.finalize()
    return target

