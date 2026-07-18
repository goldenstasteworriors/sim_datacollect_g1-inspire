from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


class EpisodeWriter:
    """原子写入一个可校验、可转 LeRobot 的多模态 episode。"""

    def __init__(self, root: str | Path, episode_index: int, metadata: dict):
        self.final = Path(root) / f"episode_{episode_index:06d}"
        self.work = self.final.with_name(self.final.name + ".incomplete")
        self.work.mkdir(parents=True, exist_ok=False)
        (self.work / "images").mkdir()
        self.metadata = metadata
        self.rows: list[dict] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.depths: list[np.ndarray] = []
        self.ee_actual: list[np.ndarray] = []
        self.ee_target: list[np.ndarray] = []

    def add_frame(self, *, timestamp: float, state: np.ndarray, action: np.ndarray,
                  images: dict[str, np.ndarray], depth: np.ndarray | None = None,
                  ee_actual: np.ndarray | None = None,
                  ee_target: np.ndarray | None = None) -> None:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        if state.shape != (13,) or action.shape != (13,):
            raise ValueError("state/action 必须是右臂 7 + 右手 6，共 13 维")
        index = len(self.rows)
        image_paths = {}
        for camera, array in images.items():
            target = self.work / "images" / f"{index:06d}_{camera}.png"
            Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(target)
            image_paths[camera] = str(target.relative_to(self.work))
        self.rows.append({"frame_index": index, "timestamp": float(timestamp), "images": image_paths})
        self.states.append(state)
        self.actions.append(action)
        if depth is not None:
            self.depths.append(np.asarray(depth, dtype=np.float32))
        if (ee_actual is None) != (ee_target is None):
            raise ValueError("ee_actual 和 ee_target 必须同时提供")
        if ee_actual is not None:
            actual = np.asarray(ee_actual, dtype=np.float32)
            target = np.asarray(ee_target, dtype=np.float32)
            if actual.shape != (7,) or target.shape != (7,):
                raise ValueError("末端位姿必须是 xyz + wxyz，共 7 维")
            self.ee_actual.append(actual)
            self.ee_target.append(target)

    def finish(self, *, status: str = "success", metrics: dict | None = None) -> Path:
        if not self.rows:
            raise RuntimeError("不能提交空 episode")
        if status not in {"success", "failed"}:
            raise ValueError(f"不支持的 episode 状态: {status}")
        arrays = {"observation.state": np.stack(self.states), "action": np.stack(self.actions)}
        if self.depths:
            if len(self.depths) != len(self.rows):
                raise RuntimeError("depth 帧数与 episode 帧数不一致")
            arrays["observation.depth"] = np.stack(self.depths)
        if self.ee_actual:
            if len(self.ee_actual) != len(self.rows):
                raise RuntimeError("末端位姿帧数与 episode 帧数不一致")
            arrays["observation.ee_pose_w"] = np.stack(self.ee_actual)
            arrays["target.ee_pose_w"] = np.stack(self.ee_target)
        np.savez_compressed(self.work / "episode.npz", **arrays)
        (self.work / "frames.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.rows), encoding="utf-8"
        )
        meta = {**self.metadata, "length": len(self.rows), "format": "lab-g1-v1",
                "status": status, "metrics": metrics or {}}
        (self.work / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        destination = self.final if status == "success" else self.final.with_name(self.final.name + ".failed")
        self.work.rename(destination)
        return destination


def validate_episode(path: str | Path) -> dict:
    root = Path(path)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(root / "episode.npz")
    rows = (root / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    length = int(meta["length"])
    if arrays["observation.state"].shape != (length, 13) or arrays["action"].shape != (length, 13):
        raise ValueError("episode 数组维度不合法")
    if len(rows) != length or not np.isfinite(arrays["action"]).all():
        raise ValueError("episode 帧数或动作值不合法")
    missing = []
    for line in rows:
        for relative in json.loads(line)["images"].values():
            if not (root / relative).is_file():
                missing.append(relative)
    if missing:
        raise FileNotFoundError(f"缺失图像: {missing[:3]}")
    return {"path": str(root), "frames": length, "state_dim": 13, "action_dim": 13}
