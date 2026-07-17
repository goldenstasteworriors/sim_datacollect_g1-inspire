from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    raw: dict[str, Any]

    @property
    def fps(self) -> int:
        return int(self.raw["task"]["fps"])

    def path(self, section: str, key: str) -> Path:
        value = Path(self.raw[section][key]).expanduser()
        return value if value.is_absolute() else self.root / value


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"robot", "task", "hug", "dataset"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"配置缺少字段: {sorted(missing)}")
    # configs/ 位于项目根目录的下一级。
    return PipelineConfig(root=config_path.parent.parent, raw=raw)

