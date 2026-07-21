"""Read G1 LowState through a remote, subscriber-only SONIC SDK helper."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np


_PREFIX = "G1_LOWSTATE_JSON "
RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class G1ArmState:
    q: np.ndarray
    dq: np.ndarray
    waist_q: np.ndarray
    waist_dq: np.ndarray
    timestamp: float
    mode_machine: int
    source: str
    body_q: np.ndarray | None = None
    body_dq: np.ndarray | None = None


def _vector(payload: dict[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(payload[key], dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"LowState {key} 应为有限值 ({size},)，实际为 {value.shape}")
    return value


def read_g1_arm_state(config: dict[str, Any]) -> G1ArmState:
    """Invoke the remote helper over SSH and return one validated snapshot."""
    ssh_host = str(config["ssh_host"])
    reader = str(config["remote_reader"])
    sonic_root = str(config["sonic_root"])
    interface = str(config.get("network_interface", "eth0"))
    timeout_s = float(config.get("timeout_s", 3.0))
    library_path = f"{sonic_root}/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/aarch64"
    remote_command = shlex.join(
        ["env", f"LD_LIBRARY_PATH={library_path}", reader, interface, str(timeout_s)]
    )
    result = subprocess.run(
        ["ssh", "--", ssh_host, remote_command],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s + 7.0,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"读取 G1 LowState 失败（exit={result.returncode}）: {detail}")
    line = next((line for line in result.stdout.splitlines() if line.startswith(_PREFIX)), None)
    if line is None:
        raise RuntimeError(f"G1 LowState 输出缺少 {_PREFIX.strip()}")
    payload = json.loads(line[len(_PREFIX):])
    q = _vector(payload, "right_arm_q", 7)
    dq = _vector(payload, "right_arm_dq", 7)
    waist_q = _vector(payload, "waist_q", 3)
    waist_dq = _vector(payload, "waist_dq", 3)
    body_q = _vector(payload, "body_q", 29) if "body_q" in payload else None
    body_dq = _vector(payload, "body_dq", 29) if "body_dq" in payload else None
    max_abs_dq = float(config.get("max_abs_dq_rad_s", 1.0))
    if float(np.max(np.abs(dq))) > max_abs_dq:
        raise RuntimeError(
            f"右臂仍在运动，|dq|max={np.max(np.abs(dq)):.3f} rad/s，"
            f"超过 dry-run 上限 {max_abs_dq:.3f} rad/s"
        )
    return G1ArmState(
        q=q,
        dq=dq,
        waist_q=waist_q,
        waist_dq=waist_dq,
        timestamp=float(payload["timestamp"]),
        mode_machine=int(payload["mode_machine"]),
        source=f"{ssh_host}:rt/lowstate[22:29]",
        body_q=body_q,
        body_dq=body_dq,
    )
