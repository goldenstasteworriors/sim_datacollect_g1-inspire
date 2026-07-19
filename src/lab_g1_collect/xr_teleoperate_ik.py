"""Persistent bridge to Unitree xr_teleoperate's official G1_29 ArmIK."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


_PREFIX = "XR_IK_JSON "


class XrTeleoperateArmIK:
    def __init__(self, project: Path, environment: str = "tv"):
        conda_envs = Path(sys.executable).resolve().parents[2]
        python = conda_envs / environment / "bin/python"
        if not python.exists():
            raise FileNotFoundError(f"xr_teleoperate conda Python not found: {python}")
        self.process = subprocess.Popen(
            [str(python), str(Path(__file__).resolve()), "--worker", str(project)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1,
        )

    def _request(self, payload: dict) -> dict:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"xr_teleoperate IK worker exited: {self.process.poll()}")
            if line.startswith(_PREFIX):
                response = json.loads(line[len(_PREFIX):])
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response
            print(f"[xr-ik-worker] {line.rstrip()}", file=sys.stderr, flush=True)

    def initialize(self, dual_q: np.ndarray) -> dict:
        return self._request({"command": "initialize", "q": np.asarray(dual_q).tolist()})

    def solve(self, right_target: np.ndarray, dual_q: np.ndarray, dual_dq: np.ndarray) -> np.ndarray:
        response = self._request({
            "command": "solve", "right_target": np.asarray(right_target).tolist(),
            "q": np.asarray(dual_q).tolist(), "dq": np.asarray(dual_dq).tolist(),
        })
        return np.asarray(response["q"], dtype=np.float32)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._request({"command": "close"})
            finally:
                self.process.terminate()
                self.process.wait(timeout=5)


def _worker(project: Path) -> None:
    xr_root = project / "third_party/xr_teleoperate"
    sys.path.insert(0, str(xr_root))
    os.chdir(xr_root / "teleop")
    cache = Path("g1_29_model_cache.pkl")
    # The IK only needs logging, while logging-mp starts a multiprocessing
    # listener that interferes with this pipe worker. Supply its stdlib-compatible
    # API without changing the vendored xr_teleoperate source.
    import logging
    import types
    logging_mp = types.ModuleType("logging_mp")
    logging_mp.getLogger = logging.getLogger
    logging_mp.basicConfig = logging.basicConfig
    for name in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        setattr(logging_mp, name, getattr(logging, name))
    sys.modules["logging_mp"] = logging_mp
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
    import pinocchio as pin

    solver = G1_29_ArmIK()
    cache.unlink(missing_ok=True)
    model = solver.reduced_robot.model
    # On a fresh (non-cached) upstream build, L_ee/R_ee are added after the
    # RobotWrapper data was created. Recreate Data so oMf contains those frames.
    solver.reduced_robot.data = model.createData()
    data = solver.reduced_robot.data

    def frames(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[solver.L_hand_id].homogeneous.copy(), data.oMf[solver.R_hand_id].homogeneous.copy()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            if command == "close":
                print(_PREFIX + json.dumps({"closed": True}), flush=True)
                break
            q = np.asarray(request["q"], dtype=np.float64)
            left, right = frames(q)
            if command == "initialize":
                response = {"left": left.tolist(), "right": right.tolist()}
            elif command == "solve":
                target = np.asarray(request["right_target"], dtype=np.float64)
                dq = np.asarray(request["dq"], dtype=np.float64)
                solution, _torque = solver.solve_ik(left, target, q, dq)
                response = {"q": np.asarray(solution).tolist()}
            else:
                raise ValueError(f"unknown command: {command}")
        except Exception as exc:
            import traceback
            response = {"error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}
        print(_PREFIX + json.dumps(response), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    arguments = parser.parse_args()
    _worker(arguments.worker)
