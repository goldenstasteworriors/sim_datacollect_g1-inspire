"""Safely initialize and replay a reviewed right-arm plan from the PC via DDS.

The process runs on the computer connected to the G1 network.  It publishes
only ``rt/arm_sdk`` slots 12:29 (waist and dual arms); leg slots 0:12 are never
written.  Publishing is impossible unless ``--execute`` is supplied and the
operator presses ENTER in an interactive terminal.
"""

from __future__ import annotations

import argparse
import json
import select
import signal
import socket
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .initial_pose import G1_ARM_SDK_JOINT_NAMES, sonicmj_initial_q


ARM_TOPIC = "rt/arm_sdk"
STATE_TOPIC = "rt/lowstate"
HAND_COMMAND_TOPIC = "rt/inspire/cmd"
HAND_STATE_TOPIC = "rt/inspire/state"
ARM_SDK_WEIGHT_INDEX = 29
ARM_SDK_MOTOR_INDICES = np.asarray(
    [*range(15, 22), *range(22, 29), *range(12, 15)], dtype=np.int64
)
LEG_MOTOR_INDICES = frozenset(range(12))
if LEG_MOTOR_INDICES.intersection(ARM_SDK_MOTOR_INDICES.tolist()):
    raise RuntimeError("安全断言失败：Arm SDK 命令集合包含腿部关节")

# G1 29-DoF limits in official Arm SDK order: left arm, right arm, waist.
JOINT_LOWER = np.asarray(
    [
        -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.614429558,
        -1.614429558, -3.0892, -2.2515, -2.618, -1.0472, -1.972222054,
        -1.614429558, -1.614429558, -2.618, -0.52, -0.52,
    ],
    dtype=np.float64,
)
JOINT_UPPER = np.asarray(
    [
        2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.614429558,
        1.614429558, 2.6704, 1.5882, 2.618, 2.0944, 1.972222054,
        1.614429558, 1.614429558, 2.618, 0.52, 0.52,
    ],
    dtype=np.float64,
)
KP = np.full(17, 60.0, dtype=np.float64)
KD = np.full(17, 1.5, dtype=np.float64)


def minimum_jerk(start: np.ndarray, goal: np.ndarray, progress: float) -> np.ndarray:
    x = float(np.clip(progress, 0.0, 1.0))
    blend = 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5
    return start + blend * (goal - start)


def _read_key() -> str | None:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1).lower()
    return None


def _validate_joint_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (17,) or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须是有限的 17DoF 向量")
    if np.any(result < JOINT_LOWER) or np.any(result > JOINT_UPPER):
        raise ValueError(f"{name} 超出 G1 URDF 关节限位")
    return result


def inspire_urdf_to_hardware(values: np.ndarray) -> np.ndarray:
    """Convert Inspire URDF radians to the DFX bridge's [0,1] open/close scale."""
    radians = np.asarray(values, dtype=np.float64)
    if radians.ndim != 2 or radians.shape[1] != 6 or not np.isfinite(radians).all():
        raise ValueError(f"Inspire 轨迹必须是有限 [T,6]，实际为 {radians.shape}")
    lower = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -0.1])
    upper = np.asarray([1.7, 1.7, 1.7, 1.7, 0.5, 1.3])
    if np.any(radians < lower - 1e-6) or np.any(radians > upper + 1e-6):
        raise ValueError("Inspire URDF 轨迹超出官方重定向范围")
    return np.clip((upper - radians) / (upper - lower), 0.0, 1.0)


def _rate_limited_commands(
    initial: np.ndarray,
    targets: np.ndarray,
    *,
    target_dt: float,
    frequency: float,
    max_speed: float,
) -> tuple[np.ndarray, float]:
    control_dt = 1.0 / frequency
    intervals = int(np.ceil((len(targets) - 1) * target_dt * frequency))
    commands = np.empty((intervals + 1, initial.size), dtype=np.float64)
    command = initial.copy()
    commands[0] = command
    max_lag = 0.0
    max_step = max_speed * control_dt
    for step in range(1, intervals + 1):
        target_index = min(int(step * control_dt / target_dt), len(targets) - 1)
        target = targets[target_index]
        command += np.clip(target - command, -max_step, max_step)
        commands[step] = command
        max_lag = max(max_lag, float(np.max(np.abs(target - command))))
    return commands, max_lag


@dataclass(frozen=True)
class ReviewedPlan:
    path: Path
    right_arm_commands: np.ndarray
    right_hand_commands: np.ndarray
    waist_q: np.ndarray
    metadata: dict[str, Any]
    arm_peak_speed: float
    hand_peak_speed: float
    arm_max_lag: float
    hand_max_lag: float


def load_reviewed_plan(
    path: Path,
    *,
    source_fps: float,
    frequency: float,
    max_arm_speed: float,
    max_hand_speed: float,
) -> ReviewedPlan:
    plan_path = path.expanduser().resolve()
    report_path = plan_path.with_name("plan.json")
    if not plan_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"抓取计划或同目录 plan.json 不存在: {plan_path}")
    metadata = json.loads(report_path.read_text(encoding="utf-8"))
    required_gates = {
        "ik_all_reachable": True,
        "speed_limit_exceeded": False,
        "simulation_review_only": False,
        "control_output_enabled": False,
    }
    failures = [
        f"{key}={metadata.get(key)!r}（要求 {expected!r}）"
        for key, expected in required_gates.items()
        if metadata.get(key) is not expected
    ]
    if failures:
        raise ValueError("真机计划安全门未通过: " + ", ".join(failures))
    with np.load(plan_path) as data:
        if "trajectory" not in data or "waist_q" not in data:
            raise ValueError("dry_run_plan.npz 缺少 trajectory 或 waist_q")
        trajectory = np.asarray(data["trajectory"], dtype=np.float64)
        waist_q = np.asarray(data["waist_q"], dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 13 or len(trajectory) < 2:
        raise ValueError(f"抓取轨迹必须是 [T,13] 且 T>=2，实际为 {trajectory.shape}")
    if not np.isfinite(trajectory).all() or waist_q.shape != (3,) or not np.isfinite(waist_q).all():
        raise ValueError("抓取轨迹或腰部姿态包含非法值")
    arm_targets = trajectory[:, :7]
    if np.any(arm_targets < JOINT_LOWER[7:14]) or np.any(arm_targets > JOINT_UPPER[7:14]):
        raise ValueError("右臂抓取轨迹超出 G1 URDF 限位")
    hand_targets = inspire_urdf_to_hardware(trajectory[:, 7:13])
    target_dt = 1.0 / source_fps
    arm_commands, arm_lag = _rate_limited_commands(
        arm_targets[0], arm_targets, target_dt=target_dt,
        frequency=frequency, max_speed=max_arm_speed,
    )
    hand_commands, hand_lag = _rate_limited_commands(
        hand_targets[0], hand_targets, target_dt=target_dt,
        frequency=frequency, max_speed=max_hand_speed,
    )
    arm_peak = float(np.max(np.abs(np.diff(arm_commands, axis=0))) * frequency)
    hand_peak = float(np.max(np.abs(np.diff(hand_commands, axis=0))) * frequency)
    if arm_peak > max_arm_speed + 1e-9 or hand_peak > max_hand_speed + 1e-9:
        raise ValueError("离线限速后的轨迹仍超过配置速度上限")
    return ReviewedPlan(
        path=plan_path,
        right_arm_commands=arm_commands,
        right_hand_commands=hand_commands,
        waist_q=waist_q,
        metadata=metadata,
        arm_peak_speed=arm_peak,
        hand_peak_speed=hand_peak,
        arm_max_lag=arm_lag,
        hand_max_lag=hand_lag,
    )


class EStop:
    def __init__(self) -> None:
        self.latched = False
        self.reason = ""

    def trigger(self, reason: str) -> None:
        if not self.latched:
            self.latched = True
            self.reason = reason
            print(f"\n[E-STOP] 已锁存：{reason}；停止轨迹并释放 Arm SDK 权重", flush=True)


class PCArmSdkDDS:
    """PC-side subscriber/publisher matching Unitree's Arm SDK example."""

    def __init__(self, network_interface: str, require_hand_state: bool) -> None:
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__MotorCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        available = {name for _, name in socket.if_nameindex()}
        if network_interface not in available:
            raise ValueError(
                f"网卡 {network_interface!r} 不存在；可用网卡: {', '.join(sorted(available))}"
            )
        ChannelFactoryInitialize(0, network_interface)
        self._ChannelPublisher = ChannelPublisher
        self._LowCmd = LowCmd_
        self._MotorCmds = MotorCmds_
        self._motor_cmd_default = unitree_go_msg_dds__MotorCmd_
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._publisher = None
        self._hand_publisher = None
        self._lock = threading.Lock()
        self._body_q: np.ndarray | None = None
        self._body_dq: np.ndarray | None = None
        self._mode_machine = 0
        self._stamp = 0.0
        self._hand_q: np.ndarray | None = None
        self._hand_stamp = 0.0
        self._last_command: np.ndarray | None = None
        self._last_weight = 0.0
        self._subscriber = ChannelSubscriber(STATE_TOPIC, LowState_)
        self._subscriber.Init(self._on_state, 10)
        self._hand_subscriber = None
        if require_hand_state:
            self._hand_subscriber = ChannelSubscriber(HAND_STATE_TOPIC, MotorStates_)
            self._hand_subscriber.Init(self._on_hand_state, 10)
        self._wait_for_state(5.0)

    def _on_state(self, message: Any) -> None:
        if len(message.motor_state) < 29:
            return
        q = np.asarray([motor.q for motor in message.motor_state[:29]], dtype=np.float64)
        dq = np.asarray([motor.dq for motor in message.motor_state[:29]], dtype=np.float64)
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            return
        with self._lock:
            self._body_q = q
            self._body_dq = dq
            self._mode_machine = int(message.mode_machine)
            self._stamp = time.monotonic()

    def _on_hand_state(self, message: Any) -> None:
        if len(message.states) < 12:
            return
        values = np.asarray([state.q for state in message.states[:12]], dtype=np.float64)
        if not np.isfinite(values).all():
            return
        with self._lock:
            self._hand_q = values
            self._hand_stamp = time.monotonic()

    def _wait_for_state(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._body_q is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError("5 秒内没有收到 rt/lowstate；请检查 enp7s0 和机器人模式")

    def state(self, max_age: float) -> tuple[np.ndarray, np.ndarray, int, float]:
        with self._lock:
            if self._body_q is None or self._body_dq is None:
                raise RuntimeError("LowState missing or stale: no state")
            age = time.monotonic() - self._stamp
            q = self._body_q[ARM_SDK_MOTOR_INDICES].copy()
            dq = self._body_dq[ARM_SDK_MOTOR_INDICES].copy()
            mode_machine = self._mode_machine
        if age > max_age:
            raise RuntimeError(
                f"LowState missing or stale: age={age * 1000.0:.1f}ms > {max_age * 1000.0:.1f}ms"
            )
        return q, dq, mode_machine, age

    def hand_state(self, max_age: float) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._hand_q is None or time.monotonic() - self._hand_stamp > max_age:
                raise RuntimeError(
                    "rt/inspire/state 缺失或过期；请先启动 inspire_modbus_hand.py --mode dds"
                )
            values = self._hand_q.copy()
        return values[:6], values[6:12]

    def enable_publishers(self, enable_hands: bool) -> None:
        if self._publisher is not None:
            return
        self._publisher = self._ChannelPublisher(ARM_TOPIC, self._LowCmd)
        self._publisher.Init()
        if enable_hands:
            self._hand_publisher = self._ChannelPublisher(
                HAND_COMMAND_TOPIC, self._MotorCmds
            )
            self._hand_publisher.Init()

    def write_arm(self, target: np.ndarray, weight: float) -> None:
        if self._publisher is None:
            raise RuntimeError("Arm SDK publisher 尚未经人工确认启用")
        command = _validate_joint_vector(target, "Arm SDK target")
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("Arm SDK weight 必须位于 [0,1]")
        with self._lock:
            self._cmd.mode_machine = self._mode_machine
        self._cmd.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = float(weight)
        for position, index, kp, kd in zip(
            command, ARM_SDK_MOTOR_INDICES, KP, KD, strict=True
        ):
            motor = self._cmd.motor_cmd[int(index)]
            motor.q = float(position)
            motor.dq = 0.0
            motor.kp = float(kp)
            motor.kd = float(kd)
            motor.tau = 0.0
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._publisher.Write(self._cmd)
        self._last_command = command.copy()
        self._last_weight = float(weight)

    def write_hands(self, right: np.ndarray, left: np.ndarray) -> None:
        if self._hand_publisher is None:
            raise RuntimeError("Inspire publisher 尚未启用")
        right_q = np.asarray(right, dtype=np.float64)
        left_q = np.asarray(left, dtype=np.float64)
        if right_q.shape != (6,) or left_q.shape != (6,):
            raise ValueError("左右 Inspire 命令都必须是 6 维")
        if not np.isfinite(right_q).all() or not np.isfinite(left_q).all():
            raise ValueError("Inspire 命令必须为有限值")
        values = np.clip(np.concatenate((right_q, left_q)), 0.0, 1.0)
        message = self._MotorCmds(
            [self._motor_cmd_default() for _ in range(12)]
        )
        for motor, value in zip(message.cmds, values, strict=True):
            motor.q = float(value)
        self._hand_publisher.Write(message)

    def release(self, measured: np.ndarray | None, frequency: float, duration: float) -> None:
        if self._publisher is None:
            return
        hold = measured if measured is not None else self._last_command
        if hold is None:
            return
        hold = _validate_joint_vector(hold, "release hold")
        start_weight = self._last_weight
        count = max(1, int(round(duration * frequency)))
        period = 1.0 / frequency
        for step in range(1, count + 1):
            progress = step / count
            weight = float(minimum_jerk(
                np.asarray([start_weight]), np.asarray([0.0]), progress
            )[0])
            self.write_arm(hold, weight)
            time.sleep(period)
        for _ in range(10):
            self.write_arm(hold, 0.0)
            time.sleep(period)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="real_grasp 生成的 dry_run_plan.npz")
    parser.add_argument("--network-interface", default="enp7s0")
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--max-speed", type=float, default=0.15)
    parser.add_argument("--max-hand-speed", type=float, default=0.5)
    parser.add_argument("--hand-frequency", type=float, default=10.0)
    parser.add_argument("--initial-duration", type=float, default=5.0)
    parser.add_argument("--initial-speed", type=float, default=0.15)
    parser.add_argument("--initial-tolerance", type=float, default=0.03)
    parser.add_argument("--stable-duration", type=float, default=1.0)
    parser.add_argument("--max-measured-speed", type=float, default=0.20)
    parser.add_argument("--lowstate-timeout", type=float, default=0.20)
    parser.add_argument("--release-duration", type=float, default=0.50)
    parser.add_argument(
        "--simulation-approved", action="store_true",
        help="确认 --plan 已在 0.76m 桌面 Isaac 场景中人工审核",
    )
    parser.add_argument("--observe", action="store_true", help="只从 PC 订阅一次 LowState")
    parser.add_argument("--execute", action="store_true", help="允许交互式真机发布")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    positive = (
        args.frequency, args.source_fps, args.max_speed, args.max_hand_speed,
        args.hand_frequency, args.initial_duration, args.initial_speed,
        args.initial_tolerance, args.stable_duration, args.max_measured_speed,
        args.lowstate_timeout, args.release_duration,
    )
    if min(positive) <= 0.0:
        raise SystemExit("频率、时长、速度和安全阈值必须为正")
    hand_stride = round(args.frequency / args.hand_frequency)
    if args.hand_frequency > args.frequency or not np.isclose(
        hand_stride * args.hand_frequency, args.frequency
    ):
        raise SystemExit("--frequency 必须是 --hand-frequency 的整数倍")
    if args.observe and args.execute:
        raise SystemExit("--observe 与 --execute 不能同时使用")

    reviewed = None
    if args.plan is not None:
        try:
            reviewed = load_reviewed_plan(
                args.plan, source_fps=args.source_fps, frequency=args.frequency,
                max_arm_speed=args.max_speed, max_hand_speed=args.max_hand_speed,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"真机轨迹离线检查失败: {exc}") from exc

    summary = {
        "mode": "execute" if args.execute else "observe" if args.observe else "dry_run",
        "dds_runs_on": "pc",
        "network_interface": args.network_interface,
        "arm_topic": ARM_TOPIC,
        "commanded_motor_indices": ARM_SDK_MOTOR_INDICES.tolist(),
        "leg_command_indices": [],
        "plan": str(reviewed.path) if reviewed else None,
        "arm_peak_speed_rad_s": reviewed.arm_peak_speed if reviewed else None,
        "hand_peak_speed_per_s": reviewed.hand_peak_speed if reviewed else None,
        "arm_max_rate_limit_lag_rad": reviewed.arm_max_lag if reviewed else None,
        "hand_max_rate_limit_lag": reviewed.hand_max_lag if reviewed else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.observe and not args.execute:
        print("[DRY RUN] 未初始化 DDS，未创建 publisher，未发送任何命令")
        return
    if args.execute and reviewed is not None and not args.simulation_approved:
        raise SystemExit("播放真机计划必须显式添加 --simulation-approved")
    if args.execute and not sys.stdin.isatty():
        raise SystemExit("真机模式必须在交互终端运行，确保 SPACE/Q 急停可用")

    robot = PCArmSdkDDS(args.network_interface, require_hand_state=reviewed is not None)
    current_q, current_dq, mode_machine, age = robot.state(args.lowstate_timeout)
    print(
        f"[LOWSTATE] mode_machine={mode_machine} age={age * 1000.0:.1f}ms "
        f"max_arm_sdk_dq={np.max(np.abs(current_dq)):.4f}rad/s"
    )
    if args.observe:
        print(json.dumps({
            "joint_names": list(G1_ARM_SDK_JOINT_NAMES),
            "q": current_q.tolist(), "dq": current_dq.tolist(),
            "publisher_created": False,
        }, ensure_ascii=False, indent=2))
        return

    sonic_target = sonicmj_initial_q(G1_ARM_SDK_JOINT_NAMES)
    initial_target = sonic_target.copy()
    if reviewed is not None:
        initial_target[7:14] = reviewed.right_arm_commands[0]
        initial_target[14:17] = reviewed.waist_q
    initial_target = _validate_joint_vector(initial_target, "初始化目标")
    max_distance = float(np.max(np.abs(initial_target - current_q)))
    # Maximum derivative of minimum jerk is 1.875.
    init_duration = max(
        args.initial_duration, 1.875 * max_distance / args.initial_speed
    )
    print(
        f"PC 端 Arm SDK 已就绪但尚未创建 publisher。\n"
        f"ENTER=启用并初始化，READY 后 L=播放，SPACE/Q/Ctrl-C=急停释放。\n"
        f"腿部索引 0-11 永不写入；预计初始化 {init_duration:.2f}s。"
    )

    estop = EStop()
    signal.signal(signal.SIGINT, lambda *_: estop.trigger("Ctrl-C"))
    signal.signal(signal.SIGTERM, lambda *_: estop.trigger("SIGTERM"))
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    phase = "DISARMED"
    enabled = False
    cycle = 0
    phase_start = 0.0
    init_start = current_q.copy()
    stable_since: float | None = None
    command_index = 0
    previous_target = current_q.copy()
    left_hand = np.ones(6, dtype=np.float64)
    right_hand = np.ones(6, dtype=np.float64)
    initial_right_hand = np.ones(6, dtype=np.float64)
    initial_left_hand = np.ones(6, dtype=np.float64)
    period = 1.0 / args.frequency
    try:
        while not estop.latched:
            cycle_start = time.monotonic()
            key = _read_key()
            if key in (" ", "q"):
                estop.trigger("keyboard")
                break
            measured_q, measured_dq, _, _ = robot.state(args.lowstate_timeout)
            if key in ("\n", "\r") and phase == "DISARMED":
                if float(np.max(np.abs(measured_dq))) > args.max_measured_speed:
                    raise RuntimeError("双臂/腰部仍在运动，拒绝初始化")
                if reviewed is not None:
                    initial_right_hand, initial_left_hand = robot.hand_state(0.5)
                    right_hand = initial_right_hand.copy()
                    left_hand = initial_left_hand.copy()
                robot.enable_publishers(enable_hands=reviewed is not None)
                enabled = True
                init_start = measured_q.copy()
                previous_target = init_start.copy()
                phase_start = time.monotonic()
                phase = "ACQUIRING"
                print("[ARMED] PC 开始接管 rt/arm_sdk，先保持实测姿态")

            target = None
            weight = 0.0
            if phase == "ACQUIRING":
                progress = (time.monotonic() - phase_start) / 1.0
                target = init_start
                weight = float(minimum_jerk(
                    np.asarray([0.0]), np.asarray([1.0]), progress
                )[0])
                if progress >= 1.0:
                    phase = "INITIALIZING"
                    phase_start = time.monotonic()
                    print("[INITIALIZING] minimum-jerk 移动到轨迹起始姿态")
            elif phase == "INITIALIZING":
                progress = (time.monotonic() - phase_start) / init_duration
                target = minimum_jerk(init_start, initial_target, progress)
                weight = 1.0
                if reviewed is not None:
                    right_hand = minimum_jerk(
                        initial_right_hand, reviewed.right_hand_commands[0], progress
                    )
                    left_hand = minimum_jerk(
                        initial_left_hand, np.ones(6), progress
                    )
                if progress >= 1.0:
                    error = float(np.max(np.abs(initial_target - measured_q)))
                    if error <= args.initial_tolerance:
                        stable_since = stable_since or time.monotonic()
                        if time.monotonic() - stable_since >= args.stable_duration:
                            phase = "READY"
                            print(
                                f"[READY] max_error={error:.4f}rad；"
                                + ("检查现场后按 L 播放" if reviewed else "保持初始化姿态，SPACE/Q 释放")
                            )
                    else:
                        stable_since = None
            elif phase == "READY":
                target = initial_target
                weight = 1.0
                if key == "l" and reviewed is not None:
                    phase = "PLAYING"
                    command_index = 0
                    print(f"[PLAYING] 开始播放 {len(reviewed.right_arm_commands)} 个 PC 端命令")
            elif phase == "PLAYING":
                assert reviewed is not None
                target = initial_target.copy()
                target[7:14] = reviewed.right_arm_commands[command_index]
                right_hand = reviewed.right_hand_commands[command_index]
                weight = 1.0
                command_index += 1
                if command_index >= len(reviewed.right_arm_commands):
                    phase = "HOLDING"
                    command_index = len(reviewed.right_arm_commands) - 1
                    print("[DONE] 轨迹播放完成，保持末姿态；SPACE/Q 释放")
            elif phase == "HOLDING":
                assert reviewed is not None
                target = initial_target.copy()
                target[7:14] = reviewed.right_arm_commands[-1]
                right_hand = reviewed.right_hand_commands[-1]
                weight = 1.0

            if target is not None:
                runtime_speed = float(np.max(np.abs(target - previous_target)) * args.frequency)
                limit = args.initial_speed if phase in ("ACQUIRING", "INITIALIZING") else args.max_speed
                if runtime_speed > limit + 1e-6:
                    raise RuntimeError(
                        f"运行时速度保护触发: {runtime_speed:.4f}>{limit:.4f}rad/s"
                    )
                robot.write_arm(target, weight)
                previous_target = target.copy()
                if reviewed is not None and cycle % hand_stride == 0:
                    robot.write_hands(right_hand, left_hand)
            cycle += 1
            time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))
    except Exception as exc:
        estop.trigger(str(exc))
    finally:
        try:
            measured = None
            if enabled:
                try:
                    measured, _, _, _ = robot.state(args.lowstate_timeout)
                except RuntimeError:
                    pass
                print(
                    f"[RELEASE] PC 连续发送 Arm SDK weight→0，"
                    f"duration={args.release_duration:.2f}s",
                    flush=True,
                )
                try:
                    robot.release(measured, args.frequency, args.release_duration)
                except Exception as exc:
                    print(f"[RELEASE ERROR] {exc}", file=sys.stderr, flush=True)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)


if __name__ == "__main__":
    main()
