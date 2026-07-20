"""Generate a bottle-pick plan from real RGB-D without controlling the robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from .arm_ik import solve_right_arm_ik
from .g1_camera_geometry import OFFICIAL_G1_MJCF_URL, official_g1_T_pelvis_camera_optical
from .hug_bridge import run_hug_capture
from .real_camera import CameraClient, CameraFrame
from .real_state import read_g1_arm_state
from .trajectory import smoothstep


# Deliberately constant: this module has no control publisher and cannot be
# enabled with a CLI flag. A future executor must live in a separate module.
CONTROL_OUTPUT_ENABLED = False


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    safety = config.get("safety", {})
    if safety.get("mode") != "dry_run" or safety.get("allow_control") is not False:
        raise ValueError("真机配置必须固定为 mode=dry_run 且 allow_control=false")
    return config


def _load_capture(path: Path) -> CameraFrame:
    data = np.load(path)
    return CameraFrame(
        rgb=data["rgb"].astype(np.uint8),
        depth_m=data["depth_m"].astype(np.float32) if "depth_m" in data else None,
        K=data["K"].astype(np.float64) if "K" in data else None,
        timestamp=float(data["timestamp"]),
        source_protocol=str(data["source_protocol"]),
    )


def _save_capture(frame: CameraFrame, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    capture = output / "capture.npz"
    values: dict[str, Any] = {
        "rgb": frame.rgb,
        "timestamp": np.float64(frame.timestamp),
        "source_protocol": np.asarray(frame.source_protocol),
    }
    if frame.depth_m is not None:
        values["depth_m"] = frame.depth_m
    if frame.K is not None:
        values["K"] = frame.K
    np.savez_compressed(capture, **values)
    Image.fromarray(frame.rgb).save(output / "rgb.png")
    return capture


def _point_to_hug_uv(u: float, v: float, width: int, height: int) -> tuple[float, float]:
    square = min(width, height)
    x_offset = (width - square) / 2.0
    y_offset = (height - square) / 2.0
    return ((u - x_offset) * 224.0 / square, (v - y_offset) * 224.0 / square)


def _robust_depth(depth_m: np.ndarray, u: float, v: float, radius: int = 4) -> float:
    x, y = int(round(u)), int(round(v))
    y0, y1 = max(0, y - radius), min(depth_m.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(depth_m.shape[1], x + radius + 1)
    values = depth_m[y0:y1, x0:x1]
    valid = values[np.isfinite(values) & (values > 0.05) & (values < 3.0)]
    if valid.size == 0:
        raise ValueError("目标点附近没有有效的 0.05~3.0 m 深度")
    return float(np.median(valid))


def _pixel_xyz(K: np.ndarray, u: float, v: float, depth_m: float) -> np.ndarray:
    return np.array(
        [
            (u - K[0, 2]) * depth_m / K[0, 0],
            (v - K[1, 2]) * depth_m / K[1, 1],
            depth_m,
        ],
        dtype=np.float64,
    )


def _camera_matrix(config: dict[str, Any], waist_q: np.ndarray) -> tuple[np.ndarray, str]:
    geometry = config.get("camera_geometry", {})
    if geometry.get("model") != "official_g1_mjcf":
        raise ValueError("camera_geometry.model 必须是 official_g1_mjcf")
    return official_g1_T_pelvis_camera_optical(waist_q), str(
        geometry.get("source", OFFICIAL_G1_MJCF_URL)
    )


def _generate_dry_run_trajectory(
    current: np.ndarray,
    pregrasp: np.ndarray,
    grasp: np.ndarray,
    lift: np.ndarray,
    hand_target: np.ndarray,
    *,
    fps: int,
    duration_s: float,
    phase_ratios: tuple[float, float, float, float],
) -> np.ndarray:
    """Build current→pregrasp→grasp→close→lift semantic joint waypoints."""
    arm_waypoints = [np.asarray(value, dtype=np.float32).reshape(7) for value in
                     (current, pregrasp, grasp, lift)]
    hand_target = np.asarray(hand_target, dtype=np.float32).reshape(6)
    ratios = np.asarray(phase_ratios, dtype=np.float64)
    if ratios.shape != (4,) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("real dry-run phase_ratios 必须是和为 1 的四个正数")
    count = max(4, int(round(fps * duration_s)))
    lengths = np.maximum(1, np.floor(count * ratios).astype(int))
    lengths[-1] += count - int(lengths.sum())
    if lengths[-1] <= 0:
        raise ValueError("规划总时长太短，无法分配四个阶段")
    cuts = np.cumsum(np.r_[0, lengths])
    trajectory = np.zeros((count, 13), dtype=np.float32)

    def blend(begin: int, end: int, start: np.ndarray, target: np.ndarray) -> None:
        alpha = smoothstep(np.linspace(0, 1, end - begin, dtype=np.float32))[:, None]
        trajectory[begin:end, :7] = start + alpha * (target - start)

    blend(cuts[0], cuts[1], arm_waypoints[0], arm_waypoints[1])
    blend(cuts[1], cuts[2], arm_waypoints[1], arm_waypoints[2])
    trajectory[cuts[2]:cuts[3], :7] = arm_waypoints[2]
    close_alpha = smoothstep(
        np.linspace(0, 1, cuts[3] - cuts[2], dtype=np.float32)
    )[:, None]
    trajectory[cuts[2]:cuts[3], 7:] = close_alpha * hand_target
    blend(cuts[3], cuts[4], arm_waypoints[2], arm_waypoints[3])
    trajectory[cuts[3]:cuts[4], 7:] = hand_target
    return trajectory


def _plan(
    *, project: Path, frame: CameraFrame, config: dict[str, Any], output: Path,
    u: float, v: float, right_arm_q: np.ndarray, waist_q: np.ndarray,
    right_arm_source: str, simulation_review: bool = False,
) -> dict[str, Any]:
    if not frame.has_metric_depth:
        raise ValueError(
            "当前是 GEAR-SONIC RGB-only 帧；规划需要 lab-g1-rgbd-v1 的无损深度和内参"
        )
    assert frame.depth_m is not None and frame.K is not None
    height, width = frame.rgb.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"目标像素 ({u}, {v}) 超出图像 {width}x{height}")
    T_base_camera, camera_geometry_source = _camera_matrix(config, waist_q)
    depth = _robust_depth(frame.depth_m, u, v)
    object_camera = _pixel_xyz(frame.K, u, v, depth)
    object_base = (T_base_camera @ np.r_[object_camera, 1.0])[:3]
    hug_uv = _point_to_hug_uv(u, v, width, height)

    grasp_camera, hand_target = run_hug_capture(
        project=project,
        episode_index=int(config["planner"].get("episode_index", 0)),
        rgb=frame.rgb,
        depth_m=frame.depth_m,
        K=frame.K,
        point_uv_224=hug_uv,
        object_name=str(config["task"].get("object_name", "bottle")),
        sampling_steps=int(config["planner"].get("hug_sampling_steps", 5)),
        candidates=int(config["planner"].get("hug_candidates", 8)),
        debug_stride=1,
        run_root=output / "hug_runtime",
    )
    grasp_base = T_base_camera @ np.asarray(grasp_camera, dtype=np.float64)
    radial = grasp_base[:3, 3] - object_base
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-6:
        raise ValueError("HUG 抓取腕位置与目标点重合，无法构造 pre-grasp")
    pregrasp_base = grasp_base.copy()
    pregrasp_base[:3, 3] += radial / radial_norm * float(config["task"]["pregrasp_offset_m"])
    lift_base = grasp_base.copy()
    lift_base[:3, 3] += np.asarray(config["task"]["lift_offset_base_m"], dtype=np.float64)

    current = np.asarray(right_arm_q, dtype=np.float64).reshape(7)
    ik_results = {}
    for name, target in (("pregrasp", pregrasp_base), ("grasp", grasp_base), ("lift", lift_base)):
        result = solve_right_arm_ik(target[:3, 3], target[:3, :3], current)
        ik_results[name] = result
        current = result["joint_positions"]
    ik_failures = [name for name, result in ik_results.items() if not result["reachable"]]
    if ik_failures and not simulation_review:
        failures = ik_failures
        raise ValueError(f"IK dry-run 未通过: {', '.join(failures)}")

    trajectory = _generate_dry_run_trajectory(
        np.asarray(right_arm_q, dtype=np.float32),
        ik_results["pregrasp"]["joint_positions"],
        ik_results["grasp"]["joint_positions"],
        ik_results["lift"]["joint_positions"],
        hand_target,
        fps=int(config["planner"]["fps"]),
        duration_s=float(config["planner"]["duration_s"]),
        phase_ratios=tuple(float(x) for x in config["planner"]["phase_ratios"]),
    )
    max_arm_speed = float(np.max(np.abs(np.diff(trajectory[:, :7], axis=0)))) * int(
        config["planner"]["fps"]
    )
    speed_limit = float(config["planner"]["max_arm_speed_rad_s"])
    speed_limit_exceeded = max_arm_speed > speed_limit
    if speed_limit_exceeded and not simulation_review:
        raise ValueError(
            f"dry-run 轨迹最大关节速度 {max_arm_speed:.3f} rad/s 超过 "
            f"配置上限 {speed_limit:.3f} rad/s"
        )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "dry_run_plan.npz",
        trajectory=trajectory,
        T_base_pregrasp=pregrasp_base,
        T_base_grasp=grasp_base,
        T_base_lift=lift_base,
        object_xyz_base=object_base,
        hand_target_rad=hand_target,
        current_right_arm_q=np.asarray(right_arm_q, dtype=np.float32),
        waist_q=np.asarray(waist_q, dtype=np.float32),
        T_base_camera=T_base_camera,
    )
    preview = Image.fromarray(frame.rgb)
    draw = ImageDraw.Draw(preview)
    draw.ellipse((u - 10, v - 10, u + 10, v + 10), outline="red", width=4)
    draw.line((u - 18, v, u + 18, v), fill="yellow", width=2)
    draw.line((u, v - 18, u, v + 18), fill="yellow", width=2)
    preview.save(output / "target_preview.png")
    metadata = {
        "mode": "dry_run",
        "simulation_review_only": simulation_review,
        "control_output_enabled": CONTROL_OUTPUT_ENABLED,
        "object_name": config["task"].get("object_name", "bottle"),
        "target_uv_full": [float(u), float(v)],
        "target_depth_m": depth,
        "object_xyz_base": object_base.tolist(),
        "current_right_arm_q": np.asarray(right_arm_q, dtype=float).tolist(),
        "waist_q": np.asarray(waist_q, dtype=float).tolist(),
        "current_right_arm_source": right_arm_source,
        "base_frame": "pelvis",
        "camera_geometry": {
            "model": "official_g1_mjcf",
            "source": camera_geometry_source,
            "T_pelvis_camera_optical": T_base_camera.tolist(),
        },
        "trajectory_shape": list(trajectory.shape),
        "max_arm_speed_rad_s": max_arm_speed,
        "configured_arm_speed_limit_rad_s": speed_limit,
        "speed_limit_exceeded": speed_limit_exceeded,
        "hand_command_semantics": "Inspire URDF radians; not a hardware command",
        "ik_all_reachable": not ik_failures,
        "ik_failures": ik_failures,
        "ik": {
            name: {key: value for key, value in result.items() if key != "joint_positions"}
            for name, result in ik_results.items()
        },
    }
    (output / "plan.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real_robot_dry_run.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/real_robot_dry_run"))
    parser.add_argument("--camera-host")
    parser.add_argument("--camera-port", type=int)
    parser.add_argument("--capture", type=Path, help="读取已有 capture.npz，不连接相机")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument(
        "--simulation-review", action="store_true",
        help="即使独立 URDF IK 预检失败也保存候选，仅允许后续 Isaac 安全复核",
    )
    parser.add_argument("--target-u", type=float)
    parser.add_argument("--target-v", type=float)
    parser.add_argument(
        "--right-arm-q", type=float, nargs=7,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="可选离线覆盖；默认通过 SSH 从 rt/lowstate 自动读取",
    )
    parser.add_argument(
        "--waist-q", type=float, nargs=3, metavar=("YAW", "ROLL", "PITCH"),
        help="可选离线腰关节角覆盖；读取历史 capture 时应传入捕获时的腰姿态",
    )
    args = parser.parse_args()

    if CONTROL_OUTPUT_ENABLED:
        raise RuntimeError("安全断言失败：dry-run 模块不得启用控制输出")
    config = _load_config(args.config.resolve())
    output = args.output.resolve()
    if args.capture:
        frame = _load_capture(args.capture.resolve())
    else:
        host = args.camera_host or config["camera"].get("host")
        if not host:
            raise SystemExit("请通过 --camera-host 指定机器人相机主机 IP/名称")
        port = args.camera_port or int(config["camera"].get("port", 5555))
        with CameraClient(host, port) as camera:
            frame = camera.receive(float(config["camera"].get("timeout_s", 10.0)))
        capture = _save_capture(frame, output)
        print(f"已保存只读相机帧: {capture}")
    if args.capture_only:
        print(
            json.dumps(
                {
                    "mode": "capture_only",
                    "control_output_enabled": CONTROL_OUTPUT_ENABLED,
                    "source_protocol": frame.source_protocol,
                    "has_metric_depth": frame.has_metric_depth,
                    "rgb_shape": list(frame.rgb.shape),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.target_u is None or args.target_v is None:
        raise SystemExit("规划时必须用 --target-u/--target-v 指定瓶子上的像素")
    if args.right_arm_q is None:
        arm_state = read_g1_arm_state(config["robot_state"])
        right_arm_q = arm_state.q
        waist_q = arm_state.waist_q if args.waist_q is None else np.asarray(args.waist_q)
        right_arm_source = arm_state.source
        print(
            json.dumps(
                {
                    "lowstate_source": arm_state.source,
                    "right_arm_q": arm_state.q.tolist(),
                    "right_arm_dq": arm_state.dq.tolist(),
                    "waist_q": arm_state.waist_q.tolist(),
                    "mode_machine": arm_state.mode_machine,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        right_arm_q = np.asarray(args.right_arm_q)
        if args.waist_q is None:
            raise SystemExit("使用 --right-arm-q 离线规划时还必须指定 --waist-q YAW ROLL PITCH")
        waist_q = np.asarray(args.waist_q)
        right_arm_source = "cli_override"
    metadata = _plan(
        project=Path(__file__).resolve().parents[2],
        frame=frame,
        config=config,
        output=output,
        u=args.target_u,
        v=args.target_v,
        right_arm_q=right_arm_q,
        waist_q=waist_q,
        right_arm_source=right_arm_source,
        simulation_review=args.simulation_review,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
