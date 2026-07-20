"""Publish aligned RealSense RGB-D without opening any robot control topic."""

from __future__ import annotations

import argparse
import time

import numpy as np


# Kept local so the robot-side camera process does not import Pillow or any
# planning dependency from the client module.
RGBD_PROTOCOL = "lab-g1-rgbd-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", help="可选的 RealSense 序列号")
    args = parser.parse_args()

    try:
        import cv2
        import msgpack
        import pyrealsense2 as rs
        import zmq
    except ImportError as exc:
        raise SystemExit(f"相机服务缺少依赖: {exc}") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_m = float(depth_sensor.get_depth_scale())

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{args.port}")
    print(
        f"只读 RGB-D 服务已启动: tcp://*:{args.port}, "
        f"{args.width}x{args.height}@{args.fps}, depth_scale={depth_scale_m}",
        flush=True,
    )
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            K = [
                [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
                [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
                [0.0, 0.0, 1.0],
            ]
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb_ok, rgb_jpeg = cv2.imencode(
                ".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            depth_ok, depth_png = cv2.imencode(".png", depth)
            if not rgb_ok or not depth_ok:
                continue
            message = {
                "protocol": RGBD_PROTOCOL,
                "timestamp": time.time(),
                "rgb_jpeg": rgb_jpeg.tobytes(),
                "depth_png": depth_png.tobytes(),
                "depth_scale_m": depth_scale_m,
                "K": K,
            }
            try:
                socket.send(msgpack.packb(message, use_bin_type=True), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
    except KeyboardInterrupt:
        print("停止只读 RGB-D 服务", flush=True)
    finally:
        pipeline.stop()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
