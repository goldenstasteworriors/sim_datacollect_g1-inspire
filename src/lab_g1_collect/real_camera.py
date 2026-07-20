"""Read-only RGB-D transport for the G1 camera computer.

The client understands both this project's lossless ``lab-g1-rgbd-v1``
messages and the RGB-only message emitted by GEAR-SONIC's composed camera.
No robot state or command DDS topic is opened here.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


RGBD_PROTOCOL = "lab-g1-rgbd-v1"


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray | None
    K: np.ndarray | None
    timestamp: float
    source_protocol: str

    @property
    def has_metric_depth(self) -> bool:
        return self.depth_m is not None and self.K is not None


def _decode_image(encoded: bytes, *, unchanged: bool = False) -> np.ndarray:
    image = Image.open(io.BytesIO(encoded))
    if unchanged:
        return np.asarray(image)
    return np.asarray(image.convert("RGB"))


def decode_camera_message(message: dict[str, Any]) -> CameraFrame:
    """Decode one RGB-D-v1 or official GEAR-SONIC camera message."""
    protocol = message.get("protocol")
    if isinstance(protocol, bytes):
        protocol = protocol.decode("utf-8")
    if protocol == RGBD_PROTOCOL:
        rgb = _decode_image(message["rgb_jpeg"])
        depth_raw = _decode_image(message["depth_png"], unchanged=True)
        if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
            raise ValueError(
                f"RGB-D 深度应为 uint16 HxW，实际为 {depth_raw.dtype} {depth_raw.shape}"
            )
        depth_scale = float(message["depth_scale_m"])
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError(f"无效的深度比例: {depth_scale}")
        K = np.asarray(message["K"], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError(f"相机内参应为有限值 (3, 3)，实际为 {K.shape}")
        if rgb.shape[:2] != depth_raw.shape:
            raise ValueError(f"RGB/深度分辨率不一致: {rgb.shape[:2]} vs {depth_raw.shape}")
        return CameraFrame(
            rgb=rgb,
            depth_m=depth_raw.astype(np.float32) * depth_scale,
            K=K,
            timestamp=float(message["timestamp"]),
            source_protocol=RGBD_PROTOCOL,
        )

    # Compatibility path for NVlabs/GR00T-WholeBodyControl composed_camera.
    # Its RealSense depth currently uses the ordinary JPEG encoder, so it is
    # deliberately not exposed as metric depth.
    images = message.get("images", {})
    timestamps = message.get("timestamps", {})
    rgb_value = images.get("ego_view")
    if rgb_value is None:
        available = ", ".join(sorted(str(key) for key in images))
        raise ValueError(f"GEAR-SONIC 消息缺少 ego_view，可用键: {available}")
    if isinstance(rgb_value, str):
        # The legacy path feeds an RGB ndarray to cv2.imencode and returns the
        # cv2-decoded BGR array. Reversing Pillow's RGB output restores the
        # original numeric RGB array expected by HUG.
        rgb = _decode_image(base64.b64decode(rgb_value))[..., ::-1].copy()
    elif isinstance(rgb_value, (bytes, bytearray)):
        rgb = _decode_image(bytes(rgb_value))
    elif isinstance(rgb_value, np.ndarray):
        rgb = np.asarray(rgb_value, dtype=np.uint8)
    else:
        raise TypeError(f"不支持的 ego_view 编码类型: {type(rgb_value).__name__}")
    return CameraFrame(
        rgb=rgb,
        depth_m=None,
        K=None,
        timestamp=float(timestamps.get("ego_view", time.time())),
        source_protocol="gear-sonic-rgb-only",
    )


class CameraClient:
    """A read-only, latest-frame ZMQ SUB client."""

    def __init__(self, host: str, port: int = 5555):
        try:
            import msgpack
            import zmq
        except ImportError as exc:
            raise RuntimeError("真机相机客户端需要安装项目的 real 可选依赖") from exc
        self._msgpack = msgpack
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, True)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self.endpoint = f"tcp://{host}:{int(port)}"
        self._socket.connect(self.endpoint)

    def receive(self, timeout_s: float = 10.0) -> CameraFrame:
        timeout_ms = max(1, int(round(float(timeout_s) * 1000)))
        if not self._socket.poll(timeout_ms):
            raise TimeoutError(f"等待相机帧超时: {self.endpoint}")
        packed = self._socket.recv()
        message = self._msgpack.unpackb(packed, raw=False, strict_map_key=False)
        return decode_camera_message(message)

    def close(self) -> None:
        self._socket.close()
        self._context.term()

    def __enter__(self) -> "CameraClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
