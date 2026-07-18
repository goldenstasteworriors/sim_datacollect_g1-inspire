"""Add an interactive-style condition point to a HUG prepared input."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


def add_condition_point(pkl_path: Path, u: float, v: float, radius: int = 12) -> None:
    """Store a model-resolution click and PNG mask in a HUG input pickle."""
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)

    camera = data["camera"]
    width = int(camera["width"] if isinstance(camera, dict) else camera.width)
    height = int(camera["height"] if isinstance(camera, dict) else camera.height)
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"condition point ({u}, {v}) is outside {width}x{height}")

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (round(u), round(v)), radius, 255, thickness=-1)
    encoded, buffer = cv2.imencode(".png", mask)
    if not encoded:
        raise RuntimeError("failed to encode HUG condition mask")

    data["condition_point"] = np.asarray([u, v], dtype=np.float32)
    data["object_mask"] = buffer.tobytes()
    with pkl_path.open("wb") as handle:
        pickle.dump(data, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pkl", type=Path)
    parser.add_argument("--u", type=float, required=True, help="x at 224px model scale")
    parser.add_argument("--v", type=float, required=True, help="y at 224px model scale")
    parser.add_argument("--radius", type=int, default=12)
    args = parser.parse_args()
    add_condition_point(args.pkl, args.u, args.v, args.radius)


if __name__ == "__main__":
    main()
