"""Run the official HUG model for one simulator RGB-D capture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .hug_condition import add_condition_point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sampling-steps", type=int, default=10)
    args = parser.parse_args()

    from hug.inference import main as infer
    from hug.prepare_inputs import prepare_pkl

    capture = np.load(args.capture)
    object_name = str(capture["object_name"]) if "object_name" in capture else "beaker"
    pkl_path = prepare_pkl(
        capture["rgb"].astype(np.uint8),
        capture["depth_mm"].astype(np.uint16),
        capture["K"].astype(np.float64),
        "sim_capture",
        args.dataset,
        object_name=object_name,
    )
    add_condition_point(pkl_path, float(capture["u_224"]), float(capture["v_224"]), radius=14)
    infer(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        sample_name="sim_capture",
        num_samples=None,
        batch_size=1,
        sampling_steps=args.sampling_steps,
    )


if __name__ == "__main__":
    main()
