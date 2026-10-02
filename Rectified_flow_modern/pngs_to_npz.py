#!/usr/bin/env python3
"""Convert a numbered PNG directory to ADM evaluator NPZ format."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()

    paths = sorted(args.images_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG images found in {args.images_dir}")
    if args.expected_count is not None and len(paths) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} PNGs, but found {len(paths)} in {args.images_dir}"
        )

    first = np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.uint8)
    images = np.empty((len(paths), *first.shape), dtype=np.uint8)
    images[0] = first
    for index, path in enumerate(paths[1:], start=1):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if image.shape != first.shape:
            raise ValueError(f"Unexpected image shape {image.shape} in {path}; expected {first.shape}")
        images[index] = image

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, images)
    print(f"Saved {args.output_npz}: shape={images.shape}, dtype={images.dtype}")


if __name__ == "__main__":
    main()
