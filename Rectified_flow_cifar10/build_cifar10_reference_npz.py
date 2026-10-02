"""Download CIFAR-10 if needed and save its 10,000 test images for ADM FID."""
import argparse
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    dataset = CIFAR10(root=args.data_root, train=False, download=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, arr_0=np.asarray(dataset.data, dtype=np.uint8))
    print(f"Wrote {args.output} with {len(dataset)} CIFAR-10 test images.")


if __name__ == "__main__":
    main()
