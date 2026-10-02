"""Generate CIFAR-10 RectifiedFlow samples at a fixed Euler budget K."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import device_default
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rf_cifar10_common import euler_sample, load_model


def save_grid(images, path, columns=8):
    images = images[: columns * columns]
    h, w = images[0].shape[:2]
    grid = Image.new("RGB", (columns * w, ((len(images) + columns - 1) // columns) * h))
    for index, image in enumerate(images):
        grid.paste(Image.fromarray(image), ((index % columns) * w, (index // columns) * h))
    grid.save(path)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=root / "ImageGeneration")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=device_default(torch.cuda.is_available()),
                        help="cpu or a logical CUDA device, e.g. cuda:0")
    args = parser.parse_args()
    if args.k < 1:
        raise ValueError("--k must be at least 1")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, _, step = load_model(args.runtime_root, args.checkpoint, device)
    args.output_dir.mkdir(parents=True)
    images = np.empty((args.num_samples, 32, 32, 3), dtype=np.uint8)
    offset = 0
    while offset < args.num_samples:
        n = min(args.batch_size, args.num_samples - offset)
        batch = euler_sample(model, n, args.k, device)
        batch = (batch.permute(0, 2, 3, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        images[offset : offset + n] = batch
        offset += n
        print(f"Generated {offset}/{args.num_samples} samples for K={args.k}", flush=True)
    np.savez(args.output_dir / "images.npz", arr_0=images)
    save_grid(images, args.output_dir / "image_grid.png")
    print(f"Checkpoint step={step}; wrote {args.output_dir / 'images.npz'} and image_grid.png")


if __name__ == "__main__":
    main()
