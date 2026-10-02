#!/usr/bin/env python3
"""Generate class-conditional CIFAR samples with Euler ODE at several K values."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import device_default

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=PROJECT_ROOT / "runtime",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing dit_config.json and dit_ema.pt",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 250])
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("fid_sweep_euler_cifar10_10k"))
    parser.add_argument("--device", default=device_default(torch.cuda.is_available()),
                        help="cpu or a logical CUDA device, e.g. cuda:0")
    return parser.parse_args()


def validate_args(args):
    if not args.runtime_root.is_dir():
        raise FileNotFoundError(f"Repository not found: {args.runtime_root}")
    for filename in ("dit_config.json", "dit_ema.pt"):
        path = args.checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint file not found: {path}")
    if any(k < 1 for k in args.steps):
        raise ValueError("Every K must be at least 1")


@torch.inference_mode()
def main():
    args = parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    validate_args(args)

    sys.path.insert(0, str(args.runtime_root))
    from rectified_flow.models.dit import DiT
    from rectified_flow.rectified_flow import RectifiedFlow
    from rectified_flow.samplers import EulerSampler

    device = torch.device(args.device)
    model = DiT.from_pretrained(
        save_directory=str(args.checkpoint_dir),
        filename="dit",
        use_ema=True,
    ).to(device).eval()

    rf = RectifiedFlow(
        data_shape=(3, 32, 32),
        velocity_field=model,
        interp="straight",
        source_distribution="normal",
        device=device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for k in args.steps:
        images_dir = args.output_dir / f"k_{k:03d}" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Resetting the seed for each K gives every solver the same noise and labels.
        generator = torch.Generator(device=device).manual_seed(args.seed)
        generated = 0
        while generated < args.num_samples:
            batch_size = min(args.batch_size, args.num_samples - generated)
            x_0 = torch.randn(
                batch_size, 3, 32, 32, device=device, generator=generator
            )
            labels = torch.arange(
                generated, generated + batch_size, device=device
            ) % 10

            sampler = EulerSampler(
                rectified_flow=rf,
                num_steps=k,
                record_traj_period=k,
            )
            result = sampler.sample_loop(x_0=x_0, y=labels)
            images = result.trajectories[-1].clamp(-1, 1).add(1).div(2)

            for offset, image in enumerate(images):
                save_image(image, images_dir / f"{generated + offset:06d}.png")
            generated += batch_size
            print(f"K={k}: {generated}/{args.num_samples}", flush=True)

        metadata = {
            "solver": "ODE/Euler",
            "k_euler_updates": k,
            "integrator_time_points": k + 1,
            "nfe": k,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "checkpoint_dir": str(args.checkpoint_dir),
            "images_dir": str(images_dir),
        }
        with (images_dir.parent / "metadata.json").open("w") as handle:
            json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
