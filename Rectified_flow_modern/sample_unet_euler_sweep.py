#!/usr/bin/env python3
"""Sample the official pretrained CIFAR-10 UNet with Euler ODE."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import resource_path, device_default

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=PROJECT_ROOT / "runtime")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=resource_path(PROJECT_ROOT / "checkpoints" / "unet_cifar10_pretrained"),
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 250])
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "fid" / "fid_sweep_unet_euler_cifar10_10k")
    parser.add_argument("--reference-npz", type=Path, default=resource_path(PROJECT_ROOT / "references" / "cifar10_test_10k.npz"))
    parser.add_argument("--adm-evaluator", type=Path, default=resource_path(PROJECT_ROOT / "evaluations" / "evaluator.py"))
    parser.add_argument("--no-fid", action="store_true", help="Generate samples and NPZ without calculating FID")
    parser.add_argument("--device", default=device_default(torch.cuda.is_available()),
                        help="cpu or a logical CUDA device, e.g. cuda:0")
    return parser.parse_args()


def pngs_to_npz(image_dir, output_npz, expected_count):
    paths = sorted(image_dir.glob("*.png"))
    if len(paths) != expected_count:
        raise RuntimeError(f"Expected {expected_count} PNGs in {image_dir}, found {len(paths)}")
    first = np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.uint8)
    images = np.empty((len(paths), *first.shape), dtype=np.uint8)
    images[0] = first
    for index, path in enumerate(paths[1:], start=1):
        images[index] = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    np.savez(output_npz, images)
    print(f"Saved {output_npz}: shape={images.shape}, dtype={images.dtype}", flush=True)


def calculate_fid(args, k, run_dir, sample_npz):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, str(args.adm_evaluator.resolve()), str(args.reference_npz.resolve()), str(sample_npz.resolve())],
        cwd=args.adm_evaluator.resolve().parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (run_dir / "adm_evaluator.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"FID failed for K={k}; see {run_dir / 'adm_evaluator.log'}")

    def metric(name):
        match = re.search(rf"^{re.escape(name)}:\s*([0-9.eE+-]+)", result.stdout, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Could not find {name} in {run_dir / 'adm_evaluator.log'}")
        return float(match.group(1))

    metrics = {
        "k_euler_updates": k,
        "integrator_time_points": k + 1,
        "nfe": k,
        "fid": metric("FID"),
        "sfid": metric("sFID"),
        "inception_score": metric("Inception Score"),
        "precision": metric("Precision"),
        "recall": metric("Recall"),
        "num_samples": args.num_samples,
        "solver": "ODE/Euler",
        "reference_npz": str(args.reference_npz.resolve()),
        "sample_npz": str(sample_npz.resolve()),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"K={k}: FID-{args.num_samples}={metrics['fid']:.6f}", flush=True)
    return metrics


def rebuild_csv(output_dir):
    rows = []
    for path in output_dir.glob("k_*/metrics.json"):
        rows.append(json.loads(path.read_text()))
    rows.sort(key=lambda row: row["k_euler_updates"])
    if not rows:
        return
    columns = ["k_euler_updates", "integrator_time_points", "nfe", "fid", "sfid", "inception_score", "precision", "recall", "num_samples", "solver", "sample_npz"]
    with (output_dir / "fid_vs_k.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main():
    args = arguments()
    for filename in ("unet_config.json", "unet_ema.pt"):
        required = args.checkpoint_dir / filename
        if not required.is_file():
            raise FileNotFoundError(required)
    if any(k < 1 for k in args.steps):
        raise ValueError("Every K must be at least 1")
    if not args.no_fid:
        for required in (args.reference_npz, args.adm_evaluator):
            if not required.is_file():
                raise FileNotFoundError(required)

    sys.path.insert(0, str(args.runtime_root.resolve()))
    from rectified_flow.models.unet import SongUNet
    from rectified_flow.rectified_flow import RectifiedFlow
    from rectified_flow.samplers import EulerSampler

    device = torch.device(args.device)
    model = SongUNet.from_pretrained(
        str(args.checkpoint_dir.resolve()), filename="unet", use_ema=True
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
        run_dir = args.output_dir / f"k_{k:03d}"
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(image_dir.glob("*.png")))
        if existing == args.num_samples:
            print(f"K={k}: found all {existing} PNGs; skipping generation.", flush=True)
        else:
            if existing:
                raise RuntimeError(f"K={k} has an incomplete set of {existing} PNGs; move it aside before rerunning")
            generator = torch.Generator(device=device).manual_seed(args.seed)
            generated = 0
            while generated < args.num_samples:
                size = min(args.batch_size, args.num_samples - generated)
                noise = torch.randn(size, 3, 32, 32, generator=generator, device=device)
                sampler = EulerSampler(rf, num_steps=k, record_traj_period=k)
                samples = sampler.sample_loop(x_0=noise).trajectories[-1]
                samples = samples.clamp(-1, 1).add(1).div(2)
                for offset, sample in enumerate(samples):
                    save_image(sample, image_dir / f"{generated + offset:06d}.png")
                generated += size
                print(f"K={k} (NFE={k}): {generated}/{args.num_samples}", flush=True)

        with (image_dir.parent / "metadata.json").open("w") as output:
            json.dump(
                {
                    "architecture": "SongUNet",
                    "solver": "ODE/Euler",
                    "k_euler_updates": k,
                    "integrator_time_points": k + 1,
                    "nfe": k,
                    "num_samples": args.num_samples,
                    "seed": args.seed,
                    "checkpoint_dir": str(args.checkpoint_dir.resolve()),
                },
                output,
                indent=2,
            )

        sample_npz = run_dir / "images.npz"
        if not sample_npz.is_file():
            pngs_to_npz(image_dir, sample_npz, args.num_samples)
        else:
            print(f"K={k}: found {sample_npz}; skipping NPZ conversion.", flush=True)

        metrics_path = run_dir / "metrics.json"
        if not args.no_fid and not metrics_path.is_file():
            calculate_fid(args, k, run_dir, sample_npz)
        elif metrics_path.is_file():
            print(f"K={k}: found {metrics_path}; skipping FID.", flush=True)
        rebuild_csv(args.output_dir)


if __name__ == "__main__":
    main()
