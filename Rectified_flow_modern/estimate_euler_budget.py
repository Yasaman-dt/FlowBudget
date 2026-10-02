#!/usr/bin/env python3
"""Estimate the Euler ODE step budget for the pretrained modern RF UNet."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import resource_path, device_default

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=resource_path(ROOT / "checkpoints" / "unet_cifar10_pretrained"))
    parser.add_argument("--runtime-root", type=Path, default=ROOT / "runtime")
    parser.add_argument("--data-root", type=Path, default=resource_path(ROOT.parent / "rectified_flow_cifar10" / "data"))
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--finite-difference-step", type=float, default=1e-3)
    parser.add_argument(
        "--derivative-method", choices=["finite_difference", "autograd"],
        default="finite_difference",
        help="Finite difference is much faster and uses less memory for this UNet.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-probe-pool", action="store_true",
                        help="Record raw q_S/d_S, ordered ratios and timings; requires batch size 1")
    parser.add_argument("--output", type=Path, default=ROOT / "euler_budget_estimate_500.json")
    parser.add_argument("--device", default=device_default(torch.cuda.is_available()),
                        help="cpu or a logical CUDA device, e.g. cuda:0")
    args = parser.parse_args()
    if args.save_probe_pool and args.batch_size != 1:
        parser.error("--save-probe-pool requires --batch-size 1")
    if args.tau <= 0:
        raise ValueError("--tau must be positive")

    sys.path.insert(0, str(args.runtime_root.resolve()))
    from rectified_flow.models.unet import SongUNet

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = SongUNet.from_pretrained(
        str(args.checkpoint_dir.resolve()), filename="unet", use_ema=True
    ).to(device).eval().requires_grad_(False)

    dataset = CIFAR10(
        root=str(args.data_root), train=True, download=False, transform=transforms.ToTensor()
    )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num_samples]
    loader = DataLoader(
        Subset(dataset, indices.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )

    ratios = []
    numerator_norms, denominator_norms = [], []
    probe_seconds = []
    started = time.time()
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    for batch_index, (x_1, _) in enumerate(loader):
        x_1 = x_1.to(device).mul(2).sub(1)
        x_0 = torch.randn(x_1.shape, generator=noise_generator, device=device, dtype=x_1.dtype)
        t = torch.rand(x_1.shape[0], generator=noise_generator, device=device)
        if args.derivative_method == "finite_difference":
            t = t * (1.0 - args.finite_difference_step)
        expanded_t = t.view(-1, 1, 1, 1)
        x_t = (1 - expanded_t) * x_0 + expanded_t * x_1

        def velocity(x, time_value):
            return model(x, time_value)

        with torch.no_grad():
            v = velocity(x_t, t)
        if args.derivative_method == "finite_difference":
            h = args.finite_difference_step
            with torch.no_grad():
                shifted_v = velocity(x_t + h * v, t + h)
            acceleration = (shifted_v - v) / h
        else:
            _, acceleration = jvp(
                velocity,
                (x_t, t),
                (v, torch.ones_like(t)),
                create_graph=False,
                strict=False,
            )
        q_norm = acceleration.flatten(1).norm(dim=1)
        d_norm = v.flatten(1).norm(dim=1)
        batch_ratios = q_norm / (d_norm + args.epsilon)
        if args.save_probe_pool:
            numerator_norms.extend(q_norm.detach().cpu().tolist())
            denominator_norms.extend(d_norm.detach().cpu().tolist())
        ratios.extend(batch_ratios.detach().cpu().tolist())
        if args.save_probe_pool:
            probe_seconds.append(time.time() - started)
        if (batch_index + 1) % 25 == 0 or len(ratios) == args.num_samples:
            print(f"Estimated {len(ratios)}/{args.num_samples}", flush=True)

    values = torch.tensor(ratios, dtype=torch.float64)
    r_theta = values.mean().item()
    result = {
        "R_theta": r_theta,
        "K_hat": max(1, math.ceil(r_theta / (2 * args.tau))),
        "K_hat_formula": "ceil(R_theta / (2 * tau))",
        "R_theta_formula": "E[||D_t v + J_x(v) v|| / (||v|| + epsilon)]",
        "tau": args.tau,
        "epsilon": args.epsilon,
        "derivative_method": args.derivative_method,
        "finite_difference_step": args.finite_difference_step if args.derivative_method == "finite_difference" else None,
        "num_samples": len(ratios),
        "solver": "deterministic_unconditional_euler_ode",
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "r_median": values.median().item(),
        "r_std": values.std(unbiased=False).item(),
        "elapsed_seconds": time.time() - started,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_probe_pool:
        result['probe_pool'] = dict(
            schema_version=2, q_S=numerator_norms, d_S=denominator_norms,
            ratios=ratios, cumulative_seconds=probe_seconds,
            dataset_indices=indices.tolist(), seed=args.seed, epsilon=args.epsilon,
            batch_size=args.batch_size, config={k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()},
            runtime_scope='Cumulative probe loop wall time, excluding model/dataset setup')
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
