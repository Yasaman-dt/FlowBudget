"""FM.pdf K-hat estimator for original CIFAR-10 RectifiedFlow Euler ODE."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import resource_path, device_default
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from rf_cifar10_common import load_model


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=root / "ImageGeneration")
    parser.add_argument("--checkpoint", type=Path, default=resource_path(root / "checkpoints/cifar10_1_rectified_flow.pth"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-probe-pool", action="store_true",
                        help="Record raw q_S/d_S, ordered ratios and timings; requires batch size 1")
    parser.add_argument("--output", type=Path, default=root / "euler_budget_estimate_500.json")
    parser.add_argument("--device", default=device_default(torch.cuda.is_available()),
                        help="cpu or a logical CUDA device, e.g. cuda:0")
    args = parser.parse_args()
    if args.save_probe_pool and args.batch_size != 1:
        parser.error("--save-probe-pool requires --batch-size 1")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, _, step = load_model(args.runtime_root, args.checkpoint, device)
    dataset = CIFAR10(args.data_root, train=True, download=True, transform=transforms.ToTensor())
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.num_samples]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    ratios = []
    numerator_norms, denominator_norms = [], []
    probe_seconds = []
    started = time.time()
    for x_data, _ in loader:
        x_data = (x_data.to(device) - 0.5) * 2.0  # original config.data.centered=True
        z = torch.randn_like(x_data)
        t = torch.rand(x_data.shape[0], device=device)
        x_t = (1 - t[:, None, None, None]) * z + t[:, None, None, None] * x_data
        def velocity(x, time_value):
            return model(x, time_value * 999.0)
        with torch.no_grad():
            v = velocity(x_t, t)
        _, a = jvp(velocity, (x_t, t), (v, torch.ones_like(t)), create_graph=False)
        q_norm = a.flatten(1).norm(dim=1)
        d_norm = v.flatten(1).norm(dim=1)
        ratios.extend((q_norm / (d_norm + args.epsilon)).detach().cpu().tolist())
        if args.save_probe_pool:
            numerator_norms.extend(q_norm.detach().cpu().tolist())
            denominator_norms.extend(d_norm.detach().cpu().tolist())
        if args.save_probe_pool:
            probe_seconds.append(time.time() - started)
    r_theta = sum(ratios) / len(ratios)
    result = {"R_theta": r_theta,
              "K_hat": max(1, math.ceil(r_theta / (2.0 * args.tau))),
              "K_hat_formula": "ceil(R_theta / (2 * tau))",
              "tau": args.tau,
              "num_samples": len(ratios), "checkpoint_step": step, "solver": "rectified_flow_ode/euler",
              "cfg_scale": "not_applicable", "r_median": float(torch.tensor(ratios).median()),
              "r_std": float(torch.tensor(ratios).std(unbiased=False)), "elapsed_seconds": time.time() - started}
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
