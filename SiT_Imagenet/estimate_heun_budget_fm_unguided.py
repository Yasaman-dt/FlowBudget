"""Estimate the deterministic Heun step budget for unguided SiT.

For the second-order Heun method, the normalized leading local error is

    ||e_H|| / ||Delta x|| ~= R_H / (12 K**2),

where R_H = E[||2 J_x v a - D^2v[(v,1),(v,1)]|| / (||v|| + eps)].
Consequently K_hat = ceil(sqrt(R_H / (12 tau))) and NFE_hat = 2 K_hat.
"""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import device_default, optional_path

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from download import find_model
from models import SiT_models


def center_crop_arr(pil_image, image_size):
    """Deterministic copy of the crop used in train.py."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )
    array = np.array(pil_image)
    y = (array.shape[0] - image_size) // 2
    x = (array.shape[1] - image_size) // 2
    return Image.fromarray(array[y : y + image_size, x : x + image_size])


def load_model(checkpoint, device):
    model = SiT_models["SiT-XL/2"](
        input_size=32, num_classes=1000, learn_sigma=True
    ).to(device)
    state_dict = find_model(
        str(checkpoint) if checkpoint else "SiT-XL-2-256x256.pt"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def enable_higher_derivative_attention():
    """Select the SDPA backend that supports the required higher derivatives."""
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def heun_derivatives(velocity, x, t):
    """Return v, a, g and c from the supplied Heun-error derivation.

    The direction q=(v, 1) is frozen at the evaluation point when computing
    c = D^2 v[q, q]. This is distinct from differentiating a=Dv[q], where q
    itself varies with (x,t); that latter derivative would produce c+g.
    """
    with torch.no_grad():
        v = velocity(x, t)
    v_direction = v.detach()
    time_direction = torch.ones_like(t)

    _, acceleration = jvp(
        velocity,
        (x, t),
        (v_direction, time_direction),
        create_graph=False,
    )
    acceleration_direction = acceleration.detach()

    _, g = jvp(
        velocity,
        (x, t),
        (acceleration_direction, torch.zeros_like(t)),
        create_graph=False,
    )

    def first_directional(x_value, t_value):
        return jvp(
            velocity,
            (x_value, t_value),
            (v_direction, time_direction),
            create_graph=True,
        )[1]

    _, c = jvp(
        first_directional,
        (x, t),
        (v_direction, time_direction),
        create_graph=False,
    )
    return v, acceleration, g, c


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=optional_path("FLOWBUDGET_SIT_CHECKPOINT"))
    parser.add_argument("--vae", choices=["ema", "mse"], default="mse")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-probe-pool", action="store_true",
                        help="Record raw q_S/d_S, ordered ratios and timings; requires batch size 1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. By default, writes "
            "results/fid/fid_sweep_heun_10k/estimate_<num-samples>.json next to this script."
        ),
    )
    parser.add_argument("--device", default=device_default(torch.cuda.is_available()),
                        help="cpu or a logical CUDA device, e.g. cuda:0")
    args = parser.parse_args()
    if args.save_probe_pool and args.batch_size != 1:
        parser.error("--save-probe-pool requires --batch-size 1")
    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent
            / "results/fid/fid_sweep_heun_10k"
            / f"estimate_{args.num_samples}.json"
        )
    if args.tau <= 0:
        parser.error("--tau must be positive")
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    if args.num_samples <= 0 or args.batch_size <= 0:
        parser.error("--num-samples and --batch-size must be positive")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    enable_higher_derivative_attention()
    print(f"Device: {device}")
    print("Estimator attention backend: math SDPA (required for nested JVPs).")

    model = load_model(args.checkpoint, device)
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae}"
    ).to(device).eval()
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    if args.num_samples > len(dataset):
        raise ValueError("--num-samples exceeds dataset size")
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num_samples]
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    ratios = []
    velocity_norms = []
    acceleration_norms = []
    g_norms = []
    c_norms = []
    error_coefficient_norms = []
    probe_seconds = []
    started = time.time()

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            x_data = vae.encode(images).latent_dist.sample().mul_(0.18215)
        noise = torch.randn_like(x_data)
        t = torch.rand(x_data.shape[0], device=device)

        # This is SiT's implemented Linear path: t=0 is noise and t=1 is data.
        x_t = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * x_data

        def velocity(x, time_value):
            return model(x, time_value, labels)

        v, acceleration, g, c = heun_derivatives(velocity, x_t, t)
        with torch.no_grad():
            error_coefficient = 2.0 * g - c
            v_norm = v.flatten(1).norm(dim=1)
            a_norm = acceleration.flatten(1).norm(dim=1)
            g_norm = g.flatten(1).norm(dim=1)
            c_norm = c.flatten(1).norm(dim=1)
            coefficient_norm = error_coefficient.flatten(1).norm(dim=1)
            ratio = coefficient_norm / (v_norm + args.epsilon)

            ratios.extend(ratio.cpu().tolist())
            if args.save_probe_pool:
                probe_seconds.append(time.time() - started)
            velocity_norms.extend(v_norm.cpu().tolist())
            acceleration_norms.extend(a_norm.cpu().tolist())
            g_norms.extend(g_norm.cpu().tolist())
            c_norms.extend(c_norm.cpu().tolist())
            error_coefficient_norms.extend(coefficient_norm.cpu().tolist())
        print(f"Processed {len(ratios)}/{args.num_samples} samples (batch {batch_index}).")

    r_theta_heun = sum(ratios) / len(ratios)
    k_hat = max(1, math.ceil(math.sqrt(r_theta_heun / (12.0 * args.tau))))
    result = {
        "R_theta_heun": r_theta_heun,
        "K_hat": k_hat,
        "integrator_time_points": k_hat + 1,
        "NFE_hat": 2 * k_hat,
        "K_hat_formula": "ceil(sqrt(R_theta_heun / (12 * tau)))",
        "R_theta_formula": "E[||2 * J_x(v) * a - D2(v)[(v,1),(v,1)]|| / (||v|| + epsilon)]",
        "K_hat_solver": "deterministic_unguided_heun2_ode",
        "compatible_sampling_method": "ODE/heun2, cfg_scale=1.0",
        "tau": args.tau,
        "epsilon": args.epsilon,
        "num_samples": len(ratios),
        "r_median": float(torch.tensor(ratios).median()),
        "r_std": float(torch.tensor(ratios).std(unbiased=False)),
        "mean_velocity_norm": sum(velocity_norms) / len(velocity_norms),
        "mean_acceleration_norm": sum(acceleration_norms) / len(acceleration_norms),
        "mean_g_norm": sum(g_norms) / len(g_norms),
        "mean_c_norm": sum(c_norms) / len(c_norms),
        "mean_error_coefficient_norm": (
            sum(error_coefficient_norms) / len(error_coefficient_norms)
        ),
        "cfg_scale": 1.0,
        "vae": args.vae,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_probe_pool:
        result['probe_pool'] = dict(
            schema_version=2, q_S=error_coefficient_norms, d_S=velocity_norms,
            ratios=ratios, cumulative_seconds=probe_seconds,
            dataset_indices=indices.tolist(), seed=args.seed, epsilon=args.epsilon,
            batch_size=args.batch_size, config={k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()},
            runtime_scope='Cumulative probe loop wall time, excluding model/dataset setup')
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
