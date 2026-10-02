"""Estimate the deterministic Euler budget K-hat from FM.pdf for unguided SiT."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import device_default, optional_path
import argparse
import json
import math
import time
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
import numpy as np
from PIL import Image
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from download import find_model
from models import SiT_models


def center_crop_arr(pil_image, image_size):
    """Deterministic copy of the crop used in train.py (without augmentation)."""
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
    model = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000, learn_sigma=True).to(device)
    state_dict = find_model(str(checkpoint) if checkpoint else "SiT-XL-2-256x256.pt")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def enable_jvp_compatible_attention():
    """Use the SDPA math backend, whose higher derivatives are implemented.

    The memory-efficient CUDA attention kernel is excellent for sampling but
    lacks the second-order derivative used internally by autograd.functional.jvp.
    This only affects the FM estimator process.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=optional_path("FLOWBUDGET_SIT_CHECKPOINT"))
    parser.add_argument("--vae", choices=["ema", "mse"], default="ema")
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
            "results/fid/fid_sweep_unguided_10k/estimate_<num-samples>.json next to this script."
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
            / "results/fid/fid_sweep_unguided_10k"
            / f"estimate_{args.num_samples}.json"
        )
    if args.tau <= 0:
        raise ValueError("--tau must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    enable_jvp_compatible_attention()
    print("FM estimator attention backend: math SDPA (required for JVP derivatives).")
    model = load_model(args.checkpoint, device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    if args.num_samples > len(dataset):
        raise ValueError("--num-samples exceeds dataset size")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.num_samples]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    ratios, velocity_norms, acceleration_norms = [], [], []
    probe_seconds = []
    started = time.time()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            x_data = vae.encode(images).latent_dist.sample().mul_(0.18215)
        z = torch.randn_like(x_data)
        t = torch.rand(x_data.shape[0], device=device)
        x_t = (1 - t[:, None, None, None]) * z + t[:, None, None, None] * x_data

        def velocity(x, time_value):
            return model(x, time_value, labels)

        with torch.no_grad():
            v = velocity(x_t, t)
        # This evaluates d/dt v_theta(x_t, t) = J_x v_theta + partial_t v_theta.
        # torch.autograd.functional.jvp is available in the PyTorch versions
        # supported by the original SiT environment.yml.
        _, a = jvp(velocity, (x_t, t), (v, torch.ones_like(t)), create_graph=False)
        with torch.no_grad():
            v_norm = v.flatten(1).norm(dim=1)
            a_norm = a.flatten(1).norm(dim=1)
            ratios.extend((a_norm / (v_norm + args.epsilon)).cpu().tolist())
            if args.save_probe_pool:
                probe_seconds.append(time.time() - started)
            velocity_norms.extend(v_norm.cpu().tolist())
            acceleration_norms.extend(a_norm.cpu().tolist())

    r_theta = sum(ratios) / len(ratios)
    result = {
        "R_theta": r_theta,
        "K_hat": max(1, math.ceil(r_theta / (2.0 * args.tau))),
        "K_hat_formula": "ceil(R_theta / (2 * tau))",
        "K_hat_solver": "deterministic_unguided_euler_ode",
        "compatible_sweep": "sweep_fid_vs_k_unguided.py (ODE/euler, cfg_scale=1.0)",
        "not_compatible": "SiT paper SDE/Euler--Maruyama and ODE/Heun results",
        "tau": args.tau, "epsilon": args.epsilon, "num_samples": len(ratios),
        "r_median": float(torch.tensor(ratios).median()),
        "r_std": float(torch.tensor(ratios).std(unbiased=False)),
        "mean_velocity_norm": sum(velocity_norms) / len(velocity_norms),
        "mean_acceleration_norm": sum(acceleration_norms) / len(acceleration_norms),
        "cfg_scale": 1.0, "vae": args.vae, "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_probe_pool:
        result['probe_pool'] = dict(
            schema_version=2, q_S=acceleration_norms, d_S=velocity_norms,
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
