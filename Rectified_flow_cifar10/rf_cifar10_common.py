"""Original RectifiedFlow CIFAR-10 model loading and Euler sampling helpers."""
import os
import sys
from pathlib import Path

import torch


def load_model(runtime_root, checkpoint, device):
    # Keep the original repository's compiled CUDA extensions with this
    # experiment instead of placing them in a shared home-directory cache.
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path(__file__).resolve().parent / "torch_extensions"))
    runtime_root = str(Path(runtime_root).resolve())
    if runtime_root not in sys.path:
        sys.path.insert(0, runtime_root)
    from configs.rectified_flow.cifar10_rf_gaussian_ddpmpp import get_config
    from models import ncsnpp  # noqa: F401 -- registers ncsnpp in the original model registry.
    from models import utils as mutils

    config = get_config()
    config.device = device
    model = mutils.create_model(config)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    # Original evaluation copies EMA weights into the model before sampling.
    for parameter, ema_parameter in zip(model.parameters(), state["ema"]["shadow_params"]):
        parameter.data.copy_(ema_parameter.to(device))
    model.eval()
    return model, config, int(state["step"])


@torch.no_grad()
def euler_sample(model, batch_size, k, device):
    """Exact deterministic branch of original sampling.py's euler_sampler.

    The official CIFAR configuration sets sigma_variance=0, so pred_sigma is
    exactly pred and no stochastic SDE term is present.
    """
    eps = 1e-3
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    dt = 1.0 / k
    for i in range(k):
        time = i / k * (1.0 - eps) + eps
        t = torch.full((batch_size,), time * 999.0, device=device)
        x = x + model(x, t) * dt
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
