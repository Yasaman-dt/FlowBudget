#!/usr/bin/env python3
"""Estimate unguided SiT EM budgets using the manuscript leading-order proxy.

Sampling time s=1-t runs noise -> data. In this convention F=v+d*score,
G=g I, A=J(F)G, B=g' I, D=partial_s F+J(F)F+0.5*g^2*Laplacian(F).
The sign changes relative to reverse time leave q^2 and ||D||^2 invariant.
Hutchinson estimates use Rademacher directions; no full Jacobian is allocated.
"""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import optional_path, device_default
import argparse
import csv
import json
import math
from pathlib import Path
import time

import torch
from torch.autograd.functional import jvp


def expand(t, x):
    return t.reshape((-1,) + (1,) * (x.ndim - 1))


def make_drift(velocity, norm=1.0):
    """Linear path, d(s)=norm*(1-s); algebraically fused drift, one model call."""
    def drift(x, s):
        return (1 + norm * expand(s, x)) * velocity(x, s) - norm * x
    return drift


@torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
def em_statistics(drift, x, s, norm=1.0, trace_probes=8, generator=None, augmented=False):
    """Estimate q2 and g2 for the leading-order manuscript budget.

    Shared Rademacher directions preserve the A/B cross term. Optional
    augmented=True computes historical higher-order statistics for validation;
    those terms are never used by the leading-order budget function.
    """
    if trace_probes < 1 or norm < 0 or not math.isfinite(norm):
        raise ValueError('Require positive trace_probes and finite norm >= 0')
    if not bool(((s >= 0) & (s < 1)).all()):
        raise ValueError('Sampling times must satisfy 0 <= s < 1')
    with torch.no_grad():
        f = drift(x, s).detach()
    advective = jvp(drift, (x, s), (f, torch.ones_like(s)))[1] if augmented else torch.zeros_like(x)
    g = torch.sqrt(2 * norm * (1 - s))
    gp = -norm / g if norm else torch.zeros_like(s)
    trace = torch.zeros_like(x)
    q2 = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for _ in range(trace_probes):
        z = torch.randint(0, 2, x.shape, device=x.device, generator=generator).to(x.dtype) * 2 - 1
        direction = (expand(g, x) * z).detach()
        _, az = jvp(lambda state: drift(state, s), x, direction)
        bz = expand(gp, x) * z
        # Equivalent to ||Az||^2+||Bz||^2+<Az,Bz>, nonnegative by construction.
        q2 += 0.5 * (az.flatten(1).square().sum(1) + bz.flatten(1).square().sum(1)
                     + (az + bz).flatten(1).square().sum(1)) / trace_probes
        def first(state):
            return jvp(lambda value: drift(value, s), state, direction, create_graph=True)[1]
        if augmented:
            _, hz = jvp(first, x, direction)
            trace += hz.detach() / trace_probes
    d = advective.detach() + 0.5 * trace
    values = dict(q2=q2, d2=d.flatten(1).square().sum(1),
                  f2=f.flatten(1).square().sum(1), g2=g.square() * x[0].numel(), s=s)
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError('Nonfinite derivative statistics; inspect model/time cutoff')
    return [{key: float(value[i].detach().cpu()) for key, value in values.items()}
            for i in range(x.shape[0])]


def leading_coefficient(rows, epsilon=1e-8):
    if not rows or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Require nonempty probes and epsilon > 0')
    for row in rows:
        if not math.isfinite(row['q2']) or row['q2'] < 0 or not math.isfinite(row['g2']) or row['g2'] <= 0:
            raise ValueError('Leading EM proxy requires finite q2 >= 0 and g2 > 0')
    return sum(math.sqrt(r['q2']) / (math.sqrt(r['g2']) + epsilon) for r in rows) / len(rows)


def relative_error(rows, k, epsilon=1e-8):
    if k < 1:
        raise ValueError('Require K >= 1')
    return leading_coefficient(rows, epsilon) / (math.sqrt(3) * k)


def estimate_budget(rows, tau, max_k=100000, epsilon=1e-8):
    if not math.isfinite(tau) or tau <= 0 or max_k < 1:
        raise ValueError('Require finite tau > 0 and max_k >= 1')
    coefficient = leading_coefficient(rows, epsilon)
    k = max(1, math.ceil(coefficient / (math.sqrt(3) * tau)))
    found = k <= max_k
    return dict(tau=tau, K_hat=k if found else None, C_EM=coefficient,
                R_theta_em=coefficient/math.sqrt(3),
                relative_error=relative_error(rows, k if found else max_k, epsilon),
                previous_relative_error=relative_error(rows, k-1, epsilon) if found and k>1 else None,
                integrator_time_points=k+1 if found else None,
                NFE_hat_ideal=k if found else None,
                NFE_hat_current_sampler=k if found else None,
                status='found' if found else f'no budget <= {max_k}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=optional_path('FLOWBUDGET_SIT_CHECKPOINT'))
    parser.add_argument('--from-statistics', type=Path, help='Reuse this estimator\'s saved JSON; no model evaluation')
    parser.add_argument('--vae', choices=['ema', 'mse'], default='ema')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--trace-probes', type=int, default=8)
    parser.add_argument('--diffusion-norm', type=float, default=1.0)
    parser.add_argument('--t-min', type=float, default=0.01, help='Manuscript time cutoff; sample s uniformly in [0,1-t_min]')
    parser.add_argument('--taus', '--tau', type=float, nargs='+', default=[0.01, 0.015, 0.02, 0.025, 0.03])
    parser.add_argument('--max-k', type=int, default=100000)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default=device_default(torch.cuda.is_available()))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'em_budget_estimates/estimate.json')
    args = parser.parse_args()
    if (not 0 < args.t_min < 1 or not math.isfinite(args.diffusion_norm) or args.diffusion_norm <= 0
            or not math.isfinite(args.epsilon) or args.epsilon <= 0
            or min(args.num_samples, args.batch_size, args.trace_probes, args.max_k) < 1
            or args.workers < 0 or any(not math.isfinite(t) or t <= 0 for t in args.taus)):
        parser.error('Invalid cutoff, diffusion norm, tolerance, epsilon, or sample/search counts')
    started = time.time()
    if args.from_statistics:
        saved = json.loads(args.from_statistics.read_text())
        if saved.get('estimator') not in ('SiT-Linear-EM-augmented-RMS-v1', 'SiT-Linear-EM-leading-order-v2'):
            parser.error('Expected statistics from estimate_em_budget.py')
        rows, metadata = saved['probes'], saved['metadata']
    else:
        if args.data_path is None:
            parser.error('--data-path is required unless --from-statistics is supplied')
        from estimate_heun_budget_fm_unguided import center_crop_arr, load_model, enable_higher_derivative_attention
        from diffusers.models import AutoencoderKL
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, Subset
        torch.manual_seed(args.seed)
        enable_higher_derivative_attention()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        model = load_model(args.checkpoint, device).requires_grad_(False)
        vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{args.vae}').to(device).eval().requires_grad_(False)
        transform = transforms.Compose([transforms.Lambda(lambda im: center_crop_arr(im, 256)),
                   transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
        dataset = ImageFolder(args.data_path, transform=transform)
        if args.num_samples > len(dataset):
            raise ValueError('num_samples exceeds dataset size')
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.num_samples].tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers)
        trace_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        rows = []
        for images, labels in loader:
            labels = labels.to(device)
            with torch.no_grad():
                data = vae.encode(images.to(device)).latent_dist.sample() * 0.18215
                noise = torch.randn_like(data)
                s = torch.rand(data.shape[0], device=device) * (1-args.t_min)
                x = expand(s, data)*data + (1-expand(s, data))*noise
            drift = make_drift(lambda state, time_value: model(state, time_value, labels), args.diffusion_norm)
            rows.extend(em_statistics(drift, x, s, args.diffusion_norm, args.trace_probes, trace_generator))
            print(f'Processed {len(rows)}/{args.num_samples} interpolation probes', flush=True)
        metadata = dict(checkpoint=str(args.checkpoint) if args.checkpoint else 'SiT-XL-2-256x256.pt',
                        data_path=str(args.data_path.resolve()), dataset_indices=indices,
                        vae=args.vae, seed=args.seed, batch_size=args.batch_size,
                        trace_probes=args.trace_probes, diffusion_form='sigma', diffusion_norm=args.diffusion_norm,
                        cfg_scale=1.0, last_step='None', t_min=args.t_min,
                        sampling_time_interval=[0, 1], probe_sampling_time_interval=[0, 1-args.t_min],
                        trace_method='Rademacher JVP; leading-order coefficient only',
                        caveat='Square root of the finite-direction q2 estimate has Monte Carlo bias. Local proxy, not a FID guarantee.')
    if not rows or any(not math.isfinite(row[key]) or row[key] < 0 for row in rows for key in ['q2','d2','f2','g2']):
        raise ValueError('Invalid saved probe statistics')
    # Save expensive statistics before searching so they survive a slow search.
    result = dict(estimator='SiT-Linear-EM-leading-order-v2', metadata=metadata,
                  num_samples=len(rows), probes=rows, epsilon=args.epsilon, max_k=args.max_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    result['budgets'] = [estimate_budget(rows, tau, args.max_k, args.epsilon) for tau in args.taus]
    result['elapsed_seconds'] = time.time()-started
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with args.output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=result['budgets'][0].keys())
        writer.writeheader()
        writer.writerows(result['budgets'])
    print(json.dumps(result['budgets'], indent=2))
    print(f'Wrote {args.output} and {args.output.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
