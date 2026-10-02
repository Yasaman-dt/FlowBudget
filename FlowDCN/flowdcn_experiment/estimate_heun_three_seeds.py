"""Estimate FlowDCN Heun budgets on three independent 2K-probe sets.

Only estimation is performed. The Heun coefficient is
E[||2 J_x(v) a - D²v[(v,1),(v,1)]|| / (||v|| + epsilon)],
where a = Dv[(v,1)]. Centered finite differences avoid second derivatives
through FlowDCN's custom Triton operation. Agreement across increments is
reported as a sensitivity check, not assumed.
"""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[2]))
from flowbudget_config import resource_path

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import torch

from run import ROOT, CHECKPOINT, load, save

INCREMENTS = (0.01, 0.02, 0.04)
SELECTED_INCREMENT = 0.02
TIME_MARGIN = 0.04
SEEDS = (0, 1, 2)


def heun_error_vector(velocity, x, t, increment):
    """Compute 2 J_x(v) a - D²v[(v,1),(v,1)] by centered differences."""
    v = velocity(x, t)
    direction = v.detach()
    v_plus = velocity(x + increment * direction, t + increment)
    v_minus = velocity(x - increment * direction, t - increment)
    acceleration = (v_plus - v_minus) / (2 * increment)
    a_direction = acceleration.detach()
    g = (velocity(x + increment * a_direction, t)
         - velocity(x - increment * a_direction, t)) / (2 * increment)
    c = (v_plus - 2 * v + v_minus) / (increment * increment)
    return v, 2 * g - c


def probe_ratios(velocity, x, t, increments=INCREMENTS, epsilon=1e-8):
    values = {}
    for h in increments:
        v, error = heun_error_vector(velocity, x, t, h)
        ratio = error.flatten(1).norm(dim=1) / (v.flatten(1).norm(dim=1) + epsilon)
        if not bool(torch.isfinite(ratio).all()):
            raise RuntimeError(f'Nonfinite Heun coefficient for increment {h}')
        values[str(h)] = ratio.double().cpu().tolist()
    return values


def budget(coefficient, tau):
    return max(1, math.ceil(math.sqrt(coefficient / (12 * tau))))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results' / 'estimates' / 'estimation_ode_heun')
    parser.add_argument('--probes', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    if args.probes < 1 or args.batch_size < 1 or not math.isfinite(args.tau) or args.tau <= 0:
        parser.error('Require positive probes, batch size, and tau')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be distinct')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for FlowDCN')

    import fcntl
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(ROOT / 'SiT_Imagenet'))
    from estimate_euler_budget_fm_unguided import center_crop_arr

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = dict(model='FlowDCN-XL-2M', solver='ODE Heun', N=args.probes,
                batch_size=args.batch_size,
                tau=args.tau, epsilon=1e-8, seeds=args.seeds, cfg_scale=1.0,
                checkpoint=str(CHECKPOINT), increments=list(INCREMENTS),
                selected_increment=SELECTED_INCREMENT,
                probe_time_interval=[TIME_MARGIN, 1 - TIME_MARGIN],
                formula='K_hat=ceil(sqrt(R_H/(12*tau))); NFE_hat=2*K_hat',
                scope='Estimation only; no sampling or FID',
                caveat='Finite differences through the custom Triton kernel are checked across increments; no analytic second derivatives are assumed.')
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Output directory belongs to different settings')
    save(plan_path, plan)

    model, vae = load()
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, 256)),
        transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3),
    ])
    dataset = ImageFolder(resource_path(ROOT.parent / 'datasets/imagenet_2012/images/train'),
                          transform=transform)
    if args.probes > len(dataset):
        raise ValueError('Probe count exceeds dataset size')

    rows = []
    for seed in args.seeds:
        folder = out / f'seed_{seed}'
        folder.mkdir(exist_ok=True)
        result_path = folder / 'estimate.json'
        indices = torch.randperm(len(dataset),
                                 generator=torch.Generator().manual_seed(seed))[:args.probes].tolist()
        if not result_path.exists():
            progress_path = folder / 'progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
            ratios = progress['ratios'] if progress else {str(h): [] for h in INCREMENTS}
            completed = progress['completed'] if progress else 0
            if completed != len(ratios[str(SELECTED_INCREMENT)]) or completed > args.probes:
                raise RuntimeError('Inconsistent progress file')
            loader = DataLoader(Subset(dataset, indices[completed:]), batch_size=args.batch_size,
                                num_workers=0)
            last_saved = completed
            for batch_index, (images, labels) in enumerate(loader):
                start = completed + batch_index * args.batch_size
                torch.manual_seed(seed * 1_000_000 + start)
                with torch.no_grad():
                    data = vae.encode(images.cuda()).latent_dist.sample() * .18215
                    noise = torch.randn_like(data)
                    t = TIME_MARGIN + (1 - 2 * TIME_MARGIN) * torch.rand(1, device='cuda')
                    x = (1 - t.reshape(-1, 1, 1, 1)) * noise + t.reshape(-1, 1, 1, 1) * data
                    label = labels.cuda()
                    velocity = lambda state, time: model(state, time, label)
                    values = probe_ratios(velocity, x, t)
                for h in INCREMENTS:
                    ratios[str(h)].extend(values[str(h)])
                count = start + images.shape[0]
                if count - last_saved >= 25 or count == args.probes:
                    save(progress_path, dict(completed=count, ratios=ratios))
                    last_saved = count
                    print(f'Seed {seed}: {count}/{args.probes} probes', flush=True)

            results = {str(h): dict(R_H=statistics.mean(ratios[str(h)]),
                                    K_hat=budget(statistics.mean(ratios[str(h)]), args.tau))
                       for h in INCREMENTS}
            chosen = results[str(SELECTED_INCREMENT)]
            spread = (max(r['K_hat'] for r in results.values())
                      - min(r['K_hat'] for r in results.values())) / chosen['K_hat']
            save(result_path, dict(seed=seed, N=args.probes, tau=args.tau,
                                   K_hat=chosen['K_hat'], NFE_hat=2 * chosen['K_hat'],
                                   R_H=chosen['R_H'], results=results, ratios=ratios,
                                   dataset_indices=indices, relative_budget_spread=spread,
                                   sensitivity_passed=spread <= .05, settings=plan))
        report = json.loads(result_path.read_text())
        if (report['seed'], report['N'], report['tau']) != (seed, args.probes, args.tau):
            raise RuntimeError(f'Unexpected result settings for seed {seed}')
        rows.append(dict(seed=seed, N=args.probes, tau=args.tau,
                         K_hat=report['K_hat'], NFE_hat=report['NFE_hat'],
                         R_H=report['R_H'], sensitivity_passed=report['sensitivity_passed'],
                         relative_budget_spread=report['relative_budget_spread'],
                         source=str(result_path)))
        write_csv(out / 'per_seed.csv', rows)
        print(f'Seed {seed}: K_hat={report["K_hat"]}, NFE_hat={report["NFE_hat"]}, '
              f'sensitivity_passed={report["sensitivity_passed"]}', flush=True)

    ks = [row['K_hat'] for row in rows]
    save(out / 'summary.json', dict(N=args.probes, tau=args.tau, seeds=args.seeds,
                                    mean_K_hat=statistics.mean(ks),
                                    std_K_hat=statistics.stdev(ks) if len(ks) > 1 else None,
                                    mean_NFE_hat=2 * statistics.mean(ks),
                                    all_sensitivity_passed=all(r['sensitivity_passed'] for r in rows),
                                    per_seed=rows, settings=plan))


if __name__ == '__main__':
    main()
