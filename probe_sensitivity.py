#!/usr/bin/env python3
"""Nested-prefix probe sensitivity using the existing estimator implementations."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
from flowbudget_config import gpu_default, load_config, resource_path
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from budget_utils import FORMULAS, predict_budget, NFE_PER_STEP

COUNTS = (25, 50, 100, 250, 500)


def summarize_pool(pool, model, solver, tau, epsilon, seed, counts=COUNTS):
    if not counts or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError('Counts must be positive integers')
    if len(set(counts)) != len(counts):
        raise ValueError('Counts must be unique')
    data = pool['probe_pool']
    values, times = data['ratios'], data['cumulative_seconds']
    if (len(values) < max(500, max(counts)) or len(times) != len(values) or data['seed'] != seed
            or data['epsilon'] != epsilon or data['batch_size'] != 1):
        raise ValueError('Insufficient probes or mismatched timings, seed, epsilon, or batch size')
    if any(not math.isfinite(v) or v < 0 for v in values + times):
        raise ValueError('Invalid probe values/timings')
    if times != sorted(times):
        raise ValueError('Timings must be cumulative')
    _, kappa, gamma = FORMULAS[solver]
    baseline = predict_budget(statistics.mean(values[:500]), kappa, gamma, tau)
    rows = []
    for n in sorted(counts):
        c = statistics.mean(values[:n])
        k = predict_budget(c, kappa, gamma, tau)
        rows.append(dict(model=model, solver=solver, N=n, seed=seed, tau=tau,
                         epsilon=epsilon, C_hat=c, K_hat=k,
                         predicted_NFE=k*NFE_PER_STEP[solver], runtime_seconds=times[n-1],
                         relative_deviation_500=(k-baseline)/baseline,
                         absolute_relative_deviation_500=abs(k-baseline)/baseline))
    return rows


def aggregate(rows):
    from scipy.stats import t
    result = []
    groups = sorted({(r['model'], r['solver'], r['N']) for r in rows})
    for model, solver, n in groups:
        group = [r for r in rows if (r['model'], r['solver'], r['N']) == (model, solver, n)]
        seeds = [r['seed'] for r in group]
        if len(set(seeds)) != len(seeds) or len(seeds) < 3:
            raise ValueError('Require at least three distinct seeds per group')
        ks = [r['K_hat'] for r in group]
        mean, sd = statistics.mean(ks), statistics.stdev(ks)
        margin = float(t.ppf(.975, len(ks)-1)) * sd/math.sqrt(len(ks))
        result.append(dict(model=model, solver=solver, N=n, num_seeds=len(ks),
                           mean_K_hat=mean, std_K_hat=sd, CV_K_hat=sd/mean,
                           CI95_low=mean-margin, CI95_high=mean+margin,
                           mean_predicted_NFE=mean*NFE_PER_STEP[solver],
                           mean_relative_deviation_500=statistics.mean(r['relative_deviation_500'] for r in group),
                           mean_absolute_relative_deviation_500=statistics.mean(r['absolute_relative_deviation_500'] for r in group),
                           mean_runtime_seconds=statistics.mean(r['runtime_seconds'] for r in group)))
    return result


def write_reports(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries = aggregate(rows)
    for filename, data in [('per_seed', rows), ('stability', summaries)]:
        with (output/f'{filename}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
        (output/f'{filename}.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    pairs = sorted({(r['model'],r['solver']) for r in summaries})
    fig, axes = plt.subplots(math.ceil(len(pairs)/3), 3, squeeze=False,
                             figsize=(10, 2.8*math.ceil(len(pairs)/3)), constrained_layout=True)
    for ax, pair in zip(axes.flat, pairs):
        data = [r for r in summaries if (r['model'],r['solver']) == pair]
        means = [r['mean_K_hat'] for r in data]
        errors = [r['CI95_high']-r['mean_K_hat'] for r in data]
        ax.errorbar([r['N'] for r in data], means, yerr=errors, fmt='o-', capsize=3)
        ax.set(title=' / '.join(pair), xlabel='Probe count N', ylabel='Predicted steps K')
        ax.set_xticks([r['N'] for r in data]); ax.tick_params(axis='x', labelsize=7, labelrotation=45)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(pairs):]: ax.set_visible(False)
    fig.suptitle(f"Nested probes: tau={rows[0]['tau']:.7g}, epsilon={rows[0]['epsilon']:g}; 95% t intervals")
    for ext in ('png','svg'): fig.savefig(output/f'probe_sensitivity.{ext}', dpi=180)
    plt.close(fig)
    return summaries


def select_models(config, models):
    """Select models in config order; retain every solver for each model."""
    if models is None:
        return config
    available = {pair['model'] for pair in config['pairs']}
    if not models or len(set(models)) != len(models) or set(models) - available:
        raise ValueError(f'Provide unique model names from: {sorted(available)}')
    return {**config, 'pairs': [pair for pair in config['pairs'] if pair['model'] in models]}


def main_ode(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Modes: --mode ode (default), --mode em (SiT EM nested probes), --mode merge (combine ODE reports).")
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--models', nargs='+', help='Select model names; default: all configured pairs')
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--gpu', default=gpu_default())
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--counts', type=int, nargs='+', default=list(COUNTS))
    args = parser.parse_args(argv)
    if not args.counts or min(args.counts) < 1 or len(set(args.counts)) != len(args.counts):
        parser.error('Provide unique positive probe counts')
    counts = sorted(args.counts)
    pool_size = max(500, max(counts))
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds)<3 or min(args.seeds)<0:
        parser.error('Provide at least 3 unique nonnegative seeds')
    if any(not math.isfinite(v) or v<=0 for v in (args.tau,args.epsilon)):
        parser.error('tau and epsilon must be finite and positive')
    try:
        config = select_models(load_config(args.config), args.models)
    except ValueError as error:
        parser.error(str(error))
    root = Path(config.get('base_dir', args.config.resolve().parent))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(config=config, tau=args.tau, epsilon=args.epsilon, seeds=args.seeds,
                    gpu=args.gpu, python=args.python, counts=counts)
    runfile = args.output_dir/'experiment.json'
    if runfile.exists() and json.loads(runfile.read_text()) != snapshot:
        raise ValueError('Output belongs to a different experiment; use a new directory')
    runfile.write_text(json.dumps(snapshot,indent=2)+'\n')
    rows=[]
    for index, spec in enumerate(config['pairs']):
        script = (root/spec['script']).resolve()
        for seed in args.seeds:
            folder=args.output_dir/f'pair_{index}'/f'seed_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            output=(folder/'pool.json').resolve()
            if not output.exists():
                if args.report_only: raise FileNotFoundError(output)
                command=[args.python,str(script),*spec['args'],'--num-samples',str(pool_size),
                         '--batch-size','1','--seed',str(seed),'--tau',str(args.tau),
                         '--epsilon',str(args.epsilon),'--save-probe-pool','--output',str(output)]
                (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                print(f"Generating {spec['model']}/{spec['solver']} seed={seed} on GPU {args.gpu}",flush=True)
                with (folder/'run.log').open('w') as log:
                    subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,
                                   PATH=str(Path(args.python).absolute().parent)+os.pathsep+os.environ.get('PATH','')),
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
            pool=json.loads(output.read_text())
            new=summarize_pool(pool,spec['model'],spec['solver'],args.tau,args.epsilon,seed,counts)
            for row in new:
                row['pool_path']=str(output)
                row['pool_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
            rows.extend(new)
        write_reports(rows,args.output_dir)
    write_reports(rows,args.output_dir)
    print(f'Wrote {len(rows)} per-seed rows and stability plots to {args.output_dir}',flush=True)


ROOT = Path(__file__).resolve().parent
EM_COUNTS = [25,50,100,250,400,500,700,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000]


def save_em_reports(rows, output, counts=EM_COUNTS):
    from scipy.stats import t
    summaries = []
    for n in counts:
        values = [r['K_hat'] for r in rows if r['N'] == n]
        if len(values) < 3:
            continue
        mean, sd = statistics.mean(values), statistics.stdev(values)
        margin = float(t.ppf(.975, len(values)-1))*sd/math.sqrt(len(values))
        summaries.append(dict(model='SiT-XL/2', solver='em', N=n, num_seeds=len(values),
            mean_K_hat=mean, std_K_hat=sd, CV_K_hat=sd/mean,
            CI95_low=mean-margin, CI95_high=mean+margin,
            mean_NFE_ideal=mean, mean_NFE_current_sampler=mean))
    for name, data in [('per_seed', rows), ('stability', summaries)]:
        if not data:
            continue
        with (output/f'{name}.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        (output/f'{name}.json').write_text(json.dumps(data, indent=2)+'\n')


def main_em(argv=None, *, default_counts=EM_COUNTS):
    parser=argparse.ArgumentParser(description=__doc__, epilog="Modes: --mode ode (default), --mode em (SiT EM nested probes), --mode merge (combine ODE reports).")
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--gpu', default=gpu_default())
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=list(default_counts),
                        help='Positive probe counts; each seed generates a pool of max(counts).')
    args=parser.parse_args(argv)
    if any(n <= 0 for n in args.counts) or len(set(args.counts)) != len(args.counts):
        parser.error('--counts must contain distinct positive integers')
    counts = sorted(args.counts)
    pool_size = max(counts)
    output=args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest=dict(model='SiT-XL/2', solver='em', counts=counts, seeds=[0,1,2], tau=.01,
        epsilon=1e-8, gpu=args.gpu, vae='ema', trace_probes=8, diffusion_norm=1.,
        t_min=.01, batch_size=1, estimator='SiT-Linear-EM-leading-order-v2',
        sampling=f'Nested prefixes of one {pool_size}-probe pool per seed')
    path=output/'experiment.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Output directory belongs to another experiment')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    if args.wait_pid:
        print(f'Waiting for process {args.wait_pid} before using GPU {args.gpu}', flush=True)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(30)
    sys.path.insert(0, str(ROOT/'SiT_Imagenet'))
    from estimate_em_budget import estimate_budget
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    rows=[]
    for seed in [0,1,2]:
        folder=output/f'seed_{seed}'
        folder.mkdir(exist_ok=True)
        pool=folder/'pool.json'
        command=[sys.executable, '-u', str(ROOT/'SiT_Imagenet/estimate_em_budget.py'),
            '--data-path',str(resource_path('datasets/imagenet_2012/images/train')),
            '--vae','ema','--num-samples',str(pool_size),'--batch-size','1','--trace-probes','8',
            '--diffusion-norm','1','--t-min','0.01','--tau','0.01','--epsilon','1e-8',
            '--seed',str(seed),'--device',os.environ.get('FLOWBUDGET_DEVICE', 'cuda'),'--output',str(pool)]
        (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
        if not pool.exists():
            print(f'Generating SiT-XL/2 EM seed={seed} on GPU {args.gpu}', flush=True)
            with (folder/'run.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        raw=pool.read_bytes()
        data=json.loads(raw)
        assert data['estimator']==manifest['estimator']
        assert len(data['probes'])==pool_size
        for key in ['seed','vae','batch_size','trace_probes','diffusion_norm','t_min']:
            assert data['metadata'][key] == (seed if key=='seed' else manifest[key]), key
        for n in counts:
            budget=estimate_budget(data['probes'][:n], .01, epsilon=1e-8)
            if budget['K_hat'] is None:
                raise ValueError(f'No budget found for seed={seed}, N={n}')
            rows.append(dict(model='SiT-XL/2', solver='em', N=n, seed=seed, tau=.01,
                epsilon=1e-8, **{k:budget[k] for k in ['K_hat','R_theta_em','relative_error',
                'NFE_hat_ideal','NFE_hat_current_sampler']}, pool_path=str(pool),
                pool_sha256=hashlib.sha256(raw).hexdigest()))
            print(f"seed={seed} N={n} K_hat={budget['K_hat']}", flush=True)
        save_em_reports(rows, output, counts)
    print(f'Completed: {output}', flush=True)


def merge(inputs, output):
    rows, sources, seen = [], [], set()
    reference = None
    for directory in inputs:
        directory = directory.resolve()
        manifest = directory / 'experiment.json'
        experiment = json.loads(manifest.read_text())
        settings = {key: experiment[key] for key in ('tau', 'epsilon', 'seeds', 'counts')}
        if settings['counts'] != list(COUNTS):
            raise ValueError(f'Unexpected probe counts: {manifest}')
        if reference is None:
            reference = settings
        elif settings != reference:
            raise ValueError('Cannot combine experiments with different tau, epsilon, seeds, or counts')
        sources.append(dict(path=str(manifest), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            gpu=experiment['gpu']))
        for index, spec in enumerate(experiment['config']['pairs']):
            pair = (spec['model'], spec['solver'])
            if pair in seen:
                raise ValueError(f'Duplicate model/solver pair: {pair}')
            seen.add(pair)
            for seed in settings['seeds']:
                pool_path = directory / f'pair_{index}' / f'seed_{seed}' / 'pool.json'
                raw = pool_path.read_bytes()
                pool = json.loads(raw)
                if pool['tau'] != settings['tau']:
                    raise ValueError(f'Pool tau differs: {pool_path}')
                group = summarize_pool(pool, *pair, settings['tau'], settings['epsilon'], seed)
                for row in group:
                    row.update(pool_path=str(pool_path), pool_sha256=hashlib.sha256(raw).hexdigest())
                rows.extend(group)
    if not rows:
        raise ValueError('No input experiments')
    output.mkdir(parents=True, exist_ok=True)
    write_reports(rows, output)
    (output / 'sources.json').write_text(json.dumps(dict(settings=reference, sources=sources,
        note='Runtime comparisons may involve different GPUs; see source manifests.'), indent=2) + '\n')
    print(f'Wrote {len(rows)} rows for {len(seen)} model/solver pairs to {output}')


def main_merge(argv=None):
    parser = argparse.ArgumentParser(description='Combine compatible ODE probe reports from completed experiment directories.')
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    merge(args.inputs, args.output_dir)


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--mode', choices=['ode', 'em', 'merge'], default='ode')
    mode, remaining = selector.parse_known_args()
    if mode.mode == 'merge':
        main_merge(remaining)
    elif mode.mode == 'em':
        main_em(remaining)
    else:
        main_ode(remaining)


if __name__ == '__main__': main()

