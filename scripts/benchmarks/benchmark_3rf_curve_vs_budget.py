"""Time the 3-RF FID curve and the single-point budget-estimation route on selected GPU."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[2]))
from flowbudget_config import resource_path, gpu_default

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RF = ROOT / 'rectified_flow_cifar10'
OUT = RF / 'results' / 'timing' / '3rf' / 'curve_vs_budget' / 'new_run'
CHECKPOINT = resource_path(RF / 'checkpoints/cifar10_3_rectified_flow.pth')
DATA = resource_path(RF / 'data')
REFERENCE = resource_path(RF / 'references/cifar10_test_10k.npz')
EVALUATOR = resource_path(ROOT.parent / 'guided-diffusion/evaluations/evaluator.py')
CURRENT_COMPLETE = ROOT / 'FlowDCN/results/fid/fid_ode_heun/complete.json'
HEUN_EXPORT_COMPLETE = ROOT / 'FlowDCN/results/fid/fid_ode_heun/images_export_complete.json'
WAIT_PID = int(os.environ.get('FLOWBUDGET_WAIT_PID', '0'))
CURVE_K = [1, 2, 4, 8, 16, 32]
IMAGES = 10000
PROBES = 2000
BATCH_SIZE = 128
SEED = 0
TAU = 0.01


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def process_start_ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def selected_gpu_uuid():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    for line in output.splitlines():
        index, uuid = (part.strip() for part in line.split(',', 1))
        if index == gpu_default():
            return uuid
    raise RuntimeError('selected GPU not found')


def selected_gpu_busy(uuid):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    return any(line.split(',', 1)[0].strip() == uuid for line in output.splitlines())


def wait_for_current_job(plan):
    ticks = plan['wait_process_start_ticks']
    while ticks is not None and process_start_ticks(WAIT_PID) == ticks:
        print(f'Waiting for current selected GPU Heun sweep PID {WAIT_PID}', flush=True)
        time.sleep(60)
    if not CURRENT_COMPLETE.is_file():
        raise RuntimeError(f'Current selected GPU sweep ended without {CURRENT_COMPLETE}; benchmark not started')
    if json.loads(CURRENT_COMPLETE.read_text()).get('status') != 'complete':
        raise RuntimeError('Current selected GPU sweep did not complete successfully')
    while not HEUN_EXPORT_COMPLETE.is_file():
        print('Waiting for FlowDCN Heun PNG export before timing 3-RF', flush=True)
        time.sleep(60)
    if json.loads(HEUN_EXPORT_COMPLETE.read_text()).get('status') != 'complete':
        raise RuntimeError('FlowDCN Heun PNG export did not complete successfully')
    uuid = selected_gpu_uuid()
    while selected_gpu_busy(uuid):
        print('Current sweep completed; waiting for selected GPU to become idle', flush=True)
        time.sleep(60)
    print('selected GPU is free; starting 3-RF benchmark', flush=True)


def timed_command(command, *, cwd, env, log):
    start = time.perf_counter()
    with log.open('w') as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def sample_and_score(phase, k):
    run = OUT / phase / f'k_{k:03d}'
    run.parent.mkdir(parents=True, exist_ok=True)
    metrics = run / 'timing.json'
    if metrics.is_file():
        row = json.loads(metrics.read_text())
        if row['k'] != k or row['num_samples'] != IMAGES:
            raise RuntimeError(f'Incompatible timing file: {metrics}')
        return row
    generation_timing = run / 'generation_timing.json'
    if generation_timing.is_file():
        generation_seconds = json.loads(generation_timing.read_text())['seconds']
        if not (run / 'images.npz').is_file():
            raise RuntimeError(f'Generation timing exists without samples in {run}')
    else:
        if run.exists():
            raise RuntimeError(f'Incomplete run at {run}; preserve and inspect it before retrying')
        command = [sys.executable, '-u', str(RF / 'sample_rf_cifar10_euler.py'),
                   '--runtime-root', str(RF / 'ImageGeneration'),
                   '--checkpoint', str(CHECKPOINT), '--k', str(k),
                   '--num-samples', str(IMAGES), '--batch-size', str(BATCH_SIZE),
                   '--seed', str(SEED), '--output-dir', str(run)]
        print(f'Sampling {phase} K=NFE={k}, {IMAGES} images', flush=True)
        start = time.perf_counter()
        # The sampler creates its output directory, so capture its log in the parent.
        with (run.parent / f'k_{k:03d}_sampling.log').open('w') as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        generation_seconds = time.perf_counter() - start
        save(generation_timing, dict(seconds=generation_seconds, k=k,
                                     images=IMAGES, scope='full sampling subprocess'))
    fid_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', TF_NUM_INTRAOP_THREADS='4',
                   TF_NUM_INTEROP_THREADS='2')
    print(f'Evaluating {phase} K=NFE={k} on CPU', flush=True)
    fid_seconds = timed_command(
        [sys.executable, '-u', str(EVALUATOR), str(REFERENCE), str(run / 'images.npz')],
        cwd=EVALUATOR.parent, env=fid_env, log=run / 'fid.log')
    match = re.search(r'\bFID\s*:\s*([0-9.eE+-]+)', (run / 'fid.log').read_text())
    if not match or not math.isfinite(float(match[1])):
        raise RuntimeError(f'Missing or nonfinite FID for {phase} K={k}')
    row = dict(phase=phase, k=k, nfe=k, num_samples=IMAGES,
               generation_seconds=generation_seconds, fid_seconds=fid_seconds,
               total_seconds=generation_seconds + fid_seconds, fid=float(match[1]))
    save(metrics, row)
    return row


def estimate_budget():
    report = OUT / 'estimate.json'
    timing = OUT / 'estimation_timing.json'
    if report.is_file() and timing.is_file():
        result = json.loads(report.read_text())
        return result['K_hat'], json.loads(timing.read_text())['seconds']
    if report.is_file() or timing.is_file():
        raise RuntimeError('Incomplete estimator timing; inspect saved files before retrying')
    command = [sys.executable, '-u', str(RF / 'estimate_euler_budget_fm.py'),
               '--runtime-root', str(RF / 'ImageGeneration'),
               '--checkpoint', str(CHECKPOINT), '--data-root', str(DATA),
               '--num-samples', str(PROBES), '--batch-size', '1',
               '--seed', str(SEED), '--tau', str(TAU), '--epsilon', '1e-8',
               '--output', str(report)]
    print(f'Estimating 3-RF budget from {PROBES} probes', flush=True)
    seconds = timed_command(command, cwd=ROOT, env=os.environ.copy(),
                            log=OUT / 'estimation.log')
    result = json.loads(report.read_text())
    if result['num_samples'] != PROBES or result['tau'] != TAU or result['K_hat'] < 1:
        raise RuntimeError('Invalid estimator output')
    save(timing, dict(seconds=seconds, probes=PROBES, seed=SEED,
                      scope='full estimator subprocess including model and dataset setup'))
    return result['K_hat'], seconds


def write_results(curve, k_hat, estimate_seconds, point):
    fields = ['phase', 'k', 'nfe', 'num_samples', 'generation_seconds',
              'fid_seconds', 'total_seconds', 'fid']
    with (OUT / 'timings.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(curve + [point])
    curve_generation = sum(row['generation_seconds'] for row in curve)
    curve_fid = sum(row['fid_seconds'] for row in curve)
    curve_total = curve_generation + curve_fid
    single_total = estimate_seconds + point['generation_seconds'] + point['fid_seconds']
    summary = dict(curve_budgets=CURVE_K, curve_count=len(CURVE_K),
                   images_per_budget=IMAGES, estimator_probes=PROBES,
                   tau=TAU, k_hat=k_hat, curve_generation_seconds=curve_generation,
                   curve_fid_seconds=curve_fid, curve_total_seconds=curve_total,
                   estimation_seconds=estimate_seconds,
                   single_generation_seconds=point['generation_seconds'],
                   single_fid_seconds=point['fid_seconds'],
                   single_total_seconds=single_total,
                   walltime_ratio=curve_total / single_total,
                   seconds_saved=curve_total - single_total,
                   timing_scope='Serial full subprocess wall time; sampling includes model setup, 10K images, NPZ and grid writing; FID uses CPU ADM evaluator')
    save(OUT / 'summary.json', summary)
    paragraph = (
        f'For 3-RF, constructing the FID–NFE curve at six budgets '
        f'(NFE = {CURVE_K}) required 10K samples per budget and took '
        f'{curve_total / 60:.1f} minutes in total '
        f'({curve_generation / 60:.1f} minutes generating samples and '
        f'{curve_fid / 60:.1f} minutes computing FID). '
        f'In contrast, the estimator used {PROBES} interpolation probes to predict '
        f'NFE = {k_hat}, then generated and evaluated 10K images at that one budget. '
        f'This took {single_total / 60:.1f} minutes '
        f'({estimate_seconds / 60:.1f} estimation, '
        f'{point["generation_seconds"] / 60:.1f} sampling, and '
        f'{point["fid_seconds"] / 60:.1f} FID), '
        f'a {curve_total / single_total:.1f}× reduction in measured wall time.\n')
    (OUT / 'paper_text.txt').write_text(paragraph)
    save(OUT / 'complete.json', dict(status='complete', summary=summary))
    print(paragraph, flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default=gpu_default())
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--wait-for-heun', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    OUT = args.output.resolve()
    for path in (CHECKPOINT, REFERENCE, EVALUATOR):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = OUT / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
    else:
        plan = dict(gpu=args.gpu,
                    wait_for_heun=args.wait_for_heun,
                    model='3-RF', solver='ODE Euler', curve_k=CURVE_K,
                    images_per_budget=IMAGES, batch_size=BATCH_SIZE,
                    estimator_probes=PROBES, tau=TAU, seed=SEED,
                    checkpoint=str(CHECKPOINT), reference=str(REFERENCE),
                    fid_device='cpu')
        save(plan_path, plan)
    if (plan['curve_k'] != CURVE_K or plan['images_per_budget'] != IMAGES
            or plan.get('gpu') != args.gpu or plan.get('wait_for_heun') != args.wait_for_heun):
        raise RuntimeError('Existing benchmark plan has different settings')
    if args.wait_for_heun:
        wait_for_current_job(plan)
    curve = [sample_and_score('curve', k) for k in CURVE_K]
    k_hat, estimate_seconds = estimate_budget()
    point = sample_and_score('predicted', k_hat)
    write_results(curve, k_hat, estimate_seconds, point)


if __name__ == '__main__':
    main()
