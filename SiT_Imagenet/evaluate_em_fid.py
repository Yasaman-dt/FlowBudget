#!/usr/bin/env python3
"""Evaluate completed unguided EM batches; resume logs and update CSV after each K.

Metadata assumes the sigma=1, last-step=None sampling command used for this
historical sweep. ``nfe`` reports the nonredundant velocity evaluations (K).
Each run's ``metrics.json`` also preserves ``nfe_executed``: the two model
calls per update actually made by the historical, unfused implementation.
The current implementation needs only one call per update.
"""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import resource_path, gpu_default
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from rebuild_heun_fid_csv import parse_metrics

ROOT = Path(__file__).resolve().parent
FIELDS = ['k_em_updates', 'h_abs', 'num_sampling_steps', 'nfe', 'fid',
          'inception_score', 'sfid', 'precision', 'recall', 'num_samples',
          'cfg_scale', 'solver', 'diffusion_form', 'diffusion_norm', 'last_step',
          'reference_npz', 'sample_npz']


def sample_count(path):
    import numpy as np
    with zipfile.ZipFile(path) as archive, archive.open('arr_0.npy') as data:
        version = np.lib.format.read_magic(data)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(data)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(data)
        else:
            raise ValueError(f'Unsupported NPY header: {version}')
    return shape[0]


def write_csv(folder, rows):
    target = folder / 'fid_vs_k.csv'
    temp = target.with_suffix('.csv.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({name: row[name] for name in FIELDS}
                         for row in sorted(rows, key=lambda row: row['k_em_updates']))
    temp.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep-dir', type=Path, default=ROOT / 'results/fid/fid_sweep_em_10k')
    parser.add_argument('--reference-npz', type=Path, default=resource_path(ROOT / 'references/imagenet_train_10k_center_crop_256.npz'))
    parser.add_argument('--evaluator', type=Path, default=resource_path(ROOT.parents[1] / 'guided-diffusion/evaluations/evaluator.py'))
    parser.add_argument('--fid-python', default=sys.executable)
    parser.add_argument('--rebuild-only', action='store_true')
    args = parser.parse_args()
    folder = args.sweep_dir.resolve()
    reference = args.reference_npz.resolve()
    evaluator = args.evaluator.resolve()
    if not reference.is_file() or not evaluator.is_file():
        raise FileNotFoundError('Reference batch or ADM evaluator is missing')
    runs = sorted((p for p in folder.glob('k_*') if p.is_dir() and p.name[2:].isdigit()),
                  key=lambda p: int(p.name[2:]))
    rows = []
    pending = []
    def record(run, metrics):
        k = int(run.name[2:])
        samples = run / 'images.npz'
        row = dict(k_em_updates=k, h_abs=1 / k, num_sampling_steps=k + 1,
                   nfe=k, nfe_executed=2 * k, **{key: metrics[key] for key in
                   ['fid', 'inception_score', 'sfid', 'precision', 'recall']},
                   num_samples=sample_count(samples), cfg_scale=1.0,
                   solver='SDE/Euler-Maruyama', diffusion_form='sigma',
                   diffusion_norm=1.0, last_step='None',
                   reference_npz=str(reference), sample_npz=str(samples))
        rows.append(row)
        (run / 'metrics.json').write_text(json.dumps(row, indent=2) + '\n')
        write_csv(folder, rows)
        print(f"K={k}: FID={row['fid']}; CSV updated", flush=True)
    for run in runs:
        if not (run / 'images.npz').is_file():
            print(f'Skipping incomplete samples: {run.name}', flush=True)
            continue
        log = run / 'adm_evaluator.log'
        try:
            metrics = parse_metrics(log)
            if not all(key in metrics for key in ['inception_score', 'sfid', 'precision', 'recall']):
                raise ValueError('Incomplete metrics')
        except (FileNotFoundError, ValueError):
            pending.append(run)
        else:
            record(run, metrics)
    write_csv(folder, rows)
    if args.rebuild_only:
        return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu_default(), PYTHONUNBUFFERED='1',
               TF_NUM_INTRAOP_THREADS='8', TF_NUM_INTEROP_THREADS='2', OMP_NUM_THREADS='8')
    for run in pending:
        print(f'Evaluating {run.name} on selected GPU...', flush=True)
        log = run / 'adm_evaluator.log'
        with log.open('w') as handle:
            subprocess.run([args.fid_python, '-u', str(evaluator), str(reference),
                            str(run / 'images.npz')], cwd=evaluator.parent,
                           env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        record(run, parse_metrics(log))
    print(f'Finished: {len(rows)} evaluated batches. Rerun to include newly completed samples.', flush=True)


if __name__ == '__main__':
    main()
