"""Reuse seed 0 and estimate seeds 1/2 only; never sample or calculate FID."""
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / 'results' / 'estimates' / 'estimation_ode_euler'
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    import fcntl
    lock = (OUTPUT/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = []
    for seed in [0, 1, 2]:
        folder = OUTPUT/f'seed_{seed}'
        folder.mkdir(exist_ok=True)
        result = folder/'estimate.json'
        if not result.exists():
            if seed == 0:
                raise RuntimeError('Expected completed seed-0 estimate')
            command = [sys.executable, '-u', str(HERE/'run.py'), 'estimate',
                       '--output', str(folder), '--probes', '2000', '--tau', '0.01', '--seed', str(seed)]
            print(f'Starting estimation only: seed {seed}', flush=True)
            with (folder/'estimator.log').open('w') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if not result.exists():
                raise RuntimeError(f'Seed {seed} failed; inspect {folder}/estimator.log')
            if process.returncode and json.loads(result.read_text())['sensitivity_passed']:
                raise RuntimeError(f'Unexpected seed {seed} failure')
        report = json.loads(result.read_text())
        assert report['seed'] == seed and report['N'] == 2000 and report['tau'] == .01
        rows.append(dict(seed=seed, N=2000, tau=.01, K_hat=report['K_hat'],
                         R_theta=report['results']['0.001']['R_theta'],
                         sensitivity_passed=report['sensitivity_passed'],
                         relative_budget_spread=report['relative_budget_spread'], source=str(result)))
        with (OUTPUT/'per_seed.csv').open('w', newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(f"Seed {seed}: K_hat={report['K_hat']}; sensitivity_passed={report['sensitivity_passed']}",flush=True)
    values=[r['K_hat'] for r in rows]
    summary=dict(N=2000,tau=.01,seeds=[0,1,2],num_seeds=3,mean_K_hat=statistics.mean(values),
                 std_K_hat=statistics.stdev(values),all_sensitivity_passed=all(r['sensitivity_passed'] for r in rows),
                 scope='Estimation only; no sampling or FID', per_seed=rows)
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Completed all three seeds',flush=True)


if __name__ == '__main__':
    main()
