"""Recompute manuscript EM budgets from preserved 2000-probe SiT pools."""
import csv
import hashlib
import json
from pathlib import Path
from estimate_em_budget import estimate_budget
import statistics

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'probe_sensitivity_forward/sde_2000_tau001_gpu3_20260915'
OUTPUT=ROOT/'probe_sensitivity_forward/sde_2000_tau001_leading_order'

def main():
    OUTPUT.mkdir(exist_ok=True)
    rows=[]
    for seed in range(3):
        path=SOURCE/f'seed_{seed}/pool.json'
        raw=path.read_bytes(); data=json.loads(raw)
        assert len(data['probes'])==2000 and data['metadata']['seed']==seed
        budget=estimate_budget(data['probes'],.01)
        row=dict(model='SiT-XL/2',solver='em',N=2000,seed=seed,**budget,
                 source_pool=str(path),source_sha256=hashlib.sha256(raw).hexdigest())
        rows.append(row)
        folder=OUTPUT/f'seed_{seed}'; folder.mkdir(exist_ok=True)
        (folder/'estimate.json').write_text(json.dumps(dict(estimator='SiT-Linear-EM-leading-order-v2',
            metadata=data['metadata'],budget=budget,source_pool=str(path),source_sha256=row['source_sha256'],
            note='Reuses original probe law and random-direction q2 estimates; no new model evaluations.'),indent=2)+'\n')
    with (OUTPUT/'per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=dict(estimator='SiT-Linear-EM-leading-order-v2',N=2000,tau=.01,
        mean_K_hat=statistics.mean(r['K_hat'] for r in rows),std_K_hat=statistics.stdev(r['K_hat'] for r in rows),per_seed=rows,
        caveats=['Original probe-time distribution retained: [0,0.99).',
                 'Eight-direction q2 estimate; square root introduces Monte Carlo bias.',
                 'Existing sampler uses two network calls per step; mathematical EM needs one drift evaluation.'])
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print([(r['seed'],r['K_hat']) for r in rows]); print('Mean',summary['mean_K_hat'],'SD',summary['std_K_hat'])

if __name__=='__main__':main()
