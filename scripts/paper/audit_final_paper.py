#!/usr/bin/env python3
"""Resolve final-paper budgets from saved coefficients and audit result provenance.

No model loading, sampling, or changes to historical experiment files.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def budget(coefficient, solver, tau):
    if not math.isfinite(coefficient) or coefficient < 0 or not math.isfinite(tau) or tau <= 0:
        raise ValueError('Require a finite nonnegative coefficient and positive tolerance')
    if solver == 'euler': value = coefficient / (2 * tau)
    elif solver == 'heun': value = math.sqrt(coefficient / (12 * tau))
    elif solver == 'em': value = coefficient / (math.sqrt(3) * tau)
    else: raise ValueError(f'Unknown solver: {solver}')
    return max(1, math.ceil(value))


def read_fids(path, k_column, images):
    """Reject malformed rows rather than interpreting shifted CSV fields."""
    points, issues = {}, []
    if not path.exists(): return points, [f'Missing FID source: {path.relative_to(ROOT)}']
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for line, row in enumerate(reader, 2):
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('column count differs from header')
                k = int(row[k_column]); fid = float(row['fid'])
                if int(row['num_samples']) != images or not math.isfinite(fid) or fid < 0:
                    raise ValueError('invalid FID or sample count')
                if k in points: raise ValueError(f'duplicate budget {k}')
                points[k] = dict(fid=fid, line=line, sampling_seed=row.get('seed', 'not recorded in CSV'))
            except (KeyError, ValueError) as error:
                issues.append(f'{path.relative_to(ROOT)}:{line}: {error}')
    return points, issues


def audit_pair(pair, config):
    rows, issues, commands = [], [], []
    for source in pair['coefficient_sources']:
        path = ROOT / source['path']
        if not path.exists():
            issues.append(f'Missing coefficient source: {source["path"]}')
            continue
        raw = path.read_bytes(); data = json.loads(raw); value = data
        for key in source['keys']: value = value[key]
        n = data.get('N', data.get('num_samples'))
        if n is None and 'metadata' in data:
            n = len(data['metadata'].get('dataset_indices', []))
        if n != config['probes']:
            raise ValueError(f'{path}: probe count {n}, expected {config["probes"]}')
        seed = data.get('seed', data.get('metadata', {}).get('seed', data.get('probe_pool', {}).get('seed')))
        if seed != source['seed']: raise ValueError(f'{path}: seed mismatch')
        if data.get('sensitivity_passed') is False:
            issues.append(f'Seed {seed}: derivative sensitivity check failed ({data.get("relative_budget_spread"):.2%} budget spread)')
        k = budget(float(value), pair['solver'], pair['tau'])
        rows.append(dict(seed=seed,coefficient=float(value),tau=pair['tau'],K=k,NFE=k*pair['nfe_per_step'],
                         source=source['path'],keys=source['keys'],sha256=hashlib.sha256(raw).hexdigest(),
                         source_tau=data.get('tau',data.get('budget',{}).get('tau'))))
    points, fid_issues = read_fids(ROOT / pair['fid_source'],pair['k_column'],config['images_per_fid'])
    issues.extend(fid_issues)
    for row in rows:
        row['fid_measurement'] = points.get(row['K'])
        if row['fid_measurement'] is None:
            issues.append(f'No valid saved FID at K={row["K"]} for probe seed {row["seed"]}')
    if len(rows) != len(config['probe_seeds']):
        issues.append('Incomplete coefficient seed set')
    mean = statistics.mean(r['NFE'] for r in rows) if rows else None
    sd = statistics.stdev(r['NFE'] for r in rows) if len(rows)>1 else None
    if mean is not None and (abs(mean-pair['reported']['nfe_mean']) > .5 or abs(sd-pair['reported']['nfe_std']) > .5):
        issues.append('Recomputed NFE mean/std do not round to the manuscript values')
    fids = [r['fid_measurement']['fid'] for r in rows if r['fid_measurement'] is not None]
    fid_mean = statistics.mean(fids) if len(fids)==3 else None
    fid_std = statistics.stdev(fids) if len(fids)==3 else None
    if fid_mean is not None and (abs(fid_mean-pair['reported']['fid_mean'])>.0051 or abs(fid_std-pair['reported']['fid_std'])>.0051):
        issues.append('FID lookup mean/std do not round to the manuscript values')
    est=pair['estimation']
    for seed in config['probe_seeds'] if est['per_seed'] else [None]:
        output=f'results/final_paper/estimation/{pair["id"]}'
        if seed is not None: output+=f'/seed_{seed}'
        if not est.get('output_directory'):output+='.json'
        args=['python',est['script'],*est['args']]
        if seed is not None:args+=['--seed',str(seed)]
        args+=['--output',output]
        commands.append(args)
    return dict(id=pair['id'],model=pair['model'],solver=pair['solver'],tau=pair['tau'],
                seeds=rows,NFE_mean=mean,NFE_std=sd,fid_lookup_mean=fid_mean,fid_lookup_std=fid_std,
                fid_source=pair['fid_source'],fid_source_sha256=hashlib.sha256((ROOT/pair['fid_source']).read_bytes()).hexdigest() if (ROOT/pair['fid_source']).exists() else None,reported=pair['reported'],issues=issues,estimation_commands=commands)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/local/final_paper_audit.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/final_paper/audit')
    parser.add_argument('--strict',action='store_true',help='Exit nonzero when results cannot be verified')
    args=parser.parse_args()
    if not args.config.is_file():
        parser.error('Local audit config is unavailable. Supply --config with a private audit config; configs/final_paper.json contains execution settings only.')
    config=json.loads(args.config.read_text())
    if any(not {'reported', 'coefficient_sources', 'fid_source', 'k_column'} <= pair.keys() for pair in config['pairs']):
        parser.error('Audit requires a private config with result mappings; use configs/local/final_paper_audit.json.')
    for p in config['pairs']:
        expected=config['sde_tau'] if p['solver']=='em' else config['ode_tau']
        if p['tau']!=expected:raise ValueError(f'{p["id"]}: inconsistent class tolerance')
    results=[audit_pair(p,config) for p in config['pairs']]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(config=str(args.config.relative_to(ROOT)) if args.config.is_relative_to(ROOT) else str(args.config),
                config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),results=results,
                fid_scope='FID lookup at each predicted budget; repeated K reuses the same saved measurement. This does not certify independent generation seeds.')
    (args.output_dir/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    commands='#!/usr/bin/env bash\nset -euo pipefail\n# Run from the repository root; these commands perform estimation only.\nsource scripts/runtime_env.sh\n'
    def shell_arg(arg):
        token = re.fullmatch(r'\$\{([A-Z_0-9]+)\}([/A-Za-z0-9_.-]*)', arg)
        if token:
            name, suffix = token.groups()
            return '"${' + name + ':?Set ' + name + '}'+suffix+'"'
        return shlex.quote(arg)
    for r in results:
        commands+='\n# '+r['id']+'\n'+'\n'.join(' '.join(shell_arg(arg) for arg in c) for c in r['estimation_commands'])+'\n'
    (args.output_dir/'estimate_commands.sh').write_text(commands)
    lines=['# Final-paper result audit','',
           'Budgets are recomputed from saved coefficients with ODE tau=0.010 and SDE tau=0.015. Sources and SHA-256 hashes are in `provenance.json`. Historical files are unchanged.','',
           '| Pair | K by probe seed 0, 1, 2 | Recomputed NFE mean ± SD | Paper NFE mean ± SD | FID lookup mean ± SD | Issues |',
           '| --- | --- | --- | --- | --- | --- |']
    def fmt(m,s):return 'unavailable' if m is None else f'{m:.4f} ± {s:.4f}'
    for r in results:
        p=r['reported']; lines.append(f'| {r["id"]} | {", ".join(str(x["K"]) for x in r["seeds"])} | {fmt(r["NFE_mean"],r["NFE_std"])} | {p["nfe_mean"]} ± {p["nfe_std"]} | {fmt(r["fid_lookup_mean"],r["fid_lookup_std"])} | {len(r["issues"])} |')
    lines+=['','FID lookup statistics reuse the saved measurement when two probe seeds select the same K. They are not evidence of independent sampling runs.','']
    for r in results:
        lines+=['## '+r['id'],'',f'FID source: `{r["fid_source"]}`.','']
        lines+=[f'- Seed {x["seed"]}: C={x["coefficient"]:.17g}; `{x["source"]}`.' for x in r['seeds']]
        lines+=['']+[f'- {issue}' for issue in r['issues']]+['']
    (args.output_dir/'AUDIT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:16]))
    print(f'\nWrote audit and estimation commands to {args.output_dir}')
    if args.strict and any(r['issues'] for r in results):return 1
    return 0


if __name__=='__main__':sys.exit(main())
