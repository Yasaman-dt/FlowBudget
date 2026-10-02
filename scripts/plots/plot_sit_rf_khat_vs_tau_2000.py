#!/usr/bin/env python3
"""Plot saved 2,000-probe coefficients at five tolerances, mean +/- seed SD."""
import csv
import json
import math
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
from plot_all_architectures_fid import SERIES, FLOWDCN_SERIES

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"
OUT = FIGURES / 'sit_rf_khat_vs_tau_2000'
TAUS = (.01, .015, .02, .025, .03)


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    sources = {}
    path = ROOT / 'probe_sensitivity_forward/extended_2000_tau001_gpu2/per_seed.csv'
    for row in read_csv(path):
        if int(row['N']) == 2000:
            key = (row['model'], row['solver'])
            sources.setdefault(key, []).append((row, float(row['C_hat']), path))
    path = ROOT / 'probe_sensitivity_forward/sde_2000_tau001_leading_order/per_seed.csv'
    sources[('SiT-XL/2', 'em')] = [(r, float(r['C_EM']), path) for r in read_csv(path)]
    for solver, folder, field in [('euler', 'ode_euler', 'R_theta'),
                                   ('heun', 'ode_heun', 'R_H'), ('em', 'sde_em', None)]:
        path = ROOT / f'FlowDCN/results/estimates/estimation_{folder}/per_seed.csv'
        group = []
        for row in read_csv(path):
            if field:
                coefficient = float(row[field])
                source = path
            else:
                source = path.parent / f"seed_{row['seed']}/estimate.json"
                raw = json.loads(source.read_text())
                assert raw['N'] == 2000
                coefficient = raw['results'][str(raw['selected_h'])]['C_EM']
            group.append((row, coefficient, source))
        sources[('FlowDCN-XL-2M', solver)] = group

    plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix',
                         'font.size': 13, 'axes.labelsize': 13, 'xtick.labelsize': 12,
                         'ytick.labelsize': 14, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
    fig.subplots_adjust(left=.1, right=.99, bottom=.23, top=.9, wspace=.18)
    exported = []
    for style in SERIES + FLOWDCN_SERIES:
        label, _, _, color, marker, linestyle, _ = style
        model, solver_text = label.split(' (')
        solver = solver_text.rstrip(')').split()[-1].lower()
        group = sources[(model, solver)]
        assert len(group) == 3 and {int(r['seed']) for r, _, _ in group} == {0, 1, 2}
        assert all(int(r['N']) == 2000 for r, _, _ in group)
        factor, power = {'euler': (.5, 1), 'heun': (1/12, .5), 'em': (1/math.sqrt(3), 1)}[solver]
        means, sds = [], []
        for tau in TAUS:
            values = [max(1, math.ceil((factor*c/tau)**power)) for _, c, _ in group]
            if tau == .01:
                assert values == [int(r['K_hat']) for r, _, _ in group], (model, solver)
            mean, sd = statistics.mean(values), statistics.stdev(values)
            nfe_factor = 2 if solver == "heun" else 1
            means.append(nfe_factor * mean); sds.append(nfe_factor * sd)
            for (r, coefficient, source), k in zip(group, values):
                exported.append(dict(model=model, solver=solver, N=2000, seed=r['seed'],
                                     tau=tau, coefficient=coefficient, K_hat=k,
                                     mean_K_hat=mean, std_K_hat=sd, NFE_hat=nfe_factor*k,
                                     mean_NFE_hat=nfe_factor*mean, std_NFE_hat=nfe_factor*sd, source=str(source.relative_to(ROOT))))
        ax = axes[0] if model.startswith(('SiT', 'FlowDCN')) else axes[1]
        display = f"{'SiT' if model.startswith('SiT') else 'FlowDCN'}, {solver_text.rstrip(')')}" if ax is axes[0] else model
        ax.fill_between(TAUS, [m-s for m,s in zip(means,sds)], [m+s for m,s in zip(means,sds)], color=color, alpha=.14, linewidth=0)
        ax.plot(TAUS, means, color=color, marker=marker, linestyle=linestyle,
                linewidth=2.5, markersize=6.5, label=display)
        print(f'{display}: tau=0.01 mean NFE_hat={means[0]:.3f}')
    for i, ax in enumerate(axes):
        ax.set_box_aspect(.8)
        ax.set_yscale('log')
        ax.set_ylim(3.5, 380)
        ax.set_yticks([5, 10, 20, 50, 100, 200])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(TAUS, [f'{t:.3f}' for t in TAUS], rotation=45, ha='right')
        ax.set_xlabel(r'$\tau$', labelpad=0)
        ax.tick_params(axis='both', which='major', width=1.2, length=4)
        ax.tick_params(axis='y', labelleft=True)
        ax.grid(True, which='both', color='#dfe3e8', linewidth=.4, alpha=.9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.legend(loc='lower left' if i == 0 else 'upper right', ncol=2, columnspacing=1.0,
                  title='Model, Solver' if i == 0 else 'Model', title_fontsize=10,
                  fontsize=8.5, framealpha=.92, facecolor='white', edgecolor='#d3d3d3',
                  handlelength=1.3, markerscale=.7, labelspacing=.25, borderpad=.3)
        ax.text(.5, -.30, ['(a) SiT and FlowDCN Models', '(b) Rectified flow Models'][i],
                transform=ax.transAxes, ha='center', va='top', fontsize=16, fontweight='bold')
    axes[0].set_ylabel(r'$\widehat{\mathrm{NFE}}$')
    fig.savefig(OUT.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)
    with OUT.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader(); writer.writerows(exported)
    OUT.with_suffix('.md').write_text('''Curves show mean estimated function evaluations (NFE) over seeds 0, 1, 2, with 2,000 estimation samples per seed. Shading shows +/- one sample standard deviation. Tolerances: 0.010, 0.015, 0.020, 0.025, 0.030.

Budgets are recomputed from saved coefficients: Euler ceil(C/(2*tau)); Heun ceil(sqrt(C/(12*tau))); EM ceil(C/(sqrt(3)*tau)). The formulas give sampling steps; plotted NFE equals 2K for Heun and K otherwise. Heun standard deviations are also multiplied by two. All 30 per-seed budgets at tau=0.01 are checked against saved results. CSV includes coefficients and source paths.

The saved leading-order SDE estimates at tau=0.01 average 317 for SiT and 266 for FlowDCN; these differ from the reference stars at 210 and 178 in the FID figure. No coefficients were adjusted to match those stars. FlowDCN Heun reports a failed finite-difference sensitivity check (19% budget spread); shading represents seed variation only, not this systematic sensitivity.
''')


if __name__ == '__main__':
    main()
