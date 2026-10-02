#!/usr/bin/env python3
"""Plot mean predicted budgets with one-sample-SD bands across seeds."""
import csv
import argparse
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from plot_matched_layout import save_matched_png
from matplotlib.ticker import ScalarFormatter
from plot_all_architectures_fid import SERIES

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"
OUTPUT = FIGURES / 'all_architectures_khat_vs_n'
SOURCES = [ROOT / 'probe_sensitivity_forward' / name for name in [
    'extended_2000_tau001_gpu2',
    'additional_3000_20000_tau001_rf_gpu2',
    'additional_3000_20000_tau001_sit_gpu3']]
TAU = 0.01
PAIRS = [('1-RF', 'euler'), ('2-RF', 'euler'), ('3-RF', 'euler'),
         ('RF-UNet', 'euler'), ('SiT-XL/2', 'euler'), ('SiT-XL/2', 'heun')]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-n', type=int, default=25)
    parser.add_argument('--max-n', type=int, default=20000)
    parser.add_argument('--use-nfe', action='store_true', help='Display NFE hat: 2K for Heun, K otherwise')
    parser.add_argument('--no-annotations', action='store_true', help='Omit legends and panel captions')
    args = parser.parse_args()
    if not 25 <= args.min_n <= args.max_n <= 20000:
        parser.error('Require 25 <= min-n <= max-n <= 20000')
    suffix = '' if (args.min_n, args.max_n) == (25, 20000) else f'_{args.min_n}_{args.max_n}'
    output = FIGURES / (OUTPUT.name + suffix)
    summaries, per_seed = [], []
    seen = set()
    for source in SOURCES:
        with (source/'stability.csv').open() as handle:
            for row in csv.DictReader(handle):
                key = (row['model'], row['solver'], int(row['N']))
                if key in seen:
                    raise ValueError(f'Duplicate summary: {key}')
                seen.add(key)
                row['source'] = str(source/'stability.csv')
                summaries.append(row)
        with (source/'per_seed.csv').open() as handle:
            per_seed.extend(csv.DictReader(handle))
    expected = {25,50,100,250,400,500,700,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000}
    if seen != {(m,s,n) for m,s in PAIRS for n in expected}:
        raise ValueError('Missing or unexpected model/count combinations')
    summaries = [r for r in summaries if args.min_n <= int(r['N']) <= args.max_n]
    plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix',
                         'font.size': 32, 'axes.labelsize': 36,
                         'xtick.labelsize': 24, 'ytick.labelsize': 30})
    compact = (args.min_n, args.max_n) == (1000, 20000)
    if compact:
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.23, top=0.9, wspace=0.18)
        for panel in axes:
            panel.set_box_aspect(0.8)
            for spine in panel.spines.values():
                spine.set_linewidth(1.2)
    else:
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.subplots_adjust(left=0.12, bottom=0.19, right=0.98, top=0.98)
        ax.set_box_aspect(0.5)
    rows = []
    counts = set()
    for pair, style in zip(PAIRS, SERIES):
        label, _, _, color, marker, linestyle, _ = style
        panel = axes[0] if compact and pair[0].startswith('SiT') else (axes[1] if compact else ax)
        factor = 2 if args.use_nfe and pair[1] == 'heun' else 1
        group = sorted((r for r in summaries if (r['model'], r['solver']) == pair),
                       key=lambda r: int(r['N']))
        if not group:
            raise ValueError(f'Missing summary for {pair}')
        for r in group:
            n = int(r['N'])
            raw = [x for x in per_seed if (x['model'], x['solver'], int(x['N'])) == (*pair, n)]
            values = [int(x['K_hat']) for x in raw]
            if len(raw) != int(r['num_seeds']) or len({x['seed'] for x in raw}) != len(raw) or len(raw) < 3:
                raise ValueError(f'Insufficient/duplicate seeds: {pair}, N={n}')
            if any(float(x['tau']) != TAU for x in raw):
                raise ValueError('Unexpected tolerance')
            mean, sd = float(r['mean_K_hat']), float(r['std_K_hat'])
            if not (math.isclose(mean, statistics.mean(values)) and math.isclose(sd, statistics.stdev(values))):
                raise ValueError(f'Summary disagrees with per-seed values: {pair}, N={n}')
            if mean-sd <= 0:
                raise ValueError('SD band is not positive; cannot use a logarithmic y axis without changing the display')
            rows.append(dict(model=pair[0], solver=pair[1], N=n, tau=TAU,
                             num_seeds=len(raw), mean_K_hat=mean, std_K_hat=sd,
                             band_low=mean-sd, band_high=mean+sd,
                             CV_K_hat=sd/mean, source=r['source']))
            if args.use_nfe:
                rows[-1].update(mean_NFE_hat=factor*mean, std_NFE_hat=factor*sd,
                                nfe_band_low=factor*(mean-sd), nfe_band_high=factor*(mean+sd))
        ns = [int(r['N']) for r in group]
        means = [factor*float(r['mean_K_hat']) for r in group]
        sds = [factor*float(r['std_K_hat']) for r in group]
        counts.update(ns)
        panel.fill_between(ns, [m-s for m,s in zip(means,sds)],
                        [m+s for m,s in zip(means,sds)], color=color, alpha=0.16, linewidth=0)
        display_label = (f'ODE {pair[1].title()}' if pair[0].startswith('SiT')
                         else pair[0]) if compact else label
        panel.plot(ns, means, label=display_label, color=color, marker=marker,
                linestyle=linestyle, linewidth=2.5 if compact else 5.5,
                markersize=6.5 if compact else 16)
    ticks = [25,50,100,250,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000]
    ticks = [n for n in ticks if args.min_n <= n <= args.max_n]
    if compact:
        ticks = [1000, 2000, 5000, 10000, 20000]
    tick_labels = [f'{n // 1000}K' if compact else f'{n:,}' for n in ticks]
    panels = axes if compact else [ax]
    for panel in panels:
        panel.set_xscale('log')
        panel.set_xticks(ticks, tick_labels, rotation=0 if compact else 65,
                         ha='center' if compact else 'right')
        panel.tick_params(axis='x', labelsize=13 if compact else 16)
        panel.set_yscale('log')
        panel.set_yticks([5, 10, 20, 50, 100, 200, 400])
        if compact:
            panel.set_yticks([10, 20, 50, 100, 200])
            panel.set_ylim(9, 300)
        panel.yaxis.set_major_formatter(ScalarFormatter())
        panel.set_xlabel(r'Estimation sample count $N$')
        panel.set_ylabel(r'Mean predicted $\widehat{\mathrm{NFE}}$' if args.use_nfe
                         else r'Mean predicted steps $\widehat{K}$')
        panel.grid(True, which='both', color='#dfe3e8', linewidth=0.8, alpha=0.9)
        if compact:
            panel.xaxis.label.set_fontsize(13)
            panel.yaxis.label.set_fontsize(13)
            panel.tick_params(axis='y', labelsize=14)
    if compact:
        if not args.no_annotations:
            axes[0].legend(fontsize=9, ncol=1, loc='lower right', title='Solver',
                       title_fontsize=12, framealpha=1, handlelength=1.6)
            axes[1].legend(fontsize=8, ncol=1, loc='center right',
                       bbox_to_anchor=(0.98, 0.56), title='Model',
                       title_fontsize=9, framealpha=1, handlelength=1.2,
                       markerscale=0.7, labelspacing=0.2, borderpad=0.3)
        axes[1].set_ylabel('')
        axes[1].tick_params(axis='y', labelleft=True)
        for panel, title in zip(axes, ['(a) SiT-XL/2 Model',
                                       '(b) Rectified flow Models']):
            if not args.no_annotations:
                panel.text(0.5, -0.30, title, transform=panel.transAxes,
                       ha='center', va='top', fontsize=16, fontweight='bold')
    elif not args.no_annotations:
        ax.legend(fontsize=18, ncol=3, loc='lower right', framealpha=0.95)
    image_path = output.with_suffix('.png')
    if compact:
        axes[0].yaxis.labelpad = -4
        save_matched_png(fig, axes, image_path, top_margin=40)
    else:
        fig.savefig(image_path, dpi=300, bbox_inches='tight', pad_inches=0.04)
    with output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.close(fig)
    print(f'Wrote {output}: means and +/-1 sample SD across three seeds; {len(rows)} points verified.')


if __name__ == '__main__':
    main()
