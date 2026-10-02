#!/usr/bin/env python3
"""Map the FID/NFE figure to inferred tolerance using 2,000-probe coefficients."""
import argparse
import csv
import math
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_matched_layout import save_matched_png
from plot_all_architectures_fid import SERIES, FLOWDCN_SERIES, read_points

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"
OUT = FIGURES / 'sit_rf_fid_vs_tau_2000'
# Reference stars from plot_all_architectures_fid.plot_side_by_side.
REFERENCES = [(148, 7.4), (19, 7.8), (15, 8.24), (140, 6.79),
              (146, 16.6), (58, 16.81), (210, 16.59),
              (150, 15.53), (42, 15.94), (178, 15.78)]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-range', action='store_true',
                        help='Show every curve point using the full axis ranges in separate files.')
    parser.add_argument('--tau-max', type=float, help='Override the upper tau axis limit.')
    parser.add_argument('--no-panel-captions', action='store_true', help='Omit panel captions.')
    parser.add_argument('--legend', action='store_true', help='Show model and solver legends.')
    parser.add_argument('--png-output', type=Path, help='Write only a PNG to this path.')
    parser.add_argument('--output-stem', type=Path, help='Output path without a file extension.')
    parser.add_argument('--match-nfe-y-axis', action='store_true',
                        help='Use the NFE figure logarithmic FID axis, from 6 to 400.')
    args = parser.parse_args()
    out = OUT.with_name(OUT.name + '_all_points') if args.full_range else OUT
    if args.output_stem is not None:
        out = args.output_stem
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.png_output:
        args.png_output.parent.mkdir(parents=True, exist_ok=True)
    with (FIGURES/'sit_rf_khat_vs_tau_2000.csv').open() as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update({'font.family':'Liberation Serif', 'mathtext.fontset':'stix',
                         'font.size':13, 'axes.labelsize':13, 'xtick.labelsize':12,
                         'ytick.labelsize':14, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
    fig.subplots_adjust(left=.1, right=.99, bottom=.23, top=.9, wspace=.18)
    output = []
    for style, reference in zip(SERIES + FLOWDCN_SERIES, REFERENCES):
        label, path, column, color, marker, linestyle, excluded = style
        model, solver_text = label.split(' (')
        solver = solver_text.rstrip(')').split()[-1].lower()
        seeds = [r for r in rows if r['model']==model and r['solver']==solver and float(r['tau'])==.01]
        assert len(seeds)==3 and {int(r['seed']) for r in seeds}=={0,1,2}
        assert all(int(r['N'])==2000 for r in seeds)
        c = statistics.mean(float(r['coefficient']) for r in seeds)
        multiplier, gamma = {'euler':(.5,1), 'heun':(1/12,2), 'em':(1/math.sqrt(3),1)}[solver]
        nfe_factor = 2 if solver=='heun' else 1
        panel = 0 if model.startswith(('SiT','FlowDCN')) else 1
        nfe_grid = {4,8,16,32,64,128,250} if panel==0 else {1,2,4,8,16,32,64,128,250}
        points = [(k, fid) for k, fid in read_points(path, column, excluded, min_k=1)
                  if nfe_factor*k in nfe_grid]
        assert points, label
        def record(k, fid, kind, source):
            tau = multiplier*c/k**gamma
            assert math.isfinite(tau) and tau>0 and math.isfinite(fid) and fid>0
            assert math.isclose((multiplier*c/tau)**(1/gamma), k, rel_tol=1e-12)
            output.append(dict(model=model, solver=solver, estimation_samples_per_seed=2000,
                               estimation_seeds=3, mean_coefficient=c, K=k, NFE=nfe_factor*k,
                               inferred_tau=tau, fid=fid, point_type=kind, fid_source=source))
            return tau, fid
        mapped = sorted(record(k,fid,'curve',str(path.relative_to(ROOT))) for k,fid in points)
        display = f"{'SiT' if model.startswith('SiT') else 'FlowDCN'}, {solver_text.rstrip(')')}" if panel==0 else f'{model}, ODE Euler'
        axes[panel].plot(*zip(*mapped), color=color, marker=marker, linestyle=linestyle,
                         linewidth=2.5, markersize=6.5, label=display)
        print(f'{display}: {len(mapped)} measured curve points; inferred tau {mapped[0][0]:.5g} to {mapped[-1][0]:.5g}')
    tau_upper = max(r['inferred_tau'] for r in output) * 1.05 if args.full_range else 0.050
    if args.tau_max is not None:
        tau_upper = args.tau_max
    visible_fids = [r['fid'] for r in output if 0.0 <= r['inferred_tau'] <= tau_upper]
    fid_lower = min(5, math.floor(min(visible_fids))) if args.full_range else 4
    fid_upper = math.ceil(max(visible_fids) + 1.0)
    if args.match_nfe_y_axis:
        fid_lower, fid_upper = 6, 400
    print(f'Maximum FID in tau range: {max(visible_fids):.6f}; y-axis: {fid_lower} to {fid_upper}')
    for i,ax in enumerate(axes):
        ax.set_box_aspect(.8)
        ax.set_xscale('linear'); ax.set_yscale('linear')
        ax.set_xlim(0.0, tau_upper)
        tau_ticks = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]
        if not args.full_range:
            ax.set_xticks(tau_ticks, [f'{t:.3f}' for t in tau_ticks], rotation=60, ha='right', fontsize=13)
        ax.set_ylim(fid_lower,fid_upper)
        if args.match_nfe_y_axis:
            ax.set_yscale('log')
            ax.set_yticks([10, 100])
        elif not args.full_range:
            ax.set_yticks(list(range(5, fid_upper+1, 5)))
        ax.set_xlabel(r'Tolerance $\tau$', labelpad=0, fontsize=13)
        ax.tick_params(axis='both',which='major',width=1.2,length=4)
        ax.tick_params(axis='y',labelleft=True)
        ax.grid(True,which='both',color='#dfe3e8',linewidth=.4,alpha=.9)
        for spine in ax.spines.values(): spine.set_linewidth(1.2)
        if not args.no_panel_captions:
            ax.text(.5,-.30,['(a) SiT and FlowDCN Models','(b) Rectified flow Models'][i],
                    transform=ax.transAxes,ha='center',va='top',fontsize=16,fontweight='bold')
    axes[0].set_ylabel('FID-10K')
    if args.legend:
        for ax in axes:
            ax.legend(loc='upper right', title='Model, Solver', title_fontsize=12.5,
                      fontsize=10.5, frameon=True, framealpha=0.92,
                      facecolor='white', edgecolor='#d3d3d3', handlelength=1.5,
                      markerscale=0.8, labelspacing=0.25, borderpad=0.3)
    if args.png_output:
        for ax in axes:
            ax.tick_params(axis='x', pad=6)
        save_matched_png(fig, axes, args.png_output, top_margin=12)
        plt.close(fig)
        return
    fig.savefig(out.with_suffix('.png'),dpi=300,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    with out.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    out.with_suffix('.md').write_text('''FID versus inferred tolerance, using the same curve NFE grid, colors and curve markers as sit_rf_fid_vs_nfe.png; reference stars are omitted. FID values are unchanged. No new model evaluations were run.

Each model/solver uses the mean saved coefficient C across estimation seeds 0, 1, 2 with 2,000 probes per seed. Convert NFE to K (K=NFE/2 for Heun; K=NFE otherwise), then infer tau=C/(2K) for Euler, C/(12K^2) for Heun, and C/(sqrt(3)K) for EM. This is the continuous inverse budget relation, i.e. the lower boundary of the tolerance interval selecting integer K under the ceiling rule, not a uniquely measured tolerance. Tau grids differ across models/solvers and follow the available NFE points.

FID is not averaged across estimation seeds; only C is. No FID uncertainty bands are inferred from estimation-seed variability. CSV records every plotted point and source.

FlowDCN Heun's saved finite-difference sensitivity check failed (19% budget spread), so its inferred tau inherits that coefficient uncertainty. Saved SDE coefficients differ from what would place the original stars at tau=0.01; no coefficients are adjusted to match those reference values.
''')

if __name__=='__main__':
    main()
