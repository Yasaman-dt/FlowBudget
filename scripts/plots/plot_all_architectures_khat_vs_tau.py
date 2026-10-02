#!/usr/bin/env python3
"""Plot saved sampling-step estimates against tau for all six model/solver series."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from plot_all_architectures_fid import SERIES


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"
OUTPUT = FIGURES / "all_architectures_khat_vs_tau"
SOURCES = [
    ("results/budget_estimates/euler_budget_tau_sweep_2tau.csv", "1-RF", "K_hat"),
    ("results/budget_estimates/euler_budget_tau_sweep_2tau.csv", "2-RF", "K_hat"),
    ("results/budget_estimates/euler_budget_tau_sweep_2tau.csv", "3-RF", "K_hat"),
    ("rectified_flow_modern/euler_budget_tau_sweep.csv", "Modern-RF-UNet", "K_hat"),
    ("results/budget_estimates/euler_budget_tau_sweep_2tau.csv", "SiT-ImageNet", "K_hat"),
    ("results/budget_estimates/heun_budget_tau_sweep.csv", "SiT-ImageNet-Heun", "K_hat_heun_updates"),
    ("SiT_Imagenet/em_budget_estimates/estimate_500.csv", None, "K_hat"),
]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Liberation Serif",
        "mathtext.fontset": "stix",
        "font.size": 32,
        "axes.labelsize": 36,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 27,
    })
    fig, ax = plt.subplots(figsize=(16, 8))
    # Keep the horizontal span while making the vertical span 15% shorter.
    fig.subplots_adjust(left=0.12, bottom=0.19, right=0.98, top=0.98)
    ax.set_box_aspect(0.5)
    taus = set()
    for (filename, model, column), series in zip(SOURCES, SERIES):
        label, _, _, color, marker, linestyle, _ = series
        source = ROOT / filename
        if model is None and not source.is_file():
            print(f"Skipping optional SiT EM series: missing {source}")
            continue
        with source.open(newline="") as handle:
            points = sorted(
                (float(row["tau"]), int(row[column]))
                for row in csv.DictReader(handle)
                if (model is None and row.get("status") == "found")
                or (model is not None and row.get("model") == model)
            )
        if not points or any(t <= 0 or k <= 0 for t, k in points):
            raise ValueError(f"Missing or invalid estimates for {model}")
        taus.update(t for t, _ in points)
        ts, ks = zip(*points)
        ax.plot(ts, ks, label=label, color=color, marker=marker,
                linestyle=linestyle, linewidth=5.5, markersize=22)

    ax.set_xticks(sorted(taus))
    ax.set_xticklabels([f"{tau:.3f}" for tau in sorted(taus)])
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 50, 100, 200, 400])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"Tolerance $\tau$")
    ax.set_ylabel(r"Estimated sampling steps $\widehat{K}$")
    ax.grid(True, which="both", color="#dfe3e8", linewidth=0.8, alpha=0.9)
    for extension in ("png",):
        path = OUTPUT.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
