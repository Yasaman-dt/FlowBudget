#!/usr/bin/env python3
"""Plot all completed empirical FID sweeps without budget/tau estimates."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "figures"
OUTPUT = FIGURES / "all_architectures_fid_vs_k"

# Use familiar colors without repeating a color between the two panels.
# Reference stars take their color from the corresponding curve.
COLORS = {
    "SiT Euler": "tab:blue",
    "SiT Heun": "tab:orange",
    "SiT EM": "tab:green",
    "FlowDCN Euler": "tab:red",
    "FlowDCN Heun": "tab:purple",
    "FlowDCN EM": "magenta",
    "1-RF": "darkviolet",
    "2-RF": "tab:brown",
    "3-RF": "tab:gray",
    "RF-UNet": "tab:cyan",
}

HIDDEN_MARKERS = {
    "RF-UNet (ODE Euler)": {147},
    "SiT-XL/2 (ODE Euler)": {149},
    "SiT-XL/2 (ODE Heun)": {29},
    "SiT-XL/2 (SDE EM)": {317},
}

SERIES = [
    (
        "1-RF (ODE Euler)",
        ROOT / "rectified_flow_cifar10/results/fid/fid_sweep_1rf_10k/fid_vs_k.csv",
        "k_euler_updates", COLORS["1-RF"], "o", "-", {70},
    ),
    (
        "2-RF (ODE Euler)",
        ROOT / "rectified_flow_cifar10/results/fid/fid_sweep_2rf_10k/fid_vs_k.csv",
        "k_euler_updates", COLORS["2-RF"], "s", "-", set(),
    ),
    (
        "3-RF (ODE Euler)",
        ROOT / "rectified_flow_cifar10/results/fid/fid_sweep_3rf_10k/fid_vs_k.csv",
        "k_euler_updates", COLORS["3-RF"], "^", "-", set(),
    ),
    (
        "RF-UNet (ODE Euler)",
        ROOT / "rectified_flow_modern/results/fid/fid_sweep_unet_euler_cifar10_10k/fid_vs_k.csv",
        "k_euler_updates", COLORS["RF-UNet"], "D", "-", {96},
    ),
    (
        "SiT-XL/2 (ODE Euler)",
        ROOT / "SiT_Imagenet/results/fid/fid_sweep_unguided_10k/fid_vs_k.csv",
        "k_euler_updates", COLORS["SiT Euler"], "P", "--", set(),
    ),
    (
        "SiT-XL/2 (ODE Heun)",
        ROOT / "SiT_Imagenet/results/fid/fid_sweep_heun_10k/fid_vs_k.csv",
        "k_heun_updates", COLORS["SiT Heun"], "X", "--", set(),
    ),
    (
        "SiT-XL/2 (SDE EM)",
        ROOT / "SiT_Imagenet/results/fid/fid_sweep_em_10k/fid_vs_k.csv",
        "k_em_updates", COLORS["SiT EM"], "v", "--", set(),
    ),
]

FLOWDCN_SERIES = [
    (
        "FlowDCN-XL-2M (ODE Euler)",
        ROOT / "FlowDCN/results/fid/fid_ode_euler/euler/fid_vs_k.csv",
        "k", COLORS["FlowDCN Euler"], "h", ":", set(),
    ),
    (
        "FlowDCN-XL-2M (ODE Heun)",
        ROOT / "FlowDCN/results/fid/fid_ode_heun/fid_vs_k.csv",
        "k", COLORS["FlowDCN Heun"], "d", ":", set(),
    ),
    (
        "FlowDCN-XL-2M (SDE EM)",
        ROOT / "FlowDCN/results/fid/fid_sde_em/fid_vs_k.csv",
        "k", COLORS["FlowDCN EM"], "^", ":", set(),
    ),
]


def read_points(path, k_column, excluded, allowed=None, min_k=4):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = sorted(
        (int(row[k_column]), float(row["fid"]))
        for row in rows
        if (int(row[k_column]) in allowed if allowed is not None else int(row[k_column]) >= min_k)
        and int(row[k_column]) not in excluded
    )
    if not points:
        raise ValueError(f"No usable FID rows in {path}")
    return points


def plot_series(series, output, keep_k=None, use_nfe=False, compact=False, ax=None):
    embedded = ax is not None
    plt.rcParams.update({
        # Times New Roman is not installed on this machine; Liberation Serif
        # is its metric-compatible Times-style substitute.
        "font.family": "Liberation Serif",
        "mathtext.fontset": "stix",
        "font.size": 32,
        "axes.titlesize": 10,
        "axes.labelsize": 36,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 24,
    })
    if compact:
        plt.rcParams.update({"font.size": 13, "axes.labelsize": 16,
                             "xtick.labelsize": 12, "ytick.labelsize": 13,
                             "legend.fontsize": 11})
        if not embedded:
            fig, ax = plt.subplots(figsize=(3.5, 2.9))
            fig.subplots_adjust(left=0.17, bottom=0.18, right=0.98, top=0.98)
        ax.set_box_aspect(0.8)
    else:
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.subplots_adjust(left=0.12, bottom=0.19, right=0.98, top=0.98)
        ax.set_box_aspect(0.5)

    all_k = set()
    for label, path, k_column, color, marker, linestyle, excluded in series:
        allowed = {
            "1-RF (ODE Euler)": {4,8,16,32,64,128,149,250},
            "3-RF (ODE Euler)": {4,8,16,32,64,128,250},
            "2-RF (ODE Euler)": {4,8,16,20,32,64,128,250},
        }.get(label)
        min_k = 4 if label.startswith("SiT") else 1
        if use_nfe and k_column == "k_heun_updates":
            min_k = 2
        if allowed is not None:
            allowed = allowed | {1, 2}
        points = read_points(path, k_column, excluded, allowed, min_k)
        if allowed is not None and {k for k, _ in points} != allowed:
            raise ValueError(f"Missing requested budgets for {label}")
        if keep_k is not None:
            # The compact panel's requested budgets are NFEs, not Heun steps.
            factor = 2 if use_nfe and "Heun" in label else 1
            series_keep_k = {nfe // factor for nfe in keep_k if nfe % factor == 0}
            points = [(k, fid) for k, fid in points if k in series_keep_k]
        ks, fids = zip(*points)
        # Plotting convention: Euler and SDE EM use K NFEs, Heun uses 2K.
        factor = 2 if use_nfe and "Heun" in label else 1
        xs = [factor * k for k in ks]
        all_k.update(xs)
        if compact:
            model = ("SiT" if label.startswith("SiT") else
                     "FlowDCN" if label.startswith("FlowDCN") else
                     label.split(" (", 1)[0])
            solver = label.split(" (", 1)[1].rstrip(")")
            display_label = f"{model}, {solver}"
        else:
            display_label = label
        ax.plot(
            xs, fids, label=display_label,
            color=color, marker=marker,
            linestyle=linestyle, linewidth=2.5 if compact else 5.5,
            markersize=6.5 if compact else 22,
            markevery=[i for i, k in enumerate(ks) if k not in HIDDEN_MARKERS.get(label, set())],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if compact:
        # Shared FID scale for side-by-side panels, including all RF values.
        # Shared tight range: all displayed FID values remain inside with margin.
        ax.set_ylim(6, 400)
        ax.set_yticks([10, 100])
    display_ticks = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 250, 500]
                     if min(all_k) <= k <= max(all_k)]
    ax.set_xticks(display_ticks)
    ax.set_xticklabels([str(k) for k in display_ticks])
    if compact:
        ax.tick_params(axis="both", which="major", width=1.2, length=4)
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    ax.set_xlabel(("NFE" if compact else "Number of function evaluations (NFE)") if use_nfe
                  else r"Sampling-step budget $K$")
    ax.set_ylabel("FID-10K")
    if compact and all(not s[0].startswith("SiT") for s in series):
        # Hide unused labels so tight export crops the blank left margin
        # without changing the physical size of the plotting axes.
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", labelleft=False)
    ax.grid(True, which="both", color="#dfe3e8", linewidth=0.4, alpha=0.9)
    if embedded:
        return
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", title="Model (Solver)",
              title_fontsize=28, fontsize=24, frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="#d3d3d3", ncol=1,
              handlelength=1.6, markerscale=0.75, labelspacing=0.35, borderpad=0.4)
    fig.savefig(output.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.png')}")


def plot_side_by_side():
    plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.23, top=0.9, wspace=0.18)
    panel_a = [s for s in SERIES if s[0].startswith("SiT")]
    panel_a += [s for s in FLOWDCN_SERIES if s[1].exists()]
    for ax, series, title, ks in [
        (axes[0], panel_a, "SiT and FlowDCN Models", {4, 8, 16, 32, 64, 128, 250}),
        (axes[1], [s for s in SERIES if not s[0].startswith("SiT")],
         "Rectified flow Models", {1, 2, 4, 8, 16, 32, 64, 128, 250}),
    ]:
        plot_series(series, None, keep_k=ks, use_nfe=True, compact=True, ax=ax)
        if ax is axes[0]:
            # Leave visual room around the endpoint ticks, as in panel (b).
            ax.set_xlim(3.5, 285)
        ax.xaxis.labelpad = 0
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", which="both", labelleft=True)

    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Requested reference points; keep the empirical curves unchanged.
    for nfe, fid, color in [(146, 16.6, COLORS["SiT Euler"]),
                            (58, 16.81, COLORS["SiT Heun"]),
                            (210, 16.59, COLORS["SiT EM"]),
                            (150, 15.53, COLORS["FlowDCN Euler"]),
                            (42, 15.94, COLORS["FlowDCN Heun"]),
                            (178, 15.78, COLORS["FlowDCN EM"])]:
        axes[0].scatter([nfe], [fid], marker="*", s=260, facecolor=color,
                        edgecolor="black", linewidth=0.8, zorder=10)
    for nfe, fid, color in [(19, 7.8, COLORS["2-RF"]),
                            (15, 8.24, COLORS["3-RF"]),
                            (148, 7.4, COLORS["1-RF"]),
                            (140, 6.79, COLORS["RF-UNet"])]:
        axes[1].scatter([nfe], [fid], marker="*", s=260, facecolor=color,
                        edgecolor="black", linewidth=0.8, zorder=10)
    output = FIGURES / "sit_rf_fid_vs_nfe"
    fig.savefig(output.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_series(SERIES, OUTPUT)
    plot_side_by_side()


if __name__ == "__main__":
    main()
