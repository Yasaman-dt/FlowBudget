#!/usr/bin/env python3
"""Plot empirical FID and the tau-to-K_hat recommendation on shared x-axis."""

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

BLUE = "#1565c0"
RED = "#d62728"
GRID = "#e5e7eb"


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid-csv", type=Path, required=True)
    parser.add_argument("--tau-csv", type=Path, required=True)
    parser.add_argument("--model", default="1-RF", help="Model label in the tau CSV.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-k", type=int, default=4,
        help="Exclude empirical FID points whose update count is below this value.",
    )
    parser.add_argument("--point-labels", choices=["k", "tau", "xy", "none"], default="xy")
    parser.add_argument("--tau-axis-min", type=float, default=0.008)
    args = parser.parse_args()

    fid_rows = [
        row for row in read_csv(args.fid_csv)
        if int(row["k_euler_updates"]) >= args.min_k
    ]
    tau_rows = [row for row in read_csv(args.tau_csv) if row["model"] == args.model]
    if not fid_rows or not tau_rows:
        raise ValueError("No FID rows or no matching tau rows were found.")

    fid_rows.sort(key=lambda row: int(row["k_euler_updates"]))
    tau_rows.sort(key=lambda row: float(row["tau"]))
    ks = [int(row["k_euler_updates"]) for row in fid_rows]
    fids = [float(row["fid"]) for row in fid_rows]
    khats = [int(row["K_hat"]) for row in tau_rows]
    taus = [float(row["tau"]) for row in tau_rows]

    width, height = 960, 560
    left, right, top, bottom = 90, 105, 48, 86
    plot_w, plot_h = width - left - right, height - top - bottom
    log_min = math.log10(min(ks + khats)) - 0.05
    log_max = math.log10(max(ks + khats)) + 0.05
    # Start both vertical scales at zero, giving the axes a broader and more
    # interpretable range than a tightly cropped view.
    fid_min, fid_max = 0.0, max(fids) * 1.07
    tau_min = min(args.tau_axis_min, min(taus))
    tau_max = max(0.032, max(taus))

    sx = lambda x: left + (math.log10(x) - log_min) / (log_max - log_min) * plot_w
    sy_fid = lambda y: top + (fid_max - y) / (fid_max - fid_min) * plot_h
    sy_tau = lambda y: top + (tau_max - y) / (tau_max - tau_min) * plot_h
    bottom_y = height - bottom

    title = args.title or f"{args.model}: FID and Euler-budget estimate"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="480" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom_y}" stroke="{BLUE}" stroke-width="1.5"/>',
        f'<line x1="{width-right}" y1="{top}" x2="{width-right}" y2="{bottom_y}" stroke="{RED}" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{bottom_y}" x2="{width-right}" y2="{bottom_y}" stroke="black" stroke-width="1.5"/>',
    ]
    for i in range(6):
        fid = fid_min + (fid_max - fid_min) * i / 5
        y = sy_fid(fid)
        parts += [
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="{GRID}"/>',
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="{BLUE}">{fid:.1f}</text>',
        ]
    for i in range(5):
        tau = tau_min + (tau_max - tau_min) * i / 4
        y = sy_tau(tau)
        parts.append(f'<text x="{width-right+10}" y="{y+4:.2f}" font-family="sans-serif" font-size="12" fill="{RED}">{tau:.3f}</text>')
    for k in sorted(set(ks + khats)):
        x = sx(k)
        parts += [
            f'<line x1="{x:.2f}" y1="{bottom_y}" x2="{x:.2f}" y2="{bottom_y+5}" stroke="black"/>',
            f'<text x="{x:.2f}" y="{bottom_y+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{k}</text>',
        ]

    fid_line = " ".join(f"{sx(k):.2f},{sy_fid(fid):.2f}" for k, fid in zip(ks, fids))
    tau_line = " ".join(f"{sx(khat):.2f},{sy_tau(tau):.2f}" for khat, tau in zip(khats, taus))
    parts.append(f'<polyline points="{fid_line}" fill="none" stroke="{BLUE}" stroke-width="3"/>')
    parts.append(f'<polyline points="{tau_line}" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="7 5"/>')
    previous_fid_x = None
    for k, fid in zip(ks, fids):
        x, y = sx(k), sy_fid(fid)
        label_x = x + 16 if k == 128 else x
        label_y = y + 15 if y < top + 18 else y - 9
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{BLUE}"/>')
        if args.point_labels == "k":
            parts.append(f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="serif" font-size="11" fill="{BLUE}"><tspan font-style="italic">K</tspan>={k}</text>')
        elif args.point_labels == "xy":
            # Stagger labels for adjacent log-scale points (for example K=60
            # and K=64) so their coordinate strings do not overlap.
            if previous_fid_x is not None and x - previous_fid_x < 45:
                label_y = y + 18
            parts.append(f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{BLUE}">({k}, {fid:.2f})</text>')
        previous_fid_x = x
    for khat, tau in zip(khats, taus):
        x, y = sx(khat), sy_tau(tau)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{RED}"/>')
        label_y = y + 18 if khat == 96 else y - 9
        if args.point_labels == "k":
            parts.append(f'<text x="{x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="serif" font-size="11" fill="{RED}"><tspan font-style="italic">K̂</tspan>={khat}</text>')
        elif args.point_labels == "tau":
            parts.append(f'<text x="{x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="serif" font-size="11" fill="{RED}"><tspan font-style="italic">τ</tspan>={tau:.3f}</text>')
        elif args.point_labels == "xy":
            parts.append(f'<text x="{x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{RED}">({khat}, {tau:.3f})</text>')
    parts += [
        f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="sans-serif" font-size="14">Euler-step budget: empirical K / estimated K-hat (log scale)</text>',
        f'<text x="23" y="{height/2}" text-anchor="middle" transform="rotate(-90 23 {height/2})" font-family="sans-serif" font-size="14" fill="{BLUE}">FID-10K (lower is better)</text>',
        f'<text x="{width-20}" y="{height/2}" text-anchor="middle" transform="rotate(90 {width-20} {height/2})" font-family="serif" font-size="16" font-style="italic" fill="{RED}">τ</text>',
        f'<rect x="112" y="54" width="13" height="3" fill="{BLUE}"/><text x="132" y="59" font-family="sans-serif" font-size="12">Empirical FID-10K</text>',
        f'<line x1="264" y1="55" x2="278" y2="55" stroke="{RED}" stroke-width="3" stroke-dasharray="5 3"/><text x="286" y="59" font-family="serif" font-size="12"><tspan font-style="italic">τ</tspan> vs K̂ = ceil(R_theta/(2<tspan font-style="italic">τ</tspan>))</text>',
        '</svg>',
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg = args.output_dir / "fid_vs_k.svg"
    png = args.output_dir / "fid_vs_k.png"
    svg.write_text("\n".join(parts) + "\n")
    if not shutil.which("rsvg-convert"):
        raise RuntimeError(f"Wrote {svg}, but rsvg-convert is required to make the matching PNG.")
    subprocess.run(["rsvg-convert", "--width=2880", "--height=1680", str(svg), "--output", str(png)], check=True)
    print(f"Wrote {svg}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
