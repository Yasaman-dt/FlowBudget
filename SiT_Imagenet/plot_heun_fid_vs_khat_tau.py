"""Plot empirical Heun FID and the tau-based Heun budget estimate."""

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
    parser.add_argument("--fid-csv", type=Path, default=Path(__file__).resolve().parent / Path("results/fid/fid_sweep_heun_10k/fid_vs_k.csv"))
    parser.add_argument("--tau-csv", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "budget_estimates" / "heun_budget_tau_sweep.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / Path("results/fid/fid_sweep_heun_10k"))
    parser.add_argument(
        "--min-k", type=int, default=1,
        help="Exclude empirical FID points whose Heun update count is below this value.",
    )
    parser.add_argument(
        "--point-labels", choices=["tau", "xy"], default="tau",
        help="Label estimate points by tau or label every point as a numeric (x, y) pair.",
    )
    args = parser.parse_args()

    fid_rows = read_csv(args.fid_csv)
    tau_rows = read_csv(args.tau_csv)
    if not fid_rows or not tau_rows:
        raise ValueError("Both input CSV files must contain data rows")

    empirical = [
        (int(r["k_heun_updates"]), float(r["fid"]))
        for r in fid_rows
        if int(r["k_heun_updates"]) >= args.min_k
    ]
    estimates = [(int(r["K_hat_heun_updates"]), float(r["tau"])) for r in tau_rows]
    empirical.sort()
    estimates.sort(key=lambda item: item[1])

    width, height = 960, 560
    left, right, top, bottom = 90, 105, 48, 86
    plot_right, plot_bottom = width - right, height - bottom
    all_k = [k for k, _ in empirical + estimates]
    log_min, log_max = math.log10(min(all_k)), math.log10(max(all_k))
    x_pad = max(0.02, (log_max - log_min) * 0.02)
    log_min, log_max = log_min - x_pad, log_max + x_pad
    fid_min, fid_max = 0.0, max(v for _, v in empirical) * 1.07
    tau_min = min(0.008, min(v for _, v in estimates))
    tau_max = max(0.032, max(v for _, v in estimates))
    sx = lambda x: left + (math.log10(x) - log_min) / (log_max - log_min) * (plot_right - left)
    sy_fid = lambda y: top + (fid_max - y) / (fid_max - fid_min) * (plot_bottom - top)
    sy_tau = lambda y: top + (tau_max - y) / (tau_max - tau_min) * (plot_bottom - top)

    fid_points = " ".join(f"{sx(k):.2f},{sy_fid(v):.2f}" for k, v in empirical)
    tau_points = " ".join(f"{sx(k):.2f},{sy_tau(v):.2f}" for k, v in estimates)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">Unguided SiT ImageNet: FID and Heun-budget estimate</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{plot_bottom}" stroke="{BLUE}" stroke-width="1.5"/>',
        f'<line x1="{plot_right}" y1="{top}" x2="{plot_right}" y2="{plot_bottom}" stroke="{RED}" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="black" stroke-width="1.5"/>',
    ]

    for i in range(6):
        fid = fid_min + (fid_max - fid_min) * i / 5
        y = sy_fid(fid)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="{BLUE}">{fid:.1f}</text>')
    for i in range(5):
        tau = tau_min + (tau_max - tau_min) * i / 4
        y = sy_tau(tau)
        parts.append(f'<text x="{plot_right+10}" y="{y+4:.2f}" font-family="sans-serif" font-size="12" fill="{RED}">{tau:.3f}</text>')

    for k in sorted(set(all_k)):
        x = sx(k)
        parts.append(f'<line x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" y2="{plot_bottom+5}" stroke="black"/>')
        parts.append(f'<text x="{x:.2f}" y="{plot_bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{k}</text>')

    parts.append(f'<polyline points="{fid_points}" fill="none" stroke="{BLUE}" stroke-width="3"/>')
    parts.append(f'<polyline points="{tau_points}" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="7 5"/>')
    for k, fid in empirical:
        parts.append(f'<circle cx="{sx(k):.2f}" cy="{sy_fid(fid):.2f}" r="4" fill="{BLUE}"/>')
        if args.point_labels == "xy":
            label_x = sx(k)
            label_y = sy_fid(fid) - 9
            anchor = "middle"
            if k == 29:
                label_x += 4
                label_y -= 4
                anchor = "end"
            elif k == 32:
                label_x += 6
                label_y = sy_fid(fid) + 18
                anchor = "start"
            parts.append(f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" font-family="sans-serif" font-size="11" fill="{BLUE}">({k}, {fid:.2f})</text>')
    for k, tau in estimates:
        x, y = sx(k), sy_tau(tau)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{RED}"/>')
        if args.point_labels == "xy":
            parts.append(f'<text x="{x:.2f}" y="{y-9:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{RED}">({k}, {tau:.3f})</text>')
        else:
            parts.append(f'<text x="{x:.2f}" y="{y-9:.2f}" text-anchor="middle" font-family="serif" font-size="11" fill="{RED}"><tspan font-style="italic">τ</tspan>={tau:.3f}</text>')

    parts.extend([
        '<text x="480" y="536" text-anchor="middle" font-family="sans-serif" font-size="14">Heun-step budget: empirical K / estimated K-hat (log scale)</text>',
        f'<text x="23" y="280" text-anchor="middle" transform="rotate(-90 23 280)" font-family="sans-serif" font-size="14" fill="{BLUE}">FID-10K (lower is better)</text>',
        f'<text x="940" y="280" text-anchor="middle" transform="rotate(90 940 280)" font-family="serif" font-size="16" font-style="italic" fill="{RED}">τ</text>',
        f'<rect x="112" y="54" width="13" height="3" fill="{BLUE}"/><text x="132" y="59" font-family="sans-serif" font-size="12">Empirical FID-10K</text>',
        f'<line x1="264" y1="55" x2="278" y2="55" stroke="{RED}" stroke-width="3" stroke-dasharray="5 3"/><text x="286" y="59" font-family="serif" font-size="12"><tspan font-style="italic">τ</tspan> vs K̂ = ceil(sqrt(R_theta/(12<tspan font-style="italic">τ</tspan>)))</text>',
        '</svg>',
    ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "fid_vs_k.svg"
    png_path = args.output_dir / "fid_vs_k.png"
    svg_path.write_text("\n".join(parts) + "\n")
    renderer = shutil.which("rsvg-convert")
    if not renderer:
        raise RuntimeError("rsvg-convert is required to render the PNG")
    subprocess.run([renderer, "--width=2880", "--height=1680", str(svg_path), "--output", str(png_path)], check=True)
    print(f"Wrote {svg_path} and {png_path}")


if __name__ == "__main__":
    main()
