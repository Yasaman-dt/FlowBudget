#!/usr/bin/env python3
"""Build a Heun FID summary CSV from completed ADM evaluator logs."""

import argparse
import csv
import re
from pathlib import Path


METRICS = {
    "inception_score": r"^Inception Score:\s*([-+0-9.eE]+)\s*$",
    "fid": r"^FID:\s*([-+0-9.eE]+)\s*$",
    "sfid": r"^sFID:\s*([-+0-9.eE]+)\s*$",
    "precision": r"^Precision:\s*([-+0-9.eE]+)\s*$",
    "recall": r"^Recall:\s*([-+0-9.eE]+)\s*$",
}


def parse_metrics(log_path):
    text = log_path.read_text(errors="replace")
    values = {}
    for name, pattern in METRICS.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if matches:
            values[name] = float(matches[-1])
    if "fid" not in values:
        raise ValueError(f"No completed FID result found in {log_path}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=Path, default=Path(__file__).resolve().parent / Path("results/fid/fid_sweep_heun_10k")
    )
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.expanduser().resolve()
    output = args.output or sweep_dir / "fid_vs_k.csv"
    rows = []
    for run_dir in sweep_dir.glob("n_*"):
        if not run_dir.is_dir():
            continue
        try:
            num_sampling_steps = int(run_dir.name.removeprefix("n_"))
        except ValueError:
            continue
        log_path = run_dir / "adm_evaluator.log"
        sample_npz = run_dir / "images.npz"
        if not log_path.is_file() or not sample_npz.is_file():
            print(f"Skipping incomplete run: {run_dir.name}")
            continue

        try:
            metrics = parse_metrics(log_path)
        except ValueError as error:
            print(f"Skipping incomplete evaluation: {run_dir.name} ({error})")
            continue
        k_heun_updates = num_sampling_steps - 1
        if k_heun_updates < 1:
            raise ValueError(f"{run_dir.name} must contain at least two time points")
        rows.append({
            "k_heun_updates": k_heun_updates,
            "h_abs": 1.0 / k_heun_updates,
            "num_sampling_steps": num_sampling_steps,
            "nfe": 2 * k_heun_updates,
            "fid": metrics["fid"],
            "inception_score": metrics.get("inception_score", ""),
            "sfid": metrics.get("sfid", ""),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "num_samples": args.num_samples,
            "cfg_scale": args.cfg_scale,
            "solver": "ODE/heun2",
            "sample_npz": str(sample_npz),
        })

    if not rows:
        raise RuntimeError(f"No completed Heun FID runs found under {sweep_dir}")
    rows.sort(key=lambda row: row["k_heun_updates"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} with {len(rows)} completed runs.")


if __name__ == "__main__":
    main()
