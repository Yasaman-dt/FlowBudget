#!/usr/bin/env python3
"""Reuse one R_theta estimate to compute Euler K-hat over several taus."""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", type=Path, default=root / "euler_budget_estimate_500.json")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.01, 0.015, 0.02, 0.025, 0.03])
    parser.add_argument("--output", type=Path, default=root / "euler_budget_tau_sweep.csv")
    args = parser.parse_args()
    if any(tau <= 0 for tau in args.taus):
        raise ValueError("Every tau must be positive")
    estimate = json.loads(args.estimate.read_text())
    r_theta = float(estimate["R_theta"])
    rows = [
        {
            "model": "Modern-RF-UNet",
            "source_estimate": str(args.estimate.resolve()),
            "R_theta": r_theta,
            "tau": tau,
            "K_hat": math.ceil(r_theta / (2 * tau)),
            "formula": "ceil(R_theta / (2 * tau))",
        }
        for tau in args.taus
    ]
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
