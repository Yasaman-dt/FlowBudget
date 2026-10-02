#!/usr/bin/env python3
"""Recompute Euler or Heun budgets from saved coefficients without model runs."""

import argparse
import csv
import json
import math
from pathlib import Path


FORMULA = "ceil(R_theta / (2 * tau))"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--estimates", type=Path, nargs="+", required=True,
                        help="JSON estimates containing R_theta (Euler) or R_theta_heun (Heun).")
    parser.add_argument("--names", nargs="*", default=None,
                        help="Optional labels for --estimates, in the same order.")
    parser.add_argument("--taus", type=float, nargs="+", required=True,
                        help="Positive tau values, e.g. 0.010 0.015 0.020.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path. A CSV with the same stem is also written.")
    return parser.parse_args()


def main():
    args = parse_args()
    if any(not math.isfinite(tau) or tau <= 0 for tau in args.taus):
        raise ValueError("All --taus values must be positive.")
    if args.names is not None and len(args.names) not in (0, len(args.estimates)):
        raise ValueError("Provide either no --names or exactly one name per estimate.")

    field = "R_theta_heun" if args.solver == "heun" else "R_theta"
    formula = "ceil(sqrt(R_theta_heun / (12 * tau)))" if args.solver == "heun" else FORMULA
    names = args.names or [path.stem for path in args.estimates]
    rows = []
    for name, path in zip(names, args.estimates):
        path = path.expanduser().resolve()
        with path.open() as handle:
            estimate = json.load(handle)
        if field not in estimate:
            raise KeyError(f"{path} does not contain {field}.")
        coefficient = float(estimate[field])
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"{path}: coefficient must be finite and nonnegative")
        for tau in args.taus:
            value = math.sqrt(coefficient / (12 * tau)) if args.solver == "heun" else coefficient / (2 * tau)
            k = max(1, math.ceil(value))
            row = dict(model=name, source_estimate=str(path))
            row[field] = coefficient
            row['tau'] = tau
            if args.solver == 'heun':
                row.update(K_hat_heun_updates=k, integrator_time_points=k+1, NFE_hat=2*k)
            else:
                row['K_hat'] = k
            row['formula'] = formula
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump({"formula": formula, "rows": rows}, handle, indent=2)
        handle.write("\n")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
