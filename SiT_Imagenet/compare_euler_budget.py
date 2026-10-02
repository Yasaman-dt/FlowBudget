"""Compare an FM Euler K-hat JSON file to the completed empirical FID sweep."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--fid-csv", type=Path, required=True)
    args = parser.parse_args()
    estimate = json.loads(args.estimate.read_text())
    with args.fid_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("FID CSV has no rows")
    best = min(rows, key=lambda row: float(row["fid"]))
    print(f"FM estimate: K_hat={estimate['K_hat']} (tau={estimate['tau']}, cfg_scale={estimate.get('cfg_scale')})")
    print(f"Empirical minimum: K={best['k_euler_updates']}, FID={float(best['fid']):.4f}")
    print("They are comparable only when both use ODE/euler, cfg_scale=1.0, the same VAE, checkpoint, and FID reference.")


if __name__ == "__main__":
    main()
