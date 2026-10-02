"""Print a compact comparison of multiple FM Euler-budget estimate JSON files."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("estimates", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [(path, json.loads(path.read_text())) for path in args.estimates]
    print(f"{'file':<28} {'N':>6} {'R_theta':>12} {'K_hat':>7} {'median':>12} {'std':>12}")
    for path, result in rows:
        print(
            f"{path.name:<28} {result['num_samples']:>6} {result['R_theta']:>12.6f} "
            f"{result['K_hat']:>7} {result['r_median']:>12.6f} {result['r_std']:>12.6f}"
        )
    taus = {result["tau"] for _, result in rows}
    if len(taus) != 1:
        print("Warning: tau differs between files; K_hat values are not directly comparable.")


if __name__ == "__main__":
    main()
