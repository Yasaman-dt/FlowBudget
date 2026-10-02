"""Record an already-computed ADM FID and rebuild the sweep CSV/SVG plot."""
import argparse
import json
from pathlib import Path

from fid_sweep_reporting import rebuild_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, help="Number of Euler updates.")
    parser.add_argument("--fid", type=float, required=True)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / Path("results/fid/fid_sweep_unguided_10k"))
    args = parser.parse_args()
    run_dir = args.output_dir / f"k_{args.k:03d}"
    sample_npz = run_dir / "images.npz"
    if not sample_npz.is_file():
        raise FileNotFoundError(f"Expected generated samples at {sample_npz}")
    metrics = {
        "k_euler_updates": args.k,
        "integrator_time_points": args.k + 1,
        "fid": args.fid,
        "num_samples": args.num_samples,
        "cfg_scale": 1.0,
        "solver": "ODE/euler",
        "sample_npz": str(sample_npz.resolve()),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    rows = rebuild_summary(args.output_dir)
    print(f"Recorded K={args.k}, FID={args.fid:.6f}.")
    print(f"Wrote {args.output_dir / 'fid_vs_k.csv'} and {args.output_dir / 'fid_vs_k.svg'} ({len(rows)} points).")


if __name__ == "__main__":
    main()
