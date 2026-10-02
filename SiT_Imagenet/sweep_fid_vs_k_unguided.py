"""Run the original SiT sampler at several *Euler update* budgets and score FID.

This is deliberately an unguided deterministic ODE experiment: cfg_scale=1.0,
torchdiffeq's fixed-step Euler method, and K Euler updates.  The original
integrator receives K+1 time points, since K points would make only K-1 updates.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fid_sweep_reporting import rebuild_summary


def read_fid(output):
    match = re.search(r"\bFID\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", output, flags=re.IGNORECASE)
    if not match:
        raise RuntimeError("Could not find `FID: <number>` in ADM evaluator output.\n" + output[-4000:])
    return float(match.group(1))


def run_one(args, k):
    run_dir = args.output_dir / f"k_{k:03d}"
    sample_dir = run_dir / "images"
    sample_npz = Path(str(sample_dir) + ".npz")
    metrics_path = run_dir / "metrics.json"
    if run_dir.exists() and args.rerun:
        shutil.rmtree(run_dir)
    if metrics_path.exists() and not args.rerun:
        return json.loads(metrics_path.read_text())
    if sample_npz.is_file() and not args.rerun:
        print(f"Reusing existing samples for K={k}: {sample_npz}")
    else:
        if run_dir.exists() and not args.rerun:
            raise FileExistsError(f"{run_dir} exists but has no complete .npz; use --rerun to replace this one run.")
        run_dir.mkdir(parents=True)
        command = [
            "torchrun", "--standalone", f"--nproc_per_node={args.gpus}", "sample_ddp.py", "ODE",
            "--model", "SiT-XL/2", "--image-size", "256", "--vae", args.vae,
            "--sample-folder", str(sample_dir), "--per-proc-batch-size", str(args.batch_size),
            "--num-fid-samples", str(args.num_samples), "--cfg-scale", "1.0",
            "--num-sampling-steps", str(k + 1), "--sampling-method", "euler",
            "--global-seed", str(args.seed), "--learn-sigma", "--save-grid",
            "--grid-num-images", str(args.grid_num_images), "--grid-cols", str(args.grid_cols),
        ]
        if args.checkpoint:
            command += ["--ckpt", str(args.checkpoint)]
        subprocess.run(command, cwd=args.repo, check=True)
    try:
        fid_env = os.environ.copy()
        if args.fid_device == "cpu":
            # TensorFlow GPU wheels can require a different cuDNN minor version
            # than the one installed for PyTorch. FID is independent of SiT
            # sampling, so CPU evaluation is the portable fallback.
            fid_env["CUDA_VISIBLE_DEVICES"] = ""
        evaluator = subprocess.run(
            [str(args.fid_python), str(args.adm_evaluator), str(args.reference_npz), str(sample_npz)],
            cwd=args.adm_evaluator.parent, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=fid_env,
        )
    except subprocess.CalledProcessError as error:
        log = error.stdout or ""
        log_path = run_dir / "adm_evaluator.log"
        log_path.write_text(log)
        raise RuntimeError(
            f"ADM FID evaluation failed; its complete output is in {log_path}.\n{log[-4000:]}"
        ) from error
    (run_dir / "adm_evaluator.log").write_text(evaluator.stdout)
    row = {
        "k_euler_updates": k,
        "integrator_time_points": k + 1,
        "fid": read_fid(evaluator.stdout),
        "num_samples": args.num_samples,
        "cfg_scale": 1.0,
        "solver": "ODE/euler",
        "sample_npz": str(sample_npz),
    }
    metrics_path.write_text(json.dumps(row, indent=2) + "\n")
    print(f"K={k}: FID-10K={row['fid']:.6f}; saved {metrics_path}")
    return row


def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 250])
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-num-images", type=int, default=64,
                        help="Number of samples shown in each qualitative image_grid.png.")
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--vae", choices=["ema", "mse"], default="ema")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Official SiT-XL/2 checkpoint. Omit to use sample_ddp.py's official auto-download.")
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--adm-evaluator", type=Path, required=True,
                        help="Path to guided-diffusion/evaluations/evaluator.py.")
    parser.add_argument("--fid-python", type=Path, default=Path(sys.executable),
                        help="Python interpreter with TensorFlow and the ADM evaluator dependencies.")
    parser.add_argument("--fid-device", choices=["cpu", "gpu"], default="cpu",
                        help="Device for TensorFlow FID. CPU avoids TensorFlow/cuDNN version mismatches.")
    parser.add_argument("--output-dir", type=Path, default=repo / "results/fid/fid_sweep_unguided_10k")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    args.repo = repo
    args.output_dir = args.output_dir.resolve()
    args.reference_npz = args.reference_npz.resolve()
    args.adm_evaluator = args.adm_evaluator.resolve()
    # Keep the virtual-environment interpreter path intact.  Resolving its
    # symlink points to the base Python and loses packages such as NumPy and
    # TensorFlow required by the ADM evaluator.
    args.fid_python = Path(os.path.abspath(args.fid_python.expanduser()))
    if not args.reference_npz.is_file() or not args.adm_evaluator.is_file() or not args.fid_python.is_file():
        raise FileNotFoundError("--reference-npz, --adm-evaluator, and --fid-python must all exist.")

    rows = []
    for k in dict.fromkeys(args.steps):
        rows.append(run_one(args, k))
        # Keep results durable even if a later K is interrupted or fails.
        rebuild_summary(args.output_dir)
    rows.sort(key=lambda row: row["k_euler_updates"])
    best = min(rows, key=lambda row: row["fid"])
    print(f"Best measured FID is {best['fid']:.4f} at K={best['k_euler_updates']} Euler updates.")


if __name__ == "__main__":
    main()
