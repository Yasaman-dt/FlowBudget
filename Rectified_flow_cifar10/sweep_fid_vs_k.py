"""Generate 10k original RectifiedFlow CIFAR-10 Euler samples and score ADM FID."""
from pathlib import Path as _ConfigPath
import sys as _config_sys
_config_sys.path.insert(0, str(_ConfigPath(__file__).resolve().parents[1]))
from flowbudget_config import resource_path
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# The shared CSV/SVG/PNG reporter is kept with the SiT-ImageNet experiment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SiT_Imagenet"))
from fid_sweep_reporting import rebuild_summary


def fid_from_output(output):
    match = re.search(r"\bFID\s*:\s*([0-9.]+)", output)
    if not match:
        raise RuntimeError("Could not parse FID from evaluator output.\n" + output[-3000:])
    return float(match.group(1))


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=root / "ImageGeneration")
    parser.add_argument("--checkpoint", type=Path, default=resource_path(root / "checkpoints/cifar10_1_rectified_flow.pth"))
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--adm-evaluator", type=Path, required=True)
    parser.add_argument("--fid-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=root / "fid_sweep_10k")
    args = parser.parse_args()
    # The ADM evaluator runs with its own working directory, so its input
    # paths must be absolute rather than relative to this experiment folder.
    args.reference_npz = args.reference_npz.resolve()
    args.adm_evaluator = args.adm_evaluator.resolve()
    # Do not resolve this symlink: resolving .venv_sit/bin/python points to
    # the base interpreter and bypasses the virtual environment's packages.
    args.fid_python = Path(os.path.abspath(args.fid_python.expanduser()))
    args.runtime_root = args.runtime_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for k in dict.fromkeys(args.steps):
        run_dir = args.output_dir / f"k_{k:03d}"
        npz = run_dir / "images.npz"
        metrics = run_dir / "metrics.json"
        if not metrics.exists():
            if not npz.exists():
                subprocess.run([
                    sys.executable, str(root / "sample_rf_cifar10_euler.py"),
                    "--runtime-root", str(args.runtime_root), "--checkpoint", str(args.checkpoint),
                    "--k", str(k), "--num-samples", str(args.num_samples),
                    "--batch-size", str(args.batch_size), "--output-dir", str(run_dir),
                ], check=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""  # FID CPU: robust to TensorFlow/cuDNN mismatch.
            try:
                result = subprocess.run([str(args.fid_python), str(args.adm_evaluator), str(args.reference_npz), str(npz)],
                                        cwd=args.adm_evaluator.parent, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=env, check=True)
            except subprocess.CalledProcessError as error:
                log = error.stdout or ""
                (run_dir / "adm_evaluator.log").write_text(log)
                raise RuntimeError(f"FID failed for K={k}; see {run_dir / 'adm_evaluator.log'}\n{log[-3000:]}") from error
            (run_dir / "adm_evaluator.log").write_text(result.stdout)
            row = {"k_euler_updates": k, "fid": fid_from_output(result.stdout), "num_samples": args.num_samples,
                   "solver": "rectified_flow_ode/euler", "cfg_scale": "not_applicable", "sample_npz": str(npz)}
            metrics.write_text(json.dumps(row, indent=2) + "\n")
            print(f"K={k}: FID-10K={row['fid']:.6f}")
        rebuild_summary(args.output_dir)


if __name__ == "__main__":
    main()
