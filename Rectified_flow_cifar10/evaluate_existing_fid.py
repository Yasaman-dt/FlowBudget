"""Compute ADM FID for already-generated RectifiedFlow CIFAR-10 .npz files."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "SiT_Imagenet"))
from fid_sweep_reporting import rebuild_summary


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--adm-evaluator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "fid_sweep_10k")
    args = parser.parse_args()
    reference = args.reference_npz.resolve()
    evaluator = args.adm_evaluator.resolve()
    output_dir = args.output_dir.resolve()
    # This preserves the active virtual environment instead of resolving its
    # python symlink to the base interpreter.
    venv = os.environ.get("VIRTUAL_ENV")
    python = Path(venv) / "bin/python" if venv else Path(sys.executable)
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python}")
    for k in dict.fromkeys(args.steps):
        run_dir = output_dir / f"k_{k:03d}"
        npz = run_dir / "images.npz"
        metrics = run_dir / "metrics.json"
        if metrics.is_file():
            print(f"K={k}: already recorded; skipping.")
            continue
        if not npz.is_file():
            raise FileNotFoundError(f"No generated samples for K={k}: {npz}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        result = subprocess.run([str(python), str(evaluator), str(reference), str(npz)],
                                cwd=evaluator.parent, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env)
        (run_dir / "adm_evaluator.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"FID failed for K={k}; see {run_dir / 'adm_evaluator.log'}\n{result.stdout[-3000:]}")
        match = re.search(r"\bFID\s*:\s*([0-9.]+)", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse FID for K={k}; see {run_dir / 'adm_evaluator.log'}")
        row = {"k_euler_updates": k, "fid": float(match.group(1)), "num_samples": 10_000,
               "solver": "rectified_flow_ode/euler", "cfg_scale": "not_applicable", "sample_npz": str(npz)}
        metrics.write_text(json.dumps(row, indent=2) + "\n")
        rebuild_summary(output_dir)
        print(f"K={k}: FID-10K={row['fid']:.6f}")


if __name__ == "__main__":
    main()
