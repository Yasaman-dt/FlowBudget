#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAU_CSV="$ROOT/../results/budget_estimates/euler_budget_tau_sweep_2tau.csv"

for RF in 1 2 3; do
  EXTRA_ARGS=()
  if [[ "$RF" == "1" ]]; then
    EXTRA_ARGS+=(--exclude-fid-k 70 --below-tau-k 173)
  fi
  python "$ROOT/plot_fid_vs_khat_tau.py" \
    --fid-csv "$ROOT/results/fid/fid_sweep_${RF}rf_10k/fid_vs_k.csv" \
    --tau-csv "$TAU_CSV" \
    --model "${RF}-RF" \
    --title "${RF}-Rectified Flow CIFAR-10: FID and Euler-budget estimate" \
    --output-dir "$ROOT/results/fid/fid_sweep_${RF}rf_10k" \
    --min-k 4 \
    --point-labels xy \
    "${EXTRA_ARGS[@]}"
done
