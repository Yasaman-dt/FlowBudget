#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  rf) models=(1-RF 2-RF 3-RF RF-UNet)
      output=probe_sensitivity_forward/additional_3000_20000_tau001_rf_gpu2 ;;
  sit) models=(SiT-XL/2)
       output=probe_sensitivity_forward/additional_3000_20000_tau001_sit_gpu3 ;;
  -h|--help) echo "Usage: $0 {rf|sit} [probe_sensitivity.py options]"; exit 0 ;;
  *) echo "Usage: $0 {rf|sit} [probe_sensitivity.py options]" >&2; exit 2 ;;
esac
shift
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime_env.sh"
exec "$FLOWBUDGET_PYTHON" -u probe_sensitivity.py \
  --config configs/probe_sensitivity.json --models "${models[@]}" --tau 0.01 --epsilon 1e-8 \
  --seeds 0 1 2 --gpu "$CUDA_VISIBLE_DEVICES" \
  --counts 3000 4000 5000 6000 7000 8000 9000 10000 15000 20000 \
  --output-dir "$output" "$@"
