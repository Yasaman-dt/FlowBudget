# Experiment configurations

- **`final_paper.json`**: the release entry point. Explicit final-paper settings,
  tolerances, probe counts, and estimator commands for all
  ten model–solver pairs. See [the estimator conventions](../docs/ESTIMATOR_FORMULAS.md).
- **`legacy/probe_sensitivity_dit_config.json`**: optional DiT probe-count
  experiment outside the main paper model set.

Legacy configs declare `base_dir: ${FLOWBUDGET_ROOT}` so script and input paths
continue to resolve from the repository root after relocation. Use the experiment
runners to expand environment placeholders; do not pass unresolved JSON arguments
directly to a shell.

Saved result JSONs remain in their original experiment directories and are
excluded from the public code selection by `.gitignore`. Copies of frozen
experiment records are retained as independent historical snapshots. Removing
one would break source references or lose a run's local provenance.

- **`local/final_paper_audit.json`**: private full copy containing reported
  numbers and saved-result mappings. Ignored by Git; used by the local audit.
  The public `final_paper.json` contains no experimental results.

## Shared probe sensitivity config

`probe_sensitivity.json` replaces the former base, extended, RF-only, and
SiT-only probe configs. RF-UNet's existing default finite-difference method
and 0.001 increment are now explicit. Select subsets with the runner:

```bash
python probe_sensitivity.py --config configs/probe_sensitivity.json --models 1-RF 2-RF 3-RF RF-UNet --tau 0.01 --output-dir results/probes_rf
python probe_sensitivity.py --config configs/probe_sensitivity.json --models SiT-XL/2 --tau 0.01 --output-dir results/probes_sit
```

Selection preserves config order and includes all solvers for selected models.
Subset output directories use contiguous `pair_0`, `pair_1`, etc., matching the
former subset configs. Existing manifests from the base config with implicit
RF-UNet defaults may reject resume because those defaults are now explicit;
use a fresh output directory. Saved manifests are not rewritten.
