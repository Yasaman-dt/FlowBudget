# FlowBudget usage guide

Start with the [root README](../README.md) for installation, the method, and a
small 3-RF example. Run all commands here from the repository root.

- [Model assets](#other-models-and-data)
- [Paths and devices](#configure-paths-and-devices)
- [Reproduction](#reproduction-workflow)
- [Analysis and benchmarks](#analysis-and-benchmark-commands)

## Other models and data

| Model | Required assets and setup |
| --- | --- |
| SiT-XL/2 | Download the 256×256 checkpoint using the links in [SiT's README](../SiT_Imagenet/README.md); pass its path with `--checkpoint`. The estimators also load a VAE. Euler defaults to `--vae ema`; Heun defaults to `--vae mse`. |
| FlowDCN | Follow [FlowDCN's checkpoint links](../FlowDCN/README.md); place `FlowDCN-XL-2M-R256.pth` under `FlowDCN/checkpoints/`. See [experiment conventions](../FlowDCN/flowdcn_experiment/README.md). |
| RF-UNet | See [RF-UNet setup](../rectified_flow_modern/README.md); the local checkpoint/config directory is `rectified_flow_modern/checkpoints/unet_cifar10_pretrained/`. |

For ImageNet probes, supply your ImageNet training directory in ImageFolder
layout (`train/<class>/<image>`) through `--data-path` where supported. The
FlowDCN runner expects its FID reference at
`SiT_Imagenet/references/imagenet_train_10k_center_crop_256.npz`.
Configure resource paths before using launchers.

### FlowDCN setup

Use the shared environment from the root README; it includes Triton. FlowDCN
requires a CUDA GPU, its checkpoint, and a locally cached
`stabilityai/sd-vae-ft-ema` VAE. See [asset setup](../FlowDCN/README.md#setup).

## Configure paths and devices

Experiment runners share settings from `flowbudget_config.py`. Export the
variables you need before launching a command; [.env.example](../.env.example)
lists example values and is not loaded automatically. Paths containing spaces
must be quoted in your shell. Relative environment paths resolved by the Python
helper are relative to the repository root. Use absolute paths in shell settings.

```bash
export FLOWBUDGET_IMAGENET="/data/imagenet/train"
export FLOWBUDGET_RF_CHECKPOINTS="/models/rectified-flow"
export FLOWBUDGET_EVALUATOR="/tools/guided-diffusion/evaluations/evaluator.py"
export CUDA_VISIBLE_DEVICES=0
```

| Setting | Purpose and default |
| --- | --- |
| `FLOWBUDGET_IMAGENET` | ImageNet training images; defaults to `data/imagenet/train`. |
| `FLOWBUDGET_CIFAR10` | CIFAR-10 data used by configs; defaults to `rectified_flow_cifar10/data`. |
| `FLOWBUDGET_RF_CHECKPOINTS` | Directory containing `cifar10_{1,2,3}_rectified_flow.pth`. |
| `FLOWBUDGET_UNET_CHECKPOINTS` | Directory containing `unet_cifar10_pretrained/`. |
| `FLOWBUDGET_FLOWDCN_CHECKPOINTS` | Directory containing `FlowDCN-XL-2M-R256.pth`. |
| `FLOWBUDGET_SIT_CHECKPOINT` | SiT estimator/sampler checkpoint; omitted to retain automatic pretrained-model lookup. |
| `FLOWBUDGET_EVALUATOR` | ADM evaluator override for runners with evaluator defaults. Commands with a required `--adm-evaluator` still need that flag. |
| `FLOWBUDGET_CIFAR_REFERENCE`, `FLOWBUDGET_IMAGENET_REFERENCE` | Override the corresponding default FID reference NPZ. |
| `CUDA_VISIBLE_DEVICES` | GPU visibility inherited by launchers; `--gpu` overrides it in runners exposing that option. |
| `FLOWBUDGET_GPU` | Fallback for launcher GPU selection when visibility is unset; otherwise defaults to GPU 0. |
| `FLOWBUDGET_DEVICE` | Default device for single-device estimators/RF samplers; an explicit `--device` wins. |
| `FLOWBUDGET_PYTHON` | Interpreter used by shell launchers; defaults to `python` on PATH. |

Checkpoint-directory defaults remain the respective model's `checkpoints/`
folder. Existing explicit command-line data/checkpoint/evaluator paths take
precedence over defaults. JSON input configs support `${FLOWBUDGET_IMAGENET}`,
`${FLOWBUDGET_CIFAR10}`, `${FLOWBUDGET_RF_CHECKPOINTS}`, and
`${FLOWBUDGET_ROOT}` placeholders, expanded by the experiment runners. Saved
result JSONs and frozen experiment records retain their original paths.

After `CUDA_VISIBLE_DEVICES=2`, `--device cuda:0` means the first visible GPU
(physical GPU 2). For CPU-capable scripts use `--device cpu`; FlowDCN's CUDA
kernels still require a GPU. Use a fresh output directory when changing
settings; existing experiment manifests can reject a mismatched resume.

## Reproduction workflow

### Estimate with the manuscript's probe count

This runs the original 3-RF Euler estimator for three independent probe seeds:

```bash
for seed in 0 1 2; do
  python rectified_flow_cifar10/estimate_euler_budget_fm.py \
    --checkpoint rectified_flow_cifar10/checkpoints/cifar10_3_rectified_flow.pth \
    --data-root rectified_flow_cifar10/data \
    --num-samples 2000 --batch-size 1 --tau 0.01 --epsilon 1e-8 \
    --seed "$seed" --save-probe-pool \
    --output "results/3rf_seed_${seed}.json"
done
```

Use the 1-RF or 2-RF checkpoint and distinct output filenames to repeat for
those models. The manuscript uses 2,000 probes per seed, three seeds,
ODE tolerance 0.010, SDE tolerance 0.015, and epsilon 1e-8. Its FID evaluation
uses 10,000 generated images at each seed's predicted budget.

### Evaluate a predicted budget

Build the separate **test-split 10K**
reference for this repository's FID evaluation with:

```bash
python rectified_flow_cifar10/build_cifar10_reference_npz.py \
  --data-root rectified_flow_cifar10/data \
  --output rectified_flow_cifar10/references/cifar10_test_10k.npz
```

These FID-10K values use this specific real reference and are not directly
comparable to upstream FID values computed with a different reference protocol.

Obtain the ADM evaluator from `openai/guided-diffusion`, with its accompanying
evaluation files. The evaluator requires TensorFlow and downloads its Inception
model when absent. Set its path and read the estimate from the preceding run:

```bash
export FLOWBUDGET_EVALUATOR=/absolute/path/to/guided-diffusion/evaluations/evaluator.py
K_HAT=$(python -c "import json; print(json.load(open('results/3rf_seed_0.json'))['K_hat'])")
python rectified_flow_cifar10/sweep_fid_vs_k.py \
  --checkpoint rectified_flow_cifar10/checkpoints/cifar10_3_rectified_flow.pth \
  --reference-npz rectified_flow_cifar10/references/cifar10_test_10k.npz \
  --adm-evaluator "$FLOWBUDGET_EVALUATOR" \
  --steps "$K_HAT" --num-samples 10000 --batch-size 128 \
  --output-dir results/3rf_predicted_fid
```

This runner evaluates FID on CPU and writes a summary CSV under the output
directory. To generate a retrospective 3-RF quality curve, use
`--steps 1 2 4 8 16 32` and a separate output directory. Such a sweep is an
evaluation of the budget estimate; it is not required to construct it.
The sweep wrapper does not expose a sampling-seed argument, so this command
alone does not reproduce the full three-seed table.

### Rebuild figures from saved experiment results

The plotting commands below read saved artifacts from paths configured in their
scripts. The original RF and RF-UNet FID curves use their model directories’
`results/fid/` folders. Outputs from a new run are not automatically selected;
set the plot inputs to the files you want to compare.

| Manuscript item | Code and required inputs |
| --- | --- |
| Table 1 | `scripts/paper/audit_final_paper.py` with the ignored local audit config maps coefficients to predicted budgets and saved FID rows, and flags unresolved discrepancies. |
| Fig. 1a: FID versus NFE | `scripts/plots/plot_all_architectures_fid.py`; model FID CSV paths are listed in `SERIES` and `FLOWDCN_SERIES`. |
| Fig. 1b: probe stability | `scripts/plots/plot_all_architectures_khat_vs_n.py`; saved `stability.csv` and per-seed results under `probe_sensitivity_forward/`. |
| Fig. 1c: FID versus tolerance | Run the coefficient export before the FID plot; requires three-seed coefficient CSV/JSON files plus FID curves. |

```bash
python scripts/plots/plot_all_architectures_fid.py
python scripts/plots/plot_all_architectures_khat_vs_n.py \
  --min-n 1000 --max-n 20000 --use-nfe --no-annotations
python scripts/plots/plot_sit_rf_khat_vs_tau_2000.py
python scripts/plots/plot_sit_rf_fid_vs_tau_2000.py
```

The tolerance generators save PNG figures and CSV/Markdown companion reports.
The FID-versus-tolerance script infers tolerances from saved coefficients and
measured budget points; it does not run new FID experiments.

### Paper settings

[configs/final_paper.json](../configs/final_paper.json) records only the paper execution settings
and estimator arguments. Reported numbers and historical source mappings are
preserved in `configs/local/final_paper_audit.json`, which is ignored by Git. `scripts/paper/audit_final_paper.py` is an optional
local audit tool requiring saved experimental artifacts, which are not included
in this code-only release. Its generated reports remain local.

See [estimator conventions](ESTIMATOR_FORMULAS.md) for formulas, NFE
accounting, and analytic checks. Model-level finite-difference accuracy still
requires validation on the chosen pretrained checkpoint.
[configs/probe_sensitivity.json](../configs/probe_sensitivity.json) contains the
shared probe-sweep inputs; [configuration usage](../configs/README.md) explains
model selection and path expansion.

## Analysis and benchmark commands

Run from the repository root:

```bash
python scripts/analysis/sweep_khat_vs_tau.py --help
python scripts/benchmarks/benchmark_3rf_curve_vs_budget.py --help
```

The tolerance sweep supports `--solver euler` and `--solver heun`.
The 3-RF benchmark measures wall-clock time for a quality sweep versus budget
estimation followed by FID evaluation. Its default output directory is
`rectified_flow_cifar10/results/timing/3rf/curve_vs_budget/new_run/`.
Use `--output` with a distinct directory for each new experiment. Shared
environment settings still configure resource paths and devices. Model-specific
sweeps remain alongside their model code.

### Unified probe-sensitivity command

`probe_sensitivity.py` supports three modes:

- `--mode ode` (default): the existing config-driven Euler/Heun probe sweeps.
- `--mode em`: the SiT EM nested-probe sweep with counts 25–20000, seeds
  0, 1, 2, and tolerance 0.01. For the final-paper SDE tolerance 0.015, use
  the estimator arguments in `configs/final_paper.json`.
- `--mode merge`: combine completed ODE experiments with matching tolerance,
  epsilon, seeds, and the default probe counts (25, 50, 100, 250, 500).
  Duplicate model/solver pairs are rejected. Inputs are preserved; use a separate
  output directory for the combined reports.

```bash
python probe_sensitivity.py --mode merge \
  --inputs results/probes_rf results/probes_sit \
  --output-dir results/probes_combined
python probe_sensitivity.py --mode em --gpu 0 --output-dir results/probe_em
```

For custom EM probe counts:

```bash
python probe_sensitivity.py --mode em --counts 500 2000 \
  --gpu 0 --output-dir results/em_custom
python scripts/run_sit_em_2000_three_seeds.py \
  --gpu 0 --output-dir results/em_2000
```

The first command generates one 2,000-probe pool per seed and summarizes both
prefixes. Counts must be distinct positive integers. The second command uses the
same implementation with a default of 2,000 probes only. Omitting `--counts`
from `--mode em` retains the 25–20,000 sweep. Both commands use the leading-order
EM estimator and report NFE = K. Resume requires a matching experiment manifest;
use a new output directory when changing settings.

### Extended probe-count sweeps

Use one launcher to select the RF or SiT models:

```bash
bash scripts/run_probe_sensitivity_20000.sh rf
bash scripts/run_probe_sensitivity_20000.sh sit
```

Both use probe counts 3000–20000, seeds 0/1/2, and tolerance 0.01.
They use their configured per-group output directories and shared device settings.
Append `--output-dir results/my_probe_run` to write a separate run, or `--gpu 0`
to select a device explicitly.

## Saved artifacts

Use the [results layout](../README.md#reproduction-and-results) to locate model
outputs. Original RF timing records are under
`rectified_flow_cifar10/results/timing/`; failed or interrupted runs are archived
there. Shared plots and companion reports default to `results/figures/`.
Explicit output arguments override defaults. Historical paths in saved logs and
JSON records describe the original run locations.

Generated data, checkpoints, figures, logs, and result reports are ignored by
Git. The paper audit also requires a local configuration and saved sources that
are excluded from this release. Preserve those records when reorganizing files.
For implementation history, see [archived cleanup notes](archive/CODE_CLEANUP.md).
