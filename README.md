# FlowBudget

**Training-Free Sampling-Budget Estimation for Flow Matching Models**


## Overview

How many sampling steps does a pretrained flow model need for a fixed solver?
FlowBudget estimates this budget from data–noise interpolation probes, without
retraining the model or running an FID-versus-NFE sweep. It measures the leading
local discretization error relative to motion and maps a chosen numerical
tolerance to a step budget using the solver order.

The method supports Euler and Heun ODE sampling and additive-noise
Euler–Maruyama (EM) SDE sampling. It requires representative data and access to
the pretrained model. The tolerance controls a numerical-error criterion; it
does not guarantee a particular FID.

| Solver | Estimated steps K | Network function evaluations (NFE) |
| --- | --- | --- |
| Euler | `ceil(C / (2τ))` | K |
| Heun | `ceil(sqrt(C / (12τ)))` | 2K |
| Euler–Maruyama | `ceil(C / (sqrt(3)τ))` | K |

Here, C is the solver-specific coefficient averaged over interpolation probes;
budgets are clamped to at least one step. See
[estimator conventions](docs/ESTIMATOR_FORMULAS.md) for coefficient definitions,
time conventions, and the uniform unguided EM assumptions.

## Paper experiments

| Models | Dataset | Solvers |
| --- | --- | --- |
| 1-RF, 2-RF, 3-RF, RF-UNet | CIFAR-10 | Euler |
| SiT-XL/2, FlowDCN-XL-2M | ImageNet 256×256 | Euler, Heun, EM |

The manuscript uses 2,000 probes per seed, seeds 0/1/2, stabilization
`epsilon = 1e-8`, ODE tolerance `0.010`, and SDE tolerance `0.015`.
FID evaluation uses 10,000 generated images per evaluated budget.
[configs/final_paper.json](configs/final_paper.json) lists the settings and
estimator arguments for all ten model–solver pairs. Some scripts retain
historical defaults; use the explicit paper settings for reproduction.

## Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate SiT
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The environment targets Linux x86_64 with Python 3.10, PyTorch 2.6.0 and
Torchvision 0.21.0 built for CUDA 12.4. GPU experiments require a working NVIDIA
driver. Original RF also requires a CUDA toolkit with `nvcc` and a C++ compiler
for its extensions. FlowDCN uses Triton, included in the environment.

## Quick start: estimate a 3-RF budget

Download the non-distilled 3-RF checkpoint from the
[original Rectified Flow checkpoint list](https://github.com/gnobitab/RectifiedFlow#pre-trained-checkpoints)
and save it as `rectified_flow_cifar10/checkpoints/cifar10_3_rectified_flow.pth`.
Then run a small 32-probe example:

```bash
CUDA_VISIBLE_DEVICES=0 python rectified_flow_cifar10/estimate_euler_budget_fm.py \
  --checkpoint rectified_flow_cifar10/checkpoints/cifar10_3_rectified_flow.pth \
  --data-root rectified_flow_cifar10/data \
  --num-samples 32 --batch-size 1 --tau 0.01 --epsilon 1e-8 --seed 0 \
  --output results/quickstart_3rf.json
```

CIFAR-10 is downloaded when needed. The output includes the estimated
coefficient `R_theta`, step budget `K_hat`, and elapsed time. For Euler,
`NFE = K_hat`. This command estimates the budget without generating images or
computing FID. Use 2,000 probes and seeds 0/1/2 for the paper protocol.

For other checkpoints and data paths, follow the
[usage guide](docs/USAGE.md#other-models-and-data) and
[environment-variable examples](.env.example).

## Reproduction and results

1. Prepare the model checkpoint and dataset.
2. Run estimation with the arguments in
   [configs/final_paper.json](configs/final_paper.json).
3. Sample at the predicted budget and evaluate FID using the stated reference
   dataset. Original RF uses a CIFAR-10 test-split reference of 10,000 images;
   comparisons with other FID protocols require care.
4. Generate figures from the saved estimates and FID measurements.

See the [reproduction commands](docs/USAGE.md#reproduction-workflow) for a
three-seed example, FID evaluation, and figure generation.
Model artifacts are organized under each model's `results/fid/` and
`results/estimates/`; shared budget reports and figures are under
`results/budget_estimates/` and `results/figures/`.

This release contains code and configurations. Checkpoints, datasets, saved
experiment outputs, and the local paper-audit configuration are excluded from
Git, so a fresh clone cannot immediately regenerate every paper figure.
Saved-source auditing has identified unresolved manuscript/result discrepancies
and a FlowDCN Heun finite-difference sensitivity failure. Full numerical
reproduction remains unverified; see the
[validation limits](docs/ESTIMATOR_FORMULAS.md#validation-and-limits).

## Code and validation

| Location | Contents |
| --- | --- |
| `rectified_flow_cifar10/`, `rectified_flow_modern/` | Original RF and RF-UNet experiments |
| `SiT_Imagenet/`, `FlowDCN/` | ImageNet model runtimes and experiments |
| `probe_sensitivity.py`, `scripts/` | Probe sweeps, launchers, plots, and benchmarks |
| `transport/` | Shared integration code |
| `configs/`, `tests/` | Experiment settings and tests |

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

A fresh CUDA environment and all 20 tests passed. One-image checkpoint sampling
also passed for 3-RF, RF-UNet, SiT, and FlowDCN in the existing runtime on A100
GPUs. These checks verify basic operation, not the full paper's FID results.
Advanced configuration and analysis commands are in the [usage guide](docs/USAGE.md).
