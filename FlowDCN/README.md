# FlowDCN runtime for FlowBudget

This directory contains the FlowDCN model and Triton kernels used by the
FlowBudget Euler, Heun, and Euler–Maruyama experiments.

Upstream: *FlowDCN: Exploring DCN-like Architectures for Fast Image Generation
with Arbitrary Resolution* (NeurIPS 2024).
The retained source files preserve their existing attribution.

## Setup

Follow the [FlowDCN setup section in the usage guide](../docs/USAGE.md#flowdcn-setup).
It uses the shared `SiT` environment and checks the Triton/CUDA dependencies.
All runtime dependencies are maintained in the root `environment.yml`, including
Triton 3.2.0. No separate FlowDCN requirements installation is needed.

Download the [FlowDCN-XL-2M-R256 checkpoint](https://huggingface.co/wangsssssss/FlowDCN/blob/main/FlowDCN-XL-2M-R256.pth)
and set `FLOWBUDGET_FLOWDCN_CHECKPOINTS` to its containing directory.
The runner loads `stabilityai/sd-vae-ft-ema` from the local Hugging Face cache;
that VAE must be available before running. Configure ImageNet and FID resources
as described in the root README.

## Commands

From the repository root:

```bash
python FlowDCN/flowdcn_experiment/run.py --help
python FlowDCN/flowdcn_experiment/run.py estimate --help
python FlowDCN/flowdcn_experiment/run.py sample --help
```

Use [configs/final_paper.json](../configs/final_paper.json) for paper estimator
settings and the [experiment guide](flowdcn_experiment/README.md) for sampling
and evaluation commands.

## Included source

- `flowdcn_experiment/`: estimation, sampling, export, and analytic tests.
- `src/models/flowdcn.py` and `base_model.py`: checkpoint-compatible model.
- `src/ops/triton_kernels/`: forward and backward kernel implementation.

The upstream Lightning training/prediction CLI, its configurations, alternative
models and kernels, and latent-cache tool were removed from this paper-focused
copy. Use the upstream repository for those workflows.
