# Applied dependency cleanup

Historical record of the cleanup at that time. For current commands and
validation status, see the [usage guide](../USAGE.md) and [README](../../README.md).

Removed 130 tracked upstream files outside the retained FlowBudget experiment
import paths. Preserved all estimation, sampling, evaluation, plotting, benchmark,
and test entry points, including the documented modern DiT sampler.

The review followed Python imports recursively, including function-local imports,
relative imports, and package initializers, and checked subprocess/configuration
references. No retained file in the pre-cleanup import graph imports a removed
module. Validation: all 17 tests in the root test suite passed in `SiT_env`; nine
configuration/paper/probe-selection tests also passed with system Python. All 110
remaining tracked Python files parsed successfully; all ten paper estimator
entry points exist; the retained import closure has no edges to removed files;
FlowDCN CLI help works; `git diff --check` passed. Full model loading and GPU
sampling have not been verified.

## Deliberately retained

- Original RF `models/ema.py`: checkpoint deserialization compatibility.
- Original RF `sde_lib.py`, the CIFAR model/configuration, and all `op/` sources.
- Modern RF training-time helper modules and every sampler imported by package
  initialization, plus DiT for the existing sampling command.
- FlowDCN forward and backward Triton kernels and both required model files.
- Dependency declarations, existing licenses and source attribution.
- Local checkpoints, data, generated results, logs, and compiled caches.

The removed FlowDCN Lightning CLI and its configuration-driven training and
sampling commands are no longer supported by this copy. Its README now describes
the retained experiment runtime. A pre-deletion backup, including the former
FlowDCN README, is outside this repository under `../release_cleanup_backups/`.

## Removed files

- `FlowDCN/configs/sampling/flowdcn_xl_2_adam2.yml`
- `FlowDCN/configs/sampling/flowdcn_xl_2_adam4.yml`
- `FlowDCN/configs/sampling/flowdcn_xl_2_euler.yml`
- `FlowDCN/configs/sampling/flowdcn_xl_2_eulerSDE.yml`
- `FlowDCN/configs/sampling/flowdcn_xl_2_neuralsolver.yml`
- `FlowDCN/configs/sampling/var_flowdcn_b_2.yml`
- `FlowDCN/configs/training/flowdcn_b_2.yml`
- `FlowDCN/configs/training/flowdcn_l_2.yml`
- `FlowDCN/configs/training/flowdcn_s_2.yml`
- `FlowDCN/configs/training/flowdcn_xl_2.yml`
- `FlowDCN/configs/training/lognorm_flowdcn_b_2.yml`
- `FlowDCN/configs/training/sit_s_2.yml`
- `FlowDCN/configs/training/sit_xl_2.yml`
- `FlowDCN/configs/training/var_flowdcn_b_2.yml`
- `FlowDCN/main.py`
- `FlowDCN/src/diffusion/__init__.py`
- `FlowDCN/src/diffusion/base/guidance.py`
- `FlowDCN/src/diffusion/base/sampling.py`
- `FlowDCN/src/diffusion/base/scheduling.py`
- `FlowDCN/src/diffusion/base/training.py`
- `FlowDCN/src/diffusion/ddpm/ddim_sampling.py`
- `FlowDCN/src/diffusion/ddpm/dpmsolver_sampling.py`
- `FlowDCN/src/diffusion/ddpm/ns_sampling.py`
- `FlowDCN/src/diffusion/ddpm/scheduling.py`
- `FlowDCN/src/diffusion/ddpm/training.py`
- `FlowDCN/src/diffusion/ddpm/vp_sampling.py`
- `FlowDCN/src/diffusion/deq/sampling.py`
- `FlowDCN/src/diffusion/deq/training.py`
- `FlowDCN/src/diffusion/flow_matching/adam_sampling.py`
- `FlowDCN/src/diffusion/flow_matching/ns_sampling.py`
- `FlowDCN/src/diffusion/flow_matching/sampling.py`
- `FlowDCN/src/diffusion/flow_matching/scheduling.py`
- `FlowDCN/src/diffusion/flow_matching/training.py`
- `FlowDCN/src/diffusion/flow_matching/var_sampling.py`
- `FlowDCN/src/diffusion/flow_matching/var_training.py`
- `FlowDCN/src/lightning_data.py`
- `FlowDCN/src/lightning_model.py`
- `FlowDCN/src/models/dit.py`
- `FlowDCN/src/ops/DCNv4_op/DCNv4/__init__.py`
- `FlowDCN/src/ops/DCNv4_op/DCNv4/functions/__init__.py`
- `FlowDCN/src/ops/DCNv4_op/DCNv4/functions/dcnv4_func.py`
- `FlowDCN/src/ops/DCNv4_op/DCNv4/functions/flash_deform_attn_func.py`
- `FlowDCN/src/ops/DCNv4_op/DCNv4/functions/table.py`
- `FlowDCN/src/ops/DCNv4_op/DCNv4/modules/__init__.py`
- `FlowDCN/src/ops/DCNv4_op/DCNv4/modules/dcnv4.py`
- `FlowDCN/src/ops/DCNv4_op/DCNv4/modules/flash_deform_attn.py`
- `FlowDCN/src/ops/DCNv4_op/MANIFEST.in`
- `FlowDCN/src/ops/DCNv4_op/__init__.py`
- `FlowDCN/src/ops/DCNv4_op/make.sh`
- `FlowDCN/src/ops/DCNv4_op/scripts/find_best.py`
- `FlowDCN/src/ops/DCNv4_op/scripts/search_bwd.sh`
- `FlowDCN/src/ops/DCNv4_op/scripts/search_dcnv4.py`
- `FlowDCN/src/ops/DCNv4_op/scripts/search_dcnv4_bwd.py`
- `FlowDCN/src/ops/DCNv4_op/scripts/search_dcnv4_bwd_engine.py`
- `FlowDCN/src/ops/DCNv4_op/scripts/search_dcnv4_engine.py`
- `FlowDCN/src/ops/DCNv4_op/scripts/search_fwd.sh`
- `FlowDCN/src/ops/DCNv4_op/scripts/test_dcnv4.py`
- `FlowDCN/src/ops/DCNv4_op/scripts/test_dcnv4_bwd.py`
- `FlowDCN/src/ops/DCNv4_op/scripts/test_flash_deform_attn.py`
- `FlowDCN/src/ops/DCNv4_op/scripts/test_flash_deform_attn_backward.py`
- `FlowDCN/src/ops/DCNv4_op/setup.py`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/common.h`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/dcnv4_col2im_cuda.cuh`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/dcnv4_cuda.cu`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/dcnv4_cuda.h`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/dcnv4_im2col_cuda.cuh`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/flash_deform_attn_cuda.cu`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/flash_deform_attn_cuda.h`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/flash_deform_col2im_cuda.cuh`
- `FlowDCN/src/ops/DCNv4_op/src/cuda/flash_deform_im2col_cuda.cuh`
- `FlowDCN/src/ops/DCNv4_op/src/dcnv4.h`
- `FlowDCN/src/ops/DCNv4_op/src/vision.cpp`
- `FlowDCN/src/ops/cuda_kernels/backward.cu`
- `FlowDCN/src/ops/cuda_kernels/bak_forward.cu`
- `FlowDCN/src/ops/cuda_kernels/forward.cu`
- `FlowDCN/src/ops/cuda_kernels/forward.py`
- `FlowDCN/src/ops/cuda_kernels/function.py`
- `FlowDCN/src/ops/cuda_kernels/setup.py`
- `FlowDCN/src/ops/triton_kernels_udcn/backward.py`
- `FlowDCN/src/ops/triton_kernels_udcn/forward.py`
- `FlowDCN/src/ops/triton_kernels_udcn/function.py`
- `FlowDCN/src/ops/triton_kernels_udcn/utils.py`
- `FlowDCN/src/utils/__init__.py`
- `FlowDCN/src/utils/callbacks.py`
- `FlowDCN/src/utils/metrics.py`
- `FlowDCN/src/utils/model_loader.py`
- `FlowDCN/src/utils/saver.py`
- `FlowDCN/src/utils/vae.py`
- `FlowDCN/tools/cache_imlatent.py`
- `rectified_flow_cifar10/ImageGeneration/configs/default_lsun_configs.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/afhq_cat_pytorch_rf_gaussian.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/bedroom_rf_gaussian.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/church_rf_gaussian.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/cifar10_rf_gaussian_reflow_distill_k=1.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/cifar10_rf_gaussian_reflow_distill_k=1_online.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/cifar10_rf_gaussian_reflow_distill_k_g_1.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/cifar10_rf_gaussian_reflow_generate_data.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/cifar10_rf_gaussian_reflow_train.py`
- `rectified_flow_cifar10/ImageGeneration/configs/rectified_flow/cifar10_rf_gaussian_reflow_train_online.py`
- `rectified_flow_cifar10/ImageGeneration/datasets.py`
- `rectified_flow_cifar10/ImageGeneration/debug.py`
- `rectified_flow_cifar10/ImageGeneration/evaluation.py`
- `rectified_flow_cifar10/ImageGeneration/likelihood.py`
- `rectified_flow_cifar10/ImageGeneration/losses.py`
- `rectified_flow_cifar10/ImageGeneration/main.py`
- `rectified_flow_cifar10/ImageGeneration/models/ddpm.py`
- `rectified_flow_cifar10/ImageGeneration/models/ncsnv2.py`
- `rectified_flow_cifar10/ImageGeneration/pytorch_datasets.py`
- `rectified_flow_cifar10/ImageGeneration/run_lib.py`
- `rectified_flow_cifar10/ImageGeneration/run_lib_pytorch.py`
- `rectified_flow_cifar10/ImageGeneration/run_lib_reflow.py`
- `rectified_flow_cifar10/ImageGeneration/sampling.py`
- `rectified_flow_cifar10/ImageGeneration/utils.py`
- `rectified_flow_modern/runtime/rectified_flow/flow_components/interpolation_convertor.py`
- `rectified_flow_modern/runtime/rectified_flow/metrics/__init__.py`
- `rectified_flow_modern/runtime/rectified_flow/metrics/clip_score.py`
- `rectified_flow_modern/runtime/rectified_flow/models/enhanced_mlp.py`
- `rectified_flow_modern/runtime/rectified_flow/models/flux_dev.py`
- `rectified_flow_modern/runtime/rectified_flow/models/gauss_analytic.py`
- `rectified_flow_modern/runtime/rectified_flow/models/kernel_method.py`
- `rectified_flow_modern/runtime/rectified_flow/models/lightningdit.py`
- `rectified_flow_modern/runtime/rectified_flow/models/lightningdit_utils.py`
- `rectified_flow_modern/runtime/rectified_flow/models/toy_mlp.py`
- `rectified_flow_modern/runtime/rectified_flow/models/utils.py`
- `rectified_flow_modern/runtime/rectified_flow/models/vae.py`
- `rectified_flow_modern/runtime/rectified_flow/pipelines/sample_ldit_imagenet256.py`
- `rectified_flow_modern/runtime/rectified_flow/pipelines/train_dit_cifar.py`
- `rectified_flow_modern/runtime/rectified_flow/pipelines/train_ldit_imagenet256.py`
- `rectified_flow_modern/runtime/rectified_flow/pipelines/train_unet_cifar.py`
