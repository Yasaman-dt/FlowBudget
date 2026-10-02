# Original RectifiedFlow CIFAR-10 Euler / FM experiment

This directory contains the official CIFAR-10 Rectified-Flow checkpoints,
experiment outputs, and a local copy of the original model source in
`rectified_flow_cifar10/ImageGeneration`.

The sampling code is the deterministic `euler_sampler` from the original
repository with `sigma_variance=0`: exactly `K` Euler updates and no guidance.

```bash
source /home/ens/Zdehghani/miniforge3/etc/profile.d/conda.sh
conda activate /home/ens/Zdehghani/SiT_env
cd /home/ens/Zdehghani/Sampling_Step_Estimation

python rectified_flow_cifar10/build_cifar10_reference_npz.py \
  --data-root rectified_flow_cifar10/data \
  --output rectified_flow_cifar10/references/cifar10_test_10k.npz

CUDA_VISIBLE_DEVICES=0 python rectified_flow_cifar10/sweep_fid_vs_k.py \
  --steps 1 2 4 8 16 32 64 128 \
  --reference-npz rectified_flow_cifar10/references/cifar10_test_10k.npz \
  --adm-evaluator /home/ens/Zdehghani/guided-diffusion/evaluations/evaluator.py \
  --fid-python /home/ens/Zdehghani/SiT_env/bin/python

CUDA_VISIBLE_DEVICES=0 python rectified_flow_cifar10/estimate_euler_budget_fm.py \
  --data-root rectified_flow_cifar10/data \
  --num-samples 500 --tau 0.05 \
  --output rectified_flow_cifar10/euler_budget_estimate_500.json
```

FID uses the CIFAR-10 **test** set as its fixed 10k real reference. This FID is
not comparable to the earlier SiT/ImageNet FID values.
