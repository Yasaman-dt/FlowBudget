# Rectified Flow CIFAR-10 Euler sampling

This directory contains a self-contained copy of the runtime code and the
official pretrained unconditional CIFAR-10 SongUNet checkpoint linked by
`lqiang67/rectified-flow`. Training is not required. After every completed K,
the runner automatically creates `images.npz`, calculates ADM FID, and updates
`fid_vs_k.csv`.

Run the full sampling sweep with:

```bash
CUDA_VISIBLE_DEVICES=3 /home/ens/Zdehghani/SiT_env/bin/python \
  sample_unet_euler_sweep.py \
  --steps 1 2 4 8 16 32 64 128 250 \
  --num-samples 10000 \
  --batch-size 128 \
  --seed 0 \
  --output-dir results/fid/fid_sweep_unet_euler_cifar10_10k
```

For Euler integration, `K` is both the number of updates and the NFE. The
time grid contains `K + 1` points. The seed is reset for every K so all sweep
points use the same initial noise samples.

Contents:

- `runtime/rectified_flow`: required Python package
- `checkpoints/unet_cifar10_pretrained`: pretrained model and configuration
- `sample_unet_euler_sweep.py`: sampling-only sweep runner
- `requirements.txt`: upstream dependencies
- `LICENSE`: upstream license

`sample_euler_sweep.py` is the DiT variant and requires a separately trained
DiT checkpoint; the upstream repository does not publish one.
