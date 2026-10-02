# Unguided Euler / FM experiment

This workflow uses the original PyTorch SiT repository. It intentionally does
**not** use classifier-free guidance: `cfg_scale=1.0`. It evaluates the
deterministic ODE with fixed-step first-order Euler, which is the solver covered
by `FM.pdf`; it is not the paper's SDE/Heun headline sampler.

Install the original dependencies first (in particular `torchdiffeq`):

```bash
conda env create -f environment.yml
conda activate SiT
```

The official README delegates FID to the ADM evaluator. Obtain that evaluator
from OpenAI's `guided-diffusion/evaluations/evaluator.py`, then make one fixed
10k ImageNet reference batch. `--data-path` must be the directory that contains
the 1,000 ImageNet training class folders.

```bash
python build_imagenet_reference_npz.py \
  --data-path /home/ens/Zdehghani/datasets/imagenet_2012/images/train \
  --num-images 10000 --seed 0 \
  --output references/imagenet_train_10k_center_crop_256.npz
```

Run the unguided sweep. `K` means the number of Euler **updates**, so the code
passes `K+1` time points to the original integrator. Supply `--checkpoint` only
when using a local official SiT-XL/2 checkpoint; omitting it uses the original
auto-download path.

```bash
CUDA_VISIBLE_DEVICES=0 python sweep_fid_vs_k_unguided.py \
  --steps 1 2 4 8 16 32 64 128 250 \
  --num-samples 10000 --gpus 1 --batch-size 8 \
  --checkpoint SiT-XL-2-256.pt \
  --reference-npz references/imagenet_train_10k_center_crop_256.npz \
  --adm-evaluator /path/to/guided-diffusion/evaluations/evaluator.py \
  --fid-python /path/to/python-with-tensorflow
```

The sweep evaluates FID on CPU by default. This does not change the SiT GPU
sampling and avoids TensorFlow/cuDNN binary mismatches. Add `--fid-device gpu`
only when TensorFlow's required cuDNN version is available.

The results are stored in `results/fid/fid_sweep_unguided_10k/fid_vs_k.csv`. These are
FID-10K comparisons, not the paper's FID-50K number.
Each `k_XXX/images/` directory also contains `image_grid.png`, an 8x8
qualitative preview of its first 64 generated images.

Estimate the FM budget on ImageNet training images, with the same checkpoint and
VAE setting used by the sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python estimate_euler_budget_fm_unguided.py \
  --data-path /home/ens/Zdehghani/datasets/imagenet_2012/images/train \
  --checkpoint SiT-XL-2-256.pt \
  --num-samples 100 --batch-size 1 --tau 0.05 \
  --output euler_budget_estimate_unguided.json

python compare_euler_budget.py \
  --estimate euler_budget_estimate_unguided.json \
  --fid-csv results/fid/fid_sweep_unguided_10k/fid_vs_k.csv
```

`K_hat = ceil(R_theta / tau)` is a heuristic FM discretization budget. Compare
it with the empirical sweep, but do not claim it predicts the best FID exactly.
