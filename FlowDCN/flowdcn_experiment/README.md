# FlowDCN-XL-2M unguided Euler experiment

`run.py` imports the original FlowDCN model and Triton forward kernel directly;
it does not require the Lightning training/metrics launcher. All inference is
float32 with TF32 disabled. Conditional ImageNet labels are retained; CFG is 1
and the unconditional branch is not evaluated. Uniform Euler has K updates and
K model evaluations from t=0 to t=1; EMA VAE scaling is 0.18215.

The estimator uses the same deterministic center-crop preprocessing and ImageNet
training source as SiT. On 2,000 identical probes it compares forward directional
finite differences with h = 0.0005, 0.001, 0.002, along (v,1). Times are sampled
uniformly on [0,0.998) for all h to keep shifted times in range. It records the
raw ratios and dataset indices. The budget is ceil(mean(||a||/(||v||+1e-8))/(2*tau)).
At tau=0.01, the full sweep is gated on (max K - min K)/K(h=0.001) <= 5%.
A failed sensitivity gate requires investigation, not silently selecting a budget.
This agreement check is not a proof of derivative accuracy.

The sweep uses ceil([0.5,0.75,1,1.25,1.5] * K_hat) and 250 (deduplicated).
Each budget uses the same seed and batch size for paired noise/label draws and
10,000 samples. FID uses the existing SiT 10K ImageNet reference and ADM evaluator
on CPU, avoiding TensorFlow/PyTorch cuDNN conflicts. Generated uint8 samples,
preview grids, logs, metrics, estimator statistics, and plans are retained.
The evaluator also computes its usual ancillary metrics; fid_vs_k.csv reports FID.

Run with the existing SiT environment:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/ens/Zdehghani/SiT_env/bin/python -u FlowDCN/flowdcn_experiment/run.py run \
  --output FlowDCN/flowdcn_experiment/fid_sweep_future \
  --probes 2000 --tau 0.01 --images 10000 --batch-size 8 --seed 0
```

Completed estimator and per-budget results are resumable with the same plan.
PNG outputs only. No existing SiT/RF results are modified.

## Three-seed estimation only (current experiment)

The original automatic sweep was stopped after seed 0 completed; sampling at
K=75 had already begun and its partial artifacts are retained. No further
sampling or FID is authorized in the current experiment.
`estimate_three_seeds.py` reuses that completed seed-0 estimate and sequentially
runs only `run.py estimate` for seeds 1 and 2, each at N=2000 and tau=0.01 with
identical finite-difference settings. Results are saved under
`FlowDCN/results/estimates/estimation_ode_euler/`, including `per_seed.csv` and `summary.json`
(mean and sample standard deviation). Sensitivity failures are explicitly
flagged in the summary; they never trigger sampling.

## Consolidated result layout

All completed estimation results are now in `FlowDCN/results/estimates/estimation_ode_euler/`:

- `seed_0/`, `seed_1/`, `seed_2/`: unchanged original estimates and logs.
- `per_seed.csv`: three-seed comparison, with current source paths.
- `summary.json`: mean and sample standard deviation.

The former `results_gpu2_tau001_n2000/` directory was moved intact to
`FlowDCN/results/estimates/estimation_ode_euler/seed_0/`. Its interrupted `k_0075/` sampling
artifacts and original sweep plans are retained as historical files; they are
not part of the estimation results. Historical logs/plans are unedited.
Seed estimates are 150, 148, and 152 (mean 150, sample SD 2); all sensitivity
checks passed. No experiment was rerun during consolidation.

The experiment folder now lives inside `FlowDCN/`. Commands above are run from
`Sampling_Step_Estimation/`. Saved historical plans and logs retain their original
paths; aggregate result source links point to the current location.

## FlowDCN Euler–Maruyama: three seeds, estimation only

`estimate_em_three_seeds.py` uses the leading-order EM formula in the pasted
methodology, not the augmented RMS proxy in the existing SiT EM code.
For the native FlowDCN `sde_preserve_step_fn` and linear schedule:

- score = (t*v - x)/(1-t), w = 1-t;
- drift f = (1+t/2)*v - x/2;
- G = sqrt(1-t)*I, G' = -I/(2*sqrt(1-t)).

Eight matched Rademacher directions estimate q² = ||Jf G||²_F + ||G'||²_F
+ <Jf G,G'>_F. Central directional finite differences use increments
0.0005, 0.001, 0.002. Seeds 0/1/2 each use 2,000 probes, t uniform on
[0.01,0.99), tau=0.01, epsilon=1e-8 and CFG=1. The selected increment is
0.001; budget spread above 5% is flagged. K_hat=ceil(mean(q/(||G||_F+epsilon))
/(sqrt(3)*tau)). The finite-direction square root introduces Monte Carlo bias.
These are empirical estimates with unverified global smoothness assumptions.

Results are under `FlowDCN/results/estimates/estimation_sde_em/seed_{0,1,2}/`
(relative to the repository root), with aggregate
`per_seed.csv` and `summary.json`. No sampling or FID is run. The estimate is for
uniform EM; it does not include the native sampler's fixed final 0.04 Euler step.
Future sampling must explicitly match this convention. Existing ODE results
are untouched. Analytic linear-field tests verify drift algebra and the EM
coefficient, including the A/B cross term.

## Sampling at a chosen budget

Use the general sampler from the repository root. Choose `euler`, `heun`, or
`em`, and provide the desired step count, seed, and output directory:

```bash
python FlowDCN/flowdcn_experiment/run.py sample --solver em \
  --k 178 --images 10000 --batch-size 8 --seed 0 \
  --output results/flowdcn_em
```

This creates `results/flowdcn_em/k_0178/images.npz`. Evaluate it with the ADM
evaluator using absolute paths for both the reference and generated NPZ files:

```bash
CUDA_VISIBLE_DEVICES="" python "$FLOWBUDGET_EVALUATOR" \
  "$FLOWBUDGET_IMAGENET_REFERENCE" "$PWD/results/flowdcn_em/k_0178/images.npz"
```

The old fixed-budget queue launchers have been removed. Existing samples and
metrics are preserved. For uniform Euler/EM, NFE=K; for Heun, NFE=2K.
The `run` mode remains the Euler-estimation-driven workflow; use `sample`
for an explicit solver and budget.
