## Current estimator: manuscript leading-order formula

`estimate_em_budget.py` now uses
`K_hat = ceil(mean(sqrt(q2)/(sqrt(g2)+epsilon))/(sqrt(3)*tau))`,
with a minimum of one step. It does not use the older K-dependent augmented
proxy described in the historical notes below. New results carry the identifier
`SiT-Linear-EM-leading-order-v2`. Random-direction JVPs approximate q2; finite
trace-probe counts and their square root remain numerical approximations.

`recompute_em_leading_order.py` reuses the unchanged original 2,000-probe pools
for seeds 0,1,2. New results are in
`../probe_sensitivity_forward/sde_2000_tau001_leading_order/`:
K_hat = 316, 316, 319; mean 317 and sample SD 1.73205 at tau=0.01.
Original results and sampler code are unchanged. The original probe-time law
[0,0.99) is retained; coefficients are nonsingular at t=0 for this SiT SDE.
The current sampler still makes two network calls per step, despite the fused
mathematical drift requiring only one. No FID experiment was launched.

## Historical augmented estimator documentation

# SiT Euler–Maruyama budget estimator

`estimate_em_budget.py` implements the supplied augmented RMS local-error proxy
for the Linear SiT path, unguided velocity model, `diffusion-form=sigma`, and
`last-step=None`. It does not estimate FID or change the sampler.

The code uses sampling time s=1-t, with noise at s=0 and data at s=1:

- d(s)=diffusion_norm*(1-s), g(s)=sqrt(2*d(s)).
- F=(1+diffusion_norm*s)*v-diffusion_norm*x, obtained by substituting the score.
- A=J_x(F)*g, B=g'(s)*I.
- D=partial_s(F)+J_x(F)*F+0.5*g^2*Laplacian(F).
- q^2=||A||_F^2+||B||_F^2+<A,B>_F.
- E(K)=mean[sqrt(q^2/3+D^2/(4K))/(sqrt(G^2+F^2/K)+epsilon)]/K.

Here D^2, G^2, and F^2 in the last expression denote squared norms. Every integer
K starting at 1 is checked, returning the first E(K)<=tau. An unmet search bound
produces a null K and an explicit status, not a fabricated recommendation.

Run on GPU 1, using the same EMA VAE as the EM sampling sweep:

```bash
cd /home/ens/Zdehghani/Sampling_Step_Estimation/SiT_Imagenet
CUDA_VISIBLE_DEVICES=1 /home/ens/Zdehghani/SiT_env/bin/python estimate_em_budget.py \
  --data-path /home/ens/Zdehghani/datasets/imagenet_2012/images/train \
  --num-samples 500 --batch-size 1 --trace-probes 8 --vae ema \
  --t-min 0.01 --diffusion-norm 1.0 \
  --taus 0.010 0.015 0.020 0.025 0.030 \
  --output em_budget_estimates/estimate_500.json
```

Supply `--checkpoint /path/to/checkpoint.pt` to select a local checkpoint;
otherwise the existing loader selects the official SiT-XL/2 checkpoint.
For a preliminary model run use `--num-samples 2 --trace-probes 2` and a separate
output filename. Full XL/2 nested derivatives are expensive; batch size 1 is
intentional.

Recompute tolerances from saved probe statistics without loading the model:

```bash
/home/ens/Zdehghani/SiT_env/bin/python estimate_em_budget.py \
  --from-statistics em_budget_estimates/estimate_500.json \
  --taus 0.005 0.01 0.02 \
  --output em_budget_estimates/tau_sweep.json
```

The JSON contains all per-probe scalar statistics and experiment metadata. Its
companion CSV contains budgets and errors at those budgets. `integrator_time_points`
is K+1. `NFE_hat_ideal` is K; the legacy field `NFE_hat_current_sampler` is 2K
for the original transport implementation, which separately evaluated drift
velocity and score.

The historical `results/fid/fid_sweep_em_10k` images were sampled before the velocity
drift was fused. Its CSV `nfe` column reports the nonredundant
velocity-evaluation budget K used for comparisons; each run's `metrics.json`
preserves the two model calls actually made per Euler-Maruyama update as
`nfe_executed` (for example, K=250 has CSV `nfe=250` and metadata
`nfe_executed=500`). The current `transport/transport.py` computes both
drift terms from one velocity call, so new samples have both counts equal K.
State which count is used when reporting measured runtime or FLOPs.

Rademacher probes estimate Jacobian norms/cross terms and the vector Hessian
trace. Averaging the trace before squaring still gives a finite-probe bias in
||D_hat||^2; compare larger `--trace-probes` and different seeds. Probe times
exclude the data endpoint using manuscript t>=t_min (code s<=1-t_min), but
sampling steps still span the unit interval. The sigma schedule has singular
g' at the data endpoint, so examine cutoff sensitivity too. This is an
interpolation-averaged local proxy, not an endpoint/global-error or FID guarantee.

Validation: `python -m unittest test_estimate_em_budget.py` checks nonlinear
analytic derivatives, fused drift equivalence, the zero-diffusion Euler limit,
and integer search boundary handling.
