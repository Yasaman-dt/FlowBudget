# Estimator and solver conventions

The code implements manuscript equations (12), (13), and (17):

| Solver | Coefficient C | Step budget K | NFE |
| --- | --- | --- | --- |
| Euler | mean of `norm(a)/(norm(v)+epsilon)` | `ceil(C/(2*tau))` | K |
| Heun | mean of `norm(2*g-c)/(norm(v)+epsilon)` | `ceil(sqrt(C/(12*tau)))` | 2K |
| Additive-noise EM | mean of `sqrt(norm(A)^2+norm(B)^2+inner(A,B))/(norm(G)+epsilon)` | `ceil(C/(sqrt(3)*tau))` | K |

All sampling budgets have a minimum of one step, including a constant field
whose leading error coefficient is zero. Coefficient averaging precedes the
budget transformation and integer ceiling.

For ODEs, `a = partial_t(v) + J(v)*v`, `g = J(v)*a`, and
`c = D²v[(v,1),(v,1)]`. The direction `(v,1)` must be held fixed when
computing `c`; differentiating acceleration instead yields an extra `g` term.
SiT computes these derivatives with JVPs; FlowDCN uses centered finite
differences for Heun because its custom kernels constrain automatic derivatives.

The manuscript uses data-to-noise time, with generation proceeding backward.
Native model code uses noise-to-data sampling time. Under `s=1-t`, velocity
changes sign; the norm-based error coefficients retain the corresponding
budget laws. Native sampler conventions are retained.

## EM diffusion normalization

In forward sampling time, the manuscript has `G=sqrt(w)*I` and drift
`v + (w/2)*score`. SiT's transport API calls `d=w/2` the diffusion coefficient
and applies noise `sqrt(2*d*dt)`.

- SiT sigma schedule: `d=norm*(1-s)`, hence `G=sqrt(2*norm*(1-s))*I`
  and fused drift `(1+norm*s)*v - norm*x`.
- FlowDCN preserve schedule: `w=1-s`, hence `G=sqrt(1-s)*I`
  and fused drift `(1+s/2)*v - x/2`.

These are distinct SDE schedules, both covered by the paper's general formula.
The estimator includes the `A/B` cross term using matched random directions.
Uniform unguided velocity EM reuses one model evaluation for drift and score.
`NFE_hat_current_sampler` now reports K; older saved reports with 2K describe
historical implementations and are not rewritten. CFG or a special terminal
step requires separate counting and is outside this K-call convention.

## Validation and limits

CPU analytic tests exercise the production Heun derivative functions on a
nonlinear time-dependent field, local error in either integration direction,
Euler/Heun solver updates and call counts, EM noise scaling, and the actual SiT
uniform EM sampler's model-call count. Existing EM tests check the drift
algebra, cross term, and rejection of the zero-diffusion limit.

```bash
python -m unittest tests.test_solver_formulas
python -m unittest discover -s SiT_Imagenet -p 'test_estimate_em_budget.py'
python -m unittest discover -s FlowDCN/flowdcn_experiment -p 'test_em.py'
```

These tests validate formulas and implementation on analytic fields. They do
not establish finite-difference accuracy on pretrained FlowDCN weights. Its
saved Heun sensitivity failure still needs a model-level convergence study;
no derivative increment was changed merely to obtain a preferred budget.
