# GPU 3 cost comparison

Run the seven pairs sequentially with the existing SiT environment:

```bash
/home/ens/Zdehghani/SiT_env/bin/python \
  /home/ens/Zdehghani/Sampling_Step_Estimation/cost_benchmark/run_sequential.py \
  --gpu 3 --estimator-repeats 3 --sweep-repeats 3 \
  --output /home/ens/Zdehghani/Sampling_Step_Estimation/cost_benchmark/results_gpu3
```

`--plan-only` saves the exact grid, K/NFE mapping and estimation configuration
without running models. Completed timing records can be resumed. Incomplete
stages deliberately stop for inspection rather than timing cached images.

## Measurement scopes

A. `estimator_only_seconds`: starts before selecting N=500 data indices and
constructing the loader, after checkpoint/VAE/dataset setup. Includes loader
iteration, host-to-device transfers, VAE encoding for SiT, noise/time generation,
interpolation, model evaluations, JVP/nested-JVP computation (and Hessian trace
probes for EM), norm ratios, mean complexity and integer budget selection.
CUDA is synchronized at both boundaries. SDE's safety checkpoint of raw probes
before its integer budget search is also included.

B. selection workflow: parent subprocess wall clock, including interpreter and
imports, warm-up, model/VAE/dataset loading, estimation and output serialization.
The worker warms up by running two probes, releases that instance, then reloads
for the measured run within the same process. Thus B includes the extra warm-up
initialization; A excludes it. Sampling workers analogously warm up a two-step
batch before their measured generation. These are **instrumented workflow**
costs, not an optimized production cold-start benchmark. Warm-up and steady-state
script timings are separately saved so this overhead is explicit.

Full empirical sweep: sum of generation and complete evaluation process times
at EVERY measured allowed grid budget. Includes warm-up, process/model loading,
10,000 new images per budget, solver updates, VAE decoding, image/NPZ/grid I/O,
reference and sample Inception features/statistics, FID, sFID, IS and precision/
recall, as executed by the existing evaluator. CPU evaluation remains CPU for
RF/CIFAR and SiT ODE; EM evaluation uses GPU 3 as in evaluate_em_fid.py.
Feature extraction, statistics and metric phases are also timed in the worker.
Model derivatives are autograd for all estimator pairs, including RF-UNet:
this intentionally overrides its current finite-difference default. Never label
these RF-UNet timings as finite-difference timings.

End-to-end FlowBudget = B + generation and the same evaluator at the predicted
budget. The paper table compares this with the full sweep; estimator-only A
is reported separately. No cached samples, reused FID features, or extrapolated
full-sweep times are substituted. The predicted budget comes from the new
estimator run (not hard-coded from an older estimate).

The sweep means include process and warm-up overhead at every budget. They do
not include queue waiting or post-measurement scratch cleanup. This mirrors
separate per-budget invocation, not a single permanently resident sampler.
Do not describe this as a lower bound on the cost of an optimized sweep.

## Repeats and interference

Three estimator runs and three full sweeps use seed 0 to hold workload constant.
They measure runtime variation, not seed uncertainty. A single blocking queue
waits for each sampler AND evaluator before advancing. GPU-idle checks occur
before every stage. A 30-second monitor invalidates a run if a foreign GPU
process is observed. A file lock prevents two copies of this benchmark using
the same GPU. It cannot prevent unrelated users starting GPU jobs, nor detect
all short-lived interference. Other GPU jobs may share CPU, filesystem, power,
and memory-bandwidth resources: sequential execution is not complete machine
isolation. Fixed CPU thread limits are recorded in the runner. For definitive
paper timing, reserve GPU 3 and a quiet host CPU allocation during the run.

## Outputs

- plan.json: N, tau, exact K and actual implementation NFE budgets, repeat counts.
- results.csv/json: completed estimator and per-budget timing records.
- pair_*/.../timing.json: GPU, torch version, component timing, raw call counts.
- pair_*/.../run.log: complete original sampler/evaluator output including FID.
- paper_table.md: final mean cost comparison after all runs complete.

Forward counters count recognized model module calls. Direct `.forward` calls
in SiT sampling bypass module hooks, so zero observed calls is NOT zero compute.
Use the grid's implementation NFE times generated image count for sampling,
and report JVP counts separately; automatic differentiation is not assigned an
invented forward-equivalent NFE. EM currently has 2*K implementation calls per
image. Estimation uses probes N=500; this differs from FID's 10,000 images.

New benchmark image artifacts are removed after successful evaluation to bound
disk use; source experiment images and all benchmark logs/timing JSON remain.
The two-probe smoke results are validation only, not paper benchmark numbers.
