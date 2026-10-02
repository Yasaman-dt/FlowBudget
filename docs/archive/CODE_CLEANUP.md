# Code-only release cleanup

Historical record of the cleanup at that time. For current commands and
validation status, see the [usage guide](../USAGE.md) and [README](../../README.md).

The root tolerance-sweep utilities are consolidated into `scripts/analysis/sweep_khat_vs_tau.py`:

```bash
python scripts/analysis/sweep_khat_vs_tau.py --solver euler --estimates path/to/euler.json --taus 0.01 0.02 --output results/euler_tau.json
python scripts/analysis/sweep_khat_vs_tau.py --solver heun --estimates path/to/heun.json --taus 0.01 0.02 --output results/heun_tau.json
```

The Euler default and both solvers' output field names are preserved. The former
root `sweep_heun_khat_vs_tau.py` is removed; use `--solver heun` instead.
Model-specific sweep scripts are retained because their command interfaces differ.

The copied audit shell file is removed: `scripts/paper/audit_final_paper.py`
regenerates commands from the input specification. Local audit reports and
provenance JSONs are excluded from Git, along with remaining smoke-run metadata
and compiled-extension caches. Saved experimental results are preserved.

Legacy JSON inputs are retained under `configs/legacy/`: their settings are
different. Empty `__init__.py` files remain because they define Python packages.
Vendored evaluator copies remain alongside their model workflows to preserve
standalone execution and upstream attribution.

## Historical launchers removed

The GPU-specific follow-up and fixed-budget queue scripts were removed after
checking that no remaining Python or shell code imports them. Their underlying
operations remain available through these general entry points:

| Model | General replacement |
| --- | --- |
| Original RF | `rectified_flow_cifar10/sweep_fid_vs_k.py --steps ...` |
| RF-UNet | `rectified_flow_modern/sample_unet_euler_sweep.py --steps ...` |
| SiT | `SiT_Imagenet/sample_ddp.py` with solver, step count, and seed arguments; ADM evaluation is separate |
| FlowDCN | `FlowDCN/flowdcn_experiment/run.py sample --solver ... --k ...`; ADM evaluation is separate |

These commands replace model sampling and evaluation. The historical job-waiting
and fixed-budget queue behavior is intentionally removed. Saved results remain
in their existing folders.

## Reduced upstream runtime copies

Unused upstream training code, unrelated models, and alternative kernels were
removed from the three RF/FlowDCN directories. The exact removal list and retained
compatibility modules are in [the dependency review](DEPENDENCY_REVIEW.md).
The FlowDCN experiment runner replaces the upstream Lightning CLI in this copy.
