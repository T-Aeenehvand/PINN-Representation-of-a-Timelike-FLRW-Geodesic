# Repository map and provenance

## Evidence hierarchy

1. `results/archived_seed42/six_cell_run/` — saved network parameters and per-cell validation outputs used for the main paper tables.
2. `results/archived_seed42/` — independent audits and reference checks performed without retraining.
3. `results/dimensional_extension/` — fresh supplementary numerical runs.
4. `notebooks/reproducibility/` — corrected/current rerun logic.
5. `notebooks/archive/` — historical notebooks retained for provenance.
6. `src/` and `scripts/` — cleaned reusable utilities and command-line workflows.

## Important provenance rule

A new execution of the cleaned scripts is a **new run**. It is not evidence that the archived checkpoint was produced by the cleaned script unless the run metadata explicitly says so.

## Generated files

Large or machine-generated files belonging to the archived numerical record are intentionally retained under `results/`. New local experiments should use `results/local_runs/` or another ignored output directory.


### Full-budget AFI continuation

`results/afi_full_budget_500/` stores the 500-iteration AFI continuation used in the final manuscript. The 220-iteration source state is retained separately under `results/afi_full_budget_incremental/`.
