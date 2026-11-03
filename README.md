# FLRW Geodesic PINN — Physics × Machine Learning Research Resource

Reproducibility repository for:

> **Physics-Informed Neural Representation of a Timelike FLRW Geodesic: Conformal First Integrals and Affine Reconstruction**

Tahereh Aeenehvand

## What is in this repository?

This **standalone code-and-data resource** accompanies the article. It contains reusable source code, saved numerical evidence behind the six-cell comparison, post-training audits, figure-generation scripts, manuscript figure assets, and supplementary Cartesian `d=1,2,3` extension records. **Manuscript LaTeX and PDF files are distributed separately** and are not in this resource. This is a crossover between computational gravitational physics, numerical analysis, and physics-informed machine learning; it is not the announcement of a formally established new scientific discipline.

The repository deliberately distinguishes **archived evidence** from **code intended for future reruns**. The saved seed-42 checkpoints reported in the paper are not silently regenerated or replaced by a cleaned implementation.

### Repository layout

```text
FLRW_Physics_AI_Research_Resource/
├── src/flrw_geodesic_pinn/        # Reusable physics and ML modules
├── scripts/                        # Training, audit and plotting commands
├── configs/                        # Experiment configurations
├── notebooks/
│   ├── reproducibility/            # Current notebook plus Python export
│   └── archive/                    # Historical notebooks; provenance
├── results/                        # Original checkpoints, arrays and logs
│   ├── archived_seed42/six_cell_run/
│   ├── dimensional_extension/
│   ├── afi_full_budget_500/
│   └── ...                         # Further representation/robustness tests
├── figures/                        # Existing figure assets; script output
├── tests/                          # Lightweight physics tests
├── docs/                           # Methods, provenance and notes
├── UPLOAD_GUIDE.md                 # Upload metadata and rights check
├── verify_release.py               # Offline SHA-256 verification
├── SHA256SUMS.txt                   # Checksums relative to this root
├── CITATION.cff                     # Author credit; DOI not assigned
├── pyproject.toml
├── Makefile
└── requirements.txt
```

## Archived evidence versus new runs

### Archived six-cell comparison

The main paper comparison is the seed-42 run under:

`results/archived_seed42/six_cell_run/`

It contains six checkpoints:

- affine second order + deterministic grid
- affine second order + RAR
- affine scaled/preconditioned + deterministic grid
- affine scaled/preconditioned + RAR
- conformal first integral + deterministic grid
- conformal first integral + RAR

The surrounding files in `results/archived_seed42/` contain the high-resolution checkpoint audit, reference-tolerance audit, PCHIP-density audit, second-order balance diagnostics, and console records.

### Supplementary dimensional extension

`results/dimensional_extension/` contains the fresh `d=1,2,3` conformal runs. These are an implementation/scaling check on the same FLRW symmetry, not a new coupled multidimensional geodesic benchmark.

## Representation-control and robustness records

The original six-cell seed-42 archive is retained. The release also contains the affine first-integral and conformal second-order controls, the direct quadrature baseline, the five-seed Adam-only robustness screen, and a full-budget AFI continuation to 500 L-BFGS iterations. These records are stored under `results/upgraded_full_controls/`, `results/seed_sweep_adam/`, and `results/afi_full_budget_500/`.

The five-seed screen uses seeds 42--46, deterministic Chebyshev collocation, and 1250 Adam updates (250 per stage for five stages). It is a robustness screen rather than a replacement for the archived 7000-update + L-BFGS protocol. The short-control seed-42 runs use the same five-stage Adam budget followed by the recorded L-BFGS phase. The separate full-budget AFI continuation uses 7000 Adam updates and resumes the saved AFI optimizer state from 220 to 500 total L-BFGS iterations.

Regenerate the new publication figures with:

```bash
PYTHONPATH=src python scripts/regenerate_upgraded_figures.py
```

The direct quadrature reference is recorded in `results/upgraded_full_controls/quadrature_baseline.csv`. The experiment design and interpretation are documented in `docs/upgraded_experiment.md`.

## Environment

The archived environment was:

- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- Matplotlib 3.10.8
- PyTorch 2.10.0+cpu

Install the project environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The exact archived environment record is retained as `docs/environment_versions_archived.txt`.

## Local configuration

No API keys or credentials are required. Copy the template if desired:

```bash
cp .env.example .env
```

The real `.env` is intentionally ignored by Git. **Do not commit secrets or machine-specific credentials.**

## Run the lightweight tests

```bash
make test
```

## Re-run the main ablation

The notebook and script export reproduce the experiment logic. A smoke profile is available for checking a fresh installation without running the full paper budget:

```bash
FLRW_PROFILE=smoke FLRW_RESULTS_ROOT=results/local_runs PYTHONPATH=src \
  python scripts/run_main_ablation.py
```

The full paper profile is substantially more expensive:

```bash
FLRW_PROFILE=paper FLRW_RESULTS_ROOT=results/new_runs PYTHONPATH=src \
  python scripts/run_main_ablation.py
```

A new run is written to a new timestamped directory. It must not be confused with `results/archived_seed42/`.

## Regenerate the publication figures

```bash
PYTHONPATH=src python scripts/regenerate_figures.py
python scripts/plot_dimensional_extension.py \
  --results results/dimensional_extension \
  --out figures/dimensional_extension
```

## Run the supplementary dimensional extension

```bash
python scripts/train_dimensional_extension.py \
  --dims 1,2,3 \
  --outdir results/new_dimensional_extension
```

The default script settings follow the supplementary protocol described in the manuscript. New outputs are deliberately directed to a separate directory.

## Audit the archived checkpoints

The audit scripts do **not** retrain the networks. They reload saved checkpoints and recompute diagnostics:

```bash
python scripts/audit_saved_checkpoints.py
python scripts/posttraining_diagnostics.py
python scripts/reference_tolerance_audit.py
```

## Scope of this release

The source and numerical records are provided as the **research-code/resource deposit**, not as the arXiv LaTeX submission. For arXiv submit the separate, clean manuscript source archive, not this resource ZIP. The included `figures/` assets are for evidence and regeneration. The preserved `notebooks/archive/` notebooks are historical, not a guarantee of reproducible bit-for-bit reruns.

Run `python verify_release.py` from the extracted root to check every file against the supplied manifest. The manifest does not include its own digest. PyTorch `.pt` files should only be opened if you trust this repository and its provenance.

## Reproducibility policy

The paper's reported six-cell values are tied to the archived checkpoints and their recorded configurations. Corrected validation logic, cleaner source organization, or a future software environment does not retroactively change those results. Any new training run should be labeled with its seed, configuration, software environment, realized optimizer counters, and output directory.

## Publication release

For the final paper release, create:

1. a public GitHub repository for the readable source;
2. an immutable Zenodo (or equivalent) archive containing the exact paper release and numerical evidence; and
3. a Git tag matching the archived DOI release.

After those exist, update `CITATION.cff` and the separate manuscript's Data and code availability section with actual repository URL and DOI. No DOI is fabricated in this resource.

## License

**No public software license has been selected in the supplied materials.** The authors must agree on the sharing/reuse license (or restricted-access terms) before depositing this archive. The presence of a ZIP file does not imply permission to redistribute or modify the code. See `UPLOAD_GUIDE.md`.

## Additional controls used in the manuscript

The current manuscript includes two supplementary control records beyond the original six-cell seed-42 baseline:

- `results/afi_full_budget/` contains the original seed-42 affine first-integral control at 7000 Adam updates and a 220-iteration L-BFGS cap (220 iterations, 239 function evaluations).
- `results/afi_full_budget_500/` contains the continuation of the same AFI checkpoint to 500 total L-BFGS iterations (527 total function evaluations), giving relative affine-grid errors of $3.72\times10^{-3}$ for $\eta(\lambda)$ and $6.54\times10^{-2}$ for $x(\lambda)$. The continuation changes no Adam parameters or collocation protocol.
- `results/seed_sweep_adam_padded_cfi/` contains the five-seed CFI Adam-only screen with the conformal training interval padded to `eta_max = 1.5`, allowing all five runs to be reconstructed on the common affine reporting interval.

The separate manuscript reports affine-grid values from these controls; native CFI errors are retained alongside them.
