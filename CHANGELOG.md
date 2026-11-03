# Changelog

## 1.0.0 — PRD-oriented manuscript release

- Added the 7000-Adam, 500-iteration L-BFGS AFI continuation and its reproducibility records.
- Updated the main article, abstract, tables, and publication figures with the continuation results and proper-time diagnostic.
- Clarified native versus reconstructed CFI errors, reference-assisted conformal-domain selection, optimizer-state recreation, RAR channel scaling, affine-grid weighting, and the scope of the multi-seed evidence.
- Corrected the trajectory-error plotting floor and Appendix B derivative notation.
- Updated the Supplemental Material and repository documentation.

## 0.1.0 — repository preparation

- Reorganized the project into a publication-oriented GitHub structure.
- Added reusable `src/flrw_geodesic_pinn` modules for background physics, references, models, sampling, training, and diagnostics.
- Added machine-readable experiment configurations.
- Preserved archived seed-42 checkpoints and audits separately from new-run locations.
- Added smoke-test support and basic source-level tests.
- Added GitHub Actions for tests and LaTeX compilation.
- Revised the manuscript prose and Data and code availability section while preserving the reported numerical values and scope.


## Upgraded representation controls

- Added the affine first-integral and conformal second-order controls.
- Added a direct trapezoid + PCHIP quadrature baseline.
- Added a five-seed Adam-only robustness screen with seeds 42--46.
- Updated the manuscript and publication figures to distinguish archived seed-42 evidence from fresh controls and robustness runs.
