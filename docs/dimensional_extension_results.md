# Fresh Cartesian dimensional-extension results

These are supplementary V12 runs, separate from the archived six-cell seed-42 ablation.

Common settings: matter background `a(eta)=eta^2`, `kappa=1`, `eta0=0.5`, fixed training interval `eta=[0.5,1.5]`, affine reporting interval `lambda=[0,0.8]`, `||P||=1`, float64 CPU, seed 42, 5 tanh hidden layers of width 64, 1250 Adam updates, final 514-point deterministic Chebyshev grid, requested L-BFGS cap 120. PyTorch recorded one realized L-BFGS iteration and two function evaluations in each run.

| d | P vector | rel L2 eta(lambda) | rel L2 vector x(lambda) | max momentum-vector defect | max tangent-normalization defect | max transverse displacement | lambda_pred(eta=1.5) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `(1)` | 1.4744e-4 | 2.7513e-4 | 1.3304e-2 | 5.2879e-2 | 0 | 0.8250629 |
| 2 | `(0.8,0.6)` | 1.3044e-4 | 5.7659e-4 | 1.1049e-2 | 7.5589e-2 | 8.1566e-5 | 0.8250895 |
| 3 | `(0.60,0.48,0.64)` | 1.1858e-4 | 1.1099e-3 | 1.0179e-2 | 1.1253e-1 | 6.9634e-5 | 0.8251191 |

Interpretation: affine-clock reconstruction is stable around `1e-4` across dimensions; the spatial-vector error grows as output dimension increases; transverse drift remains very small; the derivative-sensitive tangent normalization worsens. Because the exact Cartesian path is collinear with the fixed momentum vector, these tests do not establish performance on a genuinely coupled multidimensional geodesic system.
