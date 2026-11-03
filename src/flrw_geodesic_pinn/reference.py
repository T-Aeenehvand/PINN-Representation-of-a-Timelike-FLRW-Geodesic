"""Independent reference integrations for the benchmark."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from .physics import initial_affine_state, conformal_targets


def raw_system(lam, y, A=1.0):
    eta, x, eta_p, x_p = y
    Hc = 2.0 / eta
    return [eta_p, x_p, -Hc * (eta_p**2 + x_p**2), -2.0 * Hc * eta_p * x_p]


def solve_affine_reference(eta0=0.5, x0=0.0, p=1.0, kappa=1.0,
                           lambda0=0.0, lambda_max=0.8, A=1.0,
                           rtol=1e-11, atol=1e-11):
    eta_p0, x_p0 = initial_affine_state(eta0, p, kappa, A)
    y0 = [eta0, x0, eta_p0, x_p0]
    return solve_ivp(
        lambda lam, y: raw_system(lam, y, A),
        (lambda0, lambda_max), y0, method="DOP853",
        rtol=rtol, atol=atol, dense_output=True,
    )


def reduced_system(eta, y, p=1.0, kappa=1.0, A=1.0):
    x, lam = y
    dx, dl = conformal_targets(eta, p, kappa, A)
    return [dx, dl]


def solve_reduced_reference(eta0, eta_max, x0=0.0, lambda0=0.0,
                            p=1.0, kappa=1.0, A=1.0,
                            rtol=1e-12, atol=1e-12):
    return solve_ivp(
        lambda eta, y: reduced_system(eta, y, p, kappa, A),
        (eta0, eta_max), [x0, lambda0], method="DOP853",
        rtol=rtol, atol=atol, dense_output=True,
    )


def build_reference_bundle(eta0=0.5, x0=0.0, p=1.0, kappa=1.0,
                           lambda0=0.0, lambda_max=0.8, A=1.0,
                           n=4000):
    raw = solve_affine_reference(eta0, x0, p, kappa, lambda0, lambda_max, A)
    lam_dense = np.linspace(lambda0, lambda_max, max(n, 5000))
    vals = raw.sol(lam_dense)
    eta_raw, x_raw = vals[0], vals[1]
    eta_max = float(eta_raw[-1])
    eta_grid = np.linspace(eta0, eta_max, n)

    reduced = solve_reduced_reference(eta0, eta_max, x0, lambda0, p, kappa, A)
    x_red, lam_red = reduced.sol(eta_grid)

    order = np.argsort(eta_raw)
    eta_sorted = eta_raw[order]
    x_sorted = x_raw[order]
    lam_sorted = lam_dense[order]
    eta_unique, idx = np.unique(eta_sorted, return_index=True)
    x_of_eta = PchipInterpolator(eta_unique, x_sorted[idx])
    lambda_of_eta = PchipInterpolator(eta_unique, lam_sorted[idx])

    return {
        "raw": raw,
        "reduced": reduced,
        "eta_grid": eta_grid,
        "x_of_eta": x_of_eta,
        "lambda_of_eta": lambda_of_eta,
        "x_reduced": x_red,
        "lambda_reduced": lam_red,
        "raw_reduced_x_max_abs": float(np.max(np.abs(x_of_eta(eta_grid) - x_red))),
        "raw_reduced_lambda_max_abs": float(np.max(np.abs(lambda_of_eta(eta_grid) - lam_red))),
    }
