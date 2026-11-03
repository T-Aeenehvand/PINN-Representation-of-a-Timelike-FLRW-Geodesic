"""Analytic FLRW background and first-integral relations used by the paper."""
from __future__ import annotations

import numpy as np


def scale_factor(eta, A: float = 1.0):
    """Matter-era scale factor a(eta)=A eta^2."""
    return A * np.asarray(eta, dtype=float) ** 2


def initial_affine_state(eta0: float = 0.5, p: float = 1.0, kappa: float = 1.0, A: float = 1.0):
    """Return (deta/dlambda, dx/dlambda) at eta0."""
    a0 = A * eta0**2
    return np.sqrt(p**2 + kappa * a0**2) / a0**2, p / a0**2


def conformal_targets(eta, p: float = 1.0, kappa: float = 1.0, A: float = 1.0):
    """Return exact first-integral targets (dx/deta, dlambda/deta)."""
    a = scale_factor(eta, A)
    den = np.sqrt(p**2 + kappa * a**2)
    return p / den, a**2 / den


def peculiar_speed(eta, p: float = 1.0, A: float = 1.0):
    a = scale_factor(eta, A)
    return p / np.sqrt(p**2 + a**2)


def gamma_factor(eta, p: float = 1.0, A: float = 1.0):
    a = scale_factor(eta, A)
    return np.sqrt(1.0 + p**2 / a**2)
