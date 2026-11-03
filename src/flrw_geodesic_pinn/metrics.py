"""Error and invariant diagnostics used in the manuscript."""
from __future__ import annotations
import numpy as np


def relative_l2(pred, ref, eps=1e-15):
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.linalg.norm(pred - ref, ord=2) / (np.linalg.norm(ref, ord=2) + eps))


def max_abs(values):
    return float(np.max(np.abs(np.asarray(values))))


def invariant_defects(eta, x_lambda, eta_lambda, p=1.0, kappa=1.0, A=1.0):
    a = A * np.asarray(eta)**2
    dp = a**2 * np.asarray(x_lambda) - p
    dk = a**2 * (np.asarray(eta_lambda)**2 - np.asarray(x_lambda)**2) - kappa
    return dp, dk
