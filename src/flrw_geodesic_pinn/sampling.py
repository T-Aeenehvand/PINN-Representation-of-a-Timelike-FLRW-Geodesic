"""Deterministic Chebyshev collocation and residual-adaptive refinement."""
from __future__ import annotations
import numpy as np


def chebyshev_grid(vmin, vmax, n):
    j = np.arange(n)
    nodes = np.cos(np.pi * (2*j + 1) / (2*n))
    vals = vmin + 0.5 * (1.0 - nodes) * (vmax - vmin)
    vals = np.concatenate([[vmin, vmax], vals])
    return np.unique(np.sort(vals))
