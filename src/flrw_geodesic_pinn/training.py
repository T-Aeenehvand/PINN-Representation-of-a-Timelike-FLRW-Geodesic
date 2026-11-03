"""Small reusable training utilities shared by the public source tree."""
from __future__ import annotations
import numpy as np
import torch


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def configure_cpu_float64():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
