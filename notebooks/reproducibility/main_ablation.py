import os

# ============================================================
# 0. CONTROL PANEL — CHANGE ONLY THESE SETTINGS
# ============================================================

# Default below reproduces the paper-profile budget.
# For a quick pipeline test in a fresh runtime, temporarily set PROFILE = "smoke".
PROFILE = os.getenv("FLRW_PROFILE", "paper")          # "smoke" or "paper"
SEED = int(os.getenv("FLRW_SEED", "42"))
RUN_SECOND_SEED_CONFIRMATION = os.getenv("FLRW_SECOND_SEED", "0") == "1"   # optional; normally leave False
RUN_AFFINE_FAIRNESS = os.getenv("FLRW_AFFINE_FAIRNESS", "1") == "1"                 # add affine-specific nondimensionalization controls

# Physical matter benchmark — kept identical to the validated Paper 1 setup.
MATTER_ETA0 = 0.5
MATTER_X0 = 0.0
MATTER_P = 1.0
MATTER_KAPPA = 1.0
MATTER_A = 1.0
MATTER_LAMBDA0 = 0.0
MATTER_LAMBDA_MAX = 0.8

# Network: IDENTICAL architecture for all four cells of the 2×2 design.
WIDTH = 64
DEPTH = 5

# Optimization: IDENTICAL budget for all four cells.
ADAM_LR = 1e-3
LBFGS_LR = 0.5
LBFGS_HISTORY = 50

PROFILES = {
    "smoke": {
        "ADAM_EPOCHS_PER_STAGE": 60,
        "N_INITIAL": 96,
        "N_ADD_PER_STAGE": 24,
        "N_STAGES": 2,          # stage 0 + 2 refinements => 144 final points
        "N_CANDIDATE": 600,
        "LBFGS_ITERS": 20,
        "N_VALIDATION": 1200,
    },
    "paper": {
        "ADAM_EPOCHS_PER_STAGE": 1400,
        "N_INITIAL": 256,
        "N_ADD_PER_STAGE": 64,
        "N_STAGES": 4,          # stage 0 + 4 refinements => 512 interior-style count; 514 including endpoints
        "N_CANDIDATE": 5000,
        "LBFGS_ITERS": 220,
        "N_VALIDATION": 4000,
    },
}

if PROFILE not in PROFILES:
    raise ValueError("PROFILE must be 'smoke' or 'paper'.")
CFG = PROFILES[PROFILE]

print("PROFILE:", PROFILE)
print("SEED:", SEED)
print("CONFIG:", CFG)

# ============================================================
# 1. IMPORTS / REPRODUCIBILITY / RESULT MANAGEMENT
# ============================================================

import gc
import json
import math
import shutil
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from IPython.display import display
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

DTYPE = torch.float64
DEVICE = torch.device("cpu")
torch.set_default_dtype(DTYPE)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Each notebook execution gets a new run directory.
RESULTS_ROOT = Path(os.getenv("FLRW_RESULTS_ROOT", "results/local_runs"))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
RUN_DIR = RESULTS_ROOT / f"{RUN_ID}_2x2_ablation"
RUN_DIR.mkdir(parents=True, exist_ok=False)
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Run ID:", RUN_ID)
print("Run directory:", RUN_DIR.resolve())
print("Device:", DEVICE)
print("dtype:", torch.get_default_dtype())


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# Registry uses one stable schema for every experiment run.
REGISTRY_COLUMNS = [
    "status", "profile", "seed", "representation", "rar", "x_eta_rel_L2",
    "lambda_eta_rel_L2", "eta_lambda_rel_L2", "x_lambda_rel_L2",
    "max_R_eta", "max_R_x", "max_R_lambda", "max_momentum_residual",
    "max_kappa_residual", "min_abs_dlambda_deta", "wall_seconds",
    "collocation_points_final", "adam_epochs_total", "lbfgs_max_iter",
    "lbfgs_actual_iters", "lbfgs_func_evals",
    "reference_raw_vs_reduced_x_max", "reference_raw_vs_reduced_lambda_max",
    "run_id", "created_utc",
]


def append_registry_row(row):
    registry_path = RESULTS_ROOT / "run_registry.csv"
    row = dict(row)
    row["run_id"] = RUN_ID
    row["created_utc"] = RUN_ID.split("_")[0]
    clean = {k: row.get(k, np.nan) for k in REGISTRY_COLUMNS}
    df = pd.DataFrame([clean], columns=REGISTRY_COLUMNS)
    header = not registry_path.exists()
    df.to_csv(registry_path, mode="a", header=header, index=False)


with open(RUN_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "run_id": RUN_ID,
        "created_utc": RUN_ID.split("_")[0],
        "experiment": "2x2_ablation",
        "profile": PROFILE,
        "seed": SEED,
        "second_seed_confirmation": RUN_SECOND_SEED_CONFIRMATION,
        "network": {"width": WIDTH, "depth": DEPTH},
        "optimizer": {
            "adam_lr": ADAM_LR,
            "lbfgs_lr": LBFGS_LR,
            "lbfgs_history": LBFGS_HISTORY,
            "lbfgs_max_iter": CFG["LBFGS_ITERS"],
        },
        "training": CFG,
        "matter": {
            "A": MATTER_A, "eta0": MATTER_ETA0, "x0": MATTER_X0,
            "p": MATTER_P, "kappa": MATTER_KAPPA,
            "lambda0": MATTER_LAMBDA0, "lambda_max": MATTER_LAMBDA_MAX,
        },
    }, f, indent=2)

# ============================================================
# 2. PROBLEM DEFINITIONS + INDEPENDENT REFERENCES
# ============================================================

@dataclass(frozen=True)
class Problem:
    name: str
    A: float
    eta0: float
    x0: float
    p: float
    kappa: float
    lambda0: float
    lambda_max: float

MATTER = Problem(
    "matter", MATTER_A, MATTER_ETA0, MATTER_X0,
    MATTER_P, MATTER_KAPPA, MATTER_LAMBDA0, MATTER_LAMBDA_MAX
)


def scale_factor_np(eta, problem=MATTER):
    eta = np.asarray(eta, dtype=float)
    if problem.name == "matter":
        return problem.A * eta**2
    raise ValueError(problem.name)


def scale_factor_torch(eta, problem=MATTER):
    if problem.name == "matter":
        return problem.A * eta**2
    raise ValueError(problem.name)


def relative_l2(pred, truth):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    return float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + 1e-15))


def norm_max(x):
    return float(np.max(np.abs(np.asarray(x))))


def initial_affine_state(problem):
    a0 = problem.A * problem.eta0**2
    x0p = problem.p / a0**2
    eta0p = np.sqrt(problem.p**2 + problem.kappa * a0**2) / a0**2
    return eta0p, x0p


def raw_geodesic_system(lam, y, problem):
    eta, x, eta_p, x_p = y
    a = problem.A * eta**2
    a_prime = 2.0 * problem.A * eta
    Hc = a_prime / a
    eta_pp = -Hc * (eta_p**2 + x_p**2)
    x_pp = -2.0 * Hc * eta_p * x_p
    return [eta_p, x_p, eta_pp, x_pp]


def solve_raw_second_order(problem, n_dense=None, rtol=1e-11, atol=1e-11):
    eta0p, x0p = initial_affine_state(problem)
    y0 = [problem.eta0, problem.x0, eta0p, x0p]
    sol = solve_ivp(
        lambda lam, y: raw_geodesic_system(lam, y, problem),
        (problem.lambda0, problem.lambda_max),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def first_integral_eta_system(eta, y, problem):
    x, lam = y
    a = problem.A * eta**2
    den = np.sqrt(problem.p**2 + problem.kappa * a**2)
    dx = problem.p / den
    dl = a**2 / den
    return [dx, dl]


def build_reference_bundle(problem=MATTER, n=4000):
    raw = solve_raw_second_order(problem)
    lam_dense = np.linspace(problem.lambda0, problem.lambda_max, max(n, 5000))
    raw_vals = raw.sol(lam_dense)
    eta_raw, x_raw = raw_vals[0], raw_vals[1]

    # The raw matter trajectory is monotone in eta for this branch.
    eta_end = float(eta_raw[-1])
    eta_grid = np.linspace(problem.eta0, eta_end, n)

    # Independent reduced reference — used only as a consistency check.
    red = solve_ivp(
        lambda eta, y: first_integral_eta_system(eta, y, problem),
        (problem.eta0, eta_end),
        [problem.x0, problem.lambda0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )
    x_reduced, lam_reduced = red.sol(eta_grid)

    # Raw reference represented as functions of eta.
    order = np.argsort(eta_raw)
    eta_sorted = eta_raw[order]
    x_sorted = x_raw[order]
    lam_sorted = lam_dense[order]
    eta_unique, idx = np.unique(eta_sorted, return_index=True)
    x_unique = x_sorted[idx]
    lam_unique = lam_sorted[idx]
    x_of_eta = PchipInterpolator(eta_unique, x_unique)
    lam_of_eta = PchipInterpolator(eta_unique, lam_unique)

    # Raw-vs-reduced consistency on the common eta grid.
    x_raw_on_eta = x_of_eta(eta_grid)
    lam_raw_on_eta = lam_of_eta(eta_grid)
    raw_reduced_x_max = norm_max(x_raw_on_eta - x_reduced)
    raw_reduced_lambda_max = norm_max(lam_raw_on_eta - lam_reduced)

    return {
        "raw": raw,
        "eta_grid": eta_grid,
        "x_of_eta": x_of_eta,
        "lambda_of_eta": lam_of_eta,
        "x_reduced": x_reduced,
        "lambda_reduced": lam_reduced,
        "raw_reduced_x_max_abs": raw_reduced_x_max,
        "raw_reduced_lambda_max_abs": raw_reduced_lambda_max,
    }


print("Building independent raw + reduced reference ...")
REF = build_reference_bundle(MATTER, n=CFG["N_VALIDATION"])
ETA_MIN = float(REF["eta_grid"][0])
ETA_MAX = float(REF["eta_grid"][-1])
print("Matter eta range:", ETA_MIN, "to", ETA_MAX)
print("Raw vs reduced x max abs:", REF["raw_reduced_x_max_abs"])
print("Raw vs reduced lambda max abs:", REF["raw_reduced_lambda_max_abs"])

# ============================================================
# 3. SHARED NETWORK UTILITIES
# ============================================================

def build_mlp(in_dim=1, out_dim=2, width=64, depth=5):
    layers = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, out_dim)]
    net = nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


def make_shared_initial_state(seed):
    seed_everything(seed)
    probe = build_mlp(1, 2, WIDTH, DEPTH)
    state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    del probe
    return state


SHARED_INIT_STATE = make_shared_initial_state(SEED)
print("Shared initialization captured from seed", SEED)

# ============================================================
# 4. AFFINE SECOND-ORDER MODEL
# ============================================================

def deriv(y, x, create_graph=True):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


class AffineSecondOrderPINN(nn.Module):
    def __init__(self, problem, lam_min, lam_max, width=64, depth=5, init_state=None):
        super().__init__()
        self.problem = problem
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.net = build_mlp(1, 2, width, depth)
        if init_state is not None:
            self.net.load_state_dict(init_state)

        self.eta0p, self.x0p = initial_affine_state(problem)

    def forward(self, lam):
        lam = lam.to(device=DEVICE, dtype=DTYPE)
        s = 2.0 * (lam - self.lam_min) / (self.lam_max - self.lam_min) - 1.0
        raw = self.net(s)
        d = lam - self.problem.lambda0
        eta = self.problem.eta0 + self.eta0p * d + d**2 * raw[:, 0:1]
        x = self.problem.x0 + self.x0p * d + d**2 * raw[:, 1:2]
        return eta, x


def affine_second_order_residuals(model, lam):
    lam = lam.detach().clone().requires_grad_(True)
    eta, x = model(lam)
    eta_p = deriv(eta, lam)
    x_p = deriv(x, lam)
    eta_pp = deriv(eta_p, lam)
    x_pp = deriv(x_p, lam)

    a = scale_factor_torch(eta, model.problem)
    Hc = 2.0 / eta  # matter: a'/a = 2/eta

    R_eta = eta_pp + Hc * (eta_p**2 + x_p**2)
    R_x = x_pp + 2.0 * Hc * eta_p * x_p
    return {
        "lambda": lam,
        "eta": eta,
        "x": x,
        "eta_p": eta_p,
        "x_p": x_p,
        "R_eta": R_eta,
        "R_x": R_x,
    }


def affine_loss(model, lam_grid):
    r = affine_second_order_residuals(model, lam_grid)
    L_eta = torch.mean(r["R_eta"]**2)
    L_x = torch.mean(r["R_x"]**2)
    return r, L_eta + L_x, L_eta, L_x

# ============================================================
# 4A. AFFINE SECOND-ORDER MODEL WITH EXPLICIT NONDIMENSIONALIZATION
# ============================================================

class AffinePreconditionedSecondOrderPINN(nn.Module):
    """
    Same physical second-order affine geodesic equations as the baseline,
    but written in dimensionless affine/output variables.

    This is a fairness control, not a new physical model.
    """
    def __init__(self, problem, lam_min, lam_max, width=64, depth=5, init_state=None):
        super().__init__()
        self.problem = problem
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.delta_lam = float(lam_max - lam_min)
        self.net = build_mlp(1, 2, width, depth)
        if init_state is not None:
            self.net.load_state_dict(init_state)

        self.eta0p, self.x0p = initial_affine_state(problem)

        # Scales are determined only by known initial data and the affine interval.
        # They make the leading normalized slopes O(1) without using the reference solution.
        self.eta_scale = max(abs(self.eta0p) * self.delta_lam, 1e-12)
        self.x_scale = max(abs(self.x0p) * self.delta_lam, 1e-12)
        self.eta_slope_tau = self.eta0p * self.delta_lam / self.eta_scale
        self.x_slope_tau = self.x0p * self.delta_lam / self.x_scale

    def forward(self, lam):
        lam = lam.to(device=DEVICE, dtype=DTYPE)
        tau = (lam - self.lam_min) / self.delta_lam
        s = 2.0 * tau - 1.0
        raw = self.net(s)

        # Hard initial position and first derivative constraints in normalized variables.
        eta_hat = self.eta_slope_tau * tau + tau**2 * raw[:, 0:1]
        x_hat = self.x_slope_tau * tau + tau**2 * raw[:, 1:2]
        eta = self.problem.eta0 + self.eta_scale * eta_hat
        x = self.problem.x0 + self.x_scale * x_hat
        return eta, x


def affine_preconditioned_residuals(model, lam):
    lam = lam.detach().clone().requires_grad_(True)
    eta, x = model(lam)
    eta_p = deriv(eta, lam)
    x_p = deriv(x, lam)
    eta_pp = deriv(eta_p, lam)
    x_pp = deriv(x_p, lam)

    Hc = 2.0 / eta  # matter: a'/a = 2/eta
    R_eta_phys = eta_pp + Hc * (eta_p**2 + x_p**2)
    R_x_phys = x_pp + 2.0 * Hc * eta_p * x_p

    # Dimensionless residuals. The zero set is unchanged; only the optimization scale changes.
    R_eta = (model.delta_lam**2 / model.eta_scale) * R_eta_phys
    R_x = (model.delta_lam**2 / model.x_scale) * R_x_phys

    return {
        "lambda": lam,
        "eta": eta,
        "x": x,
        "eta_p": eta_p,
        "x_p": x_p,
        "R_eta": R_eta,
        "R_x": R_x,
        "R_eta_phys": R_eta_phys,
        "R_x_phys": R_x_phys,
    }


def affine_preconditioned_loss(model, lam_grid):
    r = affine_preconditioned_residuals(model, lam_grid)
    L_eta = torch.mean(r["R_eta"]**2)
    L_x = torch.mean(r["R_x"]**2)
    return r, L_eta + L_x, L_eta, L_x


# ============================================================
# 5. CONFORMAL FIRST-INTEGRAL MODEL
# ============================================================

class ConformalFirstIntegralPINN(nn.Module):
    def __init__(self, problem, eta_min, eta_max, width=64, depth=5, init_state=None):
        super().__init__()
        self.problem = problem
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.net = build_mlp(1, 2, width, depth)
        if init_state is not None:
            self.net.load_state_dict(init_state)

        a0 = problem.A * problem.eta0**2
        den0 = np.sqrt(problem.p**2 + problem.kappa * a0**2)
        self.dx0 = float(problem.p / den0)
        self.dlambda0 = float(a0**2 / den0)

    def forward(self, eta):
        eta = eta.to(device=DEVICE, dtype=DTYPE)
        s = 2.0 * (eta - self.eta_min) / (self.eta_max - self.eta_min) - 1.0
        raw = self.net(s)
        d = eta - self.problem.eta0
        x = self.problem.x0 + self.dx0 * d + d**2 * raw[:, 0:1]
        lam = self.problem.lambda0 + self.dlambda0 * d + d**2 * raw[:, 1:2]
        return x, lam


def conformal_residuals(model, eta):
    eta = eta.detach().clone().requires_grad_(True)
    x, lam = model(eta)
    dx = deriv(x, eta)
    dl = deriv(lam, eta)
    a = scale_factor_torch(eta, model.problem)
    den = torch.sqrt(model.problem.p**2 + model.problem.kappa * a**2)
    target_x = model.problem.p / den
    target_l = a**2 / den
    return {
        "eta": eta,
        "x": x,
        "lambda": lam,
        "dx": dx,
        "dlambda": dl,
        "R_x": dx - target_x,
        "R_lambda": dl - target_l,
    }


def conformal_loss(model, eta_grid):
    r = conformal_residuals(model, eta_grid)
    Lx = torch.mean(r["R_x"]**2)
    Ll = torch.mean(r["R_lambda"]**2)
    return r, Lx + Ll, Lx, Ll

# ============================================================
# 6. COLLOCATION GRIDS + RAR ENGINE
# ============================================================

def chebyshev_grid(vmin, vmax, n):
    j = np.arange(n)
    nodes = np.cos(np.pi * (2*j + 1) / (2*n))
    vals = vmin + 0.5 * (1.0 - nodes) * (vmax - vmin)
    vals = np.concatenate([[vmin, vmax], vals])
    vals = np.unique(np.sort(vals))
    return torch.tensor(vals.reshape(-1,1), dtype=DTYPE, device=DEVICE)


def candidate_scores(model, candidates, residual_fn, chunk=800):
    scores = []
    for start in range(0, len(candidates), chunk):
        part = candidates[start:start+chunk]
        r = residual_fn(model, part)
        vals = []
        if "R_eta" in r and "R_x" in r:
            residual_keys = ("R_eta", "R_x")
        else:
            residual_keys = ("R_x", "R_lambda")
        for key in residual_keys:
            vals.append(r[key].detach().cpu().numpy()**2)
        scores.append(np.sum(vals, axis=0).reshape(-1))
    return np.concatenate(scores)


def refine_grid(model, current, vmin, vmax, n_candidate, n_add, residual_fn):
    candidates = torch.linspace(vmin, vmax, n_candidate, dtype=DTYPE, device=DEVICE).reshape(-1,1)
    scores = candidate_scores(model, candidates, residual_fn)

    # Exclude candidate points that are already in the current collocation set.
    current_np = current.detach().cpu().numpy().reshape(-1)
    cand_np = candidates.detach().cpu().numpy().reshape(-1)
    available = np.ones(len(cand_np), dtype=bool)
    for value in current_np:
        available &= np.abs(cand_np - value) > 1e-12

    ranked = np.argsort(scores[available])[::-1]
    available_idx = np.where(available)[0]
    take = min(n_add, len(ranked))
    idx = available_idx[ranked[:take]]
    new_points = cand_np[idx]

    merged = np.concatenate([current_np, new_points])
    merged = np.unique(np.sort(merged))
    new_grid = torch.tensor(merged.reshape(-1,1), dtype=DTYPE, device=DEVICE)
    return new_grid, float(scores.max())


def matched_training_stages(vmin, vmax):
    # Exactly the point counts used by the RAR-enabled run.
    counts = [CFG["N_INITIAL"] + i * CFG["N_ADD_PER_STAGE"] for i in range(CFG["N_STAGES"] + 1)]
    return [chebyshev_grid(vmin, vmax, int(n)) for n in counts]

# ============================================================
# 7. OPTIMIZATION LOOPS — SAME BUDGET FOR ALL EXPERIMENT CELLS
# ============================================================

def run_adam_stages(model, initial_grid, vmin, vmax, residual_fn, loss_fn,
                    rar_on, verbose=True):
    current = initial_grid.detach().clone()
    history = []
    stage_records = []
    fixed_grids = matched_training_stages(vmin, vmax) if not rar_on else None

    total_stages = CFG["N_STAGES"] + 1
    total_adam_epochs = CFG["ADAM_EPOCHS_PER_STAGE"] * total_stages

    for stage in range(total_stages):
        if not rar_on and stage > 0:
            # Matched fixed-grid control: same point count as RAR condition,
            # but the points are chosen deterministically rather than by residual.
            current = fixed_grids[stage]

        opt = optim.Adam(model.parameters(), lr=ADAM_LR)
        for epoch in range(CFG["ADAM_EPOCHS_PER_STAGE"]):
            opt.zero_grad(set_to_none=True)
            r, L, L1, L2 = loss_fn(model, current)
            if not torch.isfinite(L):
                raise FloatingPointError(
                    f"Non-finite loss in stage={stage}, epoch={epoch+1}. "
                    "Inspect the saved partial run and do not interpret this condition as success."
                )
            L.backward()
            opt.step()
            history.append({
                "stage": stage,
                "epoch_in_stage": epoch + 1,
                "epoch_global": stage * CFG["ADAM_EPOCHS_PER_STAGE"] + epoch + 1,
                "loss": float(L.item()),
                "L1": float(L1.item()),
                "L2": float(L2.item()),
                "collocation_points": int(len(current)),
                "rar_on": bool(rar_on),
            })

            if verbose and (epoch + 1) in {
                1,
                CFG["ADAM_EPOCHS_PER_STAGE"] // 2,
                CFG["ADAM_EPOCHS_PER_STAGE"],
            }:
                print(
                    f"stage={stage} epoch={epoch+1}/{CFG['ADAM_EPOCHS_PER_STAGE']} "
                    f"points={len(current)} loss={L.item():.3e}"
                )

        if rar_on and stage < CFG["N_STAGES"]:
            current, max_res = refine_grid(
                model,
                current,
                vmin, vmax,
                CFG["N_CANDIDATE"],
                CFG["N_ADD_PER_STAGE"],
                residual_fn,
            )
            stage_records.append({
                "stage": stage + 1,
                "action": "RAR_add",
                "collocation_points": int(len(current)),
                "candidate_max_residual": max_res,
            })
        else:
            stage_records.append({
                "stage": stage,
                "action": "fixed_grid" if not rar_on else "final",
                "collocation_points": int(len(current)),
            })

    # Exactly the same L-BFGS configuration for every experiment cell.
    lbfgs = optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=CFG["LBFGS_ITERS"],
        history_size=LBFGS_HISTORY,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        _, L, _, _ = loss_fn(model, current)
        if not torch.isfinite(L):
            raise FloatingPointError("Non-finite L-BFGS loss.")
        L.backward()
        return L

    lbfgs.step(closure)

    # PyTorch stores the global L-BFGS counters in the state of the first parameter.
    # These are realized counters for this call, not the requested max_iter cap.
    first_param = next(iter(model.parameters()))
    lbfgs_state = lbfgs.state.get(first_param, {})
    lbfgs_actual_iters = int(lbfgs_state.get("n_iter", -1))
    lbfgs_func_evals = int(lbfgs_state.get("func_evals", -1))

    return (model, current, pd.DataFrame(history), pd.DataFrame(stage_records),
            total_adam_epochs, lbfgs_actual_iters, lbfgs_func_evals)

# ============================================================
# 8. VALIDATION HELPERS
# ============================================================

def validate_affine(model, problem, ref, n_eval, residual_fn=affine_second_order_residuals):
    lam_eval = np.linspace(problem.lambda0, problem.lambda_max, n_eval)
    lam_t = torch.tensor(lam_eval.reshape(-1,1), dtype=DTYPE, device=DEVICE)
    with torch.enable_grad():
        r = residual_fn(model, lam_t)

    eta_pred = r["eta"].detach().cpu().numpy().reshape(-1)
    x_pred = r["x"].detach().cpu().numpy().reshape(-1)
    eta_p = r["eta_p"].detach().cpu().numpy().reshape(-1)
    x_p = r["x_p"].detach().cpu().numpy().reshape(-1)
    R_eta = r["R_eta"].detach().cpu().numpy().reshape(-1)
    R_x = r["R_x"].detach().cpu().numpy().reshape(-1)

    eta_ref, x_ref = ref["raw"].sol(lam_eval)[:2]
    a = scale_factor_np(eta_pred, problem)
    p_res = a**2 * x_p - problem.p
    k_res = a**2 * (eta_p**2 - x_p**2) - problem.kappa

    metrics = {
        "x_eta_rel_L2": np.nan,
        "lambda_eta_rel_L2": np.nan,
        "eta_lambda_rel_L2": relative_l2(eta_pred, eta_ref),
        "x_lambda_rel_L2": relative_l2(x_pred, x_ref),
        "max_R_eta": norm_max(R_eta),
        "max_R_x": norm_max(R_x),
        "max_R_lambda": np.nan,
        "max_momentum_residual": norm_max(p_res),
        "max_kappa_residual": norm_max(k_res),
        "min_abs_dlambda_deta": np.nan,
    }
    arrays = {
        "lambda": lam_eval,
        "eta_pred": eta_pred,
        "x_pred": x_pred,
        "eta_ref": eta_ref,
        "x_ref": x_ref,
        "R_eta": R_eta,
        "R_x": R_x,
        "eta_prime": eta_p,
        "x_prime": x_p,
        "p_residual": p_res,
        "kappa_residual": k_res,
    }
    return metrics, arrays


def validate_conformal(model, problem, ref, n_eval):
    eta_eval = np.linspace(ETA_MIN, ETA_MAX, n_eval)
    eta_t = torch.tensor(eta_eval.reshape(-1,1), dtype=DTYPE, device=DEVICE)
    with torch.enable_grad():
        r = conformal_residuals(model, eta_t)

    x_pred = r["x"].detach().cpu().numpy().reshape(-1)
    lam_pred = r["lambda"].detach().cpu().numpy().reshape(-1)
    dx_deta = r["dx"].detach().cpu().numpy().reshape(-1)
    dl_deta = r["dlambda"].detach().cpu().numpy().reshape(-1)
    R_x = r["R_x"].detach().cpu().numpy().reshape(-1)
    R_lambda = r["R_lambda"].detach().cpu().numpy().reshape(-1)

    x_ref_eta = ref["x_of_eta"](eta_eval)
    lam_ref_eta = ref["lambda_of_eta"](eta_eval)

    # A valid future-directed reconstruction requires lambda(eta) to be
    # strictly increasing in the ORIGINAL increasing-eta order.  Sorting first
    # would destroy the very information this check is meant to test.
    if not np.all(np.isfinite(lam_pred)) or not np.all(np.isfinite(dl_deta)):
        raise RuntimeError("Conformal model produced non-finite affine reconstruction data.")
    if np.any(np.diff(lam_pred) <= 0):
        raise RuntimeError("Predicted lambda(eta) is not strictly increasing on the evaluation grid.")
    if np.any(dl_deta <= 0):
        raise RuntimeError("Predicted d lambda / d eta is non-positive on the evaluation grid.")

    # Require coverage before constructing the inverse and explicitly disable
    # extrapolation.  A small endpoint tolerance only absorbs floating-point
    # roundoff; it is not permission to extrapolate a learned trajectory.
    tol = 1e-12 * max(1.0, abs(problem.lambda_max - problem.lambda0))
    covers = (lam_pred[0] <= problem.lambda0 + tol) and (lam_pred[-1] >= problem.lambda_max - tol)

    lam_eval = np.linspace(problem.lambda0, problem.lambda_max, n_eval)
    eta_ref, x_ref = ref["raw"].sol(lam_eval)[:2]
    if covers:
        eta_recon = PchipInterpolator(lam_pred, eta_eval, extrapolate=False)(lam_eval)
        x_recon = PchipInterpolator(lam_pred, x_pred, extrapolate=False)(lam_eval)
        if np.any(~np.isfinite(eta_recon)) or np.any(~np.isfinite(x_recon)):
            raise RuntimeError("Inverse reconstruction produced non-finite values despite nominal coverage.")
        eta_lambda_rel_L2 = relative_l2(eta_recon, eta_ref)
        x_lambda_rel_L2 = relative_l2(x_recon, x_ref)
    else:
        # Do not extrapolate a learned affine map merely to manufacture a full-interval metric.
        # This path lets smoke/diagnostic runs complete while marking the common-affine-grid
        # reconstruction as unavailable.  Paper-quality runs are expected to have covers=True.
        print(
            "WARNING: predicted lambda range "
            f"[{lam_pred[0]:.16g}, {lam_pred[-1]:.16g}] does not cover "
            f"[{problem.lambda0:.16g}, {problem.lambda_max:.16g}]; "
            "affine-grid reconstruction metrics are set to NaN."
        )
        eta_recon = np.full_like(lam_eval, np.nan, dtype=float)
        x_recon = np.full_like(lam_eval, np.nan, dtype=float)
        eta_lambda_rel_L2 = np.nan
        x_lambda_rel_L2 = np.nan

    dx_dlambda = dx_deta / dl_deta
    eta_dlambda = 1.0 / dl_deta
    a = scale_factor_np(eta_eval, problem)
    p_res = a**2 * dx_dlambda - problem.p
    k_res = a**2 * (eta_dlambda**2 - dx_dlambda**2) - problem.kappa

    metrics = {
        "x_eta_rel_L2": relative_l2(x_pred, x_ref_eta),
        "lambda_eta_rel_L2": relative_l2(lam_pred, lam_ref_eta),
        "eta_lambda_rel_L2": eta_lambda_rel_L2,
        "x_lambda_rel_L2": x_lambda_rel_L2,
        "max_R_eta": np.nan,
        "max_R_x": norm_max(R_x),
        "max_R_lambda": norm_max(R_lambda),
        "max_momentum_residual": norm_max(p_res),
        "max_kappa_residual": norm_max(k_res),
        "min_abs_dlambda_deta": float(np.min(np.abs(dl_deta))),
        "min_signed_dlambda_deta": float(np.min(dl_deta)),
        "min_forward_lambda_increment": float(np.min(np.diff(lam_pred))),
        "covers_affine_interval": bool(covers),
    }
    arrays = {
        "eta": eta_eval,
        "x_pred": x_pred,
        "lambda_pred": lam_pred,
        "x_ref_eta": x_ref_eta,
        "lambda_ref_eta": lam_ref_eta,
        "eta_recon": eta_recon,
        "x_recon": x_recon,
        "eta_ref_lambda": eta_ref,
        "x_ref_lambda": x_ref,
        "R_x": R_x,
        "R_lambda": R_lambda,
        "dx_deta": dx_deta,
        "dlambda_deta": dl_deta,
        "p_residual": p_res,
        "kappa_residual": k_res,
        "lambda_eval": lam_eval,
    }
    return metrics, arrays


# ============================================================
# 9. GENERIC TRAIN / EVALUATE FUNCTION FOR ONE FACTOR CELL
# ============================================================

def train_one_cell(representation, rar_on, seed=SEED, verbose=True):
    seed_everything(seed)

    if representation == "affine_second_order":
        model = AffineSecondOrderPINN(
            MATTER, MATTER.lambda0, MATTER.lambda_max,
            width=WIDTH, depth=DEPTH,
            init_state=SHARED_INIT_STATE,
        ).to(DEVICE)
        vmin, vmax = MATTER.lambda0, MATTER.lambda_max
        grid = chebyshev_grid(vmin, vmax, CFG["N_INITIAL"])
        residual_fn = affine_second_order_residuals
        loss_fn = affine_loss
        validation_fn = lambda m: validate_affine(m, MATTER, REF, CFG["N_VALIDATION"], affine_second_order_residuals)
        fairness_note = "raw_affine"
    elif representation == "affine_preconditioned":
        model = AffinePreconditionedSecondOrderPINN(
            MATTER, MATTER.lambda0, MATTER.lambda_max,
            width=WIDTH, depth=DEPTH,
            init_state=SHARED_INIT_STATE,
        ).to(DEVICE)
        vmin, vmax = MATTER.lambda0, MATTER.lambda_max
        grid = chebyshev_grid(vmin, vmax, CFG["N_INITIAL"])
        residual_fn = affine_preconditioned_residuals
        loss_fn = affine_preconditioned_loss
        validation_fn = lambda m: validate_affine(m, MATTER, REF, CFG["N_VALIDATION"], affine_preconditioned_residuals)
        fairness_note = "affine_scaled_preconditioned"
    elif representation == "conformal_first_integral":
        model = ConformalFirstIntegralPINN(
            MATTER, ETA_MIN, ETA_MAX,
            width=WIDTH, depth=DEPTH,
            init_state=SHARED_INIT_STATE,
        ).to(DEVICE)
        vmin, vmax = ETA_MIN, ETA_MAX
        grid = chebyshev_grid(vmin, vmax, CFG["N_INITIAL"])
        residual_fn = conformal_residuals
        loss_fn = conformal_loss
        validation_fn = lambda m: validate_conformal(m, MATTER, REF, CFG["N_VALIDATION"])
        fairness_note = "conformal_first_integral"
    else:
        raise ValueError(representation)

    t0 = time.time()
    (model, final_grid, history, stage_records, total_adam_epochs,
     lbfgs_actual_iters, lbfgs_func_evals) = run_adam_stages(
        model, grid, vmin, vmax, residual_fn, loss_fn, rar_on, verbose=verbose
    )
    wall_s = time.time() - t0

    metrics, arrays = validation_fn(model)

    metrics.update({
        "representation": representation,
        "rar": bool(rar_on),
        "fairness_note": fairness_note,
        "seed": seed,
        "wall_seconds": wall_s,
        "collocation_points_final": int(len(final_grid)),
        "adam_epochs_total": int(total_adam_epochs),
        "lbfgs_max_iter": int(CFG["LBFGS_ITERS"]),
        "lbfgs_actual_iters": int(lbfgs_actual_iters),
        "lbfgs_func_evals": int(lbfgs_func_evals),
        "reference_raw_vs_reduced_x_max": REF["raw_reduced_x_max_abs"],
        "reference_raw_vs_reduced_lambda_max": REF["raw_reduced_lambda_max_abs"],
    })

    if representation == "affine_preconditioned":
        metrics.update({
            "eta_scale": float(model.eta_scale),
            "x_scale": float(model.x_scale),
            "delta_lambda": float(model.delta_lam),
            "normalized_eta_initial_slope": float(model.eta_slope_tau),
            "normalized_x_initial_slope": float(model.x_slope_tau),
        })
    else:
        metrics.update({
            "eta_scale": np.nan,
            "x_scale": np.nan,
            "delta_lambda": np.nan,
            "normalized_eta_initial_slope": np.nan,
            "normalized_x_initial_slope": np.nan,
        })

    run_name = f"{representation}__rar_{'on' if rar_on else 'off'}"
    if seed != SEED:
        run_name += f"__seed_{seed}"
    run_dir = RUN_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
    history.to_csv(run_dir / "training_history.csv", index=False)
    stage_records.to_csv(run_dir / "stage_records.csv", index=False)
    np.savez(run_dir / "validation_arrays.npz", **arrays)
    torch.save(model.state_dict(), run_dir / "model.pt")

    with open(run_dir / "cell_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "representation": representation,
            "rar_on": bool(rar_on),
            "seed": seed,
            "profile": PROFILE,
            "network": {"width": WIDTH, "depth": DEPTH},
            "training": CFG,
            "shared_initial_state": True,
            "fairness_note": fairness_note,
            "preconditioning_scales": {
                "eta_scale": float(metrics["eta_scale"]) if np.isfinite(metrics["eta_scale"]) else None,
                "x_scale": float(metrics["x_scale"]) if np.isfinite(metrics["x_scale"]) else None,
                "delta_lambda": float(metrics["delta_lambda"]) if np.isfinite(metrics["delta_lambda"]) else None,
            },
        }, f, indent=2)

    # Common affine trajectory plot.
    lam_ref_grid = np.linspace(MATTER.lambda0, MATTER.lambda_max, CFG["N_VALIDATION"])
    eta_ref_grid, x_ref_grid = REF["raw"].sol(lam_ref_grid)[:2]
    if representation in {"affine_second_order", "affine_preconditioned"}:
        lam_plot = arrays["lambda"]
        eta_plot = arrays["eta_pred"]
        x_plot = arrays["x_pred"]
    else:
        lam_plot = arrays["lambda_eval"]
        eta_plot = arrays["eta_recon"]
        x_plot = arrays["x_recon"]

    fig = plt.figure(figsize=(8,5))
    plt.plot(lam_ref_grid, eta_ref_grid, "k--", lw=2, label="raw DOP853")
    plt.plot(lam_plot, eta_plot, label=run_name)
    plt.xlabel(r"$\lambda$"); plt.ylabel(r"$\eta(\lambda)$")
    plt.title(run_name + ": affine trajectory")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    fig.savefig(run_dir / "eta_lambda.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8,5))
    plt.plot(lam_ref_grid, x_ref_grid, "k--", lw=2, label="raw DOP853")
    plt.plot(lam_plot, x_plot, label=run_name)
    plt.xlabel(r"$\lambda$"); plt.ylabel(r"$x(\lambda)$")
    plt.title(run_name + ": affine spatial trajectory")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    fig.savefig(run_dir / "x_lambda.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return metrics, arrays, history, stage_records, run_dir

# The original 2×2 factorial design.
BASE_CELL_SPECS = [
    ("affine_second_order", False),
    ("affine_second_order", True),
    ("conformal_first_integral", False),
    ("conformal_first_integral", True),
]

# The final scaling/preconditioning control for the affine equations,
# repeated with RAR OFF and ON so that scaling and sampling remain separable.
FAIRNESS_CELL_SPECS = [
    ("affine_preconditioned", False),
    ("affine_preconditioned", True),
]

CELL_SPECS = BASE_CELL_SPECS + (FAIRNESS_CELL_SPECS if RUN_AFFINE_FAIRNESS else [])

results = []
cell_outputs = {}
for representation, rar_on in CELL_SPECS:
    print("\n" + "="*90)
    print(f"RUNNING: representation={representation} | RAR={'ON' if rar_on else 'OFF'}")
    print("="*90)
    out = train_one_cell(representation, rar_on, seed=SEED, verbose=True)
    metrics, arrays, history, stage_records, run_dir = out
    results.append(metrics)
    cell_outputs[(representation, rar_on)] = out
    display(pd.DataFrame([metrics]))
    gc.collect()

ablation_df = pd.DataFrame(results)
ablation_df.to_csv(RUN_DIR / "ablation_all_results.csv", index=False)
print("\nCompleted factorial + fairness-control experiment.")
display(ablation_df[[
    "representation", "rar", "eta_lambda_rel_L2", "x_lambda_rel_L2",
    "max_R_eta", "max_R_x", "max_R_lambda",
    "max_momentum_residual", "max_kappa_residual",
    "wall_seconds", "collocation_points_final"
]])


# ============================================================
# 10. FACTORIAL + FAIRNESS SUMMARY
# ============================================================

pivot_eta = ablation_df.pivot(index="representation", columns="rar", values="eta_lambda_rel_L2")
pivot_x = ablation_df.pivot(index="representation", columns="rar", values="x_lambda_rel_L2")

print("Common affine-parameter errors: eta(lambda)")
display(pivot_eta)
print("Common affine-parameter errors: x(lambda)")
display(pivot_x)


def ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan

lookup = {
    (r, bool(rr)): ablation_df[(ablation_df["representation"] == r) & (ablation_df["rar"] == bool(rr))].iloc[0]
    for r, rr in CELL_SPECS
}

effect_rows = []

def add_effect(label, numerator_key, denominator_key):
    if numerator_key in lookup and denominator_key in lookup:
        a = lookup[numerator_key]
        b = lookup[denominator_key]
        effect_rows.append({
            "comparison": label,
            "eta_lambda_ratio": ratio(a["eta_lambda_rel_L2"], b["eta_lambda_rel_L2"]),
            "x_lambda_ratio": ratio(a["x_lambda_rel_L2"], b["x_lambda_rel_L2"]),
        })

add_effect("RAR effect | raw affine (ON/OFF)",
           ("affine_second_order", True), ("affine_second_order", False))
add_effect("RAR effect | conformal (ON/OFF)",
           ("conformal_first_integral", True), ("conformal_first_integral", False))
add_effect("Representation effect | RAR OFF (conformal/raw affine)",
           ("conformal_first_integral", False), ("affine_second_order", False))
add_effect("Representation effect | RAR ON (conformal/raw affine)",
           ("conformal_first_integral", True), ("affine_second_order", True))
if RUN_AFFINE_FAIRNESS:
    add_effect("Affine preconditioning effect | RAR OFF (precond/raw)",
               ("affine_preconditioned", False), ("affine_second_order", False))
    add_effect("Affine preconditioning effect | RAR ON (precond/raw)",
               ("affine_preconditioned", True), ("affine_second_order", True))
    add_effect("Representation effect | RAR OFF (conformal/preconditioned)",
               ("conformal_first_integral", False), ("affine_preconditioned", False))
    add_effect("Representation effect | RAR ON (conformal/preconditioned)",
               ("conformal_first_integral", True), ("affine_preconditioned", True))

effects = pd.DataFrame(effect_rows)
effects.to_csv(RUN_DIR / "factor_effects.csv", index=False)
display(effects)

# Publication-friendly bar chart on common affine errors.
plot_df = ablation_df.copy()
plot_df["label"] = plot_df.apply(lambda r: f"{r['representation']}\nRAR={'on' if r['rar'] else 'off'}", axis=1)
fig = plt.figure(figsize=(11, 6))
x = np.arange(len(plot_df))
width = 0.36
plt.bar(x - width/2, plot_df["eta_lambda_rel_L2"], width, label=r"$\eta(\lambda)$ relative $L^2$")
plt.bar(x + width/2, plot_df["x_lambda_rel_L2"], width, label=r"$x(\lambda)$ relative $L^2$")
plt.yscale("log")
plt.xticks(x, plot_df["label"], rotation=25, ha="right")
plt.ylabel("Relative $L^2$ error (log scale)")
plt.title("Controlled ablation and affine-preconditioning fairness check")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
fig.savefig(FIG_DIR / "ablation_with_fairness_control.png", dpi=240, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 11. DESIGN SANITY CHECKS
# ============================================================

assert ablation_df["seed"].nunique() == 1
assert ablation_df["seed"].iloc[0] == SEED
assert set(ablation_df["adam_epochs_total"]) == {CFG["ADAM_EPOCHS_PER_STAGE"] * (CFG["N_STAGES"] + 1)}
assert set(ablation_df["collocation_points_final"]) == {CFG["N_INITIAL"] + 2 + CFG["N_ADD_PER_STAGE"] * CFG["N_STAGES"]}
expected_cells = 4 + (2 if RUN_AFFINE_FAIRNESS else 0)
assert len(ablation_df) == expected_cells
assert REF["raw_reduced_x_max_abs"] < 1e-7, "Independent reference consistency failed."
assert REF["raw_reduced_lambda_max_abs"] < 1e-7, "Independent reference consistency failed."

# Shared initial-state bookkeeping: all cells use the same parameter state dictionary.
if RUN_AFFINE_FAIRNESS:
    for rar_flag in (False, True):
        row = ablation_df[(ablation_df["representation"] == "affine_preconditioned") & (ablation_df["rar"] == rar_flag)].iloc[0]
        assert np.isfinite(row["eta_scale"]) and np.isfinite(row["x_scale"])
        assert abs(row["normalized_eta_initial_slope"]) - 1.0 < 1e-12
        assert abs(row["normalized_x_initial_slope"]) - 1.0 < 1e-12

print("PASS: factorial cell count =", len(ablation_df))
print("PASS: same seed:", SEED)
print("PASS: same final collocation count:", int(ablation_df["collocation_points_final"].iloc[0]))
print("PASS: same total Adam epochs:", int(ablation_df["adam_epochs_total"].iloc[0]))
print("PASS: raw/reduced reference consistency is within tolerance.")
if RUN_AFFINE_FAIRNESS:
    print("PASS: affine preconditioning is based only on prescribed initial data and produces O(1) normalized slopes.")


# ============================================================
# 12. OPTIONAL SECOND-SEED CONFIRMATION
# ============================================================

if RUN_SECOND_SEED_CONFIRMATION:
    confirmation_rows = []
    confirmation_seed = 100
    shared_state_2 = make_shared_initial_state(confirmation_seed)
    old_state = SHARED_INIT_STATE
    SHARED_INIT_STATE = shared_state_2
    try:
        for representation, rar_on in CELL_SPECS:
            out = train_one_cell(representation, rar_on, seed=confirmation_seed, verbose=False)
            metrics = out[0]
            confirmation_rows.append(metrics)
        confirmation_df = pd.DataFrame(confirmation_rows)
        confirmation_df.to_csv(RUN_DIR / "second_seed_confirmation.csv", index=False)
        display(confirmation_df[[
            "representation", "rar", "eta_lambda_rel_L2", "x_lambda_rel_L2",
            "max_R_eta", "max_R_x", "max_R_lambda"
        ]])
    finally:
        SHARED_INIT_STATE = old_state
else:
    print("Second-seed confirmation disabled; primary factorial result uses seed", SEED)

# ============================================================
# 13. MANUSCRIPT-READY EVIDENCE PACK
# ============================================================

paper_table = ablation_df[[
    "representation", "rar", "eta_lambda_rel_L2", "x_lambda_rel_L2",
    "x_eta_rel_L2", "lambda_eta_rel_L2", "max_R_eta", "max_R_x",
    "max_R_lambda", "max_momentum_residual", "max_kappa_residual",
    "wall_seconds", "collocation_points_final", "adam_epochs_total",
    "lbfgs_max_iter", "lbfgs_actual_iters", "lbfgs_func_evals",
    "eta_scale", "x_scale", "delta_lambda"
]].copy()

paper_table.to_csv(RUN_DIR / "Paper1_Table_Ablation_With_Fairness_Control.csv", index=False)

report_lines = []
report_lines.append("# Paper 1 — Controlled 2×2 Ablation + Affine Fairness Control")
report_lines.append("")
report_lines.append(f"Run ID: {RUN_ID}")
report_lines.append(f"Seed: {SEED}")
report_lines.append(f"Profile: {PROFILE}")
report_lines.append("")
report_lines.append("## Controlled factors")
report_lines.append("")
report_lines.append("Primary factorial: representation (affine second-order vs conformal first-integral) × sampling (fixed vs RAR).")
report_lines.append("Scaling control: affine second-order equations with normalized input/output increments and explicit residual multipliers derived only from prescribed initial data and interval scales.")
report_lines.append("")
report_lines.append("## Measured errors")
report_lines.append("")
report_lines.append("| Representation | RAR | eta(lambda) rel L2 | x(lambda) rel L2 | max p residual | max kappa residual |")
report_lines.append("|---|---:|---:|---:|---:|---:|")
for _, row in paper_table.iterrows():
    report_lines.append(
        f"| {row['representation']} | {bool(row['rar'])} | "
        f"{row['eta_lambda_rel_L2']:.6e} | {row['x_lambda_rel_L2']:.6e} | "
        f"{row['max_momentum_residual']:.6e} | {row['max_kappa_residual']:.6e} |"
    )
report_lines.append("")
report_lines.append("## Reference consistency")
report_lines.append("")
report_lines.append(f"Raw-vs-reduced reference max |x| discrepancy: {REF['raw_reduced_x_max_abs']:.6e}")
report_lines.append(f"Raw-vs-reduced reference max |lambda| discrepancy: {REF['raw_reduced_lambda_max_abs']:.6e}")
report_lines.append("")
report_lines.append("## Interpretation discipline")
report_lines.append("")
report_lines.append("The scaling control tests whether this particular affine output/residual preconditioning materially closes the performance gap. A remaining large conformal advantage strengthens the representation-level interpretation, but it is not a proof of universal superiority outside the tested benchmark and architecture family.")

with open(RUN_DIR / "Paper1_Fairness_Control_Evidence_Report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Saved evidence files in", RUN_DIR)


# ============================================================
# 13. APPEND ALL FACTORIAL + FAIRNESS CELLS TO GLOBAL REGISTRY
# ============================================================

for _, row in ablation_df.iterrows():
    append_registry_row(row.to_dict())

print("Global registry updated:", RESULTS_ROOT / "run_registry.csv")
print(pd.read_csv(RESULTS_ROOT / "run_registry.csv").tail(len(ablation_df)))


# ============================================================
# 14. PACKAGE THE COMPLETE PAPER 1 RESULTS TREE
# ============================================================

archive_root = RESULTS_ROOT.parent
archive_base = archive_root / f"main_ablation_{RUN_ID}"
zip_path = shutil.make_archive(
    str(archive_base),
    "zip",
    root_dir=str(archive_root),
    base_dir=RESULTS_ROOT.name,
)

# Manifest for easy inspection outside Colab.
manifest_path = RUN_DIR / "EXPORT_MANIFEST.txt"
with open(manifest_path, "w", encoding="utf-8") as f:
    for p in sorted(RESULTS_ROOT.rglob("*")):
        if p.is_file():
            f.write(str(p.relative_to(RESULTS_ROOT)) + "\n")

print("Created ZIP:", zip_path)
print("ZIP size MB:", round(Path(zip_path).stat().st_size / 1024**2, 2))
print("Manifest:", manifest_path)

try:
    from google.colab import files
    files.download(zip_path)
    print("Colab download requested. If the browser does not start it, use the left Files panel and download the ZIP manually.")
except Exception:
    print("Not running inside Colab. The ZIP is still available at:", zip_path)
