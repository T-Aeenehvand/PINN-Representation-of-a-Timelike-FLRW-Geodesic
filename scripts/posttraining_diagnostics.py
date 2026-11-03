"""Additional post-training diagnostics for the archived seed-42 checkpoints.

This script does not retrain any network. It reloads the saved conformal
checkpoints and evaluates (i) interpolation-density convergence of the
lambda(eta) inversion and (ii) a dimensionless local second-order geodesic
balance ratio induced by the learned first-integral solution.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)
torch.set_num_threads(1)

A = 1.0
eta0 = 0.5
x0 = 0.0
p = 1.0
kappa = 1.0
lam0 = 0.0
lammax = 0.8
WIDTH = 64
DEPTH = 5

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "archived_seed42" / "six_cell_run"
OUT = ROOT / "results"


def build_mlp(in_dim=1, out_dim=2, width=64, depth=5):
    layers = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, out_dim)]
    return nn.Sequential(*layers)


def initial_affine_state():
    a0 = A * eta0**2
    xp0 = p / a0**2
    ep0 = np.sqrt(p**2 + kappa * a0**2) / a0**2
    return ep0, xp0


def raw_system(lam, y):
    eta, x, ep, xp = y
    Hc = 2.0 / eta
    return [ep, xp, -Hc * (ep**2 + xp**2), -2.0 * Hc * ep * xp]


def deriv(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]


class Conformal(nn.Module):
    def __init__(self, eta_max):
        super().__init__()
        self.net = build_mlp(width=WIDTH, depth=DEPTH)
        self.eta_max = float(eta_max)
        a0 = A * eta0**2
        den0 = np.sqrt(p**2 + kappa * a0**2)
        self.dx0 = p / den0
        self.dl0 = a0**2 / den0

    def forward(self, eta):
        s = 2.0 * (eta - eta0) / (self.eta_max - eta0) - 1.0
        raw = self.net(s)
        d = eta - eta0
        x = x0 + self.dx0 * d + d**2 * raw[:, 0:1]
        lam = lam0 + self.dl0 * d + d**2 * raw[:, 1:2]
        return x, lam


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


ep0, xp0 = initial_affine_state()
ref = solve_ivp(
    raw_system, (lam0, lammax), [eta0, x0, ep0, xp0],
    method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True
)
if not ref.success:
    raise RuntimeError(ref.message)
eta_max = float(ref.sol(lammax)[0])

# Fixed affine reporting grid for inversion-density convergence.
N_REPORT = 4000
lam_report = np.linspace(lam0, lammax, N_REPORT)
eta_ref_report, x_ref_report = ref.sol(lam_report)[:2]

models = [
    ("Deterministic", "conformal_first_integral__rar_off"),
    ("RAR", "conformal_first_integral__rar_on"),
]

conv_rows = []
balance_rows = []

for sampling, dirname in models:
    model = Conformal(eta_max)
    state = torch.load(RUN / dirname / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Interpolation-density convergence: only the sampling used to tabulate the
    # learned smooth lambda(eta) map changes; the network and reporting grid do not.
    for n_source in [501, 1001, 2001, 4000, 8001, 20001, 50001]:
        eta_src = np.linspace(eta0, eta_max, n_source)
        with torch.no_grad():
            x_src_t, lam_src_t = model(torch.tensor(eta_src[:, None], dtype=DTYPE))
        x_src = x_src_t.numpy().ravel()
        lam_src = lam_src_t.numpy().ravel()
        if np.any(np.diff(lam_src) <= 0):
            raise RuntimeError(f"Non-monotone lambda(eta) for {sampling}, N={n_source}")
        if lam_src[-1] < lammax:
            raise RuntimeError(f"Insufficient affine coverage for {sampling}, N={n_source}")
        eta_rec = PchipInterpolator(lam_src, eta_src, extrapolate=False)(lam_report)
        x_rec = PchipInterpolator(lam_src, x_src, extrapolate=False)(lam_report)
        conv_rows.append({
            "sampling": sampling,
            "source_points": n_source,
            "eta_lambda_rel_L2": rel_l2(eta_rec, eta_ref_report),
            "x_lambda_rel_L2": rel_l2(x_rec, x_ref_report),
            "lambda_end": float(lam_src[-1]),
        })

    # Dimensionless local cancellation diagnostic for the second-order affine
    # geodesic equations induced by the first-integral network.  Absolute affine
    # acceleration residuals are strongly scale-amplified near eta0 because
    # d lambda / d eta is small, so report both absolute and normalized balances.
    N = 20001
    eta = np.linspace(eta0, eta_max, N)
    et = torch.tensor(eta[:, None], dtype=DTYPE, requires_grad=True)
    x, lam = model(et)
    dx = deriv(x, et)
    dl = deriv(lam, et)
    d2x = deriv(dx, et)
    d2l = deriv(dl, et)

    x_n, lam_n, dx_n, dl_n, d2x_n, d2l_n = [
        z.detach().numpy().ravel() for z in (x, lam, dx, dl, d2x, d2l)
    ]
    eta_p = 1.0 / dl_n
    x_p = dx_n / dl_n
    eta_pp = -d2l_n / dl_n**3
    x_pp = d2x_n / dl_n**2 - dx_n * d2l_n / dl_n**3
    Hc = 2.0 / eta
    term_eta = Hc * (eta_p**2 + x_p**2)
    term_x = 2.0 * Hc * eta_p * x_p
    geo_eta = eta_pp + term_eta
    geo_x = x_pp + term_x

    denom_eta = np.abs(eta_pp) + np.abs(term_eta)
    denom_x = np.abs(x_pp) + np.abs(term_x)
    rho_eta = np.abs(geo_eta) / np.maximum(denom_eta, 1e-30)
    rho_x = np.abs(geo_x) / np.maximum(denom_x, 1e-30)

    a = A * eta**2
    den = np.sqrt(p**2 + kappa * a**2)
    f = p / den
    g = a**2 / den
    rx = dx_n - f
    rl = dl_n - g

    balance_rows.append({
        "sampling": sampling,
        "N": N,
        "max_abs_first_residual_x": float(np.max(np.abs(rx))),
        "max_abs_first_residual_lambda": float(np.max(np.abs(rl))),
        "max_abs_induced_geo_eta": float(np.max(np.abs(geo_eta))),
        "max_abs_induced_geo_x": float(np.max(np.abs(geo_x))),
        "max_balance_ratio_eta": float(np.max(rho_eta)),
        "max_balance_ratio_x": float(np.max(rho_x)),
        "rms_balance_ratio_eta": float(np.sqrt(np.mean(rho_eta**2))),
        "rms_balance_ratio_x": float(np.sqrt(np.mean(rho_x**2))),
        "median_balance_ratio_eta": float(np.median(rho_eta)),
        "median_balance_ratio_x": float(np.median(rho_x)),
        "eta_at_max_balance_eta": float(eta[np.argmax(rho_eta)]),
        "eta_at_max_balance_x": float(eta[np.argmax(rho_x)]),
    })

conv = pd.DataFrame(conv_rows)
balance = pd.DataFrame(balance_rows)
conv.to_csv(OUT / "pchip_inversion_convergence.csv", index=False)
balance.to_csv(OUT / "conformal_second_order_balance.csv", index=False)
print("Wrote", OUT / "pchip_inversion_convergence.csv")
print(conv.to_string(index=False))
print("\nWrote", OUT / "conformal_second_order_balance.csv")
print(balance.to_string(index=False))
