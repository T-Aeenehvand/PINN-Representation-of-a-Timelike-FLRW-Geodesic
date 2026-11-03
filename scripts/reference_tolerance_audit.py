"""Post-training reference-solver audit for the matter-era benchmark.

This script does not train neural networks.  It compares the terminal state of
DOP853 affine integrations at several tolerances with a high-precision mpmath
quadrature/root calculation of the equivalent first-integral problem.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reference_tolerance_audit.csv"
eta0 = 0.5
x0 = 0.0
p = 1.0
kappa = 1.0
lam0 = 0.0
lammax = 0.8
A = 1.0

a0 = A * eta0**2
eta_p0 = np.sqrt(p*p + kappa*a0*a0) / (a0*a0)
x_p0 = p / (a0*a0)

def rawsys(lam, y):
    eta, x, eta_p, x_p = y
    Hc = 2.0 / eta
    return [eta_p, x_p, -Hc*(eta_p*eta_p + x_p*x_p), -2.0*Hc*eta_p*x_p]

# High-precision reduced reference.
mp.mp.dps = 80
e0 = mp.mpf(str(eta0))
Ltarget = mp.mpf(str(lammax))
f = lambda u: 1 / mp.sqrt(1 + u**4)
g = lambda u: u**4 / mp.sqrt(1 + u**4)
L = lambda e: mp.quad(g, [e0, e])
eta_hp = mp.findroot(lambda e: L(e) - Ltarget, (mp.mpf("1.4"), mp.mpf("1.6")))
x_hp = mp.quad(f, [e0, eta_hp])
eta_hp_f = float(eta_hp)
x_hp_f = float(x_hp)

rows = []
for tol in [1e-9, 1e-10, 1e-11, 1e-12, 1e-13]:
    sol = solve_ivp(
        rawsys, (lam0, lammax), [eta0, x0, eta_p0, x_p0],
        method="DOP853", rtol=tol, atol=tol, dense_output=True,
    )
    eta_end, x_end = map(float, sol.sol(lammax)[:2])
    rows.append({
        "rtol_atol": tol,
        "nfev": sol.nfev,
        "eta_end": eta_end,
        "x_end": x_end,
        "abs_eta_vs_high_precision": abs(eta_end - eta_hp_f),
        "abs_x_vs_high_precision": abs(x_end - x_hp_f),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"High-precision eta(lambda=0.8) = {mp.nstr(eta_hp, 50)}")
print(f"High-precision x(lambda=0.8)   = {mp.nstr(x_hp, 50)}")
print(df.to_string(index=False, float_format=lambda z: f"{z:.16e}"))
print(f"Wrote {OUT}")
