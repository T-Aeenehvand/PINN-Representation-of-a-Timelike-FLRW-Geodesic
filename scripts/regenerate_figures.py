
# === CELL 0 ===
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import LogLocator, LogFormatterMathtext

# ---------------------------------------------------------------------
# Paths and benchmark parameters
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "archived_seed42" / "six_cell_run"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

A = 1.0
p = 1.0
kappa = 1.0
eta0 = 0.5
x0 = 0.0
lam0 = 0.0
lammax = 0.8

# Publication-quality defaults. Use vector PDF/SVG plus high-resolution PNG.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.7,
    "ytick.labelsize": 8.7,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.55,
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": True,
    "legend.framealpha": 0.94,
    "legend.edgecolor": "0.8",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
C_REF = "0.15"
C_PINN = colors[0]
C_DET = colors[0]
C_RAR = colors[1]


def save_figure(fig, stem):
    """Save a tight raster and vector copy for manuscript use."""
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight",
                facecolor="white", pad_inches=0.08)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight",
                facecolor="white", pad_inches=0.08)
    plt.close(fig)

# === CELL 1 ===
# ---------------------------------------------------------------------
# Reference solution
# ---------------------------------------------------------------------
def initial_state():
    a0 = A * eta0**2
    xp0 = p / a0**2
    ep0 = np.sqrt(p**2 + kappa * a0**2) / a0**2
    return ep0, xp0


def raw_system(lam, y):
    eta, x, ep, xp = y
    Hc = 2.0 / eta
    return [ep, xp, -Hc * (ep**2 + xp**2), -2.0 * Hc * ep * xp]


ep0, xp0 = initial_state()
ref = solve_ivp(
    raw_system,
    (lam0, lammax),
    [eta0, x0, ep0, xp0],
    method="DOP853",
    rtol=1e-12,
    atol=1e-12,
    dense_output=True,
)
if not ref.success:
    raise RuntimeError(f"Reference integration failed: {ref.message}")

eta_max = float(ref.sol(lammax)[0])
eta_cross = np.sqrt(abs(p) / A)

# === CELL 2 ===
# ---------------------------------------------------------------------
# Figure 1: physical targets
# ---------------------------------------------------------------------
eta = np.linspace(eta0, eta_max, 1400)
a = A * eta**2
beta = p / np.sqrt(p**2 + a**2)
gamma = np.sqrt(1.0 + p**2 / a**2)
g = a**2 / np.sqrt(p**2 + kappa * a**2)

fig, ax = plt.subplots(
    1, 3, figsize=(10.6, 3.15), constrained_layout=True
)

series = [
    (a, r"Scale factor $a(\eta)$", "Prescribed matter background"),
    (beta, r"$v_{\rm pec}/c=dx/d\eta$", "Peculiar speed"),
    (g, r"$d\lambda/d\eta=a/\gamma$", "Affine/proper-length pacing"),
]
for i, (vals, ylabel, title) in enumerate(series):
    ax[i].plot(eta, vals, color=colors[i], label="Matter-era solution")
    ax[i].axvline(
        eta_cross, ls=":", color="0.25", lw=1.25,
        label=r"$a=|p|$" if i == 0 else None
    )
    marker_vals = np.interp([eta0, eta_cross, eta_max], eta, vals)
    ax[i].scatter(
        [eta0, eta_cross, eta_max], marker_vals,
        s=20, color=colors[i], edgecolor="white", linewidth=0.5,
        zorder=4, label="Reference points"
    )
    ax[i].set_xlabel(r"Conformal time $\eta$")
    ax[i].set_ylabel(ylabel)
    ax[i].set_title(title, pad=8)
    ax[i].margins(x=0.04, y=0.14)
    if i == 0:
        ax[i].legend(loc="upper left", fontsize=7.5)

save_figure(fig, "physics")

# === CELL 3 ===
# ---------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------
df = pd.read_csv(RUN / "ablation_all_results.csv")
order = [
    "affine_second_order",
    "affine_preconditioned",
    "conformal_first_integral",
]
labels = {
    "affine_second_order": "Affine, raw",
    "affine_preconditioned": "Affine, scaled",
    "conformal_first_integral": "Conformal, first integral",
}


def get_metric_values(metric):
    """Return deterministic and RAR values in the declared representation order."""
    off, on = [], []
    for rep in order:
        d0 = df[(df.representation == rep) & (df.rar == False)]
        d1 = df[(df.representation == rep) & (df.rar == True)]
        if d0.empty or d1.empty:
            raise ValueError(f"Missing result row for {rep}, metric {metric}")
        off.append(float(d0.iloc[0][metric]))
        on.append(float(d1.iloc[0][metric]))
    vals = np.asarray(off + on, dtype=float)
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError(f"Metric {metric} must contain finite positive values for log plotting.")
    return np.asarray(off), np.asarray(on)


def grouped_log_bars(ax, metric, ylabel, panel_title):
    """Consistent grouped log-scale bars with readable value labels."""
    vals_off, vals_on = get_metric_values(metric)
    xpos = np.arange(len(order))
    width = 0.32

    bars1 = ax.bar(
        xpos - width / 2, vals_off, width,
        label="Deterministic grid", color=C_DET,
        edgecolor="white", linewidth=0.6, zorder=3
    )
    bars2 = ax.bar(
        xpos + width / 2, vals_on, width,
        label="Residual-adaptive (RAR)", color=C_RAR,
        edgecolor="white", linewidth=0.6, hatch="///",
        alpha=0.9, zorder=3
    )

    ax.set_yscale("log")
    ymin = min(vals_off.min(), vals_on.min()) / 3.0
    ymax = max(vals_off.max(), vals_on.max()) * 5.0
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xticks(xpos)
    ax.set_xticklabels([labels[r] for r in order], rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, pad=9, fontweight="medium")
    ax.grid(axis="y", which="major")
    ax.grid(axis="y", which="minor", alpha=0.10, linewidth=0.4)
    ax.tick_params(axis="x", pad=5)

    # Label bars horizontally above each bar. The extra log-space headroom
    # prevents labels from colliding with panel titles or being clipped.
    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.2g}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.4, rotation=0,
                clip_on=False,
            )
    return bars1, bars2

# === CELL 4 ===
# ---------------------------------------------------------------------
# Figure 2: common affine-grid trajectory errors
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2, figsize=(10.6, 4.05), constrained_layout=True
)
grouped_log_bars(
    axes[0], "eta_lambda_rel_L2",
    r"Relative $L^2$ error", r"$\eta(\lambda)$"
)
grouped_log_bars(
    axes[1], "x_lambda_rel_L2",
    r"Relative $L^2$ error", r"$x(\lambda)$"
)

# A single figure-level legend avoids duplicate legends and overlaps.
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, legend_labels, loc="upper center",
    bbox_to_anchor=(0.5, 1.035), ncol=2,
    frameon=True, fontsize=8.2
)
save_figure(fig, "errors")

# === CELL 5 ===
# ---------------------------------------------------------------------
# Figure 3: invariant diagnostics
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2, figsize=(10.6, 4.05), constrained_layout=True
)
grouped_log_bars(
    axes[0], "max_momentum_residual",
    r"Maximum absolute defect",
    r"$\max|a^2x_{,\lambda}-p|$"
)
grouped_log_bars(
    axes[1], "max_kappa_residual",
    r"Maximum absolute defect",
    r"$\max|a^2(\eta_{,\lambda}^2-x_{,\lambda}^2)-1|$"
)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, legend_labels, loc="upper center",
    bbox_to_anchor=(0.5, 1.035), ncol=2,
    frameon=True, fontsize=8.2
)
save_figure(fig, "invariants")

# === CELL 6 ===
# ---------------------------------------------------------------------
# Figure 4: conformal first-integral trajectory and pointwise errors
# ---------------------------------------------------------------------
arr = np.load(
    RUN / "conformal_first_integral__rar_off" / "validation_arrays.npz"
)
lam = arr["lambda_eval"]
eta_pred = arr["eta_recon"]
x_pred = arr["x_recon"]
eta_ref = arr["eta_ref_lambda"]
x_ref = arr["x_ref_lambda"]

fig, axes = plt.subplots(
    2, 2, figsize=(10.6, 6.5), sharex="col",
    constrained_layout=True
)

# Coordinate reconstruction: reference and PINN have explicit legends.
axes[0, 0].plot(
    lam, eta_ref, color=C_REF, lw=2.0,
    label="DOP853 reference", zorder=3
)
axes[0, 0].plot(
    lam, eta_pred, color=C_PINN, lw=1.7, ls="--",
    label="Conformal first-integral PINN", zorder=4
)
axes[0, 0].set_ylabel(r"$\eta(\lambda)$")
axes[0, 0].set_title("(a) Conformal coordinate", pad=8)
axes[0, 0].legend(loc="best")

axes[1, 0].semilogy(
    lam, np.maximum(np.abs(eta_pred - eta_ref), 1e-14),
    color=C_PINN, lw=1.7, label="Absolute error"
)
axes[1, 0].set_xlabel(r"Affine parameter $\lambda$")
axes[1, 0].set_ylabel("Absolute error")
axes[1, 0].set_title(r"(c) $|\widehat{\eta}-\eta_{\rm ref}|$", pad=8)
axes[1, 0].legend(loc="best")

axes[0, 1].plot(
    lam, x_ref, color=C_REF, lw=2.0,
    label="DOP853 reference", zorder=3
)
axes[0, 1].plot(
    lam, x_pred, color=C_PINN, lw=1.7, ls="--",
    label="Conformal first-integral PINN", zorder=4
)
axes[0, 1].set_ylabel(r"$x(\lambda)$")
axes[0, 1].set_title("(b) Comoving position", pad=8)
axes[0, 1].legend(loc="best")

axes[1, 1].semilogy(
    lam, np.maximum(np.abs(x_pred - x_ref), 1e-14),
    color=C_PINN, lw=1.7, label="Absolute error"
)
axes[1, 1].set_xlabel(r"Affine parameter $\lambda$")
axes[1, 1].set_ylabel("Absolute error")
axes[1, 1].set_title(r"(d) $|\widehat{x}-x_{\rm ref}|$", pad=8)
axes[1, 1].legend(loc="best")

for ax in axes.flat:
    ax.margins(x=0.02)
    ax.grid(True, which="major", alpha=0.22)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.4)

save_figure(fig, "trajectory")

print(f"Wrote publication-quality PNG and PDF figures to: {FIG}")

# === CELL 7 ===

