#!/usr/bin/env python3
"""Regenerate the supplementary 2D/3D Cartesian extension figure.

Reads validation_arrays.npz and metrics.csv from results/dimensional_extension/d1,d2,d3.
No retraining is performed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/dimensional_extension')
    ap.add_argument('--out', default='figures/dimensional_extension')
    args = ap.parse_args()
    root = Path(args.results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays = {d: np.load(root / f'd{d}' / 'validation_arrays.npz') for d in (1,2,3)}
    metrics = {d: pd.read_csv(root / f'd{d}' / 'metrics.csv').iloc[0] for d in (1,2,3)}

    fig = plt.figure(figsize=(12.4, 4.1), constrained_layout=True)

    ax1 = fig.add_subplot(1,3,1)
    a2 = arrays[2]
    ax1.plot(a2['x_ref'][:,0], a2['x_ref'][:,1], '--', linewidth=1.6, label='reference')
    ax1.plot(a2['x_pred'][:,0], a2['x_pred'][:,1], linewidth=1.35, label='PINN')
    ax1.set_xlabel(r'$x^1$'); ax1.set_ylabel(r'$x^2$')
    ax1.set_title('(a) Two spatial dimensions')
    ax1.grid(alpha=.25); ax1.legend(frameon=True, fontsize=8)

    ax2 = fig.add_subplot(1,3,2, projection='3d')
    a3 = arrays[3]
    ax2.plot(a3['x_ref'][:,0], a3['x_ref'][:,1], a3['x_ref'][:,2], '--', linewidth=1.6, label='reference')
    ax2.plot(a3['x_pred'][:,0], a3['x_pred'][:,1], a3['x_pred'][:,2], linewidth=1.35, label='PINN')
    ax2.set_xlabel(r'$x^1$'); ax2.set_ylabel(r'$x^2$'); ax2.set_zlabel(r'$x^3$')
    ax2.set_title('(b) Three spatial dimensions')
    ax2.legend(frameon=True, fontsize=8)

    ax3 = fig.add_subplot(1,3,3)
    ds = np.array([1,2,3])
    eta_err = np.array([float(metrics[d]['eta_lambda_rel_L2']) for d in ds])
    x_err = np.array([float(metrics[d]['x_lambda_vector_rel_L2']) for d in ds])
    w=.34
    ax3.bar(ds-w/2, eta_err, width=w, label=r'$\eta(\lambda)$')
    ax3.bar(ds+w/2, x_err, width=w, label=r'$\mathbf{x}(\lambda)$')
    ax3.set_yscale('log'); ax3.set_xticks(ds, [r'$d=1$',r'$d=2$',r'$d=3$'])
    ax3.set_ylabel('Relative $L^2$ error', labelpad=3); ax3.yaxis.set_label_coords(-0.12, 0.5); ax3.set_title('(c) Extension-run errors')
    ax3.grid(axis='y', alpha=.25, which='both'); ax3.legend(frameon=True, fontsize=8)

    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f'wrote {out.with_suffix(".pdf")} and {out.with_suffix(".png")}')

if __name__ == '__main__':
    main()
