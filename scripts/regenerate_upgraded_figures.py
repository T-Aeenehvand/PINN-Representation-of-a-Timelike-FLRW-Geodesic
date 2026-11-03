"""Regenerate publication figures for the representation controls and seed screen."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; FIG=ROOT/'figures'; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':9.5,'axes.labelsize':10.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.2,'pdf.fonttype':42,'ps.fonttype':42})
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.06); fig.savefig(FIG/f'{name}.png',dpi=600,bbox_inches='tight',pad_inches=.06); plt.close(fig)
# Short-budget deterministic controls, keeping each representation on its defined reporting grid.
afi=pd.read_csv(ROOT/'results/upgraded_full_controls/seed42/afi__rar_0/metrics.csv').iloc[0]
c2=pd.read_csv(ROOT/'results/upgraded_full_controls/seed42/c2__adam_only/metrics.csv').iloc[0]
cfi=pd.read_csv(ROOT/'results/upgraded_full_controls/seed42/cfi__rar_0/metrics.csv').iloc[0]
fig,axs=plt.subplots(1,3,figsize=(9.2,3.55),sharey=True)
for ax,title,vals,labels in [
    (axs[0],'AFI: affine reconstruction',[afi.eta_lambda_rel_L2,afi.x_lambda_rel_L2],[r'$\eta(\lambda)$',r'$x(\lambda)$']),
    (axs[1],'C2: native conformal grid',[c2.x_eta_rel_L2,c2.lambda_eta_rel_L2],[r'$x(\eta)$',r'$\lambda(\eta)$']),
    (axs[2],'CFI: native conformal grid',[cfi.x_eta_rel_L2,cfi.lambda_eta_rel_L2],[r'$x(\eta)$',r'$\lambda(\eta)$'])]:
    ax.bar(np.arange(2),vals); ax.set_xticks(np.arange(2)); ax.set_xticklabels(labels); ax.set_title(title); ax.set_yscale('log'); ax.grid(axis='y',alpha=.25)
axs[0].set_ylabel('Relative $L^2$ error'); fig.tight_layout(); save(fig,'factorized_ablation_seed42')
# Five-seed screen on the same affine reporting grid after the padded CFI training interval.
a2=pd.read_csv(ROOT/'results/seed_sweep_adam/all_runs.csv'); a2=a2[(a2.representation=='a2')&(~a2.rar)].sort_values('seed')
afi2=pd.read_csv(ROOT/'results/seed_sweep_adam/all_runs.csv'); afi2=afi2[(afi2.representation=='afi')&(~afi2.rar)].sort_values('seed')
cfi2=pd.DataFrame([pd.read_csv(f).iloc[0] for f in sorted((ROOT/'results/seed_sweep_adam_padded_cfi').glob('seed_*/metrics.csv'))]).sort_values('seed')
fig,axs=plt.subplots(1,2,figsize=(8.8,3.7),sharey=True)
for ax,metric,title in [(axs[0],'eta_lambda_rel_L2',r'$\eta(\lambda)$ on affine grid'),(axs[1],'x_lambda_rel_L2',r'$x(\lambda)$ on affine grid')]:
    ax.plot(a2.seed,a2[metric],'o-',label='A2'); ax.plot(afi2.seed,afi2[metric],'s-',label='AFI'); ax.plot(cfi2.seed,cfi2[metric],'^-',label='CFI'); ax.set_xlabel('Seed'); ax.set_title(title); ax.set_yscale('log'); ax.grid(alpha=.25)
axs[0].set_ylabel('Relative $L^2$ error'); axs[0].legend(frameon=False); fig.tight_layout(); save(fig,'phase_two_seed_sensitivity')
# Direct quadrature comparison.
q=pd.read_csv(ROOT/'results/upgraded_full_controls/quadrature_baseline.csv').iloc[0]; fig,ax=plt.subplots(figsize=(6.8,4.0)); errs=[max(q.eta_lambda_rel_L2,q.x_lambda_rel_L2),8.811672888431619e-5]; ax.bar(np.arange(2),errs); ax.set_yscale('log'); ax.set_ylabel('Maximum relative $L^2$ trajectory error'); ax.set_xticks(np.arange(2)); ax.set_xticklabels(['Trapezoid + PCHIP','Baseline CFI\nseed 42']); ax.grid(axis='y',alpha=.25); fig.tight_layout(); save(fig,'quadrature_vs_pinn')
# Full-budget AFI continuation. Both points use 7000 Adam updates; only the
# L-BFGS budget changes, so this figure is a like-for-like optimizer extension.
afi220 = pd.read_csv(ROOT/'results/afi_full_budget_incremental/metrics.csv').iloc[0]
afi500 = pd.read_csv(ROOT/'results/afi_full_budget_500/metrics.csv').iloc[0]
fig,ax=plt.subplots(figsize=(6.6,3.9))
xx=np.array([220,500])
ax.plot(xx,[afi220.eta_lambda_rel_L2,afi500.eta_lambda_rel_L2],'o-',label=r'AFI: $\eta(\lambda)$')
ax.plot(xx,[afi220.x_lambda_rel_L2,afi500.x_lambda_rel_L2],'s-',label=r'AFI: $x(\lambda)$')
ax.axhline(8.81167e-5,linestyle='--',label=r'CFI: $\eta(\lambda)$')
ax.axhline(1.70435e-4,linestyle=':',label=r'CFI: $x(\lambda)$')
ax.set_yscale('log'); ax.set_xlabel('L-BFGS iterations'); ax.set_ylabel('Relative $L^2$ error')
ax.set_xticks(xx); ax.grid(axis='y',alpha=.25); ax.legend(frameon=False,ncol=2); fig.tight_layout(); save(fig,'afi_lbfgs_extension')
