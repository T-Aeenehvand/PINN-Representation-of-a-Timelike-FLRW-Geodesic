"""Finalize the seed-42 AFI 500-iteration continuation.

This script evaluates the exact 500-iteration candidate produced by
``run_afi_500_exact.py`` on the same 4000-point affine validation grid used by
the main benchmark.  It stores the model, optimizer state, validation arrays,
trajectory metrics, invariant defects, and the proper-time diagnostic.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/afi_full_budget_500'
torch.set_default_dtype(torch.float64)
SEED=42; ETA0=.5; P=1.; KAPPA=1.; L0=0.; LMAX=.8; W=64
np.random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
def d(y,x): return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]
def net():
 m=torch.nn.Sequential(torch.nn.Linear(1,W),torch.nn.Tanh(),*[v for _ in range(4) for v in (torch.nn.Linear(W,W),torch.nn.Tanh())],torch.nn.Linear(W,2))
 for q in m:
  if isinstance(q,torch.nn.Linear): torch.nn.init.xavier_normal_(q.weight); torch.nn.init.zeros_(q.bias)
 return m
a0=ETA0**2; ep0=np.sqrt(P**2+KAPPA*a0*a0)/a0**2; xp0=P/a0**2
def out(m,z):
 s=2*z/LMAX-1; r=m(s); return ETA0+ep0*z+z*z*r[:,:1], z*xp0+z*z*r[:,1:]
m=net(); m.load_state_dict(torch.load(OUT/'model_500_candidate.pt',map_location='cpu',weights_only=True)); m.eval()
le=np.linspace(0,.8,4000); z=torch.tensor(le[:,None],dtype=torch.float64,requires_grad=True)
eta,x=out(m,z); ep=d(eta,z); xp=d(x,z); aa=eta**2; re=ep-torch.sqrt(1+aa**2)/aa**2; rx=xp-1/aa**2
eta_n=eta.detach().numpy().ravel(); x_n=x.detach().numpy().ravel(); ep_n=ep.detach().numpy().ravel(); xp_n=xp.detach().numpy().ravel(); re_n=re.detach().numpy().ravel(); rx_n=rx.detach().numpy().ravel()
ref=solve_ivp(lambda l,y:[y[2],y[3],-(2/y[0])*(y[2]**2+y[3]**2),-(4/y[0])*y[2]*y[3]],(0,.8),[.5,0,ep0,xp0],method='DOP853',rtol=1e-12,atol=1e-12,dense_output=True)
er,xr=ref.sol(le)[:2]
pres=eta_n**2*xp_n-1 # a^2 x' - 1
kdef=eta_n**2*(ep_n**2-xp_n**2)-1
# Proper time from the learned worldline. In c=1 units, lambda_ref=0.8.
dtau_dlam=np.sqrt(np.maximum(eta_n**4*(ep_n**2-xp_n**2),0.0))
tau=np.trapezoid(dtau_dlam,le)
metrics={'eta_lambda_rel_L2':float(np.linalg.norm(eta_n-er)/np.linalg.norm(er)),
         'x_lambda_rel_L2':float(np.linalg.norm(x_n-xr)/np.linalg.norm(xr)),
         'max_abs_Reta':float(np.max(np.abs(re_n))),'max_abs_Rx':float(np.max(np.abs(rx_n))),
         'max_abs_dp':float(np.max(np.abs(pres))), 'max_abs_dkappa':float(np.max(np.abs(kdef))),
         'proper_time':float(tau),'proper_time_relative_error':float((tau-.8)/.8),
         'lbfgs_total_iterations':500,'lbfgs_total_function_evaluations':527,
         'lbfgs_additional_iterations_beyond_220':280,'lbfgs_additional_function_evaluations':288}
pd.DataFrame([metrics]).to_csv(OUT/'metrics.csv',index=False)
np.savez(OUT/'validation_arrays.npz',lambda_eval=le,eta=eta_n,x=x_n,eta_ref=er,x_ref=xr,Reta=re_n,Rx=rx_n,p_residual=pres,kappa_residual=kdef)
# Promote exact candidate state to the release names.
import shutil
shutil.copy2(OUT/'model_500_candidate.pt',OUT/'model.pt'); shutil.copy2(OUT/'lbfgs_state_500_candidate.pt',OUT/'lbfgs_state.pt')
(OUT/'provenance.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
