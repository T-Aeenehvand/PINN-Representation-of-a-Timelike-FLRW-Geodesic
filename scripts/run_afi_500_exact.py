#!/usr/bin/env python3
"""Run the exact 500-iteration AFI continuation used in the manuscript.

The source checkpoint is the completed seed-42 AFI run with 7000 Adam updates
and 220 realized L-BFGS iterations.  The script restores that checkpoint and
its accumulated L-BFGS curvature history, resets only the per-call optimizer
counters, and performs exactly 280 additional L-BFGS iterations.  No Adam
updates are performed.  Results are written to ``results/afi_full_budget_500``.
"""
from pathlib import Path
import json, time
import numpy as np, pandas as pd, torch
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'results/afi_full_budget_incremental'; OUT=ROOT/'results/afi_full_budget_500'; OUT.mkdir(parents=True,exist_ok=True)
torch.set_default_dtype(torch.float64); SEED=42; ETA0=.5; P=1.; KAPPA=1.; L0=0.; LMAX=.8; W=64; BLR=.5; BH=50
np.random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
def grid(n):
 j=np.arange(n); z=np.cos(np.pi*(2*j+1)/(2*n)); vals=L0+.5*(1-z)*LMAX
 return torch.tensor(np.unique(np.sort(np.r_[L0,LMAX,vals]))[:,None],dtype=torch.float64)
def d(y,x): return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]
a0=ETA0**2; ep0=np.sqrt(P**2+KAPPA*a0*a0)/a0**2; xp0=P/a0**2
def net():
 m=torch.nn.Sequential(torch.nn.Linear(1,W),torch.nn.Tanh(),*[v for _ in range(4) for v in (torch.nn.Linear(W,W),torch.nn.Tanh())],torch.nn.Linear(W,2))
 for q in m:
  if isinstance(q,torch.nn.Linear): torch.nn.init.xavier_normal_(q.weight); torch.nn.init.zeros_(q.bias)
 return m
def out(m,z):
 s=2*z/LMAX-1; r=m(s); return ETA0+ep0*z+z*z*r[:,:1], z*xp0+z*z*r[:,1:]
def residual(m,z):
 z=z.detach().clone().requires_grad_(True); eta,x=out(m,z); ep=d(eta,z); xp=d(x,z); aa=eta**2; dd=torch.sqrt(1+aa**2)
 return eta,x,ep,xp,ep-dd/aa**2,xp-1/aa**2
m=net(); m.load_state_dict(torch.load(SRC/'model.pt',map_location='cpu',weights_only=True))
state=torch.load(SRC/'lbfgs_state.pt',map_location='cpu',weights_only=False)
lb=torch.optim.LBFGS(m.parameters(),lr=BLR,max_iter=280,history_size=BH,line_search_fn='strong_wolfe')
lb.load_state_dict(state)
pg=lb.param_groups[0]; pg['max_iter']=280; pg['max_eval']=350
st=lb.state[next(iter(m.parameters()))]
# The saved state contains cumulative n_iter/func_evals from the previous chunks.
# Reset these per-step counters while retaining the curvature history.
st['n_iter']=0; st['func_evals']=0
z=grid(512)
def closure():
 lb.zero_grad(set_to_none=True); *_,re,rx=residual(m,z); L=re.square().mean()+rx.square().mean(); L.backward(); return L
before=float(closure().detach()); t=time.time(); lb.step(closure); elapsed=time.time()-t; after=float(closure().detach())
st=lb.state[next(iter(m.parameters()))]
additional_iter=int(st.get('n_iter',0)); additional_fe=int(st.get('func_evals',0))
# The source checkpoint is the completed 220-iteration AFI run.  This continuation adds exactly 280 iterations.
prior_total=220; total=prior_total+additional_iter; total_fe_prior=239; total_fe=total_fe_prior+additional_fe
torch.save(m.state_dict(),OUT/'model_500_candidate.pt'); torch.save(lb.state_dict(),OUT/'lbfgs_state_500_candidate.pt')
meta={'prior_total_iterations':prior_total,'additional_iterations':additional_iter,'total_iterations':total,
      'prior_total_function_evaluations':total_fe_prior,'additional_function_evaluations':additional_fe,
      'total_function_evaluations':total_fe,'loss_before':before,'loss_after':after,'elapsed_seconds':elapsed}
(OUT/'lbfgs_500_candidate.json').write_text(json.dumps(meta,indent=2))
# Evaluate candidate on 4000-point affine grid.
le=np.linspace(0,.8,4000); zz=torch.tensor(le[:,None],dtype=torch.float64); eta,x,*_=residual(m,zz); eta=eta.detach().numpy().ravel(); x=x.detach().numpy().ravel()
ref=solve_ivp(lambda l,y:[y[2],y[3],-(2/y[0])*(y[2]**2+y[3]**2),-(4/y[0])*y[2]*y[3]],(0,.8),[.5,0,ep0,xp0],method='DOP853',rtol=1e-12,atol=1e-12,dense_output=True); er,xr=ref.sol(le)[:2]
metrics={'eta_lambda_rel_L2':float(np.linalg.norm(eta-er)/np.linalg.norm(er)),'x_lambda_rel_L2':float(np.linalg.norm(x-xr)/np.linalg.norm(xr)),**meta}
pd.DataFrame([metrics]).to_csv(OUT/'metrics_500_candidate.csv',index=False); print(json.dumps(metrics,indent=2))
