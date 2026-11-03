import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/afi_full_budget_incremental'; OUT.mkdir(parents=True,exist_ok=True)
torch.set_default_dtype(torch.float64)
SEED=42; A=1.; ETA0=.5; P=1.; KAPPA=1.; L0=0.; LMAX=.8; W=64; DEPTH=5; LR=1e-3; BLR=.5; BH=50

def init(): np.random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
def grid(n):
 j=np.arange(n); z=np.cos(np.pi*(2*j+1)/(2*n)); vals=L0+.5*(1-z)*LMAX; return torch.tensor(np.unique(np.sort(np.r_[L0,LMAX,vals]))[:,None],dtype=torch.float64)
def d(y,x): return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]
a0=A*ETA0**2; ep0=np.sqrt(P**2+KAPPA*a0*a0)/a0**2; xp0=P/a0**2
def net():
 m=torch.nn.Sequential(torch.nn.Linear(1,W),torch.nn.Tanh(),*[v for _ in range(4) for v in (torch.nn.Linear(W,W),torch.nn.Tanh())],torch.nn.Linear(W,2))
 for q in m:
  if isinstance(q,torch.nn.Linear): torch.nn.init.xavier_normal_(q.weight); torch.nn.init.zeros_(q.bias)
 return m
def out(m,z):
 s=2*z/LMAX-1; r=m(s); q=z; return ETA0+ep0*q+q*q*r[:,:1], q*xp0+q*q*r[:,1:]
def residual(m,z):
 z=z.detach().clone().requires_grad_(True); eta,x=out(m,z); ep=d(eta,z); xp=d(x,z); aa=eta**2; dd=torch.sqrt(1+aa**2); return eta,x,ep,xp,ep-dd/aa**2,xp-1/aa**2
ap=argparse.ArgumentParser(); ap.add_argument('--stage',type=int,default=-1); ap.add_argument('--lbfgs-chunk',type=int,default=0); args=ap.parse_args(); init(); p=OUT/'model.pt'
if not p.exists(): m=net()
else: m=net(); m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
if args.stage>=0:
 z=grid(256+64*args.stage); opt=torch.optim.Adam(m.parameters(),lr=LR); t=time.time()
 for ep in range(1400):
  opt.zero_grad(set_to_none=True); *_,re,rx=residual(m,z); L=re.square().mean()+rx.square().mean(); L.backward(); opt.step()
 print('stage',args.stage,'seconds',time.time()-t,'loss',float(L.detach()),flush=True); torch.save(m.state_dict(),p); (OUT/f'stage_{args.stage}.json').write_text(json.dumps({'stage':args.stage,'collocation':len(z),'adam_updates':1400,'loss':float(L.detach())},indent=2)); raise SystemExit
# L-BFGS chunk; optimizer state persists between chunks.
opt_path=OUT/'lbfgs_state.pt'; z=grid(512); lb=torch.optim.LBFGS(m.parameters(),lr=BLR,max_iter=args.lbfgs_chunk,history_size=BH,line_search_fn='strong_wolfe')
if opt_path.exists(): lb.load_state_dict(torch.load(opt_path,map_location='cpu',weights_only=False))
def closure():
 lb.zero_grad(set_to_none=True); *_,re,rx=residual(m,z); L=re.square().mean()+rx.square().mean(); L.backward(); return L
if args.lbfgs_chunk>0:
 t=time.time(); lb.step(closure); torch.save(m.state_dict(),p); torch.save(lb.state_dict(),opt_path); st=lb.state[next(iter(m.parameters()))]; print('lbfgs chunk',args.lbfgs_chunk,'time',time.time()-t,'n_iter',st.get('n_iter'),'fe',st.get('func_evals'),'loss',float(closure().detach()),flush=True)
else:
 # final evaluation
 le=np.linspace(0,.8,4000); zz=torch.tensor(le[:,None],dtype=torch.float64); eta,x,*_=residual(m,zz); eta=eta.detach().numpy().ravel(); x=x.detach().numpy().ravel(); ref=solve_ivp(lambda l,y:[y[2],y[3],-(2/y[0])*(y[2]**2+y[3]**2),-(4/y[0])*y[2]*y[3]],(0,.8),[.5,0,ep0,xp0],method='DOP853',rtol=1e-12,atol=1e-12,dense_output=True); er,xr=ref.sol(le)[:2]; row={'eta_lambda_rel_L2':float(np.linalg.norm(eta-er)/np.linalg.norm(er)),'x_lambda_rel_L2':float(np.linalg.norm(x-xr)/np.linalg.norm(xr))}; print(row); pd.DataFrame([row]).to_csv(OUT/'metrics.csv',index=False)
