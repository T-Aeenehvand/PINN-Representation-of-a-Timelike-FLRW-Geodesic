import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

DTYPE=torch.float64
torch.set_default_dtype(DTYPE)
try:
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
except RuntimeError: pass
DEVICE=torch.device('cpu')
SEED=42
A=1.0; ETA0=0.5; ETA_MAX_TRAIN=1.5; LAM0=0.0; LAM_MAX_REPORT=0.8; KAPPA=1.0
WIDTH=64; DEPTH=5; ADAM_LR=1e-3; LBFGS_LR=0.5; LBFGS_HISTORY=50
CFG=dict(ADAM_EPOCHS_PER_STAGE=250,N_INITIAL=256,N_ADD_PER_STAGE=64,N_STAGES=4,N_VALIDATION=4000,LBFGS_ITERS=120)

def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

def pvec(d):
    if d==1: return np.array([1.0],float)
    if d==2: return np.array([0.8,0.6],float)
    if d==3: return np.array([0.60,0.48,0.64],float)
    raise ValueError(d)

def build_mlp(out_dim):
    layers=[nn.Linear(1,WIDTH),nn.Tanh()]
    for _ in range(DEPTH-1): layers += [nn.Linear(WIDTH,WIDTH),nn.Tanh()]
    layers += [nn.Linear(WIDTH,out_dim)]
    net=nn.Sequential(*layers)
    for l in net:
        if isinstance(l,nn.Linear): nn.init.xavier_normal_(l.weight); nn.init.zeros_(l.bias)
    return net

def deriv(y,x):
    return torch.autograd.grad(y,x,grad_outputs=torch.ones_like(y),create_graph=True,retain_graph=True)[0]

def a_np(e): return A*np.asarray(e)**2
def a_t(e): return A*e**2

def raw_system(lam,y,p):
    d=len(p); eta=y[0]; ep=y[1+d]; xps=y[2+d:]; H=2/eta
    return np.concatenate(([ep],xps,[-H*(ep**2+np.dot(xps,xps))],-2*H*ep*xps))

def build_reference(d):
    p=pvec(d); P2=float(p@p); a0=A*ETA0**2; ep0=math.sqrt(P2+KAPPA*a0*a0)/a0**2; xp0=p/a0**2
    y0=np.concatenate(([ETA0],np.zeros(d),[ep0],xp0))
    raw=solve_ivp(lambda l,y:raw_system(l,y,p),(LAM0,LAM_MAX_REPORT),y0,method='DOP853',rtol=1e-12,atol=1e-12,dense_output=True)
    if not raw.success: raise RuntimeError(raw.message)
    def red(eta,y):
        aa=A*eta**2; den=math.sqrt(P2+KAPPA*aa*aa); return np.concatenate((p/den,[aa*aa/den]))
    redsol=solve_ivp(red,(ETA0,ETA_MAX_TRAIN),np.zeros(d+1),method='DOP853',rtol=1e-13,atol=1e-13,dense_output=True)
    return dict(d=d,p=p,P2=P2,raw=raw,red=redsol)

class Net(nn.Module):
    def __init__(self,ref):
        super().__init__(); self.ref=ref; self.d=ref['d']; self.net=build_mlp(self.d+1)
        aa=A*ETA0**2; den=math.sqrt(ref['P2']+KAPPA*aa*aa); self.dx0=ref['p']/den; self.dl0=aa*aa/den
    def forward(self,eta):
        s=2*(eta-ETA0)/(ETA_MAX_TRAIN-ETA0)-1; raw=self.net(s); de=eta-ETA0
        dx0=torch.tensor(self.dx0.reshape(1,-1),dtype=DTYPE,device=eta.device)
        x=dx0*de+de**2*raw[:,:self.d]
        lam=LAM0+self.dl0*de+de**2*raw[:,self.d:self.d+1]
        return x,lam

def residuals(model,eta):
    eta=eta.detach().clone().requires_grad_(True); x,lam=model(eta)
    dx=torch.cat([deriv(x[:,j:j+1],eta) for j in range(model.d)],dim=1); dl=deriv(lam,eta)
    aa=a_t(eta); den=torch.sqrt(model.ref['P2']+KAPPA*aa**2); pv=torch.tensor(model.ref['p'].reshape(1,-1),dtype=DTYPE)
    return x,lam,dx,dl,dx-pv/den,dl-aa**2/den

def loss(model,grid):
    x,lam,dx,dl,Rx,Rl=residuals(model,grid); Lx=torch.mean(Rx**2); Ll=torch.mean(Rl**2); return Lx+Ll,Lx,Ll

def cheb(n):
    j=np.arange(n); nodes=np.cos(np.pi*(2*j+1)/(2*n)); vals=ETA0+0.5*(1-nodes)*(ETA_MAX_TRAIN-ETA0); vals=np.unique(np.sort(np.r_[ETA0,ETA_MAX_TRAIN,vals])); return torch.tensor(vals.reshape(-1,1),dtype=DTYPE)

def rel(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-15))

def train_dim(d,outroot):
    seed_everything(SEED); ref=build_reference(d); model=Net(ref).to(DEVICE); hist=[]; t0=time.time(); current=None
    for st in range(CFG['N_STAGES']+1):
        current=cheb(CFG['N_INITIAL']+st*CFG['N_ADD_PER_STAGE']); opt=optim.Adam(model.parameters(),lr=ADAM_LR)
        for ep in range(CFG['ADAM_EPOCHS_PER_STAGE']):
            opt.zero_grad(set_to_none=True); L,Lx,Ll=loss(model,current); L.backward(); opt.step()
            hist.append([st,ep+1,float(L.detach()),float(Lx.detach()),float(Ll.detach()),len(current)])
            if ep in (0,CFG['ADAM_EPOCHS_PER_STAGE']//2-1,CFG['ADAM_EPOCHS_PER_STAGE']-1): print(f'd={d} stage={st} ep={ep+1} n={len(current)} loss={L.item():.3e}',flush=True)
    lb=optim.LBFGS(model.parameters(),lr=LBFGS_LR,max_iter=CFG['LBFGS_ITERS'],history_size=LBFGS_HISTORY,line_search_fn='strong_wolfe')
    def closure():
        lb.zero_grad(set_to_none=True); L,_,_=loss(model,current); L.backward(); return L
    lb.step(closure); wall=time.time()-t0; st=lb.state.get(next(iter(model.parameters())),{})
    # native grid
    eg=np.linspace(ETA0,ETA_MAX_TRAIN,CFG['N_VALIDATION']); et=torch.tensor(eg.reshape(-1,1),dtype=DTYPE); x,lp,dx,dl,Rx,Rl=residuals(model,et)
    xp=x.detach().numpy(); lp=lp.detach().numpy().ravel(); dx=dx.detach().numpy(); dl=dl.detach().numpy().ravel(); Rx=Rx.detach().numpy(); Rl=Rl.detach().numpy().ravel(); ry=ref['red'].sol(eg); xr=ry[:d].T; lr=ry[d]
    covers=bool(lp[0]<=LAM0+1e-12 and lp[-1]>=LAM_MAX_REPORT-1e-12 and np.all(np.diff(lp)>0) and np.all(dl>0))
    le=np.linspace(LAM0,LAM_MAX_REPORT,CFG['N_VALIDATION']); raw=ref['raw'].sol(le); er=raw[0]; xrl=raw[1:1+d].T
    if covers:
        erec=PchipInterpolator(lp,eg,extrapolate=False)(le); xrec=np.column_stack([PchipInterpolator(lp,xp[:,j],extrapolate=False)(le) for j in range(d)])
    else:
        erec=np.full_like(le,np.nan); xrec=np.full_like(xrl,np.nan)
    aa=a_np(eg); dxdl=dx/dl[:,None]; pres=aa[:,None]**2*dxdl-ref['p'][None,:]; kres=aa**2*((1/dl)**2-np.sum(dxdl**2,axis=1))-KAPPA
    ph=ref['p']/math.sqrt(ref['P2']); perp=xp-(xp@ph)[:,None]*ph[None,:]
    metrics=dict(d=d,p_vector=';'.join(f'{v:.8g}' for v in ref['p']),P_norm=math.sqrt(ref['P2']),eta_train_max=ETA_MAX_TRAIN,lambda_pred_at_eta_max=float(lp[-1]),covers_lambda_0p8=covers,
        x_eta_vector_rel_L2=rel(xp,xr),lambda_eta_rel_L2=rel(lp,lr),eta_lambda_rel_L2=(rel(erec,er) if covers else np.nan),x_lambda_vector_rel_L2=(rel(xrec,xrl) if covers else np.nan),
        max_spatial_residual=float(np.max(np.abs(Rx))),max_lambda_residual=float(np.max(np.abs(Rl))),max_momentum_vector_residual=float(np.max(np.linalg.norm(pres,axis=1))),max_kappa_residual=float(np.max(np.abs(kres))),max_perpendicular_displacement=float(np.max(np.linalg.norm(perp,axis=1))),min_dlambda_deta=float(np.min(dl)),
        seed=SEED,width=WIDTH,depth=DEPTH,adam_epochs_total=CFG['ADAM_EPOCHS_PER_STAGE']*(CFG['N_STAGES']+1),final_collocation=len(current),lbfgs_requested=CFG['LBFGS_ITERS'],lbfgs_actual_iters=int(st.get('n_iter',-1)),lbfgs_func_evals=int(st.get('func_evals',-1)),wall_seconds=wall)
    out=Path(outroot)/f'd{d}'; out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/'model.pt'); pd.DataFrame(hist,columns=['stage','epoch','loss','Lx','Llambda','n']).to_csv(out/'training_history.csv',index=False); pd.DataFrame([metrics]).to_csv(out/'metrics.csv',index=False); np.savez(out/'validation_arrays.npz',eta=eg,x_pred=xp,lambda_pred=lp,x_ref=xr,lambda_ref=lr,lambda_eval=le,eta_recon=erec,x_recon=xrec,eta_ref_lambda=er,x_ref_lambda=xrl,Rx=Rx,Rlambda=Rl,momentum_residual=pres,kappa_residual=kres); (out/'config.json').write_text(json.dumps(dict(A=A,eta0=ETA0,eta_train_max=ETA_MAX_TRAIN,lambda_report_max=LAM_MAX_REPORT,kappa=KAPPA,p_vector=ref['p'].tolist(),seed=SEED,width=WIDTH,depth=DEPTH,optimizer=dict(adam_lr=ADAM_LR,lbfgs_lr=LBFGS_LR,lbfgs_history=LBFGS_HISTORY),training=CFG),indent=2)); print(pd.DataFrame([metrics]).to_string(index=False),flush=True); return metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dims',default='1,2,3'); ap.add_argument('--outdir',default='multid_extension_results'); a=ap.parse_args(); Path(a.outdir).mkdir(parents=True,exist_ok=True); rows=[]
    for d in map(int,a.dims.split(',')): rows.append(train_dim(d,a.outdir))
    pd.DataFrame(rows).to_csv(Path(a.outdir)/'dimension_extension_summary.csv',index=False)
if __name__=='__main__': main()
