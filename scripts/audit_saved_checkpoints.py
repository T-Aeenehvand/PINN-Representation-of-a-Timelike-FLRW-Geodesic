import math, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

DTYPE=torch.float64; DEVICE=torch.device('cpu'); torch.set_default_dtype(DTYPE); torch.set_num_threads(1)
A=1.0; eta0=0.5; x0=0.0; p=1.0; kappa=1.0; lam0=0.0; lammax=0.8
WIDTH=64; DEPTH=5
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results'/'archived_seed42'/'six_cell_run'

def build_mlp(in_dim=1,out_dim=2,width=64,depth=5):
    layers=[nn.Linear(in_dim,width),nn.Tanh()]
    for _ in range(depth-1): layers += [nn.Linear(width,width),nn.Tanh()]
    layers += [nn.Linear(width,out_dim)]
    return nn.Sequential(*layers)

def initial_affine_state():
    a0=A*eta0**2
    return np.sqrt(p**2+kappa*a0**2)/a0**2, p/a0**2

def deriv(y,x): return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]

class Affine(nn.Module):
    def __init__(self, pre=False):
        super().__init__(); self.pre=pre; self.net=build_mlp(width=WIDTH,depth=DEPTH); self.eta0p,self.x0p=initial_affine_state(); self.L=lammax-lam0
        if pre:
            self.eta_scale=abs(self.eta0p)*self.L; self.x_scale=abs(self.x0p)*self.L
            self.es=self.eta0p*self.L/self.eta_scale; self.xs=self.x0p*self.L/self.x_scale
    def forward(self,lam):
        if not self.pre:
            s=2*(lam-lam0)/self.L-1; raw=self.net(s); d=lam-lam0
            return eta0+self.eta0p*d+d*d*raw[:,0:1], x0+self.x0p*d+d*d*raw[:,1:2]
        q=(lam-lam0)/self.L; raw=self.net(2*q-1)
        eh=self.es*q+q*q*raw[:,0:1]; xh=self.xs*q+q*q*raw[:,1:2]
        return eta0+self.eta_scale*eh, x0+self.x_scale*xh

class Conformal(nn.Module):
    def __init__(self,etamax):
        super().__init__(); self.net=build_mlp(width=WIDTH,depth=DEPTH); self.etamax=etamax
        a0=A*eta0**2; den=np.sqrt(p*p+kappa*a0*a0); self.dx0=p/den; self.dl0=a0*a0/den
    def forward(self,eta):
        s=2*(eta-eta0)/(self.etamax-eta0)-1; raw=self.net(s); d=eta-eta0
        return x0+self.dx0*d+d*d*raw[:,0:1], lam0+self.dl0*d+d*d*raw[:,1:2]

def rawsys(l,y):
    eta,x,ep,xp=y; H=2/eta; return [ep,xp,-H*(ep*ep+xp*xp),-2*H*ep*xp]
ep0,xp0=initial_affine_state(); sol=solve_ivp(rawsys,(lam0,lammax),[eta0,x0,ep0,xp0],method='DOP853',rtol=1e-12,atol=1e-12,dense_output=True)
etamax=float(sol.sol(lammax)[0])
print('etamax',etamax)

def rel(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def eval_aff(name,pre=False,N=20001):
    m=Affine(pre); m.load_state_dict(torch.load(RUN/name/'model.pt',map_location='cpu')); m.eval()
    lam=np.linspace(lam0,lammax,N); lt=torch.tensor(lam[:,None],requires_grad=True)
    eta,x=m(lt); ep=deriv(eta,lt); xp=deriv(x,lt); epp=deriv(ep,lt); xpp=deriv(xp,lt)
    H=2/eta; Reta=epp+H*(ep**2+xp**2); Rx=xpp+2*H*ep*xp
    if pre:
        Rs_eta=(m.L**2/m.eta_scale)*Reta; Rs_x=(m.L**2/m.x_scale)*Rx
    arr=[z.detach().numpy().ravel() for z in (eta,x,ep,xp,Reta,Rx)]
    eta_n,x_n,ep_n,xp_n,re_n,rx_n=arr; ref=sol.sol(lam); er, xr=ref[0],ref[1]; a=A*eta_n**2
    pres=a*a*xp_n-p; kres=a*a*(ep_n**2-xp_n**2)-1
    out=dict(name=name,N=N,eta_rel=rel(eta_n,er),x_rel=rel(x_n,xr),max_Reta_phys=np.max(np.abs(re_n)),max_Rx_phys=np.max(np.abs(rx_n)),max_p=np.max(np.abs(pres)),max_k=np.max(np.abs(kres)),min_eta=eta_n.min(),min_eta_prime=ep_n.min(),nonpos_eta=int(np.sum(eta_n<=0)),nonpos_eta_prime=int(np.sum(ep_n<=0)),nonpos_norm=int(np.sum((a*a*(ep_n**2-xp_n**2))<=0)))
    if pre:
        out['max_Reta_scaled']=float(torch.max(torch.abs((m.L**2/m.eta_scale)*Reta)).item()); out['max_Rx_scaled']=float(torch.max(torch.abs((m.L**2/m.x_scale)*Rx)).item())
    return out

def eval_conf(name,N=20001):
    m=Conformal(etamax); m.load_state_dict(torch.load(RUN/name/'model.pt',map_location='cpu')); m.eval()
    eta=np.linspace(eta0,etamax,N); et=torch.tensor(eta[:,None],requires_grad=True)
    x,l=m(et); dx=deriv(x,et); dl=deriv(l,et); d2x=deriv(dx,et); d2l=deriv(dl,et)
    a=A*et**2; den=torch.sqrt(p*p+kappa*a*a); tx=p/den; tl=a*a/den; rx=dx-tx; rl=dl-tl
    vals=[z.detach().numpy().ravel() for z in (x,l,dx,dl,rx,rl,d2x,d2l)]
    x_n,l_n,dx_n,dl_n,rx_n,rl_n,d2x_n,d2l_n=vals
    # raw ref as functions eta
    lamdense=np.linspace(lam0,lammax,max(50000,N)); raw=sol.sol(lamdense); eraw=raw[0]; xraw=raw[1]
    # monotonic raw eta; PCHIP
    xo=PchipInterpolator(eraw,xraw); lo=PchipInterpolator(eraw,lamdense)
    xr=xo(eta); lr=lo(eta)
    # reconstruction on affine grid, no extrapolation
    mono_diffs=np.diff(l_n)
    lam_eval=np.linspace(lam0,lammax,N)
    cover=(l_n[0] <= lam0+1e-12 and l_n[-1] >= lammax-1e-12)
    eta_rec=PchipInterpolator(l_n,eta,extrapolate=False)(lam_eval) if np.all(mono_diffs>0) and cover else np.full_like(lam_eval,np.nan)
    x_rec=PchipInterpolator(l_n,x_n,extrapolate=False)(lam_eval) if np.all(mono_diffs>0) and cover else np.full_like(lam_eval,np.nan)
    rr=sol.sol(lam_eval); er,xrr=rr[0],rr[1]
    dlambda=dl_n; dxdl=dx_n/dlambda; etadl=1/dlambda; an=A*eta**2; pres=an**2*dxdl-p; kres=an**2*(etadl**2-dxdl**2)-1
    # induced second-order affine residual from conformal predictions using formulas via eta
    # eta'' = -lambda_etaeta / lambda_eta^3 ; x'' = x_etaeta/lambda_eta^2 - x_eta*lambda_etaeta/lambda_eta^3
    eta_pp=-d2l_n/(dlambda**3); x_pp=d2x_n/(dlambda**2)-dx_n*d2l_n/(dlambda**3); H=2/eta; eta_p=etadl; x_p=dxdl
    geo_eta=eta_pp+H*(eta_p**2+x_p**2); geo_x=x_pp+2*H*eta_p*x_p
    return dict(name=name,N=N,x_eta_rel=rel(x_n,xr),lambda_eta_rel=rel(l_n,lr),eta_lambda_rel=rel(eta_rec,er),x_lambda_rel=rel(x_rec,xrr),max_Rx=float(np.max(np.abs(rx_n))),max_Rlambda=float(np.max(np.abs(rl_n))),min_dlambda=float(np.min(dl_n)),min_diff_lambda=float(np.min(mono_diffs)),lambda_start=float(l_n[0]),lambda_end=float(l_n[-1]),coverage=bool(cover),max_p=float(np.max(np.abs(pres))),max_k=float(np.max(np.abs(kres))),max_geo_eta=float(np.max(np.abs(geo_eta))),max_geo_x=float(np.max(np.abs(geo_x))))

rows=[]
for n,pre in [('affine_second_order__rar_off',False),('affine_second_order__rar_on',False),('affine_preconditioned__rar_off',True),('affine_preconditioned__rar_on',True)]: rows.append(eval_aff(n,pre))
for n in ['conformal_first_integral__rar_off','conformal_first_integral__rar_on']: rows.append(eval_conf(n))
df=pd.DataFrame(rows); print(df.to_string(index=False)); df.to_csv(ROOT/'results'/'highres_checkpoint_audit_recomputed.csv',index=False)
