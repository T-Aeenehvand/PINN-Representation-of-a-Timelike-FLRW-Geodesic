from __future__ import annotations
import os, gc, json, time, math, shutil
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp, cumulative_trapezoid, quad
from scipy.interpolate import PchipInterpolator

# Publication-grade upgraded experiment.
# Core design: affine/conformal x second-order/first-integral x deterministic/RAR.
# Default: five independent seeds. Seed-42 affine preconditioning sensitivity is run separately.
_seed_env=os.getenv('FLRW_UPGRADED_SEEDS','')
SEEDS=[int(x) for x in _seed_env.split(',') if x.strip()] if _seed_env else ([42,43,44,45,46] if os.getenv('FLRW_UPGRADED_PROFILE','paper')=='paper' else [42])
PROFILE = os.getenv('FLRW_UPGRADED_PROFILE','paper')
OUTROOT = Path(os.getenv('FLRW_UPGRADED_OUT','results/upgraded_ablation'))
OUTROOT.mkdir(parents=True, exist_ok=True)

A=1.0; ETA0=0.5; X0=0.0; P=1.0; KAPPA=1.0; LAMBDA0=0.0; LAMBDA_MAX=0.8
WIDTH=64; DEPTH=5; ADAM_LR=1e-3; LBFGS_LR=0.5; LBFGS_HISTORY=50
PROFILES={
 'smoke': dict(adam_epochs=40, n_initial=96, n_add=24, n_stages=2, n_candidate=600, lbfgs=20, nval=1200),
 'paper': dict(adam_epochs=250, n_initial=256, n_add=64, n_stages=4, n_candidate=1200, lbfgs=20, nval=4000),
}
CFG=PROFILES[PROFILE]
DTYPE=torch.float64; DEVICE=torch.device('cpu')
torch.set_default_dtype(DTYPE); torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass

# ---------- physics ----------
def a_np(eta): return A*np.asarray(eta,dtype=float)**2
def initial_affine():
    a0=A*ETA0**2
    return np.sqrt(P**2+KAPPA*a0*a0)/a0**2, P/a0**2

def f_np(eta):
    aa=a_np(eta); return P/np.sqrt(P**2+KAPPA*aa**2)
def g_np(eta):
    aa=a_np(eta); return aa**2/np.sqrt(P**2+KAPPA*aa**2)

def raw_rhs(lam,y):
    eta,x,ep,xp=y; Hc=2/eta
    return [ep,xp,-Hc*(ep**2+xp**2),-2*Hc*ep*xp]

def ref_bundle(n=10000):
    ep0,xp0=initial_affine()
    raw=solve_ivp(raw_rhs,(LAMBDA0,LAMBDA_MAX),[ETA0,X0,ep0,xp0],method='DOP853',rtol=1e-12,atol=1e-12,dense_output=True)
    if not raw.success: raise RuntimeError(raw.message)
    lam_dense=np.linspace(LAMBDA0,LAMBDA_MAX,max(n,10000))
    rv=raw.sol(lam_dense); eta_raw,x_raw=rv[0],rv[1]
    eta_max=float(eta_raw[-1]); eta_grid=np.linspace(ETA0,eta_max,n)
    red=solve_ivp(lambda e,y:[f_np(e),g_np(e)],(ETA0,eta_max),[X0,LAMBDA0],method='DOP853',rtol=1e-13,atol=1e-13,dense_output=True)
    xr,lr=red.sol(eta_grid)
    order=np.argsort(eta_raw); es=eta_raw[order]; xs=x_raw[order]; ls=lam_dense[order]
    eu,idx=np.unique(es,return_index=True)
    xof=PchipInterpolator(eu,xs[idx]); lof=PchipInterpolator(eu,ls[idx])
    return dict(raw=raw,eta_grid=eta_grid,x_of_eta=xof,lambda_of_eta=lof,
                raw_reduced_x_max=float(np.max(np.abs(xof(eta_grid)-xr))),
                raw_reduced_lambda_max=float(np.max(np.abs(lof(eta_grid)-lr))))
REF=ref_bundle(CFG['nval']); ETA_MIN=ETA0; ETA_MAX=float(REF['eta_grid'][-1])

def rel_l2(x,y): return float(np.linalg.norm(np.asarray(x)-np.asarray(y))/(np.linalg.norm(np.asarray(y))+1e-15))

def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)

def build_mlp(out=2):
    layers=[nn.Linear(1,WIDTH),nn.Tanh()]
    for _ in range(DEPTH-1): layers += [nn.Linear(WIDTH,WIDTH),nn.Tanh()]
    layers += [nn.Linear(WIDTH,out)]
    net=nn.Sequential(*layers)
    for m in net:
        if isinstance(m,nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    return net

def shared_state(seed):
    seed_all(seed); m=build_mlp(2); return {k:v.detach().clone() for k,v in m.state_dict().items()}

def d(y,x): return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]

def grid(vmin,vmax,n):
    j=np.arange(n); z=np.cos(np.pi*(2*j+1)/(2*n)); vals=vmin+0.5*(1-z)*(vmax-vmin)
    vals=np.unique(np.sort(np.r_[vmin,vmax,vals])); return torch.tensor(vals[:,None],dtype=DTYPE)

# ---------- models ----------
class A2(nn.Module):
    def __init__(self,state):
        super().__init__(); self.net=build_mlp(); self.net.load_state_dict(state); self.ep0,self.xp0=initial_affine()
    def forward(self,lam):
        s=2*(lam-LAMBDA0)/(LAMBDA_MAX-LAMBDA0)-1; r=self.net(s); q=lam-LAMBDA0
        return ETA0+self.ep0*q+q*q*r[:,:1], X0+self.xp0*q+q*q*r[:,1:]

def a2res(m,z):
    z=z.detach().clone().requires_grad_(True); eta,x=m(z); ep=d(eta,z); xp=d(x,z); epp=d(ep,z); xpp=d(xp,z); H=2/eta
    return dict(lam=z,eta=eta,x=x,ep=ep,xp=xp,eta2=epp,x2=xpp,Reta=epp+H*(ep**2+xp**2),Rx=xpp+2*H*ep*xp)

class AFI(nn.Module):
    def __init__(self,state):
        super().__init__(); self.net=build_mlp(); self.net.load_state_dict(state); self.ep0,self.xp0=initial_affine()
    def forward(self,lam):
        s=2*(lam-LAMBDA0)/(LAMBDA_MAX-LAMBDA0)-1; r=self.net(s); q=lam-LAMBDA0
        return ETA0+self.ep0*q+q*q*r[:,:1], X0+self.xp0*q+q*q*r[:,1:]

def afires(m,z):
    z=z.detach().clone().requires_grad_(True); eta,x=m(z); ep=d(eta,z); xp=d(x,z); aa=A*eta**2
    den=torch.sqrt(P**2+KAPPA*aa**2)
    return dict(lam=z,eta=eta,x=x,ep=ep,xp=xp,Reta=ep-den/aa**2,Rx=xp-P/aa**2)

class C2(nn.Module):
    def __init__(self,state):
        super().__init__(); self.net=build_mlp(); self.net.load_state_dict(state); self.dx0=float(f_np(ETA0)); self.dl0=float(g_np(ETA0))
    def forward(self,eta):
        s=2*(eta-ETA_MIN)/(ETA_MAX-ETA_MIN)-1; r=self.net(s); q=eta-ETA0
        return X0+self.dx0*q+q*q*r[:,:1], LAMBDA0+self.dl0*q+q*q*r[:,1:]

def c2res(m,z):
    z=z.detach().clone().requires_grad_(True); x,lam=m(z); xp=d(x,z); gp=d(lam,z); xpp=d(xp,z); gpp=d(gp,z)
    # Physical affine geodesic equations transformed to eta as independent variable.
    # eta_lambda=1/g, x_lambda=x_eta/g.
    # Multiply by g^3 to avoid explicit inverse powers while preserving zero set.
    Reta=-gpp+(2/z)*gp*(1+xp**2)
    Rx=xpp*gp-xp*gpp+(4/z)*xp*gp
    return dict(eta=z,x=x,lambda_=lam,xp=xp,gp=gp,xpp=xpp,gpp=gpp,Reta=Reta,Rx=Rx)

class CFI(nn.Module):
    def __init__(self,state):
        super().__init__(); self.net=build_mlp(); self.net.load_state_dict(state); self.dx0=float(f_np(ETA0)); self.dl0=float(g_np(ETA0))
    def forward(self,eta):
        s=2*(eta-ETA_MIN)/(ETA_MAX-ETA_MIN)-1; r=self.net(s); q=eta-ETA0
        return X0+self.dx0*q+q*q*r[:,:1], LAMBDA0+self.dl0*q+q*q*r[:,1:]

def cfires(m,z):
    z=z.detach().clone().requires_grad_(True); x,lam=m(z); xp=d(x,z); gp=d(lam,z); aa=A*z**2; den=torch.sqrt(P**2+KAPPA*aa**2)
    return dict(eta=z,x=x,lambda_=lam,xp=xp,gp=gp,Rx=xp-P/den,Rlambda=gp-aa**2/den)

# ---------- training ----------
def score_residuals(m,z,rep):
    z=z.detach().clone().requires_grad_(True)
    if rep=='a2':
        eta,x=m(z); ep=torch.autograd.grad(eta,z,torch.ones_like(eta),create_graph=True,retain_graph=True)[0]; xp=torch.autograd.grad(x,z,torch.ones_like(x),create_graph=True,retain_graph=True)[0]
        epp=torch.autograd.grad(ep,z,torch.ones_like(ep),create_graph=False,retain_graph=True)[0]; xpp=torch.autograd.grad(xp,z,torch.ones_like(xp),create_graph=False,retain_graph=True)[0]; H=2/eta; return (epp+H*(ep**2+xp**2)).detach().cpu().numpy()**2 + (xpp+2*H*ep*xp).detach().cpu().numpy()**2
    if rep=='c2':
        x,lam=m(z); xp=torch.autograd.grad(x,z,torch.ones_like(x),create_graph=True,retain_graph=True)[0]; gp=torch.autograd.grad(lam,z,torch.ones_like(lam),create_graph=True,retain_graph=True)[0]
        xpp=torch.autograd.grad(xp,z,torch.ones_like(xp),create_graph=False,retain_graph=True)[0]; gpp=torch.autograd.grad(gp,z,torch.ones_like(gp),create_graph=False,retain_graph=True)[0]
        return ((-gpp+(2/z)*gp*(1+xp**2)).detach().cpu().numpy()**2 + (xpp*gp-xp*gpp+(4/z)*xp*gp).detach().cpu().numpy()**2)
    if rep=='afi':
        eta,x=m(z); ep=torch.autograd.grad(eta,z,torch.ones_like(eta),create_graph=False,retain_graph=True)[0]; xp=torch.autograd.grad(x,z,torch.ones_like(x),create_graph=False,retain_graph=True)[0]; aa=A*eta**2; den=torch.sqrt(P**2+KAPPA*aa**2)
        return ((ep-den/aa**2).detach().cpu().numpy()**2 + (xp-P/aa**2).detach().cpu().numpy()**2)
    x,lam=m(z); xp=torch.autograd.grad(x,z,torch.ones_like(x),create_graph=False,retain_graph=True)[0]; gp=torch.autograd.grad(lam,z,torch.ones_like(lam),create_graph=False,retain_graph=True)[0]; aa=A*z**2; den=torch.sqrt(P**2+KAPPA*aa**2)
    return ((xp-P/den).detach().cpu().numpy()**2 + (gp-aa**2/den).detach().cpu().numpy()**2)

def residual_score(m,z,rep):
    return score_residuals(m,z,rep).reshape(-1)

def loss(m,z,rep):
    r=REPRES[rep](m,z); keys=('Reta','Rx') if rep in ('a2','afi','c2') else ('Rx','Rlambda'); vals=[torch.mean(r[k]**2) for k in keys]; return r,sum(vals),vals

def refine(m,z,vmin,vmax,rep):
    cand=torch.linspace(vmin,vmax,CFG['n_candidate'],dtype=DTYPE)[:,None]
    scores=[]
    for i in range(0,len(cand),800): scores.append(residual_score(m,cand[i:i+800],rep))
    scores=np.concatenate(scores); cur=z.detach().numpy().ravel(); cn=cand.numpy().ravel(); avail=np.ones(len(cn),bool)
    for q in cur: avail &= np.abs(cn-q)>1e-12
    inds=np.where(avail)[0]; rank=inds[np.argsort(scores[avail])[::-1][:CFG['n_add']]]
    return torch.tensor(np.unique(np.sort(np.r_[cur,cn[rank]]))[:,None],dtype=DTYPE),float(scores.max())

def train(rep,rar,seed,state):
    seed_all(seed)
    cls={'a2':A2,'afi':AFI,'c2':C2,'cfi':CFI}[rep]; m=cls(state)
    vmin,vmax=(LAMBDA0,LAMBDA_MAX) if rep in ('a2','afi') else (ETA_MIN,ETA_MAX)
    z=grid(vmin,vmax,CFG['n_initial']); hist=[]; stages=[]; t=time.perf_counter()
    fixed=[grid(vmin,vmax,CFG['n_initial']+i*CFG['n_add']) for i in range(CFG['n_stages']+1)]
    for st in range(CFG['n_stages']+1):
        if not rar: z=fixed[st]
        opt=optim.Adam(m.parameters(),lr=ADAM_LR)
        for ep in range(CFG['adam_epochs']):
            opt.zero_grad(set_to_none=True); r,L,vals=loss(m,z,rep)
            if not torch.isfinite(L): raise FloatingPointError(f'nonfinite {rep} seed {seed} stage {st}')
            L.backward(); opt.step(); hist.append([st,ep+1,float(L),float(vals[0]),float(vals[1]),len(z)])
        if rar and st<CFG['n_stages']:
            z,mx=refine(m,z,vmin,vmax,rep); stages.append([st+1,'RAR_add',len(z),mx])
        else: stages.append([st,'fixed' if not rar else 'final',len(z),np.nan])
    lb=optim.LBFGS(m.parameters(),lr=LBFGS_LR,max_iter=CFG['lbfgs'],history_size=LBFGS_HISTORY,line_search_fn='strong_wolfe')
    def closure():
        lb.zero_grad(set_to_none=True); _,L,_=loss(m,z,rep); L.backward(); return L
    lb.step(closure)
    p0=next(iter(m.parameters())); st=lb.state.get(p0,{})
    actual=int(st.get('n_iter',-1)); fe=int(st.get('func_evals',-1)); wall=time.perf_counter()-t
    return m,z,pd.DataFrame(hist,columns=['stage','epoch','loss','L1','L2','n']),pd.DataFrame(stages,columns=['stage','action','n','candidate_max']),wall,actual,fe

REPRES={'a2':a2res,'afi':afires,'c2':c2res,'cfi':cfires}

def evaluate(rep,m,n):
    if rep in ('a2','afi'):
        le=np.linspace(LAMBDA0,LAMBDA_MAX,n); z=torch.tensor(le[:,None],dtype=DTYPE); r=REPRES[rep](m,z)
        eta=r['eta'].detach().numpy().ravel(); x=r['x'].detach().numpy().ravel(); ep=r['ep'].detach().numpy().ravel(); xp=r['xp'].detach().numpy().ravel()
        er,xr=REF['raw'].sol(le)[:2]
        aa=a_np(eta); pres=aa**2*xp-P; kres=aa**2*(ep**2-xp**2)-KAPPA
        return dict(eta_lambda_rel_L2=rel_l2(eta,er),x_lambda_rel_L2=rel_l2(x,xr),x_eta_rel_L2=np.nan,lambda_eta_rel_L2=np.nan,max_Reta=float(np.max(np.abs(r['Reta'].detach().numpy()))),max_Rx=float(np.max(np.abs(r['Rx'].detach().numpy()))),max_Rlambda=np.nan,max_p=float(np.max(np.abs(pres))),max_kappa=float(np.max(np.abs(kres))),min_g=np.nan), dict(lambda_=le,eta=eta,x=x,eta_ref=er,x_ref=xr,Reta=r['Reta'].detach().numpy().ravel(),Rx=r['Rx'].detach().numpy().ravel(),p=pres,kappa=kres)
    ee=np.linspace(ETA_MIN,ETA_MAX,n); z=torch.tensor(ee[:,None],dtype=DTYPE); r=REPRES[rep](m,z)
    x=r['x'].detach().numpy().ravel(); lam=r['lambda_'].detach().numpy().ravel(); xp=r['xp'].detach().numpy().ravel(); gp=r['gp'].detach().numpy().ravel(); Rx=r['Rx'].detach().numpy().ravel(); R2=(r['Rlambda'] if 'Rlambda' in r else r['Reta']).detach().numpy().ravel()
    xr=REF['x_of_eta'](ee); lr=REF['lambda_of_eta'](ee); le=np.linspace(LAMBDA0,LAMBDA_MAX,n); er,xx=REF['raw'].sol(le)[:2]
    dxdl=xp/gp; dedl=1/gp; aa=a_np(ee); pres=aa**2*dxdl-P; kres=aa**2*(dedl**2-dxdl**2)-KAPPA
    good=np.all(np.isfinite(lam)) and np.all(gp>0) and np.all(np.diff(lam)>0) and lam[-1]>=LAMBDA_MAX-1e-12
    if good:
        eta_rec=PchipInterpolator(lam,ee,extrapolate=False)(le); x_rec=PchipInterpolator(lam,x,extrapolate=False)(le)
        em=rel_l2(eta_rec,er); xm=rel_l2(x_rec,xx)
    else: eta_rec=np.full_like(le,np.nan); x_rec=np.full_like(le,np.nan); em=xm=np.nan
    return dict(eta_lambda_rel_L2=em,x_lambda_rel_L2=xm,x_eta_rel_L2=rel_l2(x,xr),lambda_eta_rel_L2=rel_l2(lam,lr),max_Reta=np.nan,max_Rx=float(np.max(np.abs(Rx))),max_Rlambda=float(np.max(np.abs(R2))),max_p=float(np.max(np.abs(pres))),max_kappa=float(np.max(np.abs(kres))),min_g=float(np.min(gp))), dict(eta=ee,x=x,lambda_=lam,x_ref_eta=xr,lambda_ref_eta=lr,eta_recon=eta_rec,x_recon=x_rec,eta_ref_lambda=er,x_ref_lambda=xx,Rx=Rx,Rlambda=R2,p=pres,kappa=kres,gp=gp)

# ---------- quadrature baseline ----------
def quadrature_baseline(n=200001):
    t=time.perf_counter(); ee=np.linspace(ETA0,ETA_MAX,n); ff=f_np(ee); gg=g_np(ee); x=np.r_[X0,cumulative_trapezoid(ff,ee)]; lam=np.r_[LAMBDA0,cumulative_trapezoid(gg,ee)]; wall=time.perf_counter()-t
    le=np.linspace(LAMBDA0,LAMBDA_MAX,CFG['nval']); er,xx=REF['raw'].sol(le)[:2]; erec=PchipInterpolator(lam,ee)(le); xrec=PchipInterpolator(lam,x)(le)
    return dict(method='trapezoid+PCHIP',nodes=n,wall_seconds=wall,eta_lambda_rel_L2=rel_l2(erec,er),x_lambda_rel_L2=rel_l2(xrec,xx),eta_terminal=float(erec[-1]),x_terminal=float(xrec[-1]))

# ---------- run ----------
summary=[]; root=OUTROOT
meta={'profile':PROFILE,'seeds':SEEDS,'architecture':[WIDTH,DEPTH],'adam_epochs_per_stage':CFG['adam_epochs'],'n_stages':CFG['n_stages']+1,'lbfgs_requested':CFG['lbfgs'],'n_initial':CFG['n_initial'],'n_add':CFG['n_add'],'n_candidate':CFG['n_candidate'],'n_validation':CFG['nval']}
(root/'experiment_config.json').write_text(json.dumps(meta,indent=2))

for seed in SEEDS:
    state=shared_state(seed)
    for rep in ('a2','afi','c2','cfi'):
        for rar in (False,True):
            name=f'seed_{seed}/{rep}__rar_{"on" if rar else "off"}'; dpath=root/name; dpath.mkdir(parents=True,exist_ok=True)
            print(f'RUN {name}',flush=True)
            m,z,h,st,wall,actual,fe=train(rep,rar,seed,state)
            met,arr=evaluate(rep,m,CFG['nval'])
            row={'seed':seed,'representation':rep,'rar':rar,'wall_seconds':wall,'lbfgs_requested':CFG['lbfgs'],'lbfgs_actual_iters':actual,'lbfgs_func_evals':fe,'collocation_final':len(z),'adam_epochs_total':CFG['adam_epochs']*(CFG['n_stages']+1),**met}
            summary.append(row); pd.DataFrame([row]).to_csv(dpath/'metrics.csv',index=False); h.to_csv(dpath/'training_history.csv',index=False); st.to_csv(dpath/'stage_records.csv',index=False); np.savez(dpath/'validation_arrays.npz',**arr); torch.save(m.state_dict(),dpath/'model.pt')
            (dpath/'config.json').write_text(json.dumps({'seed':seed,'representation':rep,'rar':rar,'profile':PROFILE,'network':{'width':WIDTH,'depth':DEPTH},'training':CFG},indent=2))
            gc.collect()

summary_df=pd.DataFrame(summary); summary_df.to_csv(root/'all_runs.csv',index=False)
# Seed-level descriptive statistics.
stat=(summary_df.groupby(['representation','rar']).agg(n=('seed','count'),median_eta=('eta_lambda_rel_L2','median'),q25_eta=('eta_lambda_rel_L2',lambda x:x.quantile(.25)),q75_eta=('eta_lambda_rel_L2',lambda x:x.quantile(.75)),min_eta=('eta_lambda_rel_L2','min'),max_eta=('eta_lambda_rel_L2','max'),median_x=('x_lambda_rel_L2','median'),q25_x=('x_lambda_rel_L2',lambda x:x.quantile(.25)),q75_x=('x_lambda_rel_L2',lambda x:x.quantile(.75)),min_x=('x_lambda_rel_L2','min'),max_x=('x_lambda_rel_L2','max'),median_wall=('wall_seconds','median'),median_lbfgs=('lbfgs_actual_iters','median'),max_p=('max_p','median'),max_kappa=('max_kappa','median')).reset_index())
stat.to_csv(root/'summary_statistics.csv',index=False)
# Failure classification based on finite/common-grid metrics.
summary_df['success']=np.isfinite(summary_df['eta_lambda_rel_L2']) & np.isfinite(summary_df['x_lambda_rel_L2'])
summary_df['failure_class']=np.where(summary_df['success'],'success','reconstruction_or_training_failure')
summary_df[['seed','representation','rar','success','failure_class']].to_csv(root/'failure_summary.csv',index=False)
quad_df=pd.DataFrame([quadrature_baseline()]); quad_df.to_csv(root/'quadrature_baseline.csv',index=False)
# Seed-42 preconditioning sensitivity: reuse full paper budget but only two cells.
state=shared_state(42)
# compact copy of preconditioned implementation from the original project, evaluated with same validation.
class AP(A2):
    def __init__(self,state):
        nn.Module.__init__(self); self.net=build_mlp(); self.net.load_state_dict(state); ep,xp=initial_affine(); self.ep0=ep; self.xp0=xp; self.dlam=LAMBDA_MAX-LAMBDA0; self.eta_scale=abs(ep)*self.dlam; self.x_scale=abs(xp)*self.dlam
    def forward(self,lam):
        tau=(lam-LAMBDA0)/self.dlam; s=2*tau-1; r=self.net(s); eh=tau+tau**2*r[:,:1]; xh=tau+tau**2*r[:,1:]; return ETA0+self.eta_scale*eh,X0+self.x_scale*xh

def apres(m,z):
    z=z.detach().clone().requires_grad_(True); eta,x=m(z); ep=d(eta,z); xp=d(x,z); epp=d(ep,z); xpp=d(xp,z); H=2/eta
    re=epp+H*(ep**2+xp**2); rx=xpp+2*H*ep*xp
    return {'lam':z,'eta':eta,'x':x,'ep':ep,'xp':xp,'Reta':(m.dlam**2/m.eta_scale)*re,'Rx':(m.dlam**2/m.x_scale)*rx}

def train_ap(rar):
    m=AP(state); z=grid(LAMBDA0,LAMBDA_MAX,CFG['n_initial']); fixed=[grid(LAMBDA0,LAMBDA_MAX,CFG['n_initial']+i*CFG['n_add']) for i in range(CFG['n_stages']+1)]; t=time.perf_counter();
    for stg in range(CFG['n_stages']+1):
        if not rar: z=fixed[stg]
        opt=optim.Adam(m.parameters(),lr=ADAM_LR)
        for _ in range(CFG['adam_epochs']):
            opt.zero_grad(set_to_none=True); r=apres(m,z); L=(r['Reta']**2).mean()+(r['Rx']**2).mean(); L.backward(); opt.step()
        if rar and stg<CFG['n_stages']:
            cand=torch.linspace(LAMBDA0,LAMBDA_MAX,CFG['n_candidate'],dtype=DTYPE)[:,None]; sc=[]
            for i in range(0,len(cand),800):
                rr=apres(m,cand[i:i+800]); sc.append((rr['Reta'].detach().numpy()**2+rr['Rx'].detach().numpy()**2).ravel())
            sc=np.concatenate(sc); cur=z.numpy().ravel(); cn=cand.numpy().ravel(); av=np.ones(len(cn),bool)
            for q in cur: av &= abs(cn-q)>1e-12
            ii=np.where(av)[0]; take=ii[np.argsort(sc[av])[::-1][:CFG['n_add']]]; z=torch.tensor(np.unique(np.sort(np.r_[cur,cn[take]]))[:,None],dtype=DTYPE)
    lb=optim.LBFGS(m.parameters(),lr=LBFGS_LR,max_iter=CFG['lbfgs'],history_size=LBFGS_HISTORY,line_search_fn='strong_wolfe')
    def cl(): lb.zero_grad(); rr=apres(m,z); L=(rr['Reta']**2).mean()+(rr['Rx']**2).mean(); L.backward(); return L
    lb.step(cl); p0=next(iter(m.parameters())); ls=lb.state.get(p0,{}); return m,z,time.perf_counter()-t,int(ls.get('n_iter',-1)),int(ls.get('func_evals',-1))

ap_rows=[]
for rar in (False,True):
    m,z,wall,actual,fe=train_ap(rar); met,arr=evaluate('a2',m,CFG['nval']); ap_rows.append({'seed':42,'representation':'affine_preconditioned','rar':rar,'wall_seconds':wall,'lbfgs_requested':CFG['lbfgs'],'lbfgs_actual_iters':actual,'lbfgs_func_evals':fe,'collocation_final':len(z),'adam_epochs_total':CFG['adam_epochs']*(CFG['n_stages']+1),**met}); dpath=root/f'seed_42/affine_preconditioned__rar_{"on" if rar else "off"}'; dpath.mkdir(parents=True,exist_ok=True); pd.DataFrame([ap_rows[-1]]).to_csv(dpath/'metrics.csv',index=False); torch.save(m.state_dict(),dpath/'model.pt'); np.savez(dpath/'validation_arrays.npz',**arr)
pd.DataFrame(ap_rows).to_csv(root/'affine_preconditioning_seed42.csv',index=False)

print('\nDONE')
print(summary_df.to_string(index=False))
print('\nSTATISTICS')
print(stat.to_string(index=False))
print('\nQUADRATURE')
print(quad_df.to_string(index=False))
print('\nPRECONDITIONING')
print(pd.DataFrame(ap_rows).to_string(index=False))
