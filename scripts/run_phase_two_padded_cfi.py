"""Five-seed CFI Adam-only screen with eta training interval padded to 1.5."""
from pathlib import Path
import json, numpy as np, pandas as pd, torch
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
src=Path(__file__).with_name('run_upgraded_ablation.py').read_text(); ns={'__name__':'phase2_padded_cfi'}; exec(src[:src.index('\n# ---------- run ----------')],ns)
ns['torch'].set_num_threads(4)
ns['ETA_MAX']=1.5
ns['CFG']=dict(adam_epochs=250,n_initial=256,n_add=64,n_stages=4,n_candidate=1200,lbfgs=0,nval=4000)
OUT=Path('results/seed_sweep_adam_padded_cfi'); OUT.mkdir(parents=True,exist_ok=True)
# Reference directly on eta in the padded interval.
ee=np.linspace(ns['ETA_MIN'],1.5,20001); aa=ns['a_np'](ee); ff=ns['f_np'](ee); gg=ns['g_np'](ee)
xr=np.r_[0.0,ns['cumulative_trapezoid'](ff,ee)]; lr=np.r_[0.0,ns['cumulative_trapezoid'](gg,ee)]
# High-accuracy ODE reference for the error arrays.
red=solve_ivp(lambda e,y:[ns['f_np'](e),ns['g_np'](e)],(0.5,1.5),[0.0,0.0],method='DOP853',rtol=1e-13,atol=1e-13,dense_output=True)
xs,ls=red.sol(ee); lam_eval=np.linspace(0,0.8,4000)
raw=solve_ivp(ns['raw_rhs'],(0,0.8),[0.5,0.0,*ns['initial_affine']()],method='DOP853',rtol=1e-12,atol=1e-12,dense_output=True); eta_ref,x_ref=raw.sol(lam_eval)[:2]
def rel(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-15))
rows=[]
for seed in [45,46]:
    state=ns['shared_state'](seed); m=ns['CFI'](state); z=ns['grid'](ns['ETA_MIN'],1.5,256); hist=[]
    for st in range(5):
        z=ns['grid'](ns['ETA_MIN'],1.5,256+st*64); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
        for ep in range(250):
            opt.zero_grad(set_to_none=True); r,L,vals=ns['loss'](m,z,'cfi'); L.backward(); opt.step()
            if ep in (0,124,249): hist.append([st,ep+1,float(L.detach()),len(z)])
    et=torch.tensor(ee[:,None],dtype=torch.float64); r=ns['cfires'](m,et); xp=r['x'].detach().numpy().ravel(); lp=r['lambda_'].detach().numpy().ravel();
    native_x=rel(xp,xs); native_l=rel(lp,ls)
    good=np.all(np.diff(lp)>0) and np.all(r['gp'].detach().numpy().ravel()>0) and lp[-1]>=0.8-1e-12
    if good:
        er=PchipInterpolator(lp,ee)(lam_eval); xr2=PchipInterpolator(lp,xp)(lam_eval); aff_eta=rel(er,eta_ref); aff_x=rel(xr2,x_ref)
    else: aff_eta=aff_x=np.nan
    row=dict(seed=seed,representation='cfi',rar=False,optimizer='Adam-only',adam_epochs_total=1250,eta_train_max=1.5,native_x_eta_rel_L2=native_x,native_lambda_eta_rel_L2=native_l,eta_lambda_rel_L2=aff_eta,x_lambda_rel_L2=aff_x,lambda_terminal=float(lp[-1]),covers_lambda_0p8=good,min_dlambda_deta=float(np.min(r['gp'].detach().numpy())),max_Rx=float(np.max(np.abs(r['Rx'].detach().numpy()))),max_Rlambda=float(np.max(np.abs(r['Rlambda'].detach().numpy()))))
    rows.append(row); d=OUT/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True); pd.DataFrame([row]).to_csv(d/'metrics.csv',index=False); pd.DataFrame(hist,columns=['stage','epoch_checkpoint','loss','n']).to_csv(d/'training_history_checkpoints.csv',index=False); torch.save(m.state_dict(),d/'model.pt'); np.savez(d/'validation_arrays.npz',eta=ee,x_pred=xp,lambda_pred=lp,x_ref_eta=xs,lambda_ref_eta=ls,lambda_eval=lam_eval,eta_ref_lambda=eta_ref,x_ref_lambda=x_ref,eta_recon=(er if good else np.full_like(lam_eval,np.nan)),x_recon=(xr2 if good else np.full_like(lam_eval,np.nan)))
    print(row,flush=True)
df=pd.DataFrame(rows); df.to_csv(OUT/'all_runs.csv',index=False); pd.DataFrame({'metric':['native_x_eta','native_lambda_eta','affine_eta_lambda','affine_x_lambda'],'median':[df.native_x_eta_rel_L2.median(),df.native_lambda_eta_rel_L2.median(),df.eta_lambda_rel_L2.median(),df.x_lambda_rel_L2.median()]}).to_csv(OUT/'summary_statistics.csv',index=False); (OUT/'experiment_config.json').write_text(json.dumps({'seeds':[42,43,44,45,46],'eta_train_max':1.5,'adam_epochs_total':1250,'adam_epochs_per_stage':250,'stages':5,'lbfgs':False,'n_validation':4000},indent=2))
