from pathlib import Path
import os, pandas as pd, numpy as np, torch, time, gc
source=Path(__file__).with_name('run_upgraded_ablation.py').read_text(); ns={'__name__':'cell'}; exec(source.split('# ---------- run ----------')[0],ns)
seed=int(os.environ['SWEEP_SEED']); rep=os.environ['SWEEP_REP']; root=Path(os.environ.get('SWEEP_OUT','results/seed_sweep_adam')); d=root/f'seed_{seed}/{rep}'; d.mkdir(parents=True,exist_ok=True)
state=ns['shared_state'](seed); ns['seed_all'](seed); cls={'a2':ns['A2'],'afi':ns['AFI'],'cfi':ns['CFI']}[rep]; m=cls(state); vmin,vmax=(ns['LAMBDA0'],ns['LAMBDA_MAX']) if rep in ('a2','afi') else (ns['ETA_MIN'],ns['ETA_MAX']); z=ns['grid'](vmin,vmax,ns['CFG']['n_initial']); fixed=[ns['grid'](vmin,vmax,ns['CFG']['n_initial']+i*ns['CFG']['n_add']) for i in range(ns['CFG']['n_stages']+1)]; t=time.perf_counter(); hist=[]
for st in range(ns['CFG']['n_stages']+1):
 z=fixed[st]; opt=torch.optim.Adam(m.parameters(),lr=ns['ADAM_LR'])
 for ep in range(ns['CFG']['adam_epochs']):
  opt.zero_grad(set_to_none=True); r,L,vals=ns['loss'](m,z,rep); L.backward(); opt.step(); hist.append([st,ep+1,float(L.detach()),float(vals[0].detach()),float(vals[1].detach())])
wall=time.perf_counter()-t; met,arr=ns['evaluate'](rep,m,ns['CFG']['nval']); row={'seed':seed,'representation':rep,'rar':False,'optimizer':'Adam-only','adam_epochs_total':ns['CFG']['adam_epochs']*(ns['CFG']['n_stages']+1),'wall_seconds':wall,'lbfgs_actual_iters':0,**met}; pd.DataFrame([row]).to_csv(d/'metrics.csv',index=False); pd.DataFrame(hist,columns=['stage','epoch','loss','L1','L2']).to_csv(d/'training_history.csv',index=False); np.savez(d/'validation_arrays.npz',**arr); torch.save(m.state_dict(),d/'model.pt'); print(pd.DataFrame([row]).to_string(index=False),flush=True)
