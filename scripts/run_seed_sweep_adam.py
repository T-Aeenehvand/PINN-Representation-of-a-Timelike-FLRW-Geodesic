from pathlib import Path
import pandas as pd, numpy as np, torch
source=Path(__file__).with_name('run_upgraded_ablation.py').read_text(); ns={'__name__':'seed_sweep'}; exec(source.split('# ---------- run ----------')[0],ns)
root=Path('results/seed_sweep_adam'); root.mkdir(parents=True,exist_ok=True); rows=[]
for seed in [42,43,44,45,46]:
  state=ns['shared_state'](seed)
  for rep in ['a2','afi','cfi']:
    # Adam-only robust sweep: 250 updates per stage, 1250 total; no quasi-Newton phase.
    ns['seed_all'](seed); cls={'a2':ns['A2'],'afi':ns['AFI'],'c2':ns['C2'],'cfi':ns['CFI']}[rep]; m=cls(state); vmin,vmax=(ns['LAMBDA0'],ns['LAMBDA_MAX']) if rep in ('a2','afi') else (ns['ETA_MIN'],ns['ETA_MAX']); z=ns['grid'](vmin,vmax,ns['CFG']['n_initial']); fixed=[ns['grid'](vmin,vmax,ns['CFG']['n_initial']+i*ns['CFG']['n_add']) for i in range(ns['CFG']['n_stages']+1)]; import time; t=time.perf_counter(); hist=[]
    for st in range(ns['CFG']['n_stages']+1):
      z=fixed[st]; opt=torch.optim.Adam(m.parameters(),lr=ns['ADAM_LR'])
      for ep in range(ns['CFG']['adam_epochs']):
        opt.zero_grad(set_to_none=True); r,L,vals=ns['loss'](m,z,rep); L.backward(); opt.step(); hist.append([st,ep+1,float(L.detach()),float(vals[0].detach()),float(vals[1].detach())])
    wall=time.perf_counter()-t; met,arr=ns['evaluate'](rep,m,ns['CFG']['nval']); row={'seed':seed,'representation':rep,'rar':False,'optimizer':'Adam-only','adam_epochs_total':ns['CFG']['adam_epochs']*(ns['CFG']['n_stages']+1),'wall_seconds':wall,'lbfgs_actual_iters':0,**met}; rows.append(row); d=root/f'seed_{seed}/{rep}'; d.mkdir(parents=True,exist_ok=True); pd.DataFrame([row]).to_csv(d/'metrics.csv',index=False); pd.DataFrame(hist,columns=['stage','epoch','loss','L1','L2']).to_csv(d/'training_history.csv',index=False); np.savez(d/'validation_arrays.npz',**arr); torch.save(m.state_dict(),d/'model.pt'); print(pd.DataFrame([row]).to_string(index=False),flush=True); del m,z,hist,arr,met,row; import gc; gc.collect()
df=pd.DataFrame(rows); df.to_csv(root/'all_runs.csv',index=False); stat=df.groupby('representation').agg(n=('seed','count'),median_eta=('eta_lambda_rel_L2','median'),q25_eta=('eta_lambda_rel_L2',lambda x:x.quantile(.25)),q75_eta=('eta_lambda_rel_L2',lambda x:x.quantile(.75)),min_eta=('eta_lambda_rel_L2','min'),max_eta=('eta_lambda_rel_L2','max'),median_x=('x_lambda_rel_L2','median'),q25_x=('x_lambda_rel_L2',lambda x:x.quantile(.25)),q75_x=('x_lambda_rel_L2',lambda x:x.quantile(.75)),min_x=('x_lambda_rel_L2','min'),max_x=('x_lambda_rel_L2','max')).reset_index(); stat.to_csv(root/'summary_statistics.csv',index=False); print(stat.to_string(index=False))
