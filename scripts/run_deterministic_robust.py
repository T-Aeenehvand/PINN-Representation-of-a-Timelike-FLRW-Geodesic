import os,sys,json,glob
from pathlib import Path
import pandas as pd, numpy as np, torch
source=Path(__file__).with_name('run_upgraded_ablation.py').read_text(); ns={'__name__':'det_worker'}; exec(source.split('# ---------- run ----------')[0],ns)
seeds=[42,43,44,45,46]; root=Path('results/deterministic_robust'); root.mkdir(parents=True,exist_ok=True); rows=[]
for seed in seeds:
    state=ns['shared_state'](seed)
    for rep in ('a2','afi','cfi'):
        print('RUN',seed,rep,flush=True)
        m,z,h,st,wall,actual,fe=ns['train'](rep,False,seed,state)
        met,arr=ns['evaluate'](rep,m,ns['CFG']['nval'])
        row={'seed':seed,'representation':rep,'rar':False,'wall_seconds':wall,'lbfgs_requested':ns['CFG']['lbfgs'],'lbfgs_actual_iters':actual,'lbfgs_func_evals':fe,'collocation_final':len(z),'adam_epochs_total':ns['CFG']['adam_epochs']*(ns['CFG']['n_stages']+1),**met}; rows.append(row)
        d=root/f'seed_{seed}/{rep}__rar_off'; d.mkdir(parents=True,exist_ok=True); pd.DataFrame([row]).to_csv(d/'metrics.csv',index=False); h.to_csv(d/'training_history.csv',index=False); st.to_csv(d/'stage_records.csv',index=False); np.savez(d/'validation_arrays.npz',**arr); torch.save(m.state_dict(),d/'model.pt')
df=pd.DataFrame(rows); df.to_csv(root/'all_runs.csv',index=False); stat=df.groupby('representation').agg(n=('seed','count'),median_eta=('eta_lambda_rel_L2','median'),q25_eta=('eta_lambda_rel_L2',lambda x:x.quantile(.25)),q75_eta=('eta_lambda_rel_L2',lambda x:x.quantile(.75)),min_eta=('eta_lambda_rel_L2','min'),max_eta=('eta_lambda_rel_L2','max'),median_x=('x_lambda_rel_L2','median'),q25_x=('x_lambda_rel_L2',lambda x:x.quantile(.25)),q75_x=('x_lambda_rel_L2',lambda x:x.quantile(.75)),min_x=('x_lambda_rel_L2','min'),max_x=('x_lambda_rel_L2','max'),median_lbfgs=('lbfgs_actual_iters','median')).reset_index(); stat.to_csv(root/'summary_statistics.csv',index=False); print(stat.to_string(index=False))
