import os, sys, json, time, numpy as np, pandas as pd, torch
from pathlib import Path
# Load definitions from the upgraded experiment without executing its batch runner.
source=Path(__file__).with_name('run_upgraded_ablation.py').read_text()
ns={'__name__':'cell_worker'}
exec(source.split('# ---------- run ----------')[0],ns)
rep=os.environ['FLRW_CELL_REP']; rar=os.environ.get('FLRW_CELL_RAR','0')=='1'; seed=int(os.environ.get('FLRW_CELL_SEED','42')); out=Path(os.environ['FLRW_CELL_OUT']); out.mkdir(parents=True,exist_ok=True)
state=ns['shared_state'](seed)
print(f'RUN seed={seed} rep={rep} rar={rar}',flush=True)
m,z,h,st,wall,actual,fe=ns['train'](rep,rar,seed,state)
met,arr=ns['evaluate'](rep,m,ns['CFG']['nval'])
row={'seed':seed,'representation':rep,'rar':rar,'wall_seconds':wall,'lbfgs_requested':ns['CFG']['lbfgs'],'lbfgs_actual_iters':actual,'lbfgs_func_evals':fe,'collocation_final':len(z),'adam_epochs_total':ns['CFG']['adam_epochs']*(ns['CFG']['n_stages']+1),**met}
pd.DataFrame([row]).to_csv(out/'metrics.csv',index=False); h.to_csv(out/'training_history.csv',index=False); st.to_csv(out/'stage_records.csv',index=False); np.savez(out/'validation_arrays.npz',**arr); torch.save(m.state_dict(),out/'model.pt'); (out/'config.json').write_text(json.dumps({'seed':seed,'representation':rep,'rar':rar,'profile':ns['PROFILE'],'network':{'width':ns['WIDTH'],'depth':ns['DEPTH']},'training':ns['CFG']},indent=2)); print(pd.DataFrame([row]).to_string(index=False),flush=True)
