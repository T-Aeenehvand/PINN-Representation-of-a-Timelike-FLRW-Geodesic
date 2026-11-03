"""Reproduce the seed-42 full-budget AFI control.

The training is split into five Adam stages and eleven L-BFGS chunks so that the
optimizer state can be checkpointed between stages. The L-BFGS state is preserved
across chunks; the reported total is the final optimizer-state count.
"""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]; worker=ROOT/'scripts'/'afi_full_incremental.py'
for stage in range(5):
    subprocess.run([sys.executable,str(worker),'--stage',str(stage)],check=True)
for chunk in [20]+[100]*9+[20]:
    subprocess.run([sys.executable,str(worker),'--lbfgs-chunk',str(chunk)],check=True)
subprocess.run([sys.executable,str(worker),'--lbfgs-chunk','0'],check=True)
print('Full-budget AFI control written to results/afi_full_budget_incremental/.')
