import warnings, json, time, sys
from pathlib import Path; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from table1_perm_null_pipeline import *
OUT = str(Path(__file__).resolve().parents[2] / "paper_data" / "table1_perm_null_N200.json")
res={}
for m,t,i in CELLS:
    tag=f"{m}_{t}_{i}_L22"; t0=time.time()
    X,tg,bal,rn,g,n,ng=build_cell(m,t,i)
    folds=prepare_folds(X,g)
    real,sd=fit_with_folds(folds,X,tg,bal,rn)
    rng=np.random.RandomState(42); nulls=[]
    for p_i in range(N_PERM):
        nulls.append(fit_with_folds(folds,X,game_block_permute(tg,g,rng),bal,rn)[0])
        if (p_i+1)%50==0: print(f"   [{tag}] {p_i+1}/{N_PERM} null={np.mean(nulls):+.4f}",flush=True)
    nulls=np.array(nulls); k=int(np.sum(nulls>=real)); p=(1+k)/(1+N_PERM)
    res[tag]={"n":n,"n_games":ng,"n_features":int(X.shape[1]),"real_r2":float(real),
              "fold_sd":float(sd),"null_mean":float(nulls.mean()),
              "null_95th":float(np.percentile(nulls,95)),"n_exceed":k,
              "perm_p":float(p),"n_perm":N_PERM,"p_floor":1/(N_PERM+1)}
    print(f"== {tag}: R2={real:+.4f} null={nulls.mean():+.4f} exceed={k}/{N_PERM} p={p:.4f} ({time.time()-t0:.0f}s)",flush=True)
    json.dump(res,open(OUT,"w"),indent=2)
print("ALL DONE")
