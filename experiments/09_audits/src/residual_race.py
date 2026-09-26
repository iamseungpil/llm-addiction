"""Paper's own pipeline, three feature sets, same folds: SAE vs game-log observables vs both."""
import sys, numpy as np
sys.path.insert(0,'/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/src')
sys.path.insert(0,'/home/v-seungplee/llm-addiction/sae_v3_analysis/src')
import nested_baseline as nb
from run_perm_null_ilc import nl_deconfound_split, TOP_K, RIDGE_ALPHA
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

d=np.load('design_v2.npz', allow_pickle=True)
y,observed,raw,game = d['y'],d['observed'],d['raw'].astype(np.float64),d['game']
balance,rnd,idx = d['balance'],d['rnd'],d['idx']
sae,meta,active = nb.load_sae_block(idx,22)
state = nb.state_hashes(raw)
target=np.minimum(y,1.0)

def run(X, groups, label, topk=True):
    gkf=GroupKFold(n_splits=5); r2s=[]
    for tr,te in gkf.split(X,groups=groups):
        res_tr,res_te = nl_deconfound_split(target[tr],balance[tr],rnd[tr],target[te],balance[te],rnd[te])
        if topk and X.shape[1]>TOP_K:
            c=np.array([abs(spearmanr(X[tr,j],res_tr)[0]) if X[tr,j].std()>0 else 0 for j in range(X.shape[1])])
            sel=np.argsort(np.nan_to_num(c))[-TOP_K:]
        else:
            sel=np.arange(X.shape[1])
        sc=StandardScaler(); Xtr=sc.fit_transform(X[tr][:,sel]); Xte=sc.transform(X[te][:,sel])
        r2s.append(r2_score(res_te, Ridge(alpha=RIDGE_ALPHA).fit(Xtr,res_tr).predict(Xte)))
    print(f'  {label:34s} R2 {np.mean(r2s):+.4f} +- {np.std(r2s,ddof=1):.4f}', flush=True)
    return float(np.mean(r2s))

both=np.concatenate([observed,sae],axis=1)
for gname,groups in (('game',game),('state',state)):
    print(f'=== paper pipeline, deconfounded residual target, folds grouped by {gname} ===')
    run(sae,groups,'SAE features (the paper cell)')
    run(observed,groups,'game-log observables only')
    run(both,groups,'observables + SAE')
