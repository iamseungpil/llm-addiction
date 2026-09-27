"""표1 18셀 재계산 + 게임블록 순열 null (N=200).

프로토콜·상수·시드는 릴리스의 run_perm_null_ilc.py / run_groupkfold_recompute.py /
run_comprehensive_robustness.py 그대로. 특징 선택만 수학적으로 동등한 형태로 벡터화했다
(Spearman = 순위의 Pearson; X의 순위는 폴드 안에서 불변이므로 폴드당 1회만 계산).
"""
import warnings, json, time
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

# Root holding the HF snapshot of llm-addiction-research/llm-addiction.
# Override with LLM_ADDICTION_DATA; the two subtrees used are
#   sae_features_v3/<task>/<model>/sae_features_L22.npz
#   behavioral/<task>/...
import os
D = Path(os.environ.get("LLM_ADDICTION_DATA", "./hf_snapshot"))
DATA_ROOT, BEHAVIORAL_ROOT = D/"sae_features_v3", D/"behavioral"
TOP_K, RF_TREES, RF_DEPTH, RIDGE_ALPHA, N_PERM = 200, 50, 8, 100.0, 200
TASK_DIR = {"sm":"slot_machine","mw":"mystery_wheel","ic":"investment_choice"}

def load_sae_and_meta(model, para, layer=22):
    z = np.load(DATA_ROOT/TASK_DIR[para]/model/f"sae_features_L{layer}.npz", allow_pickle=False)
    shape = tuple(int(x) for x in z["shape"])
    sp = sparse.csr_matrix((z["values"],(z["row_indices"],z["col_indices"])), shape=shape, dtype=np.float32)
    meta = {k: z[k] for k in z.keys() if k not in ('row_indices','col_indices','values','shape')}
    return sp, meta

def _games(model, para):
    if para == 'sm':
        gp = BEHAVIORAL_ROOT/("slot_machine/gemma_v4_role/final_gemma_20260227_002507.json" if model=='gemma'
                              else "slot_machine/llama_v4_role/final_llama_20260315_062428.json")
        raw = json.load(open(gp)); g = raw.get('results', raw.get('games', []))
        return list(g.values()) if isinstance(g, dict) else g
    d = BEHAVIORAL_ROOT/("mystery_wheel/"+("gemma_v2_role" if model=='gemma' else "llama_v2_role")) if para=='mw' \
        else BEHAVIORAL_ROOT/("investment_choice/"+("v2_role_gemma" if model=='gemma' else "v2_role_llama"))
    pat = f"{model}_mysterywheel_*.json" if para=='mw' else "*.json"
    out=[]
    for f in sorted(d.glob(pat)):
        r = json.load(open(f)); r = r.get('results', r.get('games', []))
        out.extend(r.values() if isinstance(r, dict) else r)
    return out

def compute_iba(meta, model, para):
    gm = {i+1:g for i,g in enumerate(_games(model,para))}
    n = len(meta["game_ids"]); br = np.full(n,np.nan); bal_out = meta["balances"].astype(float).copy()
    for i in range(n):
        g = gm.get(int(meta["game_ids"][i]));  rn = int(meta["round_nums"][i])-1
        if g is None: continue
        raw = g.get("decisions", g.get("history", g.get("rounds", [])))
        decs = [d for d in raw if d.get("action")!="skip" and not d.get("skipped",False)]
        if rn >= len(decs): continue
        dec = decs[rn]
        bv = dec.get("parsed_bet") or dec.get("bet") or dec.get("bet_amount")
        bl = dec.get("balance_before") or dec.get("balance")
        if bv is None: continue
        try: bet=float(bv); bal=float(bl) if bl is not None else float(bal_out[i])
        except (ValueError,TypeError): continue
        if bal>0 and bet>0: br[i]=min(bet/bal,1.0); bal_out[i]=bal
    return br, bal_out

def compute_ilc(meta, model, para):
    gm = {i+1:g for i,g in enumerate(_games(model,para))}
    n = len(meta['game_ids']); lc = np.full(n,np.nan); bal_out = meta['balances'].astype(float).copy()
    for i in range(n):
        g = gm.get(int(meta['game_ids'][i])); rn = int(meta['round_nums'][i])-1
        if g is None: continue
        raw = g.get('decisions', g.get('history', []))
        decs = [d for d in raw if d.get('action')!='skip' and not d.get('skipped',False)]
        hist = g.get('history', decs)
        if rn>=len(decs) or rn<1: continue
        dec, prev = decs[rn], decs[rn-1]
        bv = dec.get('parsed_bet') or dec.get('bet') or dec.get('bet_amount')
        bl = dec.get('balance_before') or dec.get('balance')
        pb = prev.get('parsed_bet') or prev.get('bet') or prev.get('bet_amount')
        pbal = prev.get('balance_before') or prev.get('balance')
        if any(v is None for v in (bv,pb,pbal)): continue
        try:
            bet=float(bv); bal=float(bl) if bl is not None else float(bal_out[i]); p_bet=float(pb); p_bal=float(pbal)
        except (ValueError,TypeError): continue
        if bet<=0 or bal<=0 or p_bal<=0: continue
        bal_out[i]=bal
        br=min(bet/bal,1.0); p_br=min(p_bet/p_bal,1.0)
        prev_loss=False
        if rn-1 < len(hist):
            prev_loss = not hist[rn-1].get('win', str(hist[rn-1].get('result',''))=='W')
        if prev_loss and p_br>0: lc[i]=max(0.0,(br-p_br)/p_br)
    return lc, bal_out

def build_cell(model, para, ind, layer=22):
    sp, meta = load_sae_and_meta(model, para, layer)
    if ind=='i_lc': target, balances = compute_ilc(meta, model, para)
    else:
        br, balances = compute_iba(meta, model, para)
        target = br if ind=='i_ba' else np.where(np.isnan(br), np.nan, (br>=0.5).astype(float))
    bt = meta['bet_types']
    valid = (bt=='variable') & ~np.isnan(target) & ~np.isnan(balances) & (balances>0)
    if ind=='i_ba': valid = valid & (target>0)
    Xs = sp[valid]
    nnz = np.diff(Xs.tocsc().indptr); active = np.where(nnz>10)[0]
    return (Xs[:,active].toarray(), target[valid], balances[valid],
            meta['round_nums'][valid].astype(float), np.asarray(meta['game_ids'])[valid],
            int(valid.sum()), len(np.unique(np.asarray(meta['game_ids'])[valid])))

def _rf(t_tr,b_tr,r_tr,t_te,b_te,r_te):
    cov=lambda b,r: np.column_stack([b,r,b**2,np.log1p(b),b*r])
    rf=RandomForestRegressor(n_estimators=RF_TREES,max_depth=RF_DEPTH,random_state=42,n_jobs=-1)
    rf.fit(cov(b_tr,r_tr),t_tr)
    return t_tr-rf.predict(cov(b_tr,r_tr)), t_te-rf.predict(cov(b_te,r_te))

def prepare_folds(X, groups, n_splits=5):
    """폴드별로 X 순위를 미리 표준화해 둔다 (순열마다 재사용)."""
    out=[]
    for tr,te in GroupKFold(n_splits=n_splits).split(X,groups=groups):
        Rx=np.apply_along_axis(rankdata,0,X[tr]).astype(np.float64)
        sd=Rx.std(0); sd[sd==0]=1.0
        out.append((tr,te,(Rx-Rx.mean(0))/sd))
    return out

def fit_with_folds(folds, X, target, balances, rounds, k=TOP_K):
    r2s=[]
    for tr,te,Rx in folds:
        res_tr,res_te=_rf(target[tr],balances[tr],rounds[tr],target[te],balances[te],rounds[te])
        ry=rankdata(res_tr).astype(np.float64); s=ry.std()
        ry=(ry-ry.mean())/(s if s>0 else 1.0)
        corrs=np.nan_to_num(np.abs(Rx.T@ry/len(ry)))
        idx=np.argsort(corrs)[-min(k,X.shape[1]):]
        sc=StandardScaler(); Xtr=sc.fit_transform(X[tr][:,idx]); Xte=sc.transform(X[te][:,idx])
        r2s.append(r2_score(res_te, Ridge(alpha=RIDGE_ALPHA).fit(Xtr,res_tr).predict(Xte)))
    return float(np.mean(r2s)), float(np.std(r2s,ddof=1))

def game_block_permute(values, game_ids, rng):
    uq=np.unique(game_ids); gid_map=dict(zip(uq,rng.permutation(uq)))
    by={g:values[game_ids==g].copy() for g in uq}; out=np.empty_like(values)
    for g in uq:
        m=game_ids==g; src=by[gid_map[g]]
        out[m]=rng.choice(src,size=m.sum(),replace=True) if len(src)!=m.sum() else src
    return out

CELLS=[(m,t,i) for m in ('gemma','llama') for t in ('sm','ic','mw') for i in ('i_lc','i_ba','i_ec')]
