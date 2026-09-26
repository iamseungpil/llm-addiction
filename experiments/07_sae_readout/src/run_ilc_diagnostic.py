"""
I_LC Diagnostic: 2x2 Ablation — Leakage vs Feature Filtering
=============================================================
Separates two changes in the strict pipeline:
  A: Full-data RF + all features (ORIGINAL)
  B: CV-internal RF + all features (isolates leakage)
  C: Full-data RF + active features (isolates filtering)
  D: CV-internal RF + active features (STRICT)

If SM fails in B but not C → leakage was the issue
If SM fails in C but not B → feature filtering was the issue
If SM fails in both B and C → both contribute (H1+H2)
"""
import numpy as np
import json
from pathlib import Path
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

from run_perm_null_ilc import compute_loss_chasing

DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
BEHAVIORAL_ROOT = Path("/home/v-seungplee/data/llm-addiction/behavioral")
RESULTS_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 200; RF_TREES = 50; RF_DEPTH = 8; RIDGE_ALPHA = 100.0


def load_data(model, paradigm, layer):
    task_dirs = {"sm": "slot_machine", "mw": "mystery_wheel"}
    npz_path = DATA_ROOT / task_dirs[paradigm] / model / f"sae_features_L{layer}.npz"
    data = np.load(npz_path, allow_pickle=False)
    sp = sparse.csr_matrix(
        (data["values"], (data["row_indices"], data["col_indices"])),
        shape=tuple(data["shape"]), dtype=np.float32
    )
    return sp, {k: data[k] for k in data.keys() if k not in ['row_indices', 'col_indices', 'values', 'shape']}


def compute_lc(meta, model, paradigm):
    """Compute I_LC using the current source-of-truth labeling path."""
    return compute_loss_chasing(meta, model, paradigm)


def nl_deconfound_full(target, bal, rn):
    """Full-data RF deconfound (ORIGINAL — has leakage)."""
    X = np.column_stack([bal, rn, bal**2, np.log1p(bal), bal*rn])
    rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_DEPTH, random_state=42, n_jobs=-1)
    rf.fit(X, target)
    return target - rf.predict(X)


def nl_deconfound_cv(target_tr, bal_tr, rn_tr, target_te, bal_te, rn_te):
    """CV-internal RF deconfound (STRICT — no leakage)."""
    def _cov(b, r): return np.column_stack([b, r, b**2, np.log1p(b), b*r])
    rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_DEPTH, random_state=42, n_jobs=-1)
    rf.fit(_cov(bal_tr, rn_tr), target_tr)
    return target_tr - rf.predict(_cov(bal_tr, rn_tr)), target_te - rf.predict(_cov(bal_te, rn_te))


def eval_pipeline(X, target, bal, rn, deconfound_mode, tag):
    """Run 5-fold CV with specified deconfound mode."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    n_feat = X.shape[1]
    k = min(TOP_K, n_feat)
    r2s = []

    if deconfound_mode == "full":
        residual = nl_deconfound_full(target, bal, rn)
        for tr, te in kf.split(X):
            corrs = np.array([abs(spearmanr(X[tr,j], residual[tr])[0])
                              if X[tr,j].std() > 0 else 0 for j in range(n_feat)])
            idx = np.argsort(corrs)[-k:]
            sc = StandardScaler()
            pred = Ridge(RIDGE_ALPHA).fit(sc.fit_transform(X[tr][:,idx]), residual[tr]).predict(sc.transform(X[te][:,idx]))
            r2s.append(r2_score(residual[te], pred))
    else:  # cv-internal
        for tr, te in kf.split(X):
            res_tr, res_te = nl_deconfound_cv(target[tr], bal[tr], rn[tr], target[te], bal[te], rn[te])
            corrs = np.array([abs(spearmanr(X[tr,j], res_tr)[0])
                              if X[tr,j].std() > 0 else 0 for j in range(n_feat)])
            idx = np.argsort(corrs)[-k:]
            sc = StandardScaler()
            pred = Ridge(RIDGE_ALPHA).fit(sc.fit_transform(X[tr][:,idx]), res_tr).predict(sc.transform(X[te][:,idx]))
            r2s.append(r2_score(res_te, pred))

    r2 = np.mean(r2s)
    print(f"  {tag}: R²={r2:.4f}")
    return r2


def run_config(model, paradigm, layer):
    print(f"\n{'='*60}")
    print(f"  {model.upper()} {paradigm.upper()} L{layer}")
    print(f"{'='*60}")

    sp, meta = load_data(model, paradigm, layer)
    lc, balances = compute_lc(meta, model, paradigm)
    if lc is None: return None

    bt = meta["bet_types"]
    rn = meta["round_nums"].astype(float)
    valid = (bt == "variable") & ~np.isnan(lc) & ~np.isnan(balances) & (balances > 0)
    if valid.sum() < 100: return None

    X_full = sp[valid].toarray()
    lc_v, bal_v, rn_v = lc[valid], balances[valid], rn[valid]

    # Active features
    nnz = np.diff(sparse.csc_matrix(sp[valid]).indptr)
    active = np.where(nnz > 10)[0]
    X_active = X_full[:, active]

    print(f"  n={len(lc_v)}, all_features={X_full.shape[1]}, active={len(active)}, LC_rate={lc_v.mean():.3f}")

    results = {}
    # A: Full-data RF + all features (ORIGINAL)
    results["A_full_all"] = eval_pipeline(X_full, lc_v, bal_v, rn_v, "full", "A: full-data RF + all features")
    # B: CV-internal RF + all features
    results["B_cv_all"] = eval_pipeline(X_full, lc_v, bal_v, rn_v, "cv", "B: CV-internal RF + all features")
    # C: Full-data RF + active features
    results["C_full_active"] = eval_pipeline(X_active, lc_v, bal_v, rn_v, "full", "C: full-data RF + active features")
    # D: CV-internal RF + active features (STRICT)
    results["D_cv_active"] = eval_pipeline(X_active, lc_v, bal_v, rn_v, "cv", "D: CV-internal RF + active features")

    # Interpretation
    leakage_effect = results["A_full_all"] - results["B_cv_all"]
    filter_effect = results["A_full_all"] - results["C_full_active"]
    print(f"\n  Leakage effect (A-B): {leakage_effect:.4f}")
    print(f"  Filter effect (A-C):  {filter_effect:.4f}")
    if leakage_effect > filter_effect:
        print(f"  → LEAKAGE is the dominant factor")
    else:
        print(f"  → FEATURE FILTERING is the dominant factor")

    return results


if __name__ == "__main__":
    configs = [
        ("gemma", "sm", 24), ("gemma", "mw", 24),
        ("llama", "sm", 16), ("llama", "mw", 16),
    ]
    all_results = {}
    for model, para, layer in configs:
        r = run_config(model, para, layer)
        if r: all_results[f"{model}_{para}_L{layer}"] = r

    out = RESULTS_DIR / "ilc_diagnostic_2x2.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")
    print("DONE")
