"""
Permutation Null Test for SAE I_BA Probe (Critical for publication)
===================================================================
Game-level block permutation, RF deconfound inside CV.
Optimized: only 4 key configs, 200 permutations.
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

RNG = np.random.RandomState(42)
DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
BEHAVIORAL_ROOT = Path("/home/v-seungplee/data/llm-addiction/behavioral")
RESULTS_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 200
RF_TREES = 50
RF_DEPTH = 8
RIDGE_ALPHA = 100.0
N_PERM = 200


def load_sae_and_meta(model, paradigm, layer):
    task_dirs = {"sm": "slot_machine", "mw": "mystery_wheel"}
    sae_dir = DATA_ROOT / task_dirs[paradigm] / model
    npz_path = sae_dir / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None, None
    data = np.load(npz_path, allow_pickle=False)
    shape = tuple(data["shape"])
    sp = sparse.csr_matrix(
        (data["values"], (data["row_indices"], data["col_indices"])),
        shape=shape, dtype=np.float32
    )
    meta = {
        "game_ids": data["game_ids"],
        "round_nums": data["round_nums"],
        "bet_types": data["bet_types"],
        "balances": data["balances"] if "balances" in data else None,
    }
    return sp, meta


def compute_iba(meta, model, paradigm):
    if paradigm == "sm":
        if model == "gemma":
            gpath = BEHAVIORAL_ROOT / "slot_machine/gemma_v4_role/final_gemma_20260227_002507.json"
        else:
            gpath = BEHAVIORAL_ROOT / "slot_machine/llama_v4_role/final_llama_20260315_062428.json"
        with open(gpath) as f:
            raw = json.load(f)
        games_data = raw.get("results", raw.get("games", []))
        if isinstance(games_data, dict):
            games_data = list(games_data.values())
    elif paradigm == "mw":
        if model == "gemma":
            mw_dir = BEHAVIORAL_ROOT / "mystery_wheel/gemma_v2_role"
        else:
            mw_dir = BEHAVIORAL_ROOT / "mystery_wheel/llama_v2_role"
        games_data = []
        for f in sorted(mw_dir.glob(f"{model}_mysterywheel_*.json")):
            d = json.load(open(f))
            results = d.get("results", d.get("games", []))
            if isinstance(results, dict):
                games_data.extend(results.values())
            else:
                games_data.extend(results)
    else:
        return None

    game_map = {}
    for i, g in enumerate(games_data):
        game_map[g.get("game_id", i)] = g

    n = len(meta["game_ids"])
    bet_ratios = np.full(n, np.nan)
    balances = np.full(n, np.nan)
    for i in range(n):
        gid = meta["game_ids"][i]
        rn = int(meta["round_nums"][i]) - 1
        g = game_map.get(gid) or game_map.get(str(gid))
        if g is None and isinstance(gid, (np.integer, int)):
            g = game_map.get(int(gid))
        if g is None:
            continue
        decs = g.get("history", g.get("decisions", []))
        if rn >= len(decs):
            continue
        dec = decs[rn]
        bet_val = dec.get("parsed_bet") or dec.get("bet") or dec.get("bet_amount")
        bal_val = dec.get("balance_before") or dec.get("balance")
        if bet_val is None or bal_val is None:
            continue
        try:
            bet, bal = float(bet_val), float(bal_val)
        except (ValueError, TypeError):
            continue
        if bal > 0 and bet > 0:
            bet_ratios[i] = min(bet / bal, 1.0)
            balances[i] = bal
    return bet_ratios, balances


def nl_deconfound_split(target_tr, bal_tr, rn_tr, target_te, bal_te, rn_te):
    def _cov(b, r):
        return np.column_stack([b, r, b**2, np.log1p(b), b*r])
    rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_DEPTH,
                                random_state=42, n_jobs=-1)
    rf.fit(_cov(bal_tr, rn_tr), target_tr)
    return target_tr - rf.predict(_cov(bal_tr, rn_tr)), target_te - rf.predict(_cov(bal_te, rn_te))


def game_block_permute(values, game_ids):
    unique = np.unique(game_ids)
    perm_unique = RNG.permutation(unique)
    gid_map = dict(zip(unique, perm_unique))
    val_by_game = {g: values[game_ids == g].copy() for g in unique}
    out = np.empty_like(values)
    for g in unique:
        mask = game_ids == g
        src = val_by_game[gid_map[g]]
        out[mask] = RNG.choice(src, size=mask.sum(), replace=True) if len(src) != mask.sum() else src
    return out


def run_one_config(model, para, layer):
    tag = f"{model}_{para}_L{layer}"
    print(f"\n--- {tag} ---")

    sp, meta = load_sae_and_meta(model, para, layer)
    if sp is None:
        print(f"  SKIP: no SAE data"); return None

    iba_result = compute_iba(meta, model, para)
    if iba_result is None:
        print(f"  SKIP: no behavioral data"); return None

    bet_ratios, balances = iba_result
    bt = meta["bet_types"]
    valid = (
        (bt == "variable") &
        ~np.isnan(bet_ratios) & ~np.isnan(balances) &
        (balances > 0) & (bet_ratios > 0)
    )
    if valid.sum() < 100:
        print(f"  SKIP: only {valid.sum()} valid rounds"); return None

    X_sparse = sp[valid]
    iba = bet_ratios[valid]
    bal = balances[valid]
    rn = meta["round_nums"][valid].astype(float)
    gids = meta["game_ids"][valid]

    # Pre-filter: keep only features with >10 non-zero activations (99.6% are dead)
    nnz_per_col = np.diff(X_sparse.tocsc().indptr)
    active_cols = np.where(nnz_per_col > 10)[0]
    X = X_sparse[:, active_cols].toarray()
    n_feat = X.shape[1]
    k = min(TOP_K, n_feat)
    print(f"  n={len(iba)}, active_features={n_feat}/{X_sparse.shape[1]}, games={len(np.unique(gids))}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(X))

    def _eval_pipeline(target):
        r2s = []
        for tr, te in folds:
            res_tr, res_te = nl_deconfound_split(
                target[tr], bal[tr], rn[tr], target[te], bal[te], rn[te])
            corrs = np.array([abs(spearmanr(X[tr, j], res_tr)[0])
                              if X[tr, j].std() > 0 else 0
                              for j in range(n_feat)])
            top_idx = np.argsort(corrs)[-k:]
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr][:, top_idx])
            Xte = sc.transform(X[te][:, top_idx])
            pred = Ridge(RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
            r2s.append(r2_score(res_te, pred))
        return np.mean(r2s)

    # Real R²
    real_r2 = _eval_pipeline(iba)
    print(f"  Real R² = {real_r2:.4f}")

    # Game-level block permutation null
    null_r2s = []
    for perm_i in range(N_PERM):
        perm_iba = game_block_permute(iba, gids)
        null_r2s.append(_eval_pipeline(perm_iba))
        if (perm_i + 1) % 20 == 0:
            print(f"    perm {perm_i+1}/{N_PERM}: null mean={np.mean(null_r2s):.4f}")

    null_r2s = np.array(null_r2s)
    p_val = (1 + np.sum(null_r2s >= real_r2)) / (1 + N_PERM)
    print(f"  Null R²: mean={null_r2s.mean():.4f}, 95th={np.percentile(null_r2s,95):.4f}")
    print(f"  p-value = {p_val:.4f}")
    print(f"  *** {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'} ***")

    return {
        "real_r2": float(real_r2),
        "null_mean": float(null_r2s.mean()),
        "null_95th": float(np.percentile(null_r2s, 95)),
        "p_value": float(p_val),
        "n_perm": N_PERM,
        "n_samples": int(valid.sum()),
        "n_games": int(len(np.unique(gids))),
    }


if __name__ == "__main__":
    configs = [
        ("gemma", "sm", 24),
        ("gemma", "mw", 24),
        ("llama", "sm", 16),
        ("llama", "mw", 16),
    ]

    results = {}
    for model, para, layer in configs:
        r = run_one_config(model, para, layer)
        if r:
            results[f"{model}_{para}_L{layer}"] = r

    out_path = RESULTS_DIR / "permutation_null.json"
    class NpEnc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NpEnc)
    print(f"\nResults saved to {out_path}")
    print("DONE")
