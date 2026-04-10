"""
Comprehensive Robustness Analysis for RQ1, RQ2, RQ3
====================================================
Addresses ALL reviewer concerns identified via Codex debate:

CRITICAL:
  1. Permutation null distribution (full pipeline: RF→Ridge, game-level block perm)
  2. Hidden state BK probe with permutation baseline (address class imbalance)
  3. BK direction cosine + permutation baseline (HS-level structure sharing)
  4. Fixed variance diagnostic (BK counts + HS BK classification)
  5. IC paradigm full analysis

HIGH:
  6. Layer-wise CKA (all 5 layers for SM×MW×IC, random subsample alignment)
  7. HS cross-paradigm transfer (shuffle TRAIN labels, not test)

Codex-reviewed fixes applied:
  - RF deconfounding inside CV loop (no leakage)
  - StandardScaler inside CV loop (no leakage)
  - Game-level block permutation for TEST 1
  - Shuffle TRAIN labels for transfer permutation (TEST 5)
  - Permutation baseline for direction cosine (TEST 3)
  - Random subsample for CKA alignment (TEST 6)
  - No fallback to all rounds for IC (TEST 7)
"""

import numpy as np
import json
from pathlib import Path
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

RNG = np.random.RandomState(42)
HS_PCA_DIM = 200  # Reduce HS dim for faster LogReg

DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
BEHAVIORAL_ROOT = Path("/home/v-seungplee/data/llm-addiction/behavioral")
RESULTS_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 200
RF_TREES = 50
RF_DEPTH = 8
RIDGE_ALPHA = 100.0
N_PERM = 200
N_PERM_AUC = 50


# ===================================================================
# Data Loading Utilities
# ===================================================================
def load_hs_dp(model, paradigm, layer_idx=3):
    task_dirs = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
    path = DATA_ROOT / task_dirs[paradigm] / model / "hidden_states_dp.npz"
    d = np.load(path, allow_pickle=True)
    valid = d["valid_mask"].astype(bool)
    hs = d["hidden_states"][valid, layer_idx, :]
    meta = {
        "outcomes": d["game_outcomes"][valid],
        "bet_types": d["bet_types"][valid],
        "balances": d["balances"][valid],
        "round_nums": d["round_nums"][valid],
    }
    layer = d["layers"][layer_idx]
    return hs, meta, layer


def bk_labels(meta):
    return (meta["outcomes"] == "bankruptcy").astype(int)


def load_sae_and_meta(model, paradigm, layer):
    task_dirs = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
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
        "game_outcomes": data["game_outcomes"],
        "bet_types": data["bet_types"],
        "balances": data["balances"] if "balances" in data else None,
        "prompt_conditions": data["prompt_conditions"] if "prompt_conditions" in data else None,
        "bet_constraints": data["bet_constraints"] if "bet_constraints" in data else None,
        "is_last_round": data["is_last_round"] if "is_last_round" in data else None,
    }
    return sp, meta


def compute_iba(meta, model, paradigm):
    """Compute I_BA = bet/balance for each round."""
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
    elif paradigm == "ic":
        if model == "gemma":
            ic_dir = BEHAVIORAL_ROOT / "investment_choice/v2_role_gemma"
        else:
            ic_dir = BEHAVIORAL_ROOT / "investment_choice/v2_role_llama"
        games_data = []
        for f in sorted(ic_dir.glob("*.json")):
            d = json.load(open(f))
            results = d.get("results", d.get("games", []))
            if isinstance(results, dict):
                games_data.extend(results.values())
            else:
                games_data.extend(results)
    else:
        return None

    # Extraction remaps games to sequential ids starting from 1.
    game_map = {i + 1: g for i, g in enumerate(games_data)}

    n = len(meta["game_ids"])
    bet_ratios = np.full(n, np.nan)
    if meta.get("balances") is not None:
        balances = meta["balances"].astype(float).copy()
    else:
        balances = np.full(n, np.nan)

    for i in range(n):
        gid = meta["game_ids"][i]
        rn = int(meta["round_nums"][i]) - 1
        g = game_map.get(gid) or game_map.get(str(gid))
        if g is None and isinstance(gid, (np.integer, int)):
            g = game_map.get(int(gid))
        if g is None:
            continue
        raw_decs = g.get("decisions", g.get("history", g.get("rounds", [])))
        decs = [d for d in raw_decs if d.get("action") != "skip" and not d.get("skipped", False)]
        if rn >= len(decs):
            continue
        dec = decs[rn]
        bet_val = dec.get("parsed_bet") or dec.get("bet") or dec.get("bet_amount")
        bal_val = dec.get("balance_before") or dec.get("balance")
        if bet_val is None:
            continue
        try:
            bet = float(bet_val)
            bal = float(bal_val) if bal_val is not None else float(balances[i])
        except (ValueError, TypeError):
            continue
        if bal > 0 and bet > 0:
            bet_ratios[i] = min(bet / bal, 1.0)
            balances[i] = bal

    return bet_ratios, balances


def nl_deconfound_split(target_tr, bal_tr, rn_tr, target_te, bal_te, rn_te):
    """RF deconfound on train, apply to test (no leakage)."""
    def _make_cov(bal, rn):
        return np.column_stack([bal, rn, bal**2, np.log1p(bal), bal*rn])
    X_tr = _make_cov(bal_tr, rn_tr)
    X_te = _make_cov(bal_te, rn_te)
    rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_DEPTH,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, target_tr)
    return target_tr - rf.predict(X_tr), target_te - rf.predict(X_te)


def game_block_permute(values, game_ids, rng):
    """Permute values at the game level (preserve within-game structure)."""
    unique = np.unique(game_ids)
    perm_unique = rng.permutation(unique)
    gid_map = dict(zip(unique, perm_unique))
    # Gather values by game
    val_by_game = {g: values[game_ids == g].copy() for g in unique}
    out = np.empty_like(values)
    for g in unique:
        mask = game_ids == g
        src = val_by_game[gid_map[g]]
        # Match sizes (truncate/pad with random sampling)
        if len(src) == mask.sum():
            out[mask] = src
        else:
            out[mask] = rng.choice(src, size=mask.sum(), replace=True)
    return out


# ===================================================================
# TEST 1: Permutation Null for SAE I_BA Probe (RQ1 critical)
# ===================================================================
def test_permutation_null():
    """Full-pipeline permutation null with game-level block permutation."""
    print("=" * 70)
    print("TEST 1: PERMUTATION NULL — Full Pipeline (RF→Ridge)")
    print("  Game-level block permutation, deconfound inside CV")
    print("=" * 70)

    configs = [
        {"model": "gemma", "paradigm": "sm", "layer": 24},
        {"model": "gemma", "paradigm": "mw", "layer": 24},
        {"model": "llama", "paradigm": "sm", "layer": 16},
        {"model": "llama", "paradigm": "mw", "layer": 16},
    ]

    results = {}
    for cfg in configs:
        model, para, layer = cfg["model"], cfg["paradigm"], cfg["layer"]
        tag = f"{model}_{para}_L{layer}"
        print(f"\n--- {tag} ---")

        sp, meta = load_sae_and_meta(model, para, layer)
        if sp is None:
            print(f"  SKIP: no SAE data"); continue

        iba_result = compute_iba(meta, model, para)
        if iba_result is None:
            print(f"  SKIP: no behavioral data"); continue

        bet_ratios, balances = iba_result
        bt = meta["bet_types"]
        valid = (
            (bt == "variable") &
            ~np.isnan(bet_ratios) & ~np.isnan(balances) &
            (balances > 0) & (bet_ratios > 0)
        )
        if valid.sum() < 100:
            print(f"  SKIP: only {valid.sum()} valid rounds"); continue

        X = sp[valid].toarray()
        iba = bet_ratios[valid]
        bal = balances[valid]
        rn = meta["round_nums"][valid].astype(float)
        gids = meta["game_ids"][valid]
        n_feat = X.shape[1]
        k = min(TOP_K, n_feat)
        print(f"  n={len(iba)}, features={n_feat}, games={len(np.unique(gids))}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(kf.split(X))

        # Real R² (deconfound inside CV)
        real_r2s = []
        for tr, te in folds:
            res_tr, res_te = nl_deconfound_split(
                iba[tr], bal[tr], rn[tr], iba[te], bal[te], rn[te])
            corrs = np.array([abs(spearmanr(X[tr, j], res_tr)[0])
                              if X[tr, j].std() > 0 else 0
                              for j in range(n_feat)])
            top_idx = np.argsort(corrs)[-k:]
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr][:, top_idx])
            Xte = sc.transform(X[te][:, top_idx])
            pred = Ridge(RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
            real_r2s.append(r2_score(res_te, pred))
        real_r2 = np.mean(real_r2s)
        print(f"  Real R² = {real_r2:.4f}")

        # Game-level block permutation null
        null_r2s = []
        for perm_i in range(N_PERM):
            perm_iba = game_block_permute(iba, gids, RNG)
            fold_r2s = []
            for tr, te in folds:
                res_tr, res_te = nl_deconfound_split(
                    perm_iba[tr], bal[tr], rn[tr], perm_iba[te], bal[te], rn[te])
                corrs = np.array([abs(spearmanr(X[tr, j], res_tr)[0])
                                  if X[tr, j].std() > 0 else 0
                                  for j in range(n_feat)])
                top_idx = np.argsort(corrs)[-k:]
                sc = StandardScaler()
                Xtr = sc.fit_transform(X[tr][:, top_idx])
                Xte = sc.transform(X[te][:, top_idx])
                pred = Ridge(RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
                fold_r2s.append(r2_score(res_te, pred))
            null_r2s.append(np.mean(fold_r2s))
            if (perm_i + 1) % 50 == 0:
                print(f"    perm {perm_i+1}/{N_PERM}: null mean={np.mean(null_r2s):.4f}")

        null_r2s = np.array(null_r2s)
        p_val = (1 + np.sum(null_r2s >= real_r2)) / (1 + N_PERM)
        print(f"  Null R²: mean={null_r2s.mean():.4f}, 95th={np.percentile(null_r2s,95):.4f}")
        print(f"  p-value = {p_val:.4f}")
        print(f"  *** {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'} ***")

        results[tag] = {
            "real_r2": float(real_r2),
            "null_mean": float(null_r2s.mean()),
            "null_95th": float(np.percentile(null_r2s, 95)),
            "p_value": float(p_val),
            "n_perm": N_PERM,
            "n_samples": int(valid.sum()),
            "n_games": int(len(np.unique(gids))),
        }

    return results


# ===================================================================
# TEST 2: Hidden State BK Probe + Permutation Baseline
# ===================================================================
def test_hs_bk_probe():
    """BK classification from hidden states with permutation AUC baseline."""
    print("\n" + "=" * 70)
    print("TEST 2: Hidden State BK Probe + Permutation Baseline")
    print("  Scaling inside CV, permutation with fixed folds")
    print("=" * 70)

    results = {}
    for model in ["gemma", "llama"]:
        for para in ["sm", "mw", "ic"]:
            for layer_idx in [2, 3]:
                hs, meta, layer = load_hs_dp(model, para, layer_idx)
                y_bk = bk_labels(meta)
                bt = meta["bet_types"]

                # Test both variable-only and all
                for subset_name, mask in [("var", bt == "variable"), ("all", np.ones(len(bt), bool))]:
                    X = hs[mask]
                    y = y_bk[mask]

                    if y.sum() < 5 or (1-y).sum() < 5 or len(y) < 50:
                        continue

                    tag = f"{model}_{para}_L{layer}_{subset_name}"
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    folds = list(skf.split(X, y))

                    # Real AUC (PCA + scaler inside CV for speed)
                    real_aucs = []
                    for tr, te in folds:
                        sc = StandardScaler()
                        X_tr_s = sc.fit_transform(X[tr])
                        X_te_s = sc.transform(X[te])
                        pca = PCA(n_components=min(HS_PCA_DIM, X_tr_s.shape[1]), random_state=42)
                        X_tr_p = pca.fit_transform(X_tr_s)
                        X_te_p = pca.transform(X_te_s)
                        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                        clf.fit(X_tr_p, y[tr])
                        yp = clf.predict_proba(X_te_p)[:, 1]
                        real_aucs.append(roc_auc_score(y[te], yp))
                    real_auc = np.mean(real_aucs)

                    # Permutation baseline (same folds, PCA for speed)
                    null_aucs = []
                    for _ in range(N_PERM_AUC):
                        y_perm = RNG.permutation(y)
                        perm_aucs = []
                        for tr, te in folds:
                            try:
                                sc = StandardScaler()
                                X_tr_s = sc.fit_transform(X[tr])
                                X_te_s = sc.transform(X[te])
                                pca = PCA(n_components=min(HS_PCA_DIM, X_tr_s.shape[1]), random_state=42)
                                X_tr_p = pca.fit_transform(X_tr_s)
                                X_te_p = pca.transform(X_te_s)
                                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                                clf.fit(X_tr_p, y_perm[tr])
                                yp = clf.predict_proba(X_te_p)[:, 1]
                                perm_aucs.append(roc_auc_score(y_perm[te], yp))
                            except ValueError:
                                perm_aucs.append(0.5)
                        null_aucs.append(np.mean(perm_aucs))
                    null_aucs = np.array(null_aucs)
                    p_val = (1 + np.sum(null_aucs >= real_auc)) / (1 + N_PERM_AUC)

                    print(f"  {tag}: AUC={real_auc:.3f} "
                          f"(null={null_aucs.mean():.3f}±{null_aucs.std():.3f}, p={p_val:.4f})")

                    results[tag] = {
                        "auc": float(real_auc),
                        "null_mean": float(null_aucs.mean()),
                        "null_std": float(null_aucs.std()),
                        "p_value": float(p_val),
                        "n": len(y), "n_bk": int(y.sum()), "layer": int(layer),
                    }

    return results


# ===================================================================
# TEST 3: BK Direction Cosine + Permutation Baseline (RQ2)
# ===================================================================
def test_bk_direction_cosine():
    """Cosine similarity of BK directions with permutation null."""
    print("\n" + "=" * 70)
    print("TEST 3: BK Direction Cosine Similarity + Permutation Null")
    print("=" * 70)

    results = {}
    for model in ["gemma", "llama"]:
        hs_data = {}  # {para_layer: (hs, y)}
        directions = {}
        for para in ["sm", "mw", "ic"]:
            for layer_idx in [2, 3]:
                hs, meta, layer = load_hs_dp(model, para, layer_idx)
                y = bk_labels(meta)
                if y.sum() < 5:
                    continue
                key = f"{para}_L{layer}"
                hs_data[key] = (hs, y)
                direction = hs[y == 1].mean(0) - hs[y == 0].mean(0)
                direction /= np.linalg.norm(direction)
                directions[key] = direction

        keys = sorted(directions.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                # Skip same-paradigm different-layer comparisons (less interesting)
                if k1.split("_")[0] == k2.split("_")[0] and k1.split("_")[1] != k2.split("_")[1]:
                    continue
                cos_real = float(np.dot(directions[k1], directions[k2]))

                # Permutation null: shuffle BK labels independently in each paradigm
                hs1, y1 = hs_data[k1]
                hs2, y2 = hs_data[k2]
                null_cos = []
                for _ in range(N_PERM_AUC):
                    y1p = RNG.permutation(y1)
                    y2p = RNG.permutation(y2)
                    d1 = hs1[y1p == 1].mean(0) - hs1[y1p == 0].mean(0)
                    d2 = hs2[y2p == 1].mean(0) - hs2[y2p == 0].mean(0)
                    d1 /= np.linalg.norm(d1)
                    d2 /= np.linalg.norm(d2)
                    null_cos.append(np.dot(d1, d2))
                null_cos = np.array(null_cos)
                p_val = (1 + np.sum(np.abs(null_cos) >= np.abs(cos_real))) / (1 + N_PERM_AUC)

                tag = f"{model}_{k1}_vs_{k2}"
                print(f"  {tag}: cos={cos_real:.4f} "
                      f"(null={null_cos.mean():.4f}±{null_cos.std():.4f}, p={p_val:.4f})")
                results[tag] = {
                    "cosine": cos_real,
                    "null_mean": float(null_cos.mean()),
                    "null_std": float(null_cos.std()),
                    "p_value": float(p_val),
                }

    return results


# ===================================================================
# TEST 4: Fixed Variance Diagnostic (RQ3)
# ===================================================================
def test_fixed_variance():
    """Fixed betting: BK counts, balance stats, HS BK AUC with perm baseline."""
    print("\n" + "=" * 70)
    print("TEST 4: Fixed Betting Diagnostic")
    print("=" * 70)

    results = {}
    for model in ["gemma", "llama"]:
        for para in ["sm", "mw"]:
            hs, meta, layer = load_hs_dp(model, para, layer_idx=3)
            y = bk_labels(meta)
            bt, bal = meta["bet_types"], meta["balances"].astype(float)

            var_mask = bt == "variable"
            fix_mask = bt == "fixed"
            tag = f"{model}_{para}_L{layer}"

            print(f"\n--- {tag} ---")
            print(f"  Variable: n={var_mask.sum()}, BK={y[var_mask].sum()}, rate={y[var_mask].mean():.4f}")
            print(f"  Fixed:    n={fix_mask.sum()}, BK={y[fix_mask].sum()}, rate={y[fix_mask].mean():.5f}")
            print(f"  Balance std: var={bal[var_mask].std():.1f}, fix={bal[fix_mask].std():.1f}")

            def _classify_with_perm(X, y, label):
                if y.sum() < 5 or (1-y).sum() < 5:
                    print(f"  {label}: BK={y.sum()} — insufficient for classification")
                    return None, None
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                folds = list(skf.split(X, y))
                real_aucs = []
                for tr, te in folds:
                    sc = StandardScaler()
                    Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
                    pca = PCA(n_components=min(HS_PCA_DIM, Xtr.shape[1]), random_state=42)
                    Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
                    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                    clf.fit(Xtr, y[tr])
                    real_aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
                real_auc = np.mean(real_aucs)

                null_aucs = []
                for _ in range(N_PERM_AUC):
                    yp = RNG.permutation(y)
                    pa = []
                    for tr, te in folds:
                        try:
                            sc = StandardScaler()
                            Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
                            pca = PCA(n_components=min(HS_PCA_DIM, Xtr.shape[1]), random_state=42)
                            Xtr = pca.fit_transform(Xtr); Xte = pca.transform(Xte)
                            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                            clf.fit(Xtr, yp[tr])
                            pa.append(roc_auc_score(yp[te], clf.predict_proba(Xte)[:, 1]))
                        except ValueError:
                            pa.append(0.5)
                    null_aucs.append(np.mean(pa))
                null_aucs = np.array(null_aucs)
                p = (1 + np.sum(null_aucs >= real_auc)) / (1 + N_PERM_AUC)
                print(f"  {label}: AUC={real_auc:.3f} (null={null_aucs.mean():.3f}, p={p:.4f})")
                return real_auc, p

            var_auc, var_p = _classify_with_perm(hs[var_mask], y[var_mask], "Variable")
            fix_auc, fix_p = _classify_with_perm(hs[fix_mask], y[fix_mask], "Fixed")

            if fix_auc is None:
                print(f"  → EXPECTED: fixed $10 bets mechanically prevent bankruptcy")

            results[tag] = {
                "var_n": int(var_mask.sum()), "var_bk": int(y[var_mask].sum()),
                "fix_n": int(fix_mask.sum()), "fix_bk": int(y[fix_mask].sum()),
                "var_bal_std": float(bal[var_mask].std()),
                "fix_bal_std": float(bal[fix_mask].std()),
                "var_auc": var_auc, "var_p": var_p,
                "fix_auc": fix_auc, "fix_p": fix_p,
            }

    return results


# ===================================================================
# TEST 5: HS Cross-Paradigm Transfer (shuffle TRAIN labels)
# ===================================================================
def test_hs_transfer_with_perm():
    """HS BK transfer: shuffle TRAIN labels for permutation null."""
    print("\n" + "=" * 70)
    print("TEST 5: HS Cross-Paradigm BK Transfer (train-label permutation)")
    print("=" * 70)

    results = {}
    for model in ["gemma", "llama"]:
        data = {}
        for para in ["sm", "mw", "ic"]:
            hs, meta, layer = load_hs_dp(model, para, layer_idx=3)
            y = bk_labels(meta)
            if y.sum() >= 5 and (1-y).sum() >= 5:
                data[para] = (hs, y)
                print(f"  {model} {para}: n={len(y)}, BK={y.sum()}")

        pairs = [("sm", "mw"), ("mw", "sm"), ("sm", "ic"), ("ic", "sm"),
                 ("mw", "ic"), ("ic", "mw")]
        for src, tgt in pairs:
            if src not in data or tgt not in data:
                continue
            X_tr, y_tr = data[src]
            X_te, y_te = data[tgt]

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            pca = PCA(n_components=min(HS_PCA_DIM, X_tr_s.shape[1]), random_state=42)
            X_tr_p = pca.fit_transform(X_tr_s)
            X_te_p = pca.transform(X_te_s)

            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            clf.fit(X_tr_p, y_tr)
            real_auc = roc_auc_score(y_te, clf.predict_proba(X_te_p)[:, 1])

            # Permutation: shuffle TRAIN labels, retrain, evaluate on real test
            null_aucs = []
            for _ in range(N_PERM_AUC):
                y_tr_perm = RNG.permutation(y_tr)
                clf_p = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                clf_p.fit(X_tr_p, y_tr_perm)
                try:
                    null_aucs.append(roc_auc_score(y_te, clf_p.predict_proba(X_te_p)[:, 1]))
                except ValueError:
                    null_aucs.append(0.5)
            null_aucs = np.array(null_aucs)
            p_val = (1 + np.sum(null_aucs >= real_auc)) / (1 + N_PERM_AUC)

            tag = f"{model}_{src}→{tgt}"
            print(f"  {tag}: AUC={real_auc:.3f} "
                  f"(null={null_aucs.mean():.3f}±{null_aucs.std():.3f}, p={p_val:.4f})")
            results[tag] = {
                "auc": float(real_auc),
                "null_mean": float(null_aucs.mean()),
                "null_std": float(null_aucs.std()),
                "p_value": float(p_val),
            }

    return results


# ===================================================================
# TEST 6: Layer-wise CKA (random subsample alignment)
# ===================================================================
def linear_cka(X, Y, n_subsample=1600):
    """Linear CKA with random subsample alignment."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    # Random subsample to equal size
    n = min(X.shape[0], Y.shape[0], n_subsample)
    idx_x = RNG.choice(X.shape[0], n, replace=False)
    idx_y = RNG.choice(Y.shape[0], n, replace=False)
    X, Y = X[idx_x], Y[idx_y]
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def test_layerwise_cka():
    """CKA at all 5 layers (distributional similarity, not stimulus-matched)."""
    print("\n" + "=" * 70)
    print("TEST 6: Layer-wise CKA (random subsample, distributional)")
    print("=" * 70)

    results = {}
    for model in ["gemma", "llama"]:
        print(f"\n--- {model.upper()} ---")
        for layer_idx in range(5):
            reps = {}
            layer_name = None
            for para in ["sm", "mw", "ic"]:
                hs, meta, layer = load_hs_dp(model, para, layer_idx)
                reps[para] = hs
                layer_name = layer

            for p1, p2 in [("sm", "mw"), ("sm", "ic"), ("mw", "ic")]:
                # Average over 5 random subsamples for stability
                ckas = [linear_cka(reps[p1], reps[p2]) for _ in range(5)]
                cka = np.mean(ckas)
                tag = f"{model}_L{layer_name}_{p1}_{p2}"
                print(f"  L{layer_name} CKA({p1},{p2})={cka:.4f}±{np.std(ckas):.4f}")
                results[tag] = {"cka": float(cka), "cka_std": float(np.std(ckas)),
                                "layer": int(layer_name)}

    return results


# ===================================================================
# TEST 7: IC SAE I_BA Analysis (variable-only, no fallback)
# ===================================================================
def test_ic_sae_analysis():
    """IC paradigm SAE I_BA analysis — variable betting only."""
    print("\n" + "=" * 70)
    print("TEST 7: IC Paradigm SAE I_BA Analysis (variable only)")
    print("=" * 70)

    configs = [
        {"model": "gemma", "layers": [22, 24, 25]},
        {"model": "llama", "layers": [16, 22, 25]},
    ]

    results = {}
    for cfg in configs:
        model = cfg["model"]
        for layer in cfg["layers"]:
            tag = f"{model}_ic_L{layer}"
            print(f"\n--- {tag} ---")

            sp, meta = load_sae_and_meta(model, "ic", layer)
            if sp is None:
                print(f"  SKIP: no SAE data"); continue

            iba_result = compute_iba(meta, model, "ic")
            if iba_result is None:
                print(f"  SKIP: no behavioral data"); continue

            bet_ratios, balances = iba_result
            bt = meta["bet_types"]

            # Variable only — report n even if too few
            valid_var = (
                (bt == "variable") &
                ~np.isnan(bet_ratios) & ~np.isnan(balances) &
                (balances > 0) & (bet_ratios > 0)
            )
            # Also report all-rounds count
            valid_all = (
                ~np.isnan(bet_ratios) & ~np.isnan(balances) &
                (balances > 0) & (bet_ratios > 0)
            )
            print(f"  Variable rounds: {valid_var.sum()}, All valid: {valid_all.sum()}")

            if valid_var.sum() < 50:
                print(f"  SKIP: insufficient variable rounds ({valid_var.sum()}<50)")
                results[tag] = {"status": "insufficient_variable", "n_var": int(valid_var.sum()),
                                "n_all": int(valid_all.sum())}
                continue

            X = sp[valid_var].toarray()
            iba = bet_ratios[valid_var]
            bal = balances[valid_var]
            rn = meta["round_nums"][valid_var].astype(float)
            gids = meta["game_ids"][valid_var]
            n_feat = X.shape[1]
            k = min(TOP_K, n_feat)
            print(f"  n={len(iba)}, features={n_feat}")

            # CV with deconfound inside loop
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kf.split(X))
            r2s = []
            for tr, te in folds:
                res_tr, res_te = nl_deconfound_split(
                    iba[tr], bal[tr], rn[tr], iba[te], bal[te], rn[te])
                corrs = np.array([abs(spearmanr(X[tr, j], res_tr)[0])
                                  if X[tr, j].std() > 0 else 0
                                  for j in range(n_feat)])
                top_idx = np.argsort(corrs)[-k:]
                sc = StandardScaler()
                Xtr = sc.fit_transform(X[tr][:, top_idx])
                Xte = sc.transform(X[te][:, top_idx])
                pred = Ridge(RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
                r2s.append(r2_score(res_te, pred))

            mean_r2 = np.mean(r2s)
            print(f"  I_BA R² = {mean_r2:.4f} (std={np.std(r2s):.4f})")

            results[tag] = {"r2": float(mean_r2), "r2_std": float(np.std(r2s)),
                            "n": int(valid_var.sum()), "n_games": int(len(np.unique(gids)))}

    return results


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    all_results = {}

    print("\n" + "#" * 70)
    print("# COMPREHENSIVE ROBUSTNESS ANALYSIS (Codex-reviewed)")
    print("# Addresses ALL reviewer concerns for RQ1/RQ2/RQ3")
    print("#" * 70)

    # Fast tests first
    all_results["test3_bk_direction"] = test_bk_direction_cosine()
    all_results["test4_fixed_variance"] = test_fixed_variance()
    all_results["test6_layerwise_cka"] = test_layerwise_cka()

    # HS probes with permutation
    all_results["test2_hs_probe"] = test_hs_bk_probe()
    all_results["test5_hs_transfer"] = test_hs_transfer_with_perm()

    # IC paradigm
    all_results["test7_ic_sae"] = test_ic_sae_analysis()

    # Permutation null (slowest)
    all_results["test1_perm_null"] = test_permutation_null()

    # Save
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return super().default(o)

    out_path = RESULTS_DIR / "comprehensive_robustness.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\n\nAll results saved to {out_path}")
    print("DONE — Comprehensive Robustness Analysis Complete")
