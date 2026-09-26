"""
I_BA Cross-Task Probe: Train on SM, Test on MW (and vice versa)
================================================================
Tests whether SAE features encoding betting aggressiveness (I_BA) transfer
across gambling paradigms after nonlinear balance deconfounding.

Pipeline:
1. Load SAE features for SM and MW (per-round, variable betting only)
2. Compute I_BA = bet / balance for each round
3. RF deconfound: remove nonlinear balance + round effects
4. Select top-k features by |Spearman r| with residual on TRAIN set
5. Ridge regression on TRAIN residual
6. Apply same RF + feature selection to TEST set → predict → R²
7. Compare with random feature baseline (30 repeats)

Models: Gemma (L24, L18) and LLaMA (L16, L22)
"""

import numpy as np
import json
import sys
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

# ── Configuration ──────────────────────────────────────────────────
DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")

CONFIGS = [
    {"model": "gemma", "layers": [24, 18], "n_sae": 131072},
    {"model": "llama", "layers": [16, 22], "n_sae": 32768},
]

PARADIGMS = ["slot_machine", "mystery_wheel"]
PARADIGM_SHORT = {"slot_machine": "SM", "mystery_wheel": "MW"}

TOP_K = 200
N_RANDOM = 30
RF_TREES = 50
RF_DEPTH = 8
RIDGE_ALPHA = 100.0


def load_sae_and_meta(model, paradigm, layer):
    """Load sparse SAE features + metadata for a paradigm/model/layer."""
    sae_dir = DATA_ROOT / paradigm / model
    npz_path = sae_dir / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        print(f"  [SKIP] {npz_path} not found")
        return None, None

    data = np.load(npz_path, allow_pickle=False)

    # Reconstruct dense from sparse COO
    shape = tuple(data["shape"])
    row_idx = data["row_indices"]
    col_idx = data["col_indices"]
    values = data["values"]

    # Use sparse matrix for memory efficiency
    sp = sparse.csr_matrix(
        (values, (row_idx, col_idx)), shape=shape, dtype=np.float32
    )

    meta = {
        "game_ids": data["game_ids"],
        "round_nums": data["round_nums"],
        "game_outcomes": data["game_outcomes"],
        "bet_types": data["bet_types"],
        "balances": data["balances"] if "balances" in data else None,
    }
    return sp, meta


def compute_iba(meta, model, paradigm):
    """Compute I_BA (bet/balance) for each round from behavioral data."""
    behavioral_dir = Path(f"/home/v-seungplee/data/llm-addiction/behavioral")

    if paradigm == "slot_machine":
        if model == "gemma":
            games_path = behavioral_dir / "slot_machine/gemma_v4_role/final_gemma_20260227_002507.json"
        else:
            games_path = behavioral_dir / "slot_machine/llama_v4_role/final_llama_20260315_062428.json"
    elif paradigm == "mystery_wheel":
        if model == "gemma":
            mw_dir = behavioral_dir / "mystery_wheel/gemma_v2_role"
        else:
            mw_dir = behavioral_dir / "mystery_wheel/llama_v2_role"
        # MW files: {model}_mysterywheel_c30_*.json
        games_files = sorted(mw_dir.glob(f"{model}_mysterywheel_*.json"))
        if not games_files:
            print(f"  [SKIP] No MW behavioral files in {mw_dir}")
            return None
        # Load all and combine
        all_games = []
        for f in games_files:
            d = json.load(open(f))
            results = d.get("results", d.get("games", []))
            if isinstance(results, dict):
                all_games.extend(results.values())
            else:
                all_games.extend(results)
        games_path = None
        games_data = all_games
    else:
        return None

    if paradigm != "mystery_wheel":
        with open(games_path) as f:
            raw = json.load(f)
        games_data = raw.get("results", raw.get("games", []))
        if isinstance(games_data, dict):
            games_data = list(games_data.values())

    # Build game_id → game mapping
    game_map = {}
    for i, g in enumerate(games_data):
        gid = g.get("game_id", i)
        game_map[gid] = g

    # Compute bet_ratio for each round in meta
    n = len(meta["game_ids"])
    bet_ratios = np.full(n, np.nan)
    balances = np.full(n, np.nan)

    for i in range(n):
        gid = meta["game_ids"][i]
        rn = int(meta["round_nums"][i]) - 1  # 0-indexed

        # Try to find this game
        g = game_map.get(gid) or game_map.get(str(gid)) or game_map.get(int(gid) if isinstance(gid, (str, np.integer)) else gid)
        if g is None:
            continue

        # MW uses 'history', SM uses 'decisions'
        decs = g.get("history", g.get("decisions", g.get("rounds", [])))
        if rn >= len(decs):
            continue

        dec = decs[rn]
        # Handle different field names across paradigms
        bet_val = dec.get("parsed_bet") or dec.get("bet") or dec.get("bet_amount")
        bal_val = dec.get("balance_before") or dec.get("balance")

        if bet_val is None or bal_val is None:
            continue

        bet = float(bet_val)
        bal = float(bal_val)

        if bal > 0 and bet > 0:
            bet_ratios[i] = min(bet / bal, 1.0)
            balances[i] = bal

    return bet_ratios, balances


def nl_deconfound(target, balances, round_nums):
    """Remove nonlinear balance + round effects using Random Forest."""
    # Build covariate matrix with nonlinear terms
    X_cov = np.column_stack([
        balances,
        round_nums,
        balances ** 2,
        np.log1p(balances),
        balances * round_nums,
    ])

    rf = RandomForestRegressor(
        n_estimators=RF_TREES, max_depth=RF_DEPTH,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_cov, target)
    residual = target - rf.predict(X_cov)
    return residual, rf, X_cov


def select_features_and_probe(X_train, y_train, X_test, y_test, top_k=TOP_K):
    """Select top-k features on train, probe on test."""
    # Feature selection on train
    n_features = X_train.shape[1]
    corrs = np.zeros(n_features)
    for j in range(n_features):
        col = X_train[:, j]
        if col.std() == 0:
            continue
        corrs[j] = abs(spearmanr(col, y_train)[0])

    k = min(top_k, n_features)
    top_idx = np.argsort(corrs)[-k:]

    # Ridge on train
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train[:, top_idx])
    X_te = sc.transform(X_test[:, top_idx])

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_tr, y_train)
    pred = ridge.predict(X_te)
    r2 = r2_score(y_test, pred)

    return r2, top_idx


def random_baseline(X_train, y_train, X_test, y_test, top_k=TOP_K, n_repeats=N_RANDOM):
    """Random feature selection baseline."""
    r2s = []
    n_features = X_train.shape[1]
    k = min(top_k, n_features)
    for _ in range(n_repeats):
        idx = np.random.choice(n_features, k, replace=False)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train[:, idx])
        X_te = sc.transform(X_test[:, idx])
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(X_tr, y_train)
        pred = ridge.predict(X_te)
        r2s.append(r2_score(y_test, pred))
    return np.mean(r2s), np.std(r2s)


def within_task_cv(X, y, top_k=TOP_K):
    """5-fold CV within a single task (for comparison)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf.split(X):
        r2, _ = select_features_and_probe(X[tr], y[tr], X[te], y[te], top_k)
        r2s.append(r2)
    return np.mean(r2s), np.std(r2s)


def run_transfer(model_cfg):
    """Run SM↔MW transfer for one model."""
    model = model_cfg["model"]
    n_sae = model_cfg["n_sae"]
    print(f"\n{'='*70}")
    print(f"MODEL: {model.upper()}")
    print(f"{'='*70}")

    results = {}

    for layer in model_cfg["layers"]:
        print(f"\n--- Layer {layer} ---")

        # Load both paradigms
        data = {}
        for paradigm in PARADIGMS:
            sp, meta = load_sae_and_meta(model, paradigm, layer)
            if sp is None:
                continue

            # Compute I_BA
            iba_result = compute_iba(meta, model, paradigm)
            if iba_result is None:
                continue
            bet_ratios, balances = iba_result

            # Filter: variable betting only, valid bet_ratio, valid balance
            bt = meta["bet_types"]
            valid = (
                (bt == "variable") &
                ~np.isnan(bet_ratios) &
                ~np.isnan(balances) &
                (balances > 0) &
                (bet_ratios > 0)
            )

            if valid.sum() < 100:
                print(f"  {paradigm}: only {valid.sum()} valid rounds, skipping")
                continue

            # Convert sparse to dense for valid rows only
            X = sp[valid].toarray()
            iba = bet_ratios[valid]
            bal = balances[valid]
            rn = meta["round_nums"][valid].astype(float)

            # NL deconfound
            residual, rf, _ = nl_deconfound(iba, bal, rn)

            # Filter active features (>1% activation rate)
            active_rate = (X != 0).mean(axis=0)
            active_mask = active_rate >= 0.01
            X_active = X[:, active_mask]

            data[paradigm] = {
                "X": X_active,
                "y": residual,
                "n": len(residual),
                "active_mask": active_mask,
                "X_full": X,
            }

            print(f"  {PARADIGM_SHORT[paradigm]}: n={valid.sum()}, "
                  f"active_features={active_mask.sum()}, "
                  f"residual_std={residual.std():.4f}")

        if len(data) < 2:
            print("  Not enough paradigms loaded, skipping layer")
            continue

        # Find shared active features
        sm_active = set(np.where(data["slot_machine"]["active_mask"])[0])
        mw_active = set(np.where(data["mystery_wheel"]["active_mask"])[0])
        shared = sorted(sm_active & mw_active)
        print(f"  Shared active features: {len(shared)}")

        # Rebuild X with only shared features
        X_sm = data["slot_machine"]["X_full"][:, shared]
        y_sm = data["slot_machine"]["y"]
        X_mw = data["mystery_wheel"]["X_full"][:, shared]
        y_mw = data["mystery_wheel"]["y"]

        # 1. SM → MW transfer
        r2_sm_to_mw, top_idx = select_features_and_probe(X_sm, y_sm, X_mw, y_mw)
        rand_mean, rand_std = random_baseline(X_sm, y_sm, X_mw, y_mw)

        print(f"\n  SM → MW: R²={r2_sm_to_mw:.4f}  Random={rand_mean:.4f}±{rand_std:.4f}")

        # 2. MW → SM transfer
        r2_mw_to_sm, _ = select_features_and_probe(X_mw, y_mw, X_sm, y_sm)
        rand_mean2, rand_std2 = random_baseline(X_mw, y_mw, X_sm, y_sm)

        print(f"  MW → SM: R²={r2_mw_to_sm:.4f}  Random={rand_mean2:.4f}±{rand_std2:.4f}")

        # 3. Within-task CV baselines
        cv_sm, cv_sm_std = within_task_cv(X_sm, y_sm)
        cv_mw, cv_mw_std = within_task_cv(X_mw, y_mw)
        print(f"  Within SM (5-fold): R²={cv_sm:.4f}±{cv_sm_std:.4f}")
        print(f"  Within MW (5-fold): R²={cv_mw:.4f}±{cv_mw_std:.4f}")

        # 4. Transfer efficiency
        eff_sm_mw = r2_sm_to_mw / cv_mw if cv_mw > 0 else 0
        eff_mw_sm = r2_mw_to_sm / cv_sm if cv_sm > 0 else 0
        print(f"  Transfer efficiency: SM→MW={eff_sm_mw:.1%}, MW→SM={eff_mw_sm:.1%}")

        results[f"L{layer}"] = {
            "sm_to_mw": round(r2_sm_to_mw, 4),
            "mw_to_sm": round(r2_mw_to_sm, 4),
            "sm_to_mw_random": round(rand_mean, 4),
            "mw_to_sm_random": round(rand_mean2, 4),
            "within_sm": round(cv_sm, 4),
            "within_mw": round(cv_mw, 4),
            "efficiency_sm_mw": round(eff_sm_mw, 4),
            "efficiency_mw_sm": round(eff_mw_sm, 4),
            "n_sm": data["slot_machine"]["n"],
            "n_mw": data["mystery_wheel"]["n"],
            "n_shared_features": len(shared),
        }

    return results


def main():
    all_results = {}
    for cfg in CONFIGS:
        model_results = run_transfer(cfg)
        all_results[cfg["model"]] = model_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: I_BA Cross-Task Transfer (NL Deconfounded)")
    print(f"{'='*70}")
    print(f"{'Model':>8} {'Layer':>6} {'SM→MW':>8} {'MW→SM':>8} {'Rand':>8} {'Within_SM':>10} {'Within_MW':>10} {'Eff_SM→MW':>10}")

    for model, layers in all_results.items():
        for lname, r in layers.items():
            print(f"{model:>8} {lname:>6} {r['sm_to_mw']:>8.4f} {r['mw_to_sm']:>8.4f} "
                  f"{r['sm_to_mw_random']:>8.4f} {r['within_sm']:>10.4f} {r['within_mw']:>10.4f} "
                  f"{r['efficiency_sm_mw']:>10.1%}")

    # Save
    out_path = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/iba_cross_task_transfer.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json as jlib
    with open(out_path, "w") as f:
        jlib.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
