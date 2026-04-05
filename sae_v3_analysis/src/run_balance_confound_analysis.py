"""
Balance Confound Analysis for LLM Addiction Paper
==================================================
의도: 신경 분류(AUC 0.95+)가 단순 잔액 정보 읽기가 아닌,
      파산 관련 내부 표상에 기반함을 증명한다.

가설: 잔액이 동일한 게임끼리만 비교해도 분류 성능이 유지되면,
      분류기가 잔액 외의 내부 표상을 활용하고 있다.

검증 방법:
  A. Balance-Matched Classification: $95-105 구간에서 BK vs VS 분류
  B. Within-Bet-Type Classification: Variable 게임만으로 BK vs VS 분류
  C. Residual Classification: 잔액/라운드를 회귀 제거 후 분류
  D. Balance-Only Baseline: 잔액 스칼라만으로 분류 (confound 상한선)
"""

import numpy as np
import json
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)

DATA_PATH = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz")
OUT_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/balance_confound")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LAYER_IDX = 2  # layers=[8,12,22,25,30], idx=2 → L22


def load_data():
    """Load hidden states and metadata."""
    d = np.load(DATA_PATH, allow_pickle=True)
    hs = d["hidden_states"][:, TARGET_LAYER_IDX, :]  # (3200, 4096) at L22
    valid = d["valid_mask"]
    outcomes = d["game_outcomes"]
    balances = d["balances"]
    bet_types = d["bet_types"]
    round_nums = d["round_nums"]

    # Filter valid + BK/VS only (exclude max_rounds if any)
    mask = valid & np.isin(outcomes, ["bankruptcy", "voluntary_stop"])
    hs = hs[mask]
    labels = (outcomes[mask] == "bankruptcy").astype(int)
    balances = balances[mask]
    bet_types = bet_types[mask]
    round_nums = round_nums[mask]

    print(f"Data loaded: {len(labels)} games, {labels.sum()} BK, {(1-labels).sum()} VS")
    print(f"Balance range: [{balances.min():.0f}, {balances.max():.0f}], mean={balances.mean():.1f}")
    return hs, labels, balances, bet_types, round_nums


def classify_cv(X, y, n_splits=5, pca_dim=50):
    """5-fold stratified CV with PCA + LogisticRegression."""
    if len(np.unique(y)) < 2 or len(y) < 10:
        return {"auc": float("nan"), "n": len(y), "n_pos": int(y.sum())}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        actual_pca = min(pca_dim, X_tr.shape[0], X_tr.shape[1])
        if actual_pca < 2:
            continue
        pca = PCA(n_components=actual_pca, random_state=SEED)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(X_tr, y[train_idx])

        y_prob = clf.predict_proba(X_te)[:, 1]
        if len(np.unique(y[test_idx])) == 2:
            aucs.append(roc_auc_score(y[test_idx], y_prob))

    if not aucs:
        return {"auc": float("nan"), "n": len(y), "n_pos": int(y.sum())}
    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "n": len(y),
        "n_pos": int(y.sum()),
        "n_neg": int((1-y).sum()),
        "folds": len(aucs),
    }


def analysis_baseline(hs, labels):
    """Baseline: full hidden state classification at L22."""
    print("\n=== Baseline: Full Hidden State (L22) ===")
    res = classify_cv(hs, labels)
    print(f"  AUC = {res['auc']:.4f} ± {res.get('auc_std',0):.4f} (n={res['n']}, BK={res['n_pos']})")
    return res


def analysis_balance_only(balances, labels):
    """D: Balance-only baseline (confound upper bound)."""
    print("\n=== Balance-Only Baseline (confound ceiling) ===")
    X = balances.reshape(-1, 1)
    res = classify_cv(X, labels, pca_dim=1)
    print(f"  AUC = {res['auc']:.4f} ± {res.get('auc_std',0):.4f}")
    return res


def analysis_balance_matched(hs, labels, balances):
    """A: Balance-Matched Classification in narrow bins."""
    print("\n=== A: Balance-Matched Classification ===")
    bins = [
        ("$95-105", 95, 105),
        ("$85-95", 85, 95),
        ("$70-90", 70, 90),
        ("$50-70", 50, 70),
        ("$30-50", 30, 50),
        ("$10-30", 10, 30),
    ]
    results = {}
    for name, lo, hi in bins:
        mask = (balances >= lo) & (balances <= hi)
        n_total = mask.sum()
        if n_total < 20:
            print(f"  {name}: skipped (n={n_total})")
            continue
        y_bin = labels[mask]
        n_bk = y_bin.sum()
        n_vs = (1-y_bin).sum()
        if n_bk < 5 or n_vs < 5:
            print(f"  {name}: skipped (BK={n_bk}, VS={n_vs})")
            continue
        res = classify_cv(hs[mask], y_bin)
        print(f"  {name}: AUC = {res['auc']:.4f} ± {res.get('auc_std',0):.4f} (n={n_total}, BK={n_bk}, VS={n_vs})")
        results[name] = res
    return results


def analysis_within_bettype(hs, labels, bet_types):
    """B: Within-Bet-Type Classification."""
    print("\n=== B: Within-Bet-Type Classification ===")
    results = {}
    for bt in ["fixed", "variable"]:
        mask = bet_types == bt
        y_bt = labels[mask]
        n_bk, n_vs = y_bt.sum(), (1-y_bt).sum()
        if n_bk < 5 or n_vs < 5:
            print(f"  {bt}: skipped (BK={n_bk}, VS={n_vs})")
            continue
        res = classify_cv(hs[mask], y_bt)
        print(f"  {bt}: AUC = {res['auc']:.4f} ± {res.get('auc_std',0):.4f} (n={mask.sum()}, BK={n_bk}, VS={n_vs})")
        results[bt] = res
    return results


def analysis_residual(hs, labels, balances, round_nums):
    """C: Residual Classification (balance + round regressed out)."""
    print("\n=== C: Residual Classification ===")
    # Build confound matrix
    confounds = np.column_stack([balances, round_nums])

    # Project out confound directions from hidden states
    # For each confound, compute its direction in hs space and subtract
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(hs, confounds)
    predicted_confounds = reg.predict(hs)
    # Residual = hs - what confounds explain
    # Actually, we want to remove confound info FROM hs
    reg2 = LinearRegression()
    reg2.fit(confounds, hs)
    hs_predicted = reg2.predict(confounds)
    hs_residual = hs - hs_predicted

    res = classify_cv(hs_residual, labels)
    print(f"  AUC = {res['auc']:.4f} ± {res.get('auc_std',0):.4f}")
    return res


def main():
    hs, labels, balances, bet_types, round_nums = load_data()

    results = {}
    results["baseline"] = analysis_baseline(hs, labels)
    results["balance_only"] = analysis_balance_only(balances, labels)
    results["balance_matched"] = analysis_balance_matched(hs, labels, balances)
    results["within_bettype"] = analysis_within_bettype(hs, labels, bet_types)
    results["residual"] = analysis_residual(hs, labels, balances, round_nums)

    # Save results
    out_path = OUT_DIR / "balance_confound_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
