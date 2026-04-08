"""
RQ Robustness Check: Address reviewer concerns
================================================
1. RQ1 token confound: Is I_LC signal just token-level encoding?
   → Compare R² at different layers (if token-level, should be uniform across layers)
   → Check: does RF deconfound remove bet_amount too?

2. RQ2 cross-paradigm: Hidden state transfer (not SAE)
   → Train BK classifier on SM hidden states, test on MW
   → CKA between SM and MW hidden states (same model, same layer)
   → Behavioral label transfer: I_LC probe

3. RQ3 Fixed signal vanish: Is it just low variance?
   → Report I_LC variance in Fixed vs Variable
   → Alternative: hidden state BK classification in Fixed (if model encodes
     BK-relevant info even without bet choice)
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")


def load_hs_dp(model, paradigm, layer_idx=3):
    """Load hidden states at decision point."""
    task_dirs = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
    path = DATA_ROOT / task_dirs[paradigm] / model / "hidden_states_dp.npz"
    d = np.load(path, allow_pickle=True)
    valid = d["valid_mask"].astype(bool)
    hs = d["hidden_states"][valid, layer_idx, :]  # (n, hidden_dim)
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


# ===================================================================
# TEST 1: Hidden State Cross-Paradigm Transfer (RQ2)
# ===================================================================
def test_cross_paradigm_hs_transfer():
    """Train BK classifier on SM hidden states, test on MW (and vice versa)."""
    print("=" * 60)
    print("TEST 1: Hidden State Cross-Paradigm Transfer")
    print("  (Addresses: is SAE transfer failure due to prompt format?)")
    print("=" * 60)

    for model in ["gemma", "llama"]:
        for layer_idx in [3]:  # layer 25 for both
            print(f"\n--- {model.upper()} Layer idx={layer_idx} ---")

            data = {}
            for para in ["sm", "mw", "ic"]:
                hs, meta, layer = load_hs_dp(model, para, layer_idx)
                y = bk_labels(meta)
                if y.sum() < 5 or (1 - y).sum() < 5:
                    print(f"  {para}: skipped (BK={y.sum()}, Safe={(1-y).sum()})")
                    continue
                data[para] = (hs, y)
                print(f"  {para}: n={len(y)}, BK={y.sum()}, layer={layer}")

            # Within-paradigm baseline (5-fold CV)
            for para, (X, y) in data.items():
                sc = StandardScaler()
                X_s = sc.fit_transform(X)
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                scores = cross_val_score(clf, X_s, y, cv=5, scoring="roc_auc")
                print(f"  Within {para}: AUC={scores.mean():.3f}±{scores.std():.3f}")

            # Cross-paradigm transfer
            pairs = [("sm", "mw"), ("mw", "sm"), ("sm", "ic"), ("ic", "sm")]
            for src, tgt in pairs:
                if src not in data or tgt not in data:
                    continue
                X_tr, y_tr = data[src]
                X_te, y_te = data[tgt]

                sc = StandardScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)

                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                clf.fit(X_tr_s, y_tr)
                y_pred = clf.predict_proba(X_te_s)[:, 1]
                auc = roc_auc_score(y_te, y_pred)
                print(f"  Transfer {src}→{tgt}: AUC={auc:.3f}")


# ===================================================================
# TEST 2: CKA Between Paradigms (RQ2)
# ===================================================================
def linear_cka(X, Y):
    """Compute Linear CKA between two representation matrices."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    # Use min samples
    n = min(X.shape[0], Y.shape[0])
    X, Y = X[:n], Y[:n]

    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def test_cka():
    """CKA similarity between paradigm hidden states."""
    print("\n" + "=" * 60)
    print("TEST 2: CKA Representational Similarity")
    print("  (Addresses: do paradigms share hidden state structure?)")
    print("=" * 60)

    for model in ["gemma", "llama"]:
        for layer_idx in [2, 3]:  # layers 22, 25
            print(f"\n--- {model.upper()} Layer idx={layer_idx} ---")
            reps = {}
            for para in ["sm", "mw", "ic"]:
                hs, meta, layer = load_hs_dp(model, para, layer_idx)
                reps[para] = hs
                print(f"  {para}: shape={hs.shape}, layer={layer}")

            for p1 in ["sm", "ic", "mw"]:
                for p2 in ["sm", "ic", "mw"]:
                    if p1 >= p2:
                        continue
                    cka = linear_cka(reps[p1], reps[p2])
                    print(f"  CKA({p1}, {p2}) = {cka:.4f}")


# ===================================================================
# TEST 3: Fixed Variance Check (RQ3)
# ===================================================================
def test_fixed_variance():
    """Check if Fixed signal vanish is mechanical (low variance) or real."""
    print("\n" + "=" * 60)
    print("TEST 3: Fixed Betting Signal Analysis")
    print("  (Addresses: is R²≈0 due to low variance or real absence?)")
    print("=" * 60)

    for model in ["gemma", "llama"]:
        hs, meta, layer = load_hs_dp(model, "sm", layer_idx=3)
        y_bk = bk_labels(meta)
        bt = meta["bet_types"]

        var_mask = bt == "variable"
        fix_mask = bt == "fixed"

        print(f"\n--- {model.upper()} SM Layer {layer} ---")
        print(f"  Variable: n={var_mask.sum()}, BK={y_bk[var_mask].sum()}")
        print(f"  Fixed:    n={fix_mask.sum()}, BK={y_bk[fix_mask].sum()}")

        # BK classification in Fixed (hidden state, not SAE)
        if y_bk[fix_mask].sum() >= 5:
            X_fix = hs[fix_mask]
            y_fix = y_bk[fix_mask]
            sc = StandardScaler()
            X_s = sc.fit_transform(X_fix)
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
            scores = cross_val_score(clf, X_s, y_fix, cv=5, scoring="roc_auc")
            print(f"  Fixed BK classification AUC: {scores.mean():.3f}±{scores.std():.3f}")
        else:
            print(f"  Fixed: too few BK games ({y_bk[fix_mask].sum()}) for classification")

        # Variable BK classification for comparison
        X_var = hs[var_mask]
        y_var = y_bk[var_mask]
        sc = StandardScaler()
        X_s = sc.fit_transform(X_var)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        scores = cross_val_score(clf, X_s, y_var, cv=5, scoring="roc_auc")
        print(f"  Variable BK classification AUC: {scores.mean():.3f}±{scores.std():.3f}")

        # Balance variance comparison
        print(f"  Fixed balance std: {meta['balances'][fix_mask].std():.1f}")
        print(f"  Variable balance std: {meta['balances'][var_mask].std():.1f}")


# ===================================================================
# TEST 4: IC Hidden State Probe (was skipped in SAE)
# ===================================================================
def test_ic_hidden_probe():
    """Run hidden state BK probe on IC (not just SAE)."""
    print("\n" + "=" * 60)
    print("TEST 4: IC Hidden State BK Probe")
    print("  (Addresses: IC was skipped for I_BA/I_EC, but HS probe possible)")
    print("=" * 60)

    for model in ["gemma", "llama"]:
        for layer_idx in [2, 3]:
            hs, meta, layer = load_hs_dp(model, "ic", layer_idx)
            y = bk_labels(meta)
            bt = meta["bet_types"]

            print(f"\n--- {model.upper()} IC Layer {layer} ---")
            print(f"  Total: n={len(y)}, BK={y.sum()}")

            # All IC
            if y.sum() >= 5:
                sc = StandardScaler()
                X_s = sc.fit_transform(hs)
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                scores = cross_val_score(clf, X_s, y, cv=5, scoring="roc_auc")
                print(f"  All IC BK AUC: {scores.mean():.3f}±{scores.std():.3f}")

            # Variable only
            var = bt == "variable"
            if y[var].sum() >= 5:
                sc = StandardScaler()
                X_s = sc.fit_transform(hs[var])
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                scores = cross_val_score(clf, X_s, y[var], cv=5, scoring="roc_auc")
                print(f"  Variable IC BK AUC: {scores.mean():.3f}±{scores.std():.3f}")


if __name__ == "__main__":
    test_cross_paradigm_hs_transfer()
    test_cka()
    test_fixed_variance()
    test_ic_hidden_probe()
    print("\n\nDONE - All robustness checks complete.")
