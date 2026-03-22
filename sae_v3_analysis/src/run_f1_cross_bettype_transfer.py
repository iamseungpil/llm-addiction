#!/usr/bin/env python3
"""
F1: Cross-bet-type BK classifier transfer.

Train LogReg on Fixed BK → test on Variable (and vice versa).
If AUC >> 0.5, the BK representation is shared across bet types.

Runs on:
  - LLaMA IC: Hidden states (5 layers) + SAE (L22)
  - Gemma IC: SAE (L22) only (no Gemma HS available)

Output: JSON with per-direction, per-layer transfer AUC + 200-permutation p-values.
"""
import sys, json, numpy as np
from pathlib import Path
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from config import PARADIGMS, LLAMA_PARADIGMS

np.random.seed(42)

OUT = Path(__file__).parent.parent / "results" / "json"
OUT.mkdir(parents=True, exist_ok=True)

IC_HS = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3/investment_choice/llama/hidden_states_dp.npz")
N_PERM = 200


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def transfer_auc(X_train, y_train, X_test, y_test, n_perm=200):
    """PCA(50) → LogReg transfer AUC with permutation test."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    n_comp = min(50, X_tr.shape[1], X_tr.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)

    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)

    if len(np.unique(y_test)) < 2:
        return {'auc': float('nan'), 'perm_p': float('nan'), 'null_mean': float('nan')}

    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, proba)

    # Permutation test
    null_aucs = []
    for _ in range(n_perm):
        y_shuf = np.random.permutation(y_test)
        if len(np.unique(y_shuf)) < 2:
            continue
        null_aucs.append(roc_auc_score(y_shuf, proba))

    perm_p = np.mean([n >= auc for n in null_aucs]) if null_aucs else 1.0
    null_mean = np.mean(null_aucs) if null_aucs else 0.5

    return {'auc': round(auc, 4), 'perm_p': round(perm_p, 4), 'null_mean': round(null_mean, 4)}


def load_sae_dp(paradigm_cfg, layer):
    """Load SAE decision-point features for a paradigm."""
    npz_path = Path(paradigm_cfg['sae_dir']) / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None, None

    data = np.load(npz_path, allow_pickle=False)
    is_last = data['is_last_round'].astype(bool)
    dp_idx = np.where(is_last)[0]

    shape = tuple(data['shape'])
    row = data['row_indices']
    col = data['col_indices']
    vals = data['values']

    # Filter to DP rows only
    dp_set = set(dp_idx.tolist())
    mask = np.array([r in dp_set for r in row])
    row_dp = row[mask]
    col_dp = col[mask]
    vals_dp = vals[mask]

    # Remap row indices to 0..len(dp_idx)-1
    row_map = {old: new for new, old in enumerate(dp_idx)}
    row_new = np.array([row_map[r] for r in row_dp])

    dense = np.zeros((len(dp_idx), shape[1]), dtype=np.float32)
    dense[row_new, col_dp] = vals_dp

    # Filter active features (>= 1% activation rate)
    active = (dense > 0).mean(axis=0) >= 0.01
    dense = dense[:, active]

    meta = {
        'game_outcomes': data['game_outcomes'][dp_idx],
        'bet_types': data['bet_types'][dp_idx],
    }
    return dense, meta


# ══════════════════════════════════════════════
# LLaMA IC Hidden States
# ══════════════════════════════════════════════
def run_llama_hs():
    log("=" * 60)
    log("F1: CROSS-BET-TYPE TRANSFER — LLaMA IC Hidden States")
    log("=" * 60)

    if not IC_HS.exists():
        log("LLaMA IC HS not found, skipping")
        return {}

    data = np.load(IC_HS, allow_pickle=True)
    hs = data['hidden_states']
    layers = data['layers']
    outcomes = data['game_outcomes']
    bet_types = data['bet_types']
    labels = (outcomes == 'bankruptcy').astype(int) if outcomes.dtype.kind in ('U', 'O') else (outcomes == b'bankruptcy').astype(int)

    var_mask = bet_types == 'variable'
    fix_mask = bet_types == 'fixed'

    results = {}
    for j, layer in enumerate(layers):
        H = hs[:, j, :]

        X_var, y_var = H[var_mask], labels[var_mask]
        X_fix, y_fix = H[fix_mask], labels[fix_mask]

        n_var_bk, n_fix_bk = y_var.sum(), y_fix.sum()

        if n_var_bk < 5 or n_fix_bk < 5:
            log(f"  L{layer}: skipped (var_bk={n_var_bk}, fix_bk={n_fix_bk})")
            continue

        fix_to_var = transfer_auc(X_fix, y_fix, X_var, y_var, N_PERM)
        var_to_fix = transfer_auc(X_var, y_var, X_fix, y_fix, N_PERM)

        results[f'L{layer}'] = {
            'fix_to_var': fix_to_var,
            'var_to_fix': var_to_fix,
            'n_var': int(var_mask.sum()), 'n_var_bk': int(n_var_bk),
            'n_fix': int(fix_mask.sum()), 'n_fix_bk': int(n_fix_bk),
        }
        log(f"  L{layer}: Fix→Var AUC={fix_to_var['auc']:.3f} (p={fix_to_var['perm_p']:.3f}), "
            f"Var→Fix AUC={var_to_fix['auc']:.3f} (p={var_to_fix['perm_p']:.3f})")

    return results


# ══════════════════════════════════════════════
# LLaMA IC SAE
# ══════════════════════════════════════════════
def run_llama_sae():
    log("=" * 60)
    log("F1: CROSS-BET-TYPE TRANSFER — LLaMA IC SAE")
    log("=" * 60)

    results = {}
    for layer in [8, 12, 22, 25, 30]:
        dense, meta = load_sae_dp(LLAMA_PARADIGMS['ic'], layer)
        if dense is None:
            log(f"  L{layer}: SAE not found")
            continue

        labels = (meta['game_outcomes'] == 'bankruptcy').astype(int)
        bt = meta['bet_types']
        var_mask = bt == 'variable'
        fix_mask = bt == 'fixed'

        n_var_bk, n_fix_bk = labels[var_mask].sum(), labels[fix_mask].sum()
        if n_var_bk < 5 or n_fix_bk < 5:
            log(f"  L{layer}: skipped (var_bk={n_var_bk}, fix_bk={n_fix_bk})")
            continue

        fix_to_var = transfer_auc(dense[fix_mask], labels[fix_mask], dense[var_mask], labels[var_mask], N_PERM)
        var_to_fix = transfer_auc(dense[var_mask], labels[var_mask], dense[fix_mask], labels[fix_mask], N_PERM)

        results[f'L{layer}'] = {
            'fix_to_var': fix_to_var,
            'var_to_fix': var_to_fix,
            'n_active_features': dense.shape[1],
            'n_var': int(var_mask.sum()), 'n_var_bk': int(n_var_bk),
            'n_fix': int(fix_mask.sum()), 'n_fix_bk': int(n_fix_bk),
        }
        log(f"  L{layer}: Fix→Var AUC={fix_to_var['auc']:.3f} (p={fix_to_var['perm_p']:.3f}), "
            f"Var→Fix AUC={var_to_fix['auc']:.3f} (p={var_to_fix['perm_p']:.3f}) [{dense.shape[1]} features]")

    return results


# ══════════════════════════════════════════════
# Gemma IC SAE
# ══════════════════════════════════════════════
def run_gemma_sae():
    log("=" * 60)
    log("F1: CROSS-BET-TYPE TRANSFER — Gemma IC SAE")
    log("=" * 60)

    results = {}
    for layer in [10, 18, 22, 26, 30]:
        dense, meta = load_sae_dp(PARADIGMS['ic'], layer)
        if dense is None:
            log(f"  L{layer}: SAE not found")
            continue

        labels = (meta['game_outcomes'] == 'bankruptcy').astype(int)
        bt = meta['bet_types']
        var_mask = bt == 'variable'
        fix_mask = bt == 'fixed'

        n_var_bk, n_fix_bk = labels[var_mask].sum(), labels[fix_mask].sum()
        log(f"  L{layer}: Var n={var_mask.sum()} (BK={n_var_bk}), Fix n={fix_mask.sum()} (BK={n_fix_bk})")

        if n_var_bk < 5 or n_fix_bk < 5:
            log(f"  L{layer}: skipped (insufficient BK)")
            continue

        fix_to_var = transfer_auc(dense[fix_mask], labels[fix_mask], dense[var_mask], labels[var_mask], N_PERM)
        var_to_fix = transfer_auc(dense[var_mask], labels[var_mask], dense[fix_mask], labels[fix_mask], N_PERM)

        results[f'L{layer}'] = {
            'fix_to_var': fix_to_var,
            'var_to_fix': var_to_fix,
            'n_active_features': dense.shape[1],
            'n_var': int(var_mask.sum()), 'n_var_bk': int(n_var_bk),
            'n_fix': int(fix_mask.sum()), 'n_fix_bk': int(n_fix_bk),
        }
        log(f"  L{layer}: Fix→Var AUC={fix_to_var['auc']:.3f} (p={fix_to_var['perm_p']:.3f}), "
            f"Var→Fix AUC={var_to_fix['auc']:.3f} (p={var_to_fix['perm_p']:.3f}) [{dense.shape[1]} features]")

    return results


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
if __name__ == '__main__':
    import time
    start = time.time()

    all_results = {
        'metadata': {
            'analysis': 'F1_cross_bettype_transfer',
            'timestamp': datetime.now().isoformat(),
            'n_permutations': N_PERM,
            'method': 'StandardScaler → PCA(50) → LogReg(balanced) → AUC',
        },
        'llama_ic_hs': run_llama_hs(),
        'llama_ic_sae': run_llama_sae(),
        'gemma_ic_sae': run_gemma_sae(),
    }

    elapsed = time.time() - start
    log(f"\nAll F1 analyses complete in {elapsed:.0f}s")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = OUT / f"f1_cross_bettype_transfer_{ts}.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"Saved to {out_file}")
