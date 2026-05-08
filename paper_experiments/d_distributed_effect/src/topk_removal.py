"""Top-K vs random-K SAE feature removal primitive for §4.3 distributed-effect robustness.

Pipeline parity (must match `sae_v3_analysis/src/run_groupkfold_recompute.py`):
  - 5-fold GroupKFold by game_id (no shuffle; deterministic group→fold map)
  - within-fold RandomForest deconfound on [bal, rn, bal², log1p(bal), bal·rn]
  - active-feature filter: nnz > 10 across rows (matches reference)
  - top-K=200 features by |Spearman ρ| with deconfounded target on TRAIN fold
  - StandardScaler + Ridge(α=100)

Track D adds one step inside the fold loop, BEFORE the top-K=200 readout selection:
  1. Rank active features by |spearmanr(F_train[:, j], y_deconf_train)|.
  2. removal_type='top':    drop the K features with the LARGEST |ρ| → simulates
                            atomistic ablation of the most-correlated features.
     removal_type='random': drop a uniformly random K-subset of active features
                            (seeded per replicate) → null baseline.
  3. Re-rank remaining features and run the standard top-200 Ridge readout on
     the survivors.

The selection step happens in the TRAINING fold only (no leakage). This is
asserted by `tests/test_d_smoke.py::test_topk_selection_uses_training_fold_only`.

We compute Δ_G = R²_+G − R²_-G by calling fit_groupkfold_with_removal() twice
per (model, K, removal_type) — once with the +G subset filter, once with the
−G subset filter — and subtracting.

Why a NEW fit_groupkfold_with_removal instead of importing fit_groupkfold from
sae_v3_analysis: the canonical helper takes a pre-built feature matrix X and
has no parameter to drop K columns inside the fold loop. We wrap the same
nl_deconfound_split + StandardScaler + Ridge primitives but interleave the
removal step. Numerical parity verified by setting K=0 → result equals
canonical fit_groupkfold.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

# Reuse the canonical deconfound + constants — single source of truth.
SAE_V3_SRC = Path(__file__).resolve().parents[3] / "sae_v3_analysis" / "src"
if str(SAE_V3_SRC) not in sys.path:
    sys.path.insert(0, str(SAE_V3_SRC))


def _import_canonical():
    """Lazy import so test collection does not require heavyweight deps."""
    from run_perm_null_ilc import nl_deconfound_split, TOP_K, RIDGE_ALPHA
    return nl_deconfound_split, TOP_K, RIDGE_ALPHA


def select_topk_indices(
    X_train: np.ndarray,
    y_deconf_train: np.ndarray,
    K: int,
) -> np.ndarray:
    """Top-K feature indices by |Spearman ρ| with deconfounded target.

    Operates on TRAIN-fold features only — the caller must pass the train slice.
    Matches the §4.1 selection rule (run_groupkfold_recompute.fit_groupkfold).
    """
    n_feat = X_train.shape[1]
    K = int(min(K, n_feat))
    corrs = np.array([
        abs(spearmanr(X_train[:, j], y_deconf_train)[0])
        if X_train[:, j].std() > 0 else 0.0
        for j in range(n_feat)
    ])
    return np.argsort(corrs)[-K:]


def select_random_indices(
    n_active: int,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly random K-subset of {0, …, n_active − 1} (no replacement)."""
    K = int(min(K, n_active))
    return rng.choice(n_active, size=K, replace=False)


def fit_groupkfold_with_removal(
    X_dense: np.ndarray,
    target: np.ndarray,
    balances: np.ndarray,
    rounds: np.ndarray,
    groups: np.ndarray,
    K: int,
    removal_type: str,
    rng: Optional[np.random.Generator] = None,
    n_splits: int = 5,
) -> Tuple[Optional[float], Optional[float], List[float]]:
    """Same pipeline as sae_v3_analysis fit_groupkfold but with a feature-removal
    step inserted between active filter and top-K=200 Ridge selection.

    Args:
      removal_type:
        - 'none':   no removal (parity with canonical fit_groupkfold; used in
                    tests to verify equivalence at K=0).
        - 'top':    drop the K features with largest |spearman ρ| in train fold.
        - 'random': drop a random K-subset (uses rng).

    Returns: (mean_r2, std_r2, per_fold_r2_list)
    """
    nl_deconfound_split, TOP_K, RIDGE_ALPHA = _import_canonical()

    n_unique_groups = int(len(np.unique(groups)))
    if n_unique_groups < n_splits:
        n_splits = max(2, n_unique_groups // 2)
    if n_unique_groups < 4:
        return None, None, []

    n_feat = X_dense.shape[1]
    if removal_type == "random" and rng is None:
        raise ValueError("removal_type='random' requires rng")

    gkf = GroupKFold(n_splits=n_splits)
    r2s: List[float] = []
    for tr, te in gkf.split(X_dense, groups=groups):
        try:
            res_tr, res_te = nl_deconfound_split(
                target[tr], balances[tr], rounds[tr],
                target[te], balances[te], rounds[te],
            )

            if removal_type == "none" or K <= 0:
                survivors = np.arange(n_feat)
            elif removal_type == "top":
                drop_idx = select_topk_indices(X_dense[tr], res_tr, K)
                survivors = np.setdiff1d(np.arange(n_feat), drop_idx, assume_unique=True)
            elif removal_type == "random":
                drop_idx = select_random_indices(n_feat, K, rng)  # type: ignore[arg-type]
                survivors = np.setdiff1d(np.arange(n_feat), drop_idx, assume_unique=True)
            else:
                raise ValueError(f"unknown removal_type {removal_type}")

            if survivors.size == 0:
                continue

            X_tr_s = X_dense[tr][:, survivors]
            X_te_s = X_dense[te][:, survivors]
            n_surv = X_tr_s.shape[1]

            corrs = np.array([
                abs(spearmanr(X_tr_s[:, j], res_tr)[0])
                if X_tr_s[:, j].std() > 0 else 0.0
                for j in range(n_surv)
            ])
            k_top = min(TOP_K, n_surv)
            top_idx = np.argsort(corrs)[-k_top:]

            sc = StandardScaler()
            Xtr = sc.fit_transform(X_tr_s[:, top_idx])
            Xte = sc.transform(X_te_s[:, top_idx])
            pred = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
            r2s.append(r2_score(res_te, pred))
        except Exception as e:
            print(f"    fold error: {type(e).__name__}: {e}")

    if not r2s:
        return None, None, []
    return (
        float(np.mean(r2s)),
        float(np.std(r2s, ddof=1) if len(r2s) > 1 else 0.0),
        [float(x) for x in r2s],
    )


def compute_delta_g_with_removal(
    X_dense: np.ndarray,
    target: np.ndarray,
    balances: np.ndarray,
    rounds: np.ndarray,
    groups: np.ndarray,
    valid_plus_g: np.ndarray,
    valid_minus_g: np.ndarray,
    K: int,
    removal_type: str,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Δ_G = R²_+G − R²_-G under feature removal.

    `valid_plus_g`, `valid_minus_g` are boolean masks INTO the rows of X_dense
    that are already filtered to the analysis-valid superset (variable bet,
    finite target, balance > 0). The caller is responsible for that prefilter
    so it matches sae_v3_analysis fit_one_subset semantics exactly.
    """
    out: Dict = {"K": int(K), "removal_type": removal_type}
    for subset_name, mask in (("plus_G", valid_plus_g), ("minus_G", valid_minus_g)):
        if mask.sum() < 100:
            out[subset_name] = {"reason": f"n<100 ({int(mask.sum())})", "n": int(mask.sum())}
            continue
        r2m, r2s, per_fold = fit_groupkfold_with_removal(
            X_dense[mask], target[mask], balances[mask], rounds[mask], groups[mask],
            K=K, removal_type=removal_type, rng=rng,
        )
        out[subset_name] = {
            "n": int(mask.sum()),
            "n_groups": int(len(np.unique(groups[mask]))),
            "r2_mean": r2m,
            "r2_std": r2s,
            "per_fold_r2": per_fold,
        }
    plus = out.get("plus_G", {}).get("r2_mean")
    minus = out.get("minus_G", {}).get("r2_mean")
    out["delta_g"] = (float(plus) - float(minus)) if (plus is not None and minus is not None) else None
    return out
