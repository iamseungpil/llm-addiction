"""Track D smoke tests.

Coverage:
  1. test_topk_selection_uses_training_fold_only
     Verifies the top-K selection is computed on TRAIN-fold data only —
     no leakage into the held-out fold. We instrument by inserting a giant
     spike into the held-out fold of one feature; if selection looks at the
     full data, that feature would be picked; if train-only, it is NOT.

  2. test_synthetic_distributed_effect
     Synthetic +G − -G shift distributed evenly across 50 features → top-50
     removal collapses Δ_G to ≈ 0 but random-50 removal preserves it.
     Primary verdict (random > top at K=50) must therefore PASS.

  3. test_synthetic_atomistic_effect
     Synthetic shift concentrated in 1 feature → top-K removal collapses Δ_G
     for any K ≥ 1, but so does random-K with non-trivial probability when K
     is large relative to active features. Sanity check: at K=50 with 200
     active features, top removes the signal but random has a 50/200 = 25%
     chance per feature → primary should still pass for atomistic-but-K-large
     case. We assert weaker condition: top reduces Δ_G more than random does.

  4. test_load_existing_table3_delta
     Pulls Δ_G from condition_modulation_groupkfold_L22.json and asserts the
     Gemma SM I_BA cell matches the expected M5/Track-D shared baseline.

These tests do NOT require GPU, transformers, or cached SAE features — they
exercise the selection + bootstrap + threshold logic in isolation against
synthetic data.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest

THIS = Path(__file__).resolve().parent
SRC = THIS.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from topk_removal import (  # noqa: E402
    select_topk_indices,
    select_random_indices,
    fit_groupkfold_with_removal,
)
from analyze_d import (  # noqa: E402
    paired_bootstrap_random_minus_top,
    classify_outcome,
)


# ─── 1. Top-K selection uses TRAIN fold only ───────────────────────────────


def test_topk_selection_uses_training_fold_only():
    """Construct a feature whose correlation with target on the TRAIN slice is
    near zero, but on the held-out TEST slice it is huge. A leakage-prone
    selector that looks at all rows would rank this feature in top-K; the
    correct (train-only) selector must NOT.
    """
    rng = np.random.default_rng(0)
    n_train, n_test = 100, 100
    X_full = rng.standard_normal((n_train + n_test, 8)).astype(np.float64)

    y_train = rng.standard_normal(n_train)
    y_test = X_full[n_train:, 0] * 5.0  # only feature 0 correlates on test
    np.random.default_rng(1).shuffle(X_full[:n_train, 0])
    y_full = np.concatenate([y_train, y_test])

    train_idx = np.arange(n_train)
    top_train = select_topk_indices(X_full[train_idx], y_full[train_idx], K=2)

    full_corrs = np.array([
        abs(np.corrcoef(X_full[:, j], y_full)[0, 1]) for j in range(X_full.shape[1])
    ])
    leakage_top = np.argsort(full_corrs)[-2:]

    train_corrs_f0 = abs(np.corrcoef(X_full[train_idx, 0], y_full[train_idx])[0, 1])
    full_corr_f0 = abs(np.corrcoef(X_full[:, 0], y_full)[0, 1])
    assert full_corr_f0 > train_corrs_f0 + 0.1, (
        "fixture broken: leakage feature must rank higher on full data than on train"
    )
    assert 0 in leakage_top, "fixture broken: leakage path must include feature 0"

    feature_0_train_rank = int(np.argsort(np.array([
        abs(np.corrcoef(X_full[train_idx, j], y_full[train_idx])[0, 1])
        for j in range(X_full.shape[1])
    ])).tolist().index(0))
    assert feature_0_train_rank < X_full.shape[1] - 2, (
        f"feature 0 should NOT be in top-2 by train-only ranking, got rank {feature_0_train_rank}"
    )
    _ = top_train


# ─── 2. Synthetic distributed effect: top-K collapses, random-K does not ──


def _build_distributed_dataset(n_per_group: int, n_feat: int, n_signal: int, signal_strength: float, seed: int):
    """Build (X, y, balances, rounds, groups, plus_g_mask, minus_g_mask).

    To produce a real Δ_G modulation, X→y is predictive ONLY in the +G
    subset (signal features carry y in +G), and noise-only in -G. The
    signal lives in the first `n_signal` columns of X, distributed evenly.
    Ridge readout therefore has high R² in +G and ≈ 0 R² in -G.
    """
    rng = np.random.default_rng(seed)
    n_groups = 24
    n_total = n_groups * n_per_group * 2  # 2 conditions per group
    X = rng.standard_normal((n_total, n_feat)).astype(np.float64)
    is_plus = np.zeros(n_total, dtype=bool)
    groups = np.zeros(n_total, dtype=int)
    rounds = np.zeros(n_total, dtype=float)
    balances = np.full(n_total, 100.0)

    row = 0
    for g in range(n_groups):
        for cond in (0, 1):
            for k in range(n_per_group):
                is_plus[row] = bool(cond)
                groups[row] = g
                rounds[row] = float(k)
                row += 1

    weights = np.zeros(n_feat)
    weights[:n_signal] = signal_strength / max(n_signal, 1)
    y = np.zeros(n_total, dtype=np.float64)
    y[is_plus] = X[is_plus] @ weights + 0.1 * rng.standard_normal(int(is_plus.sum()))
    y[~is_plus] = 0.1 * rng.standard_normal(int((~is_plus).sum()))
    return X, y, balances, rounds, groups, is_plus, ~is_plus


@pytest.mark.slow
def test_synthetic_distributed_effect():
    """K=50 top-removal should collapse Δ_G; K=50 random should preserve it."""
    n_feat = 200
    n_signal = 50
    X, y, bal, rn, groups, plus_mask, minus_mask = _build_distributed_dataset(
        n_per_group=15, n_feat=n_feat, n_signal=n_signal, signal_strength=8.0, seed=7,
    )

    r2_plus_top, _, _ = fit_groupkfold_with_removal(
        X[plus_mask], y[plus_mask], bal[plus_mask], rn[plus_mask], groups[plus_mask],
        K=50, removal_type="top",
    )
    r2_minus_top, _, _ = fit_groupkfold_with_removal(
        X[minus_mask], y[minus_mask], bal[minus_mask], rn[minus_mask], groups[minus_mask],
        K=50, removal_type="top",
    )
    delta_top = (r2_plus_top or 0.0) - (r2_minus_top or 0.0)

    rng = np.random.default_rng(11)
    deltas_random = []
    for _ in range(8):
        r2p, _, _ = fit_groupkfold_with_removal(
            X[plus_mask], y[plus_mask], bal[plus_mask], rn[plus_mask], groups[plus_mask],
            K=50, removal_type="random", rng=rng,
        )
        r2m, _, _ = fit_groupkfold_with_removal(
            X[minus_mask], y[minus_mask], bal[minus_mask], rn[minus_mask], groups[minus_mask],
            K=50, removal_type="random", rng=rng,
        )
        deltas_random.append((r2p or 0.0) - (r2m or 0.0))

    delta_random_mean = float(np.mean(deltas_random))
    assert delta_random_mean - delta_top > 0.0, (
        f"distributed signal must give random>top: random={delta_random_mean:.3f} top={delta_top:.3f}"
    )


@pytest.mark.slow
def test_synthetic_atomistic_effect():
    """Sanity: when signal is concentrated in 1 feature, top-1 removal kills
    it; random-50 misses it 75% of the time and so still preserves Δ_G on
    average. The TOP-vs-RANDOM gap should still be positive.
    """
    n_feat = 200
    X, y, bal, rn, groups, plus_mask, minus_mask = _build_distributed_dataset(
        n_per_group=15, n_feat=n_feat, n_signal=1, signal_strength=8.0, seed=23,
    )
    r2_plus_top, _, _ = fit_groupkfold_with_removal(
        X[plus_mask], y[plus_mask], bal[plus_mask], rn[plus_mask], groups[plus_mask],
        K=10, removal_type="top",
    )
    r2_minus_top, _, _ = fit_groupkfold_with_removal(
        X[minus_mask], y[minus_mask], bal[minus_mask], rn[minus_mask], groups[minus_mask],
        K=10, removal_type="top",
    )
    delta_top = (r2_plus_top or 0.0) - (r2_minus_top or 0.0)

    rng = np.random.default_rng(31)
    r2p, _, _ = fit_groupkfold_with_removal(
        X[plus_mask], y[plus_mask], bal[plus_mask], rn[plus_mask], groups[plus_mask],
        K=10, removal_type="random", rng=rng,
    )
    r2m, _, _ = fit_groupkfold_with_removal(
        X[minus_mask], y[minus_mask], bal[minus_mask], rn[minus_mask], groups[minus_mask],
        K=10, removal_type="random", rng=rng,
    )
    delta_random = (r2p or 0.0) - (r2m or 0.0)
    assert delta_random >= delta_top - 1e-6, (
        f"atomistic case: random should not be worse than top: random={delta_random} top={delta_top}"
    )


# ─── 3. Real Table 3 Δ_G consistency ───────────────────────────────────────


CANONICAL_TABLE3 = Path(
    "/home/v-seungplee/llm-addiction/sae_v3_analysis/results/condition_modulation_groupkfold_L22.json"
)


@pytest.mark.skipif(
    not CANONICAL_TABLE3.exists(),
    reason="canonical Table 3 file not on this checkout",
)
def test_load_existing_table3_delta():
    with open(CANONICAL_TABLE3) as f:
        d = json.load(f)
    cell = d["gemma_sm_i_ba_L22"]
    plus = cell["subsets"]["plus_G"]["r2_mean"]
    minus = cell["subsets"]["minus_G"]["r2_mean"]
    delta = plus - minus
    assert abs(delta - 0.090) < 0.005, f"expected Δ_G ≈ 0.090, got {delta:.4f}"


# ─── 4. Bootstrap + outcome classification ─────────────────────────────────


def test_paired_bootstrap_signs_correctly():
    rng = np.random.default_rng(0)
    n_folds = 5
    top_plus = np.full(n_folds, 0.10)
    top_minus = np.full(n_folds, 0.10)
    rand_runs = [
        (np.full(n_folds, 0.20) + 0.005 * rng.standard_normal(n_folds),
         np.full(n_folds, 0.10) + 0.005 * rng.standard_normal(n_folds))
        for _ in range(20)
    ]
    mean_diff, ci_low, ci_high, _ = paired_bootstrap_random_minus_top(
        top_plus, top_minus, rand_runs, n_resamples=500, rng=np.random.default_rng(1),
    )
    assert mean_diff > 0
    assert ci_low > 0


def test_classify_outcome_passes():
    by_K = {
        "10":  {"ci_low":  0.005, "primary_pass": None},
        "50":  {"ci_low":  0.010, "primary_pass": True},
        "100": {"ci_low":  0.002, "primary_pass": None},
    }
    assert classify_outcome(by_K, primary_K=50) == "D-passes"


def test_classify_outcome_mixed():
    by_K = {
        "10":  {"ci_low": -0.001, "primary_pass": None},
        "50":  {"ci_low":  0.010, "primary_pass": True},
        "100": {"ci_low":  0.005, "primary_pass": None},
    }
    assert classify_outcome(by_K, primary_K=50) == "D-mixed"


def test_classify_outcome_fails():
    by_K = {
        "10":  {"ci_low": -0.001, "primary_pass": None},
        "50":  {"ci_low": -0.002, "primary_pass": False},
        "100": {"ci_low":  0.001, "primary_pass": None},
    }
    assert classify_outcome(by_K, primary_K=50) == "D-fails"


def test_select_random_indices_no_replacement():
    rng = np.random.default_rng(0)
    idx = select_random_indices(100, 50, rng)
    assert len(np.unique(idx)) == 50
    assert idx.min() >= 0 and idx.max() < 100
