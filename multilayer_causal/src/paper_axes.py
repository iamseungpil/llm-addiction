"""Build the W3 "paper-object" direction assets (offline, CPU).

Two families (RUN_PLAN_W3.md w3a / w3bk), runner npz schema throughout —
directions (42, 3584) float32 unit rows + scales (42,) = 0.03 * median
natural hidden norm of the estimation/eval subset per layer (axes.py
convention) + schema_version + gate fields recorded INSIDE the npz so a
failed builder gate auto-excludes the arm (RUN_PLAN_W3.md 구현 원칙).

saerd  The LITERAL Table-1 readout direction. PROVENANCE: the paper's §4.1
       pipeline (sae_v3_analysis/src/run_groupkfold_recompute.py via
       extract_section4_ridge_weights.py) — SAE features at L22, valid =
       variable-bet rows with I_BA>0, active features = columns with nnz>10,
       per-fold RF deconfound of I_BA on [bal, rn, bal^2, log1p(bal), bal*rn]
       (50 trees, depth 8, seed 42), top-200 by |Spearman| with the
       deconfounded target, StandardScaler + Ridge(alpha=100), 5-fold
       GroupKFold by game_id. The saved paper artifacts
       (gemma_sm_i_ba_L22.json + *_steering.json on HF, fold R^2 0.1666 =
       Table 1 Gemma SM I_BA cell) are REUSED: the npz direction is the
       stored d_unit = unit((w / scaler_scale) @ W_dec[selected]) through the
       Gemma-Scope L22 decoder (gemma-scope-9b-pt-res-canonical,
       width_131k). EVAL-STATE DISCIPLINE (round-1 leakage finding): the
       stored d_unit was fit on the full corpus, which CONTAINS the
       offset-300 eval-pool games (all 50 unique games of pool[300:350] sit
       in the 12246-row fit set), so the reproduction-gate refit, the
       per-layer scales and a steerable TWIN direction are all computed on
       the corpus with the axes.py eval-pool games EXCLUDED
       (excluded_game_ids, n_eval=500 — the W2/W3 offset-300 convention; the
       489-feature active subset stays the full-corpus one so the refit
       readout lives in the bit-identical feature space of the paper
       artifact — an nnz>10 sparsity filter only). The npz keeps the literal
       stored d_unit as `directions` (refitting it would forfeit the
       "literal Table-1 direction" claim), records BOTH the in-corpus and
       the exclusion-refit fold R^2, gates on the exclusion refit
       |R^2 - 0.167| <= 0.05, and discloses the eval-window overlap counts.
       A second runner-schema asset directions_saerd_excl.npz holds the
       exclusion-refit twin, decoder-mapped through the SAME canonical
       Gemma-Scope L22 W_dec — the W_dec source is validated in-builder by
       reconstructing the stored d_unit from the ridge artifact (cos >
       0.999). RUN_PLAN_W3.md PR-1: a positive w3a is claimed only if BOTH
       directions move behavior.

bk_sm / bk_ic / bk_sm_balres  LOTO rank-1 shared BK axes. PROVENANCE:
       endpoint labels mirror sae_v3_analysis/src/data_loader.get_labels /
       cross_domain.py (every decision row of a game carries the game's
       outcome; bankruptcy vs voluntary_stop) and the per-task contrast
       mirrors run_rq2_aligned_hidden_transfer.centroid_direction (raw L22
       states, no standardisation) with the W3-plan sign v_BK = mu_stop -
       mu_bankrupt. LOTO: the target task is EXCLUDED; the axis is the top
       right-singular vector of the stacked unit v_BK vectors of the other
       two tasks. This is a pre-registered NEW object, NOT a reproduction
       of Table 2: the Table-2 rank-1 cells (SM 0.8003 / IC 0.7355,
       v24_rq2_sweep_summary) are a different estimand — mean-centered PCA
       over the RAW centroid vectors of ALL THREE tasks (target included),
       scored by projecting the target's own fitted readout on game-level
       hidden_states_dp rows (run_rq2_aligned_hidden_transfer.py) — so
       they are not comparable and not gated against. Gates (honest
       construct criteria for the LOTO object itself):
       (a) relevance — game-level held-out projection AUC on the target,
           two-sided game-permutation p < 0.05 (statistic |AUC-0.5|,
           sign-agnostic because orientation is calibrated afterwards);
       (b) existence — game-bootstrap 2.5th percentile of
           cos(v_src1, v_src2) > 0 (a "shared rank-1 axis" must actually
           exist; near-orthogonal source pairs fail — the IC target's
           sources have pair cos ~ 0.02);
       (c) stability — game-level bootstrap (200 resamples) median axis
           cos > 0.9.
       The SAVED direction is sign-calibrated stop-ward in the TARGET's
       read geometry (orientation_sign = -1 flips the LOTO axis when its
       raw target game-AUC < 0.5, as for SM where the source-mean
       orientation is anti-aligned with SM's BK geometry); the span stays
       target-excluded, only the sign convention uses target endpoint
       labels. balres variant: source-task L22 states are residualised
       against current balance (per-dim linear regression) BEFORE the
       class means — the balance-confound control (RUN_PLAN_W3.md 비판 #4).
       PARTIAL-CONTROL DISCLOSURE (round-2 finding): source-side
       residualisation only PARTIALLY removes the balance signal — on the
       real corpora cos(bk_sm axis, bk_sm_balres axis) ~ 0.95 and the
       balres axis retains roughly half of the plain axis's
       balance alignment on target SM states. Both bk_sm npz files
       therefore record the quantification (cos_plain_vs_balres,
       balance_align_pearson/spearman on target + sources,
       cos_balance_slope_* vs per-task balance-slope directions, plus the
       counterpart axis's target numbers), and agreement between the
       w3bk_sm_p40 and w3bk_smres_p40 arms is interpreted as CONSISTENT
       WITH but NOT EXCLUDING a balance-direction account — only the
       divergence branch is discriminative (RUN_PLAN_W3.md mapping table).

bk_sm_balorth / bk_sm_balslope  DISCRIMINATIVE balance controls (round-3
       finding: cos(plain, balres) ~ 0.95 makes balres nearly powerless as
       a discriminator under n=50 steering, so a balance-confound reading
       of a positive w3bk SM result was unanswerable). Both reuse the
       plain (non-residualised) LOTO axis pipeline above plus
       balance_slope_direction on the TARGET SM states, same per-layer
       3%-norm scales as bk_sm (identical Delta-norm):
       balorth   = unit component of the oriented plain bk_sm axis
                   orthogonal to the target balance-slope direction —
                   cos(axis, balance-slope) = 0 BY CONSTRUCTION (target-
                   side, unlike source-side residualisation). A positive
                   steering effect shows the bk axis has causal content
                   outside the balance direction.
       balslope  = the pure target balance-slope direction itself,
                   sign-aligned with the plain axis's balance component —
                   the confound-POSITIVE control. If it reproduces the
                   bk_sm effect the balance account wins; if it is null
                   the balance account is EXCLUDED rather than merely
                   "not excluded" (the bk axis carries only |cos| ~ 0.4
                   of this direction at the same Delta-norm).
       Gates (construction validity, not shared-geometry claims):
       balorth — residual norm > ORTH_RESIDUAL_MIN (plain axis not
       parallel to balance-slope) + exact orthogonality; balslope —
       |cos(plain axis, balance-slope)| > BAL_COMPONENT_MIN (there must
       BE a balance component worth mimicking).

Usage:
  python -m multilayer_causal.src.paper_axes --which all \
      --dest-dir multilayer_causal/assets/w3
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .behavior_axis import CATALOG as SM_CATALOG
from .behavior_axis import HF_REPO, PHASE_A as SM_PHASE_A

# Model geometry registry — the single source of truth for (n_layers, d_model)
# (Karpathy: one MODEL_DIMS map, never scattered 42/3584 conditionals). Gemma is
# listed first and drives the module-level constants so every existing gemma
# builder/asset stays byte-identical; llama is additive (W8 model symmetry).
MODEL_DIMS = {"gemma": (42, 3584), "llama": (32, 4096)}
N_LAYERS, D_MODEL = MODEL_DIMS["gemma"]   # frozen gemma default (42, 3584)
# Model-aware §3 SM behavioural catalog (mirrors run_comprehensive_robustness
# .compute_iba's per-model SM file). SM_CATALOG stays the gemma file so every
# gemma builder is byte-identical; llama adds its v4_role catalog. The
# behavioural axis is a hidden mean-diff, so llama needs NO SAE decoder — only
# the readout axis (gemma-only) uses load_gemmascope_l22_wdec.
SM_CATALOG_BY_MODEL = {
    "gemma": SM_CATALOG,
    "llama": "behavioral/slot_machine/llama_v4_role/"
             "final_llama_20260315_062428.json",
}
# LLaMA SM all-layer hidden states (32 layers x 4096) for the W8 window scan.
# W8: full 32-layer phase_a hidden (re-extracted, round-level, fp16) — replaces
# the 5-layer game-level dp dump so LLaMA gets Gemma-identical arbitrary write
# windows. dp-parity: cos 0.9999 (axis-equivalent; ~1.5% scale offset cancels
# under scales_from_phase_a). MW mirror added for the reduced symmetric matrix.
LLAMA_SM_HIDDEN = "sae_features_v3/slot_machine/llama/checkpoints/phase_a_hidden_states.npz"
LLAMA_MW_HIDDEN = "sae_features_v3/mystery_wheel/llama/checkpoints/phase_a_hidden_states.npz"
# LLaMA IC all-layer hidden (W9 reduced symmetric matrix). W9-icextract re-dumps
# IC with full game_ids/round_nums/layers provenance (parity with the SM & MW
# W8-extract dumps) to a SEPARATE checkpoints_w9/ path so the existing count-only
# checkpoints/ IC file is left untouched (other experiments may still use it). The
# W9 loader guard (_llama_catalog_meta) refuses a count-only join, so we point at
# the provenance re-dump. (Old path: .../llama/checkpoints/phase_a_hidden_states.npz.)
LLAMA_IC_HIDDEN = "sae_features_v3/investment_choice/llama/checkpoints_w9/phase_a_hidden_states.npz"
LAYER = 22
SCALE_FRAC = 0.03
SCHEMA_VERSION = 1
GATE_TOL = 0.05
TABLE1_IBA_R2 = 0.167          # Table 1 Gemma SM I_BA cell
STABILITY_MIN_COS = 0.9
N_BOOT = 200
N_PERM = 10000                 # bk relevance gate: game-permutation draws
RELEVANCE_ALPHA = 0.05
TABLE2_NOTE = (
    "Table 2 rank-1 cells (SM 0.80 / IC 0.74) are a 3-task TARGET-INCLUSIVE "
    "quantity (mean-centered PCA of all three raw centroids, scored by the "
    "target's own fitted readout on game-level hidden_states_dp rows, "
    "run_rq2_aligned_hidden_transfer.py) — a different estimand from this "
    "target-excluded LOTO axis; NOT comparable, NOT gated against.")
# §4.1 pipeline constants (run_perm_null_ilc.py)
TOP_K, RF_TREES, RF_DEPTH, RIDGE_ALPHA, ACTIVE_MIN_NNZ = 200, 50, 8, 100.0, 10
# Eval-state discipline (round-1 leakage finding): exclusion horizon matches
# the W2/W3 axis rebuilds (axes.py --n-eval 500 covers the offset-300 waves);
# the w3a arms themselves evaluate pool[300:350] (state_offset 300, n=50).
EVAL_EXCLUDE_N = 500
W3_EVAL_OFFSET, W3_EVAL_N = 300, 50
# Gemma-Scope L22 canonical decoder for the exclusion-refit twin direction.
# "layer_22/width_131k/canonical" of release gemma-scope-9b-pt-res-canonical
# resolves to average_l0_105 (sae_lens pretrained_saes.yaml); the source is
# validated in-builder by reconstructing the stored d_unit from the ridge
# artifact through these W_dec rows (cos > DECODER_CHECK_MIN_COS).
GEMMASCOPE_REPO = "google/gemma-scope-9b-pt-res"
GEMMASCOPE_L22_PARAMS = "layer_22/width_131k/average_l0_105/params.npz"
DECODER_CHECK_MIN_COS = 0.999

SAE_L22 = {
    "sm": "sae_features_v3/slot_machine/gemma/sae_features_L22.npz",
    "ic": "sae_features_v3/investment_choice/gemma/sae_features_L22.npz",
    "mw": "sae_features_v3/mystery_wheel/gemma/sae_features_L22.npz",
}
PHASE_A = {
    "sm": SM_PHASE_A,
    "ic": "sae_features_v3/investment_choice/gemma/checkpoint/phase_a_hidden_states.npz",
    "mw": "sae_features_v3/mystery_wheel/gemma/checkpoint/phase_a_hidden_states.npz",
}
# Round-3 balance-control gates: balorth needs a non-degenerate non-balance
# residual (plain LOTO axis not parallel to the balance-slope direction;
# observed residual norm ~ 0.91); balslope needs a real balance component to
# mimic (observed |cos(plain axis, balance-slope)| ~ 0.41).
ORTH_RESIDUAL_MIN = 0.2
BAL_COMPONENT_MIN = 0.05
RIDGE_META = ("sae_v3_analysis/results/v19_multi_patching/M3prime_indicator_steering/"
              "direction_metadata/gemma_sm_i_ba_L22.json")
STEERING_META = ("sae_v3_analysis/results/v19_multi_patching/M3prime_indicator_steering/"
                 "direction_metadata/gemma_sm_i_ba_L22_steering.json")
BK_TASKS = ("sm", "ic", "mw")


# ---------------------------------------------------------------- pure math

def sm_catalog_for(model="gemma"):
    """Model-aware §3 SM behavioural catalog HF path (gemma default frozen)."""
    return SM_CATALOG_BY_MODEL[model]


def unit(v):
    """v / ||v|| (1e-12 guard)."""
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def gate_within(achieved, target, tol=GATE_TOL):
    """Reproduction gate: |achieved - target| <= tol (float-safe boundary)."""
    return bool(abs(float(achieved) - float(target)) <= tol + 1e-12)


def rf_deconfound_split(target_tr, bal_tr, rn_tr, target_te, bal_te, rn_te):
    """Residualise I_BA against balance+round with a random forest.

    EXACT mirror of run_perm_null_ilc.nl_deconfound_split (covariates
    [bal, rn, bal^2, log1p(bal), bal*rn]; RF 50 trees, depth 8, seed 42).
    """
    from sklearn.ensemble import RandomForestRegressor

    def _cov(b, r):
        return np.column_stack([b, r, b ** 2, np.log1p(b), b * r])

    rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=RF_DEPTH,
                               random_state=42, n_jobs=-1)
    rf.fit(_cov(bal_tr, rn_tr), target_tr)
    return (target_tr - rf.predict(_cov(bal_tr, rn_tr)),
            target_te - rf.predict(_cov(bal_te, rn_te)))


def spearman_topk(X, resid, k):
    """Top-k feature indices by |Spearman rho| with the deconfounded target
    (zero for constant columns), ascending-argsort tail — mirrors
    extract_section4_ridge_weights.fit_ridge_full_data."""
    from scipy.stats import spearmanr
    n_feat = X.shape[1]
    corrs = np.array([abs(spearmanr(X[:, j], resid)[0])
                      if X[:, j].std() > 0 else 0.0 for j in range(n_feat)])
    return np.argsort(corrs)[-min(k, n_feat):]


def groupkfold_readout_r2(X, target, balances, rounds, groups, n_splits=5):
    """§4.1 GroupKFold readout: per-fold deconfound + top-200 + Ridge.

    Returns (fold_r2 list, selected indices per fold) — the reproduction
    gate compares mean(fold_r2) to the Table 1 cell.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    fold_r2, fold_idx = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, groups=groups):
        res_tr, res_te = rf_deconfound_split(target[tr], balances[tr], rounds[tr],
                                             target[te], balances[te], rounds[te])
        idx = spearman_topk(X[tr], res_tr, TOP_K)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr][:, idx])
        Xte = sc.transform(X[te][:, idx])
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr)
        fold_r2.append(float(r2_score(res_te, ridge.predict(Xte))))
        fold_idx.append(idx)
    return fold_r2, fold_idx


def fit_full_ridge(X, target, balances, rounds, idx):
    """Full-data refit on the median-R^2 fold's features (paper recipe).

    Returns (w, scaler_scale) — the standardized-space Ridge coefficients
    whose raw-space readout is w / scaler_scale on the selected features.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    res_full, _ = rf_deconfound_split(target, balances, rounds,
                                      target, balances, rounds)
    sc = StandardScaler()
    Xs = sc.fit_transform(X[:, idx])
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(Xs, res_full)
    return ridge.coef_.astype(np.float64), sc.scale_.astype(np.float64)


def decoder_map(w, scaler_scale, dec_rows):
    """Map standardized-Ridge coefficients into residual space.

    pred = w . (F[idx] - mean) / scale, so the residual-space direction the
    readout responds to is sum_j (w_j / scale_j) * W_dec[idx_j] — mirrors
    compute_section4_steering_directions.py. Returns the unit vector.
    """
    w = np.asarray(w, dtype=np.float64)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
    dec_rows = np.asarray(dec_rows, dtype=np.float64)
    assert w.shape == scaler_scale.shape and len(w) == len(dec_rows)
    return unit((w / scaler_scale) @ dec_rows)


def sparse_readout_cos(w_a, idx_a, w_b, idx_b, n_feat):
    """Cosine of two readouts living on (possibly different) feature subsets."""
    va, vb = np.zeros(n_feat), np.zeros(n_feat)
    va[np.asarray(idx_a)] = w_a
    vb[np.asarray(idx_b)] = w_b
    return float(unit(va) @ unit(vb))


def exclusion_split(gids, excluded_games, eval_window_games):
    """Eval-leakage bookkeeping over fit rows (round-1 finding).

    Returns (keep, n_overlap_games, n_overlap_rows): keep = boolean mask of
    rows whose game is NOT in the axes.py excluded set (the leak-free fit
    subset); the overlap counts disclose how many of the w3a eval-window
    games / rows sit inside the ORIGINAL (in-corpus) fit set."""
    gids = np.asarray(gids)
    keep = ~np.isin(gids, sorted(excluded_games))
    in_window = np.isin(gids, sorted(eval_window_games))
    return keep, len(set(gids[in_window].tolist())), int(in_window.sum())


def vbk_axis(X, is_bk):
    """v_BK = unit(mean over voluntary-stop rows - mean over bankruptcy rows).

    Raw states, no standardisation — mirrors
    run_rq2_aligned_hidden_transfer.centroid_direction up to the W3-plan
    sign convention (mu_stop - mu_bankrupt)."""
    is_bk = np.asarray(is_bk, dtype=bool)
    assert is_bk.any() and (~is_bk).any(), "need both ending classes"
    return unit(X[~is_bk].mean(axis=0) - X[is_bk].mean(axis=0))


def loto_rank1(vs):
    """Top right-singular vector of stacked unit v_BK rows (target excluded
    upstream), sign-aligned with the mean of the stack."""
    vs = np.stack([unit(v) for v in vs], axis=0)
    _, _, vt = np.linalg.svd(vs, full_matrices=False)
    axis = vt[0]
    if axis @ vs.mean(axis=0) < 0:
        axis = -axis
    return axis


def balance_residualise(X, balances):
    """Per-dim linear regression of states on current balance; returns the
    residual states (intercept retained as the per-dim mean is removed by
    the downstream class-mean CONTRAST anyway, but kept for AUC reuse)."""
    X = np.asarray(X, dtype=np.float64)
    b = np.asarray(balances, dtype=np.float64)
    A = np.column_stack([b, np.ones_like(b)])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ coef


def balance_slope_direction(X, balances):
    """Unit per-dim linear-regression slope of states on current balance —
    the task's "balance direction" in residual space. Used only as a
    DIAGNOSTIC reference (round-2 finding): cos(saved axis, this) before vs
    after balres quantifies how much balance geometry the axis retains."""
    X = np.asarray(X, dtype=np.float64)
    b = np.asarray(balances, dtype=np.float64)
    A = np.column_stack([b, np.ones_like(b)])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return unit(coef[0])


def orthogonalise_against(v, ref):
    """Unit component of unit vector `v` orthogonal to unit vector `ref`.

    Returns (unit residual, residual norm). The residual norm equals
    cos(residual, v) = sqrt(1 - cos(v, ref)^2), so it doubles as the
    cos-to-original disclosure of the balorth control (round-3 finding)."""
    v, ref = unit(v), unit(ref)
    resid = v - float(v @ ref) * ref
    n = float(np.linalg.norm(resid))
    return unit(resid), n


def balance_alignment(X, balances, axis):
    """(Pearson, Spearman) of row projections onto `axis` vs current
    balance — the balance-readout check the round-2 audit found missing
    from the bk npz assets (auc_on_balres_states is a BK row-AUC, not a
    balance number)."""
    from scipy.stats import pearsonr, spearmanr
    proj = np.asarray(X, dtype=np.float64) @ np.asarray(axis, np.float64)
    b = np.asarray(balances, dtype=np.float64)
    return float(pearsonr(proj, b)[0]), float(spearmanr(proj, b)[0])


def projection_auc(X, is_bk, axis):
    """Centroid-distance classifier AUC: score = -(X @ axis) for the
    bankruptcy class (axis points stop-ward, so bankrupt rows project low)."""
    from sklearn.metrics import roc_auc_score
    scores = -(np.asarray(X, dtype=np.float64) @ np.asarray(axis, np.float64))
    return float(roc_auc_score(np.asarray(is_bk, dtype=int), scores))


def rank_auc(scores, labels):
    """Mann-Whitney AUC via average ranks (== roc_auc_score, tie-safe);
    rank-based so the permutation test below can reuse fixed ranks."""
    from scipy.stats import rankdata
    labels = np.asarray(labels, dtype=bool)
    r = rankdata(np.asarray(scores, dtype=np.float64))
    n1, n0 = int(labels.sum()), int((~labels).sum())
    assert n1 > 0 and n0 > 0, "need both classes"
    return float((r[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def game_projection_auc(X, game_ids, is_bk, axis):
    """Game-level centroid AUC: one score per game = -(mean row projection),
    same stop-ward sign convention as projection_auc. Outcome is a
    game-level label, so games are the exchangeable units for inference.

    Returns (auc, game_scores, game_bk)."""
    sums, counts, game_bk = per_game_sums(X, game_ids, is_bk)
    scores = -(sums @ np.asarray(axis, dtype=np.float64)) / counts
    return rank_auc(scores, game_bk), scores, game_bk


def auc_permutation_p(scores, labels, n_perm=N_PERM, seed=42):
    """Two-sided permutation p for AUC != 0.5 over exchangeable units.

    Statistic = |AUC - 0.5| (sign-agnostic: the saved direction's
    orientation is calibrated on the SAME labels afterwards, so a
    one-sided post-orientation test would double-dip). Labels are
    permuted; ranks of the fixed scores are reused. Add-one smoothing."""
    from scipy.stats import rankdata
    labels = np.asarray(labels, dtype=bool)
    r = rankdata(np.asarray(scores, dtype=np.float64))
    n1, n0 = int(labels.sum()), int((~labels).sum())
    assert n1 > 0 and n0 > 0, "need both classes"

    def _stat(lab):
        return abs((r[lab].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0) - 0.5)

    obs = _stat(labels)
    rng = np.random.default_rng(seed)
    hits = sum(_stat(rng.permutation(labels)) >= obs - 1e-12
               for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))


def per_game_sums(X, game_ids, is_bk):
    """Per-game row sums for fast game-level bootstrap.

    Returns (sums (G, d), counts (G,), game_is_bk (G,)) in unique-game order.
    Outcome is a game-level label, so every row of a game shares is_bk."""
    game_ids = np.asarray(game_ids)
    is_bk = np.asarray(is_bk, dtype=bool)
    uniq, inv = np.unique(game_ids, return_inverse=True)
    sums = np.zeros((len(uniq), X.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, np.asarray(X, dtype=np.float64))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    game_bk = np.zeros(len(uniq), dtype=bool)
    game_bk[inv] = is_bk  # consistent within game by construction
    return sums, counts, game_bk


def _vbk_from_sums(sums, counts, game_bk, pick):
    """v_BK from a bootstrap pick (game indices with replacement)."""
    bk = game_bk[pick]
    if not bk.any() or bk.all():
        return None
    mu_bk = sums[pick][bk].sum(axis=0) / counts[pick][bk].sum()
    mu_st = sums[pick][~bk].sum(axis=0) / counts[pick][~bk].sum()
    return unit(mu_st - mu_bk)


def bootstrap_axis_cos(point_axis, task_sums, n_boot=N_BOOT, seed=42,
                       return_pair_cos=False):
    """Game-level bootstrap of the LOTO axis.

    task_sums: list of (sums, counts, game_bk) per SOURCE task. Each
    resample redraws games with replacement within every source task,
    recomputes the v_BK stack and the rank-1 axis, and records its cosine
    with the point estimate. Degenerate resamples (single-class) are
    redrawn. Returns the (n_boot,) cosine array; with return_pair_cos also
    the (n_boot,) cos(v_src1, v_src2) array — the existence gate's
    bootstrap distribution (2-source LOTO only)."""
    rng = np.random.default_rng(seed)
    cosines = np.empty(n_boot)
    pair_cos = np.empty(n_boot)
    for b in range(n_boot):
        vs = []
        for sums, counts, game_bk in task_sums:
            v = None
            while v is None:
                pick = rng.integers(0, len(counts), size=len(counts))
                v = _vbk_from_sums(sums, counts, game_bk, pick)
            vs.append(v)
        cosines[b] = float(loto_rank1(vs) @ point_axis)
        pair_cos[b] = float(vs[0] @ vs[1]) if len(vs) == 2 else np.nan
    if return_pair_cos:
        return cosines, pair_cos
    return cosines


def replicate_rows(d):
    """(d_model,) -> (42, d_model) float32 unit rows: the single L22-derived
    vector is replicated to every layer row; PLACEMENT is the arm's `layers`
    field (RUN_PLAN_W3.md w3a)."""
    d = unit(d).astype(np.float32)
    return np.tile(d[None, :], (N_LAYERS, 1))


# ----------------------------------------------------------- data plumbing

def _hf_path(repo_file, token):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, repo_file, repo_type="dataset", token=token)


def load_sae_l22(task, token):
    """sae_features_L22.npz -> (csr matrix, meta dict). Mirrors
    run_perm_null_ilc.load_sae_and_meta (rows are decision rounds in
    extract_all_rounds adapter order — the SAME order as phase_a rows)."""
    from scipy import sparse
    z = np.load(_hf_path(SAE_L22[task], token), allow_pickle=False)
    sp = sparse.csr_matrix((z["values"], (z["row_indices"], z["col_indices"])),
                           shape=tuple(z["shape"]), dtype=np.float32)
    meta = {k: z[k] for k in z.files
            if k not in ("row_indices", "col_indices", "values", "shape")}
    return sp, meta


def load_phase_a_l22(task, token):
    """L22 slice of the all-layer phase_a hidden states, float32."""
    z = np.load(_hf_path(PHASE_A[task], token), mmap_mode="r")
    hs = z["hidden_states"]
    assert hs.shape[1] == N_LAYERS and hs.shape[2] == D_MODEL, hs.shape
    return np.asarray(hs[:, LAYER, :], dtype=np.float32), hs.shape[0]


def scales_from_phase_a(task, subset_idx, token):
    """0.03 * median per-layer hidden norm over subset rows — the exact
    axes.py _per_layer scales convention, one pass over the mmap."""
    z = np.load(_hf_path(PHASE_A[task], token), mmap_mode="r")
    hs = z["hidden_states"]
    scales = np.zeros(N_LAYERS, np.float32)
    for l in range(N_LAYERS):
        X = np.asarray(hs[subset_idx, l, :], dtype=np.float32)
        scales[l] = SCALE_FRAC * np.median(np.linalg.norm(X, axis=1))
    return scales


def load_gemmascope_l22_wdec(token):
    """Gemma-Scope L22 canonical W_dec (131072, 3584) from the MODEL repo
    google/gemma-scope-9b-pt-res (l0 path per GEMMASCOPE_L22_PARAMS); caller
    must run the stored-d_unit reconstruction check before trusting it."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(GEMMASCOPE_REPO, GEMMASCOPE_L22_PARAMS, token=token)
    return np.load(path)["W_dec"]


def compute_iba_sm(meta, catalog_path):
    """I_BA per SAE row for SM. Mirrors run_comprehensive_robustness
    .compute_iba: sequential 1-based game ids, non-skip decisions, bet from
    parsed_bet/bet/bet_amount, ratio = min(bet/balance_before, 1)."""
    raw = json.load(open(catalog_path))
    games = raw.get("results", raw.get("games", []))
    if isinstance(games, dict):
        games = list(games.values())
    game_map = {i + 1: g for i, g in enumerate(games)}
    n = len(meta["game_ids"])
    bet_ratios = np.full(n, np.nan)
    balances = meta["balances"].astype(float).copy()
    for i in range(n):
        g = game_map.get(int(meta["game_ids"][i]))
        if g is None:
            continue
        rn = int(meta["round_nums"][i]) - 1
        decs = [d for d in g.get("decisions", [])
                if d.get("action") != "skip" and not d.get("skipped", False)]
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


# ----------------------------------------------------------------- builders

def build_saerd(dest_dir, token):
    """w3a assets: the literal Table-1 L22 SAE-readout direction + its
    exclusion-refit twin.

    directions_saerd.npz       = stored paper d_unit (LITERAL Table-1);
    directions_saerd_excl.npz  = eval-pool-excluded refit, decoder-mapped.
    Reproduction gate (both npz) = EXCLUSION-refit fold R^2 vs Table 1
    (0.167 +- 0.05); the leaky in-corpus refit R^2 is recorded alongside
    for the paper disclosure (round-1 eval-state-leakage finding)."""
    from .axes import excluded_game_ids, minusG_pool_with_game_ids

    ridge_meta = json.load(open(_hf_path(RIDGE_META, token)))
    steer_meta = json.load(open(_hf_path(STEERING_META, token)))
    d_unit = np.asarray(steer_meta["d_unit"], dtype=np.float64)
    assert d_unit.shape == (D_MODEL,)
    assert ridge_meta["indicator"] == "i_ba" and ridge_meta["layer"] == LAYER

    sp, meta = load_sae_l22("sm", token)
    cat = _hf_path(SM_CATALOG, token)
    target, balances = compute_iba_sm(meta, cat)
    bt = meta["bet_types"]
    valid = ((bt == "variable") & ~np.isnan(target) & ~np.isnan(balances)
             & (balances > 0) & (target > 0))
    print(f"[saerd] n_valid={int(valid.sum())} "
          f"(paper n_samples={ridge_meta['n_samples']})")

    X_sparse = sp[valid]
    nnz = np.diff(X_sparse.tocsc().indptr)
    active = np.where(nnz > ACTIVE_MIN_NNZ)[0]
    X = X_sparse[:, active].toarray()
    t, bal = target[valid], balances[valid]
    rn = meta["round_nums"][valid].astype(float)
    gids = np.asarray(meta["game_ids"])[valid]
    print(f"[saerd] n_active={len(active)} "
          f"(paper {ridge_meta['n_features_total']}) -> GroupKFold refit")

    # Eval-state discipline: the eval-pool games sit inside the paper's fit
    # corpus, so gate refit / scales / twin all run on the exclusion subset
    # (axes.py excluded_game_ids, n_eval=500) and the overlap is disclosed.
    pool = minusG_pool_with_game_ids(cat)
    excluded = excluded_game_ids(pool, n_eval=EVAL_EXCLUDE_N)
    eval_window = {g for g, _ in pool[W3_EVAL_OFFSET:W3_EVAL_OFFSET + W3_EVAL_N]}
    keep, ov_games, ov_rows = exclusion_split(gids, excluded, eval_window)
    print(f"[saerd] eval-pool exclusion: {len(excluded)} games "
          f"(n_eval={EVAL_EXCLUDE_N}) -> fit rows {int(valid.sum())} -> "
          f"{int(keep.sum())} | w3a eval-window overlap in the in-corpus fit "
          f"set: {ov_games}/{len(eval_window)} games, {ov_rows} rows")

    fold_r2_ic, _ = groupkfold_readout_r2(X, t, bal, rn, gids)
    r2_in_corpus = float(np.mean(fold_r2_ic))  # leaky, artifact conditions
    Xx, tx = X[keep], t[keep]
    balx, rnx, gx = bal[keep], rn[keep], gids[keep]
    fold_r2, fold_idx = groupkfold_readout_r2(Xx, tx, balx, rnx, gx)
    r2_achieved = float(np.mean(fold_r2))
    gate = gate_within(r2_achieved, TABLE1_IBA_R2)
    median_fold = int(np.argsort(fold_r2)[len(fold_r2) // 2])
    w_refit, scale_refit = fit_full_ridge(Xx, tx, balx, rnx, fold_idx[median_fold])
    w_cos = sparse_readout_cos(
        w_refit / scale_refit, fold_idx[median_fold],
        np.asarray(ridge_meta["ridge_coef"]) / np.asarray(ridge_meta["scaler_scale"]),
        ridge_meta["feature_indices"], len(active))
    print(f"[saerd] exclusion-refit R^2={r2_achieved:.4f} "
          f"(folds {np.round(fold_r2, 4)}) gate={'PASS' if gate else 'FAIL'} | "
          f"in-corpus R^2={r2_in_corpus:.4f} | cos(refit w, stored w)={w_cos:.4f}")

    # Twin direction: exclusion refit through the SAME canonical L22 decoder.
    # W_dec source gate: the stored d_unit must reconstruct from the ridge
    # artifact through these rows (wrong l0 / repo fails loudly here).
    W_dec = load_gemmascope_l22_wdec(token)
    stored_full = np.asarray(steer_meta["feature_indices_full"])
    assert np.array_equal(active[np.asarray(ridge_meta["feature_indices"])],
                          stored_full), "stored feature ids != active mapping"
    d_check = decoder_map(np.asarray(ridge_meta["ridge_coef"]),
                          np.asarray(ridge_meta["scaler_scale"]),
                          W_dec[stored_full])
    decoder_check_cos = float(d_check @ unit(d_unit))
    assert decoder_check_cos > DECODER_CHECK_MIN_COS, \
        f"W_dec source check failed: cos={decoder_check_cos:.6f}"
    twin_full = active[np.asarray(fold_idx[median_fold])]
    d_twin = decoder_map(w_refit, scale_refit, W_dec[twin_full])
    twin_cos = float(d_twin @ unit(d_unit))
    print(f"[saerd] decoder check cos={decoder_check_cos:.6f} | "
          f"cos(twin direction, stored d_unit)={twin_cos:.4f}")

    scales = scales_from_phase_a("sm", np.where(valid)[0][keep], token)
    shared = dict(
        scales=scales, schema_version=SCHEMA_VERSION, gate_passed=gate,
        r2_achieved=r2_achieved, r2_target=TABLE1_IBA_R2, r2_tol=GATE_TOL,
        fold_r2=np.asarray(fold_r2), refit_vs_stored_w_cos=w_cos,
        r2_in_corpus=r2_in_corpus, fold_r2_in_corpus=np.asarray(fold_r2_ic),
        n_valid_rows=int(valid.sum()), n_fit_rows=int(keep.sum()),
        n_active_features=len(active), n_excluded_games=len(excluded),
        eval_exclude_n=EVAL_EXCLUDE_N, eval_window_n_games=len(eval_window),
        eval_overlap_games=ov_games, eval_overlap_rows=ov_rows,
        decoder_check_cos=decoder_check_cos, excl_vs_stored_dir_cos=twin_cos)
    common_prov = (
        f"reproduction refit per run_groupkfold_recompute.py on "
        f"{SAE_L22['sm']} with eval-pool games EXCLUDED (axes.py "
        f"excluded_game_ids, n_eval={EVAL_EXCLUDE_N}; in-corpus R^2 recorded "
        "for disclosure — the stored readout's corpus contains the offset-300 "
        "eval games); scales = 0.03*median per-layer norm of the ridge-valid "
        "eval-excluded SM phase_a rows.")
    dest = Path(dest_dir) / "directions_saerd.npz"
    np.savez(dest, directions=replicate_rows(d_unit),
             provenance=(
                 "LITERAL Table-1 Gemma SM I_BA L22 readout: stored paper "
                 f"d_unit from {STEERING_META} (Ridge w / scaler_scale through "
                 "Gemma-Scope gemma-scope-9b-pt-res-canonical width_131k L22 "
                 "W_dec rows of the 200 selected features); " + common_prov),
             **shared)
    print(f"[saerd] -> {dest}")
    dest_x = Path(dest_dir) / "directions_saerd_excl.npz"
    np.savez(dest_x, directions=replicate_rows(d_twin),
             provenance=(
                 "Exclusion-refit TWIN of the Table-1 Gemma SM I_BA L22 "
                 "readout: Ridge w / scaler_scale of the eval-excluded refit "
                 "(median-R^2 fold's features) through the SAME canonical "
                 f"Gemma-Scope L22 W_dec ({GEMMASCOPE_L22_PARAMS}, source "
                 "validated by reconstructing the stored d_unit, cos="
                 f"{decoder_check_cos:.6f}); PR-1 positive is claimed only if "
                 "BOTH this and the literal direction move behavior; "
                 + common_prov),
             **shared)
    print(f"[saerd] -> {dest_x}")
    return {"gate_passed": gate, "r2_achieved": r2_achieved,
            "r2_target": TABLE1_IBA_R2, "r2_in_corpus": r2_in_corpus,
            "refit_vs_stored_w_cos": w_cos,
            "excl_vs_stored_dir_cos": twin_cos,
            "decoder_check_cos": decoder_check_cos,
            "eval_overlap_games": ov_games, "eval_overlap_rows": ov_rows}


def _load_bk_task(task, token):
    """L22 states + endpoint labels + game ids + balances for one task,
    with the phase_a/SAE row-count alignment assert."""
    _, meta = load_sae_l22(task, token)
    X22, n_phase = load_phase_a_l22(task, token)
    assert len(meta["game_ids"]) == n_phase == len(X22), \
        f"{task}: SAE rows {len(meta['game_ids'])} != phase_a rows {n_phase}"
    is_bk = meta["game_outcomes"] == "bankruptcy"
    outcomes = sorted(set(np.asarray(meta["game_outcomes"]).tolist()))
    print(f"[bk] {task}: n={len(X22)} outcomes={outcomes} "
          f"bk_rows={int(is_bk.sum())}")
    return {"X": X22, "is_bk": np.asarray(is_bk, dtype=bool),
            "game_ids": np.asarray(meta["game_ids"]),
            "balances": meta["balances"].astype(np.float64)}


def _loto_axis_for(tasks_data, sources, balres):
    """(axis, v_BK list, source-state dict) for one variant — factored out
    so build_bk can also compute the COUNTERPART variant's axis (plain vs
    balres) for the cos_plain_vs_balres disclosure (round-2 finding)."""
    src_states = {}
    for t in sources:
        X = tasks_data[t]["X"]
        if balres:
            X = balance_residualise(X, tasks_data[t]["balances"])
        src_states[t] = X
    vs = [vbk_axis(src_states[t], tasks_data[t]["is_bk"]) for t in sources]
    return loto_rank1(vs), vs, src_states


def build_bk(dest_dir, target_task, tasks_data, token, balres=False):
    """w3bk asset: LOTO rank-1 shared BK axis for one target task.

    Pre-registered NEW object — not a Table-2 reproduction (see module
    docstring / TABLE2_NOTE). Gates: (a) relevance = game-level held-out
    AUC two-sided permutation p < RELEVANCE_ALPHA; (b) existence =
    game-bootstrap 2.5th pct of source pair cos > 0; (c) stability =
    bootstrap median axis cos > STABILITY_MIN_COS. The saved direction is
    orientation-calibrated stop-ward in the target's read geometry
    (orientation_sign recorded; the LOTO span itself stays
    target-excluded)."""
    sources = [t for t in BK_TASKS if t != target_task]
    axis, vs, src_states = _loto_axis_for(tasks_data, sources, balres)
    pair_cos = float(unit(vs[0]) @ unit(vs[1]))
    counterpart_axis, _, _ = _loto_axis_for(tasks_data, sources, not balres)

    tgt = tasks_data[target_task]
    auc_rows_raw = projection_auc(tgt["X"], tgt["is_bk"], axis)
    auc_game_raw, game_scores, game_bk = game_projection_auc(
        tgt["X"], tgt["game_ids"], tgt["is_bk"], axis)
    auc_game_p = auc_permutation_p(game_scores, game_bk)
    relevance_gate = bool(auc_game_p < RELEVANCE_ALPHA)

    orientation_sign = 1.0 if auc_game_raw >= 0.5 else -1.0
    axis_oriented = orientation_sign * axis
    auc_game_oriented = (auc_game_raw if orientation_sign > 0
                         else 1.0 - auc_game_raw)
    auc_rows_oriented = (auc_rows_raw if orientation_sign > 0
                         else 1.0 - auc_rows_raw)
    auc_balres_states = (projection_auc(
        balance_residualise(tgt["X"], tgt["balances"]), tgt["is_bk"],
        axis_oriented) if balres else np.nan)

    # Balance-alignment quantification (round-2 finding: balres is only a
    # PARTIAL control; without these numbers the npz contained no balance
    # readout at all). Counterpart = the other variant's axis, oriented by
    # ITS OWN raw target game-AUC (same calibration rule), so
    # cos_plain_vs_balres compares the two axes AS SAVED/steered.
    c_auc, _, _ = game_projection_auc(tgt["X"], tgt["game_ids"], tgt["is_bk"],
                                      counterpart_axis)
    counterpart_oriented = (1.0 if c_auc >= 0.5 else -1.0) * counterpart_axis
    cos_plain_vs_balres = float(axis_oriented @ counterpart_oriented)
    bal_pearson_tgt, bal_spearman_tgt = balance_alignment(
        tgt["X"], tgt["balances"], axis_oriented)
    cp_pearson_tgt, cp_spearman_tgt = balance_alignment(
        tgt["X"], tgt["balances"], counterpart_oriented)
    cos_bal_slope_tgt = float(axis_oriented @ balance_slope_direction(
        tgt["X"], tgt["balances"]))
    cp_cos_bal_slope_tgt = float(counterpart_oriented @ balance_slope_direction(
        tgt["X"], tgt["balances"]))
    src_bal_pearson, src_bal_spearman, src_cos_bal_slope = [], [], []
    for t in sources:
        p, s = balance_alignment(tasks_data[t]["X"], tasks_data[t]["balances"],
                                 axis_oriented)
        src_bal_pearson.append(p)
        src_bal_spearman.append(s)
        src_cos_bal_slope.append(float(axis_oriented @ balance_slope_direction(
            tasks_data[t]["X"], tasks_data[t]["balances"])))

    task_sums = [per_game_sums(src_states[t], tasks_data[t]["game_ids"],
                               tasks_data[t]["is_bk"]) for t in sources]
    cosines, boot_pair_cos = bootstrap_axis_cos(axis, task_sums,
                                                return_pair_cos=True)
    median_cos = float(np.median(cosines))
    stab_gate = bool(median_cos > STABILITY_MIN_COS)
    pair_cos_p2_5 = float(np.percentile(boot_pair_cos, 2.5))
    existence_gate = bool(pair_cos_p2_5 > 0.0)
    gate = bool(relevance_gate and existence_gate and stab_gate)

    name = f"bk_{target_task}" + ("_balres" if balres else "")
    print(f"[{name}] sources={sources} cos(v_src1, v_src2)={pair_cos:.3f} "
          f"(boot 2.5pct={pair_cos_p2_5:.3f}) "
          f"existence={'PASS' if existence_gate else 'FAIL'} | "
          f"game AUC={auc_game_raw:.4f} (rows {auc_rows_raw:.4f}) "
          f"perm p={auc_game_p:.5f} "
          f"relevance={'PASS' if relevance_gate else 'FAIL'} "
          f"orientation_sign={orientation_sign:+.0f} "
          f"-> oriented game AUC={auc_game_oriented:.4f} | "
          f"bootstrap median cos={median_cos:.4f} "
          f"stability={'PASS' if stab_gate else 'FAIL'} | "
          f"gate={'PASS' if gate else 'FAIL'}")
    print(f"[{name}] balance disclosure: cos(plain, balres)={cos_plain_vs_balres:.3f} | "
          f"target proj-balance r={bal_pearson_tgt:.3f}/rho={bal_spearman_tgt:.3f} "
          f"(counterpart {cp_pearson_tgt:.3f}/{cp_spearman_tgt:.3f}) | "
          f"cos(axis, target balance-slope)={cos_bal_slope_tgt:.3f} "
          f"(counterpart {cp_cos_bal_slope_tgt:.3f}) | "
          f"source r={np.round(src_bal_pearson, 3)} "
          f"slope-cos={np.round(src_cos_bal_slope, 3)}")

    label_idx = np.arange(len(tgt["X"]))  # all endpoint-labeled rows
    scales = scales_from_phase_a(target_task, label_idx, token)
    dest = Path(dest_dir) / f"directions_{name}.npz"
    np.savez(dest, directions=replicate_rows(axis_oriented), scales=scales,
             schema_version=SCHEMA_VERSION,
             gate_passed=gate,
             auc_game_raw=auc_game_raw, auc_game_oriented=auc_game_oriented,
             auc_game_p=auc_game_p, relevance_alpha=RELEVANCE_ALPHA,
             relevance_gate_passed=relevance_gate,
             auc_rows_raw=auc_rows_raw, auc_rows_oriented=auc_rows_oriented,
             orientation_sign=orientation_sign,
             auc_on_balres_states=auc_balres_states,
             cos_plain_vs_balres=cos_plain_vs_balres,
             balance_align_pearson_target=bal_pearson_tgt,
             balance_align_spearman_target=bal_spearman_tgt,
             cos_balance_slope_target=cos_bal_slope_tgt,
             counterpart_balance_align_pearson_target=cp_pearson_tgt,
             counterpart_balance_align_spearman_target=cp_spearman_tgt,
             counterpart_cos_balance_slope_target=cp_cos_bal_slope_tgt,
             balance_align_pearson_sources=np.asarray(src_bal_pearson),
             balance_align_spearman_sources=np.asarray(src_bal_spearman),
             cos_balance_slope_sources=np.asarray(src_cos_bal_slope),
             stability_median_cos=median_cos, stability_min_cos=STABILITY_MIN_COS,
             stability_gate_passed=stab_gate, bootstrap_cosines=cosines,
             source_pair_cos=pair_cos, bootstrap_pair_cos=boot_pair_cos,
             pair_cos_boot_p2_5=pair_cos_p2_5,
             existence_gate_passed=existence_gate,
             source_tasks=np.array(sources),
             balance_residualised=balres, table2_note=TABLE2_NOTE,
             n_target_rows=len(tgt["X"]), n_target_bk_rows=int(tgt["is_bk"].sum()),
             provenance=(
                 f"LOTO rank-1 BK axis, target={target_task} excluded; v_BK per "
                 "source task = unit(mu_stop - mu_bankrupt) of raw L22 phase_a "
                 "decision states (endpoint labels per data_loader.get_labels), "
                 "axis = top right-singular vector of the stacked unit v_BK "
                 f"vectors of {sources}; saved rows are orientation-calibrated "
                 "stop-ward in the target's read geometry "
                 f"(orientation_sign={orientation_sign:+.0f})"
                 + ("; source states balance-residualised per dim (linear) "
                    "before the contrast — a PARTIAL control only: see "
                    "cos_plain_vs_balres + balance_align_* keys; arm "
                    "agreement with the plain axis is consistent with but "
                    "does not exclude a balance-direction account"
                    if balres else
                    "; balance-alignment of this PLAIN axis is recorded in "
                    "the balance_align_* keys alongside the balres "
                    "counterpart's (round-2 disclosure: residualisation is "
                    "only a partial balance control)")
                 + "; scales = 0.03*median per-layer norm of the target task's "
                   "phase_a rows."))
    print(f"[{name}] -> {dest}")
    return {"gate_passed": gate,
            "cos_plain_vs_balres": cos_plain_vs_balres,
            "balance_align_pearson_target": bal_pearson_tgt,
            "balance_align_spearman_target": bal_spearman_tgt,
            "cos_balance_slope_target": cos_bal_slope_tgt,
            "relevance_gate_passed": relevance_gate,
            "auc_game_raw": auc_game_raw,
            "auc_game_oriented": auc_game_oriented,
            "auc_game_p": auc_game_p,
            "orientation_sign": orientation_sign,
            "existence_gate_passed": existence_gate,
            "source_pair_cos": pair_cos,
            "pair_cos_boot_p2_5": pair_cos_p2_5,
            "stability_median_cos": median_cos,
            "stability_gate_passed": stab_gate}


def build_bk_sm_balance_controls(dest_dir, tasks_data, token):
    """w3bk discriminative balance controls (round-3 finding) — two assets.

    balres only PARTIALLY removes balance (cos(plain, balres) ~ 0.95), so
    the w3bk_sm_p40/w3bk_smres_p40 pair will almost surely agree under
    n=50 steering and the balance-confound reading of a positive SM
    result stays unanswerable. These two same-Delta-norm controls make it
    answerable (see module docstring):

    directions_bk_sm_balorth.npz   plain oriented LOTO axis projected
        orthogonal to the TARGET SM balance-slope direction (cos = 0 by
        construction; orientation inherited from the plain axis).
    directions_bk_sm_balslope.npz  the pure target balance-slope
        direction, sign-aligned with the plain axis's balance component
        (+alpha pushes the same way as that component) — confound-
        positive control.

    Gates are construction-validity only (these assets claim no shared
    geometry of their own): balorth residual norm > ORTH_RESIDUAL_MIN;
    balslope |cos(plain, balance-slope)| > BAL_COMPONENT_MIN. Target
    game-AUC + permutation p and balance alignment are recorded as
    disclosures, not gates."""
    sources = [t for t in BK_TASKS if t != "sm"]
    axis, _, _ = _loto_axis_for(tasks_data, sources, balres=False)
    tgt = tasks_data["sm"]
    auc_plain_raw, _, _ = game_projection_auc(tgt["X"], tgt["game_ids"],
                                              tgt["is_bk"], axis)
    plain_oriented = (1.0 if auc_plain_raw >= 0.5 else -1.0) * axis
    bal_dir = balance_slope_direction(tgt["X"], tgt["balances"])
    cos_plain_bal = float(plain_oriented @ bal_dir)

    scales = scales_from_phase_a("sm", np.arange(len(tgt["X"])), token)
    results = {}
    variants = {
        "balorth": orthogonalise_against(plain_oriented, bal_dir),
        "balslope": ((1.0 if cos_plain_bal >= 0 else -1.0) * bal_dir,
                     abs(cos_plain_bal)),
    }
    for kind, (d, cos_to_plain) in variants.items():
        if kind == "balorth":
            gate = bool(cos_to_plain > ORTH_RESIDUAL_MIN
                        and abs(float(d @ bal_dir)) < 1e-8)
            gate_desc = (f"residual norm {cos_to_plain:.4f} > "
                         f"{ORTH_RESIDUAL_MIN} + exact orthogonality")
        else:
            gate = bool(abs(cos_plain_bal) > BAL_COMPONENT_MIN)
            gate_desc = (f"|cos(plain, balance-slope)| {abs(cos_plain_bal):.4f}"
                         f" > {BAL_COMPONENT_MIN}")
        auc_game, game_scores, game_bk = game_projection_auc(
            tgt["X"], tgt["game_ids"], tgt["is_bk"], d)
        auc_game_p = auc_permutation_p(game_scores, game_bk)
        bal_pearson, bal_spearman = balance_alignment(tgt["X"],
                                                      tgt["balances"], d)
        name = f"bk_sm_{kind}"
        print(f"[{name}] cos_to_plain={cos_to_plain:.4f} "
              f"cos(axis, target balance-slope)={float(d @ bal_dir):.4f} | "
              f"target game AUC={auc_game:.4f} (perm p={auc_game_p:.5f}, "
              f"disclosure) proj-balance r={bal_pearson:.3f}/"
              f"rho={bal_spearman:.3f} | gate({gate_desc})="
              f"{'PASS' if gate else 'FAIL'}")
        dest = Path(dest_dir) / f"directions_{name}.npz"
        np.savez(dest, directions=replicate_rows(d), scales=scales,
                 schema_version=SCHEMA_VERSION, gate_passed=gate,
                 control_kind=kind,
                 cos_to_plain_axis=cos_to_plain,
                 cos_balance_slope_target=float(d @ bal_dir),
                 plain_cos_balance_slope_target=cos_plain_bal,
                 balance_align_pearson_target=bal_pearson,
                 balance_align_spearman_target=bal_spearman,
                 auc_game_disclosure=auc_game,
                 auc_game_p_disclosure=auc_game_p,
                 orth_residual_min=ORTH_RESIDUAL_MIN,
                 bal_component_min=BAL_COMPONENT_MIN,
                 source_tasks=np.array(sources),
                 provenance=(
                     f"Round-3 discriminative balance control ({kind}) for "
                     "the w3bk SM arms; built from the SAME plain LOTO "
                     f"rank-1 BK axis as directions_bk_sm.npz (sources "
                     f"{sources}, stop-ward oriented) and the target SM "
                     "balance-slope direction (per-dim linear slope of raw "
                     "L22 phase_a states on current balance). "
                     + ("balorth = unit component of the oriented plain "
                        "axis orthogonal to the balance-slope direction — "
                        "cos(axis, balance-slope)=0 by construction; a "
                        "positive steering effect shows non-balance causal "
                        "content."
                        if kind == "balorth" else
                        "balslope = the balance-slope direction itself, "
                        "sign-aligned with the plain axis's balance "
                        "component — confound-POSITIVE control: if it "
                        "reproduces the bk_sm effect the balance account "
                        "wins, if null the account is excluded.")
                     + f" Gate = {gate_desc} (construction validity only); "
                     "AUC/balance numbers are disclosures, not gates; "
                     "scales = 0.03*median per-layer norm of the target "
                     "task's phase_a rows (same Delta-norm as bk_sm)."))
        print(f"[{name}] -> {dest}")
        results[name] = {"gate_passed": gate,
                         "cos_to_plain_axis": cos_to_plain,
                         "cos_balance_slope_target": float(d @ bal_dir),
                         "plain_cos_balance_slope_target": cos_plain_bal,
                         "balance_align_pearson_target": bal_pearson,
                         "auc_game_disclosure": auc_game,
                         "auc_game_p_disclosure": auc_game_p}
    return results


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["saerd", "bk_sm", "bk_ic", "bk_sm_balres",
                             "bk_sm_balctl", "all"])
    ap.add_argument("--dest-dir", required=True)
    args = ap.parse_args()
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    todo = (["saerd", "bk_sm", "bk_ic", "bk_sm_balres", "bk_sm_balctl"]
            if args.which == "all" else [args.which])

    results = {}
    if "saerd" in todo:
        results["saerd"] = build_saerd(dest_dir, token)
    bk_jobs = [w for w in todo if w.startswith("bk_")]
    if bk_jobs:
        tasks_data = {t: _load_bk_task(t, token) for t in BK_TASKS}
        for job in bk_jobs:
            if job == "bk_sm_balctl":
                results.update(build_bk_sm_balance_controls(
                    dest_dir, tasks_data, token))
                continue
            target = "sm" if "sm" in job else "ic"
            results[job] = build_bk(dest_dir, target, tasks_data, token,
                                    balres=job.endswith("balres"))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
