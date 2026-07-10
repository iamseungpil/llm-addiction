"""CPU-only unit tests for paper_axes.py — pure math, synthetic data only.

No downloads, no phase_a: the real builders are exercised only on synthetic
arrays through the factored pure functions (test_axes.py discipline).
"""
import numpy as np
import pytest

from multilayer_causal.src import paper_axes as pa


# ----------------------------------------------------------------- helpers

def _unit(v):
    return v / np.linalg.norm(v)


# --------------------------------------------------------------- gate logic

def test_gate_within_boundaries():
    assert pa.gate_within(0.167, 0.167)
    assert pa.gate_within(0.167 + 0.05, 0.167)
    assert pa.gate_within(0.167 - 0.05, 0.167)
    assert not pa.gate_within(0.167 + 0.0501, 0.167)
    assert not pa.gate_within(0.05, 0.167)
    assert pa.gate_within(0.80, 0.74, tol=0.06)


# -------------------------------------------------------------- decoder map

def test_decoder_map_tiny_synthetic_sae():
    # 3 selected features, d_model=4: closed-form (w/scale) @ W_dec rows.
    dec_rows = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 4.0]])
    w = np.array([2.0, 1.0, -1.0])
    scale = np.array([1.0, 2.0, 4.0])
    d = pa.decoder_map(w, scale, dec_rows)
    expect = _unit(np.array([2.0, 1.0, 0.0, -1.0]))
    assert np.allclose(d, expect)
    assert np.isclose(np.linalg.norm(d), 1.0)


def test_decoder_map_matches_standardized_readout_gradient():
    # The readout pred = w.(F[idx]-mu)/scale with F = h @ W_dec^T (linear
    # SAE) has gradient d pred/d h = (w/scale) @ W_dec[idx] — decoder_map
    # must return exactly that direction.
    rng = np.random.default_rng(0)
    n_feat, d_model = 6, 5
    W_dec = rng.standard_normal((n_feat, d_model))
    idx = np.array([0, 2, 5])
    w = rng.standard_normal(3)
    scale = rng.uniform(0.5, 2.0, 3)
    h = rng.standard_normal(d_model)
    eps = 1e-6

    def pred(hv):
        F = W_dec @ hv  # linear encode (tied weights)
        return float(w @ (F[idx] / scale))

    grad = np.array([(pred(h + eps * e) - pred(h - eps * e)) / (2 * eps)
                     for e in np.eye(d_model)])
    assert np.allclose(_unit(grad), pa.decoder_map(w, scale, W_dec[idx]),
                       atol=1e-5)


def test_sparse_readout_cos_identical_and_disjoint():
    w = np.array([1.0, 2.0])
    assert np.isclose(pa.sparse_readout_cos(w, [0, 1], w, [0, 1], 4), 1.0)
    assert np.isclose(pa.sparse_readout_cos(w, [0, 1], w, [2, 3], 4), 0.0)


# ----------------------------------------------------------------- BK axes

def test_vbk_axis_sign_is_stop_minus_bankrupt():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    v = pa.vbk_axis(X, is_bk=[False, False, True])
    assert np.allclose(v, [1.0, 0.0])  # stop centroid is +x
    with pytest.raises(AssertionError):
        pa.vbk_axis(X, is_bk=[True, True, True])


def test_loto_rank1_recovers_planted_shared_direction():
    rng = np.random.default_rng(1)
    shared = _unit(rng.standard_normal(16))
    vs = [_unit(shared + 0.05 * rng.standard_normal(16)) for _ in range(2)]
    axis = pa.loto_rank1(vs)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert axis @ shared > 0.98
    # the rank-1 axis averages out task noise: at least as close to the
    # planted direction as the worst single-task vector
    assert axis @ shared >= min(v @ shared for v in vs)


def test_loto_rank1_sign_aligned_with_stack_mean():
    v = _unit(np.array([1.0, 1.0, 0.0]))
    axis = pa.loto_rank1([v, v])
    assert axis @ v > 0.99  # not flipped to -v


def test_loto_rank1_norm_invariant_inputs():
    # Inputs are unit-normalized internally: a huge-norm task must not
    # dominate the rank-1 axis.
    a = _unit(np.array([1.0, 0.0, 0.0]))
    b = _unit(np.array([0.0, 1.0, 0.0]))
    axis_eq = pa.loto_rank1([a, b])
    axis_sc = pa.loto_rank1([1000.0 * a, b])
    assert np.allclose(np.abs(axis_eq), np.abs(axis_sc), atol=1e-9)


# ----------------------------------------------------------- residualisation

def test_balance_residualise_removes_planted_balance_component():
    rng = np.random.default_rng(2)
    n, d = 500, 8
    bal_dir = _unit(rng.standard_normal(d))
    sig_dir = _unit(rng.standard_normal(d) - (rng.standard_normal(d) @ bal_dir) * bal_dir)
    bal = rng.uniform(10, 200, n)
    sig = rng.standard_normal(n)
    X = bal[:, None] * bal_dir + sig[:, None] * sig_dir
    Xr = pa.balance_residualise(X, bal)
    # balance projection decorrelated, signal projection intact
    assert abs(np.corrcoef(Xr @ bal_dir, bal)[0, 1]) < 1e-6
    assert np.corrcoef(Xr @ sig_dir, sig)[0, 1] > 0.999


def test_balance_residualise_kills_balance_driven_vbk():
    # A fake BK contrast carried ENTIRELY by balance (bankrupt rows = low
    # balance) must vanish after residualisation.
    rng = np.random.default_rng(3)
    n, d = 400, 6
    bal_dir = _unit(rng.standard_normal(d))
    is_bk = np.arange(n) < 100
    bal = np.where(is_bk, 5.0, 150.0) + rng.uniform(0, 1, n)
    X = bal[:, None] * bal_dir + 0.01 * rng.standard_normal((n, d))
    v_raw = pa.vbk_axis(X, is_bk)
    v_res = X[~is_bk].mean(0) - X[is_bk].mean(0)
    assert abs(v_raw @ bal_dir) > 0.99
    Xr = pa.balance_residualise(X, bal)
    v_res = Xr[~is_bk].mean(0) - Xr[is_bk].mean(0)
    assert np.linalg.norm(v_res) < 0.05 * np.linalg.norm(
        X[~is_bk].mean(0) - X[is_bk].mean(0))


def test_balance_slope_direction_recovers_planted():
    rng = np.random.default_rng(20)
    n, d = 500, 8
    bal_dir = _unit(rng.standard_normal(d))
    bal = rng.uniform(10, 200, n)
    X = bal[:, None] * bal_dir + 0.1 * rng.standard_normal((n, d))
    assert pa.balance_slope_direction(X, bal) @ bal_dir > 0.99


def test_balance_alignment_quantifies_partial_removal():
    # Round-2 disclosure check: an axis carrying a balance component shows
    # strong projection-balance correlation on raw states; the same check
    # on residualised states (or an orthogonal axis) decorrelates — these
    # are the before/after numbers now recorded inside the bk npz assets.
    rng = np.random.default_rng(21)
    n, d = 600, 8
    bal_dir = _unit(rng.standard_normal(d))
    bal = rng.uniform(10, 200, n)
    X = bal[:, None] * bal_dir + 0.5 * rng.standard_normal((n, d))
    r_raw, rho_raw = pa.balance_alignment(X, bal, bal_dir)
    assert r_raw > 0.95 and rho_raw > 0.95
    r_res, rho_res = pa.balance_alignment(pa.balance_residualise(X, bal),
                                          bal, bal_dir)
    assert abs(r_res) < 0.05 and abs(rho_res) < 0.1


def test_orthogonalise_against_exact_and_decomposed():
    # Round-3 balorth control: the residual is EXACTLY orthogonal to the
    # reference, unit-norm, and its cos with the original equals
    # sqrt(1 - cos(v, ref)^2) — the recorded cos_to_plain_axis.
    rng = np.random.default_rng(22)
    d = 16
    ref = _unit(rng.standard_normal(d))
    v = _unit(rng.standard_normal(d))
    orth, resid_norm = pa.orthogonalise_against(v, ref)
    assert abs(orth @ ref) < 1e-10
    assert np.isclose(np.linalg.norm(orth), 1.0)
    assert np.isclose(orth @ v, resid_norm)
    assert np.isclose(resid_norm, np.sqrt(1.0 - float(_unit(v) @ ref) ** 2))
    # planted decomposition recovers the orthogonal part
    part = _unit(orth * 0.8 + ref * 0.6)
    o2, _ = pa.orthogonalise_against(part, ref)
    assert o2 @ orth > 0.999


# ------------------------------------------------------------ AUC + bootstrap

def test_projection_auc_sign_convention():
    # stop rows shifted +axis, bankrupt rows -axis: near-perfect AUC.
    rng = np.random.default_rng(4)
    axis = _unit(rng.standard_normal(10))
    is_bk = np.array([True] * 50 + [False] * 50)
    X = np.where(is_bk[:, None], -axis, axis) + 0.1 * rng.standard_normal((100, 10))
    assert pa.projection_auc(X, is_bk, axis) > 0.99
    assert pa.projection_auc(X, is_bk, -axis) < 0.01


def test_rank_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(10)
    scores = rng.standard_normal(80)
    labels = rng.random(80) < 0.3
    labels[0], labels[1] = True, False  # both classes guaranteed
    assert np.isclose(pa.rank_auc(scores, labels),
                      roc_auc_score(labels.astype(int), scores))


def test_game_projection_auc_separable_and_anti():
    # bankrupt games project LOW on the axis -> game-level AUC ~ 1;
    # flipping the axis must give 1 - AUC (the orientation_sign case).
    rng = np.random.default_rng(11)
    d, n_games, rows_per = 6, 40, 3
    axis = _unit(rng.standard_normal(d))
    X, gids, is_bk = [], [], []
    for g in range(n_games):
        bk = g < 10
        mu = (-2.0 if bk else 2.0) * axis
        X.append(mu + 0.1 * rng.standard_normal((rows_per, d)))
        gids += [g] * rows_per
        is_bk += [bk] * rows_per
    X = np.vstack(X)
    auc, scores, game_bk = pa.game_projection_auc(
        X, np.array(gids), np.array(is_bk), axis)
    assert len(scores) == len(game_bk) == n_games
    assert auc > 0.99
    auc_flip, _, _ = pa.game_projection_auc(
        X, np.array(gids), np.array(is_bk), -axis)
    assert np.isclose(auc_flip, 1.0 - auc)


def test_auc_permutation_p_signal_vs_null():
    rng = np.random.default_rng(12)
    n = 120
    labels = np.arange(n) < 40
    signal = np.where(labels, 1.0, -1.0) + 0.3 * rng.standard_normal(n)
    assert pa.auc_permutation_p(signal, labels, n_perm=500, seed=0) < 0.01
    noise = rng.standard_normal(n)
    assert pa.auc_permutation_p(noise, labels, n_perm=500, seed=0) > 0.05


def test_auc_permutation_p_two_sided_detects_anti_alignment():
    # An ANTI-aligned axis (AUC << 0.5, the SM case) must still be
    # significant under the |AUC - 0.5| statistic.
    rng = np.random.default_rng(13)
    n = 120
    labels = np.arange(n) < 40
    anti = np.where(labels, -1.0, 1.0) + 0.3 * rng.standard_normal(n)
    assert pa.rank_auc(anti, labels) < 0.1
    assert pa.auc_permutation_p(anti, labels, n_perm=500, seed=0) < 0.01


def test_per_game_sums_groups_rows():
    X = np.array([[1.0], [2.0], [10.0], [20.0], [30.0]])
    gids = np.array([7, 7, 9, 9, 9])
    is_bk = np.array([True, True, False, False, False])
    sums, counts, game_bk = pa.per_game_sums(X, gids, is_bk)
    assert np.allclose(sums, [[3.0], [60.0]])
    assert np.allclose(counts, [2.0, 3.0])
    assert game_bk.tolist() == [True, False]


def test_bootstrap_axis_cos_stable_synthetic():
    rng = np.random.default_rng(5)
    d, shared = 12, _unit(np.arange(1.0, 13.0))
    task_sums = []
    for _ in range(2):
        n_games = 60
        rows = []
        is_bk_rows, gid_rows = [], []
        for g in range(n_games):
            bk = g < 15
            mu = -shared if bk else shared
            rows.append(mu + 0.05 * rng.standard_normal((4, d)))
            is_bk_rows += [bk] * 4
            gid_rows += [g] * 4
        X = np.vstack(rows)
        task_sums.append(pa.per_game_sums(X, np.array(gid_rows),
                                          np.array(is_bk_rows)))
    vs = []
    for sums, counts, game_bk in task_sums:
        mu_bk = sums[game_bk].sum(0) / counts[game_bk].sum()
        mu_st = sums[~game_bk].sum(0) / counts[~game_bk].sum()
        vs.append(_unit(mu_st - mu_bk))
    point = pa.loto_rank1(vs)
    cosines = pa.bootstrap_axis_cos(point, task_sums, n_boot=50, seed=0)
    assert cosines.shape == (50,)
    assert np.median(cosines) > 0.99


def _make_task_sums(rng, dirs, n_games=60, rows_per=4, noise=0.05, d=12):
    """Synthetic (sums, counts, game_bk) per task with planted v_BK dirs."""
    task_sums = []
    for mu_dir in dirs:
        rows, is_bk_rows, gid_rows = [], [], []
        for g in range(n_games):
            bk = g < 15
            mu = -mu_dir if bk else mu_dir
            rows.append(mu + noise * rng.standard_normal((rows_per, d)))
            is_bk_rows += [bk] * rows_per
            gid_rows += [g] * rows_per
        task_sums.append(pa.per_game_sums(np.vstack(rows),
                                          np.array(gid_rows),
                                          np.array(is_bk_rows)))
    return task_sums


def test_bootstrap_pair_cos_shared_vs_orthogonal_sources():
    # Existence gate distribution: shared sources -> 2.5th pct well above
    # zero; orthogonal sources (the bk_ic pair_cos~0.02 situation) -> pair
    # cos hovers near zero, far below the shared case.
    rng = np.random.default_rng(14)
    d = 12
    shared = _unit(np.arange(1.0, d + 1.0))
    ts_shared = _make_task_sums(rng, [shared, shared], d=d)
    point = pa.loto_rank1([shared, shared])
    cos_s, pair_s = pa.bootstrap_axis_cos(point, ts_shared, n_boot=50,
                                          seed=0, return_pair_cos=True)
    assert pair_s.shape == (50,)
    assert np.percentile(pair_s, 2.5) > 0.9

    e1, e2 = np.eye(d)[0], np.eye(d)[1]
    ts_orth = _make_task_sums(rng, [e1, e2], noise=0.3, d=d)
    _, pair_o = pa.bootstrap_axis_cos(pa.loto_rank1([e1, e2]), ts_orth,
                                      n_boot=50, seed=0,
                                      return_pair_cos=True)
    assert abs(np.median(pair_o)) < 0.3
    assert np.percentile(pair_o, 2.5) < np.percentile(pair_s, 2.5) - 0.5


def test_bootstrap_axis_cos_default_signature_unchanged():
    # Frozen-call discipline: the original single-array return must hold
    # when return_pair_cos is not requested.
    rng = np.random.default_rng(15)
    shared = _unit(rng.standard_normal(8))
    ts = _make_task_sums(rng, [shared, shared], n_games=30, d=8)
    out = pa.bootstrap_axis_cos(pa.loto_rank1([shared, shared]), ts,
                                n_boot=10, seed=0)
    assert isinstance(out, np.ndarray) and out.shape == (10,)


# --------------------------------------------------- readout refit pipeline

def test_rf_deconfound_split_removes_balance_signal():
    rng = np.random.default_rng(6)
    n = 400
    bal = rng.uniform(10, 200, n)
    rn = rng.integers(1, 10, n).astype(float)
    extra = rng.standard_normal(n)
    target = 0.01 * bal + 0.5 * extra
    res_tr, res_te = pa.rf_deconfound_split(target[:300], bal[:300], rn[:300],
                                            target[300:], bal[300:], rn[300:])
    assert abs(np.corrcoef(res_te, bal[300:])[0, 1]) < 0.4
    assert np.corrcoef(res_te, extra[300:])[0, 1] > 0.6


def test_spearman_topk_selects_planted_columns():
    rng = np.random.default_rng(7)
    n, n_feat = 300, 12
    resid = rng.standard_normal(n)
    X = rng.standard_normal((n, n_feat))
    X[:, 3] = resid + 0.1 * rng.standard_normal(n)
    X[:, 8] = -resid + 0.1 * rng.standard_normal(n)  # |rho| counts
    X[:, 0] = 1.0  # constant column must score 0
    idx = pa.spearman_topk(X, resid, 2)
    assert set(idx.tolist()) == {3, 8}


def test_groupkfold_readout_r2_recovers_feature_signal():
    rng = np.random.default_rng(8)
    n, n_feat, n_games = 600, 15, 60
    groups = np.repeat(np.arange(n_games), n // n_games)
    bal = rng.uniform(10, 200, n)
    rn = rng.integers(1, 11, n).astype(float)
    X = rng.standard_normal((n, n_feat))
    target = 2.0 * X[:, 4] + 0.002 * bal + 0.1 * rng.standard_normal(n)
    fold_r2, fold_idx = pa.groupkfold_readout_r2(X, target, bal, rn, groups,
                                                 n_splits=3)
    assert len(fold_r2) == 3 and all(r > 0.5 for r in fold_r2)
    assert all(4 in idx for idx in fold_idx)
    w, scale = pa.fit_full_ridge(X, target, bal, rn, fold_idx[0])
    raw = np.zeros(n_feat)
    raw[fold_idx[0]] = w / scale
    assert np.argmax(np.abs(raw)) == 4


# ------------------------------------------------------------- npz plumbing

def test_replicate_rows_unit_and_shape():
    d = np.arange(1.0, float(pa.D_MODEL) + 1.0)
    rows = pa.replicate_rows(d)
    assert rows.shape == (pa.N_LAYERS, pa.D_MODEL)
    assert rows.dtype == np.float32
    norms = np.linalg.norm(rows, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.allclose(rows[0], rows[41])


# --------------------------------------------------- eval-pool exclusion fix

def test_exclusion_split_masks_and_overlap_counts():
    # fit rows over games 1..5; axes-excluded {2, 4}; eval window {2, 3}.
    gids = np.array([1, 2, 2, 3, 4, 5, 5])
    keep, ov_games, ov_rows = pa.exclusion_split(gids, {2, 4}, {2, 3})
    assert keep.tolist() == [True, False, False, True, False, True, True]
    assert ov_games == 2 and ov_rows == 3  # games {2,3}; rows of 2 (x2) + 3


def test_exclusion_split_window_game_absent_from_fit():
    # an eval-window game with no fit rows must not count in the overlap.
    gids = np.array([1, 1, 5])
    keep, ov_games, ov_rows = pa.exclusion_split(gids, {7}, {7, 5})
    assert keep.tolist() == [True, True, True]
    assert ov_games == 1 and ov_rows == 1


def test_exclusion_split_no_overlap_is_identity():
    gids = np.array([10, 11, 12])
    keep, ov_games, ov_rows = pa.exclusion_split(gids, set(), set())
    assert keep.all() and ov_games == 0 and ov_rows == 0
