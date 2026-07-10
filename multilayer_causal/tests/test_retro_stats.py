import numpy as np
import pytest
from scipy import stats

from multilayer_causal.src.retro_stats import (cluster_perm_test, ilc_dec,
                                               parse_excluded,
                                               tost_gap_recovery)


# ----------------------------------------------------------- cluster perm

def _clustered(rng, n_clusters, per, shift=0.0):
    vals, clusters = [], []
    for c in range(n_clusters):
        vals.extend(rng.normal(shift, 0.5, per))
        clusters.extend([c] * per)
    return vals, clusters


def test_perm_detects_planted_effect():
    rng = np.random.default_rng(1)
    a_vals, a_cl = _clustered(rng, 30, 5, shift=1.0)
    b_vals, b_cl = _clustered(rng, 30, 5, shift=0.0)
    p2 = cluster_perm_test(a_vals, a_cl, b_vals, b_cl, n_perm=2000)
    pg = cluster_perm_test(a_vals, a_cl, b_vals, b_cl, n_perm=2000,
                           alternative="greater")
    assert p2 < 0.01 and pg < 0.01


def test_perm_null_is_calibrated():
    rng = np.random.default_rng(7)
    a_vals, a_cl = _clustered(rng, 30, 5)
    b_vals, b_cl = _clustered(rng, 30, 5)
    p = cluster_perm_test(a_vals, a_cl, b_vals, b_cl, n_perm=2000, seed=0)
    assert p > 0.2


def test_perm_relabels_whole_clusters_and_is_deterministic():
    # Values constant WITHIN clusters: an iid-shuffle test would call this
    # wildly significant; with only 3+3 cluster units p is bounded below.
    a_vals = [10.0] * 5 + [10.1] * 5 + [9.9] * 5
    a_cl = [0] * 5 + [1] * 5 + [2] * 5
    b_vals = [0.0] * 5 + [0.1] * 5 + [-0.1] * 5
    b_cl = [3] * 5 + [4] * 5 + [5] * 5
    p1 = cluster_perm_test(a_vals, a_cl, b_vals, b_cl, n_perm=2000, seed=3)
    p2 = cluster_perm_test(a_vals, a_cl, b_vals, b_cl, n_perm=2000, seed=3)
    assert p1 == p2
    assert p1 >= 2 / 20          # only C(6,3)=20 distinct relabelings


# ------------------------------------------------------------------- TOST

MINUS = [0.0, 0.1, -0.1, 0.05, -0.05, 0.02]          # mean ~0
PLUS = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02]             # mean ~1.003


def test_tost_full_recovery_known_numbers():
    treat = [1.0, 0.9, 1.1, 0.95, 1.05, 1.02]
    out = tost_gap_recovery(treat, PLUS, MINUS, margin_frac=0.25)
    gap = np.mean(PLUS) - np.mean(MINUS)
    assert out["gap"] == pytest.approx(gap)
    assert out["recovery"] == pytest.approx((np.mean(treat) - np.mean(MINUS)) / gap)
    margin = 0.25 * gap
    p_lo = stats.ttest_ind(treat, np.array(PLUS) - margin, equal_var=False,
                           alternative="greater").pvalue
    p_hi = stats.ttest_ind(treat, np.array(PLUS) + margin, equal_var=False,
                           alternative="less").pvalue
    assert out["tost_p"] == pytest.approx(max(p_lo, p_hi))
    assert out["tost_p"] < 0.05                       # equivalent to +G
    assert out["ci_method"] == "normal_approx"
    lo, hi = out["recovery_ci95"]
    assert lo < out["recovery"] < hi


def test_tost_half_recovery_fails_equivalence():
    treat = [0.5, 0.4, 0.6, 0.45, 0.55, 0.52]
    out = tost_gap_recovery(treat, PLUS, MINUS, margin_frac=0.25)
    assert out["recovery"] == pytest.approx(0.5, abs=0.02)
    assert out["tost_p"] > 0.05                       # NOT within ±25% of +G


def test_tost_cluster_bootstrap_ci():
    rng = np.random.default_rng(0)
    minus, m_cl = [], []
    plus, p_cl = [], []
    treat, t_cl = [], []
    for c in range(25):
        minus.extend(rng.normal(0.0, 0.1, 4)); m_cl.extend([c] * 4)
        plus.extend(rng.normal(1.0, 0.1, 4)); p_cl.extend([c] * 4)
        treat.extend(rng.normal(1.0, 0.1, 4)); t_cl.extend([c] * 4)
    out = tost_gap_recovery(treat, plus, minus, treat_clusters=t_cl,
                            plus_clusters=p_cl, minus_clusters=m_cl)
    assert out["ci_method"] == "cluster_bootstrap"
    assert out["recovery"] == pytest.approx(1.0, abs=0.1)
    lo, hi = out["recovery_ci95"]
    assert lo < out["recovery"] < hi and hi - lo < 0.5
    with pytest.raises(AssertionError):               # partial clusters: loud
        tost_gap_recovery(treat, plus, minus, treat_clusters=t_cl)


# ---------------------------------------------------------------- I_LC^dec

def _game(prev_entries):
    return {"history": prev_entries, "bet_type": "variable"}


def test_ilc_dec_loss_reconstruction_arithmetic():
    # Displayed prev round = hist[round_idx-1] = LOSS: bet 20, balance AFTER 60
    # -> balance_before 80, r_prev = 0.25.  r_t = 0.5 -> (0.5-0.25)/0.25 = 1.0
    loss_game = _game([{"round": 1, "bet": 10, "win": True, "balance": 120},
                       {"round": 2, "bet": 20, "win": False, "balance": 60}])
    pool = [(loss_game, 2)]
    trials = [{"trial_id": 0, "phase": "e1", "action": "bet", "bet_ratio": 0.5}]
    assert ilc_dec(trials, pool) == [1.0]
    # de-escalation clips at 0
    trials[0]["bet_ratio"] = 0.1
    assert ilc_dec(trials, pool) == [0.0]
    # stop trials are excluded
    assert ilc_dec([{"trial_id": 0, "phase": "e1", "action": "stop",
                     "bet_ratio": 0.0}], pool) == []


def test_ilc_dec_skips_win_prev_and_result_field_fallback():
    win_game = _game([{"round": 1, "bet": 10, "win": True, "balance": 120}])
    # legacy entries carry "result" instead of "win" (build_prompt fallback)
    res_loss = _game([{"round": 1, "bet": 25, "result": "L", "balance": 75}])
    pool = [(win_game, 1), (res_loss, 1)]
    trials = [{"trial_id": 0, "phase": "e1", "action": "bet", "bet_ratio": 0.5},
              {"trial_id": 1, "phase": "e1", "action": "bet", "bet_ratio": 0.5}]
    out = ilc_dec(trials, pool)
    # win-prev excluded; result-loss: bal_before=100, r_prev=0.25 -> 1.0
    assert out == [1.0]


def test_ilc_dec_e1c_offset_is_100():
    loss_game = _game([{"round": 1, "bet": 20, "win": False, "balance": 60}])
    win_game = _game([{"round": 1, "bet": 10, "win": True, "balance": 120}])
    pool = [(loss_game, 1), (win_game, 1), (loss_game, 1)]
    t = {"trial_id": 0, "action": "bet", "bet_ratio": 0.5}
    # phase e1: index 0 -> loss state -> included
    assert ilc_dec([{**t, "phase": "e1"}], pool) == [1.0]
    # phase e1c: index (0+100) % 3 = 1 -> win state -> excluded
    assert ilc_dec([{**t, "phase": "e1c"}], pool) == []


# ----------------------------------------------------------- parse_excluded

def test_parse_excluded_keeps_only_parse_ok():
    trials = [{"parse_ok": True, "x": 1}, {"parse_ok": False, "x": 2},
              {"x": 3}, {"parse_ok": True, "x": 4}]
    assert [t["x"] for t in parse_excluded(trials)] == [1, 4]
