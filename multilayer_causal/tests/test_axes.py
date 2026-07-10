"""CPU-only unit tests for axes.py — pure math + pool reconstruction.

No downloads, no phase_a: the real builders are exercised only on synthetic
arrays through the factored pure functions.
"""
import json

import numpy as np
import pytest

from multilayer_causal.src import axes, states


# ---------------------------------------------------------------- pure math

def test_tercile_contrast_exact_value():
    values = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    X = values[:, None]  # 1-D feature equal to the ranking value
    # cuts at quantile(1/3)=5/3, quantile(2/3)=10/3 -> top {4,5}, bot {0,1}
    delta = axes.tercile_contrast(X, values)
    assert np.allclose(delta, [4.0])


def test_tercile_contrast_recovers_planted_direction():
    rng = np.random.default_rng(0)
    v = np.array([1.0, -2.0, 0.5, 3.0])
    v /= np.linalg.norm(v)
    values = rng.uniform(0, 1, 600)
    X = values[:, None] * v + 0.01 * rng.standard_normal((600, 4))
    delta = axes.tercile_contrast(X, values)
    cos = delta @ v / np.linalg.norm(delta)
    assert cos > 0.99


def test_binary_contrast_exact():
    X = np.array([[1.0, 0.0], [3.0, 2.0], [0.0, -1.0], [2.0, 1.0]])
    delta = axes.binary_contrast(X, [True, True, False, False])
    assert np.allclose(delta, [1.0, 1.0])
    with pytest.raises(AssertionError):
        axes.binary_contrast(X, [True, True, True, True])


def test_within_group_contrast_ignores_small_and_offset_groups():
    rng = np.random.default_rng(1)
    v = np.array([0.0, 1.0, 0.0])
    n = 40  # per class per qualifying group
    blocks, labels, groups = [], [], []
    for g, offset in [("A", 0.0), ("B", 100.0)]:  # big between-group shift
        for lab in (True, False):
            blocks.append(offset + (v if lab else 0.0)
                          + 0.01 * rng.standard_normal((n, 3)))
            labels += [lab] * n
            groups += [g] * n
    # tiny group with a HUGE opposite contrast — must be ignored (n < min_n)
    blocks.append(-50.0 * v + np.zeros((3, 3)))
    labels += [True] * 3
    groups += ["tiny"] * 3
    X = np.vstack(blocks)
    delta, n_groups = axes.within_group_contrast(X, labels, groups, min_n=20)
    assert n_groups == 2
    assert np.allclose(delta, v, atol=0.02)
    assert axes.qualifying_groups(labels, groups, min_n=20) == ["A", "B"]


def test_within_group_contrast_asserts_when_no_group_qualifies():
    X = np.zeros((4, 2))
    with pytest.raises(AssertionError):
        axes.within_group_contrast(X, [True, False, True, False],
                                   ["a", "a", "b", "b"], min_n=20)


def test_ridge_axis_recovers_planted_direction_despite_feature_scales():
    rng = np.random.default_rng(2)
    sd = np.array([0.1, 1.0, 10.0, 3.0, 0.5])
    v = np.array([1.0, -1.0, 0.5, 0.0, 2.0])
    v /= np.linalg.norm(v)
    X = rng.standard_normal((2000, 5)) * sd  # independent, anisotropic
    y = X @ v
    d = axes.ridge_axis(X, y, alpha=100.0)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5)
    # the /scaler.scale_ map must undo standardization: cos(d, v) ~ 1
    assert abs(d @ v) > 0.999 and d @ v > 0


# --------------------------------------------------------------- ILC fields

def test_ilc_fields_for_game_alignment():
    game = {
        "decisions": [
            {"action": "bet", "bet": 10, "balance_before": 100},
            {"action": "bet", "bet": 20, "balance_before": 90},
            {"action": "skip"},
            {"action": "bet", "bet": 5, "balance_before": 110},
            {"action": "stop", "balance_before": 105},
        ],
        "history": [
            {"round": 1, "bet": 10, "result": "L", "balance": 90, "win": False},
            {"round": 2, "bet": 20, "result": "W", "balance": 110, "win": True},
            {"round": 3, "bet": 5, "result": "L", "balance": 105, "win": False},
        ],
    }
    out = axes.ilc_fields_for_game(game)
    # 4 valid rows (skip dropped): (prev_loss, r_prev)
    assert out == [(False, 0.0), (True, 10 / 100), (False, 20 / 90),
                   (True, 5 / 110)]


def test_ilc_fields_win_fallback_to_result_field():
    game = {"decisions": [{"action": "bet", "bet": 10, "balance_before": 100},
                          {"action": "bet", "bet": 10, "balance_before": 90}],
            "history": [{"round": 1, "bet": 10, "result": "L", "balance": 90}]}
    assert axes.ilc_fields_for_game(game)[1][0] is True  # 'L' -> loss


# ---------------------------------------------------------------- IC labels

def test_load_ic_choices_filters_and_orders(tmp_path):
    def dec(choice, skipped=False, prompt="p"):
        return {"choice": choice, "skipped": skipped, "full_prompt": prompt}
    f1 = tmp_path / "c10.json"
    f2 = tmp_path / "c30.json"
    f1.write_text(json.dumps({"results": [
        {"decisions": [dec(1), dec(2, skipped=True), dec(3)]},
        {"decisions": [dec(4, prompt=""), dec(None)]},
    ]}))
    f2.write_text(json.dumps({"results": [{"decisions": [dec(2)]}]}))
    assert axes.load_ic_choices([str(f1), str(f2)]) == [1, 3, None, 2]
    # file order matters — reversed input, reversed rows
    assert axes.load_ic_choices([str(f2), str(f1)]) == [2, 1, 3, None]


# ---------------------------------------- eval-pool reconstruction (frozen)

def _write_catalog(tmp_path, monkeypatch):
    """PROVENANCE: same synthetic catalog fixture as tests/test_states.py."""
    d = tmp_path / "slot_machine" / "gemma_v4_role"
    d.mkdir(parents=True)
    games = []
    for combo in ["", "G", "M"]:
        for bt in ["variable", "fixed"]:
            games.append({"bet_type": bt, "prompt_combo": combo,
                          "decisions": [{"balance_before": 100 - 5 * r} for r in range(8)],
                          "history": []})
    path = d / "final_gemma_x.json"
    path.write_text(json.dumps({"results": games}))
    monkeypatch.setenv("LLM_ADDICTION_BEHAVIORAL_ROOT", str(tmp_path))
    return path, games


def test_pool_with_game_ids_matches_load_minusG_states(tmp_path, monkeypatch):
    catalog, games = _write_catalog(tmp_path, monkeypatch)
    pool = axes.minusG_pool_with_game_ids(catalog)
    ref = states.load_minusG_states("gemma", n=300)
    assert len(pool) == len(ref) == 10
    # identical round_idx sequence => identical shuffle permutation
    assert [ri for _, ri in pool] == [ri for _, ri in ref]
    # game_id is the 1-based catalog index: the referenced game must be the
    # exact dict load_minusG_states returned at every position
    for (gid, _), (game, _) in zip(pool, ref):
        assert games[gid - 1] is not game  # ref parsed its own copy...
        assert games[gid - 1]["prompt_combo"] == game["prompt_combo"]
        assert games[gid - 1]["bet_type"] == game["bet_type"]
    # only variable, no-G games may appear (catalog order: "" variable is
    # game 1, "M" variable is game 5)
    assert {gid for gid, _ in pool} == {1, 5}


def test_excluded_game_ids_takes_first_n(tmp_path, monkeypatch):
    catalog, _ = _write_catalog(tmp_path, monkeypatch)
    pool = axes.minusG_pool_with_game_ids(catalog)
    assert axes.excluded_game_ids(pool, n_eval=1) == {pool[0][0]}
    assert axes.excluded_game_ids(pool, n_eval=300) == {1, 5}
