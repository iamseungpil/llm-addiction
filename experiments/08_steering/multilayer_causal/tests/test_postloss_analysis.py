"""postloss_analysis: validated seed->state join + Q1 rung-1 verdicts.

Synthetic pool + rollout rows only (no HF / model). Covers: the per-row join
validation (a correct join passes; a shifted offset trips the >2% mismatch
guard), the post-loss labeling (build_prompt display alignment, None for
round-1-style states), and the interaction verdicts (a synthetic dose x
post-loss interaction -> LC_LIKE_INTERACTION; flat -> NO_INTERACTION; thin
cells -> UNDERPOWERED).
"""
import numpy as np
import pytest

from multilayer_causal.src import postloss_analysis as pl

SEED_BASE = 2000042
OFFSET = 3
POOL_N = 40


def _entry(round_idx, prev_win, balance):
    """(game, round_idx) pool entry in the −G catalog schema: decisions carry
    balance_before, history rows carry round/bet/win/balance (the fields
    prompts.build_prompt renders)."""
    decisions = [{"balance_before": 100} for _ in range(round_idx + 1)]
    decisions[round_idx]["balance_before"] = balance
    history = [{"round": r + 1, "bet": 10, "win": True, "balance": 100}
               for r in range(round_idx)]
    history[round_idx - 1]["win"] = prev_win
    return ({"bet_type": "variable", "prompt_combo": "",
             "decisions": decisions, "history": history}, round_idx)


def _pool():
    """40 states, alternating post-win / post-loss, varying round/balance."""
    return [_entry(2 + (k % 5), prev_win=(k % 2 == 0), balance=60 + k)
            for k in range(POOL_N)]


def _rows(pool, arm_id, alpha, n, bet_fn, seed_base=SEED_BASE, offset=OFFSET):
    rows = []
    for i in range(n):
        game, round_idx = pool[(i + offset) % len(pool)]
        loss = pl.label_postloss((game, round_idx))
        bal = int(game["decisions"][round_idx]["balance_before"])
        rows.append({
            "trial_id": i, "seed": seed_base + i * pl.SEED_STEP, "arm": arm_id,
            "alpha": alpha, "parse_ok": True,
            "bet_ratio": bet_fn(alpha, loss, i),
            "source_state": {"prompt_combo": game["prompt_combo"],
                             "round_idx": round_idx, "balance": bal},
        })
    return rows


# ------------------------------------------------------------- labeling

def test_label_postloss_reads_previous_displayed_round():
    assert pl.label_postloss(_entry(3, prev_win=False, balance=80)) is True
    assert pl.label_postloss(_entry(3, prev_win=True, balance=80)) is False
    # result=='W' fallback (the exact build_prompt win-flag read)
    game, ri = _entry(2, prev_win=True, balance=70)
    del game["history"][ri - 1]["win"]
    game["history"][ri - 1]["result"] = "L"
    assert pl.label_postloss((game, ri)) is True
    # no previous displayed round -> None (excluded)
    game2, _ = _entry(2, prev_win=True, balance=70)
    game2["history"] = []
    assert pl.label_postloss((game2, 2)) is None
    assert pl.label_postloss((game2, 0)) is None


# ------------------------------------------------------------------ join

def test_join_correct_mapping_passes():
    pool = _pool()
    rows = _rows(pool, "sec4_behavioural_ap3", 3.0, n=50,
                 bet_fn=lambda a, l, i: 0.5)
    joined = pl.join_trials_to_states(rows, pool, OFFSET, SEED_BASE)
    assert len(joined) == 50
    for row, (game, round_idx) in joined:
        assert row["source_state"]["round_idx"] == round_idx
        assert row["source_state"]["balance"] == int(
            game["decisions"][round_idx]["balance_before"])


def test_join_shifted_offset_trips_guard():
    """A wrong offset must ABORT the join (silent-misalignment guard), not
    quietly relabel the wave with the wrong states."""
    pool = _pool()
    rows = _rows(pool, "sec4_behavioural_ap3", 3.0, n=50,
                 bet_fn=lambda a, l, i: 0.5)
    with pytest.raises(ValueError, match="misaligned"):
        pl.join_trials_to_states(rows, pool, OFFSET + 1, SEED_BASE)


def test_join_wrong_seed_base_trips_guard():
    pool = _pool()
    rows = _rows(pool, "sec4_behavioural_ap3", 3.0, n=50,
                 bet_fn=lambda a, l, i: 0.5)
    with pytest.raises(ValueError, match="misaligned"):
        pl.join_trials_to_states(rows, pool, OFFSET, SEED_BASE + 1)


# -------------------------------------------------------------- analysis

def _arm_rows(pool, bet_fn, n=240, doses=(-3.0, 0.0, 3.0)):
    def _aid(a):
        return ("sec4_behavioural_a0" if a == 0 else
                f"sec4_behavioural_a{'m' if a < 0 else 'p'}{abs(int(a))}")
    return {_aid(a): _rows(pool, _aid(a), a, n, bet_fn) for a in doses}


def test_lc_like_interaction_verdict():
    """Steering raises bets MORE on post-loss states => LC_LIKE_INTERACTION."""
    rng = np.random.default_rng(0)
    pool = _pool()

    def bet(a, loss, i):
        slope = 0.06 if loss else 0.01
        return float(np.clip(0.3 + slope * a + 0.02 * rng.standard_normal(),
                             0, 1))

    res = pl.analyze_postloss(_arm_rows(pool, bet), pool, OFFSET, SEED_BASE)
    assert res["verdict"] == "LC_LIKE_INTERACTION"
    st = res["axes"]["behavioural"]
    assert st["interaction"]["coef"] > 0
    assert st["interaction"]["t"] > pl.INTERACTION_T
    assert st["min_cell_n"] >= pl.MIN_CELL_N
    assert st["spearman"]["postloss"] > st["spearman"]["postwin"]
    # every (dose x condition) cell is populated and counted
    assert len(st["cells"]) == 6
    assert res["join"]["n_joined"] == res["join"]["n_rows"]


def test_no_interaction_verdict():
    """Same slope on both conditions => NO_INTERACTION (steering writes I_BA
    but shows no LC-like post-loss amplification)."""
    rng = np.random.default_rng(1)
    pool = _pool()

    def bet(a, loss, i):
        return float(np.clip(0.3 + 0.04 * a + 0.02 * rng.standard_normal(),
                             0, 1))

    res = pl.analyze_postloss(_arm_rows(pool, bet), pool, OFFSET, SEED_BASE)
    assert res["verdict"] == "NO_INTERACTION"


def test_w1_w2_duplicate_rows_are_deduped():
    """Pseudoreplication guard: the condition-identical W2 behav_iba ladder
    duplicates the W1 behavioural rows (same alphas, same seeds) — pooling
    both must NOT change the interaction t (no sqrt(2) SE deflation)."""
    rng = np.random.default_rng(3)
    pool = _pool()

    def bet(a, loss, i):
        slope = 0.06 if loss else 0.01
        return float(np.clip(0.3 + slope * a + 0.02 * rng.standard_normal(),
                             0, 1))

    w1_only = _arm_rows(pool, bet)
    res_w1 = pl.analyze_postloss(w1_only, pool, OFFSET, SEED_BASE)
    # exact W2 duplicates: same rows under the sec4_w2_behav_iba_ prefix
    both = dict(w1_only)
    for arm_id, rows in w1_only.items():
        w2_id = arm_id.replace("sec4_behavioural_", "sec4_w2_behav_iba_")
        both[w2_id] = [dict(r, arm=w2_id) for r in rows]
    res_both = pl.analyze_postloss(both, pool, OFFSET, SEED_BASE)

    st1 = res_w1["axes"]["behavioural"]["interaction"]
    st2 = res_both["axes"]["behavioural"]["interaction"]
    assert st2["n"] == st1["n"]                       # duplicates dropped
    assert st2["t"] == pytest.approx(st1["t"])        # SE not deflated
    assert st2["coef"] == pytest.approx(st1["coef"])
    assert res_both["join"]["n_dedup"] == st1["n"]
    assert res_both["verdict"] == res_w1["verdict"]


def test_underpowered_verdict():
    pool = _pool()
    res = pl.analyze_postloss(
        _arm_rows(pool, lambda a, l, i: 0.5, n=20), pool, OFFSET, SEED_BASE)
    assert res["verdict"] == "UNDERPOWERED"


def test_outputs_written(tmp_path):
    pool = _pool()
    out_json = tmp_path / "postloss_rung1.json"
    out_png = tmp_path / "postloss_rung1.png"
    res = pl.analyze_postloss(_arm_rows(pool, lambda a, l, i: 0.5), pool,
                              OFFSET, SEED_BASE, out_json, out_png)
    assert out_json.exists()
    assert out_png.exists()
    assert res["params"]["pool_len"] == POOL_N
