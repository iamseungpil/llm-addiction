"""sec4_stats.analyze_wave2 common-axis verdict on synthetic rollouts.

Shared axis moving BOTH I_BA (bet_ratio) and I_EC (extreme) monotonically,
flat per-direction null slopes (estimated from -3/+3 pairs) => verdict
SHARED_COMMON_AXIS. A flat shared axis => NO_SHARED_AXIS.
"""
import json

import numpy as np

from multilayer_causal.src import sec4_stats


DOSES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def _sign(a):
    return "m" if a < 0 else "p"


def _aid(axis, a):
    return f"sec4_w2_{axis}_a{_sign(a)}{abs(int(a))}" if a != 0 \
        else f"sec4_w2_{axis}_a0"


def _write_arm(results_dir, arm_id, alpha, bet_mean, ec_rate, rng, n=30):
    p = results_dir / f"{arm_id}.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            bet = float(np.clip(bet_mean + 0.01 * rng.standard_normal(), 0, 1))
            extreme = bool(rng.random() < ec_rate)
            rec = {"trial_id": i, "seed": i, "arm": arm_id, "alpha": alpha,
                   "parse_ok": True, "bet_ratio": bet, "extreme": extreme,
                   "action": "bet"}
            f.write(json.dumps(rec) + "\n")


def test_shared_common_axis_verdict(tmp_path):
    rng = np.random.default_rng(0)
    results = tmp_path / "results"
    results.mkdir()

    base_bet, ba_slope = 0.30, 0.05
    base_ec, ec_slope = 0.50, 0.13
    for a in DOSES:
        # shared: BOTH indicators move monotonically with dose
        _write_arm(results, _aid("shared", a), a,
                   base_bet + ba_slope * a,
                   float(np.clip(base_ec + ec_slope * a, 0, 1)), rng)
        # per-indicator + confound axes present but flat (verdict ignores them)
        for axis in ("behav_iba", "behav_iec", "confound"):
            _write_arm(results, _aid(axis, a), a, base_bet, base_ec, rng)
    # THICK null: 6 random directions, each at -3 AND +3, flat => ~0 slope
    for k in range(1, 7):
        for a in (-3.0, 3.0):
            aid = f"sec4_w2_null{k}_a{_sign(a)}{abs(int(a))}"
            _write_arm(results, aid, a, base_bet, base_ec, rng)
    _write_arm(results, "sec4_w2_baseline", None, base_bet, base_ec, rng)

    res = sec4_stats.analyze_wave2(results, tmp_path / "assets")

    assert res["verdict"] == "SHARED_COMMON_AXIS", res
    assert res["shared_moves_both"] is True
    sh = res["axes"]["shared"]
    for ind in ("i_ba", "i_ec"):
        assert sh[ind]["monotone"] and sh[ind]["sign_ok"] and sh[ind]["above_null"]
        assert sh[ind]["slope"] > 0
    # thick null band estimated from -3/+3 pairs across 6 directions
    assert res["n_null_directions"] == 6
    for ind in ("i_ba", "i_ec"):
        assert res["null_band"][ind]["n"] == 6
        assert abs(res["null_band"][ind]["mean"]) < 0.01  # flat nulls
    # cross-indicator matrix: shared moves both, confound moves neither much
    cross = res["cross_indicator_slopes"]
    assert cross["shared"]["i_ba"] > 0 and cross["shared"]["i_ec"] > 0
    assert abs(cross["confound"]["i_ba"]) < res["null_band"]["i_ba"]["delta"]


def test_flat_shared_axis_no_common_verdict(tmp_path):
    rng = np.random.default_rng(1)
    results = tmp_path / "results"
    results.mkdir()
    for a in DOSES:
        for axis in ("shared", "behav_iba", "behav_iec", "confound"):
            _write_arm(results, _aid(axis, a), a, 0.30, 0.50, rng)
    for k in range(1, 7):
        for a in (-3.0, 3.0):
            _write_arm(results, f"sec4_w2_null{k}_a{_sign(a)}{abs(int(a))}",
                       a, 0.30, 0.50, rng)
    _write_arm(results, "sec4_w2_baseline", None, 0.30, 0.50, rng)

    res = sec4_stats.analyze_wave2(results, tmp_path / "assets")
    assert res["verdict"] == "NO_SHARED_AXIS"
    assert res["shared_moves_both"] is False
