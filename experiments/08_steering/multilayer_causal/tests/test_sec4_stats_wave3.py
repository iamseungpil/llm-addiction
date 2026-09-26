"""sec4_stats.analyze_wave3 adjudication verdicts on synthetic rollouts.

Q2 branches: SHARED_TASK_CONTROL (shared3 moves SM bets AND IC risky rate
with the IC positive control alive), TASK_SPECIFIC (IC control alive but
shared3 flat on IC), IC_LEVER_ABSENT (even IC's own axis is flat). Q3 and
Q1-rung2 slope-contrast verdicts are asserted alongside.
"""
import json

import numpy as np

from multilayer_causal.src import sec4_stats

DOSES7 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
DOSES3 = [-3.0, 0.0, 3.0]


def _suffix(a):
    return "a0" if a == 0 else f"a{'m' if a < 0 else 'p'}{abs(int(a))}"


def _write_sm(d, arm_id, alpha, bet_mean, rng, n=60):
    with open(d / f"{arm_id}.jsonl", "w") as f:
        for i in range(n):
            rec = {"trial_id": i, "seed": i, "arm": arm_id, "alpha": alpha,
                   "parse_ok": True, "action": "bet",
                   "bet_ratio": float(np.clip(
                       bet_mean + 0.01 * rng.standard_normal(), 0, 1))}
            f.write(json.dumps(rec) + "\n")


def _write_ic(d, arm_id, alpha, p_risky, rng, n=120):
    with open(d / f"{arm_id}.jsonl", "w") as f:
        for i in range(n):
            rec = {"trial_id": i, "seed": i, "arm": arm_id, "alpha": alpha,
                   "task": "ic", "parse_ok": True,
                   "risky": bool(rng.random() < p_risky), "choice": 3}
            f.write(json.dumps(rec) + "\n")


def _p(base, slope, a):
    return float(np.clip(base + slope * a, 0.02, 0.98))


def _populate(results, rng, sh3_ic_slope, ic_own_slope,
              plusG_slope=0.06, w1_slope=0.02,
              postloss_slope=0.07, postwin_slope=0.02):
    """Write a full synthetic sec4_w3 results dir; slopes parameterize the
    verdict branches."""
    for a in DOSES7:
        _write_sm(results, f"sec4_w3_sh3_sm_{_suffix(a)}", a, 0.3 + 0.05 * a, rng)
        _write_ic(results, f"sec4_w3_sh3_ic_{_suffix(a)}", a,
                  _p(0.4, sh3_ic_slope, a), rng)
        _write_sm(results, f"sec4_w3_plusG_{_suffix(a)}", a,
                  0.4 + plusG_slope * a, rng)
    for a in DOSES3:
        _write_ic(results, f"sec4_w3_ic_own_{_suffix(a)}", a,
                  _p(0.4, ic_own_slope, a), rng)
        _write_sm(results, f"sec4_w3_postloss_{_suffix(a)}", a,
                  0.3 + postloss_slope * a, rng)
        _write_sm(results, f"sec4_w3_postwin_{_suffix(a)}", a,
                  0.3 + postwin_slope * a, rng)
    _write_ic(results, "sec4_w3_ic_baseline", None, 0.4, rng)
    for k in (1, 2, 3):  # flat IC nulls, each direction at -3 AND +3
        for a in (-3.0, 3.0):
            _write_ic(results, f"sec4_w3_ic_null{k}_{_suffix(a)}", a, 0.4, rng)


def _w1_ladder(d, rng, slope=0.02):
    for a in DOSES7:
        _write_sm(d, f"sec4_behavioural_{_suffix(a)}", a, 0.4 + slope * a, rng)


def test_shared_task_control_verdict(tmp_path):
    rng = np.random.default_rng(0)
    results, w1 = tmp_path / "w3", tmp_path / "w1"
    results.mkdir(), w1.mkdir()
    _populate(results, rng, sh3_ic_slope=0.09, ic_own_slope=0.09)
    _w1_ladder(w1, rng)

    res = sec4_stats.analyze_wave3(results, w1, tmp_path / "no_w2",
                                   tmp_path / "out.json", tmp_path / "out.png")
    assert res["q2"]["verdict"] == "SHARED_TASK_CONTROL"
    for key in ("sh3_sm", "sh3_ic", "ic_own"):
        st = res["q2"][key]
        assert st["monotone"] and st["sign_ok"] and st["above_null"], (key, st)
    assert res["q2"]["ic_null_band"]["n"] == 3
    # no Wave-2 rollouts present => the weaker slope-floor gate ran, and the
    # output says so explicitly (never a silent downgrade)
    assert res["q2"]["sm_null_source"] == "floor_fallback"
    assert res["q2"]["sm_null_band"]["n"] == 0
    # Q3: +G slope 0.06 vs Wave-1 -G slope 0.02 => CI excludes 0
    assert res["q3"]["verdict"] == "CONDITION_MODULATES"
    assert res["q3"]["slope_plusG"] > res["q3"]["slope_minusG_w1"]
    assert res["q3"]["ci95"][0] > 0
    # Q1 rung-2: post-loss slope 0.07 vs post-win 0.02
    assert res["q1_rung2"]["verdict"] == "POSTLOSS_STEER_AMPLIFIED"
    assert res["q1_rung2"]["ci95"][0] > 0
    assert (tmp_path / "out.json").exists() and (tmp_path / "out.png").exists()


def test_task_specific_verdict(tmp_path):
    """IC's own axis moves risky choice but shared3 does NOT => the shared
    component's control is task-specific (a legitimate terminal result)."""
    rng = np.random.default_rng(1)
    results, w1 = tmp_path / "w3", tmp_path / "w1"
    results.mkdir(), w1.mkdir()
    _populate(results, rng, sh3_ic_slope=0.0, ic_own_slope=0.09,
              plusG_slope=0.02, postloss_slope=0.02)  # Q3/Q1 contrasts flat too
    _w1_ladder(w1, rng)

    res = sec4_stats.analyze_wave3(results, w1, tmp_path / "no_w2")
    assert res["q2"]["verdict"] == "TASK_SPECIFIC"
    assert res["q2"]["ic_own"]["above_null"]
    assert not res["q2"]["sh3_ic"]["above_null"]
    assert res["q3"]["verdict"] == "NO_MODULATION"
    assert res["q1_rung2"]["verdict"] == "NO_POSTLOSS_MODULATION"


def test_sm_null_source_thick_when_w2_nulls_present(tmp_path):
    """With the Wave-2 null rollouts on disk the Q2 SM gate uses the THICK
    per-direction slope band (n = number of null directions), and the output
    records sm_null_source='thick'."""
    rng = np.random.default_rng(3)
    results, w1, w2 = tmp_path / "w3", tmp_path / "w1", tmp_path / "w2"
    results.mkdir(), w1.mkdir(), w2.mkdir()
    _populate(results, rng, sh3_ic_slope=0.09, ic_own_slope=0.09)
    _w1_ladder(w1, rng)
    for k in (1, 2, 3):  # flat Wave-2 SM nulls, each direction at -3 AND +3
        for a in (-3.0, 3.0):
            _write_sm(w2, f"sec4_w2_null{k}_{_suffix(a)}", a, 0.4, rng)

    res = sec4_stats.analyze_wave3(results, w1, w2)
    assert res["q2"]["sm_null_source"] == "thick"
    assert res["q2"]["sm_null_band"]["n"] == 3


def test_ic_lever_absent_verdict(tmp_path):
    """Even the IC positive control is flat => the Q2 rung is unanswerable on
    IC (IC_LEVER_ABSENT), regardless of what shared3 does on SM."""
    rng = np.random.default_rng(2)
    results, w1 = tmp_path / "w3", tmp_path / "w1"
    results.mkdir(), w1.mkdir()
    _populate(results, rng, sh3_ic_slope=0.0, ic_own_slope=0.0)
    _w1_ladder(w1, rng)

    res = sec4_stats.analyze_wave3(results, w1, tmp_path / "no_w2")
    assert res["q2"]["verdict"] == "IC_LEVER_ABSENT"
    assert not res["q2"]["ic_own"]["above_null"]
