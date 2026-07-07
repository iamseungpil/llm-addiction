"""sec4_stats.analyze_wave6 verdicts on synthetic MW rollouts.

The Wave-6 metric contract (from runner._run_arm_mw's record fields): spin
rate = mean(`risky`) over parse_ok rows, bet = mean(`bet_ratio`) over parse_ok
rows (stop rows contribute 0.0). Nulls/baseline are the REUSED sec4_w4 MW
rollouts. Branches: MW_RC_CORRECTLY_SIGNED (positive monotone spin slope
above the null band), STILL_ANOMALOUS (clears the band but wrong-signed —
the W3/W4 anomaly signature), UNDERPOWERED (nothing clears the band).
Deterministic spin counts (round(n*p)) keep the branch assertions exact.
"""
import json

import numpy as np

from multilayer_causal.src import sec4_stats

DOSES7 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def _suffix(a):
    return "a0" if a == 0 else f"a{'m' if a < 0 else 'p'}{abs(int(a))}"


def _write_mw(d, arm_id, alpha, p_spin, n=200, n_bad=5):
    """Synthetic _run_arm_mw rollout: exactly round(n*p_spin) parse_ok spins
    (bet_ratio 0.3) then parse_ok stops (bet_ratio 0.0), plus n_bad defaulted
    parse-failure stops that the parse gate must exclude."""
    k = int(round(n * p_spin))
    with open(d / f"{arm_id}.jsonl", "w") as f:
        for i in range(n):
            spin = i < k
            rec = {"trial_id": i, "seed": i, "arm": arm_id, "task": "mw",
                   "alpha": alpha, "parse_ok": True,
                   "action": "spin" if spin else "stop",
                   "risky": bool(spin), "bet_ratio": 0.3 if spin else 0.0}
            f.write(json.dumps(rec) + "\n")
        for i in range(n_bad):  # parser default-stop rows (parse_ok False)
            rec = {"trial_id": n + i, "seed": n + i, "arm": arm_id,
                   "task": "mw", "alpha": alpha, "parse_ok": False,
                   "action": "stop", "risky": False, "bet_ratio": 0.0}
            f.write(json.dumps(rec) + "\n")


def _p(base, slope, a):
    return float(np.clip(base + slope * a, 0.02, 0.98))


def _populate(w6_dir, w4_dir, spin_slope, n_bad=5):
    for a in DOSES7:
        _write_mw(w6_dir, f"sec4_w6_mw_rc_{_suffix(a)}", a,
                  _p(0.4, spin_slope, a), n_bad=n_bad)
    for k in (1, 2, 3):  # flat reused W4 MW nulls, each direction at -3 AND +3
        for a in (-3.0, 3.0):
            _write_mw(w4_dir, f"sec4_w4_mw_null{k}_{_suffix(a)}", a, 0.4,
                      n_bad=n_bad)
    _write_mw(w4_dir, "sec4_w4_mw_baseline", None, 0.4, n_bad=n_bad)


def test_mw_rc_correctly_signed(tmp_path):
    w6, w4 = tmp_path / "w6", tmp_path / "w4"
    w6.mkdir(), w4.mkdir()
    _populate(w6, w4, spin_slope=0.08)

    res = sec4_stats.analyze_wave6(w6, w4, tmp_path / "out.json",
                                   tmp_path / "out.png")
    assert res["verdict"] == "MW_RC_CORRECTLY_SIGNED"
    st = res["spin"]
    assert st["monotone"] and st["sign_ok"] and st["above_null"], st
    assert abs(st["slope"] - 0.08) < 0.01
    # bet_ratio co-moves (spin implies bet 0.3): independent corroboration
    assert res["bet_corroborates"]
    assert abs(res["bet"]["slope"] - 0.08 * 0.3) < 0.01
    assert res["n_null_directions"] == 3
    assert res["null_band"]["spin"]["n"] == 3
    # reused W4 baseline is surfaced for both metrics
    assert abs(res["baseline"]["spin"] - 0.4) < 1e-9
    assert abs(res["baseline"]["bet"] - 0.4 * 0.3) < 1e-9
    # parse gate: the 5 defaulted stops per arm never entered the metric
    # (spin rate at a0 is exactly 0.4, not 80/205)
    assert abs(float(st["dose_means"]["0.0"]) - 0.4) < 1e-9
    assert (tmp_path / "out.json").exists() and (tmp_path / "out.png").exists()


def test_mw_rc_still_anomalous(tmp_path):
    """Negative slope that clears the null band = the W3/W4 MW anomaly
    signature persisting on the new object."""
    w6, w4 = tmp_path / "w6", tmp_path / "w4"
    w6.mkdir(), w4.mkdir()
    _populate(w6, w4, spin_slope=-0.08)

    res = sec4_stats.analyze_wave6(w6, w4)
    assert res["verdict"] == "STILL_ANOMALOUS"
    assert res["spin"]["above_null"] and not res["spin"]["sign_ok"]
    assert not res["bet_corroborates"]


def test_mw_rc_underpowered(tmp_path):
    """Flat ladder: nothing clears the null band."""
    w6, w4 = tmp_path / "w6", tmp_path / "w4"
    w6.mkdir(), w4.mkdir()
    _populate(w6, w4, spin_slope=0.0)

    res = sec4_stats.analyze_wave6(w6, w4)
    assert res["verdict"] == "UNDERPOWERED"
    assert not res["spin"]["above_null"]


# --------------------------------------------------- realistic parse rates
# Real MW rollouts parse at 0.19-0.73 (frozen sec4_w4 set: baseline 0.650,
# nulls 0.455-0.730). The W2 PARSE_GATE=0.8 would discard every such cell —
# ladder, nulls AND baseline — collapsing the band to the slope floor (n=0)
# and forcing UNDERPOWERED regardless of the true effect. These tests pin
# the MW-calibrated W6_PARSE_GATE (0.45) at an empirically realistic parse
# rate: 200 parse_ok rows + 105 defaulted stops = 200/305 ~ 0.656.

N_BAD_REAL = 105  # parse rate 200/305 ~ 0.656 (the W4 MW baseline regime)


def test_realistic_parse_correctly_signed(tmp_path):
    """A TRUE +0.08 spin slope at ~0.65 parse must be detected, not
    swallowed by the W2 gate (the pre-fix behaviour: slope NaN, band n=0,
    verdict UNDERPOWERED)."""
    w6, w4 = tmp_path / "w6", tmp_path / "w4"
    w6.mkdir(), w4.mkdir()
    _populate(w6, w4, spin_slope=0.08, n_bad=N_BAD_REAL)

    res = sec4_stats.analyze_wave6(w6, w4)
    assert res["verdict"] == "MW_RC_CORRECTLY_SIGNED"
    assert abs(res["spin"]["slope"] - 0.08) < 0.01
    assert len(res["spin"]["dose_means"]) == len(DOSES7)
    # the recorded gate is the MW-calibrated one, below the observed parse
    assert res["parse_gate"] == sec4_stats.W6_PARSE_GATE == 0.45
    assert all(v >= res["parse_gate"] for v in res["parse_by_dose"].values())
    # defaulted stops still excluded from the metric itself
    assert abs(float(res["spin"]["dose_means"]["0.0"]) - 0.4) < 1e-9


def test_realistic_parse_still_anomalous(tmp_path):
    w6, w4 = tmp_path / "w6", tmp_path / "w4"
    w6.mkdir(), w4.mkdir()
    _populate(w6, w4, spin_slope=-0.08, n_bad=N_BAD_REAL)

    res = sec4_stats.analyze_wave6(w6, w4)
    assert res["verdict"] == "STILL_ANOMALOUS"
    assert res["spin"]["above_null"] and not res["spin"]["sign_ok"]


def test_realistic_parse_underpowered_with_live_band(tmp_path):
    """Flat ladder at realistic parse: UNDERPOWERED must come from a LIVE
    null band (all 6 null cells kept), not from an empty floor-only band."""
    w6, w4 = tmp_path / "w6", tmp_path / "w4"
    w6.mkdir(), w4.mkdir()
    _populate(w6, w4, spin_slope=0.0, n_bad=N_BAD_REAL)

    res = sec4_stats.analyze_wave6(w6, w4)
    assert res["verdict"] == "UNDERPOWERED"
    assert res["n_null_slopes"] == {"spin": 3, "bet": 3}


def test_band_survives_half_parse_nulls(tmp_path):
    """Regression for the parse-collapse bug: nulls at exactly 0.5 parse
    (200 ok / 200 bad) still contribute all 3 slopes to the band, and the
    effective count is surfaced next to the on-disk direction count."""
    w6, w4 = tmp_path / "w6", tmp_path / "w4"
    w6.mkdir(), w4.mkdir()
    _populate(w6, w4, spin_slope=0.08, n_bad=200)  # parse 200/400 = 0.5

    res = sec4_stats.analyze_wave6(w6, w4)
    assert res["n_null_directions"] == 3
    assert res["n_null_slopes"]["spin"] == 3 > 0
    assert res["null_band"]["spin"]["n"] == 3
    assert res["verdict"] == "MW_RC_CORRECTLY_SIGNED"
