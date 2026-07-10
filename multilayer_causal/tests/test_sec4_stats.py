"""sec4_stats.analyze verdict logic on synthetic rollouts.

Monotone behavioural axis + flat readout axis + flat confound + flat nulls =>
verdict WRITE_CONFIRMED (H1 behavioural writes, H2 readout inert).
"""
import json

import numpy as np

from multilayer_causal.src import sec4_stats


DOSES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def _write_arm(results_dir, arm_id, alpha, mean_bet, rng, n=20,
               with_proj=False):
    p = results_dir / f"{arm_id}.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            bet = float(np.clip(mean_bet + 0.01 * rng.standard_normal(), 0, 1))
            rec = {"trial_id": i, "seed": i, "arm": arm_id, "alpha": alpha,
                   "parse_ok": True, "bet_ratio": bet, "action": "bet"}
            if with_proj:
                rec["vector_log"] = {"layer": 22, "proj": float(alpha * 2.0),
                                     "h_norm": 5.0}
            f.write(json.dumps(rec) + "\n")


def _write_assets(assets_dir):
    dirs = np.zeros((42, 8), np.float32)
    dirs[:, 0] = 1.0
    r = dirs.copy()
    b = np.zeros((42, 8), np.float32)
    b[:, 1] = 1.0  # orthogonal to readout => cos_read_write ~ 0
    np.savez(assets_dir / "gemma_slot_machine_i_ba_readout.npz",
             directions=r, auc=0.72, cos_read_write=0.0)
    np.savez(assets_dir / "gemma_slot_machine_i_ba_behavioural.npz",
             directions=b, auc=float("nan"), cos_read_write=0.0)
    np.savez(assets_dir / "gemma_slot_machine_i_ba_confound.npz",
             directions=dirs, auc=float("nan"), cos_read_write=0.0)


def test_write_confirmed_verdict(tmp_path):
    rng = np.random.default_rng(0)
    results = tmp_path / "results"
    assets = tmp_path / "assets"
    results.mkdir(); assets.mkdir()
    _write_assets(assets)
    (assets / "anchors.json").write_text(json.dumps({"minus": 0.15, "plus": 0.45}))

    base = 0.30
    slope = 0.05
    for a in DOSES:  # behavioural: monotone dose-response, sign-correct
        sign = "m" if a < 0 else "p"
        aid = f"sec4_behavioural_a{sign}{abs(int(a))}" if a != 0 else "sec4_behavioural_a0"
        _write_arm(results, aid, a, base + slope * a, rng, with_proj=True)
    for a in DOSES:  # readout: flat (inert)
        sign = "m" if a < 0 else "p"
        aid = f"sec4_readout_a{sign}{abs(int(a))}" if a != 0 else "sec4_readout_a0"
        _write_arm(results, aid, a, base, rng, with_proj=True)
    for a in DOSES:  # confound: flat (not a confound)
        sign = "m" if a < 0 else "p"
        aid = f"sec4_confound_a{sign}{abs(int(a))}" if a != 0 else "sec4_confound_a0"
        _write_arm(results, aid, a, base, rng)
    for k in range(1, 6):  # random nulls at baseline
        _write_arm(results, f"sec4_null_{k}", 3.0, base, rng)
    _write_arm(results, "sec4_baseline", None, base, rng)

    res = sec4_stats.analyze(results, assets)

    assert res["verdict"] == "WRITE_CONFIRMED", res["verdict"]
    assert res["h1_behavioural_writes"] is True
    assert res["h2_readout_inert"] is True
    beh = res["axes"]["behavioural"]
    assert beh["monotone"] and beh["sign_ok"] and beh["above_null"]
    assert beh["spearman"] > 0.7
    read = res["axes"]["readout"]
    assert not sec4_stats._writes(read)
    # manip-check projection was logged and must be finite (assertion path)
    assert np.isfinite(res["null_band"]["delta"])
    # decoding-AUC table surfaced the readout's stronger decoder
    assert res["decoding_auc"]["gemma_slot_machine_i_ba_readout"] == 0.72


def test_null_verdict_when_behavioural_flat(tmp_path):
    """Flat behavioural + flat readout => nothing writes => not WRITE_CONFIRMED."""
    rng = np.random.default_rng(1)
    results = tmp_path / "results"
    assets = tmp_path / "assets"
    results.mkdir(); assets.mkdir()
    _write_assets(assets)
    for a in DOSES:
        sign = "m" if a < 0 else "p"
        for axis in ("behavioural", "readout", "confound"):
            aid = (f"sec4_{axis}_a{sign}{abs(int(a))}" if a != 0
                   else f"sec4_{axis}_a0")
            _write_arm(results, aid, a, 0.30, rng)
    for k in range(1, 6):
        _write_arm(results, f"sec4_null_{k}", 3.0, 0.30, rng)
    _write_arm(results, "sec4_baseline", None, 0.30, rng)

    res = sec4_stats.analyze(results, assets)
    assert res["verdict"] != "WRITE_CONFIRMED"
    assert res["h1_behavioural_writes"] is False
