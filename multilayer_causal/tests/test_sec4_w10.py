"""W10 — LLaMA §4.1 + §4.3 model symmetry: readout builder, config/template
parity, analyzer verdicts + null-cause, gemma non-regression.

SYNTHETIC only (no torch / no HF / no real model). Covers:
  * the LlamaScope decoder loader's pure transpose/sanity core
    (wdec_from_decoder_weight) and the per-layer llama readout orchestrator
    (build_llama_readout_from_arrays) on synthetic arrays;
  * the readout array core STILL raises on a None decoder (non-regression);
  * arms_sec4_w10a/b.yaml registry counts/fields + amlt/sec4_w10.yaml.template
    consistency (both run_arms.sh lines, escaped need-list quotes, isolation);
  * analyze_w10 §4.1 dissociation verdict + §4.3 modulation, and the NULL-CAUSE
    diagnosis (behavioural null too -> window/object; readout-only null -> win);
  * gemma module geometry unchanged (42/3584), gemma decoder path untouched.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from multilayer_causal.src import indicator_axes as ia
from multilayer_causal.src import paper_axes as pa
from multilayer_causal.src import sec4_stats
from multilayer_causal.src.registry import load_arms

MLC = Path(__file__).resolve().parents[1]
W10A_YAML = MLC / "configs" / "arms_sec4_w10a.yaml"
W10B_YAML = MLC / "configs" / "arms_sec4_w10b.yaml"
W10_TEMPLATE = MLC / "amlt" / "sec4_w10.yaml.template"
W10_NPZ = ("llama_slot_machine_i_ba_readout.npz",
           "llama_slot_machine_i_ba_behavioural.npz",
           "llama_slot_machine_i_ba_confound.npz",
           "llama_slot_machine_i_ec_behavioural.npz",
           "llama_slot_machine_i_lc_behavioural.npz")


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


# -------------------------------------------------- LlamaScope decoder core

def test_wdec_from_decoder_weight_transpose_and_sanity():
    """fnlp decoder.weight (d_model, d_sae) -> (n_feat, d_model) per-feature
    rows (== gemma orientation), finite, non-degenerate. Wrong orientation
    (d_model >= d_sae) is refused LOUD."""
    rng = np.random.default_rng(0)
    d_model, d_sae = 16, 64          # scaled-down (4096, 32768)
    W = rng.standard_normal((d_model, d_sae))
    Wdec = pa.wdec_from_decoder_weight(W)
    assert Wdec.shape == (d_sae, d_model)
    assert np.allclose(Wdec, W.T)
    # wrong orientation (already transposed) must be rejected
    with pytest.raises(AssertionError):
        pa.wdec_from_decoder_weight(W.T)
    # degenerate (all-zero) decoder rejected by the row-norm gate
    with pytest.raises(AssertionError):
        pa.wdec_from_decoder_weight(np.zeros((d_model, d_sae)))


# ----------------------------------------------- per-layer llama readout

def _readout_layer_arrays(rs, n=200, k=48, d=8):
    """Synthetic (feats, decoder) for ONE readout layer: feature 0 tracks the
    residual signal, decoder row 0 points along u so the mapped readout aligns
    with u."""
    u = _unit(rs.randn(d))
    signal = rs.randn(n)
    feats = rs.randn(n, k)
    feats[:, 0] = signal + 0.05 * rs.randn(n)      # informative feature
    decoder = rs.randn(k, d) * 0.1
    decoder[0] = u
    return feats, decoder, u, signal


def test_build_llama_readout_perlayer_shape_auc():
    """The per-layer llama readout orchestrator builds ONE readout per window
    layer from that layer's own feats+decoder and stacks them: (L,d) unit rows,
    a finite mean AUC, and a per-layer AUC vector."""
    rs = np.random.RandomState(1)
    layers = [14, 15, 16]
    feats_by_layer, decoder_by_layer, us, sig = [], [], [], None
    balance = rs.rand(200)
    rounds = rs.randint(1, 20, 200).astype(float)
    groups = np.arange(200) // 4
    for _ in layers:
        f, dec, u, signal = _readout_layer_arrays(rs)
        feats_by_layer.append(f)
        decoder_by_layer.append(dec)
        us.append(u)
        sig = signal if sig is None else sig
    # indicator carries the shared signal of the LAST layer's build (any signal
    # works; the test asserts shape/auc, not cross-layer identity)
    indicator = sig
    out = ia.build_llama_readout_from_arrays(
        feats_by_layer, indicator, balance, rounds, groups,
        decoder_by_layer, layers)
    assert out["directions"].shape == (3, 8)
    assert np.allclose(np.linalg.norm(out["directions"], axis=1), 1.0, atol=1e-5)
    assert out["aucs_by_layer"].shape == (3,)
    assert "auc" in out and "provenance" in out
    assert out["scales"].shape == (3,)


def test_readout_core_still_guards_none_decoder():
    """Non-regression: the array core STILL raises NotImplementedError on a
    None decoder (W8/W9 behavioural path relies on this)."""
    with pytest.raises(NotImplementedError):
        ia.build_readout_axis_from_arrays(
            feats=np.random.RandomState(0).randn(60, 8),
            indicator=np.random.RandomState(1).randn(60),
            balance=np.random.RandomState(2).rand(60),
            rounds=np.random.RandomState(3).randint(1, 9, 60),
            groups=np.arange(60) // 4,
            decoder=None, layers=[14])


# ---------------------------------------------------- gemma non-regression

def test_gemma_geometry_and_decoder_path_untouched():
    assert pa.MODEL_DIMS["gemma"] == (42, 3584)
    assert pa.MODEL_DIMS["llama"] == (32, 4096)
    assert (ia.N_LAYERS, ia.D_MODEL) == (42, 3584)
    # gemma decoder loader + its reconstruction-check constant are unchanged
    assert hasattr(pa, "load_gemmascope_l22_wdec")
    assert pa.DECODER_CHECK_MIN_COS == 0.999
    # llama decoder constants added additively
    assert pa.LLAMASCOPE_REPO == "fnlp/Llama3_1-8B-Base-LXR-8x"
    assert pa.LLAMASCOPE_DSAE == 32768 and pa.LLAMASCOPE_DMODEL == 4096


# --------------------------------------------------- w10a/w10b registry

def test_w10a_registry_counts_and_fields():
    arms = load_arms(W10A_YAML)
    assert len(arms) == 36
    for a in arms.values():
        assert a["model"] == "llama"
        assert a["n"] == 200
        assert a["seed_base"] == 6000042
        assert a["state_offset"] == 0
        assert a["prompt_set"] == "addiction_role_gm"
        assert a["phase"] == "sec4_w10"
        assert a.get("task", "sm") == "sm"
        assert max(a["layers"]) <= 31         # llama bound
    # three i_ba axis ladders x 7 doses
    for axis, npz in (("readout", "i_ba_readout"),
                      ("behavioural", "i_ba_behavioural"),
                      ("confound", "i_ba_confound")):
        cell = [k for k in arms if k.startswith(f"sec4_w10a_{axis}_a")]
        assert len(cell) == 7, (axis, len(cell))
        for k in cell:
            assert arms[k]["directions_npz"].endswith(f"llama_slot_machine_{npz}.npz")
            assert arms[k]["layers"] == list(range(14, 20))
    # 5 random nulls at +3 on the behavioural npz, unique seeds
    nulls = [a for a in arms.values() if a.get("direction") == "random"]
    assert len(nulls) == 5
    assert all(a["alpha"] == 3.0 and a["log_vectors"] is False for a in nulls)
    assert len({a["dir_seed"] for a in nulls}) == 5
    # one no-steer baseline
    bases = [a for a in arms.values() if a["mode"] == "anchor_minus"]
    assert len(bases) == 1
    # 3 cumulative windows (H4 locality) on the behavioural npz
    cum = [k for k in arms if k.startswith("sec4_w10a_cum_")]
    assert len(cum) == 3
    assert {tuple(arms[k]["layers"]) for k in cum} == {
        tuple(range(15, 19)), tuple(range(14, 20)), tuple(range(12, 22))}
    # specificity: i_ec + i_lc behavioural, 3 doses each
    for spec, npz in (("iec", "i_ec"), ("ilc", "i_lc")):
        s = [k for k in arms if k.startswith(f"sec4_w10a_{spec}_a")]
        assert len(s) == 3
        assert all(arms[k]["directions_npz"].endswith(
            f"llama_slot_machine_{npz}_behavioural.npz") for k in s)


def test_w10b_registry_counts_and_fields():
    arms = load_arms(W10B_YAML)
    assert len(arms) == 10
    for a in arms.values():
        assert a["model"] == "llama" and a["n"] == 200
        assert a["seed_base"] == 6100042      # RNG-independent of w10a
        assert a["phase"] == "sec4_w10"
        assert a["task"] == "sm" and a["layers"] == list(range(14, 20))
    # three conditions x 3 doses on the SAME behavioural axis
    for cond, twin in (("minusG", None), ("plusG", "G"), ("plusM", "M")):
        c = [k for k in arms if k.startswith(f"sec4_w10b_{cond}_a")]
        assert len(c) == 3, (cond, len(c))
        for k in c:
            assert arms[k]["mode"] == "steer"
            assert arms[k].get("twin") == twin, (k, arms[k].get("twin"))
            assert arms[k]["directions_npz"].endswith(
                "llama_slot_machine_i_ba_behavioural.npz")
            assert arms[k]["alpha"] in (-3.0, 0.0, 3.0)
    # the plain no-hook -G anchor
    bases = [a for a in arms.values() if a["mode"] == "anchor_minus"]
    assert len(bases) == 1 and bases[0]["id"] == "sec4_w10b_base"


# ---------------------------------------------------- template parity

def test_w10_template_lists_both_registries():
    a_arms = load_arms(W10A_YAML)
    b_arms = load_arms(W10B_YAML)
    text = W10_TEMPLATE.read_text()
    runs = re.findall(r"run_arms\.sh \d+ ([^\n]+)", text)
    assert len(runs) == 2, "expected exactly two run_arms.sh fan-out lines"
    listed_a, listed_b = runs[0].split(), runs[1].split()
    assert sorted(listed_a) == sorted(a_arms), set(listed_a) ^ set(a_arms)
    assert sorted(listed_b) == sorted(b_arms), set(listed_b) ^ set(b_arms)
    # YAML guard: description must use ' - ', never ': ' (amlt parse trap)
    desc = text.splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    # build step: the §4.1 llama readout build via --wave10-llama at L14-19
    assert "--wave10-llama" in text and "--model llama" in text
    assert "14 19" in text
    for npz in W10_NPZ:
        assert npz in text
    # both arms yamls are wired via MLC_ARMS_YAML (default w10a + inline w10b)
    assert "arms_sec4_w10a.yaml" in text and "arms_sec4_w10b.yaml" in text
    # need-list double-quotes MUST be backslash-escaped in the bash-quoted
    # python -c block (an unescaped quote kills submission)
    need = re.search(r"need = \[([^\]]+)\]", text)
    assert need, "no need-list in sec4_w10.yaml.template"
    assert re.findall(r'(?<!\\)"', need.group(1)) == []
    assert need.group(1).count('\\"') == 2 * len(W10_NPZ)
    # serial llama SM catalog bootstrap before the fan-out; safetensors installed
    assert 'ensure_sm_catalog(\\"llama\\")' in text
    assert "safetensors" in text
    # isolation env
    assert "MLC_OUT: multilayer_causal/results/sec4_w10" in text
    assert 'MLC_SYNC_EVERY: "50"' in text


# ---------------------------------------------------- analyzer verdicts

DOSES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def _write_arm(results_dir, arm_id, alpha, mean_bet, rng, n=24,
               with_proj=False):
    p = results_dir / f"{arm_id}.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            bet = float(np.clip(mean_bet + 0.01 * rng.standard_normal(), 0, 1))
            rec = {"trial_id": i, "arm": arm_id, "alpha": alpha,
                   "parse_ok": True, "bet_ratio": bet, "action": "bet"}
            if with_proj:
                rec["vector_log"] = {"layer": 16, "proj": float(alpha * 2.0),
                                     "h_norm": 5.0}
            f.write(json.dumps(rec) + "\n")


def _write_w10_assets(assets_dir, readout_auc=0.71):
    r = np.zeros((32, 8), np.float32); r[:, 0] = 1.0
    b = np.zeros((32, 8), np.float32); b[:, 1] = 1.0    # cos(read,write) ~ 0
    np.savez(assets_dir / "llama_slot_machine_i_ba_readout.npz",
             directions=r, auc=readout_auc, cos_read_write=0.0)
    np.savez(assets_dir / "llama_slot_machine_i_ba_behavioural.npz",
             directions=b, auc=float("nan"), cos_read_write=0.0)
    np.savez(assets_dir / "llama_slot_machine_i_ba_confound.npz",
             directions=r, auc=float("nan"), cos_read_write=0.0)


def _aid(prefix, a):
    sign = "m" if a < 0 else "p"
    return f"{prefix}_a{sign}{abs(int(a))}" if a != 0 else f"{prefix}_a0"


def _write_w10a(results, rng, base, behav_slope, readout_slope, confound_slope):
    for a in DOSES:
        _write_arm(results, _aid("sec4_w10a_behavioural", a), a,
                   base + behav_slope * a, rng, with_proj=True)
        _write_arm(results, _aid("sec4_w10a_readout", a), a,
                   base + readout_slope * a, rng, with_proj=True)
        _write_arm(results, _aid("sec4_w10a_confound", a), a,
                   base + confound_slope * a, rng)
    for a in (-3.0, 0.0, 3.0):
        _write_arm(results, _aid("sec4_w10a_iec", a), a, base, rng)
        _write_arm(results, _aid("sec4_w10a_ilc", a), a, base, rng)
    for k in range(1, 6):
        _write_arm(results, f"sec4_w10a_null_{k}", 3.0, base, rng)
    _write_arm(results, "sec4_w10a_baseline", None, base, rng)
    for w in ("15_18", "14_19", "12_21"):
        _write_arm(results, f"sec4_w10a_cum_{w}", 3.0, base + behav_slope * 3, rng)


def _write_w10b(results, rng, base, minusG_slope, plusG_slope, plusM_slope):
    for a in (-3.0, 0.0, 3.0):
        _write_arm(results, _aid("sec4_w10b_minusG", a), a,
                   base + minusG_slope * a, rng)
        _write_arm(results, _aid("sec4_w10b_plusG", a), a,
                   base + plusG_slope * a, rng)
        _write_arm(results, _aid("sec4_w10b_plusM", a), a,
                   base + plusM_slope * a, rng)
    _write_arm(results, "sec4_w10b_base", None, base, rng)


def test_analyze_w10_monitor_neq_controller(tmp_path):
    """Behavioural writes (monotone, above null), readout INERT => §4.1 verdict
    LLAMA_MONITOR_NEQ_CONTROLLER; +G/+M steeper slope than -G => §4.3
    CONDITION_MODULATES."""
    rng = np.random.default_rng(0)
    results = tmp_path / "results"; assets = tmp_path / "assets"
    results.mkdir(); assets.mkdir()
    _write_w10_assets(assets)
    _write_w10a(results, rng, base=0.30, behav_slope=0.05,
                readout_slope=0.0, confound_slope=0.0)
    _write_w10b(results, rng, base=0.30, minusG_slope=0.01,
                plusG_slope=0.08, plusM_slope=0.07)

    res = sec4_stats.analyze_w10(results, assets)
    s41 = res["sec41"]
    assert s41["verdict"] == "LLAMA_MONITOR_NEQ_CONTROLLER", s41["verdict"]
    assert s41["null_cause"] == "none"
    assert s41["behavioural_writes"] and not s41["readout_writes"]
    assert s41["axes"]["behavioural"]["monotone"]
    assert not sec4_stats._writes(s41["axes"]["readout"])
    # readout decoding AUC surfaced (the better decoder, still inert)
    assert s41["decoding_auc"]["llama_slot_machine_i_ba_readout"] == 0.71
    s43 = res["sec43"]
    assert s43["verdict"] == "CONDITION_MODULATES", s43["diffs"]
    assert s43["diffs"]["plusG"]["excludes_zero"]


def test_analyze_w10_readout_uninformative_not_win(tmp_path):
    """Behavioural writes and the readout is INERT, but the readout's held-out
    decoding AUC is at chance (~0.5) => the inertness is no evidence of a
    monitor. Must NOT emit the LLAMA_MONITOR_NEQ_CONTROLLER win; instead
    READOUT_UNINFORMATIVE. Guards against a broken/row-misaligned readout
    fabricating the dissociation."""
    rng = np.random.default_rng(7)
    results = tmp_path / "results"; assets = tmp_path / "assets"
    results.mkdir(); assets.mkdir()
    _write_w10_assets(assets, readout_auc=0.50)   # chance-level decoder
    _write_w10a(results, rng, base=0.30, behav_slope=0.05,
                readout_slope=0.0, confound_slope=0.0)
    _write_w10b(results, rng, base=0.30, minusG_slope=0.01,
                plusG_slope=0.08, plusM_slope=0.07)

    s41 = sec4_stats.analyze_w10(results, assets)["sec41"]
    assert s41["verdict"] == "READOUT_UNINFORMATIVE", s41["verdict"]
    assert s41["verdict"] != "LLAMA_MONITOR_NEQ_CONTROLLER"
    assert s41["behavioural_writes"] and not s41["readout_writes"]
    assert not s41["readout_informative"]
    assert s41["readout_auc"] == 0.50


def test_analyze_w10_confounded_demotes_win(tmp_path):
    """Behavioural writes AND the confound (balance+round) axis ALSO writes =>
    H3 fails: balance-signalling could reproduce the betting change, so the
    headline is demoted to CONFOUNDED even though the readout is inert with a
    good decoder (mirrors gemma's _verdict H3 gate)."""
    rng = np.random.default_rng(8)
    results = tmp_path / "results"; assets = tmp_path / "assets"
    results.mkdir(); assets.mkdir()
    _write_w10_assets(assets, readout_auc=0.71)   # informative readout
    _write_w10a(results, rng, base=0.30, behav_slope=0.05,
                readout_slope=0.0, confound_slope=0.05)
    _write_w10b(results, rng, base=0.30, minusG_slope=0.01,
                plusG_slope=0.08, plusM_slope=0.07)

    s41 = sec4_stats.analyze_w10(results, assets)["sec41"]
    assert s41["verdict"] == "CONFOUNDED", s41["verdict"]
    assert s41["verdict"] != "LLAMA_MONITOR_NEQ_CONTROLLER"
    assert s41["behavioural_writes"] and s41["confound_writes"]


def test_analyze_w10_null_cause_window(tmp_path):
    """All three axes flat (nothing moves at these layers), baseline mid-range
    => §4.1 NULL with null_cause 'window'; equal condition slopes => §4.3
    NO_MODULATION_HEADROOM. This is the null-cause diagnostic path."""
    rng = np.random.default_rng(1)
    results = tmp_path / "results"; assets = tmp_path / "assets"
    results.mkdir(); assets.mkdir()
    _write_w10_assets(assets)
    _write_w10a(results, rng, base=0.30, behav_slope=0.0,
                readout_slope=0.0, confound_slope=0.0)
    _write_w10b(results, rng, base=0.30, minusG_slope=0.001,
                plusG_slope=0.001, plusM_slope=0.001)

    res = sec4_stats.analyze_w10(results, assets)
    s41 = res["sec41"]
    assert s41["verdict"] == "NULL"
    assert s41["null_cause"] == "window", s41["null_cause"]
    assert not s41["behavioural_moved"]        # behavioural null too => window/object
    assert res["sec43"]["verdict"] == "NO_MODULATION_HEADROOM"


def test_analyze_w10_null_cause_object_when_confound_moves(tmp_path):
    """Behavioural flat but the CONFOUND axis moves => the effect that exists is
    balance-signalling, not the risk object => null_cause 'object'."""
    rng = np.random.default_rng(2)
    results = tmp_path / "results"; assets = tmp_path / "assets"
    results.mkdir(); assets.mkdir()
    _write_w10_assets(assets)
    _write_w10a(results, rng, base=0.30, behav_slope=0.0,
                readout_slope=0.0, confound_slope=0.05)
    _write_w10b(results, rng, base=0.30, minusG_slope=0.001,
                plusG_slope=0.001, plusM_slope=0.001)

    s41 = sec4_stats.analyze_w10(results, assets)["sec41"]
    assert s41["verdict"] == "NULL"
    assert s41["confound_moved"] and not s41["behavioural_moved"]
    assert s41["null_cause"] == "object", s41["null_cause"]
