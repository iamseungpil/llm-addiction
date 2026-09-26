"""W11 §4.2 - LLaMA task-own write-window SCAN: per-object axis build,
w11ic/w11mw config+template parity, gemma non-regression + SM byte-identity,
and the analyze_w11 window picker on synthetic rollouts.

SYNTHETIC only (no torch / no HF / no real model): the axis build is exercised
by monkeypatching load_task_arrays with synthetic arrays; the analyzer runs on
hand-written jsonl cells.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from multilayer_causal.src import indicator_axes as ia
from multilayer_causal.src import paper_axes as pa
from multilayer_causal.src import sec4_stats as st
from multilayer_causal.src.registry import load_arms

MLC = Path(__file__).resolve().parents[1]
W11IC_YAML = MLC / "configs" / "arms_sec4_w11ic.yaml"
W11MW_YAML = MLC / "configs" / "arms_sec4_w11mw.yaml"
W11_TEMPLATE = MLC / "amlt" / "sec4_w11.yaml.template"
WINDOWS = [(10, 15), (12, 17), (14, 19), (16, 21)]
# (task-flag, full task name, object stem, npz task namespace) per config.
TASKMAP = {"ic": ("investment_choice", "ic_rc"),
           "mw": ("mystery_wheel", "mw_rc")}


# --------------------------------------------- 1. per-object window axis build

def _fake_task_arrays(calls, model="llama"):
    """A load_task_arrays stand-in: synthetic behavioural inputs, recording the
    (task, rc_keep) it was called with so the test can assert the MW rc_keep
    loader is engaged and IC/SM use the default keep."""
    n_full, d = pa.MODEL_DIMS[model]

    def _load(m, task, layers, rc_keep=False, **kw):
        calls.append((task, rc_keep))
        rs = np.random.RandomState(len(layers) + len(task))
        n = 80
        u = rs.randn(d)
        u /= np.linalg.norm(u)
        i_rc = (rs.rand(n) > 0.4).astype(np.float64)
        i_ba = rs.rand(n)
        # hidden carries the i_rc contrast so the axis is well-defined
        hidden = (i_rc[:, None, None] * u[None, None, :]
                  + 0.05 * rs.randn(n, len(layers), d))
        return {"hidden": hidden, "balance": rs.rand(n),
                "rounds": rs.randint(1, 20, n).astype(float),
                "groups": np.arange(n) // 4,
                "indicators": {"i_ba": i_ba, "i_rc": i_rc},
                "build_game_ids": np.unique(np.arange(n) // 4)}
    return _load


@pytest.mark.parametrize("task,expect_rc_keep", [
    ("slot_machine", False), ("investment_choice", False),
    ("mystery_wheel", True)])
def test_build_llama_window_per_object(tmp_path, monkeypatch, task,
                                       expect_rc_keep):
    """--build-llama-windows builds the task's OWN behavioural object and saves
    llama_{task}_{object}_behavioural_L{lo}_{hi}.npz at 32-row llama geometry,
    steering the correct indicator (SM i_ba, IC/MW i_rc) and engaging the
    rc_keep loader ONLY for MW."""
    calls = []
    monkeypatch.setattr(ia, "load_task_arrays", _fake_task_arrays(calls))
    obj = {"slot_machine": "i_ba", "investment_choice": "ic_rc",
           "mystery_wheel": "mw_rc"}[task]
    ind = {"slot_machine": "i_ba", "investment_choice": "i_rc",
           "mystery_wheel": "i_rc"}[task]
    layers = list(range(14, 20))
    dest = ia._build_llama_window(tmp_path, "llama", layers, task)
    assert dest.name == f"llama_{task}_{obj}_behavioural_L14_19.npz"
    assert calls == [(task, expect_rc_keep)]  # rc_keep only for MW
    z = np.load(dest)
    assert z["directions"].shape == (32, 4096)   # llama full-layer geometry
    assert str(z["indicator"]) == ind
    assert str(z["task"]) == task
    assert str(z["axis"]) == "behavioural"
    # every row is a unit direction (non-window rows replicate row 0, like the
    # runner npz schema); only the window layers carry a non-zero scale so only
    # they are steered.
    assert np.allclose(np.linalg.norm(z["directions"], axis=1), 1.0, atol=1e-4)
    non = [i for i in range(32) if i not in layers]
    assert np.all(z["scales"][layers] > 0)
    assert np.allclose(z["scales"][non], 0.0)


def test_build_llama_window_object_map_sm_byte_identity():
    """The SM object mapping (the 'prior llama' path) is unchanged: same key,
    stem and provenance tag the W8 window scan committed — IC/MW are purely
    additive rows."""
    m = ia.LLAMA_WINDOW_OBJECT
    assert m["slot_machine"] == ("i_ba", "i_ba", False, "W8 llama SM window scan")
    assert m["investment_choice"][:3] == ("i_rc", "ic_rc", False)
    assert m["mystery_wheel"][:3] == ("i_rc", "mw_rc", True)


# ------------------------------------------------------- gemma non-regression

def test_gemma_geometry_nonregression():
    """The additive W11 build path never touches gemma geometry."""
    assert pa.MODEL_DIMS["gemma"] == (42, 3584)
    assert pa.MODEL_DIMS["llama"] == (32, 4096)
    assert ia.N_LAYERS == 42 and ia.D_MODEL == 3584
    assert ia._replicate_to_full(np.ones((1, 4), np.float32), [0]).shape == (42, 4)


# ------------------------------------------------- 2. w11ic/w11mw config parity

@pytest.mark.parametrize("task,yaml_path,seed", [
    ("ic", W11IC_YAML, 7000042), ("mw", W11MW_YAML, 7000043)])
def test_w11_registry_counts_and_fields(task, yaml_path, seed):
    full, obj = TASKMAP[task]
    arms = load_arms(yaml_path)
    assert len(arms) == 16                          # 12 steer + 3 null + 1 base
    for a in arms.values():
        assert a["model"] == "llama"
        assert a["task"] == task
        assert a["n"] == 200                        # > W9's 150
        assert a["seed_base"] == seed
        assert a["state_offset"] == 0
        assert a["prompt_set"] == "addiction_role_gm"
        assert a["phase"] == "sec4_w11"
        assert max(a["layers"]) <= 31               # llama bound
    steer = {k: a for k, a in arms.items()
             if a["mode"] == "steer" and a.get("direction") != "random"}
    assert len(steer) == 12                         # 4 windows x 3 alphas
    seen = set()
    for a in steer.values():
        lo, hi = a["layers"][0], a["layers"][-1]
        seen.add((lo, hi, a["alpha"]))
        assert a["directions_npz"].endswith(
            f"llama_{full}_{obj}_behavioural_L{lo}_{hi}.npz")
    for lo, hi in WINDOWS:
        for al in (-3, 0, 3):
            assert (lo, hi, al) in seen, (lo, hi, al)
    # exactly one no-steer baseline + exactly 3 random nulls (their axis is the
    # middle L14-19 window)
    assert sum(1 for a in arms.values() if a["mode"] == "anchor_minus") == 1
    nulls = [a for a in arms.values() if a.get("direction") == "random"]
    assert len(nulls) == 3
    for a in nulls:
        assert a["layers"] == list(range(14, 20))
        assert a["directions_npz"].endswith(
            f"llama_{full}_{obj}_behavioural_L14_19.npz")
    # unique dir_seeds across the 3 nulls
    assert len({a["dir_seed"] for a in nulls}) == 3


# ------------------------------------------------------- 3. template parity

def test_w11_template_lists_exactly_both_registries():
    text = W11_TEMPLATE.read_text()
    runs = re.findall(r"run_arms\.sh \d+ ([^\n]+)", text)
    assert len(runs) == 2, "expected one run_arms.sh line per config"
    listed = sorted(w for line in runs for w in line.split())
    want = sorted(list(load_arms(W11IC_YAML)) + list(load_arms(W11MW_YAML)))
    assert listed == want, set(listed) ^ set(want)
    # each run_arms.sh call overrides MLC_ARMS_YAML inline (ic then mw)
    assert "MLC_ARMS_YAML=multilayer_causal/configs/arms_sec4_w11ic.yaml bash" in text
    assert "MLC_ARMS_YAML=multilayer_causal/configs/arms_sec4_w11mw.yaml bash" in text
    # YAML guard: description uses ' - ', never ': ' (amlt parse trap)
    desc = text.splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    # NEW secure docker image (torch2.8), mirrored from sec4_w8scan
    assert "aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x:latest" in text
    # build step builds each window for BOTH task objects
    assert "--build-llama-windows" in text
    assert "--task investment_choice" in text and "--task mystery_wheel" in text
    for lo, hi in WINDOWS:
        assert f"{lo} {hi}" in text                 # window passed to --layers
    # the 8 need-list npz (4 windows x 2 objects), all escaped
    need = re.search(r"need = \[([^\]]+)\]", text)
    assert need, "no need-list in sec4_w11.yaml.template"
    npz = [f"llama_{TASKMAP[t][0]}_{TASKMAP[t][1]}_behavioural_L{lo}_{hi}.npz"
           for t in ("ic", "mw") for lo, hi in WINDOWS]
    for name in npz:
        assert name in text
    assert re.findall(r'(?<!\\)"', need.group(1)) == []          # none unescaped
    assert need.group(1).count('\\"') == 2 * len(npz)            # 8 npz x 2
    # IC + MW catalog bootstraps before the fan-out (escaped-quote form)
    assert 'ensure_ic_catalog(\\"llama\\")' in text
    assert 'ensure_mw_catalog(\\"llama\\")' in text
    # isolation env
    assert "MLC_OUT: multilayer_causal/results/sec4_w11" in text
    assert 'MLC_SYNC_EVERY: "50"' in text


# ------------------------------------------------------- 4. analyze_w11 picker

def _write_cell(d: Path, arm: str, alpha, rate: float, n=200, parse=0.9):
    """A jsonl arm file: `n` rows, `parse` fraction parse_ok, and a risky/spin
    rate of `rate` among the parse_ok rows."""
    rows = []
    n_ok = int(round(n * parse))
    n_risky = int(round(n_ok * rate))
    for i in range(n):
        ok = i < n_ok
        rows.append({"parse_ok": ok, "alpha": alpha,
                     "risky": bool(ok and i < n_risky)})
    (d / f"{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def test_analyze_w11_picks_coherent_window(tmp_path):
    """A monotone rising ic_rc window (L12-17) that clears the null band is the
    chosen window; a flat window is not. MW has no moving window -> NO_WINDOW
    with an 'absent' null_cause (baseline mid-range)."""
    d = tmp_path
    # IC: baseline + 3 flat nulls at ~0.30
    _write_cell(d, "sec4_w11ic_base", None, 0.30)
    for k, al in ((1, 3.0), (2, -3.0), (3, 3.0)):
        _write_cell(d, f"sec4_w11ic_null{k}_a{'p' if al > 0 else 'm'}3", al, 0.31)
    # window L10-15 flat; L12-17 strongly rising (0.20 -> 0.30 -> 0.60);
    # L14-19 / L16-21 flat
    for lo, hi in WINDOWS:
        if (lo, hi) == (12, 17):
            rates = {-3.0: 0.20, 0.0: 0.30, 3.0: 0.60}
        else:
            rates = {-3.0: 0.30, 0.0: 0.30, 3.0: 0.30}
        for al, r in rates.items():
            suf = "a0" if al == 0 else ("ap3" if al > 0 else "am3")
            _write_cell(d, f"sec4_w11ic_L{lo}_{hi}_{suf}", al, r)
    # MW: baseline + nulls + all-flat windows -> no coherent window
    _write_cell(d, "sec4_w11mw_base", None, 0.40)
    for k, al in ((1, 3.0), (2, -3.0), (3, 3.0)):
        _write_cell(d, f"sec4_w11mw_null{k}_a{'p' if al > 0 else 'm'}3", al, 0.40)
    for lo, hi in WINDOWS:
        for al, suf in ((-3.0, "am3"), (0.0, "a0"), (3.0, "ap3")):
            _write_cell(d, f"sec4_w11mw_L{lo}_{hi}_{suf}", al, 0.40)

    res = st.analyze_w11(d)
    assert res["verdicts"]["ic"] == "IC_HAS_CAUSAL_WINDOW"
    assert res["chosen_windows"]["ic"] == "L12_17"
    assert res["any_window_significant"]["ic"] is True
    assert res["tasks"]["ic"]["windows"]["L12_17"]["coherent"] is True
    assert res["tasks"]["ic"]["windows"]["L10_15"]["coherent"] is False
    # MW: nothing moves -> NO_WINDOW, absent (baseline 0.40 is mid-range)
    assert res["verdicts"]["mw"] == "MW_NO_WINDOW"
    assert res["chosen_windows"]["mw"] is None
    assert res["any_window_significant"]["mw"] is False
    assert res["tasks"]["mw"]["null_cause"] == "absent"


def test_analyze_w11_headroom_when_baseline_at_ceiling(tmp_path):
    """No moving window AND a baseline at the rate ceiling -> null_cause
    'headroom' (no room to move), the reportable headroom-limited finding."""
    d = tmp_path
    _write_cell(d, "sec4_w11ic_base", None, 0.97)
    for k, al in ((1, 3.0), (2, -3.0), (3, 3.0)):
        _write_cell(d, f"sec4_w11ic_null{k}_a{'p' if al > 0 else 'm'}3", al, 0.97)
    for lo, hi in WINDOWS:
        for al, suf in ((-3.0, "am3"), (0.0, "a0"), (3.0, "ap3")):
            _write_cell(d, f"sec4_w11ic_L{lo}_{hi}_{suf}", al, 0.97)
    res = st.analyze_w11(d)
    assert res["verdicts"]["ic"] == "IC_NO_WINDOW"
    assert res["tasks"]["ic"]["null_cause"] == "headroom"
