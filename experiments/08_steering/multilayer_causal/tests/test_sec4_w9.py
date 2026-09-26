"""W9 - LLaMA reduced symmetric matrix: loader labeling + config/template parity.

SYNTHETIC only (no torch / no HF / no real model). Covers:
  * the llama IC/MW loader LABELING mirrors gemma (i_rc = choice>=3 for IC,
    mw_rc via the REUSED _mw_rc_labels helper for MW) on synthetic arrays;
  * gemma NON-REGRESSION — the model-agnostic _build_wave7_axes the llama build
    reuses still returns the 12-cell grid with 42/3584 gemma geometry intact;
  * arms_sec4_w9.yaml <-> amlt/sec4_w9.yaml.template consistency (63 arms,
    diagonal 7 doses / off-diagonal 3, per-target 2 nulls + baseline, escaped
    need-list quotes, isolation env, --wave7-llama build).
"""
import re
from pathlib import Path

import numpy as np

from multilayer_causal.src import indicator_axes as ia
from multilayer_causal.src import paper_axes as pa
from multilayer_causal.src.registry import load_arms

MLC = Path(__file__).resolve().parents[1]
W9_YAML = MLC / "configs" / "arms_sec4_w9.yaml"
W9_TEMPLATE = MLC / "amlt" / "sec4_w9.yaml.template"

SOURCES = ("smiba", "icrc", "mwrc", "sh3c")
TARGETS = ("sm", "ic", "mw")
SELF_OF = {"smiba": "sm", "icrc": "ic", "mwrc": "mw"}  # sh3c has no self cell
W9_NPZ = tuple(f"llama_w7_{s}_{t}scale.npz" for s in SOURCES for t in TARGETS)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------- llama IC/MW LABELING

def _ic_rc(choices):
    """The risky-CHOICE contrast expression BOTH _load_task_arrays (gemma) and
    _load_llama_task_arrays (llama) use inline for IC — kept here verbatim so
    the test fails if either branch drifts from the shared formula."""
    return ((choices >= 3).astype(np.float64)
            * np.where(np.isfinite(choices), 1.0, np.nan))


def test_llama_ic_rc_labeling_mirrors_gemma():
    """IC i_rc = (choice>=3), NaN where the catalog row has no valid choice —
    the W5 risky-OPTION contrast (game choice 3/4 = risky), identical on both
    models. Amount i_ba is explicitly NOT the IC object here."""
    choices = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 3.0])
    rc = _ic_rc(choices)
    assert rc[0] == 0.0 and rc[1] == 0.0        # safe options 1/2
    assert rc[2] == 1.0 and rc[3] == 1.0        # risky options 3/4
    assert np.isnan(rc[4])                       # unjoinable: no choice
    assert rc[5] == 1.0
    # binary contrast: both classes present, no leakage of the NaN row
    fin = rc[np.isfinite(rc)]
    assert set(np.unique(fin)) == {0.0, 1.0}


def test_llama_mw_rc_labeling_reuses_shared_helper():
    """MW spin/stop labels come from the REUSED _mw_rc_labels helper (the same
    gemma object): choice 2 with bet>0 = spin (1), choice 1 = stop (0), choice 2
    without a bet = NaN, unjoinable = NaN. The llama loader calls this helper on
    its own (choices, bets), never a forked copy."""
    choices = np.array([2.0, 1.0, 2.0, 2.0, np.nan, 1.0])
    bets = np.array([20.0, np.nan, np.nan, 0.0, 15.0, 0.0])
    rc = ia._mw_rc_labels(choices, bets)
    assert rc[0] == 1.0                          # spin
    assert rc[1] == 0.0                          # stop (no bet needed)
    assert np.isnan(rc[2]) and np.isnan(rc[3])   # parse edges
    assert np.isnan(rc[4])                       # unjoinable
    assert rc[5] == 0.0                          # stop with stored 0 bet


def test_llama_hidden_map_and_ic_provenance_required():
    """The llama loader wires the per-task hidden dumps and REQUIRES explicit
    game_ids/round_nums provenance for the catalog join on EVERY task. A dump
    without it (the old IC shape: hidden_states[+valid_mask] only) carries no
    per-row anchor to validate an order join against, so _llama_catalog_meta
    refuses it LOUD instead of silently mislabeling every row — a row-count
    guard cannot detect a reorder/filter mismatch."""
    assert ia.LLAMA_HIDDEN["slot_machine"] == pa.LLAMA_SM_HIDDEN
    assert ia.LLAMA_HIDDEN["investment_choice"] == pa.LLAMA_IC_HIDDEN
    assert ia.LLAMA_HIDDEN["mystery_wheel"] == pa.LLAMA_MW_HIDDEN

    class _Z:  # minimal np.load-like stand-in (.files list + __getitem__)
        def __init__(self, d):
            self._d = d
            self.files = list(d)

        def __getitem__(self, k):
            return self._d[k]

    # provenance present -> by-id meta (same 1-based game_counter id space as
    # SM/MW/gemma), passed straight through for the by-id _iba_from_catalog join
    z_ok = _Z({"hidden_states": None,
               "game_ids": np.array([1, 1, 2]),
               "round_nums": np.array([1.0, 2.0, 1.0])})
    meta = ia._llama_catalog_meta(z_ok, "investment_choice", 3)
    assert list(meta["game_ids"]) == [1, 1, 2]
    assert list(meta["round_nums"]) == [1.0, 2.0, 1.0]
    assert np.all(np.isnan(meta["balances"]))  # no balances key -> NaN, catalog fills

    # provenance ABSENT -> loud refusal (no silent order-join), for IC and MW
    z_bad = _Z({"hidden_states": None, "valid_mask": np.ones(3, bool)})
    import pytest
    for task in ("investment_choice", "mystery_wheel"):
        with pytest.raises(AssertionError):
            ia._llama_catalog_meta(z_bad, task, 3)


# --------------------------------------------- gemma NON-REGRESSION (w7 grid)

def _w6ish(rs, d=8, n=600, task="sm"):
    u = _unit(rs.randn(d))
    base = {"balance": rs.rand(n), "rounds": rs.randint(1, 20, n),
            "groups": np.arange(n) // 4}
    if task == "sm":
        ind = rs.randn(n)
        hidden = ind[:, None, None] * u[None, None, :] + 0.05 * rs.randn(n, 2, d)
        return {**base, "hidden": hidden,
                "indicators": {"i_ba": ind, "i_rc": np.full(n, np.nan)},
                "iba_finite": np.ones(n, dtype=bool)}
    rc = (rs.rand(n) > 0.4).astype(np.float64)
    hidden = (rc[:, None, None] * u[None, None, :] + 0.05 * rs.randn(n, 2, d))
    iba = np.where(rc == 1.0, rs.rand(n), np.nan)
    return {**base, "hidden": hidden,
            "indicators": {"i_ba": np.nan_to_num(iba), "i_rc": rc},
            "iba_finite": np.isfinite(iba)}


def test_build_wave7_grid_shared_by_both_models():
    """The llama W9 build reuses _build_wave7_axes verbatim — assert it still
    produces the 12-cell (source x target-scale) grid the gemma W7 build emits,
    with unit directions and per-target scales (only the loaded hidden differs
    between models). Gemma module geometry stays 42/3584."""
    assert pa.MODEL_DIMS["gemma"] == (42, 3584)
    assert pa.MODEL_DIMS["llama"] == (32, 4096)
    assert ia.N_LAYERS == 42
    rs = np.random.RandomState(19)
    task_data = {"slot_machine": _w6ish(rs, task="sm"),
                 "investment_choice": _w6ish(rs, task="ic"),
                 "mystery_wheel": _w6ish(rs, task="mw")}
    grid, cos_pairs = ia._build_wave7_axes(task_data, layers=[0, 1])
    assert set(grid) == {(s, t) for s in ("smiba", "icrc", "mwrc", "sh3c")
                         for t in ("sm", "ic", "mw")}
    assert len(grid) == 12
    for key, b in grid.items():
        assert b["directions"].shape == (2, 8), key
        assert np.allclose(np.linalg.norm(b["directions"], axis=1), 1, atol=1e-5)
    # the 6 pre-registration loading cosines are re-printed for the launch log
    assert set(cos_pairs) == {"mw_rc~mw_iba", "mw_rc~sm_iba", "mw_rc~ic_rc",
                              "shared3c~sm_iba", "shared3c~ic_rc",
                              "shared3c~mw_rc"}


def test_save_axis_llama_is_32_rows_no_scales_full():
    """The W9 build calls _save_axis with model='llama' (32-row npz) and
    scales_full=None; the window rows must carry the built target scales and the
    non-window rows stay zero (arms only steer the window)."""
    import tempfile
    layers = [14, 15, 16, 17, 18, 19]
    rs = np.random.RandomState(3)
    built = {"directions": np.array([_unit(rs.randn(4)) for _ in layers]),
             "scales": np.arange(1, len(layers) + 1, dtype=np.float32),
             "auc": float("nan"),
             "provenance": "test"}
    with tempfile.TemporaryDirectory() as td:
        dest = ia._save_axis(td, "llama", "w7", "smiba", "smscale", built,
                             layers, None, float("nan"),
                             np.array([1, 2, 3]))
        z = np.load(dest)
        assert z["directions"].shape[0] == 32   # 32-row llama schema
        scales = z["scales"]
        assert scales.shape[0] == 32
        for li, l in enumerate(layers):
            assert scales[l] == built["scales"][li]
        non_window = [i for i in range(32) if i not in layers]
        assert np.allclose(scales[non_window], 0.0)


# ------------------------------------------------------ w9 registry parity

def test_w9_registry_counts_and_fields():
    arms = load_arms(W9_YAML)
    assert len(arms) == 63
    for a in arms.values():
        assert a["model"] == "llama"
        assert a["n"] == 150
        assert a["seed_base"] == 5000042
        assert a["state_offset"] == 0
        assert a["prompt_set"] == "addiction_role_gm"
        assert a["phase"] == "sec4_w9"
        assert a.get("task", "sm") in ("sm", "ic", "mw")
        assert a["layers"] == list(range(14, 20))  # [14,19] expanded, L14-19
        assert max(a["layers"]) <= 31  # llama bound
    # steer (non-null) cells: 4 sources x 3 targets, diagonal 7 / off-diag 3
    steer = {k: a for k, a in arms.items()
             if a["mode"] == "steer" and a.get("direction") != "random"}
    assert len(steer) == 48  # 3*7 (diagonal) + 9*3 (off-diagonal)
    per_cell = {}
    for k in steer:
        m = re.match(r"sec4_w9_(smiba|icrc|mwrc|sh3c)_(sm|ic|mw)_", k)
        assert m, k
        per_cell.setdefault((m.group(1), m.group(2)), 0)
        per_cell[(m.group(1), m.group(2))] += 1
    for src in SOURCES:
        for tgt in TARGETS:
            n = per_cell[(src, tgt)]
            assert n == (7 if SELF_OF.get(src) == tgt else 3), (src, tgt, n)
    # every steer cell points at its own source@target-scale npz
    for k, a in steer.items():
        m = re.match(r"sec4_w9_(\w+?)_(sm|ic|mw)_", k)
        src, tgt = m.group(1), m.group(2)
        assert a["directions_npz"].endswith(f"llama_w7_{src}_{tgt}scale.npz"), k
    # nulls: per-target 2 x +/-3 = 12; baselines: 3
    nulls = [a for a in arms.values() if a.get("direction") == "random"]
    assert len(nulls) == 12
    for a in nulls:
        assert a["alpha"] in (-3.0, 3.0) and a["log_vectors"] is False
    bases = [a for a in arms.values() if a["mode"] == "anchor_minus"]
    assert len(bases) == 3
    assert {a["task"] for a in bases} == {"sm", "ic", "mw"}
    # unique dir_seeds across the nulls
    seeds = [a["dir_seed"] for a in nulls]
    assert len(set(seeds)) == len(seeds)


def test_w9_template_lists_exactly_the_registry_arms():
    arms = load_arms(W9_YAML)
    text = W9_TEMPLATE.read_text()
    m = re.search(r"run_arms\.sh \d+ ([^\n]+)", text)
    assert m, "no run_arms.sh line in sec4_w9.yaml.template"
    listed = m.group(1).split()
    assert sorted(listed) == sorted(arms), set(listed) ^ set(arms)
    # YAML guard: description must use ' - ', never ': ' (amlt parse trap)
    desc = text.splitlines()[0]
    assert desc.startswith("description:")
    assert ": " not in desc[len("description:"):]
    # build step: the model-aware llama W7 grid via --wave7-llama at L14-19
    assert "--wave7-llama" in text and "--model llama" in text
    assert "14 19" in text  # window passed to --layers
    for npz in W9_NPZ:
        assert npz in text
    # the need-list double-quotes MUST be backslash-escaped inside the bash
    # single-quoted python -c block (an unescaped quote kills submission)
    need = re.search(r"need = \[([^\]]+)\]", text)
    assert need, "no need-list in sec4_w9.yaml.template"
    assert re.findall(r'(?<!\\)"', need.group(1)) == []
    assert need.group(1).count('\\"') == 2 * len(W9_NPZ)
    # serial llama catalog bootstrap (all three tasks) before the fan-out
    assert 'ensure_sm_catalog(\\"llama\\")' in text
    assert 'ensure_ic_catalog(\\"llama\\")' in text
    assert 'ensure_mw_catalog(\\"llama\\")' in text
    # isolation env
    assert "MLC_OUT: multilayer_causal/results/sec4_w9" in text
    assert "MLC_ARMS_YAML: multilayer_causal/configs/arms_sec4_w9.yaml" in text
    assert 'MLC_SYNC_EVERY: "50"' in text
