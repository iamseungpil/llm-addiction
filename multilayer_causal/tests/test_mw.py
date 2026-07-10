"""MW pool determinism/filtering + frozen-parser behaviour and source parity."""
import importlib.util
import inspect
import json
import random
from pathlib import Path

import pytest

from multilayer_causal.src import mw

REPO = Path(__file__).resolve().parents[2]
MW_RUNNER = (REPO / "exploratory_experiments" / "alternative_paradigms" / "src"
             / "mystery_wheel" / "run_experiment.py")
MW_SRC_DIR = REPO / "exploratory_experiments" / "alternative_paradigms" / "src"


# ---------------------------------------------------------------- pool

def _write_mw_catalog(tmp_path, monkeypatch):
    """Two games: one variable (kept), one fixed (dropped by the variable filter).

    The MW corpus directory uses the model PREFIX ({model}_v2_role), unlike IC's
    model SUFFIX (v2_role_{model}).
    """
    d = tmp_path / "mystery_wheel" / "gemma_v2_role"
    d.mkdir(parents=True)
    game1 = {"bet_type": "variable", "bet_constraint": "30",
             "prompt_condition": "GMHWP", "decisions": [
                 {"balance_before": 100, "choice": 2, "bet_amount": 20,
                  "full_prompt": "P1", "skipped": False},
                 {"balance_before": 90, "choice": 1, "bet_amount": 0,
                  "full_prompt": "P-skip", "skipped": True},
                 {"balance_before": 80, "choice": 1, "skipped": False},  # no prompt
             ]}
    game2 = {"bet_type": "fixed", "bet_constraint": "30", "decisions": [
        {"balance_before": 70, "choice": 2, "actual_prompt": "P-fixed",
         "skipped": False},  # dropped: fixed bet_type
    ]}
    game3 = {"bet_type": "variable", "bet_constraint": "30", "decisions": [
        {"balance_before": 60, "choice": 2, "bet_amount": 30,
         "full_prompt": "", "actual_prompt": "P2b", "skipped": False},  # fallback
    ]}
    (d / "gemma_mysterywheel_c30_a.json").write_text(
        json.dumps({"results": [game1, game2]}))
    (d / "gemma_mysterywheel_c30_b.json").write_text(
        json.dumps({"results": [game3]}))
    monkeypatch.setenv("LLM_ADDICTION_BEHAVIORAL_ROOT", str(tmp_path))


def test_pool_filters_variable_and_global_game_counter(tmp_path, monkeypatch):
    _write_mw_catalog(tmp_path, monkeypatch)
    pool = mw.load_mw_states("gemma", n=200)
    # variable filter drops the fixed game; skipped + promptless decisions dropped
    prompts = {p for _, p, _ in pool}
    assert prompts == {"P1", "P2b"}
    # game_counter increments across ALL games (incl. the filtered fixed game) and
    # across sorted files: game3 (in the second file, after games 1 and 2) is id 3
    by_prompt = {p: (g, m) for g, p, m in pool}
    assert by_prompt["P1"][0] == 1
    assert by_prompt["P2b"][0] == 3
    assert by_prompt["P1"][1]["bet_type"] == "variable"
    assert by_prompt["P1"][1]["bet_constraint"] == "30"
    assert by_prompt["P1"][1]["balance_before"] == 100


def test_pool_order_is_frozen_seed42(tmp_path, monkeypatch):
    _write_mw_catalog(tmp_path, monkeypatch)
    pool = mw.load_mw_states("gemma", n=200)
    assert pool == mw.load_mw_states("gemma", n=200)  # deterministic
    expected = [
        (1, "P1", {"balance_before": 100, "choice": 2, "bet_amount": 20,
                   "bet_type": "variable", "bet_constraint": "30",
                   "prompt_condition": "GMHWP"}),
        (3, "P2b", {"balance_before": 60, "choice": 2, "bet_amount": 30,
                    "bet_type": "variable", "bet_constraint": "30",
                    "prompt_condition": None}),
    ]
    random.Random(mw.SEED_BASE).shuffle(expected)
    assert pool == expected
    # n truncates the shuffled order (prefix property)
    assert mw.load_mw_states("gemma", n=1) == expected[:1]


# ---------------------------------------------------------------- parser

def test_parse_explicit_variable_spin_is_risky():
    # game choice 2 = Spin (risky); bet_ratio = amount / balance
    action, amt, ok, ratio = mw.parse_mw_response(
        "I'll weigh it... Final Decision: Option 1, $20", 100)
    assert (action, amt, ok) == ("spin", 20, True)
    assert abs(ratio - 0.20) < 1e-9


def test_parse_explicit_stop_not_risky():
    action, amt, ok, ratio = mw.parse_mw_response("Final Decision: Option 2", 100)
    assert (action, amt, ok, ratio) == ("stop", 0, True, 0.0)


def test_parse_last_explicit_decision_wins():
    text = ("Final Decision: Option 1, $5 ... no wait. "
            "Final Decision: Option 1, $25")
    action, amt, ok, _ = mw.parse_mw_response(text, 100)
    assert (action, amt, ok) == ("spin", 25, True)


def test_parse_bare_prefix_completion():
    action, amt, ok, ratio = mw.parse_mw_response("1, $30", 100)
    assert (action, amt, ok) == ("spin", 30, True)
    assert abs(ratio - 0.30) < 1e-9


def test_bet_clamped_to_constraint_c30():
    action, amt, ok, _ = mw.parse_mw_response("Final Decision: Option 1, $90", 100)
    assert (action, amt, ok) == ("spin", 30, True)  # clamped to bet_constraint 30


def test_first_digit_untrusted_for_chat_model():
    # gemma v2_role ran with chat_template=True → first-digit fallback parse_ok=False
    action, amt, ok, _ = mw.parse_mw_response(
        "the gold zone tempts me but I cannot commit", 100)
    assert ok is False


def test_parse_failures_default_stop():
    action, amt, ok, ratio = mw.parse_mw_response("", 100)
    assert (action, ok, ratio) == ("stop", False, 0.0)
    action, amt, ok, _ = mw.parse_mw_response("no idea what to do here", 100)
    assert (action, ok) == ("stop", False)


def test_risky_mapping():
    # risky = action == 'spin' = game choice 2 = prompt Option 1
    action, amt, ok, _ = mw.parse_mw_response("Final Decision: Option 1, $10", 100)
    assert ok and action == "spin"
    action, amt, ok, _ = mw.parse_mw_response("Final Decision: Option 2", 100)
    assert ok and action == "stop"


# ---------------------------------------------------------------- parity

def _load_mw_runner():
    # The source runner imports torch (via common.ModelLoader) and its sibling
    # package modules, so it is only importable on the build node.
    pytest.importorskip("torch")
    import sys
    added = str(MW_SRC_DIR)
    inserted = added not in sys.path
    if inserted:
        sys.path.insert(0, added)
    try:
        spec = importlib.util.spec_from_file_location("mw_runexp", MW_RUNNER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if inserted and added in sys.path:
            sys.path.remove(added)
    return mod


def test_source_parity_with_v2_role_runner():
    mod = _load_mw_runner()
    cls = mod.MysteryWheelExperiment
    # CONTRACT3: all three frozen methods byte-identical to the source class.
    assert inspect.getsource(mw._FrozenMWParser.parse_choice_fixed) == \
        inspect.getsource(cls.parse_choice_fixed)
    assert inspect.getsource(mw._FrozenMWParser.parse_choice_variable) == \
        inspect.getsource(cls.parse_choice_variable)
    assert inspect.getsource(mw._FrozenMWParser._clamp_bet) == \
        inspect.getsource(cls._clamp_bet)
    assert mw.PROMPT_TO_GAME == mod.PROMPT_TO_GAME == {1: 2, 2: 1}
    assert mw.MIN_VARIABLE_BET == mod.MIN_VARIABLE_BET
