"""IC pool determinism/filtering + frozen-parser behaviour and source parity."""
import importlib.util
import inspect
import json
import random
from pathlib import Path

from multilayer_causal.src import ic

REPO = Path(__file__).resolve().parents[2]
IC_RUNNER = (REPO / "exploratory_experiments" / "alternative_paradigms" / "src"
             / "investment_choice" / "run_experiment.py")


# ---------------------------------------------------------------- pool

def _write_ic_catalog(tmp_path, monkeypatch):
    d = tmp_path / "investment_choice" / "v2_role_gemma"
    d.mkdir(parents=True)
    game1 = {"bet_constraint": "10", "decisions": [
        {"balance_before": 100, "choice": 3, "full_prompt": "P1", "skipped": False},
        {"balance_before": 90, "choice": 2, "full_prompt": "P-skip", "skipped": True},
        {"balance_before": 80, "choice": 1, "skipped": False},  # no stored prompt
    ]}
    game2 = {"bet_constraint": "10", "decisions": [
        {"balance_before": 70, "choice": 1, "actual_prompt": "P2-actual", "skipped": False},
        {"balance_before": 60, "choice": 4, "full_prompt": "", "actual_prompt": "P2b",
         "skipped": False},  # empty full_prompt → actual_prompt fallback
    ]}
    game3 = {"bet_constraint": "30", "decisions": [
        {"balance_before": 50, "choice": 4, "full_prompt": "P3", "skipped": False},
    ]}
    (d / "gemma_investment_c10_x.json").write_text(
        json.dumps({"results": [game1, game2]}))
    (d / "gemma_investment_c30_x.json").write_text(
        json.dumps({"results": [game3]}))
    monkeypatch.setenv("LLM_ADDICTION_BEHAVIORAL_ROOT", str(tmp_path))


def test_pool_filters_and_global_game_counter(tmp_path, monkeypatch):
    _write_ic_catalog(tmp_path, monkeypatch)
    pool = ic.load_ic_states("gemma", n=200)
    # skipped + promptless decisions dropped: 4 of 6 survive
    assert len(pool) == 4
    prompts = {p for _, p, _ in pool}
    assert prompts == {"P1", "P2-actual", "P2b", "P3"}
    # game_counter continues across sorted files: game3 (second file) is id 3
    by_prompt = {p: (g, m) for g, p, m in pool}
    assert by_prompt["P3"][0] == 3
    assert by_prompt["P3"][1] == {"balance_before": 50, "choice": 4, "bet_constraint": "30"}
    assert by_prompt["P1"][1]["bet_constraint"] == "10"


def test_pool_order_is_frozen_seed42(tmp_path, monkeypatch):
    _write_ic_catalog(tmp_path, monkeypatch)
    pool = ic.load_ic_states("gemma", n=200)
    assert pool == ic.load_ic_states("gemma", n=200)
    # exact recipe: catalog order then random.Random(42) shuffle
    expected = [
        (1, "P1", {"balance_before": 100, "choice": 3, "bet_constraint": "10"}),
        (2, "P2-actual", {"balance_before": 70, "choice": 1, "bet_constraint": "10"}),
        (2, "P2b", {"balance_before": 60, "choice": 4, "bet_constraint": "10"}),
        (3, "P3", {"balance_before": 50, "choice": 4, "bet_constraint": "30"}),
    ]
    random.Random(42).shuffle(expected)
    assert pool == expected
    # n truncates the shuffled order (prefix property)
    assert ic.load_ic_states("gemma", n=2) == expected[:2]


# ---------------------------------------------------------------- parser

def test_parse_explicit_decision_formats():
    # PRIORITY 1 regexes; prompt Option k → game choice via {1:4, 2:3, 3:2, 4:1}
    assert ic.parse_ic_response("Final Decision: Option 2, $5") == (3, True, True)
    assert ic.parse_ic_response("**Final Decision:** Option 1") == (4, True, True)
    assert ic.parse_ic_response("My decision: Option 3") == (2, True, False)
    assert ic.parse_ic_response("my choice: option 4") == (1, True, False)
    assert ic.parse_ic_response("I'll choose Option 1, investing $10") == (4, True, True)
    assert ic.parse_ic_response("I pick option 2") == (3, True, True)


def test_parse_last_explicit_decision_wins():
    text = "Final Decision: Option 1... wait, no. Final Decision: Option 4"
    assert ic.parse_ic_response(text) == (1, True, False)
    # 'final decision' pattern outranks bare 'decision' earlier in the text
    text2 = "My decision: Option 1 was tempting, but Final Decision: Option 4"
    assert ic.parse_ic_response(text2) == (1, True, False)


def test_parse_bare_number_prefix_completion():
    assert ic.parse_ic_response("2") == (3, True, True)
    assert ic.parse_ic_response("4\nbecause it is safe") == (1, True, False)
    assert ic.parse_ic_response("  1, $30") == (4, True, True)


def test_parse_first_digit_is_untrusted_for_chat_model():
    # gemma v2_role ran with chat_template=True → first-digit fallback parse_ok=False
    choice, ok, risky = ic.parse_ic_response("the 3.6x payout tempts me but I cannot say")
    assert (choice, ok, risky) == (2, False, False)


def test_parse_failures():
    assert ic.parse_ic_response("") == (None, False, False)
    assert ic.parse_ic_response(None) == (None, False, False)
    # no digit at all → conservative default stop (choice 1), invalid
    assert ic.parse_ic_response("no idea what to do") == (1, False, False)


def test_risky_mapping():
    # risky = game choice in (3,4) = prompt options 1,2 (9.0x / 3.6x branches)
    for option, expected_risky in [(1, True), (2, True), (3, False), (4, False)]:
        choice, ok, risky = ic.parse_ic_response(f"Final Decision: Option {option}")
        assert ok and choice == ic.PROMPT_TO_GAME[option] and risky == expected_risky


# ---------------------------------------------------------------- parity

def _load_ic_runner():
    spec = importlib.util.spec_from_file_location("ic_runexp", IC_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_source_parity_with_v2_role_runner():
    mod = _load_ic_runner()
    assert inspect.getsource(ic._FrozenICParser.parse_choice_fixed) == \
        inspect.getsource(mod.InvestmentChoiceExperiment.parse_choice_fixed)
    assert ic.PROMPT_TO_GAME == mod.PROMPT_TO_GAME
