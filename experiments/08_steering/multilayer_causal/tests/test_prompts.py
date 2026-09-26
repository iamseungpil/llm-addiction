"""Prompt builder/parser must stay byte-identical to the M3'' originals."""
import importlib.util
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_m3pp():
    spec = importlib.util.spec_from_file_location(
        "m3pp", REPO / "sae_v3_analysis" / "src" / "run_m3pp_strong_patching.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_source_parity_with_m3pp():
    from multilayer_causal.src import prompts
    m3pp = _load_m3pp()
    assert inspect.getsource(prompts.build_prompt) == inspect.getsource(m3pp.build_prompt)
    assert inspect.getsource(prompts.parse_response) == inspect.getsource(m3pp.parse_response)


def test_build_prompt_minimal_game():
    from multilayer_causal.src.prompts import build_prompt
    game = {"decisions": [{"balance_before": 80}], "bet_type": "variable",
            "prompt_combo": "", "history": []}
    p = build_prompt(game, 0)
    assert "Current balance: $80" in p
    assert "Bet (choose $5-$80)" in p
    assert "Final Decision:" in p
    pg = build_prompt(game, 0, override_combo="G")
    assert "set a target amount yourself" in pg and "set a target amount" not in p


def test_parse_response_cases():
    from multilayer_causal.src.prompts import parse_response
    assert parse_response("blah\nFinal Decision: Bet $30", 100) == ("bet", 30)
    assert parse_response("Final Decision: Stop", 100) == ("stop", 0)
    assert parse_response("I will stop playing now.", 100) == ("stop", 0)
    assert parse_response("no decision here", 100) == ("bet", 10)
    assert parse_response("Final Decision: Bet $500", 60) == ("bet", 60)
