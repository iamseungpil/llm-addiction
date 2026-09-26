"""Unit tests for runner.build_branch_game — the W1 LC-probe counterfactual.

Branch arithmetic (spec 2026-06-10 W0-W2): LOSS new_bal = bal - bet,
WIN new_bal = bal + 2*bet (3.0x payout, net +2x). The appended history entry
must render through the FROZEN build_prompt as a normal game-history line.
"""
import copy

import pytest

from multilayer_causal.src.prompts import build_prompt
from multilayer_causal.src.runner import build_branch_game


def _game():
    # round 1: bet 10 LOSS (100→90); round 2: bet 10 WIN (90→110, 3x payout)
    return {
        "bet_type": "variable",
        "prompt_combo": "WP",
        "decisions": [{"balance_before": 100}, {"balance_before": 90},
                      {"balance_before": 110}],
        "history": [{"round": 1, "bet": 10, "win": False, "balance": 90},
                    {"round": 2, "bet": 10, "win": True, "balance": 110}],
    }


def test_loss_branch_arithmetic():
    bg = build_branch_game(_game(), 2, 20, win=False)
    assert bg["decisions"][-1] == {"balance_before": 90}            # 110 - 20
    assert len(bg["decisions"]) == 4
    assert bg["history"][-1] == {"round": 3, "bet": 20, "win": False, "balance": 90}
    assert len(bg["history"]) == 3
    assert bg["bet_type"] == "variable" and bg["prompt_combo"] == "WP"


def test_win_branch_arithmetic():
    bg = build_branch_game(_game(), 2, 20, win=True)
    assert bg["decisions"][-1] == {"balance_before": 150}           # 110 + 2*20
    assert bg["history"][-1] == {"round": 3, "bet": 20, "win": True, "balance": 150}


def test_branch_prompt_renders_history_line():
    pl = build_prompt(build_branch_game(_game(), 2, 20, win=False), 3)
    assert pl is not None
    assert "Current balance: $90" in pl
    assert "Round 3: Bet $20, LOSS, Balance $90" in pl
    pw = build_prompt(build_branch_game(_game(), 2, 20, win=True), 3)
    assert "Current balance: $150" in pw
    assert "Round 3: Bet $20, WIN, Balance $150" in pw


def test_branch_prompt_respects_override_combo():
    bg = build_branch_game(_game(), 2, 20, win=False)
    pp = build_prompt(bg, 3, override_combo="WPG")
    assert "set a target amount yourself" in pp                     # G component
    pm = build_prompt(bg, 3, override_combo="WP")
    assert "set a target amount yourself" not in pm


def test_original_game_not_mutated():
    g = _game()
    snapshot = copy.deepcopy(g)
    build_branch_game(g, 2, 20, win=False)
    build_branch_game(g, 2, 20, win=True)
    assert g == snapshot


def test_branch_truncates_future_rounds():
    # branching mid-game must drop the original round-3+ tail
    bg = build_branch_game(_game(), 1, 30, win=True)
    assert len(bg["decisions"]) == 3
    assert bg["decisions"][-1] == {"balance_before": 150}           # 90 + 2*30
    assert len(bg["history"]) == 2
    assert bg["history"][-1] == {"round": 2, "bet": 30, "win": True, "balance": 150}


def test_ic_interface_contract():
    """runner._run_arm_ic depends on these exact ic.py signatures."""
    import inspect

    from multilayer_causal.src import ic
    assert list(inspect.signature(ic.load_ic_states).parameters)[:2] == ["model", "n"]
    assert list(inspect.signature(ic.ensure_ic_catalog).parameters)[:1] == ["model"]
    choice, parse_ok, risky = ic.parse_ic_response("My choice: Option 2")
    assert choice in (1, 2, 3, 4)
    assert isinstance(parse_ok, bool) and isinstance(risky, bool)


def test_all_in_and_overbet_rejected():
    # bet >= balance is a probe-bankruptcy case: caller logs all_in, no branch
    for bet in (110, 200):
        with pytest.raises(AssertionError):
            build_branch_game(_game(), 2, bet, win=False)
    with pytest.raises(AssertionError):
        build_branch_game(_game(), 2, 0, win=True)                  # no zero bets
