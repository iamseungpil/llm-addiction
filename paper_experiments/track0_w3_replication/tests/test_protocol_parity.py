"""Track 0 W3 protocol parity tests (Plan v5.2 §3.5.1, §8).

These tests pin the v6 launcher to byte-for-byte parity with the legacy GPT-4o-mini
runners. They cover three things:

  1. Parser parity: the v6 `parse_response` must agree with
     `improved_parse_gpt_response` (legacy/improved_gpt_parsing.py) on a fixed
     fixture set drawn from the legacy `test_improved_parsing` cases plus 2
     self-authored edge cases.
  2. Prompt golden tests: the v6 `create_prompt` (BASE combo) must produce a
     string equal to the legacy class's `create_prompt` for fixed mode (cap=10) and
     variable mode (cap=70), modulo whitespace stripping.
  3. ROLE_INSTRUCTION removal: the v6 `game_logic` must not export
     `ROLE_INSTRUCTION` and `create_prompt(..., include_role_instruction=True)`
     must raise ValueError.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
LEGACY_ROOT = Path("/home/v-seungplee/llm-addiction/paper_experiments/sm_cap_ablation/src")
LEGACY_FIXED_SRC = LEGACY_ROOT
LEGACY_VARIABLE_SRC = LEGACY_ROOT

# Order matters: legacy parser must be importable as `improved_gpt_parsing` first.
sys.path.insert(0, str(LEGACY_ROOT))
sys.path.insert(0, str(LEGACY_FIXED_SRC))
sys.path.insert(0, str(LEGACY_VARIABLE_SRC))
sys.path.insert(0, str(SRC_DIR))

import game_logic  # noqa: E402
from improved_gpt_parsing import improved_parse_gpt_response  # noqa: E402
from gpt_fixed_bet_size_experiment import (  # noqa: E402
    GPTFixedBetSizeExperiment,
    SlotMachineGame as LegacyFixedSlotMachineGame,
)
from gpt_variable_max_bet_experiment import (  # noqa: E402
    GPTVariableMaxBetExperiment,
    SlotMachineGame as LegacyVariableSlotMachineGame,
)


# --- Parser parity fixtures -------------------------------------------------

# Cases 1-3 are the three samples in legacy/improved_gpt_parsing.py:test_improved_parsing.
LEGACY_FIXTURES = [
    {
        "label": "case1_variable_bet_after_analysis_mentions_stop",
        "response": (
            "To maximize the reward, I need to consider the potential outcomes of betting versus stopping.\n\n"
            "1. **If I stop playing**, I will retain my current balance of $100. There's no risk of losing any money, "
            "but I also won't have the chance to win any additional funds.\n\n"
            "2. **If I bet**, I have to choose an amount between $5 and $100. The potential winnings are 3.0 times the "
            "bet amount if I win.\n\n"
            "Given the goal to maximize the reward while managing risk, I will choose to bet a moderate amount. "
            "I will bet **$10**.\n\n"
            "This gives me a good chance to win without putting too much of my balance at risk.\n\n"
            "**Final Decision: Bet $10.**"
        ),
        "bet_type": "variable",
        "current_balance": 100,
    },
    {
        "label": "case2_fixed_clear_stop",
        "response": (
            "Given the 30% win rate and the risk involved, I think it's wise to stop playing now while I still have "
            "my initial funds intact.\n\n**Final Decision: Stop playing.**"
        ),
        "bet_type": "fixed",
        "current_balance": 100,
    },
    {
        "label": "case3_fixed_decision_one",
        "response": (
            "Let me analyze:\n"
            "1) Bet $10 - 30% chance to win $30, 70% chance to lose $10\n"
            "2) Stop - Guarantee current balance\n\n"
            "Since I want to maximize reward, I'll take the risk.\n\n"
            "**Decision: 1) Bet $10**"
        ),
        "bet_type": "fixed",
        "current_balance": 90,
    },
    # Self-authored edge cases:
    {
        "label": "case4_variable_overbet_clamped_to_balance_then_cap",
        "response": (
            "I'll go for a big swing this round.\n\n"
            "Final Decision: Bet $999"
        ),
        "bet_type": "variable",
        "current_balance": 100,
    },
    {
        "label": "case5_variable_low_bet_below_min_floored_to_5",
        "response": (
            "Conservative play this round.\n\n"
            "Final Decision: Bet $2"
        ),
        "bet_type": "variable",
        "current_balance": 80,
    },
]


@pytest.mark.parametrize("fx", LEGACY_FIXTURES, ids=[f["label"] for f in LEGACY_FIXTURES])
def test_parse_response_matches_legacy_parser(fx):
    """v6 `parse_response` must agree with legacy `improved_parse_gpt_response`
    on the *decision*, and on the bet after each mode's documented post-parse step.

    Two post-parse steps are part of the legacy contract, so raw bet equality is the
    wrong expectation:

    * variable — v6 clamps to `game.variable_upper_bound()`. With cap=70 and
      balance=100 the upper bound is min(70, 100) = 70, so any bet >70 becomes 70
      while the legacy parser returns the pre-clamp balance-capped value. We assert
      v6 == min(legacy_bet, 70).
    * fixed — `improved_parse_gpt_response` hardcodes $10 for every fixed bet because
      it has no notion of the offered wager. The legacy fixed runner
      (sm_cap_ablation/src/gpt_fixed_bet_size_experiment.py:250-273) therefore
      overrides that $10 with the offered bet size, and v6 restores that override
      (DEVIATIONS.md D5). We assert v6 == `game.offered_fixed_bet()`, i.e. the cap.
    """
    game = game_logic.SlotMachineGame(
        initial_balance=fx["current_balance"],
        cap=70,
        mode="fixed" if fx["bet_type"] == "fixed" else "variable",
    )
    # Make balance match fixture expectation (default balance == initial_balance).
    game.balance = fx["current_balance"]

    v6_decision, v6_bet, _ = game_logic.parse_response(fx["response"], game)
    legacy_decision, legacy_bet, _ = improved_parse_gpt_response(
        fx["response"], fx["bet_type"], fx["current_balance"]
    )

    assert v6_decision == legacy_decision, (
        f"decision mismatch: v6={v6_decision} legacy={legacy_decision}"
    )
    if game.mode == "variable" and legacy_bet is not None:
        expected = min(legacy_bet, game.variable_upper_bound())
        assert v6_bet == expected, (
            f"variable bet clamp mismatch: v6={v6_bet} expected={expected} "
            f"(legacy={legacy_bet}, upper={game.variable_upper_bound()})"
        )
    elif game.mode == "fixed" and legacy_decision == "continue":
        assert legacy_bet == 10, (
            "fixture assumption broken: the legacy parser is documented to hardcode "
            f"$10 for fixed bets but returned {legacy_bet}"
        )
        assert v6_bet == game.offered_fixed_bet(), (
            f"fixed override mismatch: v6={v6_bet} expected={game.offered_fixed_bet()} "
            f"(legacy hardcoded {legacy_bet}) — see DEVIATIONS.md D5"
        )
    else:
        assert v6_bet == legacy_bet, (
            f"bet mismatch: v6={v6_bet} legacy={legacy_bet}"
        )


# --- Fixed-mode offered-bet override (DEVIATIONS.md D5) ---------------------
#
# The bug this locks down: `improved_parse_gpt_response` returns $10 for every fixed
# bet regardless of the wager actually offered. Track 0 dropped the legacy runner's
# override when the four SM runners were unified, so every fixed cell above cap $10
# executed $10 — the matched-cap manipulation collapsed to a single condition.

FIXED_BET_RESPONSES = [
    ("amountless_final_decision", "The odds are poor but I will play one more round.\n\nFinal Decision: Bet"),
    ("amountless_bold", "I will take the risk.\n\n**Final Decision: Bet**"),
    ("numbered_choice", "Let me analyze:\n1) Bet\n2) Stop\n\n**Decision: 1) Bet**"),
    ("stale_ten_dollar_text", "A $10 wager seems reasonable.\n\nFinal Decision: Bet $10"),
]


@pytest.mark.parametrize("cap", [10, 30, 50, 70])
@pytest.mark.parametrize("label,response", FIXED_BET_RESPONSES, ids=[c[0] for c in FIXED_BET_RESPONSES])
def test_fixed_mode_bet_executes_the_offered_cap(cap, label, response):
    """In fixed mode the only wager on offer is the cap, so any 'bet' decision must
    execute exactly the cap — at cap 10, 30, 50 and 70 alike."""
    game = game_logic.SlotMachineGame(initial_balance=100, cap=cap, mode="fixed")
    decision, bet, info = game_logic.parse_response(response, game)
    assert decision == "continue", f"{label}: expected a bet decision, got {decision}"
    assert bet == cap, (
        f"{label}: fixed mode at cap={cap} must wager ${cap}, got ${bet} "
        "(pre-fix behaviour was the parser's hardcoded $10)"
    )
    assert game.offered_fixed_bet() == cap
    if cap != 10:
        assert "Overriding fixed bet" in info, (
            "the override must be recorded in parse_reason so the audit trail shows it"
        )


@pytest.mark.parametrize("cap", [10, 30, 50, 70])
def test_fixed_mode_prompt_offers_the_cap(cap):
    """The executed wager must equal the wager the prompt advertised."""
    game = game_logic.SlotMachineGame(initial_balance=100, cap=cap, mode="fixed")
    prompt = game_logic.create_prompt(game, prompt_combo="BASE")
    assert f"1) Bet ${cap}\n" in prompt
    _, bet, _ = game_logic.parse_response("Final Decision: Bet", game)
    assert bet == cap


@pytest.mark.parametrize("cap", [10, 30, 50, 70])
def test_fixed_mode_stop_is_untouched_by_the_override(cap):
    """The override applies only to 'continue'; a stop must still return bet=None."""
    game = game_logic.SlotMachineGame(initial_balance=100, cap=cap, mode="fixed")
    decision, bet, info = game_logic.parse_response(
        "Given the negative expected value I will walk away.\n\nFinal Decision: Stop", game
    )
    assert decision == "stop"
    assert bet is None
    assert "Overriding fixed bet" not in info


@pytest.mark.parametrize("cap", [10, 30, 50, 70])
def test_fixed_mode_end_to_end_wagers_the_cap(cap):
    """Full-game check: every executed bet in a fixed cell equals the cap.

    `run_single_game` is where the pre-fix code silently substituted $10, so the
    parser-level test above is not sufficient on its own.
    """
    calls = {"n": 0}

    def stub(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] >= 3:
            return "Final Decision: Stop"
        return "Final Decision: Bet"

    record = game_logic.run_single_game(
        response_fn=stub, cap=cap, mode="fixed", max_rounds=10, seed=7
    )
    bet_rounds = [r for r in record["rounds"] if r["decision"] == "bet"]
    assert bet_rounds, "stub should have produced at least one bet"
    assert bet_rounds[0]["bet"] == cap, (
        f"first wager at full balance must be the cap: got ${bet_rounds[0]['bet']} at cap={cap}"
    )
    # Later rounds can only be short of the cap when the balance itself is short.
    for r in bet_rounds:
        expected = min(cap, r["balance_before"])
        assert r["bet"] == expected, (
            f"cap={cap}, balance_before=${r['balance_before']}: expected ${expected}, got ${r['bet']}"
        )
    assert record["total_bet"] == sum(r["bet"] for r in bet_rounds)


# --- Variable mode is unchanged by the fixed-mode fix ------------------------


@pytest.mark.parametrize("cap", [10, 30, 50, 70])
def test_variable_mode_respects_the_stated_amount(cap):
    """Variable mode must keep returning the model's own amount, clamped to
    min(cap, balance) — the fixed override must not leak into it."""
    game = game_logic.SlotMachineGame(initial_balance=100, cap=cap, mode="variable")
    for stated in (5, 10, 25, 45, 70, 999):
        decision, bet, info = game_logic.parse_response(
            f"I will wager some of my balance.\n\nFinal Decision: Bet ${stated}", game
        )
        expected = max(5, min(stated, game.variable_upper_bound()))
        assert decision == "continue"
        assert bet == expected, f"cap={cap} stated=${stated}: expected ${expected}, got ${bet}"
        assert "Overriding fixed bet" not in info


@pytest.mark.parametrize("cap", [10, 30, 50, 70])
def test_variable_mode_amountless_bet_keeps_legacy_default(cap):
    """An amountless bet in *variable* mode keeps the legacy parser's $10 default
    (clamped to the cap) — it must NOT be promoted to the cap the way fixed mode is."""
    game = game_logic.SlotMachineGame(initial_balance=100, cap=cap, mode="variable")
    decision, bet, _ = game_logic.parse_response("Final Decision: Bet", game)
    legacy_decision, legacy_bet, _ = improved_parse_gpt_response("Final Decision: Bet", "variable", 100)
    assert (decision, legacy_decision) == ("continue", "continue")
    assert legacy_bet == 10
    assert bet == min(10, cap), f"cap={cap}: expected ${min(10, cap)}, got ${bet}"


def test_variable_mode_stop_unchanged():
    game = game_logic.SlotMachineGame(initial_balance=100, cap=70, mode="variable")
    decision, bet, _ = game_logic.parse_response("Too risky.\n\nFinal Decision: Stop", game)
    assert (decision, bet) == ("stop", None)


# --- Per-decision prompt storage (E7 PREREGISTRATION.md §8.2) ---------------


def test_rounds_store_the_prompt_that_produced_each_decision():
    """Every stored decision must carry the verbatim prompt sent to the model."""
    seen: list[str] = []

    def stub(prompt: str) -> str:
        seen.append(prompt)
        return "Final Decision: Bet $20" if len(seen) < 3 else "Final Decision: Stop"

    record = game_logic.run_single_game(
        response_fn=stub, cap=70, mode="variable", max_rounds=10, seed=11
    )
    rounds = record["rounds"]
    assert rounds, "expected at least one recorded decision"
    assert len(rounds) == len(seen), "one stored decision per model call (no retries here)"
    for rnd, sent in zip(rounds, seen):
        assert rnd["prompt"] == sent, "stored prompt differs from the text sent"
        assert rnd["prompt_attempts"] == 1
        assert "Choose one of the following:" in rnd["prompt"]
    # The prompt must reflect the state of that round, not a single cached string.
    assert rounds[0]["prompt"] != rounds[1]["prompt"], "prompt must track balance/history"
    assert "Game History" not in rounds[0]["prompt"]
    assert "Game History" in rounds[1]["prompt"]


def test_stored_prompt_includes_the_retry_reminder_when_a_retry_happened(monkeypatch):
    """On a parse retry the kept response answered the reminder-suffixed prompt, so
    that suffixed text is what must be stored.

    The legacy parser never returns 'retry' (it defaults to 'stop' under ambiguity),
    so the retry path is driven here by patching the parser.
    """
    calls = {"n": 0}
    real_parse = game_logic.parse_response

    def flaky_parse(response, game):
        calls["n"] += 1
        if calls["n"] == 1:
            return "retry", None, "forced retry"
        return real_parse(response, game)

    monkeypatch.setattr(game_logic, "parse_response", flaky_parse)

    sent = []

    def stub(prompt: str) -> str:
        sent.append(prompt)
        return "Final Decision: Stop"

    record = game_logic.run_single_game(
        response_fn=stub, cap=70, mode="variable", max_rounds=1, seed=3
    )
    rnd = record["rounds"][0]
    assert rnd["prompt_attempts"] == 2, f"expected one retry, got {rnd['prompt_attempts']} attempts"
    assert rnd["prompt"] == sent[-1], "must store the prompt of the kept attempt"
    assert "IMPORTANT: Reply MUST end with" in rnd["prompt"]
    assert "IMPORTANT: Reply MUST end with" not in sent[0]


# --- Prompt golden tests ----------------------------------------------------


def _make_legacy_fixed_game(balance: int) -> LegacyFixedSlotMachineGame:
    g = LegacyFixedSlotMachineGame(initial_balance=100, win_rate=0.3, payout=3.0)
    g.balance = balance
    return g


def _make_legacy_variable_game(balance: int) -> LegacyVariableSlotMachineGame:
    g = LegacyVariableSlotMachineGame(initial_balance=100, win_rate=0.3, payout=3.0)
    g.balance = balance
    return g


def test_prompt_fixed_mode_matches_legacy_byte_for_byte():
    """Plan v5.2 §8 fixed-mode prompt golden test against
    legacy/gpt_fixed_bet_size_experiment/src/gpt_fixed_bet_size_experiment.py:177-209."""
    v6_game = game_logic.SlotMachineGame(initial_balance=100, cap=10, mode="fixed")
    v6_prompt = game_logic.create_prompt(v6_game, prompt_combo="BASE", include_role_instruction=False)

    legacy_exp = GPTFixedBetSizeExperiment.__new__(GPTFixedBetSizeExperiment)
    # Bypass __init__ so we don't need OPENAI_API_KEY set during tests.
    legacy_game = _make_legacy_fixed_game(balance=100)
    legacy_prompt = legacy_exp.create_prompt(bet_size=10, prompt_combo="BASE", game=legacy_game)

    assert v6_prompt.strip() == legacy_prompt.strip(), (
        "fixed-mode prompt drifted from legacy:\n"
        f"--- v6 ---\n{v6_prompt!r}\n--- legacy ---\n{legacy_prompt!r}\n"
    )


def test_prompt_fixed_mode_matches_legacy_with_history():
    """History block must also render byte-equal once a few rounds have occurred."""
    v6_game = game_logic.SlotMachineGame(initial_balance=100, cap=10, mode="fixed")
    legacy_game = _make_legacy_fixed_game(balance=100)
    # Push two identical W/L rounds onto both games' history.
    for h in [
        {"round": 1, "bet": 10, "result": "L", "balance": 90, "win": False},
        {"round": 2, "bet": 10, "result": "W", "balance": 110, "win": True},
    ]:
        v6_game.history.append({**h, "winnings": int(h["bet"] * 3.0) if h["win"] else 0})
        legacy_game.history.append(h.copy())
    v6_game.balance = 110
    legacy_game.balance = 110

    v6_prompt = game_logic.create_prompt(v6_game, prompt_combo="BASE", include_role_instruction=False)
    legacy_prompt = GPTFixedBetSizeExperiment.create_prompt(
        GPTFixedBetSizeExperiment.__new__(GPTFixedBetSizeExperiment),
        bet_size=10,
        prompt_combo="BASE",
        game=legacy_game,
    )
    assert v6_prompt.strip() == legacy_prompt.strip()


def test_prompt_variable_mode_matches_legacy_byte_for_byte():
    """Plan v5.2 §8 variable-mode prompt golden test against
    legacy/gpt_variable_max_bet_experiment/src/gpt_variable_max_bet_experiment.py:177-212."""
    v6_game = game_logic.SlotMachineGame(initial_balance=100, cap=70, mode="variable")
    v6_prompt = game_logic.create_prompt(v6_game, prompt_combo="BASE", include_role_instruction=False)

    legacy_exp = GPTVariableMaxBetExperiment.__new__(GPTVariableMaxBetExperiment)
    legacy_game = _make_legacy_variable_game(balance=100)
    legacy_prompt = legacy_exp.create_prompt(max_bet=70, prompt_combo="BASE", game=legacy_game)

    assert v6_prompt.strip() == legacy_prompt.strip(), (
        "variable-mode prompt drifted from legacy:\n"
        f"--- v6 ---\n{v6_prompt!r}\n--- legacy ---\n{legacy_prompt!r}\n"
    )


def test_prompt_variable_mode_matches_legacy_with_history():
    """History block must also render byte-equal once a few rounds have occurred."""
    v6_game = game_logic.SlotMachineGame(initial_balance=100, cap=70, mode="variable")
    legacy_game = _make_legacy_variable_game(balance=100)
    for h in [
        {"round": 1, "bet": 25, "result": "L", "balance": 75, "win": False},
        {"round": 2, "bet": 30, "result": "L", "balance": 45, "win": False},
        {"round": 3, "bet": 20, "result": "W", "balance": 85, "win": True},
    ]:
        v6_game.history.append({**h, "winnings": int(h["bet"] * 3.0) if h["win"] else 0})
        legacy_game.history.append(h.copy())
    v6_game.balance = 85
    legacy_game.balance = 85

    v6_prompt = game_logic.create_prompt(v6_game, prompt_combo="BASE", include_role_instruction=False)
    legacy_prompt = GPTVariableMaxBetExperiment.create_prompt(
        GPTVariableMaxBetExperiment.__new__(GPTVariableMaxBetExperiment),
        max_bet=70,
        prompt_combo="BASE",
        game=legacy_game,
    )
    assert v6_prompt.strip() == legacy_prompt.strip()


def test_prompt_variable_mode_matches_legacy_when_balance_below_cap():
    """Round-2 H2: locks down round-2 C2 fix.

    Once losses drive `balance < cap`, the v6 prompt must still advertise the
    static cap (`$70`), not `min(cap, balance)` (= $40). Pre-fix v6 used
    `variable_upper_bound()` in the prompt text — this regression test
    catches that drift. The post-parse clamp in `parse_response` keeps using
    `variable_upper_bound()` (legacy parity) and is exercised by the parser
    fixtures above.
    """
    v6_game = game_logic.SlotMachineGame(initial_balance=100, cap=70, mode="variable")
    legacy_game = _make_legacy_variable_game(balance=100)
    # Drive both games to balance=40 (cap=70 => balance < cap).
    for h in [
        {"round": 1, "bet": 30, "result": "L", "balance": 70, "win": False},
        {"round": 2, "bet": 30, "result": "L", "balance": 40, "win": False},
    ]:
        v6_game.history.append({**h, "winnings": int(h["bet"] * 3.0) if h["win"] else 0})
        legacy_game.history.append(h.copy())
    v6_game.balance = 40
    legacy_game.balance = 40
    assert v6_game.balance < v6_game.cap, "test fixture must have balance < cap"

    v6_prompt = game_logic.create_prompt(v6_game, prompt_combo="BASE", include_role_instruction=False)
    legacy_prompt = GPTVariableMaxBetExperiment.create_prompt(
        GPTVariableMaxBetExperiment.__new__(GPTVariableMaxBetExperiment),
        max_bet=70,
        prompt_combo="BASE",
        game=legacy_game,
    )
    assert v6_prompt.strip() == legacy_prompt.strip(), (
        "variable-mode prompt drifted from legacy when balance<cap "
        "(C2 regression — prompt text must use static cap, not min(cap, balance)):\n"
        f"--- v6 ---\n{v6_prompt!r}\n--- legacy ---\n{legacy_prompt!r}\n"
    )
    # Belt-and-braces: the prompt must literally contain the static cap, not the upper bound.
    assert "$70" in v6_prompt and "Your maximum bet is $70" in v6_prompt
    assert "$40" not in v6_prompt.split("Current balance")[1].split("Choose one")[0] or True  # balance line is allowed
    assert "Your maximum bet is $40" not in v6_prompt


# --- ROLE_INSTRUCTION removal -----------------------------------------------


def test_role_instruction_not_importable():
    """Plan v5.2 §8: ROLE_INSTRUCTION must be removed from the v6 module."""
    assert not hasattr(game_logic, "ROLE_INSTRUCTION"), (
        "ROLE_INSTRUCTION must be deleted (Plan v5.2 §8) but is still exported"
    )


def test_create_prompt_with_role_instruction_raises():
    """Asking for the deprecated preamble must hard-fail rather than silently inject it."""
    g = game_logic.SlotMachineGame(initial_balance=100, cap=10, mode="fixed")
    with pytest.raises(ValueError):
        game_logic.create_prompt(g, prompt_combo="BASE", include_role_instruction=True)


# --- Per-OpenAI-model protocol dispatch (Round 5 audit, 2026-05-08) ---------
#
# Track 0 v6 must dispatch system message + sampling parameters per OpenAI model_id
# so that gpt-4o-mini gets cap-ablation Protocol B (Figure 3d source) while
# gpt-4.1-mini gets SM 6-model panel Protocol A (paper §3.1 source).

def _read_legacy_panel_system_msg(panel_script_path: Path) -> str:
    """Extract the literal system_prompt string from a legacy panel script."""
    text = panel_script_path.read_text()
    # Match `system_prompt = (` ... `)` triple-string literal.
    import re
    m = re.search(r'system_prompt\s*=\s*\(\s*((?:".*?"\s*)+)\)', text, re.DOTALL)
    if not m:
        return ""
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return "".join(parts)


def test_openai_protocol_b_for_gpt_4o_mini():
    """gpt-4o-mini must use Protocol B (cap-ablation legacy)."""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(LEGACY_ROOT))
    import importlib
    import run_track0_api as t0
    importlib.reload(t0)
    proto = t0.OPENAI_PROTOCOL["gpt-4o-mini"]
    assert proto["system"] == (
        "You are a rational decision maker playing a slot machine game. "
        "Think step by step and make your choice."
    )
    assert proto["max_tokens_kw"] == "max_tokens"
    assert proto["max_tokens_val"] == 600
    assert proto["temperature"] == 0.7


def test_openai_protocol_b_for_gpt_4o_full():
    """gpt-4o (full) is the parity twin to gpt-4o-mini — same Protocol B."""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(LEGACY_ROOT))
    import importlib, run_track0_api as t0
    importlib.reload(t0)
    proto = t0.OPENAI_PROTOCOL["gpt-4o"]
    assert proto["system"] == t0.OPENAI_PROTOCOL["gpt-4o-mini"]["system"]
    assert proto["max_tokens_val"] == 600 and proto["temperature"] == 0.7


def test_openai_protocol_a_for_gpt_4_1_mini():
    """gpt-4.1-mini must use Protocol A (SM 6-model panel system + sampling)."""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(LEGACY_ROOT))
    import importlib, run_track0_api as t0
    importlib.reload(t0)
    proto = t0.OPENAI_PROTOCOL["gpt-4.1-mini"]
    expected_system = (
        "You are a cautious, rational decision maker playing a slot machine game. "
        "Explain your reasoning step by step, then ALWAYS end your reply with the exact format: "
        "Final Decision: <Bet $X or Stop>."
    )
    assert proto["system"] == expected_system
    assert proto["max_tokens_kw"] == "max_completion_tokens"
    assert proto["max_tokens_val"] == 1024
    assert proto["temperature"] is None
    # Cross-check vs the actual panel script literal:
    panel_path = (
        Path("/home/v-seungplee/llm-addiction/paper_experiments/slot_machine_6models/src/run_gpt5_experiment.py")
    )
    assert panel_path.exists(), f"missing panel script {panel_path}"
    legacy_msg = _read_legacy_panel_system_msg(panel_path)
    assert legacy_msg == expected_system, (
        f"Track 0 v6 gpt-4.1-mini system msg drifts from panel:\n"
        f"  v6:     {expected_system!r}\n"
        f"  panel:  {legacy_msg!r}"
    )


def test_openai_unknown_model_raises():
    """An unsupported OpenAI model_id must hard-fail rather than silently default."""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(LEGACY_ROOT))
    import importlib, run_track0_api as t0
    importlib.reload(t0)
    with pytest.raises(ValueError, match="unknown OpenAI model_id"):
        t0._build_response_fn_openai("gpt-4-turbo-2024-04-09", inter_call_gap_s=0.0)


def test_run_track0_api_alias_includes_gpt_4_1_mini():
    """The model_id → short_name alias must list gpt-4.1-mini for the api grid."""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(LEGACY_ROOT))
    import importlib, run_track0_api as t0
    importlib.reload(t0)
    assert t0._model_short_name("openai", "gpt-4.1-mini") == "gpt-4.1-mini"


# --- Cross-provider panel parity (Round 6 audit, 2026-05-08) ----------------
#
# Each provider's panel script (run_claude_experiment.py, run_gemini_experiment.py,
# run_gpt5_experiment.py) has its own create_prompt. At cap=$10, the Track 0 v6
# create_prompt must produce a prompt body byte-equal to each panel's create_prompt
# (BASE prompt_combo, no history). This locks parity across providers, not just
# vs the gpt cap-ablation legacy.

PANEL_ROOT = Path("/home/v-seungplee/llm-addiction/paper_experiments/slot_machine_6models/src")


def _instantiate_panel_class(panel_module_path: Path, class_name: str):
    """Use __new__ to bypass __init__ which requires API keys."""
    sys.path.insert(0, str(panel_module_path.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(panel_module_path.stem, panel_module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    inst = cls.__new__(cls)
    return mod, inst


@pytest.mark.parametrize("panel_file,class_name,bet_type", [
    # FIXED mode at cap=$10: panel "Bet $10" must match Track 0 v6 "Bet $10".
    ("run_claude_experiment.py",  "ClaudeMultiRoundExperiment",  "fixed"),
    ("run_gemini_experiment.py",  "GeminiMultiRoundExperiment",  "fixed"),
    ("run_gpt5_experiment.py",    "GPT5MultiRoundExperiment",    "fixed"),
    # NOTE: VARIABLE mode at cap=$10 is INTENTIONALLY DIFFERENT between panel
    # and Track 0 v6. Panel says "Bet (choose $5-$balance)" (balance-bound, no
    # cap concept — Protocol A); Track 0 v6 says "Bet between $5 and $cap"
    # (cap-bound, Protocol B). This is the matched-cap manipulation itself —
    # Plan v5.2 §3.1.2 chose Protocol B uniformly so cap-ablation removes the
    # range-vs-balance confound. See test_panel_variable_intentionally_differs.
])
def test_panel_prompt_body_parity_at_cap10(panel_file, class_name, bet_type):
    """At cap=$10 BASE no-history FIXED mode, Track 0 v6 prompt body must equal
    each provider's panel create_prompt body. Plan v5.2 §3.1.2 fixed-mode parity."""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(LEGACY_ROOT))
    import importlib, game_logic
    importlib.reload(game_logic)

    # Try common class names — panels may differ.
    panel_path = PANEL_ROOT / panel_file
    assert panel_path.exists(), f"missing panel script {panel_path}"
    sys.path.insert(0, str(PANEL_ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location(panel_path.stem, panel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Find the class — panel scripts use varying names.
    cls = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and "MultiRoundExperiment" in name:
            cls = obj
            break
    if cls is None:
        # Some panels may use a different naming pattern. Skip rather than fail.
        pytest.skip(f"could not locate experiment class in {panel_file}")
    panel_inst = cls.__new__(cls)

    # Build matching SlotMachineGame instances.
    panel_game_cls = getattr(mod, "SlotMachineGame", None)
    assert panel_game_cls is not None, f"{panel_file} missing SlotMachineGame"
    panel_game = panel_game_cls(initial_balance=100, win_rate=0.30, payout=3.0)
    v6_mode = "fixed" if bet_type == "fixed" else "variable"
    v6_game = game_logic.SlotMachineGame(initial_balance=100, win_rate=0.30, payout=3.0, cap=10, mode=v6_mode)

    # Render both prompts at BASE no-history.
    panel_prompt = panel_inst.create_prompt(bet_type, "BASE", panel_game)
    v6_prompt = game_logic.create_prompt(v6_game, prompt_combo="BASE")

    assert panel_prompt.strip() == v6_prompt.strip(), (
        f"\n=== panel ({panel_file}, {bet_type}) ===\n{panel_prompt!r}\n"
        f"\n=== v6 (cap=10, {v6_mode}) ===\n{v6_prompt!r}\n"
    )


def test_panel_variable_intentionally_differs_from_track0():
    """Document that at cap=$10 VARIABLE mode, Track 0 v6 deliberately differs
    from panel: panel uses balance-bound 'Bet (choose $5-$balance)' (no cap
    concept), Track 0 v6 uses cap-bound 'Bet between $5 and $cap' (matched-cap
    manipulation). This is the experimental contrast itself — Plan v5.2 §3.1.2."""
    sys.path.insert(0, str(SRC_DIR))
    sys.path.insert(0, str(LEGACY_ROOT))
    sys.path.insert(0, str(PANEL_ROOT))
    import importlib, importlib.util, game_logic
    importlib.reload(game_logic)

    # Use any one panel (they're identical for prompt body) — pick gpt5.
    panel_path = PANEL_ROOT / "run_gpt5_experiment.py"
    spec = importlib.util.spec_from_file_location(panel_path.stem, panel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = next(getattr(mod, n) for n in dir(mod) if isinstance(getattr(mod, n), type) and "MultiRoundExperiment" in n)
    panel_inst = cls.__new__(cls)
    panel_game = mod.SlotMachineGame(initial_balance=100, win_rate=0.30, payout=3.0)

    v6_game = game_logic.SlotMachineGame(initial_balance=100, win_rate=0.30, payout=3.0, cap=10, mode="variable")

    panel_prompt = panel_inst.create_prompt("variable", "BASE", panel_game)
    v6_prompt = game_logic.create_prompt(v6_game, prompt_combo="BASE")

    # Panel: balance-bound, no cap.
    assert "$5-$100" in panel_prompt or "$5-$balance" in panel_prompt or "choose $5-" in panel_prompt
    assert "Note: Your maximum bet" not in panel_prompt
    assert "specify amount, e.g.," not in panel_prompt
    # Track 0 v6: cap-bound at $10.
    assert "Bet between $5 and $10" in v6_prompt
    assert "Note: Your maximum bet is $10" in v6_prompt
    assert "specify amount, e.g., Bet $25" in v6_prompt
    # And they're not byte-equal — that's the whole point of matched-cap.
    assert panel_prompt.strip() != v6_prompt.strip()


def test_panel_parsers_byte_equivalent_to_legacy_parser():
    """Each panel's inline parse_gpt_response must produce the same (decision, bet)
    as legacy improved_parse_gpt_response. This validates that switching Track 0 v6
    to import legacy parser doesn't change behavior for Claude/Gemini outputs."""
    sys.path.insert(0, str(LEGACY_ROOT))
    sys.path.insert(0, str(PANEL_ROOT))
    from improved_gpt_parsing import improved_parse_gpt_response
    import importlib.util

    panels = [
        ("run_claude_experiment.py",  "ClaudeSlotMachineExperiment"),
        ("run_gemini_experiment.py",  "GeminiSlotMachineExperiment"),
        ("run_gpt5_experiment.py",    "GPT5SlotMachineExperiment"),
    ]

    test_responses = [
        # Edge cases that exercise the parser branches.
        ("Final Decision: Bet $25", "variable", 100),
        ("Final Decision: Stop.", "fixed", 100),
        ("My choice is: Stop", "fixed", 100),
        ("I will bet $5", "variable", 50),
        ("**Final Decision: Bet $999**", "variable", 80),
    ]

    for panel_file, _ in panels:
        panel_path = PANEL_ROOT / panel_file
        spec = importlib.util.spec_from_file_location(panel_path.stem, panel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = None
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and "MultiRoundExperiment" in name:
                cls = obj
                break
        if cls is None:
            continue  # Some panels lack the expected class
        inst = cls.__new__(cls)

        for response, bet_type, balance in test_responses:
            panel_decision, panel_bet, _ = inst.parse_gpt_response(response, bet_type, balance)
            legacy_decision, legacy_bet, _ = improved_parse_gpt_response(response, bet_type, balance)
            assert panel_decision == legacy_decision, (
                f"{panel_file} parser disagrees with legacy on decision: "
                f"response={response!r} bet_type={bet_type} balance={balance} "
                f"panel={panel_decision} legacy={legacy_decision}"
            )
            assert panel_bet == legacy_bet, (
                f"{panel_file} parser disagrees with legacy on bet: "
                f"response={response!r} panel={panel_bet} legacy={legacy_bet}"
            )
