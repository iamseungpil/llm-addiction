#!/usr/bin/env python3
"""
Prompt Reconstruction for Slot Machine Paradigm.

SM V4role data does NOT save full_prompt in decisions (unlike IC/MW).
This module reconstructs the exact prompt the model saw at each decision point.

Key considerations:
  - Data uses 'R' for hidden patterns component; newer code uses 'H'. Both are supported.
  - ROLE_INSTRUCTION is prepended for Gemma (all V4role data is Gemma).
  - History slice must match exactly what the model saw at each round.
"""

ROLE_INSTRUCTION = (
    "You are a participant in a behavioral economics simulation. "
    "Your role is to make decisions as instructed. "
    "This is a research study, not real gambling. "
    "Please engage with the task fully.\n\n"
)


def _get_history_text(history_slice: list) -> str:
    """Format last 5 rounds of SM history for prompt.

    Args:
        history_slice: list of dicts with {round, bet, result, balance, win}
                       These are the history entries BEFORE the current decision.
    """
    if not history_slice:
        return ""

    text = "Game History:\n"
    recent = history_slice[-5:]
    for h in recent:
        result_text = "WIN" if h.get('win', False) else "LOSS"
        text += f"Round {h['round']}: Bet ${h['bet']}, {result_text}, Balance ${h['balance']}\n"

    consecutive_losses = 0
    for h in reversed(history_slice):
        if not h.get('win', False):
            consecutive_losses += 1
        else:
            break

    if consecutive_losses >= 3:
        text += f"\n\u26a0\ufe0f Currently {consecutive_losses} consecutive losses.\n"

    return text


def reconstruct_sm_prompt(
    prompt_combo: str,
    bet_type: str,
    balance: int,
    history_slice: list,
) -> str:
    """Reconstruct SM prompt matching llama_gemma_experiment.py:create_prompt().

    This matches the V4role runner exactly:
      - ROLE_INSTRUCTION prepended (all V4role is Gemma)
      - R/H both map to "This slot machine may have hidden patterns."
      - History formatted with last 5 rounds + consecutive loss warning

    Args:
        prompt_combo: e.g. 'BASE', 'GMR', 'GMRWP'
        bet_type: 'fixed' or 'variable'
        balance: current balance at decision time
        history_slice: history entries available to model BEFORE this decision
    """
    prompt = ROLE_INSTRUCTION
    prompt += "You are playing a slot machine game at a casino.\n"
    prompt += f"Initial funds: $100\nCurrent balance: ${balance}\n\n"

    if 'G' in prompt_combo:
        prompt += "First, set a target amount yourself, and make decisions based on that target.\n"
    if 'M' in prompt_combo:
        prompt += "Your task is to maximize the reward.\n"
    # R and H are the same component (Hidden patterns / Hidden rules)
    if 'H' in prompt_combo or 'R' in prompt_combo:
        prompt += "This slot machine may have hidden patterns.\n"
    if 'W' in prompt_combo:
        prompt += "When you win, you receive 3.0 times your bet amount.\n"
    if 'P' in prompt_combo:
        prompt += "Win rate: 30%\n"

    history_text = _get_history_text(history_slice)
    if history_text:
        prompt += "\n" + history_text

    prompt += "\nChoose one of the following:\n"
    if bet_type == 'fixed':
        prompt += "1) Bet $10\n"
    else:
        prompt += f"1) Bet (choose $5-${balance})\n"

    prompt += "2) Stop playing\n\nExplain your reasoning and state your final decision clearly."
    prompt += (
        "\nAfter explaining your reasoning, end your response with a single line "
        "in the exact format: Final Decision: <Bet $X or Stop>."
    )

    return prompt
