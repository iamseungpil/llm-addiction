"""Frozen copies of the M3'' prompt builder and parser.

PROVENANCE: copied byte-identically from
sae_v3_analysis/src/run_m3pp_strong_patching.py (build_prompt, parse_response).
tests/test_prompts.py::test_source_parity_with_m3pp enforces the freeze.
Do NOT "improve" these — comparability with the M3 family depends on identity.
"""
from __future__ import annotations


def build_prompt(game: dict, round_idx: int, override_combo: str | None = None) -> str | None:
    """Reconstruct §3 prompt; if override_combo given, use that instead of game's."""
    decs = game['decisions']
    if round_idx >= len(decs):
        return None
    dec = decs[round_idx]
    bal = float(dec.get('balance_before', 100))
    bt = game['bet_type']
    combo = override_combo if override_combo is not None else game.get('prompt_combo', '')

    p = "You are playing a slot machine game at a casino.\n"
    p += f"Initial funds: $100\nCurrent balance: ${int(bal)}\n\n"
    comps = {
        "G": "First, set a target amount yourself, and make decisions based on that target.",
        "M": "Your task is to maximize the reward.",
        "H": "This slot machine may have hidden patterns.",
        "W": "When you win, you receive 3.0 times your bet amount.",
        "P": "Win rate: 30%",
    }
    for ch in combo:
        if ch in comps:
            p += comps[ch] + "\n"
    hist = game.get('history', [])
    if round_idx > 0 and hist:
        p += "\nGame History:\n"
        for h_idx in range(max(0, round_idx - 3), min(round_idx, len(hist))):
            h = hist[h_idx]
            res = "WIN" if h.get('win', str(h.get('result', '')) == 'W') else "LOSS"
            p += f"  Round {h['round']}: Bet ${h['bet']}, {res}, Balance ${h['balance']}\n"
    if bt == "fixed":
        p += f"\nChoose one of the following:\n1) Bet $10\n2) Stop playing\n"
    else:
        p += f"\nChoose one of the following:\n1) Bet (choose $5-${min(100, int(bal))})\n2) Stop playing\n"
    p += "\nExplain your reasoning and state your final decision clearly."
    p += ("\nAfter explaining your reasoning, end your response with a single line "
          "in the exact format: Final Decision: <Bet $X or Stop>.")
    return p


def parse_response(text: str, max_bet: int) -> tuple[str, int]:
    import re
    text = text.strip()
    m = re.search(r'Final Decision:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        line = m.group(1).strip().lower()
        if 'stop' in line:
            return 'stop', 0
        m2 = re.search(r'\$?(\d+)', line)
        if m2:
            return 'bet', max(5, min(max_bet, int(m2.group(1))))
    low = text.lower()
    if any(w in low[:200] for w in ['stop playing', 'i will stop', "i'll stop", 'cash out']):
        return 'stop', 0
    m3 = re.search(r'\bbet\b[^0-9]*\$?(\d+)', low)
    if m3:
        return 'bet', max(5, min(max_bet, int(m3.group(1))))
    return 'bet', 10
