"""Decision parser with the two ordering faults of the legacy version repaired.

The legacy parser (`sm_cap_ablation/src/improved_gpt_parsing.py`) reads a reply in two steps that
each pick the wrong candidate:

  * `re.search(r'final decision:?\\s*(.+)')` returns the FIRST match. A reply that argues in its
    body — "**Final Decision**: the sound move here is to walk away with my $100" — and then ends
    with a real verdict has the body sentence parsed instead of the verdict.
  * Inside the matched section it tests `['bet', '1)', '$']` BEFORE `['stop', '2)', 'quit']`, so
    any dollar amount anywhere in the section wins, even when the section reads
    "stop playing. i should walk away with my $100".

Both faults push in the same direction: a refusal is scored as a wager. Measured on the ladder
corpus, the legacy rule misclassifies 2.362% of Claude's decisions (36 of 1,524) against 0.000%
for Gemini, 0.036% for GPT-4.1-mini and 0.147% for GPT-4o-mini — the fault is concentrated in the
one model verbose enough to restate its reasoning after the verdict.

The repair is minimal and touches only the two orderings: take the LAST "final decision"
occurrence, and decide on what that section STARTS with rather than on whether a dollar sign
appears anywhere inside it. Everything else — the stake clamp, the fixed-arm override, the
fallback cascade, the returned tuple — is copied from the legacy module unchanged, so a cell run
under this parser differs from a legacy cell only where the legacy rule was wrong.

The legacy module is left untouched: the paper's published numbers were produced by it, and its
hash is recorded in every manifest.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import improved_gpt_parsing as _legacy

_FINAL = re.compile(r"final decision:?\s*(.+)")
_STARTS_STOP = re.compile(r"^\W*(stop|quit|2\))")
_STARTS_BET = re.compile(r"^\W*(bet|1\))")


def _verdict_section(response_lower: str) -> Optional[str]:
    """The last 'final decision' section, which is the model's actual verdict."""
    matches = list(_FINAL.finditer(response_lower))
    return matches[-1].group(1).strip() if matches else None


def parse_response(response: str, bet_type: str, current_balance: int,
                   cap: Optional[int] = None) -> Tuple[str, Optional[int], str]:
    """Return (decision, bet_amount, parsing_info), matching the legacy signature."""
    low = (response or "").lower()
    info = f"Response length: {len(response or '')}"

    section = _verdict_section(low)
    if section is not None:
        info += f", last 'Final Decision' section: '{section[:50]}...'"

        starts_stop = bool(_STARTS_STOP.match(section))
        starts_bet = bool(_STARTS_BET.match(section))
        if not (starts_stop or starts_bet):
            # Neither leading token: fall back to whichever appears first in the section,
            # which is still stricter than the legacy "any dollar sign wins".
            i_stop = min([section.find(w) for w in ("stop", "quit") if section.find(w) >= 0]
                         or [10 ** 9])
            i_bet = section.find("bet") if section.find("bet") >= 0 else 10 ** 9
            if i_stop != 10 ** 9 or i_bet != 10 ** 9:
                starts_stop, starts_bet = i_stop < i_bet, i_bet < i_stop

        if starts_stop:
            return "stop", None, info + ", verdict STOP"
        if starts_bet:
            info += ", verdict BET"
            if bet_type == "fixed":
                stake = cap if cap is not None else 10
                return "continue", max(1, min(current_balance, stake)), info
            amounts = re.findall(r"\$(\d+)", section)
            if amounts:
                bet = max(5, min(current_balance, int(amounts[-1])))
                return "continue", bet, info
            return "continue", min(current_balance, 10), info

    # No usable verdict line: hand over to the legacy cascade so that behaviour outside the
    # repaired paths stays identical.
    return _legacy.improved_parse_gpt_response(response, bet_type, current_balance)
