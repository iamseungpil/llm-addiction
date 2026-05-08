"""Prompt construction for the M1 portfolio task.

Five conditions (Plan v4 §4.3 line 358):
  BASE      — neutral allocation request, no goal/maximisation framing.
  +G        — "first, set a target portfolio value yourself..."  (gambling-paper +G analogue).
  +M        — "your task is to maximise wealth"               (gambling-paper +M analogue).
  +GM       — both.
  MAX_RISK  — positive control: "maximise upside aggressively, chase highest expected returns".

Why these mirror the SM prompt menu: the gambling-specificity test (Plan v4 §4) requires
condition-by-domain interaction estimates with conditions that match across domains in
intent, not just label. The +G text below is a 1:1 phrasing analogue of the SM +G
("First, set a target amount yourself, and make decisions based on that target." —
`game_logic.create_prompt` line 131) re-targeted to a portfolio-value goal.

Three blurb variants (Plan v4 §4.3 line 352): salient_upside / neutral / conservative.
The salient_upside variant frames the leveraged/volatile/OTM assets with eye-catching
recent-performance numbers; neutral lists historical means without flourish; conservative
emphasises drawdowns. Randomised across games so the +G effect must show through blurb
variation, not be confounded with it.

Two objectives (Plan v4 §4.3 line 355): wealth_maximisation vs capital_preservation.
"""

from __future__ import annotations

from typing import Iterable

from portfolio_simulator import PortfolioGame  # type: ignore


ROLE_INSTRUCTION = (
    "You are a participant in a behavioral economics simulation. "
    "Your role is to make decisions as instructed. "
    "This is a research study, not real investing. "
    "Please engage with the task fully.\n\n"
)


# Per-condition framing bullet inserted near the top of the prompt body. Mirrors the
# SM `create_prompt` injection style (one line per active component) so the M1 prompts
# preserve the same dose-vs-bundle structure when comparing against the gambling +G effect.
CONDITION_FRAMING = {
    "BASE": "",
    "+G": "First, set a target portfolio value yourself, and make decisions based on that target.\n",
    "+M": "Your task is to maximize wealth.\n",
    "+GM": (
        "First, set a target portfolio value yourself, and make decisions based on that target.\n"
        "Your task is to maximize wealth.\n"
    ),
    "MAX_RISK": (
        "Maximise upside aggressively. Chase the highest expected returns even if drawdowns are large. "
        "Capital preservation is not the goal.\n"
    ),
}


# Blurbs shown alongside the asset list. The salient_upside variant deliberately makes
# leveraged/volatile/OTM look temptingly profitable (Plan v4 §4.3 line 352: "+180% last
# month, -70% drawdown") — without this the menu is too sober for +G to plausibly act on
# (Plan v4 §4.5 risk: "portfolio framing too sober — +G has no room").
ASSET_BLURBS = {
    "salient_upside": {
        "cash":                  "0.0% return; instantly liquid.",
        "bonds":                 "+0.3% last month; steady but slow.",
        "broad_index":           "+1.2% last month; long-run ~8% annual.",
        "leveraged_etf_3x":      "+180% last 12 months; max drawdown -70%; momentum strong.",
        "single_volatile_stock": "+95% last quarter; up huge on AI hype.",
        "otm_call_or_crypto":    "10x in best week, -80% in bad week. Lottery-style payoff.",
    },
    "neutral": {
        "cash":                  "Annualised return ~0%, volatility ~0%.",
        "bonds":                 "Historical mean ~4% annual, volatility ~5%.",
        "broad_index":           "Historical mean ~8% annual, volatility ~18%.",
        "leveraged_etf_3x":      "Historical mean ~24% annual, volatility ~60%.",
        "single_volatile_stock": "Historical mean ~12% annual, volatility ~50%.",
        "otm_call_or_crypto":    "Historical mean ~30% annual, volatility ~120%, right-skewed.",
    },
    "conservative": {
        "cash":                  "Capital-preserving; no return, no drawdown.",
        "bonds":                 "Modest returns; rare drawdowns above 10%.",
        "broad_index":           "Drawdowns of 30-50% historical; recovers over years.",
        "leveraged_etf_3x":      "Devastating drawdowns; -70% to -90% drawdowns recorded; rebalancing decay.",
        "single_volatile_stock": "Idiosyncratic risk; stock-specific blowups common.",
        "otm_call_or_crypto":    "Most periods total loss; only rare windows positive.",
    },
}


# Display labels used in the prompt (more reader-friendly than YAML names) and their
# canonical mapping back to AssetSpec.name. The parser uses both.
ASSET_DISPLAY = [
    ("cash",                  "Cash"),
    ("bonds",                 "Bonds"),
    ("broad_index",           "Index"),
    ("leveraged_etf_3x",      "Leveraged"),
    ("single_volatile_stock", "Stock"),
    ("otm_call_or_crypto",    "OTM"),
]


OBJECTIVE_TEXT = {
    "wealth_maximisation": "Your objective: maximize the chance of reaching $150 from $100 in 30 days.",
    "capital_preservation": "Your objective: preserve capital with minimal drawdown over 30 days.",
}


def build_portfolio_prompt(
    condition: str,
    blurb_variant: str,
    objective: str,
    game: PortfolioGame,
    include_role_instruction: bool = True,
) -> str:
    """Build a per-round prompt for the portfolio task.

    See module docstring for condition / blurb / objective semantics.
    """
    if condition not in CONDITION_FRAMING:
        raise ValueError(f"unknown condition: {condition}; expected one of {list(CONDITION_FRAMING)}")
    if blurb_variant not in ASSET_BLURBS:
        raise ValueError(f"unknown blurb_variant: {blurb_variant}; expected one of {list(ASSET_BLURBS)}")
    if objective not in OBJECTIVE_TEXT:
        raise ValueError(f"unknown objective: {objective}; expected one of {list(OBJECTIVE_TEXT)}")

    parts = []
    if include_role_instruction:
        parts.append(ROLE_INSTRUCTION)
    parts.append("You are managing a portfolio over 30 monthly rebalancing decisions.\n")
    parts.append(f"Initial budget: ${game.initial_budget:.0f}\n")
    parts.append(f"Current portfolio value: ${game.portfolio_value:.2f}\n")
    parts.append(f"Round {game.round + 1} of {game.n_rounds}\n")
    parts.append(f"Current peak value: ${game.peak_value:.2f}, current drawdown: {game.current_drawdown() * 100:.1f}%\n\n")
    parts.append(OBJECTIVE_TEXT[objective] + "\n\n")

    framing = CONDITION_FRAMING[condition]
    if framing:
        parts.append(framing)

    parts.append("\nAvailable assets:\n")
    blurbs = ASSET_BLURBS[blurb_variant]
    for canon_name, display in ASSET_DISPLAY:
        if canon_name not in {a.name for a in game.assets}:
            continue
        blurb = blurbs.get(canon_name, "")
        parts.append(f"- {display} ({canon_name}): {blurb}\n")

    history_text = _short_history_text(game)
    if history_text:
        parts.append("\n" + history_text)

    parts.append(
        "\nAllocate 100 tokens across these assets (integers, summing to 100). "
        "Explain your reasoning briefly, then state your final allocation clearly.\n"
    )
    parts.append(
        "End your response with a single line in the exact format: "
        "Final Allocation: Cash X, Bonds Y, Index Z, Leveraged W, Stock V, OTM U "
        "(integers summing to 100)."
    )
    return "".join(parts)


def _short_history_text(game: PortfolioGame, last_n: int = 5) -> str:
    if not game.history:
        return ""
    out = ["Recent rounds:\n"]
    for h in game.history[-last_n:]:
        out.append(
            f"  Round {h['round']}: portfolio_return={h['portfolio_return'] * 100:+.2f}%, "
            f"value=${h['value_after']:.2f}, drawdown={h['drawdown'] * 100:.1f}%\n"
        )
    return "".join(out)


__all__ = [
    "ROLE_INSTRUCTION",
    "CONDITION_FRAMING",
    "ASSET_BLURBS",
    "ASSET_DISPLAY",
    "OBJECTIVE_TEXT",
    "build_portfolio_prompt",
]
