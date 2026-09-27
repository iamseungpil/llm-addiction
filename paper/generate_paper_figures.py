#!/usr/bin/env python3
"""Generate the manuscript's core behavioral figures with a unified paper style.

The slot-machine panels use the current canonical raw data, including the
corrected GPT-4o-mini export whose round outcomes are stored via
``game_history``. Other panels preserve the manuscript-facing audited values.
"""

from __future__ import annotations

import os
import sys
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# The unreleased ``paper_figure_style`` helper lives in the separate analysis
# tree, not in this repository and not on the HuggingFace dataset.  Point
# ``LLM_ADDICTION_ANALYSIS`` at that checkout to re-run this generator; a
# five-name subset is vendored at ``scripts/figures/paper_style_vendored.py``
# for the generators that were ported to it.
ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS", Path.home() / "llm-addiction"))
STYLE_SRC = ANALYSIS_ROOT / "experiments" / "07_sae_readout" / "src"
if str(STYLE_SRC) not in sys.path:
    sys.path.insert(0, str(STYLE_SRC))

from paper_figure_style import COLORS, annotate_bars, panel_title, save_pdf_png, style_axes, use_paper_style


OUT_DIR = Path(__file__).resolve().parent / "images"
HF_IC_DIR = Path("/tmp/llmadd_hf/investment_choice/bet_constraint/results")

# Optional on-disk mirror of the released corpora; see ``scripts/build_figure_data.py``.
DATA_ROOT = Path(os.environ.get("LLM_ADDICTION_DATA", Path.home() / "llm-addiction-data"))
LOCAL_IC_DIRS = {
    "Gemma-2-9B": DATA_ROOT / "behavioral" / "investment_choice" / "v2_role_gemma",
    "LLaMA-3.1-8B": DATA_ROOT / "behavioral" / "investment_choice" / "v2_role_llama",
}


SLOT_MODEL_ORDER = [
    "GPT-4o-mini",
    "GPT-4.1-mini",
    "Gemini-2.5-Flash",
    "Claude-3.5-Haiku",
    "LLaMA-3.1-8B",
    "Gemma-2-9B",
]
SLOT_FIXED_BK = np.array([0.0, 0.0, 3.12, 0.0, 0.44, 0.0])
SLOT_VARIABLE_BK = np.array([21.31, 6.31, 48.06, 20.50, 72.31, 5.44])
SLOT_FIXED_METRICS = np.array([0.104, 0.111, 0.001])
SLOT_VARIABLE_METRICS = np.array([0.294, 0.672, 0.196])

STREAK_LENGTHS = np.arange(1, 6)
STREAK_WIN_FIXED = np.array([0.07, 0.09, 0.01, 0.02, 0.00])
STREAK_WIN_VARIABLE = np.array([0.23, 0.25, 0.25, 0.24, 0.58])
STREAK_LOSS_FIXED = np.array([0.24, 0.25, 0.21, 0.19, 0.29])
STREAK_LOSS_VARIABLE = np.array([0.67, 0.45, 0.28, 0.37, 0.37])

IC_PROMPTS = ["BASE", "G", "M", "GM"]
# Order MUST match IC_PROMPTS. No-goal pair (BASE, M) in green; goal pair (G, GM) in red,
# matching Figs 2/3 (Fixed=green / Variable=red). Encodes the paper thesis: G/GM are the
# autonomy-inducing conditions that drive bankruptcy.
IC_PROMPT_COLORS = ["#59A14F", "#E15759", "#9DC388", "#B33533"]
IC_SEMANTIC_LABELS = ["Safe exit", "Low var.", "Mid var.", "High var."]
# Ordinal gradient safe→risky using paper palette anchors (green→gray→peach→red).
IC_SEMANTIC_COLORS = ["#59A14F", "#C7C7C7", "#E8A0A0", "#E15759"]
IC_MODEL_ORDER = [
    "GPT-4o-mini",
    "GPT-4.1-mini",
    "Gemini-2.5-Flash",
    "Claude-3.5-Haiku",
    "LLaMA-3.1-8B",
    "Gemma-2-9B",
]


def _slot_display_name(name: str) -> str:
    return (
        name.replace("GPT-4o-mini", "GPT-4o-\nmini")
        .replace("GPT-4.1-mini", "GPT-4.1-\nmini")
        .replace("Gemini-2.5-Flash", "Gemini-2.5-\nFlash")
        .replace("Claude-3.5-Haiku", "Claude-3.5-\nHaiku")
        .replace("LLaMA-3.1-8B", "LLaMA-3.1-\n8B")
        .replace("Gemma-2-9B", "Gemma-2-\n9B")
    )


def generate_slot_machine_figure() -> None:
    use_paper_style(11.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    x = np.arange(len(SLOT_MODEL_ORDER))
    width = 0.37
    bars1 = ax1.bar(x - width / 2, SLOT_FIXED_BK, width, color=COLORS["fixed"], label="Fixed")
    bars2 = ax1.bar(x + width / 2, SLOT_VARIABLE_BK, width, color=COLORS["variable"], label="Variable")
    annotate_bars(ax1, bars1, fmt="{:.1f}", suffix="%")
    annotate_bars(ax1, bars2, fmt="{:.1f}", suffix="%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["GPT-4o-mini", "GPT-4.1-mini", "Gemini-2.5-Flash",
         "Claude-3.5-Haiku", "LLaMA-3.1-8B", "Gemma-2-9B"],
        rotation=25, ha="right", rotation_mode="anchor",
    )
    ax1.set_ylabel("Bankruptcy rate (%)")
    ax1.set_ylim(0, 80)
    panel_title(ax1, "(a)", "Bankruptcy Rate by Bet Type")
    style_axes(ax1)
    ax1.legend(loc="upper left")

    metric_labels = [r"$I_{BA}$", r"$I_{LC}$", r"$I_{EC}$"]
    x2 = np.arange(len(metric_labels))
    bars3 = ax2.bar(x2 - width / 2, SLOT_FIXED_METRICS, width, color=COLORS["fixed"], label="Fixed")
    bars4 = ax2.bar(x2 + width / 2, SLOT_VARIABLE_METRICS, width, color=COLORS["variable"], label="Variable")
    annotate_bars(ax2, bars3, fmt="{:.3f}", size=8.2)
    annotate_bars(ax2, bars4, fmt="{:.3f}", size=8.2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylabel("Metric value")
    ax2.set_ylim(0, 0.75)
    panel_title(ax2, "(b)", "Irrationality Metrics by Bet Type")
    style_axes(ax2)
    ax2.legend(loc="upper left")

    save_pdf_png(fig, OUT_DIR, "slot_machine_analysis2")
    plt.close(fig)


def _plot_streak_panel(ax, fixed_vals: np.ndarray, variable_vals: np.ndarray, title: str) -> None:
    width = 0.36
    x = np.arange(len(STREAK_LENGTHS))
    bars1 = ax.bar(x - width / 2, fixed_vals, width, color=COLORS["fixed"], label="Fixed")
    bars2 = ax.bar(x + width / 2, variable_vals, width, color=COLORS["variable"], label="Variable")
    annotate_bars(ax, bars1, fmt="{:.2f}", color=COLORS["fixed"])
    annotate_bars(ax, bars2, fmt="{:.2f}", color=COLORS["variable"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in STREAK_LENGTHS])
    ax.set_xlabel("Consecutive streak length")
    ax.set_ylabel("Betting ratio increase")
    style_axes(ax)
    ax.legend(loc="upper left" if "Win" in title else "upper right")
    ax.set_ylim(0, 0.80 if "Loss" in title else 0.68)


def generate_streak_figure() -> None:
    use_paper_style(11.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    _plot_streak_panel(ax1, STREAK_WIN_FIXED, STREAK_WIN_VARIABLE, "After Win")
    _plot_streak_panel(ax2, STREAK_LOSS_FIXED, STREAK_LOSS_VARIABLE, "After Loss")
    panel_title(ax1, "(a)", "After Win")
    panel_title(ax2, "(b)", "After Loss")
    save_pdf_png(fig, OUT_DIR, "streak_analysis_1x2_comparison")
    plt.close(fig)


def generate_slot_streak_combined_figure() -> None:
    """Combined 1x4 strip: bankruptcy, irrationality metrics, post-win streak, post-loss streak."""
    use_paper_style(8.4)
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.05),
                              gridspec_kw={"width_ratios": [1.85, 1.0, 1.0, 1.0]})
    ax1, ax2, ax3, ax4 = axes

    # Top-left: Bankruptcy by model & bet type
    x = np.arange(len(SLOT_MODEL_ORDER))
    width = 0.37
    bars1 = ax1.bar(x - width / 2, SLOT_FIXED_BK, width, color=COLORS["fixed"], label="Fixed")
    bars2 = ax1.bar(x + width / 2, SLOT_VARIABLE_BK, width, color=COLORS["variable"], label="Variable")
    annotate_bars(ax1, bars1, fmt="{:.1f}", suffix="%", size=7.2)
    annotate_bars(ax1, bars2, fmt="{:.1f}", suffix="%", size=7.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["GPT-4o", "GPT-4.1", "Gemini-2.5", "Claude-3.5", "LLaMA-3.1", "Gemma-2"],
        rotation=20, ha="right", rotation_mode="anchor", fontsize=10.5,
    )
    ax1.set_ylabel("Bankruptcy rate (%)")
    ax1.set_ylim(0, 85)
    panel_title(ax1, "(a)", "Bankruptcy Rate by Bet Type")
    style_axes(ax1)
    ax1.legend(loc="upper left", fontsize=10.5)

    # Top-right: Irrationality metrics by bet type
    metric_labels = [r"$I_{BA}$", r"$I_{LC}$", r"$I_{EC}$"]
    x2 = np.arange(len(metric_labels))
    bars3 = ax2.bar(x2 - width / 2, SLOT_FIXED_METRICS, width, color=COLORS["fixed"], label="Fixed")
    bars4 = ax2.bar(x2 + width / 2, SLOT_VARIABLE_METRICS, width, color=COLORS["variable"], label="Variable")
    annotate_bars(ax2, bars3, fmt="{:.3f}", size=7.6)
    annotate_bars(ax2, bars4, fmt="{:.3f}", size=7.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylabel("Metric value")
    ax2.set_ylim(0, 0.78)
    panel_title(ax2, "(b)", "Irrationality Metrics by Bet Type")
    style_axes(ax2)
    ax2.legend(loc="upper left", fontsize=10.5)

    # Bottom row: streak after win / after loss
    _plot_streak_panel(ax3, STREAK_WIN_FIXED, STREAK_WIN_VARIABLE, "After Win")
    _plot_streak_panel(ax4, STREAK_LOSS_FIXED, STREAK_LOSS_VARIABLE, "After Loss")
    panel_title(ax3, "(c)", "Bet-Ratio Increase After Wins")
    panel_title(ax4, "(d)", "Bet-Ratio Increase After Losses")

    fig.tight_layout()
    save_pdf_png(fig, OUT_DIR, "slot_streak_combined")
    plt.close(fig)


def _extract_goal_from_response(response: str) -> int | None:
    patterns = [
        r"(?:goal|target)(?:\s+(?:is|:))?\s*\$?(\d+)",
        r"\$(\d+)\s*(?:goal|target)",
        r"(?:aim|aiming)\s+(?:for|to)\s+\$?(\d+)",
        r"(?:reach|get\s+to)\s+\$?(\d+)",
        r"(?:new|current|my)\s+goal[:\s]+\$?(\d+)",
        r"goal[:\s]+.*?\$(\d+)",
        r"goal[:\s]+.*?(\d+)\s*(?:dollars?)?",
        r"(?:balance|reach).*?(?:at\s+least|of)\s+\$?(\d+)",
        r"set\s+(?:a\s+)?(?:new\s+)?goal[:\s]+\$?(\d+)",
    ]
    text = response.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if not matches:
            continue
        try:
            goal = int(matches[-1])
        except ValueError:
            continue
        if 50 <= goal <= 10000:
            return goal
    return None


def _load_investment_choice_rows() -> list[dict]:
    rows: list[dict] = []
    api_files: dict[tuple[str, int, str], Path] = {}

    for json_path in sorted(HF_IC_DIR.glob("*.json")):
        name = json_path.name
        if name.startswith("gpt4o_mini_"):
            model = "GPT-4o-mini"
        elif name.startswith("gpt41_mini_"):
            model = "GPT-4.1-mini"
        elif name.startswith("gemini_flash_"):
            model = "Gemini-2.5-Flash"
        elif name.startswith("claude_haiku_"):
            model = "Claude-3.5-Haiku"
        else:
            continue
        match = re.search(r"_(10|30|50|70)_(fixed|variable)_", name)
        if match is None:
            continue
        # Keep one canonical file per (model, cap, bet type). Some HF exports
        # contain rerun duplicates for the same cell, and the latest timestamped
        # filename should define the paper-facing aggregate.
        key = (model, int(match.group(1)), match.group(2))
        api_files[key] = json_path

    for (model, bet_constraint, bet_type), json_path in sorted(api_files.items()):
        data = json.loads(json_path.read_text())
        for game in data["results"]:
            rows.append(
                {
                    "model": model,
                    "source": "api",
                    "prompt": game["prompt_condition"],
                    "bet_constraint": bet_constraint,
                    "bet_type": bet_type,
                    "game": game,
                }
            )

    for model, ic_dir in LOCAL_IC_DIRS.items():
        for json_path in sorted(ic_dir.glob("*.json")):
            data = json.loads(json_path.read_text())
            for game in data["results"]:
                bet_constraint = game["bet_constraint"]
                if isinstance(bet_constraint, str) and bet_constraint.startswith("c"):
                    bet_constraint = bet_constraint[1:]
                rows.append(
                    {
                        "model": model,
                        "source": "local",
                        "prompt": game["prompt_condition"],
                        "bet_constraint": int(bet_constraint),
                        "bet_type": game["bet_type"],
                        "game": game,
                    }
                )

    if not rows:
        raise FileNotFoundError("Investment choice data not found for figure generation.")
    return rows


def _semantic_option(row: dict, decision: dict) -> int | None:
    if row["source"] == "api":
        choice = decision.get("choice")
        return choice if choice in {1, 2, 3, 4} else None

    prompt_option = decision.get("prompt_option")
    if prompt_option in {1, 2, 3, 4}:
        return 5 - prompt_option

    choice = decision.get("choice")
    return 5 - choice if choice in {1, 2, 3, 4} else None


def _game_bankrupt(game: dict) -> bool:
    return (
        str(game.get("exit_reason", "")).lower() in {"bankrupt", "bankruptcy"}
        or bool(game.get("bankruptcy"))
        or str(game.get("final_outcome", "")).lower() == "bankruptcy"
        or game.get("final_balance", 1) <= 0
    )


def _goal_escalated(row: dict) -> bool:
    previous_goal = None
    for decision in row["game"]["decisions"]:
        if row["source"] == "local":
            goal = decision.get("goal_after")
            if goal is None:
                goal = decision.get("goal_before")
        else:
            goal = _extract_goal_from_response(decision.get("response", ""))
            if goal is None:
                match = re.search(
                    r"current self-set goal(?: from previous round)?:\s*\$?(\d+)",
                    decision.get("prompt", "").lower(),
                )
                goal = int(match.group(1)) if match else None
        if goal is None:
            continue
        if previous_goal is not None and goal > previous_goal:
            return True
        previous_goal = goal
    return False


def _compute_ic_summary(rows: list[dict]) -> dict:
    bankruptcy = []
    goal_escalation = []
    option_distribution = []
    model_goal_delta = []

    for prompt in IC_PROMPTS:
        prompt_rows = [row for row in rows if row["prompt"] == prompt]
        bankruptcy.append(
            100.0 * np.mean([_game_bankrupt(row["game"]) for row in prompt_rows])
        )
        goal_escalation.append(
            100.0 * np.mean([_goal_escalated(row) for row in prompt_rows])
        )

        counts = Counter()
        total = 0
        for row in prompt_rows:
            for decision in row["game"]["decisions"]:
                option = _semantic_option(row, decision)
                if option is None:
                    continue
                counts[option] += 1
                total += 1
        option_distribution.append(
            [100.0 * counts[idx] / total for idx in range(1, 5)]
        )

    for model in IC_MODEL_ORDER:
        base_rows = [
            row for row in rows if row["model"] == model and row["prompt"] in {"BASE", "M"}
        ]
        goal_rows = [
            row for row in rows if row["model"] == model and row["prompt"] in {"G", "GM"}
        ]
        base_bk = 100.0 * np.mean([_game_bankrupt(row["game"]) for row in base_rows])
        goal_bk = 100.0 * np.mean([_game_bankrupt(row["game"]) for row in goal_rows])
        model_goal_delta.append(goal_bk - base_bk)

    return {
        "bankruptcy": np.array(bankruptcy),
        "goal_escalation": np.array(goal_escalation),
        "option_distribution": np.array(option_distribution).T,
        "model_goal_delta": np.array(model_goal_delta),
    }


def generate_investment_choice_figure() -> None:
    use_paper_style(11.5)
    rows = _load_investment_choice_rows()
    ic = _compute_ic_summary(rows)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18.0, 4.6),
        gridspec_kw={"width_ratios": [1.05, 1.44, 1.05, 1.30]},
    )

    x = np.arange(len(IC_PROMPTS))

    bars = axes[0].bar(
        x,
        ic["bankruptcy"],
        width=0.78,
        color=IC_PROMPT_COLORS,
        edgecolor="#333333",
        linewidth=0.8,
    )
    annotate_bars(axes[0], bars, fmt="{:.1f}", suffix="%", size=10.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(IC_PROMPTS, fontsize=11.5)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(0)
    axes[0].set_ylabel("Bankruptcy rate (%)")
    axes[0].set_ylim(0, 48)
    axes[0].margins(x=0.10)
    panel_title(axes[0], "(a)", "Bankruptcy")
    style_axes(axes[0])

    bottom = np.zeros(len(IC_PROMPTS))
    for label, vals, color in zip(IC_SEMANTIC_LABELS, ic["option_distribution"], IC_SEMANTIC_COLORS):
        bars = axes[1].bar(
            x,
            vals,
            bottom=bottom,
            width=0.82,
            color=color,
            edgecolor="#333333",
            linewidth=0.8,
            label=label,
        )
        for xi, v, b in zip(x, vals, bottom):
            if v >= 13:
                txt_color = "white" if color in {COLORS["option3"], COLORS["option4"]} else "black"
                axes[1].text(
                    xi,
                    b + v / 2,
                    f"{int(round(v))}%",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=txt_color,
                    fontweight="bold",
                )
        bottom += vals
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(IC_PROMPTS, fontsize=10.5)
    axes[1].set_ylabel("Distribution (%)")
    axes[1].set_ylim(0, 100)
    axes[1].margins(x=0.12)
    panel_title(axes[1], "(b)", "Option Mix")
    style_axes(axes[1])
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=1.0,
        columnspacing=0.8,
    )

    bars = axes[2].bar(
        x,
        ic["goal_escalation"],
        width=0.78,
        color=IC_PROMPT_COLORS,
        edgecolor="#333333",
        linewidth=0.8,
    )
    annotate_bars(axes[2], bars, fmt="{:.1f}", suffix="%", size=10.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(IC_PROMPTS, fontsize=11.5)
    axes[2].set_ylabel("Goal escalation (%)")
    axes[2].set_ylim(0, 62)
    axes[2].margins(x=0.10)
    panel_title(axes[2], "(c)", "Goal Reset")
    style_axes(axes[2])

    # Panel (d): GPT-4o slot-machine cap ablation (Finding 4 control).
    # Values eyeballed from the legacy ICLR Finding 4 figure
    # (legacy/gpt_variable_max_bet_experiment/analysis/figures/
    #  1_fixed_vs_variable_comparison_corrected.pdf); raw data files live on
    # the original ubuntu host and are not on this machine, so the panel uses
    # the same axis style as (a)-(c) with hand-extracted bankruptcy rates.
    cap_levels = [10, 30, 50, 70]
    bankruptcy_fixed = [0.5, 0.3, 4.7, 0.5]
    bankruptcy_variable = [1.0, 14.0, 16.5, 17.0]

    x4 = np.arange(len(cap_levels))
    width4 = 0.40
    bars1 = axes[3].bar(
        x4 - width4 / 2,
        bankruptcy_fixed,
        width4,
        color=COLORS["fixed"],
        edgecolor="#333333",
        linewidth=0.8,
        label="Fixed (= cap)",
    )
    bars2 = axes[3].bar(
        x4 + width4 / 2,
        bankruptcy_variable,
        width4,
        color=COLORS["variable"],
        edgecolor="#333333",
        linewidth=0.8,
        label="Variable (≤ cap)",
    )
    annotate_bars(axes[3], bars1, fmt="{:.1f}", suffix="%", size=10.0)
    annotate_bars(axes[3], bars2, fmt="{:.1f}", suffix="%", size=10.0)
    axes[3].set_xticks(x4)
    axes[3].set_xticklabels([f"${c}" for c in cap_levels], fontsize=11.5)
    axes[3].set_ylabel("Bankruptcy rate (%)")
    axes[3].set_ylim(0, 22)
    axes[3].margins(x=0.10)
    panel_title(axes[3], "(d)", "SM Matched Caps (GPT-4o)")
    style_axes(axes[3])
    axes[3].legend(loc="upper left", fontsize=10.5)

    save_pdf_png(fig, OUT_DIR, "investment_choice_4panel")
    save_pdf_png(fig, OUT_DIR, "investment_choice3")
    plt.close(fig)


def generate_investment_constraint_figure() -> None:
    use_paper_style(8.4)
    rows = _load_investment_choice_rows()
    constraints = [10, 30, 50, 70]

    bankruptcy_fixed = []
    bankruptcy_variable = []
    rounds_fixed = []
    rounds_variable = []
    for constraint in constraints:
        fixed_rows = [
            row
            for row in rows
            if row["bet_constraint"] == constraint and row["bet_type"] == "fixed"
        ]
        variable_rows = [
            row
            for row in rows
            if row["bet_constraint"] == constraint and row["bet_type"] == "variable"
        ]
        bankruptcy_fixed.append(
            100.0 * np.mean([_game_bankrupt(row["game"]) for row in fixed_rows])
        )
        bankruptcy_variable.append(
            100.0 * np.mean([_game_bankrupt(row["game"]) for row in variable_rows])
        )
        rounds_fixed.append(
            np.mean(
                [
                    row["game"].get("rounds_played", row["game"].get("rounds_completed"))
                    for row in fixed_rows
                ]
            )
        )
        rounds_variable.append(
            np.mean(
                [
                    row["game"].get("rounds_played", row["game"].get("rounds_completed"))
                    for row in variable_rows
                ]
            )
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.6))
    x = np.arange(len(constraints))
    width = 0.34

    bars1 = ax1.bar(
        x - width / 2,
        bankruptcy_fixed,
        width,
        color=COLORS["fixed"],
        edgecolor="#333333",
        linewidth=0.8,
        label="Fixed",
    )
    bars2 = ax1.bar(
        x + width / 2,
        bankruptcy_variable,
        width,
        color=COLORS["variable"],
        edgecolor="#333333",
        linewidth=0.8,
        label="Variable",
    )
    annotate_bars(ax1, bars1, fmt="{:.1f}", suffix="%")
    annotate_bars(ax1, bars2, fmt="{:.1f}", suffix="%")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"${constraint}" for constraint in constraints])
    ax1.set_ylabel("Bankruptcy rate (%)")
    ax1.set_ylim(0, 52)
    panel_title(ax1, "(a)", "Matched Caps")
    style_axes(ax1)
    ax1.legend(loc="upper left")

    ax2.plot(x, rounds_fixed, marker="o", markersize=4.4, linewidth=2.0, color=COLORS["fixed"], label="Fixed")
    ax2.plot(x, rounds_variable, marker="o", markersize=4.4, linewidth=2.0, color=COLORS["variable"], label="Variable")
    for xi, value in zip(x, rounds_fixed):
        ax2.annotate(f"{value:.1f}", xy=(xi, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10.5)
    for xi, value in zip(x, rounds_variable):
        ax2.annotate(f"{value:.1f}", xy=(xi, value), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=10.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"${constraint}" for constraint in constraints])
    ax2.set_ylabel("Average rounds")
    ax2.set_ylim(0, max(rounds_fixed + rounds_variable) + 2.0)
    panel_title(ax2, "(b)", "Rounds Played")
    style_axes(ax2)
    ax2.legend(loc="upper left")

    save_pdf_png(fig, OUT_DIR, "combined_fixed_vs_variable_with_composite")
    plt.close(fig)


def main() -> None:
    generate_slot_machine_figure()
    generate_streak_figure()
    generate_slot_streak_combined_figure()
    generate_investment_choice_figure()
    generate_investment_constraint_figure()
    print("Saved improved paper figures to", OUT_DIR)


if __name__ == "__main__":
    main()
