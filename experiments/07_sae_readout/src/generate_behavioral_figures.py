"""
Generate publication-quality behavioral comparison figures for Gemma vs LLaMA.

Produces 3 figures:
  1. behavioral_gemma_vs_llama.png — 2x2 panel comparing BK rates
  2. bet_type_asymmetry.png — Fixed vs Variable BK by constraint per model
  3. prompt_component_effects.png — Marginal effects of prompt components

Data sources:
  - IC V2role Gemma: investment_choice_v2_role/gemma_investment_c{10,30,50,70}_*.json
  - IC V2role LLaMA: investment_choice_v2_role_llama/llama_investment_c{10,30,50,70}_*.json
  - SM V4role Gemma: slot_machine/experiment_0_gemma_v4_role/final_gemma_*.json
  - MW V2role Gemma: mystery_wheel_v2_role/gemma_mysterywheel_c30_*.json

No GPU or model loading required. Pure data analysis and visualization.

Usage:
    python sae_v3_analysis/src/generate_behavioral_figures.py
"""

import json
import glob
import os
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.constrained_layout.use": True,
})
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
GEMMA_COLOR = COLORS[0]   # blue
LLAMA_COLOR = COLORS[1]   # orange

DATA_ROOT = Path("/home/jovyan/beomi/llm-addiction-data")
OUTPUT_DIR = Path("/home/jovyan/llm-addiction/sae_v3_analysis/results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_ic_data(model: str) -> list[dict[str, Any]]:
    """Load Investment Choice V2role data for the given model.

    Returns a flat list of game-result dicts with keys:
        bankruptcy, bet_type, prompt_condition, bet_constraint
    """
    if model == "gemma":
        base = DATA_ROOT / "investment_choice_v2_role"
        pattern = "gemma_investment_c*_2*.json"
    elif model == "llama":
        base = DATA_ROOT / "investment_choice_v2_role_llama"
        pattern = "llama_investment_c*_2*.json"
    else:
        raise ValueError(f"Unknown model: {model}")

    results: list[dict] = []
    files = sorted(base.glob(pattern))
    # Exclude checkpoint files and unlimited files
    files = [f for f in files if "checkpoint" not in f.name and "unlimited" not in f.name]
    for fpath in files:
        with open(fpath) as fh:
            data = json.load(fh)
        results.extend(data["results"])
    print(f"  IC {model}: loaded {len(results)} games from {len(files)} files")
    return results


def load_sm_data() -> list[dict[str, Any]]:
    """Load Slot Machine V4role Gemma data.

    Returns list of dicts with normalised keys:
        bankruptcy, bet_type, prompt_condition
    """
    fpath = DATA_ROOT / "slot_machine" / "experiment_0_gemma_v4_role" / "final_gemma_20260227_002507.json"
    with open(fpath) as fh:
        data = json.load(fh)
    results = []
    for g in data["results"]:
        results.append({
            "bankruptcy": g["outcome"] == "bankruptcy",
            "bet_type": g["bet_type"],
            "prompt_condition": g["prompt_combo"],
        })
    print(f"  SM Gemma: loaded {len(results)} games")
    return results


def load_mw_data() -> list[dict[str, Any]]:
    """Load Mystery Wheel V2role Gemma data (c30 final file)."""
    fpath = DATA_ROOT / "mystery_wheel_v2_role" / "gemma_mysterywheel_c30_20260226_184400.json"
    with open(fpath) as fh:
        data = json.load(fh)
    results = data["results"]
    print(f"  MW Gemma: loaded {len(results)} games")
    return results


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def bk_rate(games: list[dict]) -> float:
    """Bankruptcy rate as a fraction [0, 1]."""
    if not games:
        return 0.0
    return sum(1 for g in games if g["bankruptcy"]) / len(games)


def bk_count(games: list[dict]) -> int:
    return sum(1 for g in games if g["bankruptcy"])


def filter_games(games: list[dict], **kwargs) -> list[dict]:
    """Filter games by matching field values."""
    out = games
    for key, val in kwargs.items():
        if isinstance(val, (list, tuple, set)):
            out = [g for g in out if g.get(key) in val]
        else:
            out = [g for g in out if g.get(key) == val]
    return out


def marginal_bk_rate(games: list[dict], component: str) -> tuple[float, float, int, int]:
    """Compute marginal bankruptcy rate when a prompt component is present vs absent.

    Returns (rate_present, rate_absent, n_present, n_absent).
    BASE is treated as having no components.
    """
    present = [g for g in games if component in g["prompt_condition"] and g["prompt_condition"] != "BASE"]
    absent = [g for g in games if component not in g["prompt_condition"] or g["prompt_condition"] == "BASE"]
    rate_p = bk_rate(present) if present else 0.0
    rate_a = bk_rate(absent) if absent else 0.0
    return rate_p, rate_a, len(present), len(absent)


def bk_rate_ci(games: list[dict], z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for bankruptcy rate."""
    n = len(games)
    if n == 0:
        return (0.0, 0.0)
    p = bk_rate(games)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Figure 1: Behavioral Gemma vs LLaMA (2x2)
# ---------------------------------------------------------------------------

def figure1_behavioral_comparison(ic_gemma: list[dict], ic_llama: list[dict]) -> None:
    """2x2 panel comparing Gemma and LLaMA IC behavioural metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Investment Choice: Gemma vs LLaMA Behavioral Comparison", fontsize=14, fontweight="bold")

    constraints = ["10", "30", "50", "70"]
    bet_types = ["fixed", "variable"]
    prompt_conds = ["BASE", "G", "GM", "M"]

    # ---- Top-left: BK rate by constraint ----
    ax = axes[0, 0]
    x = np.arange(len(constraints))
    width = 0.35
    gemma_rates = [bk_rate(filter_games(ic_gemma, bet_constraint=c)) * 100 for c in constraints]
    llama_rates = [bk_rate(filter_games(ic_llama, bet_constraint=c)) * 100 for c in constraints]
    bars_g = ax.bar(x - width / 2, gemma_rates, width, label="Gemma", color=GEMMA_COLOR, edgecolor="white", linewidth=0.5)
    bars_l = ax.bar(x + width / 2, llama_rates, width, label="LLaMA", color=LLAMA_COLOR, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{c}" for c in constraints])
    ax.set_ylabel("Bankruptcy Rate (%)")
    ax.set_title("(a) BK Rate by Bet Constraint")
    ax.legend(fontsize=9)
    # Annotate values
    for bar in list(bars_g) + list(bars_l):
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # ---- Top-right: BK rate by bet type ----
    ax = axes[0, 1]
    x = np.arange(len(bet_types))
    gemma_bt = [bk_rate(filter_games(ic_gemma, bet_type=bt)) * 100 for bt in bet_types]
    llama_bt = [bk_rate(filter_games(ic_llama, bet_type=bt)) * 100 for bt in bet_types]
    bars_g = ax.bar(x - width / 2, gemma_bt, width, label="Gemma", color=GEMMA_COLOR, edgecolor="white", linewidth=0.5)
    bars_l = ax.bar(x + width / 2, llama_bt, width, label="LLaMA", color=LLAMA_COLOR, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Fixed", "Variable"])
    ax.set_ylabel("Bankruptcy Rate (%)")
    ax.set_title("(b) BK Rate by Bet Type")
    ax.legend(fontsize=9)
    for bar in list(bars_g) + list(bars_l):
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # ---- Bottom-left: BK rate by prompt condition (c50+c70) ----
    ax = axes[1, 0]
    high_constraints = {"50", "70"}
    x = np.arange(len(prompt_conds))
    gemma_pc = [bk_rate(filter_games(ic_gemma, prompt_condition=pc, bet_constraint=high_constraints)) * 100
                for pc in prompt_conds]
    llama_pc = [bk_rate(filter_games(ic_llama, prompt_condition=pc, bet_constraint=high_constraints)) * 100
                for pc in prompt_conds]
    bars_g = ax.bar(x - width / 2, gemma_pc, width, label="Gemma", color=GEMMA_COLOR, edgecolor="white", linewidth=0.5)
    bars_l = ax.bar(x + width / 2, llama_pc, width, label="LLaMA", color=LLAMA_COLOR, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_conds)
    ax.set_ylabel("Bankruptcy Rate (%)")
    ax.set_title("(c) BK Rate by Prompt Condition (c50 + c70)")
    ax.legend(fontsize=9)
    for bar in list(bars_g) + list(bars_l):
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # ---- Bottom-right: Cumulative BK totals ----
    ax = axes[1, 1]
    gemma_total = len(ic_gemma)
    llama_total = len(ic_llama)
    gemma_bk = bk_count(ic_gemma)
    llama_bk = bk_count(ic_llama)
    labels = ["Gemma", "LLaMA"]
    bk_vals = [gemma_bk, llama_bk]
    nobk_vals = [gemma_total - gemma_bk, llama_total - llama_bk]
    x = np.arange(2)
    bars_bk = ax.bar(x, bk_vals, 0.5, label="Bankrupt", color=COLORS[3], edgecolor="white", linewidth=0.5)
    bars_nobk = ax.bar(x, nobk_vals, 0.5, bottom=bk_vals, label="Survived", color=COLORS[5], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of Games")
    ax.set_title("(d) Total Bankruptcy Counts")
    ax.legend(fontsize=9)
    for i, (bk, total) in enumerate(zip(bk_vals, [gemma_total, llama_total])):
        ax.annotate(f"{bk}/{total}\n({100*bk/total:.1f}%)",
                    xy=(x[i], bk / 2), ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    outpath = OUTPUT_DIR / "behavioral_gemma_vs_llama.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Figure 2: Bet Type Asymmetry
# ---------------------------------------------------------------------------

def figure2_bet_type_asymmetry(ic_gemma: list[dict], ic_llama: list[dict]) -> None:
    """Show Fixed>>Variable asymmetry in Gemma vs balanced pattern in LLaMA."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Bet Type Asymmetry: Gemma vs LLaMA", fontsize=14, fontweight="bold")

    constraints = ["10", "30", "50", "70"]
    bet_types = ["fixed", "variable"]
    width = 0.35

    for idx, (model_name, ic_data, color) in enumerate([
        ("Gemma", ic_gemma, GEMMA_COLOR),
        ("LLaMA", ic_llama, LLAMA_COLOR),
    ]):
        ax = axes[idx]
        x = np.arange(len(constraints))
        fixed_bk = [bk_count(filter_games(ic_data, bet_constraint=c, bet_type="fixed")) for c in constraints]
        var_bk = [bk_count(filter_games(ic_data, bet_constraint=c, bet_type="variable")) for c in constraints]
        fixed_n = [len(filter_games(ic_data, bet_constraint=c, bet_type="fixed")) for c in constraints]
        var_n = [len(filter_games(ic_data, bet_constraint=c, bet_type="variable")) for c in constraints]

        bars_f = ax.bar(x - width / 2, fixed_bk, width, label="Fixed",
                        color=color, edgecolor="white", linewidth=0.5, alpha=0.9)
        # Use a lighter shade for Variable
        var_color = COLORS[2] if idx == 0 else COLORS[4]
        bars_v = ax.bar(x + width / 2, var_bk, width, label="Variable",
                        color=var_color, edgecolor="white", linewidth=0.5, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([f"c{c}" for c in constraints])
        ax.set_xlabel("Bet Constraint")
        ax.set_ylabel("Bankruptcy Count")
        ax.set_title(f"{model_name}")
        ax.legend(fontsize=9)

        # Annotate with count and rate
        for bars, counts, totals in [(bars_f, fixed_bk, fixed_n), (bars_v, var_bk, var_n)]:
            for bar, cnt, tot in zip(bars, counts, totals):
                if cnt > 0:
                    rate = 100 * cnt / tot if tot > 0 else 0
                    ax.annotate(f"{cnt}\n({rate:.0f}%)",
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7.5)

    outpath = OUTPUT_DIR / "bet_type_asymmetry.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Figure 3: Prompt Component Marginal Effects
# ---------------------------------------------------------------------------

def figure3_prompt_component_effects(
    ic_gemma: list[dict],
    ic_llama: list[dict],
    sm_gemma: list[dict],
    mw_gemma: list[dict],
) -> None:
    """2x2 panel showing marginal BK rate difference per prompt component.

    For IC (Gemma, LLaMA): components G, M
    For SM, MW (Gemma): components G, M, R, W, P
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Prompt Component Marginal Effects on Bankruptcy Rate", fontsize=14, fontweight="bold")

    panels = [
        ("(a) IC Gemma", ic_gemma, ["G", "M"]),
        ("(b) IC LLaMA", ic_llama, ["G", "M"]),
        ("(c) SM Gemma", sm_gemma, ["G", "M", "R", "W", "P"]),
        ("(d) MW Gemma", mw_gemma, ["G", "M", "R", "W", "P"]),
    ]

    for ax, (title, data, components) in zip(axes.flat, panels):
        diffs = []
        ci_lows = []
        ci_highs = []
        labels = []

        for comp in components:
            rate_p, rate_a, n_p, n_a = marginal_bk_rate(data, comp)
            diff = (rate_p - rate_a) * 100  # percentage points

            # Bootstrap CI for the difference
            rng = np.random.default_rng(42)
            present_arr = np.array([1 if g["bankruptcy"] else 0
                                    for g in data
                                    if comp in g["prompt_condition"] and g["prompt_condition"] != "BASE"])
            absent_arr = np.array([1 if g["bankruptcy"] else 0
                                   for g in data
                                   if comp not in g["prompt_condition"] or g["prompt_condition"] == "BASE"])

            if len(present_arr) > 0 and len(absent_arr) > 0:
                boot_diffs = []
                n_boot = 2000
                for _ in range(n_boot):
                    bp = rng.choice(present_arr, size=len(present_arr), replace=True).mean()
                    ba = rng.choice(absent_arr, size=len(absent_arr), replace=True).mean()
                    boot_diffs.append((bp - ba) * 100)
                boot_diffs = np.array(boot_diffs)
                ci_low = np.percentile(boot_diffs, 2.5)
                ci_high = np.percentile(boot_diffs, 97.5)
            else:
                ci_low = ci_high = 0.0

            diffs.append(diff)
            ci_lows.append(diff - ci_low)
            ci_highs.append(ci_high - diff)
            labels.append(comp)

        x = np.arange(len(components))
        colors_bar = [COLORS[i % len(COLORS)] for i in range(len(components))]
        bars = ax.bar(x, diffs, 0.6, color=colors_bar, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, diffs, yerr=[ci_lows, ci_highs],
                    fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("BK Rate Diff (pp)")
        ax.set_title(title)

        # Add significance stars
        for i, (d, cl, ch) in enumerate(zip(diffs, ci_lows, ci_highs)):
            # Significant if CI does not cross zero
            upper = d + ch
            lower = d - cl
            is_sig = (lower > 0) or (upper < 0)
            if is_sig:
                y_pos = d + ch + 0.3 if d >= 0 else d - cl - 0.3
                ax.annotate("*", xy=(i, y_pos), ha="center", va="bottom" if d >= 0 else "top",
                            fontsize=14, fontweight="bold", color="red")

        # Annotate exact values
        for i, d in enumerate(diffs):
            y_offset = 5 if d >= 0 else -12
            ax.annotate(f"{d:+.1f}pp",
                        xy=(x[i], d),
                        xytext=(0, y_offset), textcoords="offset points",
                        ha="center", va="bottom" if d >= 0 else "top", fontsize=8)

        # Add n counts
        for i, comp in enumerate(components):
            _, _, n_p, n_a = marginal_bk_rate(data, comp)
            ax.annotate(f"n={n_p}|{n_a}",
                        xy=(x[i], 0), xytext=(0, -18), textcoords="offset points",
                        ha="center", fontsize=6.5, color="gray")

    outpath = OUTPUT_DIR / "prompt_component_effects.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    ic_gemma = load_ic_data("gemma")
    ic_llama = load_ic_data("llama")
    sm_gemma = load_sm_data()
    mw_gemma = load_mw_data()

    # Quick summary
    print("\n--- Data Summary ---")
    for label, data in [("IC Gemma", ic_gemma), ("IC LLaMA", ic_llama),
                         ("SM Gemma", sm_gemma), ("MW Gemma", mw_gemma)]:
        n = len(data)
        bk = bk_count(data)
        print(f"  {label}: N={n}, BK={bk} ({100*bk/n:.1f}%)")

    print("\n--- Generating Figures ---")
    figure1_behavioral_comparison(ic_gemma, ic_llama)
    figure2_bet_type_asymmetry(ic_gemma, ic_llama)
    figure3_prompt_component_effects(ic_gemma, ic_llama, sm_gemma, mw_gemma)

    print("\nAll figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
