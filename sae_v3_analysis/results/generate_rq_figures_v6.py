#!/usr/bin/env python3
"""
Generate publication-quality figures for the RQ-focused SAE study.

Figures:
  1. RQ1 - BK Prediction Comparison Matrix (2x3 grouped bar)
  2. RQ1 - Layer-wise AUC Curves (SAE vs Hidden, 3 paradigms)
  3. RQ2 - Cross-Domain Transfer Heatmap (SAE vs Hidden)
  4. RQ3 - Condition Encoding (horizontal bar chart)
  5. RQ3 - G Component Behavioral Effect (BK rate with/without G)

Output: /home/jovyan/llm-addiction/sae_v3_analysis/results/figures_v6/
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("/home/jovyan/llm-addiction/sae_v3_analysis/results")
JSON_DIR = RESULTS_DIR / "json"
OUT_DIR = RESULTS_DIR / "figures_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RQ_FILE = RESULTS_DIR / "rq_comprehensive_results.json"
ALL_ANALYSES_FILE = JSON_DIR / "all_analyses_20260306_091055.json"
COMPREHENSIVE_GEMMA_FILE = JSON_DIR / "comprehensive_gemma_20260309_063511.json"
CONDITION_V2_FILE = JSON_DIR / "condition_v2_20260308_151953.json"

# ---------------------------------------------------------------------------
# Matplotlib global settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
})

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]

# Paradigm display names
PARADIGM_LABELS = {"ic": "Investment\nChoice", "sm": "Slot\nMachine", "mw": "Mystery\nWheel"}
PARADIGM_SHORT = {"ic": "IC", "sm": "SM", "mw": "MW"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(RQ_FILE) as f:
    rq = json.load(f)
with open(ALL_ANALYSES_FILE) as f:
    all_analyses = json.load(f)
with open(COMPREHENSIVE_GEMMA_FILE) as f:
    comp_gemma = json.load(f)
with open(CONDITION_V2_FILE) as f:
    cond_v2 = json.load(f)

rq1 = rq["RQ1_bankruptcy_prediction"]
rq2 = rq["RQ2_domain_invariant"]
rq3 = rq["RQ3_condition_differences"]


# ===================================================================
# FIGURE 1: RQ1 - BK Prediction Comparison Matrix
# ===================================================================
def fig1_bk_prediction_matrix():
    """2x3 grouped bar chart: rows=[SAE, Hidden], cols=[IC, SM, MW],
    3 bars per cell: DP (green), R1 (blue), Balance-Matched (orange)."""

    paradigms = ["ic", "sm", "mw"]

    # Gather AUC values -- SAE row
    sae_dp = {p: rq1["sae_bk_dp"][p]["best_auc"] for p in paradigms}
    sae_r1 = {p: rq1["sae_r1_classification"][p]["best_auc"] for p in paradigms}
    sae_bm = {p: rq1["sae_bk_balance_matched"][p]["best_auc"] for p in paradigms}

    # Hidden row
    hidden_dp = {p: rq1["hidden_bk_dp"][p]["best_auc"] for p in paradigms}
    hidden_r1 = {p: rq1["hidden_bk_r1"][p]["best_auc"] for p in paradigms}
    # Hidden balance-matched: use the _dp keys from hidden_bk_balance_matched
    hidden_bm = {}
    hbm = rq1["hidden_bk_balance_matched"]
    for p in paradigms:
        key = f"{p}_dp"
        if key in hbm:
            hidden_bm[p] = hbm[key]["best_auc"]
        else:
            hidden_bm[p] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bar_width = 0.22
    x = np.arange(len(paradigms))
    bar_colors = ["#009E73", "#0072B2", "#D55E00"]  # DP green, R1 blue, BM orange
    bar_labels = ["Decision-Point", "Round 1", "Balance-Matched"]

    for ax_idx, (row_label, dp_vals, r1_vals, bm_vals) in enumerate([
        ("SAE Features", sae_dp, sae_r1, sae_bm),
        ("Hidden States", hidden_dp, hidden_r1, hidden_bm),
    ]):
        ax = axes[ax_idx]
        for i, (vals, color, label) in enumerate(zip(
            [dp_vals, r1_vals, bm_vals], bar_colors, bar_labels
        )):
            positions = x + (i - 1) * bar_width
            heights = [vals[p] for p in paradigms]
            bars = ax.bar(positions, heights, bar_width, color=color, label=label,
                          edgecolor="white", linewidth=0.5, zorder=3)
            # Annotate
            for bar, h in zip(bars, heights):
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8.5,
                            fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([PARADIGM_LABELS[p] for p in paradigms], fontsize=11)
        ax.set_title(row_label, fontsize=14, fontweight="bold")
        ax.set_ylim(0.5, 1.08)
        ax.set_ylabel("AUC" if ax_idx == 0 else "")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle("RQ1: Bankruptcy Prediction Performance", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig1_bk_prediction_matrix.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# FIGURE 2: RQ1 - Layer-wise AUC Curves
# ===================================================================
def fig2_layer_curves():
    """1x3 subplots (IC, SM, MW). Each: SAE DP line + Hidden DP line across layers."""

    paradigms = ["ic", "sm", "mw"]
    ga = all_analyses["goal_a_classification"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    # Per-paradigm annotation offsets: (sae_xy, hidden_xy) to avoid overlap
    # Each is (x_offset, y_offset) in points
    annot_offsets = {
        "ic": {"sae": (14, 8), "hidden": (-70, -20)},
        "sm": {"sae": (14, -25), "hidden": (-70, 12)},
        "mw": {"sae": (14, -20), "hidden": (-70, 10)},
    }

    for idx, p in enumerate(paradigms):
        ax = axes[idx]

        # SAE per-layer (42 layers, every layer)
        sae_layers_data = ga[p]["sae"]  # list of dicts
        sae_layers = [d["layer"] for d in sae_layers_data]
        sae_aucs = [d["auc"] for d in sae_layers_data]

        # Hidden per-layer (12 sampled layers)
        hidden_layers_data = ga[p]["hidden"]  # list of dicts
        hidden_layers = [d["layer"] for d in hidden_layers_data]
        hidden_aucs = [d["auc"] for d in hidden_layers_data]

        # Plot
        ax.plot(sae_layers, sae_aucs, "-o", color=COLORS[0], markersize=3,
                label="SAE Features", zorder=3)
        ax.plot(hidden_layers, hidden_aucs, "-s", color=COLORS[1], markersize=5,
                label="Hidden States", zorder=3)

        # Mark peak layers
        sae_peak_idx = int(np.argmax(sae_aucs))
        hidden_peak_idx = int(np.argmax(hidden_aucs))

        sae_off = annot_offsets[p]["sae"]
        hid_off = annot_offsets[p]["hidden"]

        ax.plot(sae_layers[sae_peak_idx], sae_aucs[sae_peak_idx], "*",
                color=COLORS[0], markersize=16, zorder=5)
        ax.annotate(f"L{sae_layers[sae_peak_idx]}: {sae_aucs[sae_peak_idx]:.3f}",
                    (sae_layers[sae_peak_idx], sae_aucs[sae_peak_idx]),
                    textcoords="offset points", xytext=sae_off,
                    fontsize=9.5, color=COLORS[0], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.2))

        ax.plot(hidden_layers[hidden_peak_idx], hidden_aucs[hidden_peak_idx], "*",
                color=COLORS[1], markersize=16, zorder=5)
        ax.annotate(f"L{hidden_layers[hidden_peak_idx]}: {hidden_aucs[hidden_peak_idx]:.3f}",
                    (hidden_layers[hidden_peak_idx], hidden_aucs[hidden_peak_idx]),
                    textcoords="offset points", xytext=hid_off,
                    fontsize=9.5, color=COLORS[1], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.2))

        ax.set_title(f"{PARADIGM_SHORT[p]}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Layer")
        if idx == 0:
            ax.set_ylabel("AUC (Decision-Point BK)")
        ax.set_xlim(-1, 42)

        # Set per-paradigm y-limits to show variation better
        all_vals = sae_aucs + hidden_aucs
        y_lo = min(all_vals) - 0.01
        y_hi = max(all_vals) + 0.015
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.suptitle("RQ1: Layer-wise BK Prediction (SAE vs Hidden States)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig2_layer_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# FIGURE 3: RQ2 - Cross-Domain Transfer Heatmap
# ===================================================================
def fig3_cross_domain_heatmap():
    """2x1 subplot: SAE transfer matrix (left), Hidden transfer matrix (right).
    3x3 heatmap: rows=train, cols=test. Diagonal=within-domain."""

    paradigms = ["ic", "sm", "mw"]
    paradigm_labels = ["IC", "SM", "MW"]

    # Build 3x3 matrices
    # SAE transfer
    sae_ct = rq2["sae_cross_domain_transfer"]
    sae_dp = rq1["sae_bk_dp"]

    sae_matrix = np.zeros((3, 3))
    for i, src in enumerate(paradigms):
        for j, tgt in enumerate(paradigms):
            if i == j:
                sae_matrix[i, j] = sae_dp[src]["best_auc"]
            else:
                key = f"{src}_to_{tgt}"
                sae_matrix[i, j] = sae_ct[key]["best_transfer_auc"]

    # Hidden transfer
    hid_ct = rq2["hidden_cross_domain_transfer"]
    hid_dp = rq1["hidden_bk_dp"]

    hid_matrix = np.zeros((3, 3))
    for i, src in enumerate(paradigms):
        for j, tgt in enumerate(paradigms):
            if i == j:
                hid_matrix[i, j] = hid_dp[src]["best_auc"]
            else:
                key = f"{src}_to_{tgt}"
                hid_matrix[i, j] = hid_ct[key]["best_auc"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rg", ["#D32F2F", "#FFEB3B", "#388E3C"], N=256)

    for ax, matrix, title in zip(axes,
                                  [sae_matrix, hid_matrix],
                                  ["SAE Features", "Hidden States"]):
        im = ax.imshow(matrix, cmap=cmap, vmin=0.5, vmax=1.0, aspect="equal")

        # Annotate cells
        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                text_color = "white" if val < 0.7 else "black"
                fontw = "bold" if i == j else "normal"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=13, fontweight=fontw, color=text_color)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(paradigm_labels, fontsize=12)
        ax.set_yticklabels(paradigm_labels, fontsize=12)
        ax.set_xlabel("Test Paradigm", fontsize=12)
        ax.set_ylabel("Train Paradigm", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add grid lines
        for edge in [-0.5, 0.5, 1.5, 2.5]:
            ax.axhline(edge, color="white", linewidth=2)
            ax.axvline(edge, color="white", linewidth=2)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("AUC", fontsize=12)

    fig.suptitle("RQ2: Cross-Domain Transfer Performance",
                 fontsize=15, fontweight="bold", y=1.02)
    out = OUT_DIR / "fig3_cross_domain_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# FIGURE 4: RQ3 - Condition Encoding
# ===================================================================
def fig4_condition_encoding():
    """Horizontal bar chart showing AUC for each condition type,
    grouped by paradigm with color coding."""

    ce = rq3["condition_encoding"]
    cm = rq3["condition_marginal_effects"]

    # Build items: (label, auc, paradigm_group)
    items = []

    # Bet type (binary) -- always AUC=1.0 for all
    items.append(("Bet Type (IC)", ce["bet_type_ic"]["best_auc"], "ic"))
    items.append(("Bet Type (SM)", ce["bet_type_sm"]["best_auc"], "sm"))
    items.append(("Bet Type (MW)", ce["bet_type_mw"]["best_auc"], "mw"))

    # Bet constraint (4-class, IC only)
    items.append(("Bet Constraint\n4-class (IC)", ce["ic_bet_constraint"]["best_auc"], "ic"))

    # Prompt condition (4-class, IC only)
    items.append(("Prompt Condition\n4-class (IC)", ce["ic_prompt_condition"]["best_auc"], "ic"))

    # Prompt components from condition_marginal_effects
    # SM has G, M, R, W, P components
    sm_comp = cm["sm"]["component_marginal"]
    for comp in ["G", "M", "R", "W", "P"]:
        if comp in sm_comp:
            auc_with = sm_comp[comp].get("auc_with", np.nan)
            auc_without = sm_comp[comp].get("auc_without", np.nan)
            # Use the overall BK-AUC difference is not so meaningful;
            # instead report the per-component BK rate difference as supplementary.
            # For encoding, use the SAE classification AUC for that component from condition_encoding
            pass

    # Since prompt_components is empty in condition_encoding, we use the
    # component_marginal BK rate differences instead. But let's focus on
    # what has actual encoding AUC values.

    # Color mapping
    paradigm_colors = {"ic": COLORS[0], "sm": COLORS[1], "mw": COLORS[2]}

    # Reverse for horizontal bar (top-to-bottom reading)
    items = items[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = np.arange(len(items))
    labels = [it[0] for it in items]
    aucs = [it[1] for it in items]
    colors = [paradigm_colors[it[2]] for it in items]

    bars = ax.barh(y_pos, aucs, color=colors, edgecolor="white", linewidth=0.5, height=0.6, zorder=3)

    # Annotate
    for bar, auc in zip(bars, aucs):
        if not np.isnan(auc):
            ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{auc:.3f}", ha="left", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("AUC", fontsize=12)
    ax.set_xlim(0.5, 1.10)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=0.6, alpha=0.3)

    # Legend for paradigm colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[0], label="Investment Choice"),
        Patch(facecolor=COLORS[1], label="Slot Machine"),
        Patch(facecolor=COLORS[2], label="Mystery Wheel"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10, framealpha=0.9)

    ax.set_title("RQ3: Experimental Condition Encoding in SAE Features",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "fig4_condition_encoding.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# FIGURE 5: RQ3 - G Component Behavioral Effect
# ===================================================================
def fig5_g_component_effect():
    """Bar chart: BK rate with_G vs without_G for each paradigm.
    Add Fisher p-value annotations."""

    paradigms = ["ic", "sm", "mw"]

    # Gather G component marginal data
    g_data = {}
    for p in paradigms:
        comp = cond_v2[p]["component_marginal"]
        if "G" in comp:
            g_data[p] = comp["G"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(paradigms))
    bar_width = 0.3

    with_g_rates = []
    without_g_rates = []
    fisher_ps = []

    for p in paradigms:
        gd = g_data[p]
        with_g_rates.append(gd["bk_rate_with"] * 100)
        without_g_rates.append(gd["bk_rate_without"] * 100)
        fisher_ps.append(gd["fisher_p"])

    bars_without = ax.bar(x - bar_width / 2, without_g_rates, bar_width,
                          color="#56B4E9", label="Without G (Goal)", edgecolor="white",
                          linewidth=0.5, zorder=3)
    bars_with = ax.bar(x + bar_width / 2, with_g_rates, bar_width,
                       color="#D55E00", label="With G (Goal)", edgecolor="white",
                       linewidth=0.5, zorder=3)

    # Annotate bars
    for bars in [bars_without, bars_with]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.15,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add Fisher p-value annotations with brackets
    for i, (p_name, fp) in enumerate(zip(paradigms, fisher_ps)):
        y_max = max(with_g_rates[i], without_g_rates[i])
        bracket_y = y_max + 1.5

        # Draw bracket
        ax.plot([x[i] - bar_width / 2, x[i] - bar_width / 2,
                 x[i] + bar_width / 2, x[i] + bar_width / 2],
                [bracket_y - 0.3, bracket_y, bracket_y, bracket_y - 0.3],
                color="black", linewidth=1.0, zorder=4)

        # p-value text
        if fp < 0.001:
            p_text = f"p < 0.001"
        elif fp < 0.01:
            p_text = f"p = {fp:.3f}"
        elif fp < 0.05:
            p_text = f"p = {fp:.3f}*"
        else:
            p_text = f"p = {fp:.3f} (n.s.)"

        ax.text(x[i], bracket_y + 0.3, p_text, ha="center", va="bottom",
                fontsize=9.5, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([PARADIGM_SHORT[p] for p in paradigms], fontsize=13, fontweight="bold")
    ax.set_ylabel("Bankruptcy Rate (%)", fontsize=12)
    ax.set_ylim(0, max(with_g_rates) * 1.45)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # Add context box
    n_info = []
    for p in paradigms:
        gd = g_data[p]
        n_info.append(f"{PARADIGM_SHORT[p]}: n_with={gd['n_with']}, n_without={gd['n_without']}")
    info_text = "\n".join(n_info)
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    ax.set_title("RQ3: Effect of Goal Prompt Component (G) on Bankruptcy Rate",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "fig5_g_component_effect.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating RQ figures (v6)...")
    print()

    print("[1/5] BK Prediction Comparison Matrix")
    fig1_bk_prediction_matrix()

    print("[2/5] Layer-wise AUC Curves")
    fig2_layer_curves()

    print("[3/5] Cross-Domain Transfer Heatmap")
    fig3_cross_domain_heatmap()

    print("[4/5] Condition Encoding")
    fig4_condition_encoding()

    print("[5/5] G Component Behavioral Effect")
    fig5_g_component_effect()

    print()
    print("All 5 figures saved to:", OUT_DIR)
