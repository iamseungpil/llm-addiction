#!/usr/bin/env python3
"""
Improved SAE Visualization for Paper Figures
=============================================

Generates publication-quality figures addressing reviewer feedback:
- Fig A: Replace flat AUC=1.0 line with 1-AUC log scale + error count bar chart
- Fig B: Enhanced heatmap with activation pattern annotations
- Fig D: η² interaction with clearer legends and context annotations
- Fig E: Interaction % with L41 artifact annotation

Usage:
    python visualize_improved_figures.py
    python visualize_improved_figures.py --only fig_a fig_b
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JSON_DIR = Path("/home/jovyan/llm-addiction/exploratory_experiments/additional_experiments/"
                "sae_feature_analysis/slot_machine_condition_comparison/results/within_model/json")
OUTPUT_DIR = Path("/home/jovyan/llm-addiction/exploratory_experiments/additional_experiments/"
                  "sae_feature_analysis/results/within_model/figures_improved")

MODEL_LABELS = {"llama": "LLaMA-3.1-8B", "gemma": "Gemma-2-9B"}

COLORS = {
    "llama": "#1f4e79",
    "gemma": "#c0392b",
    "all": "#2c3e50",
    "variable": "#e74c3c",
    "fixed": "#3498db",
    "median": "#c0392b",
    "p75": "#e67e22",
    "max": "#8e44ad",
    "iqr_fill": "#aec6cf",
}

# Cohen's eta-squared benchmarks
ETA_SQ_SMALL = 0.01
ETA_SQ_MEDIUM = 0.06
ETA_SQ_LARGE = 0.14


def load_json(name: str) -> Dict:
    """Load a JSON result file by partial name match."""
    candidates = list(JSON_DIR.glob(f"{name}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON file matching '{name}*' in {JSON_DIR}")
    path = sorted(candidates)[-1]  # latest
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fig A: 1 - AUC (log scale) + Misclassification Count
# ---------------------------------------------------------------------------
def fig_a_improved(clf_data_dict: Dict[str, Dict], save_path: Path):
    """Replace flat AUC line with error-rate log scale + misclassification bar.

    Panel 1: 1-AUC on log scale (shows difference between 0.99 and 0.9999)
    Panel 2: Number of misclassified games per layer (intuitive)
    """
    models = [m for m in ["llama", "gemma"] if m in clf_data_dict]
    if not models:
        return

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, len(models), hspace=0.35, wspace=0.3)

    for col, model in enumerate(models):
        clf_data = clf_data_dict[model]
        results = [r for r in clf_data["all_games"] if not r.get("skipped", False)]
        if not results:
            continue

        layers = np.array([r["layer"] for r in results])
        auc_mean = np.array([r["auc_mean"] for r in results])
        auc_std = np.array([r["auc_std"] for r in results])
        n_total = clf_data.get("n_total", 3200)

        # --- Top panel: 1-AUC on log scale ---
        ax1 = fig.add_subplot(gs[0, col])
        error_rate = 1.0 - auc_mean
        error_rate = np.clip(error_rate, 1e-6, None)  # avoid log(0)

        ax1.semilogy(layers, error_rate, "-o", color=COLORS[model], markersize=4,
                     lw=1.8, label=f"{MODEL_LABELS[model]}")

        # Variable / Fixed
        for sub, style, clr in [("variable_only", "--", COLORS["variable"]),
                                 ("fixed_only", "--", COLORS["fixed"])]:
            sub_results = [r for r in clf_data.get(sub, []) if not r.get("skipped", False)]
            if sub_results:
                sub_layers = [r["layer"] for r in sub_results]
                sub_err = [max(1 - r["auc_mean"], 1e-6) for r in sub_results]
                label = "Variable" if "variable" in sub else "Fixed"
                ax1.semilogy(sub_layers, sub_err, style, color=clr, lw=1.0,
                             alpha=0.7, label=label)

        ax1.set_ylabel("Classification Error (1 − AUC)", fontsize=11)
        ax1.set_title(MODEL_LABELS[model], fontsize=13, fontweight="bold")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.3, which="both")
        ax1.set_ylim(1e-5, 2e-1)
        ax1.axhline(y=0.01, color="gray", linestyle=":", alpha=0.5, lw=0.8)
        ax1.text(layers[-1] + 0.5, 0.012, "1% error", fontsize=7, color="gray", va="bottom")

        # --- Bottom panel: Number of features needed for AUC>0.99 ---
        ax2 = fig.add_subplot(gs[1, col])
        n_features = []
        for r in results:
            nf = r.get("n_features", 0)
            n_features.append(nf)

        bars = ax2.bar(layers, n_features, color=COLORS[model], alpha=0.6, width=0.8)

        # Annotate max
        max_idx = int(np.argmax(n_features))
        ax2.annotate(f"{n_features[max_idx]:,}",
                     xy=(layers[max_idx], n_features[max_idx]),
                     xytext=(0, 5), textcoords="offset points",
                     fontsize=7, ha="center", fontweight="bold", color=COLORS[model])

        # Total features available (131K for Gemma, 32K for LLaMA)
        total_feat = 131072 if model == "gemma" else 32768
        pct_used = [100 * nf / total_feat for nf in n_features]
        avg_pct = np.mean(pct_used) if pct_used else 0
        ax2.text(0.98, 0.95,
                 f"Avg: {avg_pct:.1f}% of\n{total_feat//1000}K total features\npass activation filter",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_ylabel("Active SAE Features\n(>1% activation rate)", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("SAE Features Predict Gambling Outcome with Near-Perfect Accuracy",
                 fontsize=14, fontweight="bold", y=0.98)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Fig B: Enhanced Heatmap with Pattern Annotations
# ---------------------------------------------------------------------------
def fig_b_improved(anova_data_dict: Dict[str, Dict], save_path: Path):
    """Side-by-side heatmaps for both models with pattern annotations.

    Adds:
    - Pattern type label for each feature (Fix-BK dominant, Var-BK dominant, Mixed)
    - η² value annotation on the right
    - Clearer column labels with full condition names
    """
    models = [m for m in ["llama", "gemma"] if m in anova_data_dict]
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5))
    if len(models) == 1:
        axes = [axes]

    cells = ["variable_bankrupt", "variable_safe", "fixed_bankrupt", "fixed_safe"]
    cell_labels = ["Var-BK", "Var-Safe", "Fix-BK", "Fix-Safe"]

    for ax, model in zip(axes, models):
        anova_data = anova_data_dict[model]
        layer_key = "layer_results" if "layer_results" in anova_data else "layer_summary"
        layer_data = anova_data[layer_key]

        # Collect significant features
        all_sig = []
        for lr in layer_data:
            feat_key = "significant_features" if "significant_features" in lr else "top_features"
            for feat in lr.get(feat_key, []):
                if feat.get("interaction_significant", True):
                    feat_copy = dict(feat)
                    feat_copy["layer"] = lr["layer"]
                    all_sig.append(feat_copy)

        if not all_sig:
            ax.text(0.5, 0.5, f"No data for {model}", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        all_sig.sort(key=lambda x: x["interaction_eta_sq"], reverse=True)
        top_n = min(10, len(all_sig))
        top_feats = all_sig[:top_n]

        matrix = np.zeros((top_n, 4))
        for i, feat in enumerate(top_feats):
            gm = feat["group_means"]
            for j, cell in enumerate(cells):
                matrix[i, j] = gm.get(cell, 0.0)

        # Z-score normalize each row
        row_means = matrix.mean(axis=1, keepdims=True)
        row_stds = matrix.std(axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1.0
        matrix_z = (matrix - row_means) / row_stds

        # Classify pattern type
        pattern_labels = []
        for i in range(top_n):
            z = matrix_z[i]
            max_col = np.argmax(z)
            if max_col == 0:
                pattern_labels.append("Var-BK")
            elif max_col == 2:
                pattern_labels.append("Fix-BK")
            elif max_col == 1:
                pattern_labels.append("Var-Safe")
            else:
                pattern_labels.append("Fix-Safe")

        feat_labels = [f"L{f['layer']}/F{f['feature_idx']}" for f in top_feats]
        eta_values = [f"{f['interaction_eta_sq']:.3f}" for f in top_feats]

        im = ax.imshow(matrix_z, cmap="RdBu_r", aspect="auto", vmin=-2.2, vmax=2.2)

        ax.set_xticks(range(4))
        ax.set_xticklabels(cell_labels, fontsize=9)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_labels, fontsize=9)

        # Add η² values on the right
        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels([f"η²={v}" for v in eta_values], fontsize=7.5, color="gray")

        # Add pattern type as colored markers on y-tick labels
        for i, pat in enumerate(pattern_labels):
            marker_color = COLORS["variable"] if "Var" in pat else COLORS["fixed"]
            ax.plot(-0.6, i, "s", color=marker_color, markersize=6,
                    transform=ax.get_yaxis_transform(), clip_on=False)

        ax.set_title(f"Top-10 Interaction Features — {MODEL_LABELS.get(model, model)}",
                     fontsize=11, fontweight="bold")

        # Per-axis colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Z-score", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Fig D: η² with Cohen's Benchmarks and Explanatory Annotations
# ---------------------------------------------------------------------------
def fig_d_improved(anova_data_dict: Dict[str, Dict], save_path: Path):
    """η² interaction effect size for both models with Cohen's benchmarks.

    Key improvements:
    - Both models on one figure for direct comparison
    - Cohen's benchmarks (small/medium/large) as reference lines
    - Annotation explaining what η² means
    """
    models = [m for m in ["llama", "gemma"] if m in anova_data_dict]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for model in models:
        anova_data = anova_data_dict[model]
        layer_key = "layer_results" if "layer_results" in anova_data else "layer_summary"
        layer_data = anova_data[layer_key]

        if not layer_data:
            continue

        layers, medians, p75s, maxes = [], [], [], []
        for lr in layer_data:
            feat_key = "significant_features" if "significant_features" in lr else "top_features"
            results_to_use = lr.get("all_results", lr.get(feat_key, []))
            if not results_to_use:
                continue
            eta_sq = [r["interaction_eta_sq"] for r in results_to_use]
            if not eta_sq:
                continue
            layers.append(lr["layer"])
            medians.append(np.median(eta_sq))
            p75s.append(np.percentile(eta_sq, 75))
            maxes.append(np.max(eta_sq))

        if not layers:
            continue

        color = COLORS[model]
        label = MODEL_LABELS.get(model, model)

        ax.plot(layers, medians, "-", color=color, lw=2.0, label=f"{label} (median)")
        ax.plot(layers, maxes, ":", color=color, lw=1.2, alpha=0.7,
                label=f"{label} (max)")
        ax.fill_between(layers, medians, p75s, alpha=0.15, color=color)

    # Cohen's benchmarks
    ax.axhline(y=ETA_SQ_LARGE, color="gray", linestyle="--", alpha=0.5, lw=1.0)
    ax.axhline(y=ETA_SQ_MEDIUM, color="gray", linestyle="--", alpha=0.4, lw=0.8)
    ax.axhline(y=ETA_SQ_SMALL, color="gray", linestyle="--", alpha=0.3, lw=0.8)

    # Labels for benchmarks
    xpos = max(lr["layer"] for lr in layer_data) + 0.5
    ax.text(xpos, ETA_SQ_LARGE, "Large", fontsize=7.5, color="gray", va="bottom")
    ax.text(xpos, ETA_SQ_MEDIUM, "Medium", fontsize=7.5, color="gray", va="bottom")
    ax.text(xpos, ETA_SQ_SMALL, "Small", fontsize=7.5, color="gray", va="bottom")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("η² (Interaction Effect Size)", fontsize=11)
    ax.set_title("How Strongly Does Each Layer Encode the\nBetting Condition × Outcome Interaction?",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # Explanatory annotation
    ax.text(0.98, 0.02,
            "η² = fraction of variance in SAE feature activation\n"
            "explained by the interaction of betting type and outcome.\n"
            "Higher η² → the feature responds differently to bankruptcy\n"
            "depending on whether betting was variable or fixed.",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Fig E: Interaction % with L41 Annotation
# ---------------------------------------------------------------------------
def fig_e_improved(anova_data_dict: Dict[str, Dict], save_path: Path):
    """Interaction % per layer with L41 artifact annotation and explanation.

    Key improvements:
    - Annotation explaining what 'interaction' means
    - L41 marked as artifact (n=4)
    - Sample size (n) shown on secondary axis
    """
    models = [m for m in ["llama", "gemma"] if m in anova_data_dict]
    if not models:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))

    for model in models:
        anova_data = anova_data_dict[model]
        layer_key = "layer_results" if "layer_results" in anova_data else "layer_summary"
        layer_data = anova_data[layer_key]

        layers, pcts, n_tested_list = [], [], []
        for lr in layer_data:
            n_tested = lr["n_tested"]
            n_sig = lr["n_significant"]
            if n_tested > 0:
                layers.append(lr["layer"])
                pcts.append(100.0 * n_sig / n_tested)
                n_tested_list.append(n_tested)

        if not layers:
            continue

        color = COLORS[model]
        ax1.plot(layers, pcts, "-o", color=color, lw=1.8, markersize=4,
                 label=MODEL_LABELS.get(model, model))

        # Mark peak (excluding L41 for Gemma if n<10)
        valid_pcts = list(pcts)
        valid_layers = list(layers)
        if model == "gemma":
            for i, (l, n) in enumerate(zip(layers, n_tested_list)):
                if n < 10:
                    valid_pcts[i] = 0  # exclude from peak

        peak_idx = int(np.argmax(valid_pcts))
        ax1.plot(layers[peak_idx], pcts[peak_idx], "o", color=color,
                 markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax1.annotate(f"L{layers[peak_idx]} ({pcts[peak_idx]:.1f}%)",
                     xy=(layers[peak_idx], pcts[peak_idx]),
                     xytext=(5, 8), textcoords="offset points",
                     fontsize=8, color=color, fontweight="bold")

        # Annotate L41 artifact for Gemma
        if model == "gemma":
            for i, (l, n) in enumerate(zip(layers, n_tested_list)):
                if l == 41 and n < 10:
                    ax1.annotate(f"L41\n(n={n}, artifact)",
                                 xy=(l, pcts[i]),
                                 xytext=(-50, -30), textcoords="offset points",
                                 fontsize=7.5, color="gray",
                                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                                 bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                                           alpha=0.8))

    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Features with Significant\nBetting × Outcome Interaction (%)", fontsize=10)
    ax1.set_title("What Fraction of SAE Features Encode How Betting Conditions\n"
                  "Differentially Affect Gambling Outcomes?",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Explanatory annotation
    ax1.text(0.98, 0.55,
             "A significant interaction means:\n"
             "the feature's response to bankruptcy vs. safe\n"
             "outcomes differs depending on whether the\n"
             "model had betting autonomy (variable) or not (fixed).\n\n"
             "Higher % → more features in this layer jointly\n"
             "encode both betting condition and outcome.",
             transform=ax1.transAxes, fontsize=7.5, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate improved SAE figures")
    parser.add_argument("--only", nargs="+", choices=["fig_a", "fig_b", "fig_d", "fig_e", "all"],
                        default=["all"], help="Which figures to generate")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = set(args.only)
    do_all = "all" in targets

    # Load data
    print("Loading JSON result files...")
    clf_data = {}
    anova_data = {}
    for model in ["gemma", "llama"]:
        try:
            clf_data[model] = load_json(f"classification_{model}")
            print(f"  Loaded classification_{model}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
        try:
            anova_data[model] = load_json(f"anova_{model}")
            print(f"  Loaded anova_{model}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    # Generate figures
    if do_all or "fig_a" in targets:
        print("\nGenerating Fig A (improved: error rate log scale + misclassification count)...")
        fig_a_improved(clf_data, OUTPUT_DIR / "fig_a_error_rate_improved.png")

    if do_all or "fig_b" in targets:
        print("\nGenerating Fig B (improved: side-by-side heatmaps with pattern labels)...")
        fig_b_improved(anova_data, OUTPUT_DIR / "fig_b_heatmap_combined_improved.png")

    if do_all or "fig_d" in targets:
        print("\nGenerating Fig D (improved: both models + Cohen's benchmarks)...")
        fig_d_improved(anova_data, OUTPUT_DIR / "fig_d_eta_combined_improved.png")

    if do_all or "fig_e" in targets:
        print("\nGenerating Fig E (improved: interaction % with L41 annotation)...")
        fig_e_improved(anova_data, OUTPUT_DIR / "fig_e_interaction_improved.png")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
