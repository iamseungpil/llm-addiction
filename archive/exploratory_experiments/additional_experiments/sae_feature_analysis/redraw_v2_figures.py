#!/usr/bin/env python3
"""
Redraw V2 within-model figures with fixed scaling and layout.

Fixes:
- Fig A: Linear AUC scale (not log error) — LLaMA was invisible with log scale
- Fig A: Dual-panel layout with AUC on top, active features on bottom
- Fig B: Shared colorbar, consistent eta2 annotation positioning
- Fig D: Separate panels for LLaMA and Gemma (different scales)
- Fig E: Already fine, minor polish
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

JSON_DIR = Path(__file__).parent / "results" / "within_model_v2" / "json"
FIG_DIR = Path(__file__).parent / "results" / "within_model_v2" / "figures_improved"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "llama": "#1f77b4",
    "gemma": "#d62728",
    "variable": "#ff7f0e",
    "fixed": "#2ca02c",
}
MODEL_LABELS = {"llama": "LLaMA-3.1-8B", "gemma": "Gemma-2-9B"}

# Load data
clf_dict = {}
anova_dict = {}
for model in ["llama", "gemma"]:
    with open(JSON_DIR / f"classification_{model}_20260302_150156.json") as f:
        clf_dict[model] = json.load(f)
    with open(JSON_DIR / f"anova_{model}_20260302_150156.json") as f:
        anova_dict[model] = json.load(f)


# =====================================================================
# Fig A: Classification AUC (linear scale) + Active Features
# =====================================================================
def fig_a():
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    for col, model in enumerate(["llama", "gemma"]):
        clf = clf_dict[model]
        results = [r for r in clf["all_games"] if not r.get("skipped", False)]
        if not results:
            continue

        layers = [r["layer"] for r in results]
        auc_mean = [r["auc_mean"] for r in results]
        auc_std = [r["auc_std"] for r in results]

        # --- Top: AUC (linear scale) ---
        ax1 = fig.add_subplot(gs[0, col])
        ax1.plot(layers, auc_mean, "-o", color=COLORS[model], markersize=4, lw=1.8,
                 label=f"{MODEL_LABELS[model]} (all)")
        ax1.fill_between(layers,
                         [m - s for m, s in zip(auc_mean, auc_std)],
                         [m + s for m, s in zip(auc_mean, auc_std)],
                         alpha=0.15, color=COLORS[model])

        # Variable/Fixed subsets
        for sub, style, clr, label in [
            ("variable_only", "--", COLORS["variable"], "Variable"),
            ("fixed_only", "--", COLORS["fixed"], "Fixed"),
        ]:
            sr = [r for r in clf.get(sub, []) if not r.get("skipped", False)]
            if sr:
                ax1.plot([r["layer"] for r in sr],
                         [r["auc_mean"] for r in sr],
                         style, color=clr, lw=1.0, alpha=0.7, label=label)

        ax1.axhline(0.5, color="gray", ls=":", alpha=0.5, lw=0.8, label="Chance")
        ax1.set_ylabel("Classification AUC", fontsize=11)
        ax1.set_title(MODEL_LABELS[model], fontsize=13, fontweight="bold")
        ax1.legend(fontsize=7, loc="best")
        ax1.grid(True, alpha=0.3)

        # Set y-axis range based on data
        min_auc = min(auc_mean) - 0.05
        max_auc = max(auc_mean) + 0.05
        ax1.set_ylim(max(0.45, min_auc), min(1.0, max_auc))

        # Annotate best layer
        best_idx = int(np.argmax(auc_mean))
        ax1.annotate(f"L{layers[best_idx]}: {auc_mean[best_idx]:.3f}",
                     xy=(layers[best_idx], auc_mean[best_idx]),
                     xytext=(10, -15), textcoords="offset points",
                     fontsize=8, fontweight="bold", color=COLORS[model],
                     arrowprops=dict(arrowstyle="->", color=COLORS[model], lw=0.8))

        # --- Bottom: Active features ---
        ax2 = fig.add_subplot(gs[1, col])
        nf = [r.get("n_features", 0) for r in results]
        ax2.bar(layers, nf, color=COLORS[model], alpha=0.6, width=0.8)
        max_idx = int(np.argmax(nf))
        ax2.annotate(f"{nf[max_idx]:,}", xy=(layers[max_idx], nf[max_idx]),
                     xytext=(0, 5), textcoords="offset points",
                     fontsize=7, ha="center", fontweight="bold", color=COLORS[model])
        total = 131072 if model == "gemma" else 32768
        avg_pct = np.mean([100 * n / total for n in nf])
        ax2.text(0.98, 0.95,
                 f"Avg: {avg_pct:.1f}% of {total // 1000}K total\npass activation filter",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_ylabel("Active SAE Features\n(>1% activation rate)", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("SAE Features Predict Gambling Outcome — V2 (Decision-Point Fix)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_a_error_rate_improved.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_a_error_rate_improved.png")


# =====================================================================
# Fig B: Top-10 Interaction Heatmap (improved layout)
# =====================================================================
def fig_b():
    cells = ["variable_bankrupt", "variable_safe", "fixed_bankrupt", "fixed_safe"]
    cell_labels = ["Var-BK", "Var-Safe", "Fix-BK", "Fix-Safe"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, model in zip(axes, ["llama", "gemma"]):
        anova = anova_dict[model]
        lk = "layer_summary"

        all_sig = []
        for lr in anova[lk]:
            fk = "top_features"
            for feat in lr.get(fk, []):
                if feat.get("interaction_significant", True):
                    fc = dict(feat)
                    fc["layer"] = lr["layer"]
                    all_sig.append(fc)

        if not all_sig:
            ax.text(0.5, 0.5, f"No significant features\n({MODEL_LABELS[model]})",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)
            ax.set_title(f"Top-10 Interaction — {MODEL_LABELS[model]}", fontsize=11, fontweight="bold")
            continue

        all_sig.sort(key=lambda x: x["interaction_eta_sq"], reverse=True)
        top_n = min(10, len(all_sig))
        top = all_sig[:top_n]

        matrix = np.zeros((top_n, 4))
        for i, feat in enumerate(top):
            gm = feat["group_means"]
            for j, c in enumerate(cells):
                matrix[i, j] = gm.get(c, 0.0)

        row_means = matrix.mean(axis=1, keepdims=True)
        row_stds = matrix.std(axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1.0
        matrix_z = (matrix - row_means) / row_stds

        feat_labels = [f"L{f['layer']}/F{f['feature_idx']}" for f in top]
        eta_strs = [f"eta2={f['interaction_eta_sq']:.3f}" for f in top]

        im = ax.imshow(matrix_z, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(cell_labels, fontsize=10)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_labels, fontsize=9)

        # eta2 as text inside the plot
        for i, eta_str in enumerate(eta_strs):
            ax.text(4.2, i, eta_str, fontsize=7, va="center", color="gray")

        ax.set_title(f"Top-10 Interaction — {MODEL_LABELS[model]}", fontsize=11, fontweight="bold")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.15)
        cbar.set_label("Z-score", fontsize=9)

    fig.suptitle("V2: Top Interaction Features (bet_type × outcome)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_b_heatmap_combined_improved.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_b_heatmap_combined_improved.png")


# =====================================================================
# Fig D: Interaction eta² — separate panels for different scales
# =====================================================================
def fig_d():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    ETA_SQ_SMALL = 0.01
    ETA_SQ_MEDIUM = 0.06
    ETA_SQ_LARGE = 0.14

    for ax, model in zip(axes, ["llama", "gemma"]):
        anova = anova_dict[model]
        lk = "layer_summary"

        layers, medians, p75s, maxes = [], [], [], []
        for lr in anova[lk]:
            fk = "top_features"
            eta = [r["interaction_eta_sq"] for r in lr.get(fk, []) if r.get("interaction_significant", False)]
            if not eta:
                layers.append(lr["layer"])
                medians.append(0)
                p75s.append(0)
                maxes.append(0)
                continue
            layers.append(lr["layer"])
            medians.append(np.median(eta))
            p75s.append(np.percentile(eta, 75))
            maxes.append(np.max(eta))

        color = COLORS[model]
        ax.plot(layers, medians, "-", color=color, lw=2.0, label="Median")
        ax.plot(layers, maxes, ":", color=color, lw=1.2, alpha=0.7, label="Max")
        ax.fill_between(layers, medians, p75s, alpha=0.2, color=color)

        # Reference lines
        y_max = max(maxes) * 1.2 if max(maxes) > 0 else 0.02
        for y, label in [(ETA_SQ_SMALL, "Small"), (ETA_SQ_MEDIUM, "Medium"), (ETA_SQ_LARGE, "Large")]:
            if y < y_max:
                ax.axhline(y=y, color="gray", ls="--", alpha=0.4, lw=0.8)
                ax.text(max(layers) + 0.5, y, label, fontsize=7, color="gray", va="bottom")

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("eta² (Interaction Effect Size)", fontsize=10)
        ax.set_title(f"{MODEL_LABELS[model]}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=0, top=y_max)

        # Annotate peak
        peak_idx = int(np.argmax(maxes))
        if maxes[peak_idx] > 0:
            ax.annotate(f"L{layers[peak_idx]}\n{maxes[peak_idx]:.4f}",
                        xy=(layers[peak_idx], maxes[peak_idx]),
                        xytext=(10, -10), textcoords="offset points",
                        fontsize=8, fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    fig.suptitle("Betting Condition × Outcome Interaction Effect (V2 Decision-Point Fix)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_d_eta_combined_improved.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_d_eta_combined_improved.png")


# =====================================================================
# Fig E: Interaction Feature Ratio
# =====================================================================
def fig_e():
    fig, ax = plt.subplots(figsize=(12, 5))

    for model in ["llama", "gemma"]:
        anova = anova_dict[model]
        lk = "layer_summary"

        layers, pcts, n_tested_list = [], [], []
        for lr in anova[lk]:
            nt = lr["n_tested"]
            ns = lr["n_significant"]
            if nt > 0:
                layers.append(lr["layer"])
                pcts.append(100.0 * ns / nt)
                n_tested_list.append(nt)

        if not layers:
            continue

        color = COLORS[model]
        ax.plot(layers, pcts, "-o", color=color, lw=1.8, markersize=4, label=MODEL_LABELS[model])

        # Find peak (exclude artifacts with tiny n)
        valid_pcts = list(pcts)
        for i, n in enumerate(n_tested_list):
            if n < 10:
                valid_pcts[i] = -1

        peak_idx = int(np.argmax(valid_pcts))
        ax.plot(layers[peak_idx], pcts[peak_idx], "o", color=color,
                markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.annotate(f"L{layers[peak_idx]} ({pcts[peak_idx]:.1f}%)",
                    xy=(layers[peak_idx], pcts[peak_idx]),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold")

        # Mark L41 artifact
        if model == "gemma":
            for i, (l, n) in enumerate(zip(layers, n_tested_list)):
                if l == 41 and n < 10:
                    ax.annotate(f"L41 (n={n}, artifact)",
                                xy=(l, pcts[i]),
                                xytext=(-60, -25), textcoords="offset points",
                                fontsize=7.5, color="gray",
                                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Features with Significant\nBetting × Outcome Interaction (%)", fontsize=10)
    ax.set_title("Interaction Feature Ratio by Layer (V2 Decision-Point Fix)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(-2, 105)
    ax.grid(axis="y", alpha=0.3)

    ax.text(0.98, 0.45,
            "Significant interaction = feature responds\n"
            "differently to bankruptcy vs safe outcomes\n"
            "depending on betting autonomy (var/fixed).\n\n"
            "V2: fixed circular reasoning in bankruptcy prompts.",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_e_interaction_improved.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_e_interaction_improved.png")


# =====================================================================
if __name__ == "__main__":
    print("Redrawing V2 figures...")
    fig_a()
    fig_b()
    fig_d()
    fig_e()
    print("Done!")
