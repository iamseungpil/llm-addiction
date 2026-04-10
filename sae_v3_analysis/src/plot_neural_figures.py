#!/usr/bin/env python3
"""
Generate paper figures from the audited neural metrics manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
AUDIT_PATH = ROOT / "results" / "paper_neural_audit.json"
OUT_DIR = Path("/home/v-seungplee/LLM_Addiction_NMT_KOR/images")
OUT_DIR.mkdir(exist_ok=True)


def load_audit() -> dict:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {AUDIT_PATH}. Run build_paper_neural_audit.py first."
        )
    return json.loads(AUDIT_PATH.read_text())


def build_neural_summary(audit: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"width_ratios": [1.1, 1]})

    # Panel (a): archived I_LC readout across six model-task combinations.
    ilc_order = [
        ("gemma_sm", "Gemma\nSM"),
        ("gemma_ic", "Gemma\nIC"),
        ("gemma_mw", "Gemma\nMW"),
        ("llama_sm", "LLaMA\nSM"),
        ("llama_ic", "LLaMA\nIC"),
        ("llama_mw", "LLaMA\nMW"),
    ]
    ilc_vals = [audit["rq1_ilc"][key]["r2"] for key, _ in ilc_order]
    ilc_colors = ["#d95f02"] * 3 + ["#1b9e77"] * 3
    x = np.arange(len(ilc_order))
    ax1.bar(x, ilc_vals, color=ilc_colors, alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels([label for _, label in ilc_order], fontsize=9)
    ax1.set_ylabel(r"$R^2$", fontsize=11)
    ax1.set_title("(a) I$_{LC}$ Readout Across Tasks", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(ilc_vals) * 1.18)
    for xi, val in zip(x, ilc_vals):
        ax1.text(xi, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(0, color="black", lw=0.8, alpha=0.35)

    # Panel (b): audited condition modulation for I_BA in slot machine.
    cond_labels = ["All\nVar", "+G", "-G", "Fixed"]
    gemma = audit["rq3_condition_i_ba"]["gemma_sm_i_ba"]["subsets"]
    llama = audit["rq3_condition_i_ba"]["llama_sm_i_ba"]["subsets"]
    gemma_vals = [gemma["all_variable"]["r2"], gemma["plus_G"]["r2"], gemma["minus_G"]["r2"], gemma["fixed_all"]["r2"]]
    llama_vals = [llama["all_variable"]["r2"], llama["plus_G"]["r2"], llama["minus_G"]["r2"], llama["fixed_all"]["r2"]]
    x2 = np.arange(len(cond_labels))
    width = 0.35
    ax2.bar(x2 - width / 2, gemma_vals, width, label="Gemma", color="#d95f02", alpha=0.82)
    ax2.bar(x2 + width / 2, llama_vals, width, label="LLaMA", color="#1b9e77", alpha=0.82)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cond_labels, fontsize=9)
    ax2.set_ylabel(r"$R^2$", fontsize=11)
    ax2.set_title("(b) Condition Modulation of I$_{BA}$", fontsize=12, fontweight="bold")
    ax2.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax2.legend(fontsize=9, loc="upper right")
    ymax = max(gemma_vals + llama_vals)
    ax2.set_ylim(-0.04, ymax * 1.22)

    ratio = gemma["plus_G"]["r2"] / gemma["minus_G"]["r2"]
    ax2.annotate(
        f"{ratio:.2f}x",
        xy=(1 - width / 2, gemma["plus_G"]["r2"]),
        xytext=(0.45, ymax * 1.08),
        textcoords="data",
        fontsize=9,
        color="#d95f02",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.2),
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "neural_analysis_combined.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUT_DIR / "neural_analysis_combined.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def build_cross_transfer() -> None:
    # Representative archived cross-task transfer matrix from the paper's negative-transfer story.
    paradigms = ["SM", "IC", "MW"]
    matrix = np.array(
        [
            [0.24, -0.05, -2.01],
            [-0.10, 0.48, -0.08],
            [-0.06, -0.04, 0.48],
        ]
    )
    display = np.clip(matrix, -0.5, 0.6)

    fig, ax = plt.subplots(figsize=(5.1, 4.1))
    im = ax.imshow(display, cmap="RdYlGn", vmin=-0.5, vmax=0.6, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val < 0 else "black"
            weight = "bold" if i == j else "normal"
            label = f"{val:.2f}" if abs(val) < 1 else f"{val:.1f}"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=11, fontweight=weight)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(paradigms, fontsize=11)
    ax.set_yticklabels(paradigms, fontsize=11)
    ax.set_xlabel("Test Paradigm", fontsize=11)
    ax.set_ylabel("Train Paradigm", fontsize=11)
    ax.set_title("Cross-Paradigm Transfer ($R^2$, Representative Gemma Run)", fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(r"$R^2$", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cross_paradigm_transfer.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(OUT_DIR / "cross_paradigm_transfer.png", bbox_inches="tight", dpi=160)
    plt.close(fig)


def main() -> None:
    audit = load_audit()
    build_neural_summary(audit)
    build_cross_transfer()
    print("Saved audited paper figures.")


if __name__ == "__main__":
    main()
