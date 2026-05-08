from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    # Model identity reuses the behavioral fixed/variable palette so Figures 5/7
    # match Figure 3's green/red saturation level. Gemma -> green (fixed anchor),
    # LLaMA -> red (variable anchor).
    "gemma": "#59A14F",      # green, matches fixed
    "llama": "#E15759",      # red, matches variable
    "fixed": "#59A14F",
    "variable": "#E15759",
    "option1": "#A0CBE8",
    "option2": "#BAB0AC",
    "option3": "#7F7F7F",
    "option4": "#E74C3C",
    "neutral": "#9D9D9D",
    "grid": "#D9D9D9",
    "accent": "#EDC948",
}


def use_paper_style(base_size: float = 12.0) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": base_size,
            "axes.titlesize": base_size + 0.5,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 0.5,
            "ytick.labelsize": base_size - 0.5,
            "legend.fontsize": base_size - 1.0,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.45,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CFCFCF",
            "legend.framealpha": 0.95,
            "figure.constrained_layout.use": True,
        }
    )


def style_axes(ax, *, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.6, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_title(ax, label: str, title: str) -> None:
    text = f"{label} {title}".strip()
    ax.set_title(text, loc="left", fontweight="bold", pad=6)


def annotate_bars(ax, bars, *, fmt: str = "{:.1f}", suffix: str = "", color: str = "black", size: float = 8.5) -> None:
    for bar in bars:
        height = bar.get_height()
        if abs(height) < 1e-12:
            continue
        va = "bottom" if height >= 0 else "top"
        offset = 3 if height >= 0 else -5
        ax.annotate(
            f"{fmt.format(height)}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=size,
            color=color,
        )


def save_pdf_png(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=180)
