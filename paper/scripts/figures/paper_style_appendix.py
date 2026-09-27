"""Vendored subset of the (unreleased) ``paper_figure_style`` helper module.

The appendix generators on the HF dataset import ``paper_figure_style``; that module
is not part of this repository and is not in the HF release either.  Rather than
importing it, the few constants and helpers the appendix figures actually use are
vendored here.  Everything below was read back out of the *submitted* PDFs
(``images/*.pdf``) so that the regenerated figures keep the submitted look:

  * series colours          -> fill/stroke operators in the PDF content streams
  * tick / spine colour     -> 0.2666666667 grey  = #444444
  * grid colour and width   -> 0.8666666667 grey  = #DDDDDD at 0.6 pt
  * spine width             -> 0.8 pt, top/right hidden
  * bold sans title, left aligned above the axes

What is *not* vendored: the original module's canvas sizing.  These figures were
drawn on canvases far wider than the 397.5 pt (5.5 in) NeurIPS ``\textwidth`` and
were therefore shrunk by LaTeX to as little as 0.41x, which put their labels below
5 pt on paper.  ``use_paper_style`` here sets point sizes that are correct at
scale 1.0, i.e. the canvas is authored at the width it is printed at.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- palette ----
COLORS = {
    "fixed": "#59A14F",           # 0.3490196078 0.631372549  0.3098039216
    "variable": "#E15759",        # 0.8823529412 0.3411764706 0.3490196078
    "variable_light": "#E8A0A0",  # 0.9098039216 0.6274509804 0.6274509804
    "neutral": "#C7C7C7",         # 0.7803921569 0.7803921569 0.7803921569
    "text": "#444444",
    "grid": "#DDDDDD",
}

# Stacked investment-choice options, in the submitted order (Option 1 -> 4).
CHOICE_COLORS = [COLORS["fixed"], COLORS["neutral"], COLORS["variable_light"], COLORS["variable"]]

# Diverging colormap used by the distortion heatmap.  This is *not* ``RdBu_r``:
# the submitted heatmap runs between a muted #B23A3A and #3B6A99 through white,
# with #E8A0A0 (the palette's ``variable_light``) sitting three quarters of the
# way up the red half.  ``RdBu_r`` is nearly three times darker at both ends
# (max channel error 78/255 against the submitted ramp; the nine anchors below
# reproduce it to 2/255).  The anchors were sampled off the colourbar raster
# embedded in the submitted ``images/distortion_multimodel_summary.pdf``.
HEATMAP_COLORS = ["#3B6A99", "#6F95B9", "#A6C2DB", "#D2E0ED", "#FFFEFE",
                  "#F4D0D0", "#E8A0A0", "#CD6D6D", "#B23A3A"]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("paper_diverging", HEATMAP_COLORS)

# Nothing may print below this many points.
MIN_PT = 7.0


def use_paper_style(base: float = 8.0) -> None:
    """Point sizes are *printed* point sizes: author the canvas at print width."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": base,
        "axes.labelsize": base + 1.0,
        "axes.titlesize": base + 1.5,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "axes.edgecolor": COLORS["text"],
        "axes.linewidth": 0.8,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#CCCCCC",
        "legend.borderpad": 0.4,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,
    })


def style_axes(ax, grid_axis: str = "y") -> None:
    """Hide top/right spines, put a light grid behind the data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["text"])
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, tag: str, text: str, size: float = 9.5, pad: float = 6.0) -> None:
    """Left-aligned bold title, optionally prefixed with a ``(a)``-style tag."""
    label = f"({tag}) {text}" if tag else text
    ax.set_title(label, fontsize=size, fontweight="bold", loc="left",
                 color="#000000", pad=pad)


def save_pdf_png(fig, out_dir, stem: str, png: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    if png:
        fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)
