"""Vendored subset of ``paper_figure_style``.

The original module lives outside this repository, in the separate analysis
tree at ``sae_v3_analysis/src/paper_figure_style.py`` (reachable by setting
``LLM_ADDICTION_ANALYSIS``), and is not mirrored on the HuggingFace dataset, so
the body-figure generators could not be re-run anywhere else.  Only five names were ever imported --
``COLORS``, ``use_paper_style``, ``style_axes``, ``panel_title`` and
``save_pdf_png`` -- and they are reproduced here rather than imported, so
``scripts/figures/*.py`` runs with nothing but matplotlib and numpy.

The palette values were recovered from the submitted PDFs themselves
(``images/fig02_slot_machine.pdf``, ``images/fig04_causal_battery.pdf``): every
fill in those files is one of the Tableau-10 hues below, so the vendored colours
are the shipped colours, not a re-pick.

  fixed           #59A14F      variable          #E15759
  option2         #BAB0AC      option3           #7F7F7F
  accent          #EDC948      (drawn at alpha 0.35 over white -> #F8ECBF,
                                which is what the submitted PDF contains)

Model identity has since been split off from the betting condition.  Gemma and
LLaMA were drawn in the *same* two hues as Fixed and Variable, so green meant
"Fixed" on Figure 2 and "Gemma" on Figure 4 with nothing on the page to
disambiguate them; they are now Dark2 teal #1B9E77 and purple #7570B3, with
the readout control at #666666 and the balance control at Dark2 amber #E6AB02.
That pair is used by fig04, fig04b, fig_xctx_ladders and fig_axis_alignment.

Grid #EBEBEB at 0.7 pt, top and right spines removed, DejaVu Sans -- all read
back off the same two files.

One deliberate departure: :func:`save_pdf_png` writes the figure at its declared
canvas size instead of cropping with ``bbox_inches="tight"``.  The tight crop is
what broke the print scale.  A tight bbox *grows* the page whenever an artist
overhangs the canvas -- a legend anchored outside its axes, a text column past
the axis limit -- so a 5.5 in canvas shipped as a 5.98 in page, and LaTeX then
scaled it down by 0.88 to land inside \\textwidth.  Writing the canvas verbatim
makes the PDF's page width equal the printed width, so nominal point sizes are
printed point sizes.  Anything that would overhang has to be brought inside the
canvas by the generator, which is the point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------- palette

COLORS = {
    # Betting condition.  Green = Fixed, red = Variable, everywhere in the
    # paper: Figure 2, Figure 3, Figure 5 and every appendix bar panel.
    "fixed": "#59A14F",
    "variable": "#E15759",
    # Model identity.  These used to be the same two hues as the betting
    # condition, which made green mean "Fixed" in Figure 2 and "Gemma" in
    # Figure 4 with nothing on the page to tell a reader which.  Model identity
    # now has its own pair, from Dark2, used by every causal figure
    # (fig04, fig04b, fig_xctx_ladders, fig_axis_alignment):
    # Since the camera-ready restyle the causal figures use the palette of
    # Figures 2 and 3 instead: the risk-raising behaviour-built direction in the
    # red of the risk-raising arm (Gemma #E15759, LLaMA the darker GM red
    # #B33533), the balance control in the green of the restrained arm and the
    # readout in Figure 3's neutral grey.
    "gemma": "#E15759",
    "llama": "#B33533",
    # The two control directions in Figure 4, also Dark2: the readout is the
    # neutral grey it has always been, the balance control an amber that
    # separates from both model hues in greyscale as well as in colour.
    "readout": "#7F7F7F",
    "balance": "#59A14F",
    "option2": "#BAB0AC",
    "option3": "#7F7F7F",
    # The random-direction band.  It was #EDC948, a yellow that is now the
    # balance control's neighbour, so the band -- a background reference, not a
    # series -- is drawn in neutral grey instead.
    "null_band": "#BDBDBD",
    "accent": "#EDC948",
    # Present in the original module and kept for callers that reference them.
    "option1": "#4E79A7",
    "neutral": "#9A9A9A",
    "ink": "#333333",
}

GRID_COLOR = "#EBEBEB"
GRID_LW = 0.7


def use_paper_style(base: float = 9.0) -> None:
    """rcParams for a figure whose canvas width equals its printed width.

    ``base`` is the body size in points.  Because the canvas is written
    verbatim, these nominal sizes are the sizes that print.  The relative
    ladder -- title ``base+0.5`` bold, axis labels ``base``, ticks
    ``base-0.5``, legend ``base-1`` -- is the one the submitted PDFs use
    (9.5 bold / 9 / 8.5 / 8 at ``base=9``).
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base + 0.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 1.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": GRID_COLOR,
        "grid.linewidth": GRID_LW,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.constrained_layout.use": False,
        "pdf.fonttype": 3,
        "savefig.dpi": 400,
    })


def style_axes(ax, grid_axis: str = "y") -> None:
    """Drop the top and right spines; a hairline grid on one axis only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID_COLOR, linewidth=GRID_LW, zorder=0)
    ax.grid(False, axis="x" if grid_axis == "y" else "y")


def panel_title(ax, letter: str, title: str, *, pad: float | None = None,
                fontsize: float | None = None, loc: str = "center") -> None:
    """``(a) Title``, bold, above the axes."""
    kw = {}
    if pad is not None:
        kw["pad"] = pad
    if fontsize is not None:
        kw["fontsize"] = fontsize
    ax.set_title(f"{letter} {title}", fontweight="bold", loc=loc, **kw)


def fit_titles(fig, pad: float = 4.0) -> None:
    """Nudge any axes title that would run off the canvas back inside it.

    With the page written at the canvas size there is no tight bbox to grow and
    absorb an overhang, and a panel title is routinely wider than the panel it
    sits over.  Rather than shrink the type or re-word the title, the title is
    slid along its own axes until both ends are inside the figure; the shift is
    a couple of points and the title stays visually centred.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches() * fig.dpi
    for ax in fig.axes:
        t = ax.title
        if not t.get_text():
            continue
        bb = t.get_window_extent(renderer=r)
        shift = 0.0
        if bb.x1 > fw - pad:
            shift = -(bb.x1 - (fw - pad))
        elif bb.x0 < pad:
            shift = pad - bb.x0
        if shift:
            axw = ax.get_window_extent(renderer=r).width
            t.set_x(t.get_position()[0] + shift / axw)


def save_pdf_png(fig, out_dir, stem: str, dpi: int = 400) -> tuple[Path, Path]:
    """Write ``<stem>.pdf`` and ``<stem>.png`` at the figure's declared size.

    No ``bbox_inches='tight'``: the PDF page must equal the canvas so that the
    LaTeX include scales it by 1.0.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    return pdf, png
