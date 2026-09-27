#!/usr/bin/env python
r"""Figure 1 (§1 Introduction): the two-phase overview schematic.

Outputs (images/):
  representative_flow_diagram.pdf

PRINT-SIZE CONTRACT
-------------------
\textwidth is 5.5in.  A PDF point is 1/72in, so that is exactly 396.0pt on the
page -- NOT the 397.5 a TeX log reports (TeX's pt is 1/72.27in, so 5.5in is
397.485 TeX pt = 396.0 PDF pt).  The float includes this figure at
0.9\textwidth, so the printed width is 0.9 x 396.0 = 356.4pt = 4.95in.  The
canvas is drawn at exactly that width, so \includegraphics scales it by 1.000
and every font size below is the size that reaches the page.

The whole schematic is drawn in ONE axes whose data units ARE points: xlim is
(0, W_PT) and ylim is (0, H_PT).  So every coordinate in this file is a printed
point, and a `size=7` label really is 7pt on paper.  Nothing is drawn below 7pt
-- including the "777" / "1234" glyphs inside the pictographs, which is why
those icons are as large as they are.  savefig() is called WITHOUT
bbox_inches="tight" so the PDF MediaBox equals figsize exactly.

WHY THIS FILE EXISTS
--------------------
The committed images/representative_flow_diagram.pdf is a hand-made vector
drawing (Type3 fonts, no producer metadata, 3727pt wide) with NO generator in
the repo or on HF.  build_overview_figure.py -- the script the figure README
points at, byte-identical in the repo and in three places on HF -- draws a
completely different picture (two flat panels plus a thesis band, no icons, no
cuboid), so it cannot be the source, and re-running it would replace the
submitted composition rather than fix its size.

At 3727pt wide included at 0.9\textwidth the drawing was scaled to 0.096, so
its 30-60pt type printed at 2.9-5.8pt.  No pure scale change can fix that: the
content is far too dense for a 2.93:1 strip at 356.4pt.  This file re-draws
the SAME composition -- same blocks, same order, same labels, same colour
semantics -- reflowed from one very wide row into two full-width rows (Phase 1
above, Phase 2 below) so the type can be 7-9pt.  The former vertical rule
between the phases becomes the horizontal dashed rule between the rows.

HEIGHT BUDGET (the layout is compacted, the type is not)
-------------------------------------------------------
The canvas is 215pt tall, down from 288, with no glyph below 7pt.  What paid
for the 73pt:

  * Phase 1's title and its italic gloss share one line: the pair measures
    341.8pt on a 356.4pt canvas.  Phase 2's pair would need 388.5pt, so that
    header stays two lines.  Both glosses are 7.0pt rather than 7.5 -- the
    print floor, which is what makes the Phase 1 pair fit.
  * Inside each outcome panel, "risk metrics" and the verb share one line, and
    the panel is 56pt instead of 80 (a 17pt bar block instead of 25pt).
  * The autonomy contrast is a row of [dot][two-line caption] pairs instead of
    a dot stacked above its caption.
  * The audited-task strip labels each task on one line instead of two.
  * The cuboid is 80pt instead of 104 and its two halves are asymmetric: 35pt
    above the shared axis, 45 below.  They cannot be equal.  In each half the
    context caption sits at the left and the vector fan sweeps past it to the
    right; the low-autonomy fan is the divergent one, so its steepest vector
    needs more room to clear the caption than the aligned fan does.
  * The SAE readout scatters are 19pt tall instead of 30.

Every number was checked against a 400dpi render, not assumed.  The tight
spots, measured from the rendered text extents:

    top vector clearing "High Alignment"      2.65pt
    steepest low-autonomy vector clearing
      "Low Autonomy"                          2.89pt
    strip caption inside its grey panel       0.0pt (82.1pt of text in 82)
    V_IC against V_MW's subscript             0.7pt of side bearing, glyphs
                                              clear

The last two are the horizontal budget talking, not the vertical one: the
strip caption, the context caption, three V labels and the tip cloud have to
share the 210pt left of the SAE column, and none of them can shrink below 7pt.

The pictographs (slot machine, ATM/keypad, mystery wheel, magnifier) are
redrawn as simplified vector glyphs; the original artwork is not recoverable.
No data file is required -- the scatter clouds are illustrative, as in the
original, and are drawn from fixed seeds so the figure is deterministic.

Vendored from `paper_figure_style` (absent from this repo and from HF): only
the two palette entries this figure uses,
    COLORS["fixed"]    = "#59A14F"   (green -> low autonomy / fixed bet)
    COLORS["variable"] = "#E15759"   (red   -> high autonomy / variable bet)
plus the DejaVu Sans / pdf.fonttype=42 rcParams that use_paper_style() sets.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "images")

# --- print geometry (data units == printed points) ------------------------
TEXTWIDTH_PT = 396.0                 # 5.5in NeurIPS text block, in PDF points
W_PT = 0.9 * TEXTWIDTH_PT            # 356.4pt = 4.95in
H_PT = 215.0

# --- vendored palette (see module docstring) ------------------------------
C_LOW = "#59A14F"      # COLORS["fixed"]     green = low autonomy
C_HIGH = "#E15759"     # COLORS["variable"]  red   = high autonomy
C_LOW_BG = "#E9F3E3"
C_HIGH_BG = "#FBEAEA"
C_INK = "#1A1A1A"
C_MUTED = "#6B6B6B"
C_STRIP = "#EFEFEF"
C_WIRE = "#B8B8B8"
C_VSM, C_VIC, C_VMW = "#C0392B", "#2E6DB4", "#E8A33D"

# --- layout ---------------------------------------------------------------
# Phase 1
P1_HDR_Y = 206.5
P1_ICON_Y = 186.0
P1_NAME_Y = 169.0
P1_DOT_Y = 154.0
PANEL_X = (166.0, 260.0)
PANEL_Y, PANEL_W, PANEL_H = 142.0, 88.0, 56.0
DIVIDER_Y = 136.0
# Phase 2
P2_HDR_Y = 127.0
P2_SUB_Y = 117.5
STRIP_X, STRIP_Y, STRIP_W, STRIP_H = 3.0, 3.0, 82.0, 104.0
STRIP_CY = (86.0, 54.5, 23.0)
CUB_X0, CUB_X1 = 93.0, 210.0         # front face, left/right
CUB_Y0, CUB_YMID, CUB_Y1 = 4.0, 49.0 , 84.0
CUB_DX, CUB_DY = 8.0, 7.0
MECH_HDR_Y = (107.5, 98.5)
READ_HDR_Y = 107.5
MINI_X, MINI_W, MINI_H = 290.0, 62.0, 19.0
MINI_Y = (64.0, 14.0)                # high autonomy, low autonomy


# ------------------------------------------------------------------ helpers
def txt(ax, x, y, s, *, size=7.0, color=C_INK, weight="normal", ha="center",
        va="center", style="normal", rot=0, lsp=1.1, zorder=5, alpha=1.0):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, fontstyle=style, rotation=rot,
                   linespacing=lsp, zorder=zorder, alpha=alpha)


def span(ax, t):
    """True printed width of an already-placed Text, in points (= data units)."""
    fig = ax.figure
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    x0, _ = inv.transform((bb.x0, bb.y0))
    x1, _ = inv.transform((bb.x1, bb.y1))
    return x1 - x0


def rbox(ax, x, y, w, h, *, fc, ec=C_INK, lw=0.7, r=2.0, zorder=1):
    b = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                       boxstyle=f"round,pad={r}", facecolor=fc, edgecolor=ec,
                       linewidth=lw, zorder=zorder)
    ax.add_patch(b)
    return b


def rect(ax, x, y, w, h, *, fc, ec="none", lw=0.0, zorder=3):
    p = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw,
                      zorder=zorder)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, *, color=C_INK, lw=1.0, ms=5, zorder=4,
          alpha=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=ms, color=color, linewidth=lw,
                        shrinkA=0, shrinkB=0, zorder=zorder, alpha=alpha)
    ax.add_patch(a)
    return a


def underline(ax, t, *, pad=2.0, lw=0.7, color=C_INK):
    """Underline an already-placed Text using its true rendered extent."""
    fig = ax.figure
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((bb.x0, bb.y0))
    x1, _ = inv.transform((bb.x1, bb.y1))
    ax.plot([x0, x1], [y0 - pad, y0 - pad], color=color, lw=lw, zorder=6)


def vlab(ax, x, y, sub, color, *, alpha=1.0, base=8.0, subsz=7.0):
    """Draw V with a real subscript, keeping the subscript at >=7pt.

    mathtext renders `V$_{\\rm SM}$` with the subscript at 0.7x the base, which
    would be 4.9pt at a 7pt base -- below the print floor. So the two runs are
    set as separate Text objects instead.

    A white halo sits behind the pair: in the compacted layout the aligned fan
    spans only ~14pt at its tips, so the three labels sit shoulder to shoulder
    and a neighbouring arrow can still clip a label at the best perpendicular
    offset. The halo keeps the label legible where that happens instead of
    letting a line run through the glyphs.
    """
    wv = base * 0.60
    ws = len(sub) * subsz * 0.62
    x0 = x - wv - 1.2
    rect(ax, x0, y - base * 0.72, wv + ws + 2.6, base * 1.30, fc="white",
         zorder=4.5)
    txt(ax, x, y, "V", size=base, color=color, ha="right", va="center",
        alpha=alpha, zorder=5)
    txt(ax, x + 0.6, y - base * 0.30, sub, size=subsz, color=color,
        ha="left", va="center", alpha=alpha, zorder=5)


def perp_label(ax, ox, oy, tx, ty, f, side, sub, color, *, d=8.0, alpha=1.0):
    """Place a V label at fraction f along a vector, offset perpendicular to it.

    Offsetting along the normal (rather than straight up) is what keeps the
    label off its own arrow: these arrows are steep, so a purely vertical
    offset lets the line climb back into the label further along.
    """
    dxa, dya = tx - ox, ty - oy
    L = float(np.hypot(dxa, dya))
    nx, ny = -dya / L, dxa / L
    px, py = ox + f * dxa, oy + f * dya
    vlab(ax, px + side * d * nx, py + side * d * ny, sub, color, alpha=alpha)


# -------------------------------------------------------------------- icons
def icon_slot(ax, cx, cy, w=22.0, h=19.0):
    """Slot machine: body with a 777 reel window and a side lever."""
    x, y = cx - w / 2, cy - h / 2
    rect(ax, x, y, w, 3.0, fc=C_INK, zorder=3)                 # plinth
    rbox(ax, x, y + 2.5, w, h - 2.5, fc=C_INK, ec=C_INK, lw=0.5, r=1.6,
         zorder=3)
    rect(ax, cx - 0.36 * w, cy + 0.02 * h, 0.72 * w, 7.6, fc="white",
         zorder=4)                                             # reel window
    txt(ax, cx, cy + 0.02 * h + 3.8, "777", size=7.0, weight="bold",
        color=C_INK, zorder=5)
    rect(ax, cx - 0.26 * w, cy - 0.30 * h, 0.52 * w, 2.4, fc="white",
         zorder=4)                                             # payout slot
    ax.plot([cx + w / 2 + 1.4, cx + w / 2 + 1.4], [cy - 1.0, cy + 6.5],
            color=C_INK, lw=1.1, zorder=3, solid_capstyle="round")
    ax.add_patch(Circle((cx + w / 2 + 1.4, cy + 7.6), 1.9, facecolor=C_INK,
                        edgecolor="none", zorder=4))


def icon_atm(ax, cx, cy, w=27.0, h=19.0):
    """Investment choice: ATM-style unit with a 1234 keypad readout."""
    x, y = cx - w / 2, cy - h / 2
    rect(ax, cx - 0.30 * w, cy + h / 2 - 1.0, 0.60 * w, 2.6, fc=C_INK,
         zorder=3)                                             # top cap
    rbox(ax, x, y, w, h - 1.0, fc=C_INK, ec=C_INK, lw=0.5, r=1.6, zorder=3)
    rect(ax, cx - 0.38 * w, cy + 0.03 * h, 0.76 * w, 7.6, fc="white",
         zorder=4)
    txt(ax, cx, cy + 0.03 * h + 3.8, "1234", size=7.0, weight="bold",
        color=C_INK, zorder=5)
    rect(ax, cx - 0.24 * w, cy - 0.30 * h, 0.48 * w, 2.4, fc="white",
         zorder=4)                                             # card slot


def icon_wheel(ax, cx, cy, r=8.0):
    """Mystery wheel: spoked dial with a pointer and a stand."""
    ax.add_patch(Polygon([[cx - 4.2, cy - r - 3.2], [cx + 4.2, cy - r - 3.2],
                          [cx, cy - r + 1.0]], closed=True, facecolor=C_INK,
                         edgecolor="none", zorder=3))
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=C_INK,
                        lw=1.2, zorder=4))
    for k in range(8):
        a = np.pi / 8 + k * np.pi / 4
        ax.plot([cx, cx + r * 0.85 * np.cos(a)],
                [cy, cy + r * 0.85 * np.sin(a)],
                color=C_INK, lw=0.5, zorder=5)
    ax.add_patch(Circle((cx, cy), r * 0.22, facecolor=C_INK, edgecolor="none",
                        zorder=6))
    ax.add_patch(Polygon([[cx - 2.0, cy + r + 1.0], [cx + 2.0, cy + r + 1.0],
                          [cx, cy + r - 2.2]], closed=True, facecolor=C_INK,
                         edgecolor="none", zorder=7))


def icon_magnifier(ax, cx, cy, r=7.0):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=C_INK,
                        lw=1.6, zorder=4))
    a = np.deg2rad(232)
    ax.plot([cx + r * np.cos(a), cx + (r + 6.0) * np.cos(a)],
            [cy + r * np.sin(a), cy + (r + 6.0) * np.sin(a)],
            color=C_INK, lw=2.2, zorder=4, solid_capstyle="round")


# ------------------------------------------------------------------ headers
def header(ax, y, title, gloss, *, one_line):
    """Section header: bold title plus its italic gloss.

    Phase 1's pair measures 336pt and shares one line; Phase 2's measures
    371pt on a 356pt canvas and cannot, so it wraps. The gloss is 7.0pt (the
    print floor) rather than 7.5 so the Phase 1 pair clears the right margin.
    """
    t = txt(ax, 5, y, title, size=9.0, weight="bold", ha="left")
    if one_line:
        txt(ax, 5 + span(ax, t) + 7, y, gloss, size=7.0, style="italic",
            color=C_MUTED, ha="left")
    else:
        txt(ax, 5, y - 9.5, gloss, size=7.0, style="italic", color=C_MUTED,
            ha="left")


# ------------------------------------------------------------------ phase 1
def phase1(ax):
    header(ax, P1_HDR_Y, "Phase 1: Behavioral Experiment",
           "Autonomy amplifies gambling-like irrationality", one_line=True)

    # --- the two behavioural tasks
    icon_slot(ax, 40, P1_ICON_Y)
    icon_atm(ax, 116, P1_ICON_Y)
    txt(ax, 40, P1_NAME_Y, "Slot Machine", size=7.0)
    txt(ax, 116, P1_NAME_Y, "Investment Choice", size=7.0)

    # --- the autonomy contrast: dot beside its caption, not above it
    _contrast(ax, 12, C_LOW, 0.75, "Fixed bet", "External Goal")
    txt(ax, 84, P1_DOT_Y, "vs.", size=8.0, weight="bold")
    _contrast(ax, 98, C_HIGH, 0.85, "Variable bet", "Self-set Goal")

    # --- the two outcome panels
    _outcome(ax, PANEL_X[0], PANEL_Y, PANEL_W, PANEL_H, title="Low Autonomy",
             verb="stay low", color=C_LOW, bg=C_LOW_BG,
             vals=[0.26, 0.30, 0.24])
    _outcome(ax, PANEL_X[1], PANEL_Y, PANEL_W, PANEL_H, title="High Autonomy",
             verb="rise", color=C_HIGH, bg=C_HIGH_BG, vals=[0.50, 0.68, 0.92])


def _contrast(ax, cx, color, alpha, l1, l2):
    ax.add_patch(Circle((cx, P1_DOT_Y), 6.5, facecolor=color, edgecolor="none",
                        alpha=alpha, zorder=3))
    txt(ax, cx + 11, P1_DOT_Y + 4.0, l1, size=7.0, weight="bold", ha="left")
    txt(ax, cx + 11, P1_DOT_Y - 5.0, l2, size=7.0, weight="bold", ha="left")


def _outcome(ax, x, y, w, h, *, title, verb, color, bg, vals):
    rbox(ax, x, y, w, h, fc=bg, ec=color, lw=0.9, r=2.5, zorder=1)
    cx = x + w / 2
    txt(ax, cx, y + h - 9, title, size=7.5, weight="bold", color=color)

    # "risk metrics" and the verb share one line: two lines of gloss above a
    # 17pt bar block does not fit in 56pt.  Measure, then centre the pair.
    probe = txt(ax, 0, -50, "risk metrics", size=7.0, style="italic")
    w1 = span(ax, probe)
    probe.remove()
    probe = txt(ax, 0, -50, verb, size=7.5, style="italic", weight="bold")
    w2 = span(ax, probe)
    probe.remove()
    x_gloss = cx - (w1 + 4 + w2) / 2
    txt(ax, x_gloss, y + h - 19.5, "risk metrics", size=7.0, style="italic",
        color=C_MUTED, ha="left")
    t = txt(ax, x_gloss + w1 + 4, y + h - 19.5, verb, size=7.5, style="italic",
            weight="bold", ha="left")
    underline(ax, t, pad=1.6)

    base, top, bw = y + 12, y + h - 27, 13.0
    xs = [cx - 24, cx, cx + 24]
    for xi, v in zip(xs, vals):
        rect(ax, xi - bw / 2, base, bw, (top - base) * v, fc=color, zorder=3)
    ax.plot([x + 6, x + w - 6], [base, base], color=C_INK, lw=0.6, zorder=4)
    for xi, lab in zip(xs, ["I_BA", "I_LC", "I_EC"]):
        txt(ax, xi, base - 6.0, lab, size=7.0)


# ------------------------------------------------------------------ phase 2
def phase2(ax):
    header(ax, P2_HDR_Y, "Phase 2: Internal Representation Audit",
           "Partial Risk Sharing with Task-Specific Readouts", one_line=False)

    # --- the three audited tasks (one caption line each, not two)
    rbox(ax, STRIP_X, STRIP_Y, STRIP_W, STRIP_H, fc=C_STRIP, ec="none", lw=0,
         r=3.0, zorder=1)
    cx = STRIP_X + STRIP_W / 2
    for cy, icon, lab in [(STRIP_CY[0], icon_slot, "Slot Machine (SM)"),
                          (STRIP_CY[1], icon_atm, "Investment Choice (IC)"),
                          (STRIP_CY[2], icon_wheel, "Mystery Wheel (MW)")]:
        icon(ax, cx, cy + 6.5)
        txt(ax, cx, cy - 8.0, lab, size=7.0)

    _mechanism(ax)
    _readout(ax)


def _mechanism(ax):
    x0, x1, y0, y1 = CUB_X0, CUB_X1, CUB_Y0, CUB_Y1
    ymid = CUB_YMID
    dx, dy = CUB_DX, CUB_DY
    cx = (x0 + x1) / 2 - 1
    txt(ax, cx, MECH_HDR_Y[0], "Internal Mechanism:", size=7.5, weight="bold")
    txt(ax, cx, MECH_HDR_Y[1], "Goal-Framed Partial Sharing", size=7.5,
        weight="bold")

    # wireframe cuboid
    front = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    back = [(px + dx, py + dy) for px, py in front]
    for poly in (front, back):
        ax.add_patch(Polygon(poly, closed=True, facecolor="none",
                             edgecolor=C_WIRE, lw=0.6, zorder=2))
    for (px, py), (qx, qy) in zip(front, back):
        ax.plot([px, qx], [py, qy], color=C_WIRE, lw=0.6, zorder=2)
    ax.plot([x0, x1], [ymid, ymid], color=C_WIRE, lw=0.6, zorder=2)
    ax.plot([x1, x1 + dx], [ymid, ymid + dy], color=C_WIRE, lw=0.6, zorder=2)
    ax.plot([x0 + dx, x1 + dx], [ymid + dy, ymid + dy], color=C_WIRE, lw=0.6,
            zorder=2)

    rng = np.random.default_rng(11)
    ox, oy = x0 + 16, ymid + 3          # common origin of the task vectors
    tcx = x0 + 24                       # centre of both context captions

    # --- high-autonomy context: vectors align, tight cluster
    # The caption sits top-left and the fan sweeps past it to the right. The
    # top vector clears the bottom caption line by ~2pt: that clearance, not
    # the type, is what sets the 35pt height of this half.
    txt(ax, tcx, y1 - 1, "High Autonomy", size=7.0, weight="bold", color=C_HIGH)
    txt(ax, tcx, y1 - 9.5, "Context", size=7.0, weight="bold", color=C_HIGH)
    txt(ax, tcx, y1 - 18, "High Alignment", size=7.0)
    # Tip spread (+-7pt) and the per-label fractions below are chosen so the
    # three V labels sit side by side without overlapping their neighbours or
    # the tip cloud; a wider fan would need a taller half.
    # All three labels have to clear the TOP vector, because the fan is only
    # 12pt deep and an 11pt label cannot sit between two of its arrows. So the
    # perpendicular distances differ (10 / 14 / 16pt) and the fractions run
    # right-to-left, which is what keeps the labels in the same vertical order
    # as the vectors they name and keeps every arrowhead out of a white halo.
    tipx, tipy = x1 - 20, y1 - 21
    for c, lab, off, lx, ld in [(C_VSM, "SM", 6.0, 0.98, 10.0),
                                (C_VIC, "IC", 0.0, 0.78, 14.0),
                                (C_VMW, "MW", -6.0, 0.53, 16.0)]:
        ty = tipy + off
        arrow(ax, ox, oy, tipx, ty, color=c, lw=1.0, ms=5, zorder=4)
        perp_label(ax, ox, oy, tipx, ty, lx, +1, lab, c, d=ld)
    cl = rng.normal(0, 1, (34, 2)) * np.array([2.2, 2.4])
    ax.scatter(tipx + 15 + cl[:, 0], tipy + cl[:, 1], s=1.5, c="#8A8A8A",
               edgecolors="none", zorder=3)

    # --- the shared risk axis
    arrow(ax, x0 + 7, ymid, x1 - 5, ymid, color="#7A7A7A", lw=5.5, ms=12,
          zorder=5)
    txt(ax, (x0 + x1) / 2 - 2, ymid, "Shared Risk Axis", size=7.0,
        weight="bold", color="white", zorder=6)

    # --- low-autonomy context: vectors fan out, diffuse cloud
    txt(ax, tcx, y0 + 21, "Low Autonomy", size=7.0, weight="bold", color=C_LOW)
    txt(ax, tcx, y0 + 12.5, "Context", size=7.0, weight="bold", color=C_LOW)
    txt(ax, tcx, y0 + 4, "Low Alignment", size=7.0)
    # The steepest vector has to stay above the caption until it is clear of
    # it, which is why this half is 45pt while the aligned half is 35.
    # V_SM rides in the wedge between the SM and IC vectors, which is 10pt
    # clear at that x -- just under the 11pt label -- so its offset is 6.5, not
    # the 8 the other two use.
    tg = [(C_VSM, "SM", (x1 - 4, ymid - 14), 0.85, 6.5),
          (C_VIC, "IC", (x1 - 9, y0 + 16), 0.55, 8.0),
          (C_VMW, "MW", (x1 - 15, y0 + 2), 0.70, 8.0)]
    for c, lab, tgt, lx, ld in tg:
        arrow(ax, ox, oy, tgt[0], tgt[1], color=c, lw=0.9, ms=5, zorder=3,
              alpha=0.40)
        perp_label(ax, ox, oy, tgt[0], tgt[1], lx, -1, lab, c, alpha=0.75,
                   d=ld)
    # The diffuse cloud sits in the wedge between the IC and MW vectors, so it
    # is sheared along their bisector instead of being drawn axis-aligned:
    # an upright cloud this close to the origin spills across both arrows.
    mid = ((tg[1][2][0] + tg[2][2][0]) / 2, (tg[1][2][1] + tg[2][2][1]) / 2)
    slope = (mid[1] - oy) / (mid[0] - ox)
    ccx = ox + 0.80 * (mid[0] - ox)
    df = rng.normal(0, 1, (58, 2)) * np.array([7.0, 3.0])
    ax.scatter(ccx + df[:, 0], oy + slope * (ccx - ox + df[:, 0]) + df[:, 1],
               s=1.5, c="#A8A8A8", edgecolors="none", zorder=3)


def _readout(ax):
    arrow(ax, 222, CUB_YMID, 237, CUB_YMID, color=C_INK, lw=1.3, ms=6)
    txt(ax, 255, CUB_YMID + 19, "SAE", size=7.5)
    txt(ax, 255, CUB_YMID + 10, "Decoder", size=7.5)
    icon_magnifier(ax, 255, CUB_YMID - 4)

    txt(ax, MINI_X - 9, READ_HDR_Y, "SAE Decoder & Context Output",
        size=7.5, weight="bold")

    _mini(ax, MINI_X, MINI_Y[0], MINI_W, MINI_H, title="High Autonomy",
          sub="Sharp Signal", color=C_HIGH, tight=True, seed=3)
    _mini(ax, MINI_X, MINI_Y[1], MINI_W, MINI_H, title="Low Autonomy",
          sub="Noisy Signal", color=C_LOW, tight=False, seed=4)


def _mini(ax, x, y, w, h, *, title, sub, color, tight, seed):
    txt(ax, x + w / 2, y + h + 13, title, size=7.0, weight="bold", color=color)
    txt(ax, x + w / 2, y + h + 4, sub, size=7.0)
    ax.plot([x, x], [y, y + h], color=C_INK, lw=0.8, zorder=3)
    ax.plot([x, x + w], [y, y], color=C_INK, lw=0.8, zorder=3)
    rng = np.random.default_rng(seed)
    n = 68
    u = rng.uniform(0.08, 0.94, n)
    if tight:
        v = u + rng.normal(0, 0.07, n)
        ax.plot([x + 0.06 * w, x + 0.96 * w], [y + 0.06 * h, y + 0.96 * h],
                color="#555555", lw=0.5, zorder=4)
        col = "#3A3A3A"
    else:
        v = rng.uniform(0.06, 0.94, n)
        col = "#9A9A9A"
    v = np.clip(v, 0.04, 0.96)
    ax.scatter(x + u * w, y + v * h, s=1.4, c=col, edgecolors="none", zorder=4)
    txt(ax, x - 9.5, y + h / 2, "Actual\nBehavior", size=7.0, rot=90, lsp=1.0)
    txt(ax, x + w / 2, y - 7.0, "Internal Readout", size=7.0)


# -------------------------------------------------------------------- build
def build():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(W_PT / 72.0, H_PT / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(0, H_PT)
    ax.axis("off")

    phase1(ax)
    # divider between the phases (the vertical rule of the original 1-row art)
    ax.plot([6, W_PT - 6], [DIVIDER_Y, DIVIDER_Y], color="#C4C4C4", lw=0.8,
            ls=(0, (4, 3)), zorder=1)
    phase2(ax)

    out = os.path.join(OUT, "representative_flow_diagram.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}  ({W_PT:.2f} x {H_PT:.2f} pt)")
    return out


if __name__ == "__main__":
    build()
