"""Figure 5: the matched-cap control, drawn from data instead of read off a PDF.

Why this file exists
--------------------
The submitted paper carries the matched-cap result as panel (d) of the
investment-choice figure, where it does not belong: panels (a)--(c) are the
six-model investment-choice experiment and (d) is a separate GPT-4o-mini
slot-machine ablation whose numbers were hand-read off an older PDF.  This
script promotes that control to a figure of its own and recomputes both halves
of it from the released corpora.

Panel (a) is the GPT-4o-mini cap ablation, taken from
``scripts/build_figure_data.build_cap_ablation`` so that the loader is not
duplicated.  Two facts that the submitted panel hides are drawn explicitly:
there is no fixed arm at a $10 cap, because the fixed run was collected at bet
sizes 30, 50 and 70 only, and the two arms do not share a round ceiling (100
rounds fixed against 50 variable), so the rates are not exchangeable across
arms.  Both ceilings are read out of the corpus, not typed in.

Panel (b) is the matched-cap replication on four further API models, 64 cells
(4 models x 4 caps x {fixed, variable} x {BASE, GMPRW}, 50 games each) from the
local ``mc32`` corpus.  Every pair gets a percentile bootstrap interval over
games, and the three pairs that run the other way are drawn and labelled rather
than dropped.

Outputs
-------
``images/fig05_matched_cap.pdf`` / ``.png`` and the sidecar
``paper_data/fig05_matched_cap.json``.

Run with::

    from the repository root, \
        HF_HUB_DISABLE_XET=1 python3 scripts/figures/fig05_matched_cap.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
# ``paper_figure_style`` lives in the separate analysis tree -- it is neither in
# this repository nor on the HuggingFace dataset.  Point ``LLM_ADDICTION_ANALYSIS``
# at that checkout to re-run this generator; ``paper_style_vendored.py`` under
# scripts/figures carries the same five names but writes untight PDF pages, so it
# is not substituted here silently.
ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS", Path.home() / "llm-addiction"))
sys.path.insert(0, str(ANALYSIS_ROOT / "experiments" / "07_sae_readout" / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from build_figure_data import REPO_ID, build_cap_ablation, wilson  # noqa: E402
from paper_figure_style import (  # noqa: E402
    COLORS,
    panel_title,
    save_pdf_png,
    style_axes,
    use_paper_style,
)

# The matched-cap corpus.  ``HF_MC32`` is where it lives in the release and is
# what the sidecar records; ``LOCAL_MC32`` is an optional on-disk mirror under
# ``$LLM_ADDICTION_DATA``.  ``mc32_dir()`` prefers the mirror and otherwise
# pulls the same 64 files from the public dataset.
HF_MC32 = "rebuttal_neurips_2026/matched_cap_mc32"
DATA_ROOT = Path(os.environ.get("LLM_ADDICTION_DATA", Path.home() / "llm-addiction-data"))
LOCAL_MC32 = DATA_ROOT / "mc32"
MC32_CACHE = Path("/tmp/hfcache_fig05")
IMAGES = REPO / "images"
SIDECAR = REPO / "paper_data" / "fig05_matched_cap.json"
STEM = "fig05_matched_cap"

N_BOOT = 2000
SEED = 24231
RNG = np.random.default_rng(SEED)

CAPS = (10, 30, 50, 70)
COMBOS = ("BASE", "GMPRW")
# Display names only; every quantity below is read from the corpus.
MODEL_ORDER = ("gpt-4o-mini", "gpt-4.1-mini", "gemini-flash", "claude-haiku-4-5-20251001")
MODEL_LABEL = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gemini-flash": "Gemini-2.5-Flash",
    "claude-haiku-4-5-20251001": "Claude-Haiku-4.5",
}
COMBO_LABEL = {"BASE": "BASE prompt", "GMPRW": "GMPRW (five modules)"}
COMBO_COLOR = {"BASE": COLORS["option3"], "GMPRW": COLORS["option4"]}


# ------------------------------------------------------------------ statistics


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]], by summing tables no more
    likely than the observed one.  Written out because scipy is not installed in
    the interpreter this repository's figure code runs under."""
    n = a + b + c + d
    row1, row2, col1 = a + b, c + d, a + c

    def prob(x: int) -> float:
        return (math.comb(row1, x) * math.comb(row2, col1 - x)) / math.comb(n, col1)

    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    p_obs = prob(a)
    return float(min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))))


def boot_diff_ci(k_f: int, n_f: int, k_v: int, n_v: int, n_boot: int = N_BOOT):
    """Percentile interval for (variable - fixed) bankruptcy, in points.

    The resampling unit is the game: each arm is resampled independently with
    replacement at its own n.  When neither arm ever ruins the interval is
    degenerate at zero -- that is a property of the estimator at this sample
    size, not a measurement, so ``newcombe_diff_ci`` is reported alongside it.
    """
    if n_f == 0 or n_v == 0:
        return (float("nan"), float("nan"), float("nan"))
    f = np.zeros(n_f)
    f[:k_f] = 1.0
    v = np.zeros(n_v)
    v[:k_v] = 1.0
    fb = f[RNG.integers(0, n_f, size=(n_boot, n_f))].mean(axis=1)
    vb = v[RNG.integers(0, n_v, size=(n_boot, n_v))].mean(axis=1)
    diffs = 100.0 * (vb - fb)
    point = 100.0 * (k_v / n_v - k_f / n_f)
    return (point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def newcombe_diff_ci(k_f: int, n_f: int, k_v: int, n_v: int):
    """Newcombe hybrid-score interval for the difference of two proportions.

    Kept as a second interval because the bootstrap collapses to a point
    whenever both arms are 0/50, and a collapsed interval must not be read as
    certainty about a zero difference.
    """
    lo_f, hi_f = [x / 100.0 for x in wilson(k_f, n_f)]
    lo_v, hi_v = [x / 100.0 for x in wilson(k_v, n_v)]
    p_f, p_v = k_f / n_f, k_v / n_v
    lower = (p_v - p_f) - math.sqrt((p_v - lo_v) ** 2 + (hi_f - p_f) ** 2)
    upper = (p_v - p_f) + math.sqrt((hi_v - p_v) ** 2 + (p_f - lo_f) ** 2)
    return (100.0 * lower, 100.0 * upper)


# ------------------------------------------------------------------ mc32 load


def mc32_dir() -> Path:
    """The matched-cap corpus, from the local mirror or the public dataset."""
    if LOCAL_MC32.is_dir() and any(LOCAL_MC32.glob("final_*.json")):
        return LOCAL_MC32
    from huggingface_hub import snapshot_download

    root = snapshot_download(REPO_ID, repo_type="dataset", cache_dir=str(MC32_CACHE),
                             allow_patterns=[f"{HF_MC32}/final_*.json"])
    return Path(root) / HF_MC32


def load_mc32() -> dict:
    """One cell per (model, combo, cap, mode), read from ``final_*.json``.

    Only the top level of ``mc32`` is read.  ``QUARANTINE/``,
    ``LEGACY_PARSER_claude/`` and ``TRUNCATED_claude_maxtok300/`` are superseded
    collections that sit in subdirectories and are deliberately not globbed.
    """
    cells, files, ceilings = {}, {}, {}
    for path in sorted(mc32_dir().glob("final_*.json")):
        blob = json.load(open(path))
        games = blob["results"]
        key = (blob["model"], blob["prompt_combo"], int(blob["cap"]), blob["mode"])
        if key in cells:
            raise RuntimeError(f"duplicate cell {key}: {path}")
        cells[key] = {
            "k": sum(1 for g in games if g.get("bankrupt")),
            "n": len(games),
            "n_wagering": sum(1 for g in games if (g.get("total_rounds") or 0) > 0),
            "rounds_per_game": float(np.mean([g.get("total_rounds") or 0 for g in games])),
        }
        files[key] = path.name
        gen = blob["config_snapshot"]["generation"]
        ceilings[blob["mode"]] = gen["max_rounds_fixed"] if blob["mode"] == "fixed" else gen["max_rounds_variable"]
    if len(cells) != 64:
        raise RuntimeError(f"expected 64 mc32 cells, found {len(cells)}")
    return {"cells": cells, "files": files, "ceilings": ceilings}


def mc32_pairs(corpus: dict) -> list[dict]:
    rows = []
    for model in MODEL_ORDER:
        for combo in COMBOS:
            for cap in CAPS:
                f = corpus["cells"][(model, combo, cap, "fixed")]
                v = corpus["cells"][(model, combo, cap, "variable")]
                point, lo, hi = boot_diff_ci(f["k"], f["n"], v["k"], v["n"])
                n_lo, n_hi = newcombe_diff_ci(f["k"], f["n"], v["k"], v["n"])
                rows.append({
                    "model": model,
                    "model_label": MODEL_LABEL[model],
                    "prompt_combo": combo,
                    "cap": cap,
                    "fixed": {"k": f["k"], "n": f["n"], "pct": 100 * f["k"] / f["n"],
                              "ci": list(wilson(f["k"], f["n"])),
                              "n_wagering": f["n_wagering"], "mean_rounds": f["rounds_per_game"]},
                    "variable": {"k": v["k"], "n": v["n"], "pct": 100 * v["k"] / v["n"],
                                 "ci": list(wilson(v["k"], v["n"])),
                                 "n_wagering": v["n_wagering"], "mean_rounds": v["rounds_per_game"]},
                    "delta_pp": point,
                    "delta_boot_ci": [lo, hi],
                    "delta_boot_degenerate": bool(lo == hi),
                    "delta_newcombe_ci": [n_lo, n_hi],
                    "fisher_p": fisher_exact_two_sided(v["k"], v["n"] - v["k"], f["k"], f["n"] - f["k"]),
                    "files": {"fixed": corpus["files"][(model, combo, cap, "fixed")],
                              "variable": corpus["files"][(model, combo, cap, "variable")]},
                })
    return rows


# ------------------------------------------------------------------ panel (a)


def draw_panel_a(ax, cap_data: dict) -> None:
    rows = cap_data["rows"]
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.55)

    for arm, off, colour in (("fixed", -width / 2, COLORS["fixed"]), ("variable", +width / 2, COLORS["variable"])):
        for i, row in enumerate(rows):
            cell = row[arm]
            xi = x[i] + off
            if cell is None:
                # No fixed arm at a $10 cap: the fixed run covers bet sizes
                # 30/50/70 only.  Draw the hole, do not leave the slot blank.
                ax.add_patch(Rectangle((xi - width / 2, 0), width, 0.9, facecolor="none",
                                       edgecolor=COLORS["neutral"], hatch="///", linewidth=0.9,
                                       linestyle=(0, (3, 2)), zorder=2))
                ax.text(xi - 0.08, 1.05, "no fixed arm\n(bet sizes 30/50/70)", ha="center",
                        va="bottom", fontsize=7.8, color="#555555", rotation=90,
                        linespacing=1.05)
                continue
            ax.bar(xi, cell["pct"], width, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
            lo, hi = cell["ci"]
            # Labels are single-line and vertical: rotated 90 deg each stays in
            # its own bar's column, where the old two-line boxed labels were
            # wider than the bar pair itself.
            if cell["k"] == 0:
                # One-sided bracket to the Wilson upper bound.
                ax.plot([xi, xi], [0, hi], color="#333333", linewidth=1.1, zorder=5)
                ax.plot([xi - width * 0.28, xi + width * 0.28], [hi, hi], color="#333333",
                        linewidth=1.1, zorder=5)
                ax.text(xi, hi * 1.25, f"0/{cell['n']} $\\leq$ {hi:.2f}%", ha="center",
                        va="bottom", fontsize=7.8, color="#333333", rotation=90, zorder=6)
            else:
                ax.plot([xi, xi], [lo, hi], color="#333333", linewidth=1.1, zorder=5)
                for y in (lo, hi):
                    ax.plot([xi - width * 0.22, xi + width * 0.22], [y, y], color="#333333",
                            linewidth=1.1, zorder=5)
                ax.text(xi, hi * 1.15, f"{cell['pct']:.1f}% ({cell['k']}/{cell['n']})",
                        ha="center", va="bottom", fontsize=7.8, color="#333333",
                        rotation=90, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"\\${r['cap']}" for r in rows])
    ax.set_xlabel("Matched maximum bet (cap)")
    ax.set_ylabel("Bankruptcy rate (%)")
    # Extra log-scale headroom keeps the vertical labels inside the frame.
    ax.set_ylim(0, 700)
    ax.set_yticks([0, 0.25, 0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0", "0.25", "0.5", "1", "2", "5", "10", "20"])
    ax.set_xlim(-0.65, len(rows) - 0.35)
    style_axes(ax)

    rc_f, rc_v = cap_data["round_cap_fixed"], cap_data["round_cap_variable"]
    handles = [
        Rectangle((0, 0), 1, 1, color=COLORS["fixed"],
                  label=f"Fixed (= cap), $\\leq${rc_f} rounds"),
        Rectangle((0, 0), 1, 1, color=COLORS["variable"],
                  label=f"Variable ($\\leq$ cap), $\\leq${rc_v} rounds"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=COLORS["neutral"], hatch="///",
                  label="Arm not collected"),
    ]
    # The legend sits below the axis: at print size the panel interior has no
    # free corner once the vertical value labels are legible.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              fontsize=7.8, ncol=1, borderpad=0.35, labelspacing=0.35,
              handlelength=1.3, handletextpad=0.5, frameon=False)
    ax.text(0.5, -0.52,
            "Vertical axis linear above 1%, logarithmic\n"
            "below it, so sub-percent intervals stay visible.\n"
            "Arms are matched on the cap, not on game length\n"
            f"({rc_f}-round ceiling fixed vs {rc_v} variable).",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="#555555",
            linespacing=1.3)
    model_disp = cap_data["model"].replace("gpt", "GPT")
    panel_title(ax, "(a)", f"Cap Ablation\n({model_disp}, slot machine)")


# ------------------------------------------------------------------ panel (b)


def draw_panel_b(ax, pairs: list[dict]) -> None:
    by_key = {(p["model"], p["prompt_combo"], p["cap"]): p for p in pairs}
    rows = [(m, c) for m in MODEL_ORDER for c in CAPS]
    y_of = {rc: len(rows) - 1 - i for i, rc in enumerate(rows)}
    offset = {"BASE": +0.19, "GMPRW": -0.19}

    ax.axvline(0.0, color="#444444", linewidth=1.0, zorder=1)
    for i, (model, cap) in enumerate(rows):
        if i % 2 == 0:
            ax.axhspan(y_of[(model, cap)] - 0.5, y_of[(model, cap)] + 0.5,
                       color=COLORS["grid"], alpha=0.28, zorder=0, linewidth=0)

    for combo in COMBOS:
        colour = COMBO_COLOR[combo]
        for model, cap in rows:
            p = by_key[(model, combo, cap)]
            y = y_of[(model, cap)] + offset[combo]
            lo, hi = p["delta_boot_ci"]
            sig = p["fisher_p"] < 0.05
            if p["delta_boot_degenerate"]:
                nlo, nhi = p["delta_newcombe_ci"]
                ax.plot([nlo, nhi], [y, y], color=colour, linewidth=0.8, alpha=0.45, zorder=2)
            else:
                ax.plot([lo, hi], [y, y], color=colour, linewidth=1.8, alpha=0.9, zorder=3)
                for b in (lo, hi):
                    ax.plot([b, b], [y - 0.10, y + 0.10], color=colour, linewidth=1.2, zorder=3)
            ax.plot([p["delta_pp"]], [y], marker="o", markersize=5.2,
                    markerfacecolor=colour if sig else "white",
                    markeredgecolor=colour, markeredgewidth=1.2, zorder=4)

    # The three pairs that run the other way are named, not dropped.
    reversed_pairs = [p for p in pairs if p["delta_pp"] < 0]
    for p in reversed_pairs:
        y = y_of[(p["model"], p["cap"])] + offset[p["prompt_combo"]]
        anchor = max(p["delta_pp"], p["delta_boot_ci"][1])
        ax.annotate(f"{p['delta_pp']:+.0f} pp, Fisher $p$={p['fisher_p']:.3f}",
                    xy=(max(anchor, 2.0), y), xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7.8, color="#8B1A1A", zorder=7,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.4))

    n_rev = len(reversed_pairs)
    ax.text(0.985, 0.99, f"{n_rev} of {len(pairs)} pairs run the\nother way (labelled)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color="#8B1A1A",
            linespacing=1.25)

    ax.set_yticks([y_of[r] for r in rows])
    ax.set_yticklabels([f"\\${cap}" for _, cap in rows])
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("Bankruptcy change, variable $-$ fixed (points)")
    ax.set_xlim(-30, 76)
    style_axes(ax, grid_axis="x")

    # Model names are longer than their four-row blocks at print size, so
    # alternate them over two columns; the bracket line stays put.
    for k, model in enumerate(MODEL_ORDER):
        ys = [y_of[(model, cap)] for cap in CAPS]
        ax.text(-0.165 - 0.075 * (k % 2), (min(ys) + max(ys)) / 2, MODEL_LABEL[model],
                transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=7.8, fontweight="bold")
        ax.plot([-0.105, -0.105], [min(ys) - 0.42, max(ys) + 0.42], transform=ax.get_yaxis_transform(),
                color="#999999", linewidth=1.0, clip_on=False)

    handles = [
        Line2D([], [], color=COMBO_COLOR["BASE"], marker="o", markersize=6.0, linewidth=1.8,
               label=COMBO_LABEL["BASE"]),
        Line2D([], [], color=COMBO_COLOR["GMPRW"], marker="o", markersize=6.0, linewidth=1.8,
               label=COMBO_LABEL["GMPRW"]),
        Line2D([], [], color="#444444", marker="o", markersize=6.0, linestyle="none",
               label="Fisher $p<0.05$ (filled)"),
        Line2D([], [], color="#444444", marker="o", markersize=6.0, linestyle="none",
               markerfacecolor="white", label="not significant (open)"),
        Line2D([], [], color="#777777", linewidth=0.8, alpha=0.6,
               label="score interval\n(bootstrap degenerate)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.44, -0.17),
              fontsize=7.8, ncol=2, columnspacing=1.0, handlelength=1.4,
              handletextpad=0.5, labelspacing=0.4, frameon=False)
    panel_title(ax, "(b)", "Matched-Cap Replication\n(4 models, 32 pairs)")


# ------------------------------------------------------------------ main


def main() -> None:
    cap_data = build_cap_ablation()
    corpus = load_mc32()
    pairs = mc32_pairs(corpus)

    # Native width ~= the 397 pt NeurIPS text block, so the \textwidth include
    # scales by ~1 and nominal point sizes are printed sizes.  Floor: 7.5 pt.
    # Manual layout: constrained layout crushes the axes once legends and
    # tick labels are set at print-legible sizes.
    use_paper_style(9.0)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(5.5, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.34,
                          left=0.085, right=0.99, top=0.86, bottom=0.30)
    draw_panel_a(fig.add_subplot(gs[0, 0]), cap_data)
    draw_panel_b(fig.add_subplot(gs[0, 1]), pairs)
    save_pdf_png(fig, IMAGES, STEM)
    plt.close(fig)

    reversed_pairs = [p for p in pairs if p["delta_pp"] < 0]
    n_games = sum(p["fixed"]["n"] + p["variable"]["n"] for p in pairs)
    caption = (
        "Matched-cap control. "
        "(a) GPT-4o-mini slot machine at four maximum bets, three of which carry both arms. The "
        "fixed arm was collected at bet sizes 30, 50 and 70 only, so there is no fixed cell at the "
        "$10 cap and the comparison is matched at $30, $50 and $70; the missing slot is drawn "
        "hatched. The two arms also differ in their round ceiling, "
        f"{cap_data['round_cap_fixed']} rounds for fixed against {cap_data['round_cap_variable']} for "
        "variable, so the arms are matched on the bet cap and not on game length. Bars carry 95% "
        "Wilson intervals; where a rate is 0/N the bracket runs one-sided to the Wilson upper bound "
        "and the count is printed. The vertical axis is linear above 1% and logarithmic below it so "
        "that sub-percent intervals remain visible. "
        f"(b) The same contrast on four further API models under the paper's own condition set: "
        f"{len(pairs)} pairs (4 models x 4 caps x 2 prompt combinations), 50 games per cell, "
        f"{n_games} games in total. Points are variable minus fixed bankruptcy in percentage points "
        "with a 95% percentile bootstrap interval over games; filled points are significant at "
        "Fisher exact p < 0.05. Where both arms ruin no games the bootstrap interval collapses to a "
        "point, and the thin line is the Newcombe score interval, which shows what the design can "
        f"resolve at n = 50. {len(reversed_pairs)} of {len(pairs)} pairs run the other way and are "
        "labelled: " + "; ".join(
            f"{p['model_label']} {p['prompt_combo']} ${p['cap']} ({p['delta_pp']:+.1f} pp, "
            f"p={p['fisher_p']:.3f})" for p in reversed_pairs) + "."
    )

    payload = {
        "generated_by": "scripts/figures/fig05_matched_cap.py",
        "figure": f"images/{STEM}.pdf",
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": SEED,
        "panel_a": cap_data,
        "panel_b": {
            "source_dir": HF_MC32,
            "n_cells": 64,
            "n_pairs": len(pairs),
            "games_per_cell": 50,
            "round_ceilings": corpus["ceilings"],
            "pairs": pairs,
            "reversed_pairs": [
                {"model": p["model_label"], "prompt_combo": p["prompt_combo"], "cap": p["cap"],
                 "delta_pp": p["delta_pp"], "fisher_p": p["fisher_p"],
                 "fixed": f"{p['fixed']['k']}/{p['fixed']['n']}",
                 "variable": f"{p['variable']['k']}/{p['variable']['n']}"}
                for p in reversed_pairs],
            "note": ("Subdirectories QUARANTINE/, LEGACY_PARSER_claude/ and TRUNCATED_claude_maxtok300/ "
                     "are superseded collections and are not read. mc32 shares the cap ablation's "
                     "round-ceiling asymmetry (100 fixed vs 50 variable) and runs with the persona "
                     "prefix on, held constant across the two arms of every pair."),
        },
        "caption": caption,
    }
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(payload, indent=1))

    print(f"wrote {IMAGES / (STEM + '.pdf')}")
    print(f"wrote {IMAGES / (STEM + '.png')}")
    print(f"wrote {SIDECAR}")

    print("\npanel (a): GPT-4o-mini cap ablation "
          f"(round ceiling {cap_data['round_cap_fixed']} fixed / {cap_data['round_cap_variable']} variable)")
    for r in cap_data["rows"]:
        f, v = r["fixed"], r["variable"]
        fs = "  no fixed arm collected      " if f is None else \
            f"{f['k']:5d}/{f['n']:<5d} {f['pct']:6.2f}% [{f['ci'][0]:.2f},{f['ci'][1]:.2f}]"
        print(f"  ${r['cap']:<3d} fixed {fs}   variable {v['k']:5d}/{v['n']:<5d} "
              f"{v['pct']:6.2f}% [{v['ci'][0]:.2f},{v['ci'][1]:.2f}]")

    print("\npanel (b): mc32 matched-cap pairs, variable - fixed in points")
    for p in pairs:
        flag = "  <-- runs the other way" if p["delta_pp"] < 0 else ""
        star = "*" if p["fisher_p"] < 0.05 else " "
        ci = p["delta_newcombe_ci"] if p["delta_boot_degenerate"] else p["delta_boot_ci"]
        tag = "score" if p["delta_boot_degenerate"] else "boot "
        print(f"  {p['model_label']:17s} {p['prompt_combo']:6s} ${p['cap']:<3d} "
              f"fixed {p['fixed']['k']:2d}/{p['fixed']['n']:<3d} variable {p['variable']['k']:2d}/{p['variable']['n']:<3d} "
              f"delta {p['delta_pp']:+6.1f} {tag}[{ci[0]:+6.1f},{ci[1]:+6.1f}] "
              f"p={p['fisher_p']:.4f}{star}{flag}")

    print(f"\nreversed pairs: {len(reversed_pairs)} of {len(pairs)}")
    for p in reversed_pairs:
        print(f"  {p['model_label']} {p['prompt_combo']} ${p['cap']}: "
              f"fixed {p['fixed']['k']}/{p['fixed']['n']} vs variable {p['variable']['k']}/{p['variable']['n']}, "
              f"{p['delta_pp']:+.1f} pp, Fisher p={p['fisher_p']:.4f}")


if __name__ == "__main__":
    main()
