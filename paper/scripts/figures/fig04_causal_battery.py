"""Body Figure 4 — the causal battery — rebuilt from the raw §4 rollouts.

Why this file exists
--------------------
``images/fig_causal_battery.pdf`` shipped with the submission has no generator
anywhere on this machine.  This script reconstructs it from the raw per-trial
records so the panel can be regenerated, and it fixes four things the shipped
version got wrong or hid:

0. **The float now carries the whole causal battery.**  §4.4 consolidates the
   write results that were scattered across §4.1--4.3, so the figure gained
   two panels: (d) the pre-registered cross-task transfer matrix and (e) the
   condition-writability contrasts, both previously appendix-only.  Neither is
   a new measurement; both are re-read from what the appendix already
   reports, and (d)'s verdicts are re-derived and checked against the appendix
   table on every draw.  The canvas went from 376.2 x 341.7 pt at
   0.95\\textwidth to 396.0 x 252.0 pt at full \\textwidth, so the include line
   is ``width=\\textwidth`` and the print scale is 1.0.

1. **The random-direction band was a continuous ribbon.**  The random directions
   were only *run* at some doses: twenty directions at :math:`\\alpha=\\pm 3` on
   Gemma (``sec4_w2_null1..20_{am3,ap3}``) and five at :math:`\\alpha=+3` only on
   LLaMA (``sec4_w10a_null_1..5``).  Here the band is drawn only at the doses
   that were run; every other dose is marked as unrun.  :math:`\\alpha=0` is
   marked separately, because at zero dose the hook is the identity and all
   directions coincide by construction (all three Gemma arms return the same
   0.06084), so there is nothing for a random direction to differ from.
2. **The dose means are over parse-ok subsets, not over the 200 games run.**
   Parse rate falls as |dose| rises (Gemma balance at :math:`-3` drops to 0.790,
   below the pre-registered 0.80 gate; LLaMA behavioural at :math:`+3` drops to
   0.765).  The companion row under each dose panel plots the parse rate per
   dose per direction with Wilson intervals, so the shrinking denominator is
   visible, and gate-failing cells are drawn hollow and excluded from the slope.
3. **Removal is seed-paired.**  Every removal arm replays the same seeded state
   slice as its baseline, so records pair by ``seed``.  Panel (c) plots the
   paired within-seed delta as the bar and overlays the unpaired arm-mean
   difference as an open diamond, each with its own bootstrap interval.
4. **On Gemma, removing the balance control raises betting.**  Panel (c) is
   signed (``removed - baseline``, the sign convention of Table 2), diverging
   about zero, so the ``+0.046`` reads as an increase instead of being flattened
   into a bar length.

5. **Green and red meant two different things in the same paper.**  Gemma and
   LLaMA were drawn in ``#59A14F`` and ``#E15759``, which are the Fixed and
   Variable betting conditions on Figure 2, Figure 3, Figure 5 and every
   appendix bar panel.  Model identity now has its own pair -- Dark2 teal
   ``#1B9E77`` for Gemma, purple ``#7570B3`` for LLaMA -- with the readout
   control at ``#666666`` and the balance control at Dark2 amber ``#E6AB02``.
   The same four colours are used by ``fig_cross_context_write.py``, so the
   whole causal family reads as one set.  The random-direction band moved off
   ``#EDC948`` to neutral grey: it is a background reference, not a series,
   and its old yellow now neighbours the balance control's amber.
6. **The grey subtitles are gone and the keys are boxed.**  A grey second title
   under every panel head is not a device any submitted figure in this paper
   uses; the layer bands moved into the titles and the marker conventions into
   the keys, which are now framed like every other legend in the paper.  Both
   keys stay at figure level: the dose axes are 156 x 54 pt and the removal
   forest 84 x 52 pt, so no panel here can hold its own key inside its frame.

Statistics
----------
* Dose means: percentile bootstrap over the parse-ok trials of the cell.
* Parse rates: Wilson score interval.
* Null bands: mean +/- 2 SD across random directions, the "thick null"
  convention of ``multilayer_causal/src/sec4_stats.py``.
* Removal: percentile bootstrap over seed pairs (paired) or over trials
  (unpaired), plus an exact two-sided sign test on the nonzero within-seed
  differences.

Nothing is hardcoded: every number drawn or written comes from the rollouts.

Sources
-------
``sec4_p0``  Gemma natural-prompt ladder, three directions x seven doses.
``sec4_w2``  Gemma twenty random-direction nulls at +/-3.
``sec4_w10`` LLaMA ladder (``sec4_w10a_*``) and five nulls at +3.
``sec4_w13`` removal, base vs behavioural vs balance vs readout, seed-paired.

Each phase is read from the local mirror when present and otherwise pulled from
``llm-addiction-research/llm-addiction`` under
``experiments/sec4_causal/checkpoints/<phase>/``.

Panels and the claims they carry
--------------------------------
(a), (b)  sufficiency: dose-response on Gemma L16--21 and LLaMA L14--19, each
          over its parse-rate strip.
(c)       necessity: the seed-paired removal delta for all six model x
          direction cells.
(d)       cross-task transfer: the pre-registered 12-cell steering matrix.
(e)       condition writability: the prompt-frame contrasts on the dose slope.

Outputs
-------
``images/fig04_causal_battery_detail.pdf`` / ``.png``   (396.0 x 144.0 pt; the body
  Figure 4 is drawn from the same sidecar by fig04_causal_battery_paperstyle.py)
``images/fig04b_causal_removal.pdf`` / ``.png``  (396.0 x 120.0 pt)
both included at ``width=\\textwidth``
``paper_data/fig04_causal_battery.json``
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ``paper_figure_style`` is not in this repository and not on the dataset; the
# five names it supplied are vendored beside this file.
from paper_style_vendored import (COLORS, fit_titles, panel_title,  # noqa: E402
                                  save_pdf_png, style_axes, use_paper_style)

# The interval helpers the rest of the camera-ready uses.  Imported, not copied.
# They come with the corpus loaders, which only the --recompute path needs.
try:
    from build_figure_data import REPO_ID, wilson  # noqa: E402
except Exception:  # pragma: no cover
    REPO_ID = "llm-addiction-research/llm-addiction"
    wilson = None

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Optional local mirror of the §4 rollout checkpoints.  Point
# ``LLM_ADDICTION_ANALYSIS`` at the analysis checkout to read them from disk;
# ``phase_dir`` below falls back to ``HF_PREFIX`` on the public dataset.
ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS", Path.home() / "llm-addiction"))
LOCAL_RESULTS = ANALYSIS_ROOT / "experiments" / "08_steering" / "multilayer_causal" / "results"
HF_PREFIX = "experiments/sec4_causal/checkpoints"
CACHE = Path("/tmp/hfcache_fig04")
IMAGES = REPO_ROOT / "images"
SIDECAR = REPO_ROOT / "paper_data" / "fig04_causal_battery.json"

RNG = np.random.default_rng(24231)
N_BOOT = 2000

DOSES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
SUFFIX = {-3.0: "am3", -2.0: "am2", -1.0: "am1", 0.0: "a0",
          1.0: "ap1", 2.0: "ap2", 3.0: "ap3"}

# Pre-registered parse gates (appendix §causal-battery): Gemma SM 0.8,
# LLaMA SM 0.5.  Cells below the gate are dropped before behaviour is read.
PARSE_GATE = {"gemma": 0.80, "llama": 0.50}

# Direction identity is shared across panels: the model-coloured line is the one
# that writes, the two greys are the controls it has to beat.
AXIS_ORDER = ("behavioural", "readout", "confound")
AXIS_LABEL = {"behavioural": "behaviour-built direction", "readout": "readout direction",
              "confound": "balance control"}
AXIS_SHORT = {"behavioural": "behavioural", "readout": "readout",
              "confound": "balance"}
AXIS_STYLE = {"behavioural": dict(marker="o", ls="-", lw=2.0, ms=5.0),
              "readout": dict(marker="s", ls="--", lw=1.5, ms=4.0),
              "confound": dict(marker="^", ls=":", lw=1.5, ms=4.5)}
# Model identity is teal/purple, not green/red: green and red are the betting
# condition everywhere else in the paper (Figure 2, Figure 3, Figure 5) and one
# pair of hues cannot carry two meanings across one document.  The two control
# directions are a neutral grey and a Dark2 amber.
AXIS_COLOR = {"readout": COLORS["readout"], "confound": COLORS["balance"]}
MODEL_COLOR = {"gemma": COLORS["gemma"], "llama": COLORS["llama"]}
# The random-direction band is a background reference, not a fourth series, so
# it is drawn in neutral grey.  It used to be #EDC948, which now sits next to
# the balance control's amber.
NULL_COLOR = COLORS["null_band"]
NULL_MEAN_COLOR = "#8A8A8A"
NULL_TEXT_COLOR = "#707070"


# ------------------------------------------------------------------ loading


def phase_dir(phase: str) -> Path:
    """Local mirror of a §4 phase, downloading it from HF when absent."""
    local = LOCAL_RESULTS / phase
    if local.is_dir() and any(local.glob("*.jsonl")):
        return local
    from huggingface_hub import snapshot_download

    root = snapshot_download(REPO_ID, repo_type="dataset", cache_dir=str(CACHE),
                             allow_patterns=[f"{HF_PREFIX}/{phase}/*.jsonl"])
    return Path(root) / HF_PREFIX / phase


def rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cell(path: Path) -> dict | None:
    """One dose cell: n run, the parse-ok bet ratios, and the parse rate.

    Mirrors ``sec4_stats._arm_summary``: parse-fail rows are dropped before any
    behaviour is read, so ``n_parse_ok`` is the denominator of the mean and
    ``n_run`` is the number of games actually replayed.
    """
    if not path.exists():
        return None
    rs = rows(path)
    if not rs:
        return None
    ok = [r for r in rs if r.get("parse_ok")]
    bets = [float(r["bet_ratio"]) for r in ok if r.get("bet_ratio") is not None]
    return {"n_run": len(rs), "n_parse_ok": len(ok), "bets": bets,
            "parse_rate": len(ok) / len(rs)}


def by_seed(path: Path) -> dict[int, float]:
    """Parse-ok bet ratio keyed by seed, for the matched-pairs removal test."""
    out = {}
    for r in rows(path):
        if r.get("parse_ok") and r.get("bet_ratio") is not None \
                and r.get("seed") is not None:
            out[int(r["seed"])] = float(r["bet_ratio"])
    return out


# --------------------------------------------------------------- statistics


def boot_ci(values, n_boot: int = N_BOOT) -> tuple[float, float, float, int]:
    """Percentile interval for a mean over the unit the values were taken over."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]), 1)
    idx = RNG.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(arr.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), int(arr.size))


def mean_ci(values) -> dict:
    """Mean with an interval, degrading to a one-sided bracket at an all-zero cell.

    A dose cell where every parse-ok game bet nothing has a degenerate bootstrap
    (every resample is 0).  Rather than draw no interval, the upper end is the
    Wilson upper bound on the rate of betting at all, 0/n, which is the largest
    mean bet ratio compatible with the observation since bet ratio is at most 1.
    """
    arr = np.asarray(values, dtype=float)
    m, lo, hi, n = boot_ci(arr)
    one_sided = bool(n > 0 and np.all(arr == 0.0))
    if one_sided:
        hi = wilson(0, n)[1] / 100.0
        lo = 0.0
    return {"mean": m, "lo": lo, "hi": hi, "n": n, "one_sided": one_sided,
            "zero_count": int(n) if one_sided else None}


def ols_slope(xs, ys) -> float:
    """Trial-level OLS slope of behaviour on dose (``sec4_stats._ols_slope``)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 3 or float(x.std()) < 1e-12:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / x.var())


def sign_test_two_sided(diffs) -> tuple[float, int, int]:
    """Exact two-sided sign test on the nonzero within-pair differences."""
    d = np.asarray(diffs, dtype=float)
    nz = d[d != 0.0]
    n = int(nz.size)
    up = int((nz > 0).sum())
    if n == 0:
        return (float("nan"), 0, 0)
    tail = sum(math.comb(n, i) for i in range(min(up, n - up) + 1))
    return (min(1.0, 2.0 * tail * 0.5 ** n), up, n - up)


# ----------------------------------------------------------------- ladders


def ladder(model: str, results: Path, stem: str) -> dict:
    """Dose ladder for one model: per axis, per dose, mean + interval + parse.

    ``stem`` is the arm-id prefix: ``sec4_`` on Gemma (``sec4_behavioural_ap3``),
    ``sec4_w10a_`` on LLaMA.
    """
    gate = PARSE_GATE[model]
    out = {}
    for axis in AXIS_ORDER:
        cells = {}
        xs, ys = [], []
        for dose in DOSES:
            c = cell(results / f"{stem}{axis}_{SUFFIX[dose]}.jsonl")
            if c is None:
                continue
            est = mean_ci(c["bets"])
            gated = c["parse_rate"] >= gate
            plo, phi = wilson(c["n_parse_ok"], c["n_run"])
            cells[dose] = {
                "mean_bet_ratio": est["mean"], "ci95": [est["lo"], est["hi"]],
                "one_sided_bracket": est["one_sided"],
                "n_run": c["n_run"], "n_parse_ok": c["n_parse_ok"],
                "parse_rate": c["parse_rate"],
                "parse_rate_wilson95": [plo / 100.0, phi / 100.0],
                "passes_parse_gate": gated,
            }
            if gated:
                xs.extend([dose] * len(c["bets"]))
                ys.extend(c["bets"])
        out[axis] = {"doses": cells, "slope_parse_gated": ols_slope(xs, ys),
                     "n_trials_in_slope": len(ys)}
    return out


def null_band(files: dict[float, list[Path]]) -> dict:
    """Random-direction band per dose: mean +/- 2 SD of the per-direction means.

    ``files[dose]`` is the list of rollouts for that dose, one per direction.
    Doses absent from the mapping were never run and get no band.
    """
    out = {}
    for dose, paths in sorted(files.items()):
        means, n_dirs = [], 0
        for p in paths:
            c = cell(p)
            if c is None or not c["bets"]:
                continue
            means.append(float(np.mean(c["bets"])))
            n_dirs += 1
        if not means:
            continue
        arr = np.asarray(means)
        sd = float(arr.std()) if arr.size > 1 else 0.0
        out[dose] = {"n_directions": n_dirs, "mean": float(arr.mean()), "sd": sd,
                     "lo": float(arr.mean() - 2 * sd),
                     "hi": float(arr.mean() + 2 * sd),
                     "min": float(arr.min()), "max": float(arr.max()),
                     "per_direction_means": means}
    return out


def gemma_null_slopes(w2: Path) -> dict:
    """Twenty random-direction slopes from their -3/+3 pairs (the Gemma z band).

    This is the band Table 2's ``z`` column is scored against, reproduced here so
    the figure's slope annotations are checkable.
    """
    slopes = []
    for k in range(1, 21):
        xs, ys = [], []
        for dose in (-3.0, 3.0):
            c = cell(w2 / f"sec4_w2_null{k}_{SUFFIX[dose]}.jsonl")
            if c is None or c["parse_rate"] < PARSE_GATE["gemma"]:
                continue
            xs.extend([dose] * len(c["bets"]))
            ys.extend(c["bets"])
        s = ols_slope(xs, ys)
        if np.isfinite(s):
            slopes.append(s)
    arr = np.asarray(slopes)
    return {"n_directions": int(arr.size), "mean": float(arr.mean()),
            "sd": float(arr.std()), "slopes": [float(s) for s in arr]}


# ----------------------------------------------------------------- removal


def removal(model: str, w13: Path) -> dict:
    """Seed-paired removal table for one model, plus the unpaired arm means."""
    base_path = w13 / f"sec4_w13_{model}_base.jsonl"
    base_cell = cell(base_path)
    base_pairs = by_seed(base_path)
    b_mean, b_lo, b_hi, _ = boot_ci(base_cell["bets"])

    axes = {}
    for axis in AXIS_ORDER:
        p = w13 / f"sec4_w13_{model}_{axis}.jsonl"
        c = cell(p)
        if c is None:
            continue
        arm = by_seed(p)
        seeds = sorted(set(base_pairs) & set(arm))
        diffs = np.array([arm[s] - base_pairs[s] for s in seeds])
        pd_mean, pd_lo, pd_hi, _ = boot_ci(diffs)
        pval, n_up, n_down = sign_test_two_sided(diffs)

        # Unpaired: difference of the two independent arm means, bootstrapped
        # by resampling each arm's trials separately.
        a = np.asarray(c["bets"], dtype=float)
        b = np.asarray(base_cell["bets"], dtype=float)
        ia = RNG.integers(0, a.size, size=(N_BOOT, a.size))
        ib = RNG.integers(0, b.size, size=(N_BOOT, b.size))
        ud = a[ia].mean(axis=1) - b[ib].mean(axis=1)

        arm_mean, arm_lo, arm_hi, _ = boot_ci(a)
        axes[axis] = {
            "arm_mean_bet_ratio": arm_mean, "arm_ci95": [arm_lo, arm_hi],
            "n_run": c["n_run"], "n_parse_ok": c["n_parse_ok"],
            "parse_rate": c["parse_rate"],
            "delta_paired": pd_mean, "delta_paired_ci95": [pd_lo, pd_hi],
            "n_pairs": len(seeds), "n_pairs_up": n_up, "n_pairs_down": n_down,
            "p_paired_sign_test": pval,
            "delta_unpaired": float(a.mean() - b.mean()),
            "delta_unpaired_ci95": [float(np.percentile(ud, 2.5)),
                                    float(np.percentile(ud, 97.5))],
        }

    nulls = {}
    for p in sorted(w13.glob(f"sec4_w13_{model}_null*.jsonl")):
        c = cell(p)
        arm = by_seed(p)
        seeds = sorted(set(base_pairs) & set(arm))
        nulls[p.stem] = {
            "arm_mean_bet_ratio": float(np.mean(c["bets"])),
            "delta_paired": float(np.mean([arm[s] - base_pairs[s]
                                           for s in seeds])),
            "n_pairs": len(seeds), "parse_rate": c["parse_rate"]}

    return {"baseline_mean_bet_ratio": b_mean, "baseline_ci95": [b_lo, b_hi],
            "baseline_n_run": base_cell["n_run"],
            "baseline_n_parse_ok": base_cell["n_parse_ok"],
            "axes": axes, "random_direction_nulls": nulls}


# ------------------------------------------------- transfer and writability
# Panels (d) and (e) carry claims that used to live only in the appendix, so
# neither reads a rollout: (d) replays the pre-registered sign matrix already
# serialised for the appendix sign-map figure, and (e) quotes the appendix
# contrast table.  Both are copied into this figure's sidecar so the body float
# and the appendix cannot drift apart unnoticed.

# Written by ``scripts/figures/fig_cross_context_write.py``; the same twelve z
# values appear in appendix Table ``tab:causal-transfer-matrix``.
XCTX_VALUES = REPO_ROOT / "images" / "fig_cross_context_write_values.json"

# Row order = the four steering axes, column order = the three targets, both as
# in the appendix table.
TRANSFER_AXES = (("smiba", "SM betting"), ("icrc", "IC risky"),
                 ("mwrc", "MW spin"), ("sh3c", "shared"))
TRANSFER_TARGETS = (("sm", "SM"), ("ic", "IC"), ("mw", "MW"))

# The verdict column of appendix Table ``tab:causal-transfer-matrix``, verbatim.
# It is not drawn from here: the verdicts are re-derived from the pre-registered
# sign and the observed z below, and this table is the assertion they are
# checked against, so a silent change on either side fails the build.
TRANSFER_VERDICT_TABLE = {
    "smiba->sm": "hit",      "smiba->ic": "hit", "smiba->mw": "sign hit",
    "icrc->sm": "hit",       "icrc->ic": "hit",  "icrc->mw": "null",
    "mwrc->sm": "sign hit",  "mwrc->ic": "hit",  "mwrc->mw": "null",
    "sh3c->sm": "hit",       "sh3c->ic": "hit",  "sh3c->mw": "sign miss",
}
# |z| threshold the matrix was pre-registered against (appendix
# §appendix:causal-transfer: "seven of the ten primary cells move with the
# pre-registered sign at |z|>2").
TRANSFER_Z_GATE = 2.0

VERDICT_FACE = {"hit": "#D9E9D5", "sign hit": "#D9E9D5",
                "null": "#F0F0F0", "sign miss": "#F8DDDD"}
VERDICT_EDGE = {"hit": "#59A14F", "sign hit": "#59A14F",
                "null": "#C8C8C8", "sign miss": "#E15759"}

# Appendix Table ``tab:causal-condition-writability``, contrast rows verbatim:
# (model, row label, contrast, CI low, CI high).  The contrasts are differences
# of dose slopes on the common grid {-3,0,+3}, with 1000x bootstrap intervals.
# The twin conditions graft the goal / reward-maximisation module onto one
# matched replay pool; the bare -G pool is the control.
# Row labels name the grafted module in words rather than as $+G^{\text{twin}}$:
# a mathtext superscript at a 7 pt base prints at 4.9 pt, under this figure's
# floor, and the caption carries the mapping back to the appendix notation.
# The minus is U+2212 so the labels need no mathtext at all.
WRITABILITY = (
    ("gemma", "goal \u2212 (\u2212G)", +0.0111, +0.0069, +0.0156),
    ("gemma", "reward \u2212 (\u2212G)", -0.0140, -0.0218, -0.0066),
    ("gemma", "goal \u2212 reward", +0.0237, +0.0179, +0.0297),
    ("llama", "goal \u2212 reward", -0.0156, -0.0261, -0.0046),
)
# Per-condition dose slopes from the same table, on the same common grid; they
# are the levels the contrasts above are differences of.  Reported in the
# sidecar, annotated on the panel as its axis note.
WRITABILITY_SLOPES = {"matched -G pool": 0.0358, "+G twin": 0.0469,
                      "+M twin": 0.0218}


WHAT = ("The causal battery, drawn as two floats. Body Figure 4 "
        "(fig04_causal_battery) holds the dose ladders recomputed from the "
        "raw §4 rollouts: (a) Gemma dose-response, (b) LLaMA dose-response. "
        "Appendix Figure fig:causal-removal (fig04b_causal_removal) holds "
        "(a) removal, also recomputed from the rollouts, plus two panels "
        "carrying no new measurement: (b) cross-task transfer replays the "
        "pre-registered sign matrix serialised in "
        "images/fig_cross_context_write_values.json and re-checks every "
        "verdict against appendix Table tab:causal-transfer-matrix; (c) "
        "condition writability quotes appendix Table "
        "tab:causal-condition-writability. The sidecar keys keep their "
        "panel_a..panel_e names from the single-float layout.")


def side_panels(tm: dict) -> dict:
    """The sidecar blocks for (d) and (e), which read no rollout."""
    return {
        "panel_d_transfer": {
            "source": "images/fig_cross_context_write_values.json "
                      "(signmap_prereg_cells); verdicts cross-checked against "
                      "appendix Table tab:causal-transfer-matrix",
            "z_gate": TRANSFER_Z_GATE,
            **tm,
        },
        "panel_e_writability": {
            "source": "appendix Table tab:causal-condition-writability "
                      "(neurips_content_en/appendix.tex)",
            "units": "difference of dose slopes on the common grid {-3,0,+3}; "
                     "1000x bootstrap CIs",
            "contrasts": [{"model": m, "label": lab, "delta": d,
                           "ci95": [lo, hi]}
                          for m, lab, d, lo, hi in WRITABILITY],
            "per_condition_slopes": WRITABILITY_SLOPES,
        },
    }


def transfer_matrix() -> dict:
    """The twelve pre-registered transfer cells, verdict re-derived and checked.

    A cell hits when the observed z carries the pre-registered sign at
    ``|z| > 2``; it is null when the sign agrees but ``|z|`` does not clear the
    gate; it is a sign miss when the sign disagrees.  The two cells pre-marked
    low-confidence (steering axis nearly orthogonal to the target's own axis)
    are excluded from the ten-cell primary count and are recorded as "sign hit".
    """
    cells = json.loads(XCTX_VALUES.read_text())["signmap_prereg_cells"]
    out = {}
    for key, c in cells.items():
        z, pred, low = float(c["z"]), c["pred"], bool(c["low_conf"])
        agrees = (z > 0) == (pred == "+")
        if not agrees:
            verdict = "sign miss"
        elif abs(z) < TRANSFER_Z_GATE:
            verdict = "null"
        else:
            verdict = "sign hit" if low else "hit"
        want = TRANSFER_VERDICT_TABLE[key]
        if verdict != want:
            raise SystemExit(
                f"transfer cell {key}: derived verdict {verdict!r} disagrees "
                f"with appendix Table tab:causal-transfer-matrix ({want!r})")
        out[key] = {"pred_sign": pred, "z": z, "low_confidence": low,
                    "verdict": verdict}
    primary = [k for k, v in out.items() if not v["low_confidence"]]
    hits = [k for k in primary if out[k]["verdict"] == "hit"]
    signs = [k for k, v in out.items() if v["verdict"] != "sign miss"]
    return {"cells": out, "n_primary": len(primary), "n_primary_hits": len(hits),
            "n_sign_agree": len(signs), "n_cells": len(out)}


# ------------------------------------------------------------------ drawing

# Every dose panel reserves the right slice of its own x range for the slope
# and z annotations, so those never sit on the data and never overhang the
# canvas.  ``PLOT_FRAC`` is the share of the panel the seven doses occupy.
PLOT_FRAC = 0.615
X_LO, X_HI = -3.55, 3.55
X_HEAD_C = X_HEAD_D = X_HEAD_E = 0.0
X_MAX = X_LO + (X_HI - X_LO) / PLOT_FRAC


def panel_head(ax, letter: str, title: str, pad: float = 4.0,
               x: float | None = None):
    """Bold ``(x) Title`` above the axes.  Returns the artists it created.

    The grey one-line subtitle this used to set under every head is gone.  It
    was the only grey secondary title in the paper, it is not a device any of
    the submitted figures use, and every fact it carried -- the layer band, the
    null definition, the paired/unpaired marker convention -- is either in the
    title now, in the panel's own boxed key, or in the caption.

    ``x`` is an axes-fraction anchor: ``None`` centres the head over the axes,
    which is what the two wide dose columns want.  Row 2 packs three panels
    whose label gutters are wider than their axes, so those heads are
    left-aligned to the panel box instead, at a negative ``x``.  A left title
    cannot ride on ``ax.title`` -- matplotlib parks that one in the centre slot
    and ``fit_titles`` would then be nudging an empty string -- so the
    x-anchored head is drawn as one annotation instead.
    """
    if x is None:
        panel_title(ax, letter, title, pad=pad, fontsize=8.5)
        return []
    return [ax.annotate(f"{letter} {title}", xy=(x, 1.0),
                        xycoords="axes fraction", xytext=(0, 2.5),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=8.5, fontweight="bold",
                        annotation_clip=False)]


def declutter(fracs, min_gap: float):
    """Push a column of labels apart to ``min_gap`` without reordering them.

    ``fracs`` are axes-fraction y positions, ``min_gap`` the smallest fraction
    of the panel height two 7 pt lines may sit apart.  Labels are separated
    downward from the top, then the whole stack is slid back inside [0, 1] if
    the separation pushed it out.
    """
    order = sorted(range(len(fracs)), key=lambda i: -fracs[i])
    out = list(fracs)
    for k, i in enumerate(order):
        if k:
            out[i] = min(out[i], out[order[k - 1]] - min_gap)
    lowest = out[order[-1]]
    if lowest < min_gap * 0.6:
        shift = min_gap * 0.6 - lowest
        out = [v + shift for v in out]
    top = max(out)
    if top > 1.0 - min_gap * 0.6:
        out = [v - (top - (1.0 - min_gap * 0.6)) for v in out]
    return out


def draw_ladder(ax, ax_parse, model, lad, band, title, letter,
                zkey, h_dose, first=False):
    """One dose-response column: the ladder, its parse strip, and the
    per-direction slope and z in the panel's own right-hand gutter.

    The identity of the three curves comes from the figure-wide key under the
    dose block, so nothing here needs a per-panel legend; what stays on the
    panel is the pair of numbers that cannot be read off the curve, the
    parse-gated dose slope and its z against that model's random-direction
    null.  ``zkey`` names which of the two z fields the model carries: Gemma
    has a twenty-direction slope band, LLaMA only a five-direction level at
    the strongest positive dose.
    """
    colour = {**AXIS_COLOR, "behavioural": MODEL_COLOR[model]}
    gate = PARSE_GATE[model]

    # The band, only where random directions were actually run.
    for dose in sorted(band):
        nb = band[dose]
        ax.add_patch(plt.Rectangle((dose - 0.30, nb["lo"]), 0.60,
                                   max(nb["hi"] - nb["lo"], 1e-4),
                                   facecolor=NULL_COLOR, alpha=0.55,
                                   edgecolor=NULL_MEAN_COLOR, lw=0.8, zorder=1))
        ax.plot([dose - 0.30, dose + 0.30], [nb["mean"]] * 2,
                color=NULL_MEAN_COLOR, lw=1.0, zorder=2)

    for axis in AXIS_ORDER:
        cells = lad[axis]["doses"]
        ds = sorted(cells)
        m = [cells[d]["mean_bet_ratio"] for d in ds]
        lo = [cells[d]["mean_bet_ratio"] - cells[d]["ci95"][0] for d in ds]
        hi = [cells[d]["ci95"][1] - cells[d]["mean_bet_ratio"] for d in ds]
        st = AXIS_STYLE[axis]
        ax.errorbar(ds, m, yerr=[lo, hi], color=colour[axis], ecolor=colour[axis],
                    elinewidth=1.0, capsize=2.0, zorder=4, **st)
        # One-sided bracket where the whole cell bet nothing.
        for d in ds:
            c = cells[d]
            if c["one_sided_bracket"]:
                ax.annotate(f"0/{c['n_parse_ok']}", xy=(d, c["ci95"][1]),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.0, color=colour[axis])
        # Gate-failing cells: hollow, and out of the slope.
        fail = [d for d in ds if not cells[d]["passes_parse_gate"]]
        if fail:
            ax.plot(fail, [cells[d]["mean_bet_ratio"] for d in fail], "o",
                    mfc="white", mec=colour[axis], mew=1.4, ms=8.0, zorder=5)

    # Doses where no random direction was run.  alpha=0 is marked apart: the
    # hook is the identity there, so every direction gives the same cell by
    # construction and there is nothing for a random direction to differ from.
    # Footer strip: per dose, either the number of random directions run
    # (gold, matching the band) or a marker for "none run" / the alpha=0 case.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.06 * (hi - lo))
    lo, hi = ax.get_ylim()
    foot = lo - (7.5 / h_dose) * (hi - lo)     # 7.5 pt clear of the x spine
    for dose in DOSES:
        if dose in band:
            ax.text(dose, foot, str(band[dose]["n_directions"]), ha="center",
                    va="center", fontsize=7.0, color=NULL_TEXT_COLOR,
                    fontweight="bold", clip_on=False, zorder=3)
        elif dose == 0.0:
            ax.plot([dose], [foot], marker="o", ms=4.0, mfc="white",
                    mec="#9A9A9A", mew=1.0, clip_on=False, zorder=3)
        else:
            ax.plot([dose], [foot], marker=7, ms=4.5, color="#9A9A9A",
                    clip_on=False, zorder=3)

    # Slope and z, right-aligned in the gutter the x range reserves for them,
    # each on its own curve's line where the curves allow it and pushed apart
    # where they do not.
    ends = [lad[a]["doses"][max(lad[a]["doses"])]["mean_bet_ratio"]
            for a in AXIS_ORDER]
    fracs = declutter([(v - lo) / (hi - lo) for v in ends], 9.2 / h_dose)
    for axis, f in zip(AXIS_ORDER, fracs):
        e = lad[axis]
        ax.text(0.997, f, f"{e['slope_parse_gated']:+.3f}  z {e[zkey]:+.2f}",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=7.0, color=colour[axis], clip_on=False)
    ax.set_xlim(X_LO, X_MAX)
    ax.plot([X_HI, X_HI], [lo, hi], color="#E4E4E4", lw=0.7, zorder=0)

    ax.axvline(0.0, color="#BBBBBB", lw=0.7, zorder=0)
    ax.set_xticks(list(DOSES))
    # The parse strip directly below carries the shared dose axis labels.
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=7.4)
    if first:
        # Both rotated two-line labels are longer than the short axes they
        # sit beside, so they are pushed apart along their own axes: the dose
        # label rides high on its panel, the parse label low on its strip.
        ax.set_ylabel("mean bet ratio\n(parse-ok games)", fontsize=7.6,
                      linespacing=1.05, y=0.55)
    style_axes(ax)
    head_art = panel_head(ax, letter, title)

    # Companion row: the denominator behind every mean above.  It records the
    # pre-registered dose exclusion, so it stays even in the compact layout.
    for axis in AXIS_ORDER:
        cells = lad[axis]["doses"]
        ds = sorted(cells)
        pr = [cells[d]["parse_rate"] for d in ds]
        plo = [pr[i] - cells[d]["parse_rate_wilson95"][0]
               for i, d in enumerate(ds)]
        phi = [cells[d]["parse_rate_wilson95"][1] - pr[i]
               for i, d in enumerate(ds)]
        st = dict(AXIS_STYLE[axis])
        st["lw"] = 1.0
        st["ms"] = 3.0
        ax_parse.errorbar(ds, pr, yerr=[plo, phi], color=colour[axis],
                          ecolor=colour[axis], elinewidth=0.8, capsize=1.4, **st)
        for d in [d for d in ds if not cells[d]["passes_parse_gate"]]:
            ax_parse.plot([d], [cells[d]["parse_rate"]], "o", mfc="none",
                          mec="#C0392B", mew=1.4, ms=6.5, zorder=6)
            ax_parse.annotate(f"{cells[d]['parse_rate']:.3f} below gate",
                              xy=(d, cells[d]["parse_rate"]), xytext=(5, -1),
                              textcoords="offset points", ha="left", va="top",
                              fontsize=7.0, color="#C0392B")
    ax_parse.axhline(gate, color="#C0392B", lw=0.9, ls="--")
    # In the panel's own right-hand gutter, not over the dose range: at X_HI
    # this label lay across the parse curves, which run at 0.98-1.00.
    ax_parse.annotate(f"parse gate {gate:.2f}", xy=(X_MAX, gate),
                      xytext=(0, 2), textcoords="offset points", ha="right",
                      fontsize=7.0, color="#C0392B")
    ax_parse.set_ylim(0.45, 1.12)
    ax_parse.set_yticks([0.5, 0.75, 1.0])
    ax_parse.set_xticks(list(DOSES))
    ax_parse.set_xlim(X_LO, X_MAX)
    # The dose axis is named in the panel's own right-hand gutter, on the tick
    # row itself.  Under the ticks it would need a band of its own, and there
    # is no band to spare between the parse strip and the figure key; the
    # gutter beside the tick row is empty in both dose panels.
    ax_parse.annotate("steering dose", xy=(1.0, 0.0),
                      xycoords="axes fraction", xytext=(0, -8.5),
                      textcoords="offset points", ha="right", va="center",
                      fontsize=7.4, annotation_clip=False)
    if first:
        ax_parse.set_ylabel("parse rate\n(n=200)", fontsize=7.2,
                            linespacing=1.05, y=0.40)
    ax_parse.tick_params(axis="both", labelsize=7.2)
    style_axes(ax_parse)
    return head_art


def draw_removal(ax, rem, letter):
    """Six-row forest of the seed-paired removal delta, diverging about zero.

    Signed as ``removed - baseline`` (the sign convention of appendix
    Table~2), so Gemma's balance control reads as the increase it is.  The
    numbers behind the bars -- delta, sign-test p, and the pair counts -- are
    in that table; the panel carries the intervals, which the table cannot.
    """
    order = [(m, a) for m in ("gemma", "llama") for a in AXIS_ORDER]
    ys = np.arange(len(order))[::-1]
    top = float(ys.max())
    # Greyscale safety: the two control bars carry hatches, so behavioural
    # (solid model colour), readout and balance separate without colour.
    hatch = {"behavioural": None, "readout": "////", "confound": "\\\\\\\\"}
    labels = []
    for (model, axis), y in zip(order, ys):
        e = rem[model]["axes"][axis]
        d = e["delta_paired"]
        lo = d - e["delta_paired_ci95"][0]
        hi = e["delta_paired_ci95"][1] - d
        colour = (MODEL_COLOR[model] if axis == "behavioural"
                  else AXIS_COLOR[axis])
        ax.barh(y, d, height=0.60, color=colour, alpha=0.92, zorder=3,
                edgecolor="white", lw=0.5, hatch=hatch[axis])
        ax.errorbar(d, y, xerr=[[lo], [hi]], fmt="none", ecolor="#333333",
                    elinewidth=1.0, capsize=2.0, zorder=5)
        u = e["delta_unpaired"]
        ax.errorbar(u, y + 0.30, xerr=[[u - e["delta_unpaired_ci95"][0]],
                                       [e["delta_unpaired_ci95"][1] - u]],
                    fmt="D", mfc="white", mec="#333333", ms=3.0, mew=0.9,
                    ecolor="#888888", elinewidth=0.8, capsize=1.5, zorder=6)
        labels.append(AXIS_SHORT[axis])

    ax.axvline(0.0, color="#444444", lw=1.0, zorder=2)
    ax.axhline(2.5, color="#E4E4E4", lw=0.7, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.0)
    ax.tick_params(axis="y", pad=3, length=0)
    # Row pitch here is about 9 pt, so a two-line tick label would overrun its
    # row; the model name is a rotated group label spanning its own three rows.
    for model, y_mid in (("Gemma", 4.1), ("LLaMA", 0.9)):
        ax.text(-0.66, y_mid, model, transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=7.0,
                color=MODEL_COLOR[model.lower()], clip_on=False)
        ax.plot([-0.615, -0.615], [y_mid - 1.3, y_mid + 1.3],
                transform=ax.get_yaxis_transform(), color="#BBBBBB", lw=0.9,
                clip_on=False, zorder=1)
    ax.set_ylim(-0.55, top + 0.55)
    ax.set_xlabel(r"$\Delta$ bet ratio (removed $-$ baseline)", fontsize=7.6,
                  labelpad=1.5)
    ax.set_xlim(-0.095, 0.075)
    ax.set_xticks([-0.05, 0.0, 0.05])
    ax.set_xticklabels(["$-$.05", "0", "+.05"])
    ax.tick_params(axis="x", labelsize=7.4)
    style_axes(ax, grid_axis="x")
    return panel_head(ax, letter, "removal", x=X_HEAD_C)


def draw_transfer(ax, tm, letter):
    """The pre-registered 12-cell cross-task steering matrix.

    Rows are the four steering axes, columns the three targets.  Each cell
    carries the pre-registered direction as a triangle and the observed z as
    text; the fill is the verdict those two produce.  Nothing here is a new
    measurement: the z values are the ones the appendix table reports.
    """
    cells = tm["cells"]
    nrow, ncol = len(TRANSFER_AXES), len(TRANSFER_TARGETS)
    ax.set_xlim(0, ncol)
    # A header row is reserved inside the axes, so the column names cannot
    # collide with the panel head sitting just above the axes.
    ax.set_ylim(nrow, -0.62)
    for j, (_, tlab) in enumerate(TRANSFER_TARGETS):
        grey = tlab == "MW"
        ax.text(j + 0.5, -0.30, tlab, ha="center", va="center", fontsize=7.2,
                fontweight="bold", color="#8A8A8A" if grey else "#333333")
    for i, (akey, alab) in enumerate(TRANSFER_AXES):
        ax.text(-0.06, i + 0.5, alab, ha="right", va="center", fontsize=7.0,
                color="#333333")
        for j, (tkey, _) in enumerate(TRANSFER_TARGETS):
            c = cells[f"{akey}->{tkey}"]
            v = c["verdict"]
            ax.add_patch(plt.Rectangle(
                (j + 0.05, i + 0.09), 0.90, 0.82,
                facecolor=VERDICT_FACE[v], edgecolor=VERDICT_EDGE[v],
                lw=0.9, ls=(0, (1.6, 1.2)) if c["low_confidence"] else "-",
                zorder=2))
            ax.plot([j + 0.24], [i + 0.50],
                    marker="^" if c["pred_sign"] == "+" else "v",
                    ms=3.2, color="#6B6B6B", zorder=4)
            ax.text(j + 0.63, i + 0.50, f"{c['z']:+.1f}", ha="center",
                    va="center", fontsize=7.2, color="#222222", zorder=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("target (Gemma)", fontsize=7.6, labelpad=2.0, color="#555555")
    return panel_head(ax, letter, "cross-task transfer", x=X_HEAD_D)


def draw_writability(ax, letter):
    """Condition writability as a contrast forest of dose slopes.

    Each row is one difference of dose slopes on the common grid
    ``{-3, 0, +3}`` with its bootstrap interval, quoted from appendix
    Table~``tab:causal-condition-writability``.  Model identity rides on the
    colour the dose panels already established.
    """
    ys = np.arange(len(WRITABILITY))[::-1]
    for (model, lab, d, lo, hi), y in zip(WRITABILITY, ys):
        colour = MODEL_COLOR[model]
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", ms=3.6,
                    color=colour, ecolor=colour, elinewidth=1.1, capsize=2.0,
                    zorder=4)
    ax.axvline(0.0, color="#444444", lw=1.0, zorder=2)
    ax.axhline(0.5, color="#E4E4E4", lw=0.7, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([w[1] for w in WRITABILITY], fontsize=7.0)
    for tick, (model, *_rest) in zip(ax.get_yticklabels(), WRITABILITY):
        tick.set_color(MODEL_COLOR[model])
    ax.tick_params(axis="y", pad=3, length=0)
    ax.set_ylim(-0.62, float(ys.max()) + 0.62)
    ax.set_xlim(-0.033, 0.036)
    ax.set_xticks([-0.02, 0.0, 0.02])
    ax.set_xticklabels(["$-$.02", "0", "+.02"], fontsize=7.0)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_xlabel(r"$\Delta$ dose slope", fontsize=7.6, labelpad=1.5)
    style_axes(ax, grid_axis="x")
    return panel_head(ax, letter, "condition writability", x=X_HEAD_E)


# --------------------------------------------------------------------- figure


def _finish(fig, Wp, Hp, subs, named, stem) -> None:
    """Shared tail: fit the heads, nudge overhanging subtitles, report, write.

    ``subs`` are the annotation artists ``fit_titles`` cannot see; ``named`` is
    the (label, artist) list the FIG04_DEBUG band report prints.
    """
    fit_titles(fig)
    # The subtitles are not axes titles, so fit_titles does not see them.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for art in subs:
        bb = art.get_window_extent(renderer=r)
        s = 72.0 / fig.dpi
        shift = 0.0
        if bb.x1 * s > Wp - 2.0:
            shift = -(bb.x1 * s - (Wp - 2.0))
        elif bb.x0 * s < 2.0:
            shift = 2.0 - bb.x0 * s
        if shift:
            art.xyann = (art.xyann[0] + shift, art.xyann[1])

    if os.environ.get("FIG04_DEBUG"):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        s = 72.0 / fig.dpi

        def band(art):
            bb = art.get_window_extent(renderer=r)
            return (Hp - bb.y1 * s, Hp - bb.y0 * s)   # (top, bottom) in pt

        print(f"{stem}: canvas {Wp:.1f} x {Hp:.1f} pt")
        named = list(named)
        named += [(f"key{i}", lg) for i, lg in enumerate(fig.legends)]
        for name, art in named:
            t, b = band(art)
            print(f"  {name:12s} top={t:7.1f}  bottom={b:7.1f}  h={b - t:5.1f}")
        import matplotlib.text as mtext

        def drawn_texts():
            """Only the Text artists that are actually rendered.

            ``fig.findobj`` also returns the tick labels matplotlib keeps
            cached from an earlier autoscale; they carry stale strings at
            stale positions and would report phantom collisions.
            """
            def in_view(axis, ticks, labels):
                v0, v1 = sorted(axis.get_view_interval())
                span = v1 - v0
                return [lb for t, lb in zip(ticks, labels)
                        if v0 - 1e-9 * span <= t <= v1 + 1e-9 * span]

            out = []
            for a in fig.axes:
                out += [a.title, a.xaxis.label, a.yaxis.label]
                # The locator hands back ticks outside the view limits; their
                # labels exist and report a position but are never drawn.
                out += in_view(a.xaxis, a.get_xticks(), a.get_xticklabels())
                out += in_view(a.yaxis, a.get_yticks(), a.get_yticklabels())
                out += list(a.texts)
                if a.get_legend() is not None:
                    out += list(a.get_legend().findobj(mtext.Text))
            for lg in fig.legends:
                out += list(lg.findobj(mtext.Text))
            out += list(fig.texts)
            return out

        worst_t, worst_b, worst_l, worst_r = Hp, 0.0, Wp, 0.0
        smallest = 99.0
        for art in drawn_texts():
            if not art.get_text() or not art.get_visible():
                continue
            bb = art.get_window_extent(renderer=r)
            if bb.width <= 0 or bb.height <= 0:
                continue
            smallest = min(smallest, art.get_fontsize())
            t, b = Hp - bb.y1 * s, Hp - bb.y0 * s
            l, rt = bb.x0 * s, bb.x1 * s
            worst_t, worst_b = min(worst_t, t), max(worst_b, b)
            worst_l, worst_r = min(worst_l, l), max(worst_r, rt)
            if l < -0.5 or rt > Wp + 0.5 or t < -0.5 or b > Hp + 0.5:
                print(f"    OVERHANG {art.get_text()[:34]!r} "
                      f"l={l:.1f} r={rt:.1f} t={t:.1f} b={b:.1f}")
        print(f"  text box: top={worst_t:.1f} bottom={worst_b:.1f} "
              f"left={worst_l:.1f} right={worst_r:.1f}")
        print(f"  smallest font in use: {smallest:.2f} pt")

        # Pairwise overlap report.  Two labels whose ink boxes intersect by
        # more than a hairline are a layout bug, and at this density they are
        # easy to introduce and hard to see.  Text inside one legend is
        # skipped: matplotlib packs those and they cannot collide.
        boxes = []
        for art in drawn_texts():
            t = art.get_text()
            if not t or not art.get_visible():
                continue
            bb = art.get_window_extent(renderer=r)
            if bb.width <= 0 or bb.height <= 0:
                continue
            owner = None
            for lg in fig.legends:
                if art in lg.findobj(mtext.Text):
                    owner = lg
            boxes.append((t, owner, bb.x0 * s, bb.x1 * s,
                          Hp - bb.y1 * s, Hp - bb.y0 * s))
        n_hit = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[1] is not None and a[1] is b[1]:
                    continue
                ox = min(a[3], b[3]) - max(a[2], b[2])
                oy = min(a[5], b[5]) - max(a[4], b[4])
                if ox > 0.7 and oy > 0.7:
                    n_hit += 1
                    print(f"    OVERLAP {a[0][:22]!r} x {b[0][:22]!r} "
                          f"({ox:.1f} x {oy:.1f} pt)")
        print(f"  overlapping text pairs: {n_hit}")

    save_pdf_png(fig, IMAGES, stem)
    plt.close(fig)


def render(gemma, llama, g_band, l_band, g_slope_band, lnb, rem, tm) -> None:
    """Draw and write both floats.  Pure layout: it reads no data source."""
    # ------------------------------------------------------------- figures
    # Both floats are included at \textwidth.  \textwidth is 5.5 in and
    # matplotlib writes 72 PDF points to the inch, so each canvas is exactly
    # 396.0 bp wide and the LaTeX scale is 1.000: every nominal point size
    # below IS the printed size.  Floor: 7.0 pt.
    #
    # The battery no longer rides on a single body float.  The camera-ready is
    # one page over the limit, so the body keeps the two dose ladders -- the
    # sufficiency evidence, read over its necessity companion -- and the three
    # derived panels move to their own appendix float, next to the tables they
    # were re-read from.  No plotted value changes; only which page it prints
    # on.  The two canvases:
    #
    #   fig04_causal_battery   396.0 x 144.0 pt  (a) Gemma ladder, (b) LLaMA
    #   fig04b_causal_removal  396.0 x 120.0 pt  (a) removal, (b) transfer,
    #                                            (c) condition writability
    #
    # The panel rows are packed at their measured heights, unchanged from the
    # five-panel canvas they were cut from; only the head band and the key
    # band moved, when the grey subtitles came off and the keys were boxed:
    #
    #    13  panel heads, bold 8.5 pt, no subtitle
    #    54  dose ladders
    #     8  the random-direction footer strip, hung below the axes
    #    20  parse strips
    #    17  the shared dose ticks and axis label
    #    23  the boxed two-row figure key for the dose block
    #    15  appendix panel heads
    #    52  removal six forest rows, transfer four matrix rows, writability four
    #    16  their ticks and axis labels
    #    25  the boxed two-row key for the appendix row
    #
    # Run with FIG04_DEBUG=1 to print those bands back out of the drawn
    # figures, with an overhang check on every Text artist.
    use_paper_style(9.0)
    plt.rcParams["figure.constrained_layout.use"] = False
    plt.rcParams["axes.labelpad"] = 2.5
    plt.rcParams["ytick.major.pad"] = 2.5

    render_body(gemma, llama, g_band, l_band, lnb)
    render_appendix(rem, tm)


def render_body(gemma, llama, g_band, l_band, lnb) -> None:
    """The body float: the two dose ladders over their parse strips."""
    FIG_W, FIG_H = 396.0 / 72.0, 144.0 / 72.0
    Wp, Hp = FIG_W * 72.0, FIG_H * 72.0

    def rect(x0, y_top, w, h):
        """Axes rectangle from points measured down from the top-left."""
        return [x0 / Wp, (Hp - y_top - h) / Hp, w / Wp, h / Hp]

    # The two dose columns.  Left margin holds the parse strip's two-line y
    # label and its "1.00" ticks; the column gap holds (b)'s ticks.
    L, PW, COL_GAP = 46.0, 156.0, 32.0
    # Dropping the grey subtitle takes the head band from 21 pt to 13, and the
    # eight points go to the boxed key, which needs a border and a padding the
    # frameless two-row key did not.
    Y_DOSE, H_DOSE = 13.0, 54.0
    Y_PARSE, H_PARSE = 81.0, 20.0
    Y_KEY = 130.0                          # centre of the boxed key

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a = fig.add_axes(rect(L, Y_DOSE, PW, H_DOSE))
    ax_ap = fig.add_axes(rect(L, Y_PARSE, PW, H_PARSE))
    ax_b = fig.add_axes(rect(L + PW + COL_GAP, Y_DOSE, PW, H_DOSE))
    ax_bp = fig.add_axes(rect(L + PW + COL_GAP, Y_PARSE, PW, H_PARSE))

    subs = []
    for art in [
        draw_ladder(ax_a, ax_ap, "gemma", gemma, g_band,
                    "Gemma dose–response (L16–21)",
                    "(a)", "z_vs_null_slope_band", H_DOSE, first=True),
        draw_ladder(ax_b, ax_bp, "llama", llama, l_band,
                    "LLaMA dose–response (L14–19)",
                    "(b)", "z_vs_null_level_at_plus3", H_DOSE),
    ]:
        subs.extend(art)

    # The dose panels' marker conventions, in one boxed key under the dose
    # block.  A key inside a panel is what the rest of the paper does, and it
    # is what these two panels cannot do: each axes is 156 x 54 pt and (b)'s
    # six curves leave no corner free, so the key is boxed at figure level and
    # shared, which is also honest -- every entry applies to both panels.
    h_band = Patch(fc=NULL_COLOR, alpha=0.55, ec=NULL_MEAN_COLOR)
    h_n = Line2D([], [], marker="$n$", ls="none", color=NULL_TEXT_COLOR, ms=6)
    h_none = Line2D([], [], marker=7, ls="none", color="#9A9A9A", ms=5)
    h_zero = Line2D([], [], marker="o", ls="none", mfc="white", mec="#9A9A9A",
                    mew=1.0, ms=4.5)
    h_gate = Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555",
                    mew=1.4, ms=7)
    # The behaviour-built direction is drawn in its model's own colour, so its key entry
    # is the two model colours side by side under one label rather than a grey
    # stand-in -- grey is the readout direction on this figure.
    h_beh = tuple(Line2D([], [], color=MODEL_COLOR[m],
                         **AXIS_STYLE["behavioural"]) for m in ("gemma", "llama"))
    h_read = Line2D([], [], color=AXIS_COLOR["readout"],
                    **AXIS_STYLE["readout"])
    h_conf = Line2D([], [], color=AXIS_COLOR["confound"],
                    **AXIS_STYLE["confound"])

    # ncol=4 fills column-major, so the entries are ordered down each column:
    # row 1 is the three directions plus the gate marker, row 2 the
    # random-direction conventions.
    handles = [h_beh, h_band, h_read, h_n, h_conf, h_none, h_gate, h_zero]
    labels = ["behaviour-built direction",
              r"random directions, mean$\pm$2 SD (at doses run)",
              "readout", "n dirs run",
              "balance control", "none run",
              "below gate", "dose 0: identity"]
    fig.legend(handles=handles, labels=labels, loc="center",
               bbox_to_anchor=(0.5, (Hp - Y_KEY) / Hp), ncol=4,
               fontsize=6.8, handlelength=1.9, columnspacing=1.0,
               handletextpad=0.4, borderpad=0.45, borderaxespad=0.0,
               labelspacing=0.35, frameon=True, edgecolor="#CCCCCC",
               framealpha=1.0, fancybox=False,
               handler_map={tuple: HandlerTuple(ndivide=2, pad=0.0)})

    _finish(fig, Wp, Hp, subs,
            [("(a) title", ax_a.title), ("(b) title", ax_b.title),
             ("(a) xlabel", ax_ap.xaxis.label)],
            "fig04_causal_battery_detail")


def render_appendix(rem, tm) -> None:
    """The appendix float: removal, cross-task transfer, condition writability.

    The same three panels the five-panel body float carried as (c)-(e), at the
    same sizes and with the same content, re-lettered (a)-(c) as a standalone
    figure.
    """
    # 396 x 120 pt.  The canvas grew 9 pt: the key is boxed now and carries
    # the two marker conventions panel (a)'s grey subtitle used to state, which
    # costs a second key row and a border, and none of the three panels may be
    # shrunk to pay for it -- (a) is six forest rows at a 8.7 pt pitch already.
    # This is an appendix float, so the 9 pt does not come off the body's ten
    # pages.
    FIG_W, FIG_H = 396.0 / 72.0, 120.0 / 72.0
    Wp, Hp = FIG_W * 72.0, FIG_H * 72.0

    def rect(x0, y_top, w, h):
        return [x0 / Wp, (Hp - y_top - h) / Hp, w / Wp, h / Hp]

    # The grey subtitles are gone, so the head band is 15 pt instead of 23 and
    # the eight points go to the key, which is now boxed and carries the two
    # marker conventions panel (a)'s subtitle used to state.
    Y_ROW, H_ROW = 15.0, 52.0
    # The row packs three panels whose label gutters are wider than their axes,
    # so each panel is a box and its head is left-aligned to that box.  The
    # boxes are 2-146, 150-272 and 276-394 pt.
    X_C, W_C = 62.0, 84.0                  # removal axes; 62 pt of labels left
    X_D, W_D = 192.0, 80.0                 # transfer grid; row labels 150..192
    X_E, W_E = 332.0, 60.0                 # writability axes; labels 276..332
    Y_KEY_D = 105.0                        # centre of the boxed key

    # Head anchors, as axes fractions of each panel's own axes.
    global X_HEAD_C, X_HEAD_D, X_HEAD_E
    X_HEAD_C = (2.0 - X_C) / W_C
    X_HEAD_D = (150.0 - X_D) / W_D
    X_HEAD_E = (276.0 - X_E) / W_E

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_c = fig.add_axes(rect(X_C, Y_ROW, W_C, H_ROW))
    ax_d = fig.add_axes(rect(X_D, Y_ROW, W_D, H_ROW))
    ax_e = fig.add_axes(rect(X_E, Y_ROW, W_E, H_ROW))

    subs = []
    for art in [
        draw_removal(ax_c, rem, "(a)"),
        draw_transfer(ax_d, tm, "(b)"),
        draw_writability(ax_e, "(c)"),
    ]:
        subs.extend(art)

    # One boxed key for the row.  Panel (a)'s bar/diamond convention used to be
    # a grey subtitle over its axes; it is a key entry now, beside the verdict
    # fills it shares the row with.  (a) is 84 x 52 pt with six forest rows and
    # (b) is a 3 x 4 matrix, so neither can hold a key inside its own frame.
    # Two patches under one label: the behavioural bar takes its model's own
    # colour, so a single-colour swatch would name one model and not the other.
    h_bar = (Patch(fc=MODEL_COLOR["gemma"], ec="white", lw=0.5),
             Patch(fc=MODEL_COLOR["llama"], ec="white", lw=0.5))
    h_diam = Line2D([], [], marker="D", ls="none", mfc="white", mec="#333333",
                    ms=3.0, mew=0.9)
    h_hit = Patch(fc=VERDICT_FACE["hit"], ec=VERDICT_EDGE["hit"], lw=0.9)
    h_null = Patch(fc=VERDICT_FACE["null"], ec=VERDICT_EDGE["null"], lw=0.9)
    h_miss = Patch(fc=VERDICT_FACE["sign miss"], ec=VERDICT_EDGE["sign miss"],
                   lw=0.9)
    h_low = Patch(fc=VERDICT_FACE["hit"], ec=VERDICT_EDGE["hit"], lw=0.9,
                  ls=(0, (1.6, 1.2)))
    # ncol=3 fills column-major: row 1 is (a)'s two marker conventions plus the
    # low-confidence cell, row 2 the three (b) verdict fills.
    fig.legend(handles=[h_bar, h_hit, h_diam, h_null, h_low, h_miss],
               labels=[r"(a) bar: seed-paired $\Delta$",
                       r"(b) predicted sign, $|z|>2$",
                       "(a) $\\diamond$: unpaired arms", "null, $|z|<2$",
                       "low-confidence cell", "sign miss"],
               loc="center", bbox_to_anchor=(0.5, (Hp - Y_KEY_D) / Hp),
               ncol=3, fontsize=7.0, handlelength=1.3, columnspacing=1.6,
               handletextpad=0.4, borderpad=0.45, borderaxespad=0.0,
               labelspacing=0.35, frameon=True, edgecolor="#CCCCCC",
               framealpha=1.0, fancybox=False,
               handler_map={tuple: HandlerTuple(ndivide=2, pad=0.0)})

    _finish(fig, Wp, Hp, subs,
            [("(a) xlabel", ax_c.xaxis.label), ("(c) xlabel", ax_e.xaxis.label)],
            "fig04b_causal_removal")


# --------------------------------------------------------------------- replot


def from_sidecar() -> tuple:
    """The plotted quantities, read back from the sidecar this script writes.

    ``main()`` needs the four raw rollout phases (a local mirror, or the gated
    HuggingFace dataset).  Everything the figure draws is already serialised
    into ``paper_data/fig04_causal_battery.json``, so a pure re-draw -- which is
    what a layout change is -- reads that file and touches no data source.
    JSON object keys are strings, so the dose keys are cast back to float.
    """
    d = json.loads(SIDECAR.read_text())

    def ladder_of(node):
        out = {}
        for axis, e in node.items():
            out[axis] = dict(e)
            out[axis]["doses"] = {float(k): v for k, v in e["doses"].items()}
        return out

    def band_of(node):
        return {float(k): v for k, v in node.items()}

    gemma = ladder_of(d["panel_a_gemma"]["ladder"])
    llama = ladder_of(d["panel_b_llama"]["ladder"])
    g_band = band_of(d["panel_a_gemma"]["null_band_by_dose"])
    l_band = band_of(d["panel_b_llama"]["null_band_by_dose"])
    g_slope_band = d["panel_a_gemma"]["null_slope_band"]
    # (d) and (e) are not rollout products: (d) is re-derived and re-checked
    # against the appendix verdict column on every draw, (e) is quoted from the
    # appendix table.  Neither is read back from the sidecar, so a stale
    # sidecar cannot smuggle a wrong transfer cell into the body float.
    return (gemma, llama, g_band, l_band, g_slope_band, l_band[3.0],
            d["panel_c_removal"], transfer_matrix())


def replot() -> None:
    args = from_sidecar()
    render(*args)
    # (d) and (e) are appendix-sourced, so a redraw can refresh their sidecar
    # blocks without any rollout.  The rollout-derived blocks are left alone.
    d = json.loads(SIDECAR.read_text())
    d["what"] = WHAT
    d.update(side_panels(args[-1]))
    SIDECAR.write_text(json.dumps(d, indent=1))
    print(f"refreshed {SIDECAR} panels (d), (e)")
    for stem in ("fig04_causal_battery_detail", "fig04b_causal_removal"):
        print(f"wrote {IMAGES / (stem + '.pdf')}")
        print(f"wrote {IMAGES / (stem + '.png')}")


# --------------------------------------------------------------------- main


def main() -> None:
    if wilson is None:
        raise SystemExit("build_figure_data is unavailable; run with --replot "
                         "to redraw from paper_data/fig04_causal_battery.json")
    p0 = phase_dir("sec4_p0")
    w2 = phase_dir("sec4_w2")
    w10 = phase_dir("sec4_w10")
    w13 = phase_dir("sec4_w13")

    gemma = ladder("gemma", p0, "sec4_")
    llama = ladder("llama", w10, "sec4_w10a_")

    g_band = null_band({d: [w2 / f"sec4_w2_null{k}_{SUFFIX[d]}.jsonl"
                            for k in range(1, 21)] for d in (-3.0, 3.0)})
    l_band = null_band({3.0: [w10 / f"sec4_w10a_null_{k}.jsonl"
                              for k in range(1, 6)]})
    g_slope_band = gemma_null_slopes(w2)

    rem = {m: removal(m, w13) for m in ("gemma", "llama")}

    # Table 2's z column: slope against the twenty-direction slope band.
    def zscore(slope):
        return ((slope - g_slope_band["mean"]) / g_slope_band["sd"]
                if g_slope_band["sd"] > 1e-12 else float("nan"))

    for axis in AXIS_ORDER:
        gemma[axis]["z_vs_null_slope_band"] = zscore(gemma[axis]["slope_parse_gated"])
    # LLaMA random directions ran at +3 only, so there is no slope band there.
    # The honest LLaMA contrast is a level contrast at +3.
    lnb = l_band[3.0]
    for axis in AXIS_ORDER:
        top = llama[axis]["doses"][3.0]["mean_bet_ratio"]
        llama[axis]["z_vs_null_level_at_plus3"] = (
            (top - lnb["mean"]) / lnb["sd"] if lnb["sd"] > 1e-12 else float("nan"))
        llama[axis]["z_vs_null_slope_band"] = None

    tm = transfer_matrix()
    render(gemma, llama, g_band, l_band, g_slope_band, lnb, rem, tm)

    # ------------------------------------------------------------ sidecar
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(float(o)) else float(o)
        if isinstance(o, (np.integer, int)):
            return int(o)
        return o

    payload = {
        "what": WHAT,
        # Recorded as dataset-relative paths, not as whatever local mirror or
        # HF cache directory this run happened to read them from: the sidecar is
        # a provenance record for readers, and an absolute path names a machine
        # nobody else has.
        "sources": {
            "gemma_ladder": f"{HF_PREFIX}/sec4_p0",
            "gemma_nulls": f"{HF_PREFIX}/sec4_w2",
            "llama_ladder_and_nulls": f"{HF_PREFIX}/sec4_w10",
            "removal": f"{HF_PREFIX}/sec4_w13",
            "hf_repo": REPO_ID, "hf_prefix": HF_PREFIX,
        },
        "conventions": {
            "dose_mean": "mean bet ratio over parse-ok rows; n_run is the games "
                         "replayed, n_parse_ok the denominator of the mean",
            "dose_ci": f"percentile bootstrap over trials, {N_BOOT} resamples",
            "parse_rate_ci": "Wilson score interval",
            "null_band": "mean +/- 2 SD of the per-direction means, at the doses "
                         "the random directions were actually run",
            "removal_delta": "removed - baseline (Table 2 sign convention); "
                             "paired = mean within-seed difference, unpaired = "
                             "difference of arm means",
            "removal_p": "exact two-sided sign test on nonzero within-seed "
                         "differences; n_pairs_up and n_pairs_down count only "
                         "those nonzero pairs, so they need not sum to n_pairs",
            "parse_gates": PARSE_GATE,
        },
        "panel_a_gemma": {"ladder": gemma, "null_band_by_dose": g_band,
                          "null_slope_band": g_slope_band,
                          "doses_without_random_directions":
                              [d for d in DOSES if d not in g_band]},
        "panel_b_llama": {"ladder": llama, "null_band_by_dose": l_band,
                          "null_slope_band": None,
                          "doses_without_random_directions":
                              [d for d in DOSES if d not in l_band]},
        "panel_c_removal": rem,
        **side_panels(tm),
    }
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(clean(payload), indent=1))

    # ------------------------------------------------------------- report
    print("=== (a) Gemma ladder, sec4_p0 ===")
    for axis in AXIS_ORDER:
        e = gemma[axis]
        print(f"  {axis:12s} slope={e['slope_parse_gated']:+.6f} "
              f"z={e['z_vs_null_slope_band']:+.4f}")
        for d in sorted(e["doses"]):
            c = e["doses"][d]
            flag = "" if c["passes_parse_gate"] else "  <-- BELOW PARSE GATE"
            print(f"      a={d:+.0f} mean={c['mean_bet_ratio']:.5f} "
                  f"CI=[{c['ci95'][0]:.5f},{c['ci95'][1]:.5f}] "
                  f"n_ok={c['n_parse_ok']}/{c['n_run']} "
                  f"parse={c['parse_rate']:.3f}{flag}")
    print(f"  null slope band: n={g_slope_band['n_directions']} "
          f"mean={g_slope_band['mean']:.6f} sd={g_slope_band['sd']:.6f}")
    for d, nb in sorted(g_band.items()):
        print(f"  null level band a={d:+.0f}: {nb['n_directions']} dirs "
              f"mean={nb['mean']:.5f} sd={nb['sd']:.5f} "
              f"band=[{nb['lo']:.5f},{nb['hi']:.5f}]")
    print(f"  doses with NO random-direction run: "
          f"{[d for d in DOSES if d not in g_band]}")

    print("=== (b) LLaMA ladder, sec4_w10a ===")
    for axis in AXIS_ORDER:
        e = llama[axis]
        print(f"  {axis:12s} slope={e['slope_parse_gated']:+.6f} "
              f"level-z at +3={e['z_vs_null_level_at_plus3']:+.4f}")
        for d in sorted(e["doses"]):
            c = e["doses"][d]
            flag = "" if c["passes_parse_gate"] else "  <-- BELOW PARSE GATE"
            print(f"      a={d:+.0f} mean={c['mean_bet_ratio']:.5f} "
                  f"CI=[{c['ci95'][0]:.5f},{c['ci95'][1]:.5f}] "
                  f"n_ok={c['n_parse_ok']}/{c['n_run']} "
                  f"parse={c['parse_rate']:.3f}{flag}")
    print(f"  null level band a=+3: {lnb['n_directions']} dirs "
          f"mean={lnb['mean']:.5f} sd={lnb['sd']:.5f} "
          f"band=[{lnb['lo']:.5f},{lnb['hi']:.5f}]")
    print(f"  doses with NO random-direction run: "
          f"{[d for d in DOSES if d not in l_band]}")

    print("=== (c) removal, sec4_w13 ===")
    for model in ("gemma", "llama"):
        r = rem[model]
        print(f"  {model}: baseline={r['baseline_mean_bet_ratio']:.5f} "
              f"CI=[{r['baseline_ci95'][0]:.5f},{r['baseline_ci95'][1]:.5f}] "
              f"n_ok={r['baseline_n_parse_ok']}/{r['baseline_n_run']}")
        for axis in AXIS_ORDER:
            e = r["axes"][axis]
            print(f"    {axis:12s} arm={e['arm_mean_bet_ratio']:.5f} "
                  f"paired={e['delta_paired']:+.5f} "
                  f"CI=[{e['delta_paired_ci95'][0]:+.5f},"
                  f"{e['delta_paired_ci95'][1]:+.5f}] "
                  f"unpaired={e['delta_unpaired']:+.5f} "
                  f"CI=[{e['delta_unpaired_ci95'][0]:+.5f},"
                  f"{e['delta_unpaired_ci95'][1]:+.5f}] "
                  f"pairs={e['n_pairs']} ({e['n_pairs_down']}dn/"
                  f"{e['n_pairs_up']}up) p_sign={e['p_paired_sign_test']:.4f}")
        for k, v in r["random_direction_nulls"].items():
            print(f"    [null] {k}: arm={v['arm_mean_bet_ratio']:.5f} "
                  f"paired={v['delta_paired']:+.5f}")

    for stem in ("fig04_causal_battery_detail", "fig04b_causal_removal"):
        print(f"wrote {IMAGES / (stem + '.pdf')}")
        print(f"wrote {IMAGES / (stem + '.png')}")
    print(f"wrote {SIDECAR}")


if __name__ == "__main__":
    # Default: redraw from the sidecar, which needs no rollouts.  --recompute
    # re-reads the four raw phases and rewrites the sidecar as well.
    if "--recompute" in sys.argv:
        main()
    else:
        replot()
