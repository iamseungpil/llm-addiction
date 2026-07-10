"""Stage C — cross-task BK steering DOSE-SWEEP analysis (gemma).

The alpha=4.0 transfer wave (xtask_stats / experiments/xtask) collapsed the
geometrically-misaligned off-diagonal cells: forcing a foreign BK axis in at the
within-task-calibrated dose silenced generation (sm->mw 0% parse), so behaviour
transfer could not be read. This module combines that wave (alpha 4) with the
lower-dose wave (alpha 1/2/3, experiments/xtask_dose) to:

  1. trace, per cell (source axis s -> target task t), parse_ok(alpha) and
     behaviour(alpha) curves;
  2. pick, per target, the COHERENCE-PRESERVING dose = the largest alpha whose
     diagonal (s==t) parse_ok stays within `coh_tol` of the no-steer baseline;
  3. read off-diagonal transfer at that coherent dose (null-subtracted), and
  4. test whether the per-cell behaviour shift tracks the source-axis cosine
     (the sm-column finding from Stage B: sign(shift) == sign(cos)).

Behaviour metric = mean_risky (propensity to gamble/continue; defined for all
three tasks — ic/mw expose `risky`, sm derives it from action=='bet'). Pure
matrix math is unit-tested in __main__ (--selftest); no fabricated cells — an
absent arm is reported, never invented.

Reuses xtask_stats primitives UNMODIFIED (_read_jsonl, summarize_arm,
load_source_axes, cosine_matrix). Isolated: reads results/xtask_dose +
results/xtask, writes results/xtask_dose/xtask_dose.json + .png. Stage B and the
alpha=4 cells are read-only inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from multilayer_causal.src import xtask_stats as xs

TASKS = ("sm", "ic", "mw")
ALPHAS = (1.0, 2.0, 3.0, 4.0)            # 1-3 = dose wave, 4 = Stage B wave
COH_TOL = 0.15                            # diagonal parse may drop this far below
#                                           baseline and still count as coherent
_MLC = Path(__file__).resolve().parents[1]
DOSE_DIR = _MLC / "results" / "xtask_dose"
BASE_DIR = _MLC / "results" / "xtask"     # Stage B (alpha=4 cells + baselines)
ASSET_DIR = _MLC / "assets" / "xtask"
OUT_JSON = DOSE_DIR / "xtask_dose.json"
OUT_PNG = DOSE_DIR / "xtask_dose.png"
METRIC = "mean_risky"


# --------------------------------------------------------------------------- #
# arm-id <-> (kind, source/target, alpha) bridge                              #
# --------------------------------------------------------------------------- #
def cell_id(s: str, t: str, alpha: float) -> str:
    """Steer cell id. alpha 4 is the Stage-B wave (no suffix); 1-3 carry _a{n}."""
    return f"xtask_{s}_to_{t}" if alpha == 4.0 else f"xtask_{s}_to_{t}_a{int(alpha)}"


def null_id(t: str, alpha: float) -> str:
    return f"xtask_rnd_{t}_s0" if alpha == 4.0 else f"xtask_rnd_{t}_a{int(alpha)}"


def base_id(t: str) -> str:
    return f"xtask_base_{t}"


# --------------------------------------------------------------------------- #
# data loading                                                                #
# --------------------------------------------------------------------------- #
def load_all_summaries() -> Dict[str, dict]:
    """Merge per-arm summaries from the dose wave and the Stage-B wave. On an id
    collision the dose wave wins (it should not happen — disjoint ids)."""
    merged: Dict[str, dict] = {}
    for d in (BASE_DIR, DOSE_DIR):
        if d.is_dir():
            merged.update(xs.load_arm_summaries(d))
    return merged


def _metric(summ: Optional[dict]) -> float:
    return float("nan") if not summ else float(summ.get(METRIC, float("nan")))


def _parse(summ: Optional[dict]) -> float:
    return float("nan") if not summ else float(summ.get("parse_ok_rate", float("nan")))


# --------------------------------------------------------------------------- #
# dose curves + coherent-dose selection                                       #
# --------------------------------------------------------------------------- #
def dose_curves(summ: Dict[str, dict]) -> dict:
    """Per (s,t): parse_ok(alpha) and behaviour(alpha) over ALPHAS, plus the
    per-target no-steer baseline and the per-(t,alpha) random-null curve."""
    cells: Dict[str, Dict[str, dict]] = {}
    for s in TASKS:
        cells[s] = {}
        for t in TASKS:
            pa = {a: _parse(summ.get(cell_id(s, t, a))) for a in ALPHAS}
            be = {a: _metric(summ.get(cell_id(s, t, a))) for a in ALPHAS}
            cells[s][t] = {"parse": pa, "behavior": be}
    baseline = {t: {"parse": _parse(summ.get(base_id(t))),
                    "behavior": _metric(summ.get(base_id(t)))} for t in TASKS}
    nulls = {t: {"parse": {a: _parse(summ.get(null_id(t, a))) for a in ALPHAS},
                 "behavior": {a: _metric(summ.get(null_id(t, a))) for a in ALPHAS}}
             for t in TASKS}
    return {"cells": cells, "baseline": baseline, "nulls": nulls}


def coherent_dose(curves: dict, tol: float = COH_TOL) -> Dict[str, Optional[float]]:
    """Per target t: the LARGEST alpha whose diagonal (t->t) parse_ok is within
    `tol` of t's no-steer baseline parse. None if no dose qualifies (target too
    fragile at every dose)."""
    out: Dict[str, Optional[float]] = {}
    for t in TASKS:
        base_p = curves["baseline"][t]["parse"]
        diag = curves["cells"][t][t]["parse"]
        ok = [a for a in ALPHAS
              if np.isfinite(diag[a]) and np.isfinite(base_p)
              and diag[a] >= base_p - tol]
        out[t] = max(ok) if ok else None
    return out


def transfer_at_coherent_dose(curves: dict, dose: Dict[str, Optional[float]]) -> dict:
    """At each target's coherent dose, the null-subtracted behaviour shift per
    source, and the transfer fraction vs the within-task diagonal shift.

      spec_shift(s->t)   = (cell - baseline) - (null - baseline) = cell - null
      transfer_fraction  = spec_shift(s->t) / spec_shift(t->t)
    Cells whose own parse at that dose is too sparse (< min_parse) are flagged
    coherence_limited and their behaviour value is kept but marked low-trust."""
    res = {}
    for t in TASKS:
        a = dose[t]
        col = {"dose": a, "cells": {}}
        if a is None:
            res[t] = col
            continue
        base_b = curves["baseline"][t]["behavior"]
        null_b = curves["nulls"][t]["behavior"].get(a, float("nan"))
        diag_spec = (curves["cells"][t][t]["behavior"].get(a, float("nan")) - null_b)
        for s in TASKS:
            cell_b = curves["cells"][s][t]["behavior"].get(a, float("nan"))
            cell_p = curves["cells"][s][t]["parse"].get(a, float("nan"))
            spec = cell_b - null_b
            frac = spec / diag_spec if np.isfinite(diag_spec) and abs(diag_spec) > 1e-9 else float("nan")
            col["cells"][s] = {
                "behavior": cell_b, "parse": cell_p,
                "raw_shift": cell_b - base_b if np.isfinite(base_b) else float("nan"),
                "spec_shift": spec, "transfer_fraction": frac,
                "coherence_limited": bool(not np.isfinite(cell_p) or cell_p < 0.30),
            }
        col["baseline_behavior"] = base_b
        col["null_behavior"] = null_b
        col["diag_spec_shift"] = diag_spec
        res[t] = col
    return res


def cosine_vs_transfer(transfer: dict, cos: np.ndarray) -> dict:
    """Across the OFF-diagonal cells measured at a coherent dose with adequate
    parse, correlate the source-axis cosine with the specific behaviour shift —
    the Stage-B sm-column prediction generalised. Returns the paired points + a
    Pearson r (None if < 3 usable points)."""
    idx = {t: i for i, t in enumerate(TASKS)}
    pts = []
    for t in TASKS:
        col = transfer.get(t, {})
        for s in TASKS:
            if s == t:
                continue
            c = col.get("cells", {}).get(s)
            if not c or c["coherence_limited"] or not np.isfinite(c["spec_shift"]):
                continue
            pts.append({"source": s, "target": t,
                        "cosine": float(cos[idx[s], idx[t]]),
                        "spec_shift": float(c["spec_shift"])})
    r = None
    if len(pts) >= 3:
        xs_ = np.array([p["cosine"] for p in pts])
        ys_ = np.array([p["spec_shift"] for p in pts])
        if xs_.std() > 1e-9 and ys_.std() > 1e-9:
            r = float(np.corrcoef(xs_, ys_)[0, 1])
    return {"points": pts, "pearson_r": r, "n": len(pts)}


# --------------------------------------------------------------------------- #
# figure                                                                       #
# --------------------------------------------------------------------------- #
def make_figure(curves: dict, transfer: dict, cvt: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 4)
    al = list(ALPHAS)
    # 3x3 small multiples: parse(alpha) [solid] + behaviour(alpha) [dashed,twin]
    for ti, t in enumerate(TASKS):
        for si, s in enumerate(TASKS):
            ax = fig.add_subplot(gs[ti, si])
            c = curves["cells"][s][t]
            ax.plot(al, [c["parse"][a] for a in al], "-o", color="C0", ms=3,
                    label="parse")
            ax.axhline(curves["baseline"][t]["parse"], color="C0", lw=0.6, ls=":")
            axb = ax.twinx()
            axb.plot(al, [c["behavior"][a] for a in al], "--s", color="C3", ms=3,
                     label="risky")
            axb.axhline(curves["baseline"][t]["behavior"], color="C3", lw=0.6, ls=":")
            axb.set_ylim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{s}->{t}" + (" (diag)" if s == t else ""), fontsize=8)
            if ti == 2:
                ax.set_xlabel("alpha")
            if si == 0:
                ax.set_ylabel("parse", color="C0")
            if si == 2:
                axb.set_ylabel("risky", color="C3")
    # right column: cosine vs spec_shift scatter
    axc = fig.add_subplot(gs[:, 3])
    for p in cvt["points"]:
        axc.scatter(p["cosine"], p["spec_shift"], s=40)
        axc.annotate(f"{p['source']}->{p['target']}",
                     (p["cosine"], p["spec_shift"]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    axc.axhline(0, color="0.6", lw=0.6)
    axc.axvline(0, color="0.6", lw=0.6)
    axc.set_xlabel("source-axis cosine(s, t)")
    axc.set_ylabel("specific behaviour shift (risky)")
    r = cvt["pearson_r"]
    axc.set_title(f"off-diagonal transfer vs cosine\n"
                  f"r={r:.2f} (n={cvt['n']})" if r is not None
                  else f"off-diagonal transfer vs cosine\n(n={cvt['n']}, r n/a)")
    fig.suptitle("Stage C: cross-task BK steering dose sweep "
                 "(parse + behaviour vs alpha; transfer vs geometry)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def analyze(dose_dir: Path = DOSE_DIR, base_dir: Path = BASE_DIR,
            asset_dir: Path = ASSET_DIR, out_json: Path = OUT_JSON,
            out_png: Path = OUT_PNG) -> dict:
    global DOSE_DIR, BASE_DIR
    DOSE_DIR, BASE_DIR = dose_dir, base_dir
    summ = load_all_summaries()
    curves = dose_curves(summ)
    dose = coherent_dose(curves)
    transfer = transfer_at_coherent_dose(curves, dose)
    try:
        axes = xs.load_source_axes(asset_dir)
        cos = xs.cosine_matrix(axes)
    except Exception:
        cos = np.full((3, 3), np.nan)
    cvt = cosine_vs_transfer(transfer, cos)
    result = {
        "tasks": list(TASKS), "alphas": list(ALPHAS), "metric": METRIC,
        "coh_tol": COH_TOL, "n_arms_found": len(summ),
        "coherent_dose": dose, "curves": curves, "transfer": transfer,
        "source_axis_cosine": cos.tolist(), "cosine_vs_transfer": cvt,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1))
    try:
        make_figure(curves, transfer, cvt, out_png)
    except Exception as e:  # plotting is non-essential; never lose the json
        print(f"[xtask_dose] figure skipped: {e}")
    print(f"[xtask_dose] {len(summ)} arms -> {out_json}")
    return result


def _selftest() -> None:
    """Synthetic check of the dose pipeline with no node data."""
    # baseline parse 0.9 / risky 0.20 for every target. diagonal stays coherent
    # at every alpha; off-diagonals: ic->sm coherent (parse 0.9) with a +cosine
    # raising risky, sm->mw collapses (parse 0.0) above alpha 1.
    def S(parse, risky):
        return {"parse_ok_rate": parse, "mean_risky": risky}
    summ = {}
    for t in TASKS:
        summ[base_id(t)] = S(0.9, 0.20)
        for a in ALPHAS:
            summ[null_id(t, a)] = S(0.9, 0.25)
            for s in TASKS:
                if s == t:
                    summ[cell_id(s, t, a)] = S(0.9, 0.20 + 0.4)          # diag: +0.4
                elif (s, t) == ("ic", "sm"):
                    summ[cell_id(s, t, a)] = S(0.9, 0.20 + 0.3)          # transfer +0.3
                else:
                    summ[cell_id(s, t, a)] = S(0.0 if a > 1 else 0.5, 0.20)
    global DOSE_DIR, BASE_DIR
    cur = dose_curves(summ)
    dose = coherent_dose(cur)
    assert dose["sm"] == 4.0, dose                       # sm diagonal coherent everywhere
    tr = transfer_at_coherent_dose(cur, dose)
    # ic->sm spec_shift = cell(0.50) - null(0.25) = 0.25; diag spec = 0.60-0.25=0.35
    c = tr["sm"]["cells"]["ic"]
    assert abs(c["spec_shift"] - 0.25) < 1e-9, c
    assert abs(c["transfer_fraction"] - 0.25 / 0.35) < 1e-9, c
    assert tr["sm"]["cells"]["sm"]["coherence_limited"] is False
    cos = np.array([[1, -0.3, 0.0], [-0.3, 1, 0.5], [0.0, 0.5, 1]])
    cvt = cosine_vs_transfer(tr, cos)
    assert cvt["n"] >= 1
    print("xtask_dose_stats selftest: ALL PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose-dir", default=str(DOSE_DIR))
    ap.add_argument("--base-dir", default=str(BASE_DIR))
    ap.add_argument("--asset-dir", default=str(ASSET_DIR))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-png", default=str(OUT_PNG))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(Path(args.dose_dir), Path(args.base_dir), Path(args.asset_dir),
            Path(args.out_json), Path(args.out_png))


if __name__ == "__main__":
    main()
