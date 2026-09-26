"""Component D — 3x3 cross-task BK steering TRANSFER analysis (xtask namespace).

Reads the xtask rollout jsonls (one per arm, written by ArmCheckpoint as
{results_dir}/{arm_id}.jsonl), reduces each arm to a behaviour summary
(parse_ok rate, mean risky/spin-rate, mean bet_ratio, n on the parse_ok rows),
and assembles the 3x3 CAUSAL transfer matrix:

    rows  = SINGLE-TASK BK source axis  (sm | ic | mw)
    cols  = APPLY target task           (sm | ic | mw)
    cell  = behaviour SHIFT of (source axis steered into target) vs that
            target's NO-STEER baseline population.

BASELINE CHOICE (documented, load-bearing):
    The no-steer reference per target is the target's own `anchor_minus` arm —
    the SAME decision-state pool, state_offset, seed_base and model as the steer
    arms, run with NO steering hook installed (runner.py: mode != steer means no
    MultiLayerSteerer). This is the most faithful baseline because it differs
    from a steer arm by exactly one factor (the hook), holding population, seeds
    and prompt construction byte-identical. It is strictly preferred over the
    W2 +G/-G twin anchors (which encode a prompt-level contrast, not the steer
    population) for measuring a steer-induced shift. Per-arm baselines are keyed
    by target task; we fall back to a "no_steer"/"anchor" arm id substring match
    so the analysis is robust to the final arm-id spelling in arms_xtask.yaml.

NORMALISATION (transfer fraction) and SPECIFICITY (null subtraction):
    raw_shift[src][tgt]  = metric(steer src->tgt)  - metric(baseline tgt)
    null_shift[tgt]      = mean over the 8 per-target random-null seeds of
                           metric(rnd->tgt) - metric(baseline tgt)
    spec_shift[src][tgt] = raw_shift - null_shift[tgt]    (specificity: remove
                           the same-Δnorm random-direction band for that target)
    transfer_fraction[src][tgt] = spec_shift[src][tgt] / spec_shift[tgt][tgt]
                           (off-diagonal normalised by the within-task diagonal
                           of the SAME target; diagonal == 1.0 by construction).

The diagonal (src==tgt) is the within-task causal effect (upper bound); the
transfer fraction reports how much of that a foreign-task source axis reproduces
on the same target population at the same Δnorm.

Outputs (results/xtask/):
    xtask_transfer.json  — full numeric summary (per-arm + matrices + cosines).
    xtask_transfer.png   — panel 1: transfer-fraction heatmap;
                           panel 2: READ-side cross-task transfer AUC heatmap
                           (from the spine crosstask42 read profiles) so the
                           causal-transfer and read-transfer geometries sit
                           side by side.

This module is read-only over rollout data; it never re-runs a model. It is
isolated in the xtask namespace and imports nothing from the runner.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

TASKS = ("sm", "ic", "mw")
# Per-target random-null seed count (matches the 8-seed w3rnd null band width).
N_NULL_SEEDS = 8

_HERE = Path(__file__).resolve().parent           # .../multilayer_causal/src
_MLC = _HERE.parent                               # .../multilayer_causal
DEFAULT_RESULTS_DIR = _MLC / "results" / "xtask"
DEFAULT_SPINE_DIR = _MLC / "results" / "spine"
DEFAULT_OUT_JSON = DEFAULT_RESULTS_DIR / "xtask_transfer.json"
DEFAULT_OUT_PNG = DEFAULT_RESULTS_DIR / "xtask_transfer.png"


# ---------------------------------------------------------------------------
# jsonl reduction
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_arm(rows: List[dict]) -> dict:
    """Reduce one arm's rollout rows to a behaviour summary.

    The behaviour-shift metric is the binary risk flag (CONTRACT3): mw/ic use
    `risky` (spin / risky-choice), sm has no `risky` field so we derive it from
    action=='bet'. bet_ratio is averaged as a secondary continuous metric.
    All behaviour means are taken over parse_ok==True rows only (R5 contract);
    parse_ok rate itself is over ALL rows.
    """
    n_total = len(rows)
    n_ok = 0
    risky_vals, bet_ratio_vals = [], []
    for r in rows:
        ok = bool(r.get("parse_ok"))
        if not ok:
            continue
        n_ok += 1
        # risky: explicit field (ic/mw) or derived from sm action.
        if "risky" in r:
            risky = bool(r["risky"])
        elif "action" in r:
            risky = (r.get("action") == "bet")
        else:
            risky = False
        risky_vals.append(1.0 if risky else 0.0)
        if "bet_ratio" in r and r["bet_ratio"] is not None:
            bet_ratio_vals.append(float(r["bet_ratio"]))
    return {
        "n_rows": n_total,
        "n_parse_ok": n_ok,
        "parse_ok_rate": (n_ok / n_total) if n_total else 0.0,
        "mean_risky": float(np.mean(risky_vals)) if risky_vals else float("nan"),
        "stop_rate": float(1.0 - np.mean(risky_vals)) if risky_vals else float("nan"),
        "mean_bet_ratio": float(np.mean(bet_ratio_vals)) if bet_ratio_vals else float("nan"),
    }


def load_arm_summaries(results_dir: Path) -> Dict[str, dict]:
    """Map arm_id -> behaviour summary for every *.jsonl in results_dir."""
    summaries = {}
    for jl in sorted(Path(results_dir).glob("*.jsonl")):
        arm_id = jl.stem
        rows = _read_jsonl(jl)
        if rows:
            summaries[arm_id] = summarize_arm(rows)
    return summaries


# ---------------------------------------------------------------------------
# arm-id resolution (robust to the final arms_xtask.yaml spelling)
# ---------------------------------------------------------------------------

def _find_arm(summaries: Dict[str, dict], *needles: str) -> Optional[str]:
    """Return the single arm id containing ALL lowercased needles, else None.

    Prefers an exact-token match (needles joined by '2' or '_') but falls back
    to a substring-AND match so we tolerate either xt_sm2sm or xt_sm_to_sm style
    ids without hard-coding the convention.
    """
    needles = tuple(s.lower() for s in needles)
    hits = [a for a in summaries if all(n in a.lower() for n in needles)]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    # Disambiguate: shortest id (most specific) wins; deterministic on ties.
    return sorted(hits, key=lambda a: (len(a), a))[0]


def resolve_cells(summaries: Dict[str, dict],
                  metric: str = "mean_risky") -> dict:
    """Resolve baseline / transfer-cell / null arm ids per target and read metric.

    Returns a structured dict of metric values (np.nan where an arm is missing,
    so the math is exercised even on a partial run).
    """
    def val(arm_id: Optional[str]) -> float:
        if arm_id is None or arm_id not in summaries:
            return float("nan")
        return float(summaries[arm_id][metric])

    baseline_ids, baseline = {}, {}
    for tgt in TASKS:
        # Baseline = target's no-steer arm: prefer an explicit xtask baseline,
        # else any anchor_minus / "no_steer" / "anchor" arm for that target.
        aid = (_find_arm(summaries, "xt", tgt, "base")
               or _find_arm(summaries, "xt", tgt, "anchor")
               or _find_arm(summaries, tgt, "anchor"))
        baseline_ids[tgt] = aid
        baseline[tgt] = val(aid)

    cell_ids = {s: {} for s in TASKS}
    cell = {s: {} for s in TASKS}
    for src in TASKS:
        for tgt in TASKS:
            # Resolve by the canonical id first; the directed "{src}_to_{tgt}"
            # connector is unique per cell, so even the fuzzy fallback can no
            # longer collide a diagonal (src==tgt) with an off-diagonal arm.
            exact = f"xtask_{src}_to_{tgt}"
            aid = (exact if exact in summaries
                   else _find_arm(summaries, "xt", f"{src}_to_{tgt}")
                   or _find_arm(summaries, "xt", f"{src}2{tgt}"))
            cell_ids[src][tgt] = aid
            cell[src][tgt] = val(aid)

    null_ids = {t: [] for t in TASKS}
    null = {}
    for tgt in TASKS:
        vals = []
        for s in range(N_NULL_SEEDS):
            aid = (_find_arm(summaries, "xt", "rnd", tgt, f"s{s}")
                   or _find_arm(summaries, "rnd", tgt, f"s{s}"))
            null_ids[tgt].append(aid)
            v = val(aid)
            if not np.isnan(v):
                vals.append(v)
        null[tgt] = float(np.mean(vals)) if vals else float("nan")

    return {
        "metric": metric,
        "baseline": baseline, "baseline_ids": baseline_ids,
        "cell": cell, "cell_ids": cell_ids,
        "null": null, "null_ids": null_ids,
    }


# ---------------------------------------------------------------------------
# transfer matrix math (pure; unit-tested in __main__)
# ---------------------------------------------------------------------------

def build_transfer_matrix(cell: Dict[str, Dict[str, float]],
                          baseline: Dict[str, float],
                          null: Dict[str, float]) -> dict:
    """Assemble raw / specificity / transfer-fraction 3x3 matrices.

    cell[src][tgt]  = metric of (src axis steered into tgt).
    baseline[tgt]   = no-steer metric on tgt population.
    null[tgt]       = mean random-null metric on tgt population.

    raw_shift   = cell - baseline[tgt]
    spec_shift  = raw_shift - (null[tgt] - baseline[tgt])   # subtract null band
    transfer_fraction[src][tgt] = spec_shift[src][tgt] / spec_shift[tgt][tgt]

    Returns dict of 3x3 numpy arrays (rows=src, cols=tgt) + the diagonal.
    """
    n = len(TASKS)
    raw = np.full((n, n), np.nan)
    spec = np.full((n, n), np.nan)
    frac = np.full((n, n), np.nan)

    null_shift = np.array([null[t] - baseline[t] for t in TASKS])  # per-target
    for i, src in enumerate(TASKS):
        for j, tgt in enumerate(TASKS):
            raw[i, j] = cell[src][tgt] - baseline[tgt]
            spec[i, j] = raw[i, j] - null_shift[j]

    diag = np.array([spec[j, j] for j in range(n)])  # within-task spec shift
    for i in range(n):
        for j in range(n):
            d = diag[j]
            # Divide-by-(near)-zero diagonal -> undefined transfer fraction.
            frac[i, j] = spec[i, j] / d if abs(d) > 1e-9 else np.nan

    return {
        "raw_shift": raw,
        "null_shift": null_shift,
        "spec_shift": spec,
        "transfer_fraction": frac,
        "diagonal_spec_shift": diag,
    }


def cosine_matrix(axes: Dict[str, np.ndarray]) -> np.ndarray:
    """Pairwise cosine between the 3 single-task source axis vectors (3x3)."""
    n = len(TASKS)
    cos = np.full((n, n), np.nan)
    for i, a in enumerate(TASKS):
        for j, b in enumerate(TASKS):
            va, vb = axes.get(a), axes.get(b)
            if va is None or vb is None:
                continue
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na > 0 and nb > 0:
                cos[i, j] = float(np.dot(va, vb) / (na * nb))
    return cos


def load_source_axes(asset_dir: Path, layer_row: int = 22) -> Dict[str, np.ndarray]:
    """Load the single-task BK axis (one layer row) for each source task.

    The xtask assets replicate the L22 axis across all 42 layer rows, so any row
    carries the same unit direction; row 22 is the native L22 axis row.
    """
    asset_dir = Path(asset_dir)
    axes = {}
    for t in TASKS:
        # Tolerate the xtask single-task spelling (directions_bk_single_{t}.npz,
        # what xtask_axes actually writes) plus the bk1/bk legacy names.
        cand = [asset_dir / f"directions_bk_single_{t}.npz",
                asset_dir / f"directions_bk1_{t}.npz",
                asset_dir / f"directions_bk_{t}.npz"]
        npz = next((p for p in cand if p.exists()), None)
        if npz is None:
            continue
        z = np.load(npz)
        d = z["directions"]
        row = layer_row if layer_row < d.shape[0] else 0
        axes[t] = np.asarray(d[row], dtype=np.float64)
    return axes


# ---------------------------------------------------------------------------
# read-side comparison (spine crosstask42 profiles)
# ---------------------------------------------------------------------------

def load_read_transfer_auc(spine_dir: Path, model: str = "gemma",
                           layer: int = 22, variant: str = "baseline") -> np.ndarray:
    """3x3 read-side transfer-AUC matrix from read_profile_{model}_crosstask42_L{L}.

    The profile stores directed pairs (sm_to_ic, ...). Diagonal (within-task) is
    not measured on the read side, so it is left NaN. rows=src, cols=tgt.
    """
    path = Path(spine_dir) / f"read_profile_{model}_crosstask42_L{layer}.json"
    n = len(TASKS)
    mat = np.full((n, n), np.nan)
    if not path.exists():
        return mat
    data = json.load(open(path))
    auc = data["result"]["variants"][variant]["transfer_auc"]
    for i, src in enumerate(TASKS):
        for j, tgt in enumerate(TASKS):
            if src == tgt:
                continue
            mat[i, j] = float(auc.get(f"{src}_to_{tgt}", np.nan))
    return mat


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _heatmap(ax, mat, title, fmt="{:.2f}", cmap="RdBu_r",
             vmin=None, vmax=None, center=None):
    import numpy as _np
    m = _np.asarray(mat, dtype=float)
    if center is not None and vmin is None and vmax is None:
        span = _np.nanmax(_np.abs(m - center)) if _np.isfinite(m).any() else 1.0
        vmin, vmax = center - span, center + span
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(TASKS)), [t.upper() for t in TASKS])
    ax.set_yticks(range(len(TASKS)), [t.upper() for t in TASKS])
    ax.set_xlabel("apply target (column)")
    ax.set_ylabel("source axis (row)")
    ax.set_title(title)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            txt = "—" if not _np.isfinite(v) else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    return im


def make_figure(transfer_fraction, read_auc, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = _heatmap(axes[0], transfer_fraction,
                   "CAUSAL transfer fraction\n(offdiag / diagonal, null-subtracted)",
                   center=0.0)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = _heatmap(axes[1], read_auc,
                   "READ-side cross-task transfer AUC\n(spine crosstask42 L22)",
                   center=0.5, cmap="RdBu_r")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("3x3 cross-task BK steering: causal vs read transfer", y=1.02)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _mat_to_lists(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = np.where(np.isfinite(v), v, None).tolist()
        else:
            out[k] = v
    return out


def analyze(results_dir: Path = DEFAULT_RESULTS_DIR,
            asset_dir: Path = _MLC / "assets" / "xtask",
            spine_dir: Path = DEFAULT_SPINE_DIR,
            out_json: Path = DEFAULT_OUT_JSON,
            out_png: Path = DEFAULT_OUT_PNG,
            metric: str = "mean_risky",
            model: str = "gemma", layer: int = 22) -> dict:
    """Full Component D pipeline over the xtask rollout jsonls."""
    results_dir, asset_dir = Path(results_dir), Path(asset_dir)
    spine_dir, out_json, out_png = Path(spine_dir), Path(out_json), Path(out_png)
    summaries = load_arm_summaries(results_dir)
    resolved = resolve_cells(summaries, metric=metric)
    tm = build_transfer_matrix(resolved["cell"], resolved["baseline"],
                               resolved["null"])
    axes = load_source_axes(asset_dir, layer_row=layer)
    cos = cosine_matrix(axes)
    read_auc = load_read_transfer_auc(spine_dir, model=model, layer=layer)

    result = {
        "tasks": list(TASKS),
        "metric": metric,
        "baseline_choice": (
            "per-target no-steer anchor_minus arm (same pool/offset/seed_base "
            "as the steer arms, hook absent); off-diagonals null-subtracted with "
            "the per-target 8-seed random-direction band; transfer fraction = "
            "spec_shift / within-task diagonal spec_shift."),
        "n_arms_found": len(summaries),
        "arm_summaries": summaries,
        "resolved_ids": {
            "baseline": resolved["baseline_ids"],
            "cell": resolved["cell_ids"],
            "null": resolved["null_ids"],
        },
        "values": {
            "baseline": resolved["baseline"],
            "cell": resolved["cell"],
            "null": resolved["null"],
        },
        "transfer": _mat_to_lists(tm),
        "source_axis_cosine": np.where(np.isfinite(cos), cos, None).tolist(),
        "read_transfer_auc": np.where(np.isfinite(read_auc), read_auc, None).tolist(),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    try:
        make_figure(tm["transfer_fraction"], read_auc, out_png)
        result["png"] = str(out_png)
    except Exception as e:  # plotting must never block the numeric artifact
        print(f"[xtask_stats] figure skipped: {type(e).__name__}: {e}", flush=True)
    return result


# ---------------------------------------------------------------------------
# self-test: synthetic-data unit test of the matrix / normalisation math
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Synthetic-data unit test of build_transfer_matrix + cosine + read load."""
    # Construct a controlled scenario with known answers.
    #   baseline metric per target = 0.20 (no-steer risky rate).
    #   null shifts the metric by +0.05 per target (random-direction band).
    #   diagonal (within-task) steer raises risky by +0.40 over baseline.
    #   off-diagonal sm->ic reproduces HALF the ic diagonal; ic->sm reproduces a
    #   QUARTER; everything else = exactly the null (zero specific transfer).
    base = {"sm": 0.20, "ic": 0.20, "mw": 0.20}
    null = {"sm": 0.25, "ic": 0.25, "mw": 0.25}        # +0.05 null shift
    cell = {s: {t: null[t] for t in TASKS} for s in TASKS}  # default = null band
    for t in TASKS:
        cell[t][t] = base[t] + 0.40                    # diagonal: +0.40 raw
    # Specific (null-subtracted) diagonal magnitude = 0.40 - 0.05 = 0.35.
    cell["sm"]["ic"] = base["ic"] + 0.05 + 0.35 * 0.50  # half of ic diagonal
    cell["ic"]["sm"] = base["sm"] + 0.05 + 0.35 * 0.25  # quarter of sm diagonal

    tm = build_transfer_matrix(cell, base, null)
    frac = tm["transfer_fraction"]
    spec = tm["spec_shift"]
    idx = {t: i for i, t in enumerate(TASKS)}

    # Diagonal transfer fraction must be exactly 1.0.
    for t in TASKS:
        assert abs(frac[idx[t], idx[t]] - 1.0) < 1e-9, f"diag {t} != 1.0"
    # Specific diagonal shift recovered as 0.35.
    for t in TASKS:
        assert abs(spec[idx[t], idx[t]] - 0.35) < 1e-9, f"spec diag {t} != 0.35"
    # Programmed off-diagonal fractions.
    assert abs(frac[idx["sm"], idx["ic"]] - 0.50) < 1e-9, "sm->ic frac != 0.50"
    assert abs(frac[idx["ic"], idx["sm"]] - 0.25) < 1e-9, "ic->sm frac != 0.25"
    # Null-only cells -> zero specific transfer -> zero fraction.
    assert abs(frac[idx["mw"], idx["ic"]] - 0.0) < 1e-9, "null cell frac != 0"
    assert abs(frac[idx["sm"], idx["mw"]] - 0.0) < 1e-9, "null cell frac != 0"
    # null_shift recovered per target.
    assert np.allclose(tm["null_shift"], 0.05), "null_shift != 0.05"

    # Divide-by-zero diagonal -> NaN fraction (no crash).
    base2 = {t: 0.3 for t in TASKS}
    null2 = {t: 0.3 for t in TASKS}                    # null == baseline
    cell2 = {s: {t: 0.3 for t in TASKS} for s in TASKS}  # zero diagonal shift
    tm2 = build_transfer_matrix(cell2, base2, null2)
    assert np.isnan(tm2["transfer_fraction"]).all(), "zero diag should be NaN"

    # summarize_arm: parse_ok gating + sm-derived risky + bet_ratio mean.
    rows = [
        {"parse_ok": True, "risky": True, "bet_ratio": 0.30},
        {"parse_ok": True, "risky": False, "bet_ratio": 0.0},
        {"parse_ok": False, "risky": True, "bet_ratio": 0.99},   # dropped
        {"parse_ok": True, "action": "bet", "bet_ratio": 0.10},  # sm-style risky
        {"parse_ok": True, "action": "stop", "bet_ratio": 0.0},
    ]
    s = summarize_arm(rows)
    assert s["n_rows"] == 5 and s["n_parse_ok"] == 4, "parse_ok gating wrong"
    assert abs(s["parse_ok_rate"] - 0.8) < 1e-9, "parse_ok_rate wrong"
    assert abs(s["mean_risky"] - 0.5) < 1e-9, "mean_risky wrong"      # 2 of 4
    assert abs(s["stop_rate"] - 0.5) < 1e-9, "stop_rate wrong"
    assert abs(s["mean_bet_ratio"] - 0.10) < 1e-9, "mean_bet_ratio wrong"  # (.3+0+.1+0)/4

    # cosine_matrix: orthogonal + identical axes.
    axes = {"sm": np.array([1.0, 0.0, 0.0]),
            "ic": np.array([0.0, 2.0, 0.0]),
            "mw": np.array([1.0, 0.0, 0.0])}
    cm = cosine_matrix(axes)
    assert abs(cm[0, 0] - 1.0) < 1e-9 and abs(cm[0, 1]) < 1e-9, "cosine wrong"
    assert abs(cm[idx["sm"], idx["mw"]] - 1.0) < 1e-9, "identical axis cos != 1"

    print("xtask_stats self-test: ALL PASSED")


def main():
    import argparse
    p = argparse.ArgumentParser(description="3x3 cross-task BK transfer analysis")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--asset-dir", default=str(_MLC / "assets" / "xtask"))
    p.add_argument("--spine-dir", default=str(DEFAULT_SPINE_DIR))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-png", default=str(DEFAULT_OUT_PNG))
    p.add_argument("--metric", default="mean_risky",
                   choices=["mean_risky", "mean_bet_ratio", "stop_rate"])
    p.add_argument("--model", default="gemma")
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--selftest", action="store_true",
                   help="run the synthetic-data math unit test and exit")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return
    if not Path(args.results_dir).exists() or not list(Path(args.results_dir).glob("*.jsonl")):
        print(f"[xtask_stats] no rollout jsonls in {args.results_dir}; "
              f"running self-test instead.", flush=True)
        _selftest()
        return
    res = analyze(results_dir=Path(args.results_dir),
                  asset_dir=Path(args.asset_dir),
                  spine_dir=Path(args.spine_dir),
                  out_json=Path(args.out_json),
                  out_png=Path(args.out_png),
                  metric=args.metric, model=args.model, layer=args.layer)
    print(f"[xtask_stats] {res['n_arms_found']} arms -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
