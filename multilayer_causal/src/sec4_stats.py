"""§4 causal-wave Phase-1 analysis: dose / sign / parse / null / cos / AUC.

Reads the per-arm rollout jsonls produced by the runner (results_dir) and the
axis assets (assets_dir), and scores the pre-registered hypotheses:

  H1 (writable window) behavioural axis moves betting MONOTONICALLY with alpha,
     REVERSES sign, stays coherent (parse >= 0.8), and clears the random-null
     band.
  H2 (monitor != controller) the readout axis is INERT at matched sigma-units
     even though it is the better decoder (report both decoding AUCs).
  H3 (not a confound) the confound axis reproduces at most a fraction.

Per axis it returns {recovery_by_dose, spearman, sign_ok, parse_ok_by_dose,
above_null, monotone}; overall it returns a verdict in {WRITE_CONFIRMED,
READOUT_INERT, NULL, CONFOUNDED}, the null band, cos_read_write, and the
decoding-AUC table. Parse gate: dose cells with parse rate < 0.8 are dropped
before any behaviour is read. Manipulation-check assertion: mean manip_proj
(vector_log.proj) must be finite wherever it is logged.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_MLC = Path(__file__).resolve().parents[1]
RESULTS_DIR = _MLC / "results" / "sec4_p0"
ASSETS_DIR = _MLC / "assets" / "sec4"
OUT_JSON = RESULTS_DIR / "sec4_p0_analysis.json"
OUT_PNG = RESULTS_DIR / "sec4_p0_dose.png"

PARSE_GATE = 0.8
SPEARMAN_MIN = 0.7
AXES = ("behavioural", "readout", "confound")


# ------------------------------------------------------------ io

def _rows(path: Path) -> List[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _bet(r: dict) -> Optional[float]:
    """Behaviour metric per record: SM/MW bet_ratio, IC risky as 0/1."""
    if r.get("bet_ratio") is not None:
        return float(r["bet_ratio"])
    if "risky" in r:
        return 1.0 if r["risky"] else 0.0
    return None


def _manip_proj(r: dict):
    """Manipulation check: vector_log.proj (key-path fix) or a top-level
    manip_proj fallback. None when the arm did not log it."""
    v = r.get("vector_log", {})
    if isinstance(v, dict) and v.get("proj") is not None:
        return float(v["proj"])
    if r.get("manip_proj") is not None:
        return float(r["manip_proj"])
    return None


def _axis_of(arm_id: str) -> str:
    """Axis (or role) an arm belongs to, from its id.
    sec4_<axis>_a<dose> | sec4_null_<k> | sec4_baseline | sec4_cum_<...>."""
    if arm_id.startswith("sec4_null"):
        return "null"
    if arm_id.startswith("sec4_baseline"):
        return "baseline"
    if arm_id.startswith("sec4_cum"):
        return "cum"
    parts = arm_id.split("_")
    return parts[1] if len(parts) > 1 else arm_id


def _dose_of(rows: List[dict], arm_id: str) -> Optional[float]:
    """Steering alpha for an arm: record 'alpha' first, else parse the id."""
    for r in rows:
        if r.get("alpha") is not None:
            return float(r["alpha"])
    m = re.search(r"_a(m|p)?(\d+)$", arm_id)
    if m:
        return -float(m.group(2)) if m.group(1) == "m" else float(m.group(2))
    return None


# ------------------------------------------------------- arm summaries

def _arm_summary(rows: List[dict]) -> dict:
    n = len(rows)
    parse_ok = [r for r in rows if r.get("parse_ok")]
    parse_rate = len(parse_ok) / n if n else float("nan")
    bets = [b for r in parse_ok if (b := _bet(r)) is not None]
    projs = [p for r in rows if (p := _manip_proj(r)) is not None]
    return {
        "n": n, "parse_rate": parse_rate,
        "mean_bet": float(np.mean(bets)) if bets else float("nan"),
        "n_parsed_bet": len(bets),
        "manip_proj": float(np.mean(projs)) if projs else float("nan"),
        "bets": bets,
    }


def _anchor_means(assets_dir: Path, results_dir: Path):
    """(-G, +G) anchor mean bet_ratio for the recovery metric.

    Preference order: a local anchors.json {minus, plus} in assets_dir or
    results_dir, else the frozen W2 +/-G anchors pulled from HF (like
    xtaskd_stats). Returns (minus, plus) or (nan, nan) when unavailable — the
    verdict never depends on recovery, only on sign/monotone/null."""
    for d in (assets_dir, results_dir):
        p = Path(d) / "anchors.json"
        if p.exists():
            j = json.loads(p.read_text())
            return float(j["minus"]), float(j["plus"])
    try:  # optional HF fallback (skipped offline / in unit tests)
        from huggingface_hub import hf_hub_download
        out = {}
        for tag, arm in (("minus", "w2/w2_anchor_minus"),
                         ("plus", "w2/w2_anchor_plus")):
            fp = hf_hub_download(
                "llm-addiction-research/llm-addiction",
                f"experiments/multilayer_causal/checkpoints/{arm}.jsonl",
                repo_type="dataset")
            out[tag] = _arm_summary(_rows(Path(fp)))["mean_bet"]
        return out["minus"], out["plus"]
    except Exception:
        return float("nan"), float("nan")


# ----------------------------------------------------------- analysis

def analyze(results_dir=RESULTS_DIR, assets_dir=ASSETS_DIR,
            out_json: Optional[Path] = None,
            out_png: Optional[Path] = None) -> dict:
    results_dir = Path(results_dir)
    assets_dir = Path(assets_dir)
    minus, plus = _anchor_means(assets_dir, results_dir)
    gap = plus - minus

    def recovery(m):
        return ((m - minus) / gap
                if np.isfinite(m) and np.isfinite(gap) and abs(gap) > 1e-9
                else float("nan"))

    # gather arms grouped by axis role
    by_axis: Dict[str, Dict[float, dict]] = {a: {} for a in AXES}
    nulls, baseline = [], None
    for fp in sorted(results_dir.glob("sec4_*.jsonl")):
        arm = fp.stem
        rows = _rows(fp)
        summ = _arm_summary(rows)
        # manip-check invariant: a logged projection must be finite.
        if summ["n_parsed_bet"] and any(_manip_proj(r) is not None for r in rows):
            assert np.isfinite(summ["manip_proj"]), \
                f"{arm}: manip_proj is NaN despite logged vector_log.proj"
        role = _axis_of(arm)
        if role == "null":
            nulls.append(summ)
        elif role == "baseline":
            baseline = summ
        elif role in by_axis:
            dose = _dose_of(rows, arm)
            if dose is not None:
                by_axis[role][dose] = summ

    # random-null band: mean +/- 2sd of the null arms' mean bet.
    null_bets = [s["mean_bet"] for s in nulls if np.isfinite(s["mean_bet"])]
    null_mean = float(np.mean(null_bets)) if null_bets else float("nan")
    null_sd = float(np.std(null_bets)) if len(null_bets) > 1 else 0.0
    base_bet = baseline["mean_bet"] if baseline else (
        by_axis["behavioural"].get(0.0, {}).get("mean_bet", float("nan")))
    # effect that must clear the null band = 2sd, floored so a degenerate
    # zero-variance null does not make every axis "significant".
    null_delta = max(2.0 * null_sd, 0.03)

    axes_out = {}
    for axis in AXES:
        cells = by_axis[axis]
        # parse gate: drop dose cells below the coherence floor.
        kept = {d: c for d, c in cells.items() if c["parse_rate"] >= PARSE_GATE}
        doses = sorted(kept)
        recovery_by_dose = {d: recovery(kept[d]["mean_bet"]) for d in doses}
        parse_ok_by_dose = {d: cells[d]["parse_rate"] for d in sorted(cells)}
        # per-trial spearman(alpha, bet) over kept cells (more power than means)
        xs, ys = [], []
        for d in doses:
            for b in kept[d]["bets"]:
                xs.append(d); ys.append(b)
        rho = _spearman(xs, ys)
        sign_ok = bool(np.isfinite(rho) and rho > 0)  # +alpha => more betting
        monotone = bool(np.isfinite(rho) and abs(rho) >= SPEARMAN_MIN and rho > 0)
        # above-null: extreme-dose effect vs baseline clears the null band.
        eff = _extreme_effect(kept, base_bet)
        above_null = bool(np.isfinite(eff) and abs(eff) > null_delta)
        axes_out[axis] = {
            "recovery_by_dose": {str(k): v for k, v in recovery_by_dose.items()},
            "spearman": rho, "sign_ok": sign_ok,
            "parse_ok_by_dose": {str(k): v for k, v in parse_ok_by_dose.items()},
            "above_null": above_null, "monotone": monotone,
            "extreme_effect": eff, "n_doses": len(doses),
        }

    verdict = _verdict(axes_out)
    cos_rw = _cos_read_write(assets_dir)
    dec_auc = _decoding_auc(assets_dir)

    result = {
        "verdict": verdict,
        "axes": axes_out,
        "anchors": {"minus": minus, "plus": plus, "gap": gap},
        "null_band": {"mean": null_mean, "sd": null_sd, "delta": null_delta,
                      "n": len(null_bets)},
        "baseline_bet": base_bet,
        "cos_read_write": cos_rw,
        "decoding_auc": dec_auc,
        # H1/H2 booleans surfaced for the ledger.
        "h1_behavioural_writes": _writes(axes_out["behavioural"]),
        "h2_readout_inert": not _writes(axes_out["readout"]),
    }
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
    if out_png is not None:
        try:
            make_figure(result, by_axis, base_bet, Path(out_png))
        except Exception as e:  # figure is a nicety, never fail analysis on it
            print(f"[sec4] figure skipped: {e}")
    return result


def _spearman(xs, ys):
    if len(set(xs)) < 2 or len(xs) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(xs, ys)[0])
    except Exception:
        return float("nan")


def _extreme_effect(kept, base_bet):
    """Signed effect of the largest-|dose| kept cell relative to baseline."""
    if not kept or not np.isfinite(base_bet):
        return float("nan")
    d = max(kept, key=abs)
    return kept[d]["mean_bet"] - base_bet


def _writes(axis_stat: dict) -> bool:
    return bool(axis_stat["monotone"] and axis_stat["sign_ok"]
               and axis_stat["above_null"])


def _verdict(axes_out: dict) -> str:
    b = _writes(axes_out["behavioural"])
    r = _writes(axes_out["readout"])
    c = _writes(axes_out["confound"])
    if b and not r and not c:
        return "WRITE_CONFIRMED"      # H1 + H2 (readout inert) + H3 (no confound)
    if b and c:
        return "CONFOUNDED"           # H3 fails: balance-signalling explains it
    if not b and not r:
        return "READOUT_INERT"        # nothing writes; readout confirmed inert
    return "NULL"                     # behavioural inert while readout writes, etc.


def _cos_read_write(assets_dir: Path) -> Dict[str, float]:
    """cos(readout, behavioural) at each layer's row from the saved assets,
    keyed by indicator. Reads the stored cos_read_write field, else recomputes
    from the direction rows."""
    out = {}
    for fp in sorted(Path(assets_dir).glob("*_readout.npz")):
        try:
            z = np.load(fp)
            key = fp.stem.replace("_readout", "")
            if "cos_read_write" in z.files:
                out[key] = float(z["cos_read_write"])
                continue
            bfp = Path(assets_dir) / f"{key}_behavioural.npz"
            if bfp.exists():
                r = np.load(fp)["directions"]
                b = np.load(bfp)["directions"]
                out[key] = float(_row_cos(r, b))
        except Exception:
            continue
    return out


def _row_cos(a, b):
    a0 = a[a.any(axis=1)][0] if a.ndim == 2 else a
    b0 = b[b.any(axis=1)][0] if b.ndim == 2 else b
    a0 = a0 / (np.linalg.norm(a0) + 1e-12)
    b0 = b0 / (np.linalg.norm(b0) + 1e-12)
    return float(a0 @ b0)


def _decoding_auc(assets_dir: Path) -> Dict[str, float]:
    """Decoding-AUC table {asset_key: auc} from the saved axis npz (H2: the
    better decoder should still be inert)."""
    out = {}
    for fp in sorted(Path(assets_dir).glob("*.npz")):
        try:
            z = np.load(fp)
            if "auc" in z.files:
                out[fp.stem] = float(z["auc"])
        except Exception:
            continue
    return out


def make_figure(res: dict, by_axis: dict, base_bet: float, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"behavioural": "C0", "readout": "C1", "confound": "C2"}
    for axis in AXES:
        cells = by_axis.get(axis, {})
        doses = sorted(d for d, c in cells.items() if c["parse_rate"] >= PARSE_GATE)
        if not doses:
            continue
        ax.plot(doses, [cells[d]["mean_bet"] for d in doses], "-o",
                color=colors[axis], label=axis)
    nb = res["null_band"]
    if np.isfinite(nb["mean"]):
        ax.axhspan(nb["mean"] - nb["delta"], nb["mean"] + nb["delta"],
                   color="0.85", label="random-null band")
    if np.isfinite(base_bet):
        ax.axhline(base_bet, color="k", lw=0.5, ls=":", label="baseline")
    ax.set_xlabel("steering alpha (sigma-units)")
    ax.set_ylabel("mean bet ratio")
    ax.set_title(f"§4 Phase-1 dose-response — verdict: {res['verdict']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


# ============================================================ WAVE 2
# Common-axis analysis: does ONE shared internal axis drive BOTH the I_BA
# (bet_ratio) and I_EC (extreme) irrationality indicators, above a THICK null
# slope band estimated per random-null direction from its own -3/+3 dose pair
# (fixing Wave-1's single-dose thin null)?

W2_AXES = ("behav_iba", "behav_iec", "shared", "confound")
W2_INDICATORS = ("i_ba", "i_ec")
NULL_SLOPE_FLOOR = 0.005  # min half-band so a zero-variance null is not "clear"


def _ec(r: dict):
    """I_EC value per record: the runner's boolean `extreme` field as 0/1, with
    a bet_ratio>=0.5 fallback when `extreme` was not logged."""
    if r.get("extreme") is not None:
        return 1.0 if r["extreme"] else 0.0
    b = _bet(r)
    return None if b is None else (1.0 if b >= 0.5 else 0.0)


_W2_METRIC = {"i_ba": _bet, "i_ec": _ec}


def _w2_axis_of(arm_id: str) -> Optional[str]:
    """Wave-2 axis role from an arm id, or None if not a Wave-2 dose arm.
    sec4_w2_<axis>_a{m,p}<k> for axis in W2_AXES; nulls / baseline handled apart."""
    if not arm_id.startswith("sec4_w2_"):
        return None
    if arm_id.startswith("sec4_w2_null") or arm_id.startswith("sec4_w2_baseline"):
        return None
    for axis in W2_AXES:
        if arm_id.startswith(f"sec4_w2_{axis}_a"):
            return axis
    return None


def _ols_slope(xs, ys) -> float:
    """OLS slope of ys on xs (dose->indicator). NaN when <3 points or no dose
    spread."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 3 or float(xs.std()) < 1e-12:
        return float("nan")
    return float(np.cov(xs, ys, bias=True)[0, 1] / xs.var())


def _dose_cells(results_dir: Path):
    """Group Wave-2 rollouts: {axis: {dose: summary}}, plus per-null-direction
    {dose: summary} pairs and the baseline summary. Each summary carries both
    indicators' per-trial values so slopes/spearman use trial-level data."""
    by_axis: Dict[str, Dict[float, dict]] = {a: {} for a in W2_AXES}
    nulls: Dict[str, Dict[float, dict]] = {}
    baseline = None
    for fp in sorted(results_dir.glob("sec4_w2_*.jsonl")):
        arm = fp.stem
        rows = _rows(fp)
        parse_ok = [r for r in rows if r.get("parse_ok")]
        parse_rate = len(parse_ok) / len(rows) if rows else float("nan")
        vals = {ind: [v for r in parse_ok if (v := _W2_METRIC[ind](r)) is not None]
                for ind in W2_INDICATORS}
        summ = {"parse_rate": parse_rate, "n": len(rows), "values": vals}
        if arm.startswith("sec4_w2_baseline"):
            baseline = summ
            continue
        dose = _dose_of(rows, arm)
        if dose is None:
            continue
        if arm.startswith("sec4_w2_null"):
            key = re.sub(r"_a[mp]?\d+$", "", arm)  # collapse the dose suffix
            nulls.setdefault(key, {})[dose] = summ
            continue
        axis = _w2_axis_of(arm)
        if axis is not None:
            by_axis[axis][dose] = summ
    return by_axis, nulls, baseline


def _pooled(cells: Dict[float, dict], ind: str):
    """(doses, values) trial-level arrays over parse-gated dose cells."""
    xs, ys, dose_means = [], [], {}
    for d in sorted(cells):
        c = cells[d]
        if c["parse_rate"] < PARSE_GATE:
            continue
        vs = c["values"][ind]
        if not vs:
            continue
        dose_means[d] = float(np.mean(vs))
        for v in vs:
            xs.append(d); ys.append(v)
    return xs, ys, dose_means


def _null_band(nulls: Dict[str, Dict[float, dict]], ind: str) -> dict:
    """THICK null: one slope per random-null direction from its -3/+3 pair,
    reported as mean +/- 2sd (half-band = max(2sd, floor))."""
    slopes = []
    for key, cells in nulls.items():
        xs, ys, _ = _pooled(cells, ind)
        s = _ols_slope(xs, ys)
        if np.isfinite(s):
            slopes.append(s)
    mean = float(np.mean(slopes)) if slopes else 0.0
    sd = float(np.std(slopes)) if len(slopes) > 1 else 0.0
    return {"mean": mean, "sd": sd, "delta": max(2.0 * sd, NULL_SLOPE_FLOOR),
            "n": len(slopes)}


def _w2_indicator_stat(cells, ind, band) -> dict:
    """Per (axis, indicator): monotone dose-response + slope vs the null band."""
    xs, ys, dose_means = _pooled(cells, ind)
    doses = sorted(dose_means)
    rho = _spearman(doses, [dose_means[d] for d in doses])  # per-dose-mean trend
    slope = _ols_slope(xs, ys)
    z = ((slope - band["mean"]) / band["sd"]
         if band["sd"] > 1e-12 and np.isfinite(slope) else float("nan"))
    sign_ok = bool(np.isfinite(rho) and rho > 0)
    monotone = bool(np.isfinite(rho) and abs(rho) >= SPEARMAN_MIN and rho > 0)
    above_null = bool(np.isfinite(slope)
                      and abs(slope - band["mean"]) > band["delta"])
    return {"spearman": rho, "slope": slope, "z": z, "sign_ok": sign_ok,
            "monotone": monotone, "above_null": above_null,
            "dose_means": {str(k): v for k, v in dose_means.items()}}


def _w2_moves(stat: dict) -> bool:
    return bool(stat["monotone"] and stat["sign_ok"] and stat["above_null"])


def analyze_wave2(results_dir=None, assets_dir=None,
                  out_json: Optional[Path] = None,
                  out_png: Optional[Path] = None) -> dict:
    """Wave-2 COMMON-AXIS analysis. For each axis, the dose-response of BOTH
    I_BA (bet_ratio) and I_EC (extreme). Returns:
      axes[axis][ind] = monotone/sign/slope/z vs the thick null band,
      cross_indicator_slopes[axis][ind] = the axis's slope on each indicator,
      null_band[ind] = per-direction slope band (mean, sd, +/-2sd delta, n),
    and a verdict SHARED_COMMON_AXIS iff the SHARED axis moves BOTH indicators
    monotonically and above the null slope band, else NO_SHARED_AXIS.
    """
    results_dir = Path(results_dir) if results_dir is not None \
        else _MLC / "results" / "sec4_w2"
    assets_dir = Path(assets_dir) if assets_dir is not None else ASSETS_DIR

    by_axis, nulls, baseline = _dose_cells(results_dir)
    band = {ind: _null_band(nulls, ind) for ind in W2_INDICATORS}

    axes_out, cross = {}, {}
    for axis in W2_AXES:
        cells = by_axis[axis]
        axes_out[axis] = {ind: _w2_indicator_stat(cells, ind, band[ind])
                          for ind in W2_INDICATORS}
        cross[axis] = {ind: axes_out[axis][ind]["slope"]
                       for ind in W2_INDICATORS}

    shared = axes_out["shared"]
    shared_moves_both = bool(_w2_moves(shared["i_ba"]) and _w2_moves(shared["i_ec"]))
    verdict = "SHARED_COMMON_AXIS" if shared_moves_both else "NO_SHARED_AXIS"

    result = {
        "verdict": verdict,
        "shared_moves_both": shared_moves_both,
        "axes": axes_out,
        "cross_indicator_slopes": cross,
        "null_band": band,
        "n_null_directions": len(nulls),
        "baseline": ({ind: (float(np.mean(baseline["values"][ind]))
                            if baseline["values"][ind] else float("nan"))
                     for ind in W2_INDICATORS} if baseline else None),
    }
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
    if out_png is not None:
        try:
            make_figure_wave2(result, by_axis, band, Path(out_png))
        except Exception as e:
            print(f"[sec4-w2] figure skipped: {e}")
    return result


def make_figure_wave2(res: dict, by_axis: dict, band: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = {"behav_iba": "C0", "behav_iec": "C3", "shared": "C1",
              "confound": "C2"}
    labels = {"i_ba": "I_BA (bet ratio)", "i_ec": "I_EC (extreme rate)"}
    for ax, ind in zip(axs, W2_INDICATORS):
        for axis in W2_AXES:
            cells = by_axis.get(axis, {})
            _, _, dose_means = _pooled(cells, ind)
            doses = sorted(dose_means)
            if not doses:
                continue
            ax.plot(doses, [dose_means[d] for d in doses], "-o",
                    color=colors[axis], label=axis,
                    lw=2.2 if axis == "shared" else 1.2)
        ax.set_xlabel("steering alpha (sigma-units)")
        ax.set_ylabel(f"mean {labels[ind]}")
        ax.set_title(labels[ind])
        ax.legend(fontsize=8)
    fig.suptitle(f"§4 Wave-2 common axis — verdict: {res['verdict']}")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--assets-dir", default=str(ASSETS_DIR))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-png", default=str(OUT_PNG))
    ap.add_argument("--wave2", action="store_true",
                    help="run the Wave-2 common-axis analysis instead of Wave-1")
    args = ap.parse_args()
    if args.wave2:
        res = analyze_wave2(Path(args.results_dir), Path(args.assets_dir),
                            Path(args.out_json), Path(args.out_png))
        print(json.dumps({"verdict": res["verdict"],
                          "shared_moves_both": res["shared_moves_both"],
                          "cross_indicator_slopes": res["cross_indicator_slopes"]},
                         indent=2))
        return
    res = analyze(Path(args.results_dir), Path(args.assets_dir),
                  Path(args.out_json), Path(args.out_png))
    print(json.dumps({"verdict": res["verdict"],
                      "h1": res["h1_behavioural_writes"],
                      "h2": res["h2_readout_inert"]}, indent=2))


if __name__ == "__main__":
    main()
