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
import os
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


def _pooled(cells: Dict[float, dict], ind: str, parse_gate: float = PARSE_GATE):
    """(doses, values) trial-level arrays over parse-gated dose cells."""
    xs, ys, dose_means = [], [], {}
    for d in sorted(cells):
        c = cells[d]
        if c["parse_rate"] < parse_gate:
            continue
        vs = c["values"][ind]
        if not vs:
            continue
        dose_means[d] = float(np.mean(vs))
        for v in vs:
            xs.append(d); ys.append(v)
    return xs, ys, dose_means


def _null_band(nulls: Dict[str, Dict[float, dict]], ind: str,
               parse_gate: float = PARSE_GATE) -> dict:
    """THICK null: one slope per random-null direction from its -3/+3 pair,
    reported as mean +/- 2sd (half-band = max(2sd, floor))."""
    slopes = []
    for key, cells in nulls.items():
        xs, ys, _ = _pooled(cells, ind, parse_gate)
        s = _ols_slope(xs, ys)
        if np.isfinite(s):
            slopes.append(s)
    mean = float(np.mean(slopes)) if slopes else 0.0
    sd = float(np.std(slopes)) if len(slopes) > 1 else 0.0
    return {"mean": mean, "sd": sd, "delta": max(2.0 * sd, NULL_SLOPE_FLOOR),
            "n": len(slopes)}


def _w2_indicator_stat(cells, ind, band, parse_gate: float = PARSE_GATE) -> dict:
    """Per (axis, indicator): monotone dose-response + slope vs the null band."""
    xs, ys, dose_means = _pooled(cells, ind, parse_gate)
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


# ============================================================ WAVE 3
# goals-v2 adjudication rung-1 analyses (configs/arms_sec4_w3.yaml):
#   Q2 expression-matched control — does the 3-task shared3 axis move the SM
#      bet_ratio AND the IC risky rate, against IC's own behavioural axis as a
#      positive control and per-target IC random nulls?
#   Q3 condition modulation — is the +G-prompt dose slope different from the
#      Wave-1 −G behavioural dose slope (bootstrap CI on the difference)?
#   Q1 rung-2 — is the behavioural dose slope larger on post-loss than on
#      post-win states (conditional steering contrast)?

HF_CKPT_BASE = "experiments/sec4_causal/checkpoints"
W3_BOOT = 1000
W3_BOOT_SEED = 42
# arm-id prefixes of the sec4_w3 dose ladders (nulls/baselines handled apart)
W3_LADDERS = {
    "sh3_sm": "sec4_w3_sh3_sm_",
    "sh3_ic": "sec4_w3_sh3_ic_",
    "ic_own": "sec4_w3_ic_own_",
    "plusG": "sec4_w3_plusG_",
    "postloss": "sec4_w3_postloss_",
    "postwin": "sec4_w3_postwin_",
}
W1_BEHAV_PREFIX = "sec4_behavioural_"   # Wave-1 −G behavioural ladder (Q3 ref)


def fetch_hf_rollouts(phase: str, dest_dir: Path, prefix: str = "sec4_") -> int:
    """Download {HF_CKPT_BASE}/{phase}/*.jsonl rollouts into dest_dir (skipping
    files already present). Used by the wave3/postloss CLIs to pull the frozen
    Wave-1/2 ladders off HF for re-analysis; returns the download count."""
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    base = f"{HF_CKPT_BASE}/{phase}/"
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in api.list_repo_files("llm-addiction-research/llm-addiction",
                                 repo_type="dataset"):
        if not (f.startswith(base) and f.endswith(".jsonl")):
            continue
        name = Path(f).name
        if not name.startswith(prefix):
            continue
        dest = dest_dir / name
        if dest.exists():
            continue
        p = hf_hub_download("llm-addiction-research/llm-addiction", f,
                            repo_type="dataset", token=token)
        dest.write_bytes(Path(p).read_bytes())
        n += 1
    return n


def _w3_cells(results_dir: Path, prefix: str) -> Dict[float, dict]:
    """{dose: summary} for the `{prefix}a{m,p}k` dose ladder, in the _pooled /
    _w2_indicator_stat cell shape with the single metric key 'm' (_bet:
    bet_ratio for SM rows, risky 0/1 for IC rows — expression-matched)."""
    cells = {}
    for fp in sorted(Path(results_dir).glob(f"{prefix}a*.jsonl")):
        rows = _rows(fp)
        parse_ok = [r for r in rows if r.get("parse_ok")]
        dose = _dose_of(rows, fp.stem)
        if dose is None:
            continue
        cells[dose] = {
            "parse_rate": len(parse_ok) / len(rows) if rows else float("nan"),
            "n": len(rows),
            "values": {"m": [v for r in parse_ok
                             if (v := _bet(r)) is not None]},
        }
    return cells


def _w3_null_cells(results_dir: Path,
                   prefix: str = "sec4_w3_ic_null") -> Dict[str, dict]:
    """Per-null-direction {dose: summary} pairs (metric key 'm') for the IC
    random-direction nulls — feeds the same _null_band machinery as Wave-2."""
    nulls: Dict[str, Dict[float, dict]] = {}
    for fp in sorted(Path(results_dir).glob(f"{prefix}*.jsonl")):
        rows = _rows(fp)
        parse_ok = [r for r in rows if r.get("parse_ok")]
        dose = _dose_of(rows, fp.stem)
        if dose is None:
            continue
        key = re.sub(r"_a[mp]?\d+$", "", fp.stem)
        nulls.setdefault(key, {})[dose] = {
            "parse_rate": len(parse_ok) / len(rows) if rows else float("nan"),
            "n": len(rows),
            "values": {"m": [v for r in parse_ok
                             if (v := _bet(r)) is not None]},
        }
    return nulls


def _boot_slope_diff(cells_a, cells_b, n_boot=W3_BOOT, seed=W3_BOOT_SEED,
                     parse_gate=PARSE_GATE):
    """Point estimate + bootstrap 95% CI of slope(cells_a) - slope(cells_b),
    resampling (dose, value) trials with replacement within each ladder.
    parse_gate defaults to the W2/W3 0.8 (wave3 callers byte-identical); the
    W10 llama caller passes a looser SM-calibrated gate."""
    xa, ya, _ = _pooled(cells_a, "m", parse_gate)
    xb, yb, _ = _pooled(cells_b, "m", parse_gate)
    sa, sb = _ols_slope(xa, ya), _ols_slope(xb, yb)
    point = (sa - sb if np.isfinite(sa) and np.isfinite(sb) else float("nan"))
    if not np.isfinite(point):
        return point, [float("nan"), float("nan")], sa, sb
    xa, ya = np.asarray(xa), np.asarray(ya)
    xb, yb = np.asarray(xb), np.asarray(yb)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(xa), len(xa))
        ib = rng.integers(0, len(xb), len(xb))
        d = _ols_slope(xa[ia], ya[ia]) - _ols_slope(xb[ib], yb[ib])
        if np.isfinite(d):
            diffs.append(d)
    if len(diffs) < n_boot // 2:
        return point, [float("nan"), float("nan")], sa, sb
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return point, [float(lo), float(hi)], sa, sb


def analyze_wave3(results_dir=None, w1_results_dir=None, w2_results_dir=None,
                  out_json: Optional[Path] = None,
                  out_png: Optional[Path] = None) -> dict:
    """Wave-3 adjudication analysis. Returns per-question blocks:

    q2 (expression-matched): shared3-on-SM / shared3-on-IC / IC-own dose stats
       (_w2_indicator_stat vs the IC null slope band; the SM side reuses the
       Wave-2 thick SM null when w2_results_dir has it, else the slope floor).
       verdict: IC_LEVER_ABSENT when even IC's own behavioural axis is flat
       (no IC headroom — the control question is unanswerable), else
       SHARED_TASK_CONTROL when shared3 moves BOTH tasks, else TASK_SPECIFIC.
    q3: slope(+G ladder) vs slope(Wave-1 −G behavioural ladder) with a
       bootstrap CI on the difference. verdict CONDITION_MODULATES when the CI
       excludes 0, else NO_MODULATION.
    q1_rung2: postloss vs postwin steering slope contrast (same bootstrap).
       verdict POSTLOSS_STEER_AMPLIFIED when the post-loss slope is larger
       with CI > 0, else NO_POSTLOSS_MODULATION.
    """
    results_dir = Path(results_dir) if results_dir is not None \
        else _MLC / "results" / "sec4_w3"
    w1_results_dir = Path(w1_results_dir) if w1_results_dir is not None \
        else RESULTS_DIR
    w2_results_dir = Path(w2_results_dir) if w2_results_dir is not None \
        else _MLC / "results" / "sec4_w2"

    cells = {k: _w3_cells(results_dir, p) for k, p in W3_LADDERS.items()}

    # ---- Q2: expression-matched shared3 vs IC positive control vs IC nulls
    ic_band = _null_band(_w3_null_cells(results_dir), "m")
    _, w2_nulls, _ = _dose_cells(w2_results_dir)
    # sm_null_source makes the band's provenance explicit in the output: the
    # design gate is the THICK 20-direction Wave-2 SM null; the slope floor is
    # a far weaker fallback that must be visible, never silent.
    sm_null_source = "thick" if w2_nulls else "floor_fallback"
    sm_band = (_null_band(w2_nulls, "i_ba") if w2_nulls
               else {"mean": 0.0, "sd": 0.0, "delta": NULL_SLOPE_FLOOR, "n": 0})
    sh3_sm = _w2_indicator_stat(cells["sh3_sm"], "m", sm_band)
    sh3_ic = _w2_indicator_stat(cells["sh3_ic"], "m", ic_band)
    ic_own = _w2_indicator_stat(cells["ic_own"], "m", ic_band)
    if not _w2_moves(ic_own):
        q2_verdict = "IC_LEVER_ABSENT"
    elif _w2_moves(sh3_sm) and _w2_moves(sh3_ic):
        q2_verdict = "SHARED_TASK_CONTROL"
    else:
        q2_verdict = "TASK_SPECIFIC"

    # ---- Q3: +G dose slope vs the Wave-1 −G behavioural dose slope
    w1_cells = _w3_cells(w1_results_dir, W1_BEHAV_PREFIX)
    q3_diff, q3_ci, s_plus, s_minus = _boot_slope_diff(cells["plusG"], w1_cells)
    q3_verdict = ("CONDITION_MODULATES"
                  if np.isfinite(q3_diff) and np.isfinite(q3_ci[0])
                  and (q3_ci[0] > 0 or q3_ci[1] < 0)
                  else "NO_MODULATION")

    # ---- Q1 rung-2: post-loss vs post-win conditional steering contrast
    q1_diff, q1_ci, s_loss, s_win = _boot_slope_diff(cells["postloss"],
                                                     cells["postwin"])
    q1_verdict = ("POSTLOSS_STEER_AMPLIFIED"
                  if np.isfinite(q1_diff) and q1_diff > 0
                  and np.isfinite(q1_ci[0]) and q1_ci[0] > 0
                  else "NO_POSTLOSS_MODULATION")

    result = {
        "q2": {"verdict": q2_verdict, "sh3_sm": sh3_sm, "sh3_ic": sh3_ic,
               "ic_own": ic_own, "ic_null_band": ic_band,
               "sm_null_band": sm_band, "sm_null_source": sm_null_source},
        "q3": {"verdict": q3_verdict, "slope_plusG": s_plus,
               "slope_minusG_w1": s_minus, "diff": q3_diff, "ci95": q3_ci,
               "n_boot": W3_BOOT},
        "q1_rung2": {"verdict": q1_verdict, "slope_postloss": s_loss,
                     "slope_postwin": s_win, "diff": q1_diff, "ci95": q1_ci,
                     "n_boot": W3_BOOT},
        "verdicts": {"q2": q2_verdict, "q3": q3_verdict,
                     "q1_rung2": q1_verdict},
    }
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
    if out_png is not None:
        try:
            make_figure_wave3(result, cells, w1_cells, Path(out_png))
        except Exception as e:
            print(f"[sec4-w3] figure skipped: {e}")
    return result


def make_figure_wave3(res: dict, cells: dict, w1_cells: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

    def _plot(ax, cell, label, color, lw=1.4):
        _, _, dm = _pooled(cell, "m")
        doses = sorted(dm)
        if doses:
            ax.plot(doses, [dm[d] for d in doses], "-o", color=color,
                    label=label, lw=lw)

    _plot(axs[0], cells["sh3_sm"], "shared3 -> SM (bet ratio)", "C1", 2.2)
    _plot(axs[0], cells["sh3_ic"], "shared3 -> IC (risky rate)", "C4", 2.2)
    _plot(axs[0], cells["ic_own"], "IC own axis -> IC (risky rate)", "C2")
    axs[0].set_title(f"Q2 expression-matched — {res['q2']['verdict']}")
    _plot(axs[1], cells["plusG"], "+G prompt (w3)", "C3", 2.2)
    _plot(axs[1], w1_cells, "-G prompt (Wave-1)", "C0")
    axs[1].set_title(f"Q3 condition slope — {res['q3']['verdict']}")
    _plot(axs[2], cells["postloss"], "post-loss states", "C3", 2.2)
    _plot(axs[2], cells["postwin"], "post-win states", "C0")
    axs[2].set_title(f"Q1 rung-2 conditional steer — {res['q1_rung2']['verdict']}")
    for ax in axs:
        ax.set_xlabel("steering alpha (sigma-units)")
        ax.set_ylabel("mean indicator")
        ax.legend(fontsize=8)
    fig.suptitle("§4 Wave-3 adjudication (Q2 / Q3 / Q1-rung2)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


# ============================================================ WAVE 6
# Q2 MW object fix (configs/arms_sec4_w6.yaml): does MW's spin-vs-stop axis
# (mw_rc, built with the rc_keep loader so stop rows survive) write MW
# CORRECTLY SIGNED — the within-task control that resolved the same anomaly
# on IC in Wave-5? Nulls + baseline are REUSED from the sec4_w4 MW rollouts
# (same frozen pool, same MW sigma-unit convention), so this wave only spends
# rollout budget on the 7-dose mw_rc ladder.

W6_LADDER_PREFIX = "sec4_w6_mw_rc_"
W6_MW_NULL_PREFIX = "sec4_w4_mw_null"
W6_MW_BASELINE = "sec4_w4_mw_baseline"
W6_METRICS = ("spin", "bet")
# MW-calibrated parse gate. Real MW rollouts parse at 0.19-0.73 (measured on
# the frozen sec4_w4 set: baseline 0.650, nulls 0.455-0.730, mw_own_ap3
# 0.190), so the W2 PARSE_GATE=0.8 would discard EVERY real MW cell — nulls
# included — silently collapsing the band to NULL_SLOPE_FLOOR and forcing
# UNDERPOWERED. 0.45 is the same MW/IC low-parse sensitivity gate the
# INDEX.md Q2-rung1 analysis used; it keeps all six W4 nulls and the
# baseline while still excluding parse-cratered cells like mw_own_ap3.
W6_PARSE_GATE = 0.45


def _mw_spin(r) -> Optional[float]:
    """Spin/stop outcome per MW record: the runner's `risky` boolean as 0/1
    (risky = action == 'spin'); None when absent (non-MW record)."""
    v = r.get("risky")
    return None if v is None else (1.0 if v else 0.0)


def _w6_summ(rows: List[dict]) -> dict:
    """Per-arm summary in the _pooled cell shape with BOTH Wave-6 metrics:
    'spin' (spin rate) and 'bet' (bet_ratio, 0.0 on stop rows)."""
    parse_ok = [r for r in rows if r.get("parse_ok")]
    return {
        "parse_rate": len(parse_ok) / len(rows) if rows else float("nan"),
        "n": len(rows),
        "values": {
            "spin": [v for r in parse_ok if (v := _mw_spin(r)) is not None],
            "bet": [float(r["bet_ratio"]) for r in parse_ok
                    if r.get("bet_ratio") is not None],
        },
    }


def _w6_cells(results_dir: Path,
              prefix: str = W6_LADDER_PREFIX) -> Dict[float, dict]:
    """{dose: summary} for the mw_rc dose ladder (both metrics per cell)."""
    cells = {}
    for fp in sorted(Path(results_dir).glob(f"{prefix}a*.jsonl")):
        rows = _rows(fp)
        dose = _dose_of(rows, fp.stem)
        if dose is None:
            continue
        cells[dose] = _w6_summ(rows)
    return cells


def _w6_null_cells(w4_results_dir: Path) -> Dict[str, dict]:
    """Per-null-direction {dose: summary} pairs from the REUSED sec4_w4 MW
    null rollouts (sec4_w4_mw_null<k>_{am3,ap3}) — feeds _null_band for both
    Wave-6 metrics."""
    nulls: Dict[str, Dict[float, dict]] = {}
    for fp in sorted(Path(w4_results_dir).glob(f"{W6_MW_NULL_PREFIX}*.jsonl")):
        rows = _rows(fp)
        dose = _dose_of(rows, fp.stem)
        if dose is None:
            continue
        key = re.sub(r"_a[mp]?\d+$", "", fp.stem)
        nulls.setdefault(key, {})[dose] = _w6_summ(rows)
    return nulls


def analyze_wave6(results_dir=None, w4_results_dir=None,
                  out_json: Optional[Path] = None,
                  out_png: Optional[Path] = None) -> dict:
    """Wave-6 adjudication: mw_rc dose-response on BOTH the spin/stop outcome
    AND bet_ratio, against the sec4_w4 MW null band + baseline.

    SPIN-RATE METRIC (decided from runner._run_arm_mw): every MW rollout row
    records `action` ('spin'/'stop' from the frozen parser's game choice:
    2=spin, 1=stop), `risky` = (action == 'spin'), `bet_ratio` (amount /
    max(balance,1) when spinning, 0.0 on stop), and `parse_ok`. A STOP is an
    ordinary parse_ok=True row (e.g. 'Final Decision: Option 2' -> choice 1,
    bet 0, valid) with risky=False and bet_ratio=0.0; a parse FAILURE also
    defaults to stop but with parse_ok=False. Therefore:
      spin rate = mean(`risky`) over parse_ok rows (parse-gated so defaulted
                  stops never count as behaviour), and
      bet metric = mean(`bet_ratio`) over parse_ok rows (stop rows contribute
                  0.0 — unconditional bet intensity, the W4-comparable object).

    Cell inclusion uses the MW-calibrated W6_PARSE_GATE (0.45), NOT the W2
    PARSE_GATE (0.8): real MW cells parse at 0.19-0.73, so the 0.8 gate would
    discard every ladder/null/baseline cell and force UNDERPOWERED with an
    empty (floor-only) null band. The result records `parse_gate` and the
    EFFECTIVE per-metric band count `n_null_slopes` (what actually gated the
    verdict) alongside `n_null_directions` (directions found on disk).

    Verdict (primary outcome = the spin metric, the axis's own object):
      MW_RC_CORRECTLY_SIGNED  spin rate rises monotonically with +alpha and
                              the slope clears the MW null band,
      STILL_ANOMALOUS         the slope clears the null band but wrong-signed
                              or incoherent (the W3/W4 anomaly signature),
      UNDERPOWERED            nothing clears the null band.
    bet_corroborates reports whether bet_ratio independently moves the same
    way (spin implies bet>0, so it should co-move if the write is real)."""
    results_dir = Path(results_dir) if results_dir is not None \
        else _MLC / "results" / "sec4_w6"
    w4_results_dir = Path(w4_results_dir) if w4_results_dir is not None \
        else _MLC / "results" / "sec4_w4"

    cells = _w6_cells(results_dir)
    nulls = _w6_null_cells(w4_results_dir)
    band = {m: _null_band(nulls, m, W6_PARSE_GATE) for m in W6_METRICS}
    stats = {m: _w2_indicator_stat(cells, m, band[m], W6_PARSE_GATE)
             for m in W6_METRICS}
    parse_by_dose = {str(d): cells[d]["parse_rate"] for d in sorted(cells)}

    base_fp = Path(w4_results_dir) / f"{W6_MW_BASELINE}.jsonl"
    baseline = _w6_summ(_rows(base_fp)) if base_fp.exists() else None
    base_means = ({m: (float(np.mean(baseline["values"][m]))
                       if baseline["values"][m] else float("nan"))
                   for m in W6_METRICS} if baseline else None)

    spin = stats["spin"]
    if _w2_moves(spin):
        verdict = "MW_RC_CORRECTLY_SIGNED"
    elif spin["above_null"]:
        verdict = "STILL_ANOMALOUS"
    else:
        verdict = "UNDERPOWERED"

    result = {
        "verdict": verdict,
        "spin": stats["spin"],
        "bet": stats["bet"],
        "bet_corroborates": _w2_moves(stats["bet"]),
        "null_band": band,
        "n_null_directions": len(nulls),
        "n_null_slopes": {m: band[m]["n"] for m in W6_METRICS},
        "parse_gate": W6_PARSE_GATE,
        "baseline": base_means,
        "parse_by_dose": parse_by_dose,
    }
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
    if out_png is not None:
        try:
            make_figure_wave6(result, cells, Path(out_png))
        except Exception as e:
            print(f"[sec4-w6] figure skipped: {e}")
    return result


def make_figure_wave6(res: dict, cells: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    labels = {"spin": "spin rate (risky)", "bet": "bet ratio"}
    for ax, m in zip(axs, W6_METRICS):
        _, _, dm = _pooled(cells, m, W6_PARSE_GATE)
        doses = sorted(dm)
        if doses:
            ax.plot(doses, [dm[d] for d in doses], "-o", color="C3",
                    label=f"mw_rc -> {labels[m]}", lw=2.2)
        base = (res["baseline"] or {}).get(m)
        if base is not None and np.isfinite(base):
            ax.axhline(base, color="k", lw=0.5, ls=":", label="W4 MW baseline")
        ax.set_xlabel("steering alpha (sigma-units)")
        ax.set_ylabel(f"mean {labels[m]}")
        ax.set_title(labels[m])
        ax.legend(fontsize=8)
    fig.suptitle(f"§4 Wave-6 MW spin/stop axis — verdict: {res['verdict']}")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


# ============================================================ WAVE 10
# §4.1 + §4.3 LLaMA model symmetry (configs/arms_sec4_w10a/b.yaml):
#   §4.1 read-vs-write dissociation — does the READOUT axis (the better
#        decoder, built per-layer through the LlamaScope decoder.weight.T) stay
#        INERT while the BEHAVIOURAL axis moves betting, at the SAME L14-19
#        layers (monitor != controller)? Absolute-effect metric (llama gap
#        0.0225 small). NULL-CAUSE diagnosis distinguishes wrong-object vs
#        headroom vs wrong-window when nothing writes.
#   §4.3 condition modulation — does +G / +M change the WRITABILITY (dose
#        slope) of the shared behavioural axis vs the -G control (bootstrap CI)?

W10_RESULTS_DIR = _MLC / "results" / "sec4_w10"
# LLaMA SM parse floor: looser than the gemma 0.8 (llama SM parses lower); cells
# below this are dropped before behaviour is read, and every axis reports its
# per-dose parse so the null-cause diagnosis can flag a parse-gated collapse.
W10_PARSE_GATE = 0.5
W10A_AXES = ("behavioural", "readout", "confound")
W10A_SPEC = ("iec", "ilc")               # indicator-specificity behavioural axes
W10B_CONDS = {"minusG": "sec4_w10b_minusG_", "plusG": "sec4_w10b_plusG_",
              "plusM": "sec4_w10b_plusM_"}
W10_CEIL, W10_FLOOR = 0.9, 0.05          # headroom thresholds on the baseline bet
# The 'monitor != controller' win requires the readout to be a GENUINELY
# informative decoder (an inert-AND-uninformative readout is no evidence of a
# monitor, just a broken axis). Gate the win on above-chance held-out decoding;
# 0.5 is chance, and we keep a margin because no bootstrap LB is stored.
W10_READOUT_AUC_MIN = 0.55
W10A_READOUT_KEY = "llama_slot_machine_i_ba_readout"  # readout asset stem (auc)


def _w10a_role(arm: str) -> Optional[str]:
    """Axis/role of a sec4_w10a_* arm: one of W10A_AXES/W10A_SPEC, or
    'null'/'baseline'/'cum'. None if not a w10a arm."""
    if not arm.startswith("sec4_w10a_"):
        return None
    rest = arm[len("sec4_w10a_"):]
    if rest.startswith("null"):
        return "null"
    if rest.startswith("baseline"):
        return "baseline"
    if rest.startswith("cum"):
        return "cum"
    for a in W10A_AXES + W10A_SPEC:
        if rest.startswith(a + "_a"):
            return a
    return None


def _w10a_axis_stat(cells, base_bet, null_delta) -> dict:
    """Per-axis dose stat (absolute effect): trial-level spearman/slope, sign,
    monotone, and whether the extreme-dose effect clears the null band. Reports
    per-dose parse so a parse-gated collapse is visible."""
    kept = {d: c for d, c in cells.items() if c["parse_rate"] >= W10_PARSE_GATE}
    doses = sorted(kept)
    xs, ys = [], []
    for d in doses:
        for b in kept[d]["bets"]:
            xs.append(d); ys.append(b)
    rho = _spearman(xs, ys)
    slope = _ols_slope(xs, ys)
    sign_ok = bool(np.isfinite(rho) and rho > 0)
    monotone = bool(np.isfinite(rho) and abs(rho) >= SPEARMAN_MIN and rho > 0)
    eff = _extreme_effect(kept, base_bet)
    above_null = bool(np.isfinite(eff) and abs(eff) > null_delta)
    parse_by_dose = {str(d): cells[d]["parse_rate"] for d in sorted(cells)}
    return {"spearman": rho, "slope": slope, "sign_ok": sign_ok,
            "monotone": monotone, "above_null": above_null,
            "extreme_effect": eff, "n_doses": len(doses),
            "parse_by_dose": parse_by_dose,
            "mean_parse": (float(np.mean(list(parse_by_dose.values())))
                           if parse_by_dose else float("nan"))}


def analyze_w10_sec41(results_dir, assets_dir) -> dict:
    """§4.1 llama read-vs-write dissociation. Verdict:
      LLAMA_MONITOR_NEQ_CONTROLLER  behavioural writes, readout INERT *and*
                                    above-chance decoding, confound clean (win);
      READOUT_ALSO_WRITES           both write (no dissociation);
      CONFOUNDED                    behavioural AND confound both write — H3
                                    fails, balance-signal could carry it;
      READOUT_UNINFORMATIVE         behavioural writes, readout inert BUT its
                                    decoding AUC <= chance floor (no real
                                    monitor -> not a win);
      NULL                          behavioural does not write — with a
                                    null_cause hint in {object, headroom,
                                    window} (see below).
    NULL-CAUSE: behavioural moved but incoherent/wrong-signed -> 'object'
    (wrong indicator/expression, the MW-anomaly signature); baseline at
    ceiling/floor -> 'headroom' (no room to move); confound moves but
    behavioural does not -> 'object'; nothing at these layers moves ->
    'window' (wrong write window). Fields report parse rates, effect vs the
    null band, and whether behavioural/readout/confound each moved so the
    caller can tell a genuine dissociation (readout-only null) from a
    window/object failure (behavioural null too)."""
    results_dir, assets_dir = Path(results_dir), Path(assets_dir)
    by_axis = {a: {} for a in W10A_AXES + W10A_SPEC}
    nulls, baseline, cum = [], None, {}
    for fp in sorted(results_dir.glob("sec4_w10a_*.jsonl")):
        arm = fp.stem
        rows = _rows(fp)
        summ = _arm_summary(rows)
        if summ["n_parsed_bet"] and any(_manip_proj(r) is not None for r in rows):
            assert np.isfinite(summ["manip_proj"]), \
                f"{arm}: manip_proj is NaN despite logged vector_log.proj"
        role = _w10a_role(arm)
        if role == "null":
            nulls.append(summ)
        elif role == "baseline":
            baseline = summ
        elif role == "cum":
            cum[arm] = summ["mean_bet"]
        elif role in by_axis:
            dose = _dose_of(rows, arm)
            if dose is not None:
                by_axis[role][dose] = summ

    null_bets = [s["mean_bet"] for s in nulls if np.isfinite(s["mean_bet"])]
    null_mean = float(np.mean(null_bets)) if null_bets else float("nan")
    null_sd = float(np.std(null_bets)) if len(null_bets) > 1 else 0.0
    null_delta = max(2.0 * null_sd, 0.03)
    base_bet = (baseline["mean_bet"] if baseline
                else by_axis["behavioural"].get(0.0, {}).get("mean_bet",
                                                             float("nan")))

    axes_out = {a: _w10a_axis_stat(by_axis[a], base_bet, null_delta)
                for a in W10A_AXES + W10A_SPEC}
    behav, read, conf = (axes_out["behavioural"], axes_out["readout"],
                         axes_out["confound"])
    behav_writes, read_writes = _writes(behav), _writes(read)
    confound_writes = _writes(conf)          # H3 gate (mirrors gemma _verdict)
    behavioural_moved = bool(behav["above_null"])
    readout_moved = bool(read["above_null"])
    confound_moved = bool(conf["above_null"])
    # The readout must DECODE the indicator above chance for its inertness to be
    # evidence of an informative-but-non-steering monitor; a broken/row-misaligned
    # readout is ALSO inert AND has AUC~0.5, and must NOT credit the dissociation.
    dec_auc = _decoding_auc(assets_dir)
    readout_auc = float(dec_auc.get(W10A_READOUT_KEY, float("nan")))
    readout_informative = bool(np.isfinite(readout_auc)
                               and readout_auc > W10_READOUT_AUC_MIN)

    if behav_writes and read_writes:
        verdict, null_cause = "READOUT_ALSO_WRITES", "none"  # no dissociation
    elif behav_writes and confound_writes:
        # H3 fails: balance-signalling reproduces the betting change -> demote
        # the headline (the confound, not the risk object, could carry it).
        verdict, null_cause = "CONFOUNDED", "confound"
    elif behav_writes and not read_writes:
        if readout_informative:
            verdict, null_cause = "LLAMA_MONITOR_NEQ_CONTROLLER", "none"
        else:
            # readout is inert AND cannot decode above chance -> no monitor to
            # speak of; the 'dissociation' is uninformative, not a win.
            verdict, null_cause = "READOUT_UNINFORMATIVE", "readout_broken"
    else:
        verdict = "NULL"
        at_ceiling = bool(np.isfinite(base_bet)
                          and (base_bet > W10_CEIL or base_bet < W10_FLOOR))
        if behavioural_moved:            # moved but incoherent = wrong object
            null_cause = "object"
        elif at_ceiling:                 # no room to move = headroom
            null_cause = "headroom"
        elif confound_moved:             # only balance-signal moves = object
            null_cause = "object"
        else:                            # nothing at these layers = window
            null_cause = "window"

    return {
        "verdict": verdict,
        "null_cause": null_cause,
        "axes": axes_out,
        "behavioural_writes": behav_writes,
        "readout_writes": read_writes,
        "confound_writes": confound_writes,
        "behavioural_moved": behavioural_moved,
        "readout_moved": readout_moved,
        "confound_moved": confound_moved,
        "readout_auc": readout_auc,
        "readout_informative": readout_informative,
        "null_band": {"mean": null_mean, "sd": null_sd, "delta": null_delta,
                      "n": len(null_bets)},
        "baseline_bet": base_bet,
        "cos_read_write": _cos_read_write(assets_dir),
        "decoding_auc": dec_auc,
        "cumulative_window_bet": cum,    # H4 locality (disclosure, not gated)
    }


def _w10b_cells(results_dir: Path, prefix: str) -> Dict[float, dict]:
    """{dose: summary} for a sec4_w10b_<cond>_a{m,p}k ladder (metric key 'm' =
    _bet, in the _pooled cell shape)."""
    cells = {}
    for fp in sorted(Path(results_dir).glob(f"{prefix}a*.jsonl")):
        rows = _rows(fp)
        parse_ok = [r for r in rows if r.get("parse_ok")]
        dose = _dose_of(rows, fp.stem)
        if dose is None:
            continue
        cells[dose] = {
            "parse_rate": len(parse_ok) / len(rows) if rows else float("nan"),
            "n": len(rows),
            "values": {"m": [v for r in parse_ok if (v := _bet(r)) is not None]},
        }
    return cells


def analyze_w10_sec43(results_dir) -> dict:
    """§4.3 llama condition modulation. slope(+G) and slope(+M) vs the -G
    control slope, each with a bootstrap CI on the difference. Verdict
    CONDITION_MODULATES when ANY difference CI excludes 0, else
    NO_MODULATION_HEADROOM (the expected small-gap outcome). Reports per-
    condition parse and a headroom flag (all baselines at ceiling/floor)."""
    results_dir = Path(results_dir)
    cells = {k: _w10b_cells(results_dir, p) for k, p in W10B_CONDS.items()}
    slopes, cond_parse, base_bets = {}, {}, {}
    for k, c in cells.items():
        xs, ys, dm = _pooled(c, "m", W10_PARSE_GATE)
        slopes[k] = _ols_slope(xs, ys)
        cond_parse[k] = {str(d): c[d]["parse_rate"] for d in sorted(c)}
        base_bets[k] = dm.get(0.0, float("nan"))   # alpha-0 = condition floor
    diffs = {}
    for k in ("plusG", "plusM"):
        diff, ci, s_k, s_m = _boot_slope_diff(cells[k], cells["minusG"],
                                              parse_gate=W10_PARSE_GATE)
        diffs[k] = {"diff": diff, "ci95": ci, "slope_cond": s_k,
                    "slope_minusG": s_m,
                    "excludes_zero": bool(np.isfinite(ci[0])
                                          and (ci[0] > 0 or ci[1] < 0))}
    modulates = any(d["excludes_zero"] for d in diffs.values())
    headroom = all((not np.isfinite(b)) or b > W10_CEIL or b < W10_FLOOR
                   for b in base_bets.values())
    verdict = ("CONDITION_MODULATES" if modulates
               else "NO_MODULATION_HEADROOM")
    return {
        "verdict": verdict,
        "slopes": slopes,
        "diffs": diffs,
        "condition_baseline_bet": base_bets,
        "headroom_all_conditions": headroom,
        "parse_by_condition": cond_parse,
        "n_boot": W3_BOOT,
    }


def analyze_w10(results_dir=None, assets_dir=None,
                out_json: Optional[Path] = None) -> dict:
    """W10 combined §4.1 + §4.3 llama analysis. Returns
    {'sec41': analyze_w10_sec41, 'sec43': analyze_w10_sec43, 'verdicts': ...}."""
    results_dir = Path(results_dir) if results_dir is not None \
        else W10_RESULTS_DIR
    assets_dir = Path(assets_dir) if assets_dir is not None else ASSETS_DIR
    sec41 = analyze_w10_sec41(results_dir, assets_dir)
    sec43 = analyze_w10_sec43(results_dir)
    result = {"sec41": sec41, "sec43": sec43,
              "verdicts": {"sec41": sec41["verdict"],
                           "sec41_null_cause": sec41["null_cause"],
                           "sec43": sec43["verdict"]}}
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
    return result


# ============================================================ WAVE 11
# §4.2 LLaMA task-own write-window SCAN (configs/arms_sec4_w11ic/mw.yaml):
# W9's llama matrix reused the SM write window (L14-19) for IC and MW, so its
# self-cells were weak (icrc->ic wrong-signed, mwrc->mw n=16). W11 scans each
# task's OWN behavioural axis (IC ic_rc risky-choice, MW mw_rc spin/stop) across
# four candidate write windows to find where each task's causal write lives
# BEFORE the matrix re-run (the follow-up). Absolute-effect metric (the llama
# +/-G gap is small): IC risky rate, MW spin rate = mean(`risky`) over parse_ok
# rows — the SAME field both task runners record (IC risky = choice>=3, MW risky
# = spin), so the outcome extractor is shared (_mw_spin). Per task: 4 windows x
# {-3,0,+3} (the a0 cell = that window's own no-steer baseline) + a task-level
# no-steer baseline + 3 random nulls (the null band on the absolute rate).

W11_RESULTS_DIR = _MLC / "results" / "sec4_w11"
# IC/MW low-parse gate (same as the W6 MW / INDEX Q2 sensitivity gate); the W2
# 0.8 gate would discard every real IC/MW cell.
W11_PARSE_GATE = 0.45
W11_WINDOWS = ((10, 15), (12, 17), (14, 19), (16, 21))
W11_TASKS = ("ic", "mw")
W11_NULL_FLOOR = 0.03            # min half-band on the absolute rate effect
W11_CEIL, W11_FLOOR = 0.9, 0.05  # headroom thresholds on the baseline rate


def _w11_summ(rows: List[dict]) -> dict:
    """Per-arm summary in the _pooled cell shape: parse rate + the risky/spin
    RATE values (mean(`risky`) is the IC risky rate / MW spin rate). Reuses the
    shared _mw_spin outcome extractor (both task runners record `risky`)."""
    parse_ok = [r for r in rows if r.get("parse_ok")]
    return {
        "parse_rate": len(parse_ok) / len(rows) if rows else float("nan"),
        "n": len(rows),
        "values": {"rate": [v for r in parse_ok
                            if (v := _mw_spin(r)) is not None]},
    }


def _w11_window_cells(results_dir: Path, task: str,
                      lo: int, hi: int) -> Dict[float, dict]:
    """{dose: summary} for one window's {-3,0,+3} ladder (metric key 'rate')."""
    cells = {}
    for fp in sorted(Path(results_dir).glob(
            f"sec4_w11{task}_L{lo}_{hi}_a*.jsonl")):
        rows = _rows(fp)
        dose = _dose_of(rows, fp.stem)
        if dose is not None:
            cells[dose] = _w11_summ(rows)
    return cells


def _w11_null_rates(results_dir: Path, task: str) -> List[float]:
    """Mean rate of each parse-gated random-null arm (the absolute-effect null
    spread — mirrors analyze_w10_sec41's null_bets, not a per-direction slope)."""
    out = []
    for fp in sorted(Path(results_dir).glob(f"sec4_w11{task}_null*.jsonl")):
        s = _w11_summ(_rows(fp))
        vs = s["values"]["rate"]
        if vs and s["parse_rate"] >= W11_PARSE_GATE:
            out.append(float(np.mean(vs)))
    return out


def _w11_baseline_rate(results_dir: Path, task: str) -> float:
    """Task-level no-steer baseline rate (sec4_w11{task}_base), or NaN."""
    fp = Path(results_dir) / f"sec4_w11{task}_base.jsonl"
    if not fp.exists():
        return float("nan")
    s = _w11_summ(_rows(fp))
    vs = s["values"]["rate"]
    return (float(np.mean(vs))
            if vs and s["parse_rate"] >= W11_PARSE_GATE else float("nan"))


def _w11_window_stat(cells: Dict[float, dict], base_rate: float,
                     null_delta: float) -> dict:
    """Per-window dose-response on the risky/spin rate: trial-level spearman +
    slope, the ABSOLUTE effect (dose-mean furthest from the window's own a0
    baseline), and whether it clears the random-null band. 'coherent' = the
    effect clears the band AND rises monotonically with +alpha (sign_ok)."""
    xs, ys, dm = _pooled(cells, "rate", W11_PARSE_GATE)
    doses = sorted(dm)
    a0 = dm.get(0.0, base_rate)                       # per-window baseline
    rho = _spearman(doses, [dm[d] for d in doses])
    slope = _ols_slope(xs, ys)
    eff = float("nan")
    if dm and np.isfinite(a0):
        far = max(dm, key=lambda d: abs(dm[d] - a0))  # extreme-dose displacement
        eff = dm[far] - a0
    sign_ok = bool(np.isfinite(rho) and rho > 0)
    monotone = bool(np.isfinite(rho) and abs(rho) >= SPEARMAN_MIN and rho > 0)
    above_null = bool(np.isfinite(eff) and abs(eff) > null_delta)
    return {
        "spearman": rho, "slope": slope, "extreme_effect": eff,
        "a0_rate": a0, "sign_ok": sign_ok, "monotone": monotone,
        "above_null": above_null,
        "coherent": bool(sign_ok and monotone and above_null),
        "n_doses": len(doses),
        "dose_means": {str(k): v for k, v in dm.items()},
        "parse_by_dose": {str(d): cells[d]["parse_rate"] for d in sorted(cells)},
    }


def _w11_task(results_dir: Path, task: str) -> dict:
    """Per-task window scan. Picks the strongest COHERENT window (max |effect|
    among windows that clear the null band AND rise with +alpha); falls back to
    the strongest above-null-but-incoherent window; else NULL. On NULL,
    null_cause distinguishes 'headroom' (baseline at ceiling/floor, no room to
    move) from 'absent' (genuinely no causal write at these layers — the
    reportable 'llama IC/MW write is headroom-limited' finding)."""
    base_rate = _w11_baseline_rate(results_dir, task)
    null_rates = _w11_null_rates(results_dir, task)
    null_sd = float(np.std(null_rates)) if len(null_rates) > 1 else 0.0
    null_delta = max(2.0 * null_sd, W11_NULL_FLOOR)

    windows = {f"L{lo}_{hi}":
               _w11_window_stat(_w11_window_cells(results_dir, task, lo, hi),
                                base_rate, null_delta)
               for lo, hi in W11_WINDOWS}
    coherent = {k: w for k, w in windows.items() if w["coherent"]}
    above = {k: w for k, w in windows.items() if w["above_null"]}

    chosen, null_cause = None, "none"
    if coherent:
        chosen = max(coherent, key=lambda k: abs(coherent[k]["extreme_effect"]))
        verdict = f"{task.upper()}_HAS_CAUSAL_WINDOW"
    elif above:
        chosen = max(above, key=lambda k: abs(above[k]["extreme_effect"]))
        verdict = f"{task.upper()}_WINDOW_INCOHERENT"
    else:
        verdict = f"{task.upper()}_NO_WINDOW"
        at_headroom = bool(np.isfinite(base_rate)
                           and (base_rate > W11_CEIL or base_rate < W11_FLOOR))
        null_cause = "headroom" if at_headroom else "absent"

    return {
        "verdict": verdict,
        "any_window_significant": bool(coherent),
        "chosen_window": chosen,
        "chosen_effect": (windows[chosen]["extreme_effect"]
                          if chosen else float("nan")),
        "null_cause": null_cause,
        "windows": windows,
        "baseline_rate": base_rate,
        "null_band": {"delta": null_delta, "sd": null_sd, "n": len(null_rates)},
        "parse_gate": W11_PARSE_GATE,
    }


def analyze_w11(results_dir=None, out_json: Optional[Path] = None,
                out_png: Optional[Path] = None) -> dict:
    """§4.2 llama task-own write-window scan (IC ic_rc, MW mw_rc). Per task it
    returns the per-window dose-response (absolute risky/spin rate effect vs the
    random-null band), the strongest coherent window, and whether ANY window
    writes significantly. The chosen windows feed the FOLLOW-UP matrix re-run
    (NOT this workflow); when neither task has a coherent window the scan
    reports that llama IC/MW causal write is genuinely absent / headroom-limited
    at these layers (a reportable finding, per-task null_cause)."""
    results_dir = Path(results_dir) if results_dir is not None \
        else W11_RESULTS_DIR
    tasks = {t: _w11_task(results_dir, t) for t in W11_TASKS}
    result = {
        "tasks": tasks,
        "verdicts": {t: tasks[t]["verdict"] for t in W11_TASKS},
        "chosen_windows": {t: tasks[t]["chosen_window"] for t in W11_TASKS},
        "any_window_significant": {t: tasks[t]["any_window_significant"]
                                   for t in W11_TASKS},
    }
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
    if out_png is not None:
        try:
            make_figure_w11(result, Path(results_dir), Path(out_png))
        except Exception as e:
            print(f"[sec4-w11] figure skipped: {e}")
    return result


def make_figure_w11(res: dict, results_dir: Path, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    labels = {"ic": "IC risky rate", "mw": "MW spin rate"}
    for ax, task in zip(axs, W11_TASKS):
        for lo, hi in W11_WINDOWS:
            cells = _w11_window_cells(results_dir, task, lo, hi)
            _, _, dm = _pooled(cells, "rate", W11_PARSE_GATE)
            doses = sorted(dm)
            if doses:
                ax.plot(doses, [dm[d] for d in doses], "-o",
                        label=f"L{lo}-{hi}", lw=1.8)
        chosen = res["tasks"][task]["chosen_window"]
        ax.set_xlabel("steering alpha (sigma-units)")
        ax.set_ylabel(f"mean {labels[task]}")
        ax.set_title(f"{labels[task]} — chosen: {chosen or 'none'}")
        ax.legend(fontsize=8)
    fig.suptitle("§4.2 Wave-11 LLaMA task-own write-window scan")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


# ============================================================ WAVE 13
# §4 NECESSITY via PROJECT-OUT (configs/arms_sec4_w13.yaml). Every prior wave
# tested SUFFICIENCY (add alpha*dir -> behaviour moves). W13 REMOVES the axis
# from the residual stream (h -= (h·û)û at every write-window layer, all
# positions, every forward) and asks whether betting DROPS. The KEY pattern:
# the BEHAVIOURAL axis is NECESSARY (project-out drops betting below the
# random-direction project-out null band) while the CONFOUND and READOUT axes
# are NOT (their project-out leaves betting inside the null band) — pure +
# necessary, upgrading §4 to sufficiency+necessity. Reported PER MODEL (gemma
# L16-21, llama L14-19). Absolute-effect (llama gap small): mean bet_ratio drop
# vs the no-hook baseline, gated against the random-null project-out band.
#
# Balance as a confound is removed BY CONSTRUCTION (matched pairs, not post-hoc
# stratification): the baseline and every project-out arm replay the SAME
# held-out state slice with the SAME alpha-independent seeds, so records pair by
# seed. Within a pair both arms see the identical prompt and thus the identical
# shown balance (`balance` is parsed from the prompt, a function of the
# seed/state, not of the model output the hook perturbs). The paired drop is
# therefore a within-balance drop for every pair; project-out cannot shift which
# balances were sampled, so a balance-mix artefact is impossible. `by_bin` (the
# §4.3 fixed-balance-window read-out) is DESCRIPTIVE only — a count-weighted mean
# over its bins is the raw paired drop by construction, so it is not an
# independent artefact guard.

W13_RESULTS_DIR = _MLC / "results" / "sec4_w13"
# LLaMA SM parses lower than gemma; reuse the W10 SM floor so a real llama cell
# is not discarded. Gemma SM parses well above it, so the gate is a no-op there.
W13_PARSE_GATE = 0.5
W13_MODELS = ("gemma", "llama")
W13_AXES = ("behavioural", "confound", "readout")
W13_NULL_FLOOR = 0.02            # min half-band on the absolute bet drop
W13_BAL_BINS = (0, 40, 80, 120, 200, 100000)   # balance strata edges ($)


def _bal_of(r: dict) -> Optional[float]:
    """Balance the trial was decided at (source_state.balance), for the
    descriptive within-balance read-out. None when the record omits it."""
    ss = r.get("source_state")
    if isinstance(ss, dict) and ss.get("balance") is not None:
        return float(ss["balance"])
    return None


def _w13_arm_rows(results_dir: Path, arm_id: str) -> List[dict]:
    fp = Path(results_dir) / f"{arm_id}.jsonl"
    return _rows(fp) if fp.exists() else []


def _w13_bets(rows: List[dict]) -> List[float]:
    """Parse-gated per-trial bet_ratios (drops parse-fail rows before betting is
    read, mirroring _arm_summary)."""
    return [b for r in rows if r.get("parse_ok") and (b := _bet(r)) is not None]


def _w13_paired_drop(base_rows, proj_rows) -> dict:
    """Matched-pairs drop baseline - project_out. Pairs parse-ok rows by `seed`
    (the arms replay the identical seeded slice), so each pair shares the same
    source state and therefore the same shown balance (`balance` is parsed from
    the prompt at runner.py:534, a function of the seed/state, NOT of the model
    output the hook perturbs). Returns the mean within-pair drop (positive =
    betting fell). Balance is removed as a confound BY CONSTRUCTION here: because
    both arms see the identical balance for every pair, the drop cannot be a
    balance-mix artefact — there is nothing to re-weight. `by_bin` reports the
    drop within each shown-balance stratum for descriptive read-out only (it is
    NOT an independent artefact guard: the paired design already holds balance
    fixed, and a count-weighted mean over these bins is the raw mean by
    construction)."""
    def _by_seed(rows):
        return {r["seed"]: r for r in rows
                if r.get("parse_ok") and _bet(r) is not None
                and r.get("seed") is not None}
    b, p = _by_seed(base_rows), _by_seed(proj_rows)
    seeds = sorted(set(b) & set(p))
    if not seeds:
        return {"n_pairs": 0, "paired_drop": float("nan"), "by_bin": {}}
    diffs = [(_bet(b[s]) - _bet(p[s])) for s in seeds]
    # descriptive: within-pair drop grouped by the (arm-invariant) shown balance.
    bins: Dict[int, List[float]] = {}
    for s in seeds:
        bal = _bal_of(b[s])
        k = (int(np.digitize([bal], W13_BAL_BINS)[0]) if bal is not None else -1)
        bins.setdefault(k, []).append(_bet(b[s]) - _bet(p[s]))
    by_bin = {str(k): {"n": len(v), "mean_drop": float(np.mean(v))}
              for k, v in sorted(bins.items())}
    return {"n_pairs": len(seeds), "paired_drop": float(np.mean(diffs)),
            "by_bin": by_bin}


def _w13_model(results_dir: Path, model: str) -> dict:
    """Per-model necessity table. For each axis: project-out mean bet vs the
    no-hook baseline and the random-direction project-out null band, plus the
    matched-pairs drop. Verdict per axis NECESSARY (mean bet drops below
    the null band AND the matched-pairs drop is positive) / NOT_NECESSARY.
    The KEY §4 pattern = behavioural NECESSARY while confound + readout NOT."""
    base_rows = _w13_arm_rows(results_dir, f"sec4_w13_{model}_base")
    base_bets = _w13_bets(base_rows)
    base_bet = float(np.mean(base_bets)) if base_bets else float("nan")

    # random-direction project-out null band: mean bet under removing a RANDOM
    # axis (should leave betting near baseline). mean +/- 2sd, floored.
    null_means, null_drops = [], []
    for fp in sorted(Path(results_dir).glob(f"sec4_w13_{model}_null*.jsonl")):
        rows = _rows(fp)
        bets = _w13_bets(rows)
        if bets and (len(rows) == 0 or (len([r for r in rows if r.get("parse_ok")])
                                        / len(rows)) >= W13_PARSE_GATE):
            null_means.append(float(np.mean(bets)))
            null_drops.append(_w13_paired_drop(base_rows, rows)["paired_drop"])
    null_mean = float(np.mean(null_means)) if null_means else base_bet
    null_sd = float(np.std(null_means)) if len(null_means) > 1 else 0.0
    null_delta = max(2.0 * null_sd, W13_NULL_FLOOR)
    # the largest random-null paired drop that still counts as "no necessity":
    # a real drop must exceed BOTH the mean-bet band AND the null paired-drop.
    null_drop_ceiling = (max([d for d in null_drops if np.isfinite(d)],
                             default=0.0))

    axes_out = {}
    for axis in W13_AXES:
        rows = _w13_arm_rows(results_dir, f"sec4_w13_{model}_{axis}")
        bets = _w13_bets(rows)
        parse_rate = (len([r for r in rows if r.get("parse_ok")]) / len(rows)
                      if rows else float("nan"))
        proj_bet = float(np.mean(bets)) if bets else float("nan")
        # manip check: project-out drives the post-removal projection to ~0.
        projs = [p for r in rows if (p := _manip_proj(r)) is not None]
        pair = _w13_paired_drop(base_rows, rows)
        # NECESSARY: project-out lowers betting below the null band (proj_bet <
        # null_mean - delta) AND the matched-pairs drop clears the random-null
        # paired-drop ceiling (positive within-balance effect, balance held fixed
        # per pair by construction).
        below_band = bool(np.isfinite(proj_bet)
                          and proj_bet < null_mean - null_delta)
        paired_ok = bool(np.isfinite(pair["paired_drop"])
                         and pair["paired_drop"] > null_drop_ceiling
                         and pair["paired_drop"] > W13_NULL_FLOOR)
        necessary = bool(below_band and paired_ok
                         and parse_rate >= W13_PARSE_GATE)
        axes_out[axis] = {
            "verdict": "NECESSARY" if necessary else "NOT_NECESSARY",
            "necessary": necessary,
            "proj_bet": proj_bet, "parse_rate": parse_rate,
            "below_null_band": below_band, "paired_drop_clears_null": paired_ok,
            "paired": pair,
            "manip_proj": (float(np.mean(projs)) if projs else float("nan")),
            "n": len(rows),
        }

    behav = axes_out["behavioural"]["necessary"]
    conf = axes_out["confound"]["necessary"]
    read = axes_out["readout"]["necessary"]
    if behav and not conf and not read:
        verdict = "BEHAVIOURAL_NECESSARY_AND_PURE"     # the target §4 pattern
    elif behav and (conf or read):
        verdict = "NECESSARY_BUT_IMPURE"               # confound/readout also drops
    elif not behav:
        verdict = "BEHAVIOURAL_NOT_NECESSARY"          # project-out did not lower
    else:
        verdict = "AMBIGUOUS"
    return {
        "verdict": verdict,
        "axes": axes_out,
        "baseline_bet": base_bet,
        "null_band": {"mean": null_mean, "sd": null_sd, "delta": null_delta,
                      "n": len(null_means),
                      "paired_drop_ceiling": null_drop_ceiling},
        "parse_gate": W13_PARSE_GATE,
    }


def analyze_w13(results_dir=None, out_json: Optional[Path] = None,
                out_png: Optional[Path] = None) -> dict:
    """§4 NECESSITY (project-out) analysis, per model. For each model it returns
    the per-axis necessity table (project-out mean bet vs the no-hook baseline
    and the random-direction project-out null band, plus the matched-pairs
    drop), a per-axis verdict NECESSARY / NOT_NECESSARY, and a model-level
    verdict whose target value is BEHAVIOURAL_NECESSARY_AND_PURE (behavioural
    project-out drops betting while confound + readout project-out do not)."""
    results_dir = Path(results_dir) if results_dir is not None \
        else W13_RESULTS_DIR
    models = {m: _w13_model(results_dir, m) for m in W13_MODELS}
    result = {
        "models": models,
        "verdicts": {m: models[m]["verdict"] for m in W13_MODELS},
        "necessity_table": {
            m: {a: models[m]["axes"][a]["verdict"] for a in W13_AXES}
            for m in W13_MODELS},
    }
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
    if out_png is not None:
        try:
            make_figure_w13(result, Path(results_dir), Path(out_png))
        except Exception as e:
            print(f"[sec4-w13] figure skipped: {e}")
    return result


def make_figure_w13(res: dict, results_dir: Path, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, model in zip(axs, W13_MODELS):
        m = res["models"][model]
        base = m["baseline_bet"]
        labels = ["baseline"] + list(W13_AXES)
        vals = [base] + [m["axes"][a]["proj_bet"] for a in W13_AXES]
        colors = ["k"] + ["C0" if m["axes"][a]["necessary"] else "0.6"
                          for a in W13_AXES]
        ax.bar(range(len(vals)), vals, color=colors)
        nb = m["null_band"]
        if np.isfinite(nb["mean"]):
            ax.axhspan(nb["mean"] - nb["delta"], nb["mean"] + nb["delta"],
                       color="C1", alpha=0.2, label="random-null band")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_ylabel("mean bet ratio (project-out)")
        ax.set_title(f"{model} — {m['verdict']}")
        ax.legend(fontsize=8)
    fig.suptitle("§4 Wave-13 NECESSITY (project-out): behavioural should DROP")
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
    ap.add_argument("--wave3", action="store_true",
                    help="run the Wave-3 adjudication analysis (Q2/Q3/Q1-rung2)")
    ap.add_argument("--wave6", action="store_true",
                    help="run the Wave-6 MW spin/stop adjudication (mw_rc "
                         "ladder vs the reused sec4_w4 MW nulls/baseline)")
    ap.add_argument("--wave10", action="store_true",
                    help="run the Wave-10 llama §4.1 read-vs-write dissociation "
                         "+ §4.3 condition modulation analysis (results/sec4_w10)")
    ap.add_argument("--w11", action="store_true",
                    help="run the Wave-11 llama §4.2 task-own write-window scan "
                         "(IC ic_rc + MW mw_rc dose-response per window vs the "
                         "random-null band; picks each task's window)")
    ap.add_argument("--w13", action="store_true",
                    help="run the Wave-13 §4 NECESSITY (project-out) analysis "
                         "per model: behavioural/confound/readout project-out "
                         "mean bet vs baseline + random-null band, balance-"
                         "matched paired drop, per-axis NECESSARY verdict "
                         "(results/sec4_w13)")
    ap.add_argument("--w1-results-dir", default=str(RESULTS_DIR),
                    help="wave3: dir with the Wave-1 sec4_behavioural_* ladder "
                         "(fetched from the sec4_p0 HF checkpoints if absent)")
    ap.add_argument("--w2-results-dir",
                    default=str(_MLC / "results" / "sec4_w2"),
                    help="wave3: dir with the Wave-2 thick SM null rollouts")
    ap.add_argument("--w4-results-dir",
                    default=str(_MLC / "results" / "sec4_w4"),
                    help="wave6: dir with the sec4_w4 MW null/baseline "
                         "rollouts (fetched from HF if absent)")
    args = ap.parse_args()
    if args.w13:
        w13_dir = W13_RESULTS_DIR
        if args.results_dir == str(RESULTS_DIR):
            args.results_dir = str(w13_dir)
        out_json = (str(w13_dir / "sec4_w13_analysis.json")
                    if args.out_json == str(OUT_JSON) else args.out_json)
        out_png = (str(w13_dir / "sec4_w13_analysis.png")
                   if args.out_png == str(OUT_PNG) else args.out_png)
        res = analyze_w13(Path(args.results_dir), Path(out_json), Path(out_png))
        print(json.dumps({"verdicts": res["verdicts"],
                          "necessity_table": res["necessity_table"]}, indent=2))
        return
    if args.w11:
        # untouched defaults point at the Wave-1 output paths — redirect them to
        # the wave11 namespace so --w11 never overwrites the p0 analysis.
        w11_dir = W11_RESULTS_DIR
        if args.results_dir == str(RESULTS_DIR):
            args.results_dir = str(w11_dir)
        out_json = (str(w11_dir / "sec4_w11_analysis.json")
                    if args.out_json == str(OUT_JSON) else args.out_json)
        out_png = (str(w11_dir / "sec4_w11_analysis.png")
                   if args.out_png == str(OUT_PNG) else args.out_png)
        res = analyze_w11(Path(args.results_dir), Path(out_json), Path(out_png))
        print(json.dumps({"verdicts": res["verdicts"],
                          "chosen_windows": res["chosen_windows"],
                          "any_window_significant": res["any_window_significant"]},
                         indent=2))
        return
    if args.wave10:
        w10_dir = W10_RESULTS_DIR
        if args.results_dir == str(RESULTS_DIR):
            args.results_dir = str(w10_dir)
        out_json = (str(w10_dir / "sec4_w10_analysis.json")
                    if args.out_json == str(OUT_JSON) else args.out_json)
        res = analyze_w10(Path(args.results_dir), Path(args.assets_dir),
                          Path(out_json))
        print(json.dumps({"verdicts": res["verdicts"],
                          "sec41_null_band": res["sec41"]["null_band"],
                          "sec43_diffs": {k: v["diff"]
                                          for k, v in res["sec43"]["diffs"].items()}},
                         indent=2))
        return
    if args.wave6:
        # untouched defaults point at the Wave-1 output paths — redirect them
        # to the wave6 namespace so --wave6 never overwrites the p0 analysis.
        w6_dir = _MLC / "results" / "sec4_w6"
        if args.results_dir == str(RESULTS_DIR):
            args.results_dir = str(w6_dir)
        if args.out_json == str(OUT_JSON):
            args.out_json = str(w6_dir / "sec4_w6_analysis.json")
        if args.out_png == str(OUT_PNG):
            args.out_png = str(w6_dir / "sec4_w6_analysis.png")
        w4_dir = Path(args.w4_results_dir)
        if not list(w4_dir.glob(f"{W6_MW_NULL_PREFIX}*.jsonl")):
            try:  # pull the frozen Wave-4 MW nulls/baseline off HF — without
                # them the band degrades to the slope floor (n_null_slopes in
                # the output records the EFFECTIVE per-metric band count that
                # actually gated the verdict; n_null_directions is on-disk).
                n = fetch_hf_rollouts("sec4_w4", w4_dir, prefix="sec4_w4_mw_")
                print(f"[sec4-w6] fetched {n} Wave-4 MW rollouts -> {w4_dir}")
            except Exception as e:
                print(f"[sec4-w6] Wave-4 HF fetch skipped: {e}")
        res = analyze_wave6(Path(args.results_dir), w4_dir,
                            Path(args.out_json), Path(args.out_png))
        print(json.dumps({"verdict": res["verdict"],
                          "bet_corroborates": res["bet_corroborates"],
                          "spin_slope": res["spin"]["slope"],
                          "bet_slope": res["bet"]["slope"]}, indent=2))
        return
    if args.wave3:
        # untouched defaults point at the Wave-1 output paths — redirect them
        # to the wave3 namespace so --wave3 never overwrites the p0 analysis.
        w3_dir = _MLC / "results" / "sec4_w3"
        if args.results_dir == str(RESULTS_DIR):
            args.results_dir = str(w3_dir)
        if args.out_json == str(OUT_JSON):
            args.out_json = str(w3_dir / "sec4_w3_analysis.json")
        if args.out_png == str(OUT_PNG):
            args.out_png = str(w3_dir / "sec4_w3_analysis.png")
        w1_dir = Path(args.w1_results_dir)
        if not list(w1_dir.glob(f"{W1_BEHAV_PREFIX}a*.jsonl")):
            try:  # pull the frozen Wave-1 ladder off HF (skipped offline)
                n = fetch_hf_rollouts("sec4_p0", w1_dir)
                print(f"[sec4-w3] fetched {n} Wave-1 rollouts -> {w1_dir}")
            except Exception as e:
                print(f"[sec4-w3] Wave-1 HF fetch skipped: {e}")
        w2_dir = Path(args.w2_results_dir)
        if not list(w2_dir.glob("sec4_w2_null*.jsonl")):
            try:  # pull the Wave-2 rollouts (thick SM null) off HF likewise —
                # without them the Q2 sh3_sm gate silently degrades to the
                # slope-floor fallback (q2.sm_null_source records which ran).
                n = fetch_hf_rollouts("sec4_w2", w2_dir)
                print(f"[sec4-w3] fetched {n} Wave-2 rollouts -> {w2_dir}")
            except Exception as e:
                print(f"[sec4-w3] Wave-2 HF fetch skipped: {e}")
        res = analyze_wave3(Path(args.results_dir), w1_dir, w2_dir,
                            Path(args.out_json), Path(args.out_png))
        print(json.dumps({"verdicts": res["verdicts"]}, indent=2))
        return
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
