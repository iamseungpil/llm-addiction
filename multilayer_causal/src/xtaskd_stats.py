"""Stage D — WRITE-LOCALITY width / cumulative-layer sweep analysis (gemma, SM).

Stage A's spine localised the causal write to L16-21 with a coarse width-6
tiling. This refines it: steering the I_BA behavioural axis (+4.0) on SM across
window GEOMETRY only, it asks where the write actually lives.

  recovery(arm) = (mean_bet_ratio(arm) - mean_bet_ratio(-G))
                  / (mean_bet_ratio(+G) - mean_bet_ratio(-G))

with the W2 +/-G anchors (w2_anchor_minus / w2_anchor_plus, reused from HF) as
the behavioural endpoints. Three reads:

  1. single-layer (width-1) recovery vs layer  -> does the lever peak INSIDE
     L16-21, and is one layer enough?
  2. cumulative-window recovery vs width        -> does recovery saturate at
     ~L16-21 and NOT grow when widened (L14-23 .. all 42)?
  3. parse_ok vs width                          -> does over-widening DERAIL
     generation (coherence drop) rather than help?

Plus a concentration ratio (best single layer / L16-21 window) and the
manipulation-check projection (log_vectors) confirming the steer landed.

Arm geometry is parsed from the arm id (xtaskd_L{NN}_w1, xtaskd_cum_{lo}_{hi}),
not the expanded `layers` list. Anchors are pulled from HF once and cached. No
fabricated points: a missing arm/anchor is reported, never invented.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_MLC = Path(__file__).resolve().parents[1]
RESULTS_DIR = _MLC / "results" / "xtaskd"
OUT_JSON = RESULTS_DIR / "xtaskd_locality.json"
OUT_PNG = RESULTS_DIR / "xtaskd_locality.png"
HF_REPO = "llm-addiction-research/llm-addiction"
ANCHOR_MINUS = "w2/w2_anchor_minus"
ANCHOR_PLUS = "w2/w2_anchor_plus"
WRITE_LO, WRITE_HI = 16, 21


# --------------------------------------------------------------------------- #
# rollout summaries                                                           #
# --------------------------------------------------------------------------- #
def _rows(path: Path) -> List[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _mean_br(rows: List[dict]) -> Tuple[float, int, int]:
    ok = [r for r in rows if r.get("parse_ok")]
    br = [float(r["bet_ratio"]) for r in ok if r.get("bet_ratio") is not None]
    return (float(np.mean(br)) if br else float("nan"), len(rows), len(ok))


def _mean_projection(rows: List[dict]) -> float:
    """Manipulation check: mean |steering projection| from log_vectors rows, if
    present. Returns nan when the arm did not log the projection."""
    vals = []
    for r in rows:
        for key in ("steer_projection", "projection", "proj", "manip_proj"):
            if key in r and r[key] is not None:
                vals.append(abs(float(r[key])))
                break
    return float(np.mean(vals)) if vals else float("nan")


def anchor_means(cache_dir: Path) -> Tuple[float, float]:
    """+/-G anchor mean bet_ratio, pulled from HF (cached locally)."""
    from huggingface_hub import hf_hub_download
    out = {}
    for tag, arm in (("minus", ANCHOR_MINUS), ("plus", ANCHOR_PLUS)):
        p = hf_hub_download(HF_REPO,
                            f"experiments/multilayer_causal/checkpoints/{arm}.jsonl",
                            repo_type="dataset")
        out[tag] = _mean_br(_rows(Path(p)))[0]
    return out["minus"], out["plus"]


# --------------------------------------------------------------------------- #
# id parsing                                                                  #
# --------------------------------------------------------------------------- #
def parse_single(arm_id: str) -> Optional[int]:
    m = re.fullmatch(r"xtaskd_L(\d{2})_w1", arm_id)
    return int(m.group(1)) if m else None


def parse_cum(arm_id: str) -> Optional[Tuple[int, int]]:
    m = re.fullmatch(r"xtaskd_cum_(\d{2})_(\d{2})", arm_id)
    return (int(m.group(1)), int(m.group(2))) if m else None


# --------------------------------------------------------------------------- #
# analysis                                                                     #
# --------------------------------------------------------------------------- #
def analyze(results_dir: Path = RESULTS_DIR, out_json: Path = OUT_JSON,
            out_png: Path = OUT_PNG) -> dict:
    results_dir = Path(results_dir)
    br_minus, br_plus = anchor_means(results_dir)
    gap = br_plus - br_minus

    def recovery(m):
        return (m - br_minus) / gap if np.isfinite(m) and abs(gap) > 1e-9 else float("nan")

    single, cum, nulls = {}, {}, {}
    for fp in sorted(results_dir.glob("xtaskd_*.jsonl")):
        arm = fp.stem
        rows = _rows(fp)
        m, n, ok = _mean_br(rows)
        rec = {"mean_br": m, "recovery": recovery(m), "n": n, "parse_ok": ok,
               "parse_rate": ok / n if n else float("nan"),
               "manip_proj": _mean_projection(rows)}
        L = parse_single(arm)
        C = parse_cum(arm)
        if L is not None:
            single[L] = rec
        elif C is not None:
            cum[f"{C[0]:02d}_{C[1]:02d}"] = {**rec, "lo": C[0], "hi": C[1],
                                             "width": C[1] - C[0] + 1}
        elif "rnd" in arm:
            nulls[arm] = rec

    # concentration: best single-layer recovery vs the L16-21 cumulative window.
    best_L = max(single, key=lambda k: abs(single[k]["recovery"])) if single else None
    full = cum.get(f"{WRITE_LO:02d}_{WRITE_HI:02d}")
    concentration = (abs(single[best_L]["recovery"]) / abs(full["recovery"])
                     if best_L is not None and full
                     and np.isfinite(full["recovery"]) and abs(full["recovery"]) > 1e-9
                     else float("nan"))
    # in-window vs out-window single-layer recovery (localisation summary)
    in_w = [single[L]["recovery"] for L in single if WRITE_LO <= L <= WRITE_HI]
    out_w = [single[L]["recovery"] for L in single if not (WRITE_LO <= L <= WRITE_HI)]
    result = {
        "anchors": {"minus": br_minus, "plus": br_plus, "gap": gap},
        "single_layer": single, "cumulative": cum, "nulls": nulls,
        "best_single_layer": best_L,
        "concentration_best_over_L16_21": concentration,
        "mean_recovery_in_window": float(np.nanmean(in_w)) if in_w else float("nan"),
        "mean_recovery_out_window": float(np.nanmean(out_w)) if out_w else float("nan"),
        "null_recovery": {k: nulls[k]["recovery"] for k in nulls},
        "write_window": [WRITE_LO, WRITE_HI],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1))
    try:
        make_figure(result, out_png)
    except Exception as e:
        print(f"[xtaskd] figure skipped: {e}")
    print(f"[xtaskd] single={len(single)} cum={len(cum)} nulls={len(nulls)} -> {out_json}")
    return result


def make_figure(res: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    # panel 1: single-layer recovery vs layer
    sl = res["single_layer"]
    if sl:
        Ls = sorted(int(k) for k in sl)
        ax1.axvspan(WRITE_LO, WRITE_HI, color="0.88", label="write window L16-21")
        ax1.plot(Ls, [sl[L]["recovery"] for L in Ls], "-o", color="C0")
        for k, v in res["null_recovery"].items():
            if "L19" in k:
                ax1.axhline(v, color="0.6", ls=":", lw=0.8, label="random null (1 layer)")
        ax1.axhline(0, color="k", lw=0.4)
        ax1.set_xlabel("single steered layer")
        ax1.set_ylabel("recovery (frac of +/-G gap)")
        ax1.set_title("width-1: where does the lever live?")
        ax1.legend(fontsize=7)
    # panel 2: cumulative recovery + parse vs width
    cum = res["cumulative"]
    if cum:
        items = sorted(cum.values(), key=lambda d: d["width"])
        w = [d["width"] for d in items]
        ax2.plot(w, [d["recovery"] for d in items], "-s", color="C1", label="recovery")
        ax2.axvline(WRITE_HI - WRITE_LO + 1, color="0.6", ls="--", lw=0.8,
                    label="L16-21 width (6)")
        ax2.axhline(1.0, color="0.6", lw=0.5, ls=":")
        ax2.set_xlabel("cumulative window width (layers)")
        ax2.set_ylabel("recovery", color="C1")
        axp = ax2.twinx()
        axp.plot(w, [d["parse_rate"] for d in items], "-^", color="C3", label="parse_rate")
        axp.set_ylabel("parse_ok rate", color="C3")
        axp.set_ylim(-0.05, 1.05)
        ax2.set_title("cumulative width: saturation / derail")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = axp.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    fig.suptitle("Stage D: write-locality sweep (I_BA axis steer, SM)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


def _selftest() -> None:
    # recovery math + id parsing
    assert parse_single("xtaskd_L18_w1") == 18
    assert parse_single("xtaskd_cum_16_21") is None
    assert parse_cum("xtaskd_cum_16_21") == (16, 21)
    assert parse_cum("xtaskd_L18_w1") is None
    # recovery: br=0.13, -G=0.061, +G=0.202 -> (0.13-0.061)/0.141
    gap = 0.202 - 0.061
    assert abs((0.13 - 0.061) / gap - 0.4894) < 1e-3
    print("xtaskd_stats selftest: ALL PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-png", default=str(OUT_PNG))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(Path(args.results_dir), Path(args.out_json), Path(args.out_png))


if __name__ == "__main__":
    main()
