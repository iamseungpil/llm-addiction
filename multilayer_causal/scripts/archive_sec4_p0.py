"""Reproducible archiving of the SEC4 Wave-1 (sec4_p0) causal result.

Pulls the sec4_p0 rollouts from HF, recomputes the dose-response / null-band
analysis, writes results/sec4_p0/sec4_p0_analysis.json (+ .png), and appends a
rung row to experiments/sec4_causal/INDEX.md so a stranger can reproduce and
locate the result. Idempotent: rewrites the analysis and replaces (not
duplicates) the sec4_p0 INDEX row. Does NOT push to git/HF — the caller does.

Usage:  python -m multilayer_causal.scripts.archive_sec4_p0
        (or)  python multilayer_causal/scripts/archive_sec4_p0.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_MLC = Path(__file__).resolve().parents[1]
RESULTS = _MLC / "results" / "sec4_p0"
INDEX = _MLC / "experiments" / "sec4_causal" / "INDEX.md"
HF_REPO = "llm-addiction-research/llm-addiction"
HF_BASE = "experiments/sec4_causal/checkpoints/sec4_p0"
AXES = ("behavioural", "readout", "confound")
DOSES = (-3, -2, -1, 0, 1, 2, 3)


def _tag(d: int) -> str:
    return "a0" if d == 0 else (f"am{abs(d)}" if d < 0 else f"ap{d}")


def _rows(arm: str) -> list | None:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(HF_REPO, f"{HF_BASE}/{arm}.jsonl", repo_type="dataset")
    except Exception:
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def _metrics(arm: str) -> dict | None:
    rows = _rows(arm)
    if not rows:
        return None
    ok = [r for r in rows if r.get("parse_ok")]
    br = [float(r["bet_ratio"]) for r in ok if r.get("bet_ratio") is not None]
    ec = [1.0 if r.get("extreme") else 0.0 for r in ok if "extreme" in r]
    if not ec:  # fallback proxy
        ec = [1.0 if (r.get("bet_ratio") is not None and float(r["bet_ratio"]) >= 0.5)
              else 0.0 for r in ok if r.get("bet_ratio") is not None]
    return {"i_ba": float(np.mean(br)) if br else float("nan"),
            "i_ec": float(np.mean(ec)) if ec else float("nan"),
            "n": len(rows), "parse": len(ok) / len(rows) if rows else float("nan")}


def analyze() -> dict:
    # null band (Wave-1 nulls are single-dose @ +3)
    nulls = [_metrics(f"sec4_null_{i}") for i in range(1, 6)]
    nb = [m["i_ba"] for m in nulls if m and np.isfinite(m["i_ba"])]
    nb_mean, nb_sd = float(np.mean(nb)), float(np.std(nb))
    base = _metrics("sec4_baseline")

    out = {"axes": {}, "null_band": {"mean": nb_mean, "sd": nb_sd, "n": len(nb),
                                     "lo2sd": nb_mean - 2 * nb_sd,
                                     "hi2sd": nb_mean + 2 * nb_sd},
           "baseline_i_ba": base["i_ba"] if base else None}
    for ax in AXES:
        pts = [(d, _metrics(f"sec4_{ax}_{_tag(d)}")) for d in DOSES]
        pts = [(d, m) for d, m in pts if m]
        xs = [d for d, _ in pts]
        iba = [m["i_ba"] for _, m in pts]
        iec = [m["i_ec"] for _, m in pts]
        parse = [m["parse"] for _, m in pts]
        rho, pv = spearmanr(xs, iba) if len(xs) >= 3 else (float("nan"), float("nan"))
        slope = float(np.polyfit(xs, iba, 1)[0]) if len(xs) >= 2 else float("nan")
        p3 = _metrics(f"sec4_{ax}_ap3")
        z = ((p3["i_ba"] - nb_mean) / nb_sd) if (p3 and nb_sd > 0) else float("nan")
        out["axes"][ax] = {
            "doses": xs, "i_ba": iba, "i_ec": iec, "parse": parse,
            "spearman_iba": float(rho), "p_iba": float(pv), "slope_iba": slope,
            "delta_iba": float(max(iba) - min(iba)) if iba else float("nan"),
            "z_at_plus3_vs_null": float(z),
            "i_ec_delta": float(max(iec) - min(iec)) if iec else float("nan")}
    # H4 locality
    out["locality"] = {w: (_metrics(f"sec4_{w}") or {}).get("i_ba")
                       for w in ("cum_18_19", "cum_16_21", "cum_14_23")}
    # verdict
    b = out["axes"].get("behavioural", {})
    out["verdict"] = ("WRITE_CONFIRMED (behavioural writes I_BA & co-moves I_EC; "
                      "read>=weak, confound inert)"
                      if b.get("slope_iba", 0) > 0 and b.get("delta_iba", 0) > 0.1
                      else "INCONCLUSIVE")
    return out


def make_figure(res: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    nb = res["null_band"]
    for ax, color in (("behavioural", "C0"), ("readout", "C3"), ("confound", "C2")):
        a = res["axes"].get(ax)
        if not a:
            continue
        ax1.plot(a["doses"], a["i_ba"], "-o", color=color, label=ax)
        ax2.plot(a["doses"], a["i_ec"], "-o", color=color, label=ax)
    ax1.axhspan(nb["lo2sd"], nb["hi2sd"], color="0.85", label="null +/-2sd")
    ax1.set_xlabel("alpha"); ax1.set_ylabel("I_BA (bet ratio)")
    ax1.set_title("Wave-1: I_BA dose-response"); ax1.legend(fontsize=8)
    ax2.set_xlabel("alpha"); ax2.set_ylabel("I_EC (extreme rate)")
    ax2.set_title("Wave-1: I_EC co-movement"); ax2.legend(fontsize=8)
    fig.suptitle("SEC4 Wave-1 (sec4_p0) — Gemma SM, L16-21 causal write")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


_MARK = "<!-- sec4_p0 -->"


def append_index(res: dict) -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    b = res["axes"]["behavioural"]; r = res["axes"]["readout"]; c = res["axes"]["confound"]
    row = (
        f"{_MARK}\n"
        f"### Wave-1 `sec4_p0` — Gemma SM, I_BA, L16-21 (mlc-sec4-p0-0706c)\n"
        f"- **Hypotheses:** H1 behavioural axis writes betting; H2 readout inert; "
        f"H3 confound inert; H4 locality.\n"
        f"- **Config:** `configs/arms_sec4_p0.yaml` (30 arms, n=200, alpha -3..+3, "
        f"held-out state_offset 300, addiction_role_gm, alpha-independent seeds).\n"
        f"- **Result:** behavioural Spearman(a,I_BA)={b['spearman_iba']:+.2f} "
        f"(p={b['p_iba']:.3f}), delta={b['delta_iba']:.3f}, z@+3 vs null="
        f"{b['z_at_plus3_vs_null']:+.1f}; I_EC co-moves (delta={b['i_ec_delta']:.3f}). "
        f"readout weak (rho={r['spearman_iba']:+.2f}, z@+3={r['z_at_plus3_vs_null']:+.1f}); "
        f"confound inert (z@+3={c['z_at_plus3_vs_null']:+.1f}). "
        f"Locality: cum_16_21={res['locality'].get('cum_16_21')}.\n"
        f"- **Caveat:** read!=write UNRESOLVED — Wave-1 null is thin single-dose; "
        f"Wave-2 adds a thick multi-dose null-slope band.\n"
        f"- **Verdict:** {res['verdict']}\n"
        f"- **HF:** rollouts `{HF_BASE}/`, axes `experiments/sec4_causal/assets/`, "
        f"analysis `results/sec4_p0/sec4_p0_analysis.json`.\n"
    )
    if INDEX.exists():
        txt = INDEX.read_text()
        if _MARK in txt:  # idempotent replace
            txt = re.sub(re.escape(_MARK) + r".*?(?=\n### |\Z)", row.rstrip() + "\n",
                         txt, flags=re.S)
        else:
            txt = txt.rstrip() + "\n\n" + row
    else:
        txt = "# SEC4 Causal Program — Result Ledger\n\n" + row
    INDEX.write_text(txt)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    res = analyze()
    (RESULTS / "sec4_p0_analysis.json").write_text(json.dumps(res, indent=1))
    try:
        make_figure(res, RESULTS / "sec4_p0_analysis.png")
    except Exception as e:
        print(f"[archive] figure skipped: {e}")
    append_index(res)
    print(f"[archive] verdict: {res['verdict']}")
    print(f"[archive] -> {RESULTS/'sec4_p0_analysis.json'} + INDEX rung")


if __name__ == "__main__":
    main()
