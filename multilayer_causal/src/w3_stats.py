"""W3 discovery-wave analysis — endpoints exactly as RUN_PLAN_W3.md.

Gemma arms (offset 300, seeds 2000042) compare against the W2 anchors
(w2_anchor_minus/plus, same states+seeds). LLaMA arms compare against their
own n=878 anchors (offset 0, seeds 42, llama pool).

Sections:
  w3a   PR-1: saerd + saerdx dose trend on I_BA(bet-only) + mixed bet_ratio.
  w3bk  BK shared-axis steering: mixed bet_ratio + stop rate vs -G, dose
        across {-2,+2,+4}; balance discrimination (smres/smorth vs smbal);
        manipulation check from vector_log projections.
  w3pos harness anchor: I_BA axis reproduces the W2-scale effect?
  w3rnd 8-direction null band + rank of each treatment effect.
  w3m   +M twin patch per window vs -G anchor + crude natural +M yardstick.
  w3L   PR-2: anchor gate (n=878) then 8-window gap recovery; top-1 rule.

Usage:
  python -m multilayer_causal.src.w3_stats \
      --w3 /tmp/w3_ckpt --w2 /tmp/w2_ckpt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .retro_stats import (cluster_perm_test, tost_gap_recovery,
                          parse_excluded, DEFAULT_BEHAV_ROOT)
from .w1_stats import arm_summary, iba_rows, probe_rows, load, perm_test
from .w2_stats import build_pool_550, clusters300

GEMMA_STEER = ["w3a_saerd_m20", "w3a_saerd_p20", "w3a_saerd_p40",
               "w3a_saerdx_m20", "w3a_saerdx_p20", "w3a_saerdx_p40",
               "w3bk_sm_m20", "w3bk_sm_p20", "w3bk_sm_p40",
               "w3bk_smres_p40", "w3bk_smorth_p40", "w3bk_smbal_p40",
               "w3pos_iba_p40"] + [f"w3rnd_s{i}" for i in range(8)]
M_ARMS = ["w3m_0813", "w3m_1621", "w3m_2429"]
L_WINDOWS = ["w3L_w00", "w3L_w04", "w3L_w08", "w3L_w12",
             "w3L_w16", "w3L_w20", "w3L_w24", "w3L_w28"]


def mixed_vals(trials):
    """Mixed bet_ratio (stop counts as 0) on parse_ok rows — the M3/anchor
    yardstick metric used across W1-W3."""
    return [float(t["bet_ratio"]) for t in parse_excluded(trials)]


def build_llama_pool(n=928):
    from .states import ensure_sm_catalog, load_minusG_states
    os.environ.setdefault("LLM_ADDICTION_BEHAVIORAL_ROOT",
                          str(DEFAULT_BEHAV_ROOT))
    ensure_sm_catalog("llama")
    pool = load_minusG_states("llama", n=n)
    gid = {}
    for i, (game, _) in enumerate(pool):
        gid.setdefault(id(game), i)
    return pool, gid


def clusters0(trials, pool, gid):
    return [gid[id(pool[t["trial_id"] % len(pool)][0])] for t in trials]


def vs_minus(trials, minus_ok, mv, mc, pool, gid, cluster_fn):
    ok = parse_excluded(trials)
    tv = [float(t["bet_ratio"]) for t in ok]
    tc = cluster_fn(ok, pool, gid)
    d, p = perm_test(tv, tc, mv, mc)
    sv = [float(t["action"] == "stop") for t in ok]
    smv = [float(t["action"] == "stop") for t in minus_ok]
    ds, ps = perm_test(sv, tc, smv, mc)
    return {"bet_diff": d, "bet_p": p, "stop_diff": ds, "stop_p": ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w3", default="/tmp/w3_ckpt")
    ap.add_argument("--w2", default="/tmp/w2_ckpt")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "out" / "w3_stats.json"))
    args = ap.parse_args()
    pool, gid = build_pool_550()
    R = {}

    minus = load(args.w2, "w2_anchor_minus")
    plus = load(args.w2, "w2_anchor_plus")
    mok, pok = parse_excluded(minus), parse_excluded(plus)
    mv, mc = mixed_vals(minus), clusters300(mok, pool, gid)
    pv, pc = mixed_vals(plus), clusters300(pok, pool, gid)

    # ---- per-arm summaries + vs -G anchor --------------------------------
    arms = {}
    for arm in GEMMA_STEER + M_ARMS:
        t = load(args.w3, arm)
        e = arm_summary(t)
        e.update(vs_minus(t, mok, mv, mc, pool, gid, clusters300))
        ok = parse_excluded(t)
        if ok and len(ok) >= 4:
            tv = [float(x["bet_ratio"]) for x in ok]
            tc = clusters300(ok, pool, gid)
            e["gap"] = tost_gap_recovery(tv, pv, mv, treat_clusters=tc,
                                         plus_clusters=pc, minus_clusters=mc)
        vl = [t_["vector_log"]["proj"] for t_ in t if t_.get("vector_log")]
        if vl:
            e["proj_mean"] = float(np.mean(vl))
        arms[arm] = e
    R["arms"] = arms

    # ---- w3a PR-1 trend ---------------------------------------------------
    def trend(prefix):
        xs, out = [(-2, "m20"), (2, "p20"), (4, "p40")], []
        for a, sfx in xs:
            e = arms[f"{prefix}_{sfx}"]
            out.append({"alpha": a, "iba": e["iba"], "bet_diff": e["bet_diff"],
                        "bet_p": e["bet_p"], "parse": e["parse"]})
        return out
    R["w3a"] = {"saerd": trend("w3a_saerd"), "saerdx": trend("w3a_saerdx")}

    # ---- w3rnd null band + ranks -----------------------------------------
    rnd_effects = [arms[f"w3rnd_s{i}"]["bet_diff"] for i in range(8)]
    rnd_stop = [arms[f"w3rnd_s{i}"]["stop_diff"] for i in range(8)]
    R["w3rnd"] = {"bet_diff_band": [float(np.min(rnd_effects)),
                                    float(np.max(rnd_effects))],
                  "stop_diff_band": [float(np.min(rnd_stop)),
                                     float(np.max(rnd_stop))]}

    def rank_among_rnd(val, band_vals):
        return int(1 + sum(abs(v) >= abs(val) for v in band_vals))

    for arm in ("w3bk_sm_p40", "w3bk_smres_p40", "w3bk_smorth_p40",
                "w3bk_smbal_p40", "w3a_saerd_p40", "w3a_saerdx_p40",
                "w3pos_iba_p40"):
        arms[arm]["rank_vs_8rnd_bet"] = rank_among_rnd(
            arms[arm]["bet_diff"], rnd_effects)
        arms[arm]["rank_vs_8rnd_stop"] = rank_among_rnd(
            arms[arm]["stop_diff"], rnd_stop)

    # ---- w3m: natural +M yardstick (crude corpus mean) -------------------
    from .states import ensure_sm_catalog
    os.environ.setdefault("LLM_ADDICTION_BEHAVIORAL_ROOT",
                          str(DEFAULT_BEHAV_ROOT))
    cat_dir = Path(ensure_sm_catalog("gemma"))
    cat_file = cat_dir if cat_dir.is_file() else next(cat_dir.glob("*.json"))
    cat = json.load(open(cat_file))
    m_rows = []
    for g in cat["results"]:
        if g.get("bet_type") != "variable" or g.get("prompt_combo") != "M":
            continue
        for d in g.get("decisions", []):
            bal = d.get("balance_before")
            if not bal:
                continue
            m_rows.append((d.get("bet", 0) or 0) / bal
                          if d.get("action") == "bet" else 0.0)
    R["natural_plusM_mixed"] = {"mean": float(np.mean(m_rows)),
                                "n_rows": len(m_rows),
                                "note": "crude corpus mean, all rounds"}

    # ---- w3L PR-2 ---------------------------------------------------------
    lpool, lgid = build_llama_pool()
    lm = load(args.w3, "w3L_anchor_minus")
    lp = load(args.w3, "w3L_anchor_plus")
    lmok, lpok = parse_excluded(lm), parse_excluded(lp)
    lmv, lmc = mixed_vals(lm), clusters0(lmok, lpool, lgid)
    lpv, lpc = mixed_vals(lp), clusters0(lpok, lpool, lgid)
    d, p = perm_test(lpv, lpc, lmv, lmc)
    gate = {"minus": arm_summary(lm), "plus": arm_summary(lp),
            "gap": d, "p": p, "pass": p < 0.05}
    lw = {"gate": gate}
    if gate["pass"]:
        for arm in L_WINDOWS:
            t = load(args.w3, arm)
            e = arm_summary(t)
            ok = parse_excluded(t)
            tv = [float(x["bet_ratio"]) for x in ok]
            tc = clusters0(ok, lpool, lgid)
            e["gap"] = tost_gap_recovery(tv, lpv, lmv, treat_clusters=tc,
                                         plus_clusters=lpc, minus_clusters=lmc)
            dd, pp = perm_test(tv, tc, lmv, lmc)
            e["vs_minus"] = {"diff": dd, "p": pp}
            lw[arm] = e
        ranked = sorted(L_WINDOWS, key=lambda a: -lw[a]["gap"]["recovery"])
        lw["top1_by_recovery"] = ranked[0]
    R["w3L"] = lw

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
