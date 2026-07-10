"""W2 confirmatory analysis — pre-registered endpoints from RUN_PLAN_W1.md §W2.

Differences from w1_stats: pool is n=550 and every w2 trial maps to state
pool[(trial_id + 300) % 550] (state_offset 300, NOT the phase-keyed offset
retro_stats._state_offset knows about), anchors are w2_anchor_minus/plus.

Promotion rule per arm (pre-registered): confirmatory p<0.05 + separation
from random + parse>=0.9 + gap recovery with cluster-bootstrap CI reported.

Usage:
  python -m multilayer_causal.src.w2_stats --ckpt-dir /tmp/w2_ckpt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .retro_stats import (cluster_perm_test, tost_gap_recovery,
                          parse_excluded, DEFAULT_BEHAV_ROOT)
from .w1_stats import (arm_summary, iba_rows, parse_rate, probe_rows,
                       load, perm_test)

POOL_N = 550          # offset 300 + n 200 + 50 slack (runner contract)
OFFSET = 300


def build_pool_550():
    from .states import ensure_sm_catalog, load_minusG_states
    os.environ.setdefault("LLM_ADDICTION_BEHAVIORAL_ROOT",
                          str(DEFAULT_BEHAV_ROOT))
    ensure_sm_catalog("gemma")
    pool = load_minusG_states("gemma", n=POOL_N)
    assert len(pool) == POOL_N, f"pool {len(pool)} != {POOL_N}"
    gid = {}
    for i, (game, _) in enumerate(pool):
        gid.setdefault(id(game), i)
    return pool, gid


def clusters300(trials, pool, gid):
    """w2 cluster labels: every arm in this wave ran at state_offset 300."""
    return [gid[id(pool[(t["trial_id"] + OFFSET) % len(pool)][0])]
            for t in trials]


def perm300(a_rows, b_rows, key, pool, gid, one_sided=False):
    return perm_test([r[key] for r in a_rows], clusters300(a_rows, pool, gid),
                     [r[key] for r in b_rows], clusters300(b_rows, pool, gid),
                     one_sided=one_sided)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/tmp/w2_ckpt")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "out" / "w2_stats.json"))
    args = ap.parse_args()
    pool, gid = build_pool_550()
    R = {}

    minus = load(args.ckpt_dir, "w2_anchor_minus")
    plus = load(args.ckpt_dir, "w2_anchor_plus")
    R["anchors"] = {"minus": arm_summary(minus), "plus": arm_summary(plus)}

    # ---- R7 gate re-check on held-out states ----------------------------
    pm, pp = probe_rows(minus), probe_rows(plus)
    d, p = perm300(pp, pm, "ilc_loss", pool, gid, one_sided=True)
    R["gate_r7"] = {"probe_n": [len(pp), len(pm)],
                    "ilc_loss_plus": float(np.mean([r["ilc_loss"] for r in pp])),
                    "ilc_loss_minus": float(np.mean([r["ilc_loss"] for r in pm])),
                    "diff": d, "one_sided_p": p,
                    "pass": d > 0 and p < 0.05}

    # ---- w2e confirmatory gap recovery -----------------------------------
    mok, pok = parse_excluded(minus), parse_excluded(plus)
    mv, mc = [t["bet_ratio"] for t in mok], clusters300(mok, pool, gid)
    pv, pc = [t["bet_ratio"] for t in pok], clusters300(pok, pool, gid)
    for arm in ("w2e_1617", "w2e_1821"):
        t = load(args.ckpt_dir, arm)
        ok = parse_excluded(t)
        tv, tc = [x["bet_ratio"] for x in ok], clusters300(ok, pool, gid)
        e = arm_summary(t)
        e["gap"] = tost_gap_recovery(tv, pv, mv, treat_clusters=tc,
                                     plus_clusters=pc, minus_clusters=mc)
        dd, p2 = perm_test(tv, tc, mv, mc)
        e["vs_minus"] = {"diff": dd, "p": p2}
        e["confirmed"] = bool(e["gap"]["recovery"] >= 0.5 and p2 < 0.05
                              and e["parse"] >= 0.9)
        R[arm] = e

    # ---- probe steering arms (the "one axis, all three indicators" test) -
    for arm in ("w2lc_iba_p40", "w2lc_ilc_p20", "w2rnd_p40"):
        t = load(args.ckpt_dir, arm)
        pr = probe_rows(t)
        e = {"stage1": arm_summary(t), "probe_n": len(pr)}
        b, mb = iba_rows(t), iba_rows(minus)
        d, p = perm_test([x["bet_ratio"] for x in b], clusters300(b, pool, gid),
                         [x["bet_ratio"] for x in mb],
                         clusters300(mb, pool, gid))
        e["iba_vs_minus"] = {"diff": d, "p": p}
        ok = parse_excluded(t)
        d, p = perm_test([float(x["extreme"]) for x in ok],
                         clusters300(ok, pool, gid),
                         [float(x["extreme"]) for x in mok],
                         clusters300(mok, pool, gid))
        e["iec_vs_minus"] = {"diff": d, "p": p}
        if pr:
            e["ilc_loss"] = float(np.mean([r["ilc_loss"] for r in pr]))
            e["spec"] = float(np.mean([r["spec"] for r in pr]))
            if len(pr) >= 4 and len(pm) >= 4:
                d, p = perm300(pr, pm, "spec", pool, gid, one_sided=True)
                e["spec_vs_minus"] = {"diff": d, "one_sided_p": p}
        R[arm] = e

    # ---- w2ro5: readout axis on the paper's 5-layer read grid ------------
    t = load(args.ckpt_dir, "w2ro5_p20")
    e = arm_summary(t)
    b = iba_rows(t)
    mb = iba_rows(minus)
    if len(b) >= 4:
        d, p = perm_test([x["bet_ratio"] for x in b], clusters300(b, pool, gid),
                         [x["bet_ratio"] for x in mb],
                         clusters300(mb, pool, gid))
        e["iba_vs_minus"] = {"diff": d, "p": p}
    R["w2ro5_p20"] = e

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
