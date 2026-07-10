"""W1 discovery-wave analysis — primary endpoints exactly as RUN_PLAN_W1.md.

Reads downloaded checkpoints from --ckpt-dir (default /tmp/w1_ckpt), rebuilds
the deterministic -G pool for game-cluster labels (same alignment contract as
retro_stats), and reports per RUN_PLAN:

  w1ro  Spearman rho(alpha, I_BA bet-only) with within-cluster permutation;
        R5 gate: arms with parse<0.9 are dropped from the trend.
  w1lc  R7 gate first (anchor_plus loss-branch I_LC > anchor_minus, one-sided
        cluster perm); then per-arm loss-specificity vs anchor_minus.
  w1s   I_BA / I_EC / stop per arm vs anchor_minus (cluster perm) — matrix.
  w1ic  risky rate (choice in {3,4}, parse_ok) vs w1ic_anchor; clusters =
        IC game_id; plus Fisher exact.
  w1b/e +G gap recovery with cluster-bootstrap CI (tost_gap_recovery) against
        the fresh same-offset anchors w1lc_anchor_minus/plus (stage-1).

Usage:
  python -m multilayer_causal.src.w1_stats --ckpt-dir /tmp/w1_ckpt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .retro_stats import (cluster_perm_test, tost_gap_recovery,
                          parse_excluded, _build_pool, _clusters,
                          _state_offset)

RO_ARMS = [("w1ro_m20", -2.0), ("w1ro_a00", 0.0), ("w1ro_p20", 2.0),
           ("w1ro_p40", 4.0)]
LC_ARMS = ["w1lc_patch", "w1lc_iba_p40", "w1lc_ilc_p20", "w1lc_ilc_p40",
           "w1lc_rnd"]
S_ARMS = ["w1s_ilc_p20", "w1s_ilc_p40", "w1s_iec_p20", "w1s_iec_p40"]
IC_ARMS = ["w1ic_smiba_p20", "w1ic_smiba_p40", "w1ic_icax_p40",
           "w1ic_rnd_s0", "w1ic_rnd_s1"]
GAP_ARMS = ["w1b_patch5", "w1b_steer5",
            "w1e_1617", "w1e_1819", "w1e_2021", "w1e_2223", "w1e_2425",
            "w1e_1821", "w1e_2023"]


def load(ckpt_dir, arm):
    p = Path(ckpt_dir) / f"{arm}.jsonl"
    assert p.exists(), f"missing checkpoint {p}"
    return [json.loads(l) for l in open(p)]


def parse_rate(trials):
    return float(np.mean([t["parse_ok"] for t in trials]))


def iba_rows(trials):
    """Paper-definition I_BA rows: bet decisions only, parse_ok only."""
    return [t for t in parse_excluded(trials) if t["action"] == "bet"]


def arm_summary(trials):
    ok = parse_excluded(trials)
    bets = iba_rows(trials)
    return {
        "n": len(trials), "parse": parse_rate(trials),
        "stop": float(np.mean([t["action"] == "stop" for t in ok])) if ok else None,
        "iba": float(np.mean([t["bet_ratio"] for t in bets])) if bets else None,
        "iec": float(np.mean([t["extreme"] for t in ok])) if ok else None,
    }


# ----------------------------------------------------------------- w1ro

def ro_trend(rows, n_perm=10000, seed=0):
    """rows: list of (alpha, value, cluster). Spearman rho with a null built
    by permuting alpha labels WITHIN each cluster (paired design: every
    cluster contributes one trial per arm)."""
    alphas = np.array([r[0] for r in rows], float)
    vals = np.array([r[1] for r in rows], float)
    clusters = np.array([r[2] for r in rows])
    obs = stats.spearmanr(alphas, vals).statistic
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = alphas.copy()
        for c in np.unique(clusters):
            idx = np.where(clusters == c)[0]
            perm[idx] = rng.permutation(perm[idx])
        if abs(stats.spearmanr(perm, vals).statistic) >= abs(obs) - 1e-12:
            hits += 1
    return float(obs), (hits + 1) / (n_perm + 1)


# ----------------------------------------------------------------- w1lc

def probe_rows(trials):
    """Valid LC-probe rows: stage-1 bet (not stop/bankrupt), r_t>0, both
    branches parse_ok with a recorded ratio.  Returns per-trial dicts with
    ilc_loss (paper formula, loss branch) and specificity (LOSS - WIN)."""
    out = []
    for t in trials:
        p = t.get("probe") or {}
        if p.get("stage1_stop") or not t.get("parse_ok"):
            continue
        r_t = p.get("r_t")
        loss, win = p.get("loss") or {}, p.get("win") or {}
        if not r_t or not loss.get("parse_ok") or not win.get("parse_ok"):
            continue
        if loss.get("r") is None or win.get("r") is None:
            continue
        d_loss = (loss["r"] - r_t) / r_t
        d_win = (win["r"] - r_t) / r_t
        out.append({"trial_id": t["trial_id"], "arm": t["arm"],
                    "state_offset": _state_offset(t),
                    "ilc_loss": max(0.0, d_loss),
                    "spec": d_loss - d_win})
    return out


def perm_test(av, ac, bv, bc, one_sided=False, n_perm=10000):
    """diff of means + cluster permutation p (one-sided uses 'greater')."""
    diff = float(np.mean(av) - np.mean(bv))
    p = cluster_perm_test(av, ac, bv, bc, n_perm=n_perm,
                          alternative="greater" if one_sided else "two-sided")
    return diff, float(p)


def perm_rows(a_rows, b_rows, key, pool, gid, one_sided=False, n_perm=10000):
    return perm_test([r[key] for r in a_rows], _clusters(a_rows, pool, gid),
                     [r[key] for r in b_rows], _clusters(b_rows, pool, gid),
                     one_sided=one_sided, n_perm=n_perm)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/tmp/w1_ckpt")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "out" / "w1_stats.json"))
    args = ap.parse_args()
    pool, gid = _build_pool()
    R = {}

    minus = load(args.ckpt_dir, "w1lc_anchor_minus")
    plus = load(args.ckpt_dir, "w1lc_anchor_plus")
    R["anchors"] = {"minus": arm_summary(minus), "plus": arm_summary(plus)}

    # ---- w1ro -----------------------------------------------------------
    ro = {}
    rows = []
    for arm, alpha in RO_ARMS:
        t = load(args.ckpt_dir, arm)
        ro[arm] = {"alpha": alpha, **arm_summary(t)}
        if ro[arm]["parse"] >= 0.9:                                   # R5
            for x in iba_rows(t):
                c = _clusters([x], pool, gid)[0]
                rows.append((alpha, x["bet_ratio"], c))
        else:
            ro[arm]["excluded_r5"] = True
    alphas_in = sorted({a for a, _, _ in rows})
    if len(alphas_in) >= 3:
        rho, p = ro_trend(rows)
        ro["trend"] = {"alphas_used": alphas_in, "spearman_rho": rho,
                       "perm_p": p}
    else:
        ro["trend"] = {"alphas_used": alphas_in,
                       "note": "fewer than 3 valid alpha levels after R5"}
    R["w1ro"] = ro

    # ---- w1lc -----------------------------------------------------------
    lc = {}
    pm = probe_rows(minus)
    pp = probe_rows(plus)
    lc["anchor_minus"] = {"probe_n": len(pm),
                          "ilc_loss": float(np.mean([r["ilc_loss"] for r in pm])) if pm else None,
                          "spec": float(np.mean([r["spec"] for r in pm])) if pm else None}
    lc["anchor_plus"] = {"probe_n": len(pp),
                         "ilc_loss": float(np.mean([r["ilc_loss"] for r in pp])) if pp else None,
                         "spec": float(np.mean([r["spec"] for r in pp])) if pp else None}
    gate = None
    if len(pm) >= 4 and len(pp) >= 4:
        d, p = perm_rows(pp, pm, "ilc_loss", pool, gid, one_sided=True)
        gate = {"diff": d, "one_sided_p": p, "pass": d > 0 and p < 0.05}
    lc["gate_r7"] = gate or {"pass": False, "note": "too few valid probe rows"}
    for arm in LC_ARMS:
        t = load(args.ckpt_dir, arm)
        pr = probe_rows(t)
        e = {"stage1": arm_summary(t), "probe_n": len(pr)}
        if pr:
            e["ilc_loss"] = float(np.mean([r["ilc_loss"] for r in pr]))
            e["spec"] = float(np.mean([r["spec"] for r in pr]))
            if len(pr) >= 4 and len(pm) >= 4:
                d, p = perm_rows(pr, pm, "spec", pool, gid, one_sided=True)
                e["spec_vs_minus"] = {"diff": d, "one_sided_p": p}
        lc[arm] = e
    R["w1lc"] = lc

    # ---- w1s ------------------------------------------------------------
    sm = {}
    mb = iba_rows(minus)
    for arm in S_ARMS:
        t = load(args.ckpt_dir, arm)
        e = arm_summary(t)
        b = iba_rows(t)
        if len(b) >= 4 and len(mb) >= 4:
            av, ac = [x["bet_ratio"] for x in b], _clusters(b, pool, gid)
            bv, bc = [x["bet_ratio"] for x in mb], _clusters(mb, pool, gid)
            d, p = perm_test(av, ac, bv, bc)
            e["iba_vs_minus"] = {"diff": d, "p": p}
        ok, mok = parse_excluded(t), parse_excluded(minus)
        av, ac = [float(x["extreme"]) for x in ok], _clusters(ok, pool, gid)
        bv, bc = [float(x["extreme"]) for x in mok], _clusters(mok, pool, gid)
        d, p = perm_test(av, ac, bv, bc)
        e["iec_vs_minus"] = {"diff": d, "p": p}
        sm[arm] = e
    R["w1s"] = sm

    # ---- w1ic -----------------------------------------------------------
    ic = {}
    ica = load(args.ckpt_dir, "w1ic_anchor")

    def risky(trials):
        ok = [t for t in trials if t["parse_ok"]]
        return ok, (float(np.mean([t["risky"] for t in ok])) if ok else None)

    a_ok, a_rate = risky(ica)
    ic["w1ic_anchor"] = {"n": len(ica), "parse": parse_rate(ica),
                         "risky": a_rate}
    for arm in IC_ARMS:
        t = load(args.ckpt_dir, arm)
        ok, rate = risky(t)
        e = {"n": len(t), "parse": parse_rate(t), "risky": rate}
        if ok and a_ok:
            av = [float(x["risky"]) for x in ok]
            ac = [x["game_id"] for x in ok]
            bv = [float(x["risky"]) for x in a_ok]
            bc = [x["game_id"] for x in a_ok]
            d, p = perm_test(av, ac, bv, bc)
            k = [sum(x["risky"] for x in ok), sum(not x["risky"] for x in ok)]
            m = [sum(x["risky"] for x in a_ok),
                 sum(not x["risky"] for x in a_ok)]
            e["vs_anchor"] = {"diff": d, "cluster_perm_p": p,
                              "fisher_p": float(stats.fisher_exact(
                                  [k, m])[1])}
        ic[arm] = e
    R["w1ic"] = ic

    # ---- w1b / w1e ------------------------------------------------------
    gaps = {}
    pv = [t["bet_ratio"] for t in parse_excluded(plus)]
    pc = _clusters(parse_excluded(plus), pool, gid)
    mv = [t["bet_ratio"] for t in parse_excluded(minus)]
    mc = _clusters(parse_excluded(minus), pool, gid)
    for arm in GAP_ARMS:
        t = load(args.ckpt_dir, arm)
        e = arm_summary(t)
        ok = parse_excluded(t)
        if len(ok) >= 4:
            tv = [x["bet_ratio"] for x in ok]
            tc = _clusters(ok, pool, gid)
            e["gap"] = tost_gap_recovery(tv, pv, mv, treat_clusters=tc,
                                         plus_clusters=pc, minus_clusters=mc)
            d, p = perm_test(tv, tc, mv, mc)
            e["vs_minus"] = {"diff": d, "p": p}
        gaps[arm] = e
    R["w1be"] = gaps

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
