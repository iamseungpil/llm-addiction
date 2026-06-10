"""Retroactive paper-grade statistics over collected arm checkpoints (CPU).

Pure, unit-tested pieces: cluster_perm_test (rulebook R4), tost_gap_recovery
(R3), ilc_dec (I_LC^dec on displayed-loss states), parse_excluded (R5).
main() downloads the arm JSONLs from HF, rebuilds the deterministic −G state
pool for game-cluster labels, recomputes the headline comparisons, writes
multilayer_causal/out/retro_stats.json and prints a markdown summary.

Pool/trial alignment: runner.run_arm draws states[(i + state_offset) % len]
from load_minusG_states(n = n + offset + 50).  load_minusG_states shuffles
the FULL state list with Random(42) then truncates, so any n=350 pool is a
prefix-superset of every pool the arms actually used (max index 100+199=299).
state_offset is 100 for the e1c confirmatory phase and 0 everywhere else.

Usage:
  python -m multilayer_causal.src.retro_stats
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

HF_REPO = "llm-addiction-research/llm-addiction"
HF_BASE = "experiments/multilayer_causal/checkpoints"
PHASES = ("e1", "e1c", "e2", "e3a", "e3i")
POOL_N = 350
DEFAULT_BEHAV_ROOT = Path(__file__).resolve().parents[1] / "out" / "behav"
OUT_JSON = Path(__file__).resolve().parents[1] / "out" / "retro_stats.json"
# (name, treat, plus_anchor, minus_anchor) — headline re-tests
HEADLINE = [
    ("e1c_win18", "e1c_win18", "e1c_anchor_plus", "e1c_anchor_minus"),
    ("e1_win_18", "e1_win_18", "e1_anchor_plus", "e1_anchor_minus"),
    ("e1_cum_b22", "e1_cum_b22", "e1_anchor_plus", "e1_anchor_minus"),
]
E2_PRIMARY = ("e2_r1", "e2_r2", "e2_r8", "e2_r32", "e2_r128")


# ---------------------------------------------------------------- pure stats

def _cluster_groups(vals, clusters):
    """Group values by cluster label (order of first appearance)."""
    groups = {}
    for v, c in zip(vals, clusters):
        groups.setdefault(c, []).append(float(v))
    return [np.asarray(g) for g in groups.values()]


def cluster_perm_test(a_vals, a_clusters, b_vals, b_clusters, n_perm=10000,
                      alternative="two-sided", seed=0):
    """Permutation p for mean(a) − mean(b), relabeling WHOLE clusters (R4).

    Cluster (game-id) units are pooled across both groups and reassigned to
    group a / b keeping the observed cluster counts; the statistic is the
    plain difference of pooled means (cluster-size weighted).
    """
    assert len(a_vals) == len(a_clusters), "a vals/clusters length mismatch"
    assert len(b_vals) == len(b_clusters), "b vals/clusters length mismatch"
    assert alternative in ("two-sided", "greater"), alternative
    ga = _cluster_groups(a_vals, a_clusters)
    gb = _cluster_groups(b_vals, b_clusters)
    assert len(ga) >= 2 and len(gb) >= 2, "need >=2 clusters per group"
    sums = np.array([g.sum() for g in ga + gb])
    cnts = np.array([g.size for g in ga + gb], dtype=float)
    tot_s, tot_c = sums.sum(), cnts.sum()
    n_a, k = len(ga), len(ga) + len(gb)

    def diff(sel):
        sa, ca = sums[sel].sum(), cnts[sel].sum()
        return sa / ca - (tot_s - sa) / (tot_c - ca)

    obs = diff(np.arange(n_a))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        s = diff(rng.permutation(k)[:n_a])
        if alternative == "greater":
            hits += s >= obs - 1e-12
        else:
            hits += abs(s) >= abs(obs) - 1e-12
    return float((hits + 1) / (n_perm + 1))


def _boot_mean(sums, cnts, rng):
    sel = rng.integers(0, len(sums), len(sums))
    return sums[sel].sum() / cnts[sel].sum()


def tost_gap_recovery(treat, plus_anchor, minus_anchor, margin_frac=0.25,
                      treat_clusters=None, plus_clusters=None,
                      minus_clusters=None, n_boot=2000, seed=0):
    """Gap-recovery % + CI and TOST equivalence vs the +G anchor (R3).

    gap = mean(plus) − mean(minus); recovery = (mean(treat) − mean(minus))/gap.
    TOST: two one-sided Welch tests that mean(treat) lies within
    ±margin_frac·|gap| of mean(plus); tost_p = max of the two one-sided ps.
    CI on recovery: cluster bootstrap if cluster labels given (all three),
    else delta-method normal approximation.
    """
    treat = np.asarray(treat, float)
    plus = np.asarray(plus_anchor, float)
    minus = np.asarray(minus_anchor, float)
    assert treat.size >= 2 and plus.size >= 2 and minus.size >= 2, "tiny group"
    mt, mp, mm = treat.mean(), plus.mean(), minus.mean()
    gap = mp - mm
    assert abs(gap) > 1e-12, "anchors do not separate (R7 gate)"
    recovery = (mt - mm) / gap
    margin = abs(margin_frac * gap)
    p_lower = stats.ttest_ind(treat, plus - margin, equal_var=False,
                              alternative="greater").pvalue
    p_upper = stats.ttest_ind(treat, plus + margin, equal_var=False,
                              alternative="less").pvalue

    have_clusters = [c is not None for c in
                     (treat_clusters, plus_clusters, minus_clusters)]
    assert all(have_clusters) or not any(have_clusters), \
        "give cluster labels for all three groups or none"
    if all(have_clusters):
        rng = np.random.default_rng(seed)
        sc = []
        for vals, cl in ((treat, treat_clusters), (plus, plus_clusters),
                         (minus, minus_clusters)):
            g = _cluster_groups(vals, cl)
            sc.append((np.array([x.sum() for x in g]),
                       np.array([float(x.size) for x in g])))
        recs = []
        for _ in range(n_boot):
            bt, bp, bm = (_boot_mean(s, c, rng) for s, c in sc)
            bg = bp - bm
            if abs(bg) < 1e-9:
                continue
            recs.append((bt - bm) / bg)
        assert len(recs) >= n_boot // 2, "degenerate bootstrap (gap ~ 0)"
        ci = [float(x) for x in np.percentile(recs, [2.5, 97.5])]
        ci_method = "cluster_bootstrap"
    else:
        ses = [x.std(ddof=1) / np.sqrt(x.size) for x in (treat, plus, minus)]
        num = mt - mm
        var = (ses[0] ** 2 / gap ** 2 + num ** 2 * ses[1] ** 2 / gap ** 4
               + (num - gap) ** 2 * ses[2] ** 2 / gap ** 4)
        half = 1.96 * float(np.sqrt(var))
        ci = [float(recovery - half), float(recovery + half)]
        ci_method = "normal_approx"
    return {
        "recovery": float(recovery), "recovery_ci95": ci,
        "ci_method": ci_method, "gap": float(gap), "margin": float(margin),
        "tost_p": float(max(p_lower, p_upper)),
        "tost_p_lower": float(p_lower), "tost_p_upper": float(p_upper),
        "mean_treat": float(mt), "mean_plus": float(mp), "mean_minus": float(mm),
        "n": [int(treat.size), int(plus.size), int(minus.size)],
    }


def _state_offset(trial):
    """state_offset the runner used — 100 for e1c confirmatory, else 0."""
    return 100 if trial.get("phase") == "e1c" else 0


def ilc_dec(trials, pool):
    """I_LC^dec = max(0, (r_t − r_prev)/r_prev) on displayed-loss bet trials.

    The displayed previous round is the LAST history line build_prompt shows:
    hist[min(round_idx, len(hist)) − 1].  For a LOSS the displayed balance is
    balance_after = balance_before − bet, so balance_before = balance + bet
    and r_prev = bet / balance_before.  WIN-previous and stop trials are
    skipped.  Pool lookup mirrors runner.run_arm:
    pool[(trial_id + state_offset) % len(pool)].
    """
    assert pool, "empty pool"
    out = []
    for t in trials:
        if t.get("action") != "bet":
            continue
        game, round_idx = pool[(t["trial_id"] + _state_offset(t)) % len(pool)]
        hist = game.get("history", [])
        if round_idx <= 0 or not hist:
            continue
        h = hist[min(round_idx, len(hist)) - 1]
        # PROVENANCE: win flag logic identical to prompts.build_prompt.
        win = h.get("win", str(h.get("result", "")) == "W")
        if win:
            continue
        bet = float(h["bet"])
        bal_before = float(h["balance"]) + bet
        if bet <= 0 or bal_before <= 0:
            continue
        r_prev = bet / bal_before
        out.append(max(0.0, (float(t["bet_ratio"]) - r_prev) / r_prev))
    return out


def parse_excluded(trials):
    """R5: behavioural metrics are computed only on parse_ok trials."""
    return [t for t in trials if t.get("parse_ok")]


# ------------------------------------------------------------------- main()

def _download_arm(phase, arm_id):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(HF_REPO, f"{HF_BASE}/{phase}/{arm_id}.jsonl",
                            repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
    except Exception:
        return []
    return [json.loads(l) for l in open(p)]


def _build_pool():
    from .states import ensure_sm_catalog, load_minusG_states
    os.environ.setdefault("LLM_ADDICTION_BEHAVIORAL_ROOT",
                          str(DEFAULT_BEHAV_ROOT))
    ensure_sm_catalog("gemma")
    pool = load_minusG_states("gemma", n=POOL_N)
    assert len(pool) == POOL_N, f"pool {len(pool)} != {POOL_N}"
    gid = {}                              # id(game) -> first pool index seen
    for i, (game, _) in enumerate(pool):
        gid.setdefault(id(game), i)
    return pool, gid


def _clusters(trials, pool, gid):
    out = []
    for t in trials:
        game, _ = pool[(t["trial_id"] + _state_offset(t)) % len(pool)]
        out.append(gid[id(game)])
    return out


def _vals(trials):
    return [float(t["bet_ratio"]) for t in trials]


def _headline_entry(treat, plus, minus, pool, gid):
    tv, pv, mv = _vals(treat), _vals(plus), _vals(minus)
    tc, pc, mc = (_clusters(x, pool, gid) for x in (treat, plus, minus))
    entry = tost_gap_recovery(tv, pv, mv, treat_clusters=tc,
                              plus_clusters=pc, minus_clusters=mc)
    entry["perm_p_vs_minus"] = cluster_perm_test(tv, tc, mv, mc)
    entry["perm_p_vs_plus"] = cluster_perm_test(tv, tc, pv, pc)
    return entry


def _summ(trials):
    br = np.array([t["bet_ratio"] for t in trials], float)
    return {"n": len(trials), "iba": float(br.mean()) if len(trials) else None,
            "stop_rate": float(np.mean([t["action"] == "stop"
                                        for t in trials])) if trials else None}


def main():
    pool, gid = _build_pool()
    from .registry import load_arms
    arms = load_arms()
    trials_by_arm = {}
    for aid, a in arms.items():
        if a["phase"] not in PHASES:
            continue
        tr = _download_arm(a["phase"], aid)
        if tr:
            trials_by_arm[aid] = tr
    print(f"[retro] {len(trials_by_arm)} arms with checkpoints", flush=True)

    report = {"headline": {}, "e2_parse_sensitivity": {}, "ilc_dec": {}}
    for name, t_id, p_id, m_id in HEADLINE:
        groups = [parse_excluded(trials_by_arm.get(x, []))
                  for x in (t_id, p_id, m_id)]
        if any(len(g) < 2 for g in groups):
            report["headline"][name] = {"skipped": "missing arm data"}
            continue
        report["headline"][name] = _headline_entry(*groups, pool, gid)

    for aid in E2_PRIMARY:
        tr = trials_by_arm.get(aid)
        if not tr:
            continue
        ok = parse_excluded(tr)
        report["e2_parse_sensitivity"][aid] = {
            "raw": _summ(tr), "parse_ok": _summ(ok),
            "parse_rate": len(ok) / len(tr),
        }

    for aid in sorted(trials_by_arm):
        vals = ilc_dec(parse_excluded(trials_by_arm[aid]), pool)
        report["ilc_dec"][aid] = {
            "mean": float(np.mean(vals)) if vals else None, "n": len(vals)}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"[retro] wrote {OUT_JSON}", flush=True)
    _print_markdown(report)


def _print_markdown(report):
    def f(x, d=3):
        return "—" if x is None else f"{x:.{d}f}"
    # NOTE: tost_p is a non-clustered Welch TOST (anticonservative under game
    # clustering, R4) — read it alongside the cluster-bootstrap CI and the two
    # cluster permutation ps; W2 confirmatory will use a clustered variant.
    print("\n## Headline (game-cluster perm + TOST gap-recovery, parse_ok)")
    print("| comparison | n(T/+/−) | recovery | ci95 | tost_p(non-clustered) "
          "| perm_p vs −G | perm_p vs +G |")
    print("|---|---|---|---|---|---|---|")
    for name, e in report["headline"].items():
        if "skipped" in e:
            print(f"| {name} | {e['skipped']} | | | | | |")
            continue
        ci = e["recovery_ci95"]
        print(f"| {name} | {'/'.join(map(str, e['n']))} | {f(e['recovery'])} "
              f"| [{f(ci[0])}, {f(ci[1])}] | {f(e['tost_p'], 4)} "
              f"| {f(e['perm_p_vs_minus'], 4)} | {f(e['perm_p_vs_plus'], 4)} |")
    print("\n## E2 parse-excluded sensitivity (R5)")
    print("| arm | n | parse_rate | I_BA raw | I_BA ok | stop raw | stop ok |")
    print("|---|---|---|---|---|---|---|")
    for aid, e in report["e2_parse_sensitivity"].items():
        print(f"| {aid} | {e['raw']['n']} | {f(e['parse_rate'])} "
              f"| {f(e['raw']['iba'])} | {f(e['parse_ok']['iba'])} "
              f"| {f(e['raw']['stop_rate'])} | {f(e['parse_ok']['stop_rate'])} |")
    print("\n## I_LC^dec by arm (displayed-loss bet trials, parse_ok)")
    print("| arm | mean | n |")
    print("|---|---|---|")
    for aid, e in report["ilc_dec"].items():
        print(f"| {aid} | {f(e['mean'])} | {e['n']} |")


if __name__ == "__main__":
    main()
