"""Per-arm summaries, anchor comparisons, G1 gate, S* selection, direction build.

Usage:
  python -m multilayer_causal.src.analyze summary --out multilayer_causal/out
  python -m multilayer_causal.src.analyze directions \
      --vectors multilayer_causal/out/e1_full_vectors.npz \
      --dest multilayer_causal/out/directions.npz
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def load_arm(out_dir, arm_id):
    p = Path(out_dir) / f"{arm_id}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def summarize(trials):
    if not trials:
        return None
    br = np.array([t["bet_ratio"] for t in trials], float)
    rng = np.random.default_rng(0)
    boot = [rng.choice(br, len(br)).mean() for _ in range(1000)]
    return {
        "n": len(trials),
        "bet_ratio_mean": float(br.mean()),
        "bet_ratio_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "stop_rate": float(np.mean([t["action"] == "stop" for t in trials])),
        "extreme_rate": float(np.mean([t["extreme"] for t in trials])),
        "parse_rate": float(np.mean([t["parse_ok"] for t in trials])),
    }


def welch_vs(trials_a, trials_b, key="bet_ratio"):
    a = np.array([t[key] for t in trials_a], float)
    b = np.array([t[key] for t in trials_b], float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def cohen_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def gate_g1(out_dir):
    """Full-layer patch must be indistinguishable from natural_plusG."""
    full = load_arm(out_dir, "e1_full")
    plus = load_arm(out_dir, "e1_anchor_plus")
    if not full or not plus:
        return {"pass": False, "reason": "missing arms"}
    _, p = welch_vs(full, plus)
    ds = abs(summarize(full)["stop_rate"] - summarize(plus)["stop_rate"])
    return {"pass": bool(p > 0.05 and ds < 0.15), "welch_p_vs_plus": p,
            "stop_rate_gap": ds}


def select_s_star(out_dir, arms):
    """Smallest passing layer set; ties → deeper window. Pre-registered rule."""
    plus = load_arm(out_dir, "e1_anchor_plus")
    if not plus:
        return None
    passing = []
    for aid, a in arms.items():
        if a["phase"] != "e1" or a["mode"] != "patch":
            continue
        tr = load_arm(out_dir, aid)
        if len(tr) < 30:
            continue
        _, p = welch_vs(tr, plus)
        ds = abs(summarize(tr)["stop_rate"] - summarize(plus)["stop_rate"])
        if p > 0.05 and ds < 0.15:
            passing.append((len(a["layers"]), -min(a["layers"]), aid, a["layers"]))
    if not passing:
        return None
    passing.sort()
    return {"arm": passing[0][2], "layers": passing[0][3],
            "all_passing": [x[2] for x in passing]}


def build_directions(vectors_npz, dest):
    """v̂_l = normalize(mean(plus − minus)); scale_l = 0.03 · median ||minus_l||."""
    z = np.load(vectors_npz)
    minus, plus = z["minus"].astype(np.float32), z["plus"].astype(np.float32)
    assert minus.shape == plus.shape and minus.ndim == 3
    delta = (plus - minus).mean(axis=0)                              # (L, D)
    dirs = delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-9)
    scales = 0.03 * np.median(np.linalg.norm(minus, axis=2), axis=0)  # (L,)
    np.savez(dest, directions=dirs, scales=scales, n_pairs=minus.shape[0])
    print(f"directions -> {dest} (n_pairs={minus.shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--out", required=True)
    d = sub.add_parser("directions")
    d.add_argument("--vectors", required=True)
    d.add_argument("--dest", required=True)
    args = ap.parse_args()
    if args.cmd == "directions":
        build_directions(args.vectors, args.dest)
        return
    from .registry import load_arms
    arms = load_arms()
    plus = load_arm(args.out, "e1_anchor_plus")
    minus = load_arm(args.out, "e1_anchor_minus")
    rows = {}
    for aid in arms:
        tr = load_arm(args.out, aid)
        if not tr:
            continue
        srow = summarize(tr)
        if plus and aid != "e1_anchor_plus":
            srow["welch_p_vs_plusG"] = welch_vs(tr, plus)[1]
        if minus and aid != "e1_anchor_minus":
            srow["welch_p_vs_minusG"] = welch_vs(tr, minus)[1]
        rows[aid] = srow
    report = {"arms": rows, "gate_g1": gate_g1(args.out),
              "s_star": select_s_star(args.out, arms)}
    dest = Path(args.out) / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
