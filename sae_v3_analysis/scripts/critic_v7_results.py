#!/usr/bin/env python3
"""Critic v7 result JSONs — validate structure and flag anomalies.

Called after each monitor-tick pulls new combined_*.json from e8.
Returns structured pass/fail report + stats summary per experiment.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
JSON_DIR = REPO / "results" / "json"


def _get(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _extract_sweep_and_stats(exp_block: dict):
    """Return (main_sweep, main_stats, task_label) unified across Exp A/B/C layouts.

    Exp C: has main_sweep + main_stats at top.
    Exp A/B: has task_results = {task: {main_sweep, main_stats, ...}}.
    """
    if "main_sweep" in exp_block:
        return exp_block.get("main_sweep", []), exp_block.get("main_stats", {}), exp_block.get("task")
    tr = exp_block.get("task_results", {})
    if isinstance(tr, dict) and tr:
        # Take the first (and usually only) task
        task_name = next(iter(tr))
        tblock = tr[task_name]
        return tblock.get("main_sweep", []), tblock.get("main_stats", {}), task_name
    return [], {}, None


def critic_aligned(payload: dict) -> dict:
    """Critic for aligned_factor_steering combined JSON."""
    issues = []
    args = payload.get("args", {})
    results = payload.get("results", {})
    exp_block = None
    exp_name = None
    for k in ("experiment_a", "experiment_b", "experiment_c"):
        if k in results:
            exp_block = results[k]
            exp_name = k
            break
    if exp_block is None:
        return {"ok": False, "issues": ["no experiment_a/b/c block"], "summary": {}}

    alphas = exp_block.get("alpha_values", [])
    if len(alphas) != 7:
        issues.append(f"alpha_values len={len(alphas)} (expected 7)")

    main_sweep, main_stats, task_label = _extract_sweep_and_stats(exp_block)
    if len(main_sweep) != len(alphas):
        issues.append(f"main_sweep len={len(main_sweep)} != alphas len={len(alphas)}")

    ca = _get(main_stats, "cochran_armitage_bk", "Z")
    rho = _get(main_stats, "spearman", "bk_rate", "rho")
    if ca is None or (isinstance(ca, float) and math.isnan(ca)):
        issues.append("CA Z missing/NaN")
    if rho is None or (isinstance(rho, float) and math.isnan(rho)):
        issues.append("spearman rho missing/NaN")

    null = exp_block.get("null_distribution", {})
    n_dirs = null.get("n_dirs", 0)
    if n_dirs < 10:
        issues.append(f"null n_dirs={n_dirs} (expected >= 20)")
    perm = exp_block.get("permutation_tests", {})
    perm_p = _get(perm, "spearman_rho", "perm_p") or _get(perm, "cochran_armitage_Z", "perm_p")

    bk_rates = []
    for r in main_sweep:
        br = _get(r, "behavioral", "bk_rate") or _get(r, "stats", "bk_rate") or r.get("bk_rate")
        if br is not None and not (isinstance(br, float) and math.isnan(br)):
            bk_rates.append(br)
    bk_range = (min(bk_rates), max(bk_rates)) if bk_rates else (None, None)

    return {
        "ok": len(issues) == 0,
        "experiment": exp_name,
        "model": args.get("model"),
        "task_filter": args.get("task_filter") or task_label,
        "n_games": args.get("n_games"),
        "issues": issues,
        "summary": {
            "alphas": alphas,
            "n_main": len(main_sweep),
            "n_null_dirs": n_dirs,
            "ca_z": ca,
            "spearman_rho": rho,
            "perm_p": perm_p,
            "bk_rate_range": bk_range,
        },
    }


def critic_shared_axis(payload: dict) -> dict:
    """Critic for shared_axis_steering JSON."""
    issues = []
    args = payload.get("args", {})
    result = payload.get("result", {})
    alphas = result.get("alpha_values", [])
    if len(alphas) != 7:
        issues.append(f"alphas len={len(alphas)}")
    main = result.get("main_sweep", [])
    if len(main) != len(alphas):
        issues.append(f"main_sweep len={len(main)}")
    axis_meta = result.get("axis_meta", {})
    if not axis_meta.get("eigenvalue"):
        issues.append("axis eigenvalue missing")

    perm = result.get("permutation_tests", {})
    perm_p = _get(perm, "spearman_rho", "perm_p")
    ca_z = _get(result, "main_stats", "cochran_armitage_bk", "Z")
    rho = _get(result, "main_stats", "spearman", "bk_rate", "rho")

    bk_rates = []
    for r in main:
        br = _get(r, "behavioral", "bk_rate") or _get(r, "stats", "bk_rate") or r.get("bk_rate")
        if br is not None and not math.isnan(br):
            bk_rates.append(br)

    return {
        "ok": len(issues) == 0,
        "experiment": "shared_axis",
        "model": args.get("model"),
        "task": args.get("task"),
        "layer": result.get("layer"),
        "issues": issues,
        "summary": {
            "axis_eigenvalue": axis_meta.get("eigenvalue"),
            "alphas": alphas,
            "n_main": len(main),
            "n_null_dirs": _get(result, "null_distribution", "n_dirs"),
            "ca_z": ca_z,
            "spearman_rho": rho,
            "perm_p": perm_p,
            "bk_rate_range": (min(bk_rates), max(bk_rates)) if bk_rates else (None, None),
        },
    }


def main():
    glob_pat = sys.argv[1] if len(sys.argv) > 1 else "*.json"
    files = sorted(JSON_DIR.glob(glob_pat))
    reports = []
    for f in files:
        try:
            with f.open() as h:
                payload = json.load(h)
        except Exception as e:
            reports.append({"file": f.name, "ok": False, "issues": [f"load: {e}"]})
            continue
        script = payload.get("script", "")
        if "shared_axis" in script or "shared_axis" in f.name:
            r = critic_shared_axis(payload)
        elif "aligned_factor" in script or "aligned_steering" in f.name:
            r = critic_aligned(payload)
        else:
            continue
        r["file"] = f.name
        reports.append(r)

    # Print summary
    passed = sum(1 for r in reports if r.get("ok"))
    print(f"=== Critic v7 — {len(reports)} files, {passed} pass, {len(reports)-passed} fail ===\n")
    for r in reports:
        status = "PASS" if r.get("ok") else "FAIL"
        head = f"[{status}] {r['file']}"
        tag = f" {r.get('experiment','?')}/{r.get('model','?')}/{r.get('task') or r.get('task_filter') or '-'}"
        print(head + tag)
        s = r.get("summary", {})
        if s:
            print(f"    CA_Z={s.get('ca_z')}, rho={s.get('spearman_rho')}, perm_p={s.get('perm_p')}, "
                  f"BK range={s.get('bk_rate_range')}, n_null_dirs={s.get('n_null_dirs')}")
        if r.get("issues"):
            for iss in r["issues"]:
                print(f"    - {iss}")
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
