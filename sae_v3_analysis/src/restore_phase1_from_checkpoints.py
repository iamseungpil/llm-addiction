#!/usr/bin/env python3
"""Reconstruct Phase 1 dose-response JSON from aligned_factor_steering checkpoints.

Input: ckpt_{A,B,C}_{model}_{task}.json under results/checkpoints_backup/
Output: results/json/phase1_restored_20260420/{experiment}_{model}_{task}.json

Each checkpoint `completed_alphas` holds 7 dicts (one per alpha) with:
  bk_count, stop_count, bk_rate, stop_rate, mean_terminal_wealth,
  mean_iba, mean_rounds, game_outcomes (list of per-game dicts, n=200).

This restorer computes the same Phase 1 statistics that the original script
would have written to the final JSON: OLS slope (BK, stop, wealth, IBA),
Cochran-Armitage trend test, Spearman rho. No mutation of the raw per-game
list; we simply aggregate.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

CKPT_DIR = Path("/tmp/hf_ckpt/sae_v3_analysis/results/checkpoints")
EXTRA_CKPT = CKPT_DIR / "ckpt_C_llama_sm.json"
OUT_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json/phase1_restored_20260420")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP_LAYERS = {"A_llama": 25, "B_gemma": 12, "C_llama": 25}


def cochran_armitage(bk_counts, n_counts, alphas):
    bk = np.asarray(bk_counts, dtype=float)
    n = np.asarray(n_counts, dtype=float)
    s = np.asarray(alphas, dtype=float)
    N = n.sum()
    p = bk.sum() / N
    if p == 0 or p == 1:
        return {"z": 0.0, "p_value": 1.0}
    num = np.sum(s * (bk - n * p))
    var = p * (1 - p) * (np.sum(n * s * s) - (np.sum(n * s) ** 2) / N)
    z = num / np.sqrt(var) if var > 0 else 0.0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p_value": float(pval)}


def ols_stats(y, x):
    slope, intercept, r, p, se = stats.linregress(x, y)
    t = slope / se if se > 0 else 0.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "t_stat": float(t),
        "p_value": float(p),
        "r_squared": float(r ** 2),
        "std_err": float(se),
    }


def build_main_sweep(ca):
    """Return list of per-alpha summary dicts, sorted by alpha."""
    rows = sorted(ca, key=lambda r: r["alpha"])
    out = []
    for r in rows:
        out.append({
            "alpha": r["alpha"],
            "task": r["task"],
            "n_games": r["n_games"],
            "bk_count": r["bk_count"],
            "stop_count": r["stop_count"],
            "bk_rate": r["bk_rate"],
            "stop_rate": r["stop_rate"],
            "mean_terminal_wealth": r["mean_terminal_wealth"],
            "mean_iba": r["mean_iba"],
            "mean_rounds": r["mean_rounds"],
            "parse_failures": r.get("parse_failures", 0),
        })
    return out


def compute_main_stats(sweep):
    alphas = [r["alpha"] for r in sweep]
    bk_rates = [r["bk_rate"] for r in sweep]
    stop_rates = [r["stop_rate"] for r in sweep]
    wealth = [r["mean_terminal_wealth"] for r in sweep]
    ibas = [r["mean_iba"] for r in sweep]
    bk_counts = [r["bk_count"] for r in sweep]
    n_counts = [r["n_games"] for r in sweep]
    return {
        "cochran_armitage_bk": cochran_armitage(bk_counts, n_counts, alphas),
        "ols_bk_rate": ols_stats(bk_rates, alphas),
        "ols_stop_rate": ols_stats(stop_rates, alphas),
        "ols_wealth": ols_stats(wealth, alphas),
        "ols_iba": ols_stats(ibas, alphas),
        "spearman": {
            "rho_bk": float(stats.spearmanr(alphas, bk_rates).statistic),
            "p_bk": float(stats.spearmanr(alphas, bk_rates).pvalue),
            "rho_iba": float(stats.spearmanr(alphas, ibas).statistic),
            "p_iba": float(stats.spearmanr(alphas, ibas).pvalue),
        },
    }


def restore(ckpt_path: Path, exp_char: str) -> dict:
    d = json.load(open(ckpt_path))
    meta = d["metadata"]
    model = meta["model"]
    task = meta["task"]
    n_games = meta["n_games"]

    sweep = build_main_sweep(d["completed_alphas"])
    main_stats = compute_main_stats(sweep)

    out = {
        "experiment": exp_char,
        "model": model,
        "task": task,
        "layer": EXP_LAYERS.get(f"{exp_char}_{model}"),
        "n_games_per_alpha": n_games,
        "alpha_values": [r["alpha"] for r in sweep],
        "main_sweep": sweep,
        "main_stats": main_stats,
        "null_distribution": {
            "status": "not_computed_phase2_lost",
            "note": "Phase 2 permutation null was running at checkpoint time; the container was reset on 2026-04-20 and the partial Phase 2 results were lost. Phase 1 main-sweep data below is intact.",
        },
        "source_checkpoint": str(ckpt_path),
        "source_checkpoint_timestamp": d.get("timestamp"),
        "restoration_script": "src/restore_phase1_from_checkpoints.py",
        "restored_at": datetime.utcnow().isoformat() + "Z",
        "bridge": "exact_behavioral_replay",
    }
    return out


def main():
    jobs = [
        ("A", "llama", "ic"),
        ("A", "llama", "mw"),
        ("A", "llama", "sm"),
        ("B", "gemma", "ic"),
        ("B", "gemma", "mw"),
        ("B", "gemma", "sm"),
        ("C", "llama", "sm"),
    ]

    results = {}
    for exp, model, task in jobs:
        fname = f"ckpt_{exp}_{model}_{task}.json"
        local = CKPT_DIR / fname
        if not local.exists() and exp == "C":
            local = EXTRA_CKPT
        if not local.exists():
            print(f"SKIP {fname}: not found")
            continue
        out = restore(local, exp)
        out_path = OUT_DIR / f"phase1_{exp}_{model}_{task}.json"
        json.dump(out, open(out_path, "w"), indent=2)
        summary = {
            "exp": f"{exp}_{model}_{task}",
            "layer": out["layer"],
            "bk_rates": [f"{r['bk_rate']:.3f}" for r in out["main_sweep"]],
            "ca_bk_z": f"{out['main_stats']['cochran_armitage_bk']['z']:.3f}",
            "ca_bk_p": f"{out['main_stats']['cochran_armitage_bk']['p_value']:.4f}",
            "ols_bk_slope": f"{out['main_stats']['ols_bk_rate']['slope']:.4f}",
            "ols_bk_p": f"{out['main_stats']['ols_bk_rate']['p_value']:.4f}",
            "rho_bk": f"{out['main_stats']['spearman']['rho_bk']:.3f}",
            "p_bk": f"{out['main_stats']['spearman']['p_bk']:.4f}",
        }
        results[f"{exp}_{model}_{task}"] = summary
        print(f"=> {out_path.name}: ρ_bk={summary['rho_bk']} (p={summary['p_bk']}), CA Z={summary['ca_bk_z']} (p={summary['ca_bk_p']})")

    summary_path = OUT_DIR / "phase1_restoration_summary.json"
    json.dump({
        "restored_at": datetime.utcnow().isoformat() + "Z",
        "restoration_source": "aligned_factor_steering Phase 1 checkpoints (Apr 17-19, 2026)",
        "phase2_status": "Lost (container reset 2026-04-20 before Phase 2 completion)",
        "bridge": "exact_behavioral_replay",
        "n_games_per_alpha": 200,
        "n_alphas": 7,
        "per_experiment": results,
    }, open(summary_path, "w"), indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
