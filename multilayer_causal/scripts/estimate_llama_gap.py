#!/usr/bin/env python3
"""CPU pre-estimate of LLaMA's natural −G vs +G behavioural gap (w3L gate).

RUN_PLAN_W3 w3L requires the anchor gate's n=50 power to be checked BEFORE
submission, from the existing behavioural corpus. This script mirrors the
frozen state-pool semantics (states.load_minusG_states: variable betting,
decision indices 2–6) on the llama_v4_role catalog, splits by 'G' in
prompt_combo, and computes the mixed bet_ratio convention used by every
W-track table: stop counts as 0.0, bet counts as bet/balance_before
(skipped/invalid decisions are excluded, matching the runner's parse path
which always yields bet-or-stop).

Power: simulation — resample n per arm with replacement from the empirical
pools, Welch t-test, fraction of sims with p < 0.05. If n=50 is underpowered
(< 0.8), the smallest adequate n from the sweep is reported.

MDE (round-2 audit): the anchor gate is power-sized (n=878), but the 8 sweep
windows stay discovery n=50 and the promoted window's W4 confirmatory stays
n=200 — both underpowered against a gap-sized (d=0.134) window effect. To
pre-register what those stages CAN claim, this script also computes the
minimal detectable effect at n=50 and n=200: synthetic shift on the −G pool
(both arms resampled from −G, one shifted, isolating effect size from the
real gap), z-approx seed d=(z_a+z_b)*sqrt(2/n), then simulation-verified and
bumped until power >= 0.8. Gemma-magnitude power (W2 gap 0.141) is reported
alongside to justify the discovery sizing for the actual target effect.

Output: multilayer_causal/out/llama_gap_estimate.json
Usage:  python multilayer_causal/scripts/estimate_llama_gap.py [--model llama]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from multilayer_causal.src.states import (behavioral_root,  # noqa: E402
                                          ensure_sm_catalog)

ALPHA = 0.05
N_SIMS = 5000
N_SWEEP = (50, 75, 100, 150, 200, 300)
SIM_SEED = 2000042  # w3 seed family


def mixed_bet_ratios(model: str) -> tuple[list[float], list[float]]:
    """(−G pool, +G pool) of mixed bet_ratio values, frozen-pool semantics."""
    path = behavioral_root() / "slot_machine" / f"{model}_v4_role"
    minus, plus = [], []
    for game_file in sorted(path.glob("*.json")):
        d = json.load(open(game_file))
        games = d.get("results", d.get("games", []))
        if isinstance(games, dict):
            games = list(games.values())
        for game in games:
            if game.get("bet_type") != "variable":
                continue
            sink = plus if "G" in game.get("prompt_combo", "") else minus
            decs = game.get("decisions", [])
            for round_idx in range(2, min(7, len(decs))):
                dec = decs[round_idx]
                action, bal = dec.get("action"), dec.get("balance_before", 0)
                if action == "stop":
                    sink.append(0.0)
                elif action == "bet" and dec.get("bet") and bal and bal > 0:
                    sink.append(min(float(dec["bet"]), float(bal)) / float(bal))
    return minus, plus


def welch_power(minus, plus, n: int, n_sims: int = N_SIMS,
                seed: int = SIM_SEED) -> float:
    rng = np.random.default_rng(seed + n)
    minus, plus = np.asarray(minus), np.asarray(plus)
    hits = 0
    for _ in range(n_sims):
        a = rng.choice(minus, size=n, replace=True)
        b = rng.choice(plus, size=n, replace=True)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        hits += bool(p < ALPHA)
    return float(hits) / n_sims


def shift_power(minus, n: int, shift: float, n_sims: int = N_SIMS,
                seed: int = SIM_SEED) -> float:
    """Power for a synthetic effect: both arms resample the −G pool, one arm
    shifted by `shift` — effect size is controlled, pool geometry is real."""
    rng = np.random.default_rng(seed + 31 * n + int(round(shift * 1e6)))
    minus = np.asarray(minus)
    hits = 0
    for _ in range(n_sims):
        a = rng.choice(minus, size=n, replace=True)
        b = rng.choice(minus, size=n, replace=True) + shift
        _, p = stats.ttest_ind(a, b, equal_var=False)
        hits += bool(p < ALPHA)
    return float(hits) / n_sims


def mde_at_n(minus, n: int, sd_pooled: float) -> dict:
    """Smallest shift with simulation power >= 0.8 at n/arm: z-approx seed,
    then 5% bumps until the empirical-resampling simulation confirms."""
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.8)
    shift = float(z * np.sqrt(2.0 / n) * sd_pooled)
    power = shift_power(minus, n, shift)
    while power < 0.8:
        shift *= 1.05
        power = shift_power(minus, n, shift)
    return {"shift": round(shift, 4),
            "cohens_d": round(shift / sd_pooled, 3),
            "sim_power": round(power, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "out" / "llama_gap_estimate.json"))
    args = ap.parse_args()

    ensure_sm_catalog(args.model)
    minus, plus = mixed_bet_ratios(args.model)
    assert len(minus) >= 100 and len(plus) >= 100, "pool too small for estimate"
    m, p = np.asarray(minus), np.asarray(plus)
    gap = float(p.mean() - m.mean())

    power = {n: welch_power(minus, plus, n) for n in N_SWEEP}
    power50 = power[50]
    n_needed = next((n for n in N_SWEEP if power[n] >= 0.8), None)
    # sweep exhausted → analytic two-sample n (z-approx), simulation-verified
    if n_needed is None:
        z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.8)
        var_sum = float(m.var(ddof=1) + p.var(ddof=1))
        n_analytic = int(np.ceil(z ** 2 * var_sum / gap ** 2))
        power[n_analytic] = welch_power(minus, plus, n_analytic)
        n_needed = n_analytic
        while power[n_needed] < 0.8:  # resampling can undershoot the z-approx
            n_needed = int(np.ceil(n_needed * 1.15))
            power[n_needed] = welch_power(minus, plus, n_needed)

    sd_pooled = float(np.sqrt((m.var(ddof=1) + p.var(ddof=1)) / 2))
    d_pooled = float(gap / sd_pooled)
    gemma_gap = 0.202 - 0.061  # W2 n=200 reference, same metric
    mde = {}
    for n in (50, 200):
        cell = mde_at_n(minus, n, sd_pooled)
        cell["gap_multiple"] = round(cell["shift"] / gap, 2)
        mde[f"n{n}"] = cell
    result = {
        "model": args.model,
        "metric": "mixed bet_ratio (stop=0, bet=bet/balance_before, "
                  "decision indices 2-6, variable betting)",
        "n_minusG": int(m.size), "n_plusG": int(p.size),
        "mean_minusG": round(float(m.mean()), 4),
        "mean_plusG": round(float(p.mean()), 4),
        "sd_minusG": round(float(m.std(ddof=1)), 4),
        "sd_plusG": round(float(p.std(ddof=1)), 4),
        "gap": round(gap, 4),
        "cohens_d": round(d_pooled, 3),
        "welch_power": {str(n): round(v, 3) for n, v in power.items()},
        "power_n50": round(power50, 3),
        "n50_adequate": bool(power50 >= 0.8),
        "n_per_arm_for_80pct": n_needed,
        "mde": {
            "method": "synthetic shift on −G pool (both arms resampled from "
                      "−G, one shifted), z-approx seed, simulation-verified "
                      "to power >= 0.8",
            "n50_discovery": mde["n50"],
            "n200_confirmatory": mde["n200"],
        },
        "gemma_magnitude_power": {
            "shift": round(gemma_gap, 4),
            "cohens_d": round(gemma_gap / sd_pooled, 3),
            "n50": round(shift_power(minus, 50, gemma_gap), 3),
            "n200": round(shift_power(minus, 200, gemma_gap), 3),
        },
        "sim": {"n_sims": N_SIMS, "alpha": ALPHA, "seed": SIM_SEED,
                "method": "empirical resampling, Welch t-test"},
        "gemma_w2_reference": {"minusG": 0.061, "plusG": 0.202, "n": 200},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
