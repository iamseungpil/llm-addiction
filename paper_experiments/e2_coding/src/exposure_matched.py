"""Compare ruin between the fixed and variable arms at matched cumulative exposure.

Reviewer gbSA's Weakness 4 is that variable betting changes two things at once: it lets the
model choose the stake, and it lets the model stay in the game longer, so the arms differ in
cumulative exposure as well as in discretion. Matching the cap removes the first confound
(range expansion) but not the second.

This script removes the second. For each variable game we walk the rounds and record the
cumulative amount staked. A game counts as ruined *at exposure X* if it went bankrupt while
its cumulative stake was at most X. Sweeping X and reading both arms off the same axis gives
a like-for-like comparison: at the point where the fixed arm has staked as much as it ever
does, has the variable arm ruined more or less?

The comparison is only meaningful where both arms actually play, so the script reports
participation alongside every cell and refuses to interpret a cell whose arms differ.

Corpus: the matched-cap-with-conditions run (mc32), cap $70, persona on, n = 50 per cell,
BASE and the paper's full five-module condition.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

DEFAULT_GLOB = "/home/v-seungplee/data/llm-addiction/mc32/*.json"


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def walk(game: dict) -> tuple[float, float | None]:
    """Return (total staked, cumulative stake at the moment of ruin or None)."""
    total = 0.0
    ruin_at: float | None = None
    for rnd in game.get("rounds", []):
        if rnd.get("decision") != "bet":
            continue
        total += rnd.get("bet") or 0
        if ruin_at is None and rnd.get("balance_after") == 0:
            ruin_at = total
    if ruin_at is None and (game.get("outcome") == "bankrupt" or game.get("final_balance") == 0):
        # Ruin recorded at game level without a zero balance_after on any round.
        ruin_at = total
    return total, ruin_at


def load(pattern: str) -> dict:
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for path in sorted(glob.glob(pattern)):
        payload = json.load(open(path))
        key = (payload["model"], payload.get("prompt_combo", "BASE"), payload["mode"])
        cells[key] = payload["results"]
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/exposure_matched.json")
    args = ap.parse_args()

    cells = load(args.glob)
    report: dict = {}

    models = sorted({k[0] for k in cells})
    conditions = sorted({k[1] for k in cells})

    for model in models:
        for cond in conditions:
            kf, kv = (model, cond, "fixed"), (model, cond, "variable")
            if kf not in cells or kv not in cells:
                continue
            fixed, variable = cells[kf], cells[kv]

            f_walk = [walk(g) for g in fixed]
            v_walk = [walk(g) for g in variable]
            f_play = sum(1 for g in fixed if any(r.get("decision") == "bet" for r in g.get("rounds", [])))
            v_play = sum(1 for g in variable if any(r.get("decision") == "bet" for r in g.get("rounds", [])))

            f_ruin = sum(1 for _t, r in f_walk if r is not None)
            v_ruin = sum(1 for _t, r in v_walk if r is not None)
            if not f_ruin and not v_ruin:
                continue

            # The exposure the fixed arm actually reaches, averaged over its games.
            budget = sum(t for t, _r in f_walk) / len(f_walk)
            v_ruin_at_budget = sum(1 for _t, r in v_walk if r is not None and r <= budget)

            nf, nv = len(fixed), len(variable)
            lo_f, hi_f = wilson(f_ruin, nf)
            lo_v, hi_v = wilson(v_ruin, nv)
            lo_m, hi_m = wilson(v_ruin_at_budget, nv)

            label = "GMHWP" if cond == "GMPRW" else cond
            matched = "yes" if f_play == v_play else f"NO ({f_play} vs {v_play})"
            print(f"{model[:20]:<22}{label:<7} participation matched: {matched}")
            print(f"    fixed     ruin {100*f_ruin/nf:5.1f} [{lo_f:.1f},{hi_f:.1f}]   "
                  f"mean total staked ${budget:,.0f}")
            print(f"    variable  ruin {100*v_ruin/nv:5.1f} [{lo_v:.1f},{hi_v:.1f}]   "
                  f"mean total staked ${sum(t for t,_ in v_walk)/nv:,.0f}")
            print(f"    variable, truncated at the fixed arm's exposure (${budget:,.0f}): "
                  f"{100*v_ruin_at_budget/nv:5.1f} [{lo_m:.1f},{hi_m:.1f}]")
            print()

            report[f"{model}|{label}"] = {
                "n_fixed": nf, "n_variable": nv,
                "participation_fixed": f_play, "participation_variable": v_play,
                "fixed_ruin_pct": 100 * f_ruin / nf, "fixed_ruin_ci": [lo_f, hi_f],
                "variable_ruin_pct": 100 * v_ruin / nv, "variable_ruin_ci": [lo_v, hi_v],
                "fixed_mean_total_stake": budget,
                "variable_mean_total_stake": sum(t for t, _ in v_walk) / nv,
                "variable_ruin_at_fixed_exposure_pct": 100 * v_ruin_at_budget / nv,
                "variable_ruin_at_fixed_exposure_ci": [lo_m, hi_m],
            }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
