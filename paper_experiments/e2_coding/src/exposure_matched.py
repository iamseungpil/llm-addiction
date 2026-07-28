"""Ruin in the two arms compared at the same cumulative stake.

Reviewer gbSA's Weakness 4 is that letting the model choose its wager changes two things at once:
the size of each stake, and how long the game lasts. Capping both arms at the same maximum bet
removes the first. This script removes the second, by asking how much ruin each arm has produced
by the time it has staked a given total.

An earlier version of this script was degenerate and its numbers must not be reused. It truncated
the choosing arm at the *mean total stake of every fixed-arm game, including the games where the
model never wagered at all*, which put the threshold at $34, $85 and $90 in three of four cells.
Ruin from a $100 opening balance requires a cumulative stake of at least $100 — at the moment of
ruin, 0 = 100 - S + 3*W with W >= 0, so S >= 100, and the corpus confirms it: of 172 ruined games
the smallest cumulative stake is exactly $100 and none is below. Below a $100 threshold no data
of any kind can return a non-zero rate, so those three cells reported an arithmetic identity as a
finding. The comparison was also asymmetric: the fixed arm's ruin was counted in full while the
choosing arm's was truncated.

This version fixes both. Only games in which the model actually wagered enter the comparison, the
same threshold is applied to both arms, and the threshold is swept from $100 upward rather than
taken from one arm's mean.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

DEFAULT_GLOB = "/home/v-seungplee/data/llm-addiction/mc32/final_*.json"
GRID = [100, 150, 200, 300, 500, None]      # None = no truncation


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def walk(game: dict) -> tuple[bool, float, float | None]:
    """Return (wagered at all, total staked, cumulative stake at ruin or None)."""
    total = 0.0
    ruin_at: float | None = None
    wagered = False
    for rnd in game.get("rounds", []):
        if rnd.get("decision") != "bet":
            continue
        wagered = True
        total += rnd.get("bet") or 0
        if ruin_at is None and rnd.get("balance_after") == 0:
            ruin_at = total
    # The runner writes "bankruptcy"; accept either spelling and fall back to the final balance.
    if ruin_at is None and (str(game.get("outcome", "")).startswith("bankrupt")
                            or game.get("final_balance") == 0):
        ruin_at = total if wagered else None
    return wagered, total, ruin_at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/exposure_matched.json")
    args = ap.parse_args()

    cells: dict[tuple[str, str, str], list[dict]] = {}
    for path in sorted(glob.glob(args.glob)):
        payload = json.load(open(path))
        cells[(payload["model"], payload.get("prompt_combo", "BASE"), payload["mode"])] = payload["results"]

    report: dict = {}
    for model in sorted({k[0] for k in cells}):
        for cond in sorted({k[1] for k in cells}):
            kf, kv = (model, cond, "fixed"), (model, cond, "variable")
            if kf not in cells or kv not in cells:
                continue
            f = [walk(g) for g in cells[kf]]
            v = [walk(g) for g in cells[kv]]
            fp = [x for x in f if x[0]]          # games in which the model actually wagered
            vp = [x for x in v if x[0]]
            if not fp or not vp:
                continue
            if not any(x[2] is not None for x in fp + vp):
                continue

            label = "GMHWP" if cond == "GMPRW" else cond
            print(f"{model[:20]:<22}{label:<7} wagering games: fixed {len(fp)}/{len(f)}, "
                  f"choosing {len(vp)}/{len(v)}")
            rows = []
            for x in GRID:
                fk = sum(1 for _w, _t, r in fp if r is not None and (x is None or r <= x))
                vk = sum(1 for _w, _t, r in vp if r is not None and (x is None or r <= x))
                flo, fhi = wilson(fk, len(fp))
                vlo, vhi = wilson(vk, len(vp))
                tag = "no cap" if x is None else f"<= ${x}"
                print(f"    stake {tag:<9} forced {100*fk/len(fp):5.1f} [{flo:4.1f},{fhi:5.1f}]   "
                      f"choosing {100*vk/len(vp):5.1f} [{vlo:4.1f},{vhi:5.1f}]   "
                      f"delta {100*vk/len(vp) - 100*fk/len(fp):+6.1f}")
                rows.append({"threshold": x,
                             "fixed_pct": 100 * fk / len(fp), "fixed_ci": [flo, fhi],
                             "variable_pct": 100 * vk / len(vp), "variable_ci": [vlo, vhi]})
            print()
            report[f"{model}|{label}"] = {
                "n_fixed_wagering": len(fp), "n_variable_wagering": len(vp),
                "n_fixed_total": len(f), "n_variable_total": len(v), "sweep": rows}

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
