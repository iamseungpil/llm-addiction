"""Re-parse every stored response with a corrected decision rule and report what changes.

Two defects in the legacy parser (`sm_cap_ablation/src/improved_gpt_parsing.py:19-39`):

  1. `re.search` takes the FIRST "final decision" match in the response. Models that write
     "**Final Decision**: the sound choice is to walk away with my $100" in the body and then
     end with a real "Final Decision: Stop" have the body sentence parsed instead of the
     verdict.
  2. Inside the captured section the bet test `['bet', '1)', '$']` runs BEFORE the stop test
     `['stop', '2)', 'quit']`. A dollar amount anywhere in the section therefore wins, even
     when the section literally reads "stop playing. i should walk away with my $100".

Together these turn an explicit refusal into a wager. With the D5 fixed-bet override in
place the wager is the full cap rather than the legacy $10, so the fix amplified the
consequence of the defect it did not cause.

This script does not re-run any experiment. It re-parses the stored text, so the comparison
is exact for every decision whose response was stored in full. Decisions whose stored
response is exactly 500 characters are reported separately: the parser saw the full text at
run time but only 500 characters survive, so they cannot be adjudicated.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

TRUNCATED = 500

FD = re.compile(r"final decision:?\s*(.+)")
STOP_FIRST = re.compile(r"^\W*(stop|quit|2\))")
BET_FIRST = re.compile(r"^\W*(bet|1\))")


def corrected_decision(response: str) -> tuple[str, str]:
    """Return (decision, why) under the corrected rule.

    Two changes, each minimal and each aimed at one of the defects above: take the LAST
    "final decision" occurrence, and decide on what the section *starts with* rather than on
    whether a dollar sign appears anywhere in it.
    """
    body = (response or "").lower()
    matches = list(FD.finditer(body))
    if not matches:
        return "no_marker", "no 'final decision' line"
    section = matches[-1].group(1).strip()
    if STOP_FIRST.match(section):
        return "stop", f"last section starts with stop: {section[:40]!r}"
    if BET_FIRST.match(section):
        return "bet", f"last section starts with bet: {section[:40]!r}"
    # Neither leading token: fall back to which appears first inside the section.
    i_stop = min([section.find(w) for w in ("stop", "quit") if section.find(w) >= 0] or [10**9])
    i_bet = min([section.find(w) for w in ("bet",) if section.find(w) >= 0] or [10**9])
    if i_stop == i_bet == 10**9:
        return "ambiguous", f"neither token: {section[:40]!r}"
    return ("stop" if i_stop < i_bet else "bet"), f"first token wins: {section[:40]!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--globs", nargs="+", default=[
        "/home/v-seungplee/data/llm-addiction/track0_rerun/final_*.json",
        "/home/v-seungplee/data/llm-addiction/track0_w3/final_*.json",
        "/home/v-seungplee/data/llm-addiction/e7_factorial/e7_*.json",
    ])
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/reparse_audit.json")
    args = ap.parse_args()

    report = {}
    for pattern in args.globs:
        corpus = Path(pattern).parent.name
        agg = collections.Counter()
        per_cell = {}
        for path in sorted(glob.glob(pattern)):
            payload = json.load(open(path))
            games = payload.get("results", [])
            cell = payload.get("cell") or f"{payload.get('model')}_cap{payload.get('cap')}_{payload.get('mode')}"
            c = collections.Counter()
            for game in games:
                for rnd in game.get("rounds", []):
                    stored = rnd.get("response") or ""
                    recorded = rnd.get("decision")
                    c["decisions"] += 1
                    if len(stored) == TRUNCATED:
                        c["truncated"] += 1
                        continue
                    new, _why = corrected_decision(stored)
                    if new in ("no_marker", "ambiguous"):
                        c["unadjudicable"] += 1
                        continue
                    old = "stop" if recorded == "stop" else "bet"
                    if new != old:
                        c[f"flip_{old}_to_{new}"] += 1
            per_cell[cell] = dict(c)
            agg.update(c)
        report[corpus] = {"total": dict(agg), "cells": per_cell}

        print(f"=== {corpus} ===")
        d = agg["decisions"]
        print(f"  decisions {d}  truncated {agg['truncated']}  unadjudicable {agg['unadjudicable']}")
        print(f"  bet -> stop  {agg['flip_bet_to_stop']}")
        print(f"  stop -> bet  {agg['flip_stop_to_bet']}")
        adjudicable = d - agg["truncated"] - agg["unadjudicable"]
        flips = agg["flip_bet_to_stop"] + agg["flip_stop_to_bet"]
        if adjudicable:
            print(f"  flip rate on adjudicable decisions: {100 * flips / adjudicable:.3f}%")
        worst = sorted(
            ((k, v) for k, v in per_cell.items() if v.get("flip_bet_to_stop", 0) or v.get("flip_stop_to_bet", 0)),
            key=lambda kv: -(kv[1].get("flip_bet_to_stop", 0) + kv[1].get("flip_stop_to_bet", 0)),
        )[:8]
        for cell, c in worst:
            print(f"    {cell:<48} bet->stop {c.get('flip_bet_to_stop',0):>4}  stop->bet {c.get('flip_stop_to_bet',0):>4}")
        print()

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
