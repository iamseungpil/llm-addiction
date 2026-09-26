"""Sanity checks per Plan v4 §1bis.3.

The "freedom-to-choose at root" mechanism, if real, predicts two model-level signatures
under the matched-cap protocol:

  S1. Variable-mode mean bet < cap value at every cap > $10. The model uses
      discretion below the cap rather than betting the maximum (the "freedom not
      range" signature).
  S2. Variable-mode mean game length > Fixed-mode mean game length at matched cap
      (more rounds at smaller bets).

This script outputs a markdown table summarising per-model PASS/FAIL.

A model can pass S1 and fail S2 (or vice-versa); the markdown reports both so the
reviewer/analyst can read the full picture.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List


def load_long_records(input_dir: str) -> List[Dict]:
    rows = []
    for path in sorted(Path(input_dir).glob("final_*.json")):
        with open(path, "r") as f:
            payload = json.load(f)
        model = payload.get("model")
        cap = payload.get("cap")
        mode = payload.get("mode")
        for record in payload.get("results", []):
            mean_bet = None
            if record.get("history"):
                mean_bet = mean(h["bet"] for h in record["history"])
            rows.append({
                "model": model,
                "cap": cap,
                "mode": mode,
                "mean_bet": mean_bet,
                "n_rounds": record.get("total_rounds"),
            })
    return rows


def compute_per_model(rows: List[Dict]) -> List[Dict]:
    bucket = defaultdict(lambda: {"variable_mean_bets_by_cap": defaultdict(list),
                                  "variable_rounds_by_cap": defaultdict(list),
                                  "fixed_rounds_by_cap": defaultdict(list)})
    for r in rows:
        m, c, mode = r["model"], r["cap"], r["mode"]
        if r["mean_bet"] is not None and mode == "variable":
            bucket[m]["variable_mean_bets_by_cap"][c].append(r["mean_bet"])
        if r["n_rounds"] is not None:
            if mode == "variable":
                bucket[m]["variable_rounds_by_cap"][c].append(r["n_rounds"])
            elif mode == "fixed":
                bucket[m]["fixed_rounds_by_cap"][c].append(r["n_rounds"])

    out = []
    for m, b in sorted(bucket.items()):
        # S1: variable mean bet < cap value at every cap > 10
        s1_pass = True
        s1_detail = {}
        for c, vals in sorted(b["variable_mean_bets_by_cap"].items()):
            if c is None or vals == []:
                continue
            mb = mean(vals)
            s1_detail[c] = mb
            if c > 10 and not (mb < c):
                s1_pass = False

        # S2: variable_rounds > fixed_rounds at each matched cap
        s2_pass = True
        s2_detail = {}
        for c in sorted(set(list(b["variable_rounds_by_cap"].keys()) + list(b["fixed_rounds_by_cap"].keys()))):
            if c is None:
                continue
            v = b["variable_rounds_by_cap"].get(c, [])
            f = b["fixed_rounds_by_cap"].get(c, [])
            if not v or not f:
                continue
            vm, fm = mean(v), mean(f)
            s2_detail[c] = (vm, fm)
            if not (vm > fm):
                s2_pass = False

        out.append({
            "model": m,
            "S1_freedom_not_range": s1_pass,
            "S1_variable_mean_bet_by_cap": s1_detail,
            "S2_more_rounds_under_variable": s2_pass,
            "S2_var_vs_fix_rounds_by_cap": s2_detail,
        })
    return out


def render_markdown(per_model: List[Dict]) -> str:
    lines = []
    lines.append("# Track 0 sanity checks")
    lines.append("")
    lines.append("S1: variable-mode mean bet < cap at every cap > $10  (freedom-not-range)")
    lines.append("S2: variable-mode mean game length > fixed-mode mean game length at matched cap")
    lines.append("")
    lines.append("| model | S1 | S2 | mean_bet@cap30 | mean_bet@cap50 | mean_bet@cap70 |")
    lines.append("|---|---|---|---|---|---|")
    for r in per_model:
        s1 = "PASS" if r["S1_freedom_not_range"] else "FAIL"
        s2 = "PASS" if r["S2_more_rounds_under_variable"] else "FAIL"
        mb = r["S1_variable_mean_bet_by_cap"]
        cells = [f"{mb.get(c, float('nan')):.1f}" if isinstance(mb.get(c), (int, float)) else "n/a" for c in (30, 50, 70)]
        lines.append(f"| {r['model']} | {s1} | {s2} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    lines.append("Per-model S2 detail (variable_rounds, fixed_rounds) by cap:")
    for r in per_model:
        detail = r["S2_var_vs_fix_rounds_by_cap"]
        if not detail:
            continue
        lines.append(f"- {r['model']}: " + "; ".join(f"cap${c}: var={v:.1f} vs fix={f:.1f}" for c, (v, f) in sorted(detail.items())))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    rows = load_long_records(args.input_dir)
    per_model = compute_per_model(rows)
    md = render_markdown(per_model)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"[sanity_checks] wrote {out}")
    print(md)


if __name__ == "__main__":
    main()
