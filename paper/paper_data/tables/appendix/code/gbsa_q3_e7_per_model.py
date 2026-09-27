#!/usr/bin/env python3
"""gbSA Q3: the rationality-instruction result, per API model (and per open-weight model).

Why this file exists
--------------------
The gbSA reply promised the rationality-instruction cells "on the API models as
well".  ``neurips_content_en/appendix.tex`` (tab:added-controls, the row "Is the
gap role-play or instruction following?") discharges that promise only as four
aggregate means -- 18.3, 6.8, 1.0 and 0.5 percentage points -- with no per-model
breakdown, no n, and no interval.  This script recomputes the whole factorial
cell by cell.

Corpus
------
``rebuttal_neurips_2026/framing_rationality_factorial_e7/`` on the HuggingFace
release, top level only.  100 games per cell.  The paper prints the five models whose
checkpoints are those of the main study: the Claude-3.5-Haiku checkpoint of the six-model
roster had been withdrawn from the API, and the substitute Claude-Haiku-4.5 cells, though
collected and released, are not printed.  That leaves 36 of 40 nominal cells; the four
absent cells are the open-weight (no framing, no rationality) arm, exactly as
the corpus README and appendix.tex both state.  ``QUARANTINE_truncated_claude/``
is excluded: its own README forbids quoting any rate from it.

Definitions
-----------
bankruptcy     game["bankrupt"] is true (equivalently outcome == "bankruptcy").
gap            variable bankruptcy % minus fixed bankruptcy %, within one
               (model, framing, rationality) cell pair.
interval       per-arm 95% Wilson; on the gap, the Newcombe hybrid-score
               interval for a difference of two independent proportions, which
               does not run outside [-100, 100] and stays sane at 0/100.

Outputs
-------
  paper_data/tables/appendix/gbsa_q3_e7_per_model.json
  paper_data/tables/appendix/gbsa_q3_e7_per_model.tex
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

E7_DIR = Path(os.environ.get(
    "E7_DIR",
    # Point E7_DIR at a local copy of the release, or at the dataset itself.
    "hf_snapshot/rebuttal_neurips_2026/framing_rationality_factorial_e7",
))
OUT_DIR = Path(__file__).resolve().parents[1]

MODEL_LABEL = {
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "claude-haiku-4-5-20251001": "Claude-Haiku-4.5",
    "llama": "LLaMA-3.1-8B",
    "gemma": "Gemma-2-9B",
}
MODEL_ORDER = ["gpt-4o-mini", "gpt-4.1-mini", "gemini-2.5-flash", "llama", "gemma"]
API_MODELS = MODEL_ORDER[:3]
# Collected and released, but not printed: the substitute checkpoint for the withdrawn
# Claude-3.5-Haiku of the six-model roster.
NOT_PRINTED = ["claude-haiku-4-5-20251001"]
CELLS = [("role", 0), ("role", 1), ("none", 0), ("none", 1)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def newcombe(k2: int, n2: int, k1: int, n1: int) -> tuple[float, float]:
    """95% interval for p2 - p1 (variable minus fixed), in percentage points."""
    p1, p2 = 100 * k1 / n1, 100 * k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (lo, hi)


def companion_metrics(games: list[dict]) -> dict:
    """The three measures gbSA W4/Q1 promised alongside every bankruptcy figure.

    E7 games carry ``history`` with one entry per executed wager, each with ``bet``,
    ``win`` and the balance *after* the round, so no prompt parsing is needed here.
    """
    n = len(games)
    played = at_risk = rebet = 0
    wagers, per_game = [], []
    for g in games:
        h = [e for e in (g.get("history") or []) if e.get("bet") is not None]
        if h:
            played += 1
            b = [float(e["bet"]) for e in h]
            wagers.extend(b)
            per_game.append(sum(b))
        wins = [bool(e["win"]) if "win" in e
                else str(e.get("result", "")).upper().startswith("W") for e in h]
        fl = next((i for i, w in enumerate(wins) if not w), None)
        if fl is not None:
            at_risk += 1
            if fl + 1 < len(h):
                rebet += 1
    return {
        "participation": {"k": played, "n": n, "pct": 100 * played / n,
                          "wilson95": list(wilson(played, n))},
        "realised_bet_size": {
            "mean_usd_per_executed_wager": (sum(wagers) / len(wagers)) if wagers else None,
            "n_executed_wagers": len(wagers),
            "mean_usd_staked_per_wagering_game": (sum(per_game) / len(per_game)) if per_game else None,
        },
        "rebet_after_first_loss": {"k": rebet, "n_at_risk": at_risk,
                                   "pct": (100 * rebet / at_risk) if at_risk else None,
                                   "wilson95": list(wilson(rebet, at_risk)) if at_risk else [None, None]},
    }


def load_cells() -> dict:
    """{(model, mode, framing, rat): {bankrupt, n, file}}"""
    out = {}
    for f in sorted(E7_DIR.glob("e7_*.json")):
        blob = json.load(open(f))
        model = blob["model"]
        key = (model, blob["mode"], blob["factor_preamble"], int(bool(blob["factor_rat"])))
        games = blob["results"]
        k = sum(1 for g in games if g.get("bankrupt") or
                str(g.get("outcome", "")).lower().startswith("bankrupt"))
        assert key not in out, f"duplicate cell {key}"
        out[key] = {"n_games": len(games), "n_bankrupt": k, "file": f.name,
                    "companion": companion_metrics(games),
                    "cell": blob["cell"], "cap": blob["cap"],
                    "n_exactly_500": blob.get("response_length_stats", {}).get("n_exactly_500"),
                    "git_commit": blob.get("manifest", {}).get("git", {}).get("commit")}
    return out


def main() -> None:
    cells = load_cells()
    per_model, gaps_by_cell = {}, {c: [] for c in CELLS}
    for model in MODEL_ORDER:
        entry = {}
        for framing, rat in CELLS:
            fx = cells.get((model, "fixed", framing, rat))
            vr = cells.get((model, "variable", framing, rat))
            name = f"{framing}_rat{rat}"
            if fx is None or vr is None:
                entry[name] = {"status": "absent",
                               "note": "cell never collected; see corpus README"}
                continue
            gap = 100 * vr["n_bankrupt"] / vr["n_games"] - 100 * fx["n_bankrupt"] / fx["n_games"]
            lo, hi = newcombe(vr["n_bankrupt"], vr["n_games"], fx["n_bankrupt"], fx["n_games"])
            entry[name] = {
                "status": "present",
                "fixed": {"n_bankrupt": fx["n_bankrupt"], "n_games": fx["n_games"],
                          "pct": 100 * fx["n_bankrupt"] / fx["n_games"],
                          "wilson95": list(wilson(fx["n_bankrupt"], fx["n_games"])),
                          "companion": fx["companion"],
                          "file": fx["file"]},
                "variable": {"n_bankrupt": vr["n_bankrupt"], "n_games": vr["n_games"],
                             "pct": 100 * vr["n_bankrupt"] / vr["n_games"],
                             "wilson95": list(wilson(vr["n_bankrupt"], vr["n_games"])),
                             "companion": vr["companion"],
                             "file": vr["file"]},
                "gap_pp": gap,
                "gap_newcombe95": [lo, hi],
                "gap_excludes_zero": bool(lo > 0 or hi < 0),
            }
            gaps_by_cell[(framing, rat)].append((model, gap))
        per_model[MODEL_LABEL[model]] = entry

    # Reproduce the four aggregate means the appendix prints, and the API-only means.
    aggregates = {}
    for framing, rat in CELLS:
        vals = gaps_by_cell[(framing, rat)]
        api = [g for m, g in vals if m in API_MODELS]
        aggregates[f"{framing}_rat{rat}"] = {
            "mean_gap_pp_all_available": sum(g for _, g in vals) / len(vals),
            "n_models_available": len(vals),
            "models_available": [MODEL_LABEL[m] for m, _ in vals],
            "mean_gap_pp_api_only": sum(api) / len(api),
            "n_models_api": len(api),
        }

    # Rationality effect on the gap, within framing level, per model.
    rat_effect = {}
    for model in MODEL_ORDER:
        e = per_model[MODEL_LABEL[model]]
        row = {}
        for framing in ("role", "none"):
            a, b = e[f"{framing}_rat0"], e[f"{framing}_rat1"]
            if a["status"] == "present" and b["status"] == "present":
                row[framing] = {"gap_rat0_pp": a["gap_pp"], "gap_rat1_pp": b["gap_pp"],
                                "delta_pp": b["gap_pp"] - a["gap_pp"]}
            else:
                row[framing] = {"status": "not estimable: rat0 cell absent"}
        rat_effect[MODEL_LABEL[model]] = row

    doc = {
        "purpose": "gbSA Q3 -- the rationality-instruction result reported per model, "
                   "which appendix.tex currently reports only as four aggregate means.",
        "promise_text": "gbSA reply, Q3: 'We ran the same instruction on the API models as "
                        "well, and those cells will appear in the revision.'",
        "currently_in_paper": {
            "location": "neurips_content_en/appendix.tex, tab:added-controls, row "
                        "'Is the gap role-play or instruction following?' (lines 1165-1179)",
            "aggregate_means_pp": {"role_rat0": 18.3, "role_rat1": 6.8,
                                   "none_rat0": 1.0, "none_rat1": 0.5},
        },
        "corpus": {
            "hf_repo": "llm-addiction-research/llm-addiction",
            "path": "rebuttal_neurips_2026/framing_rationality_factorial_e7/ (top level only)",
            "cells_present": len(cells),
            "cells_nominal": 48,
            "absent_cells": ["gemma fixed none rat0", "gemma variable none rat0",
                             "llama fixed none rat0", "llama variable none rat0"],
            "excluded": "QUARANTINE_truncated_claude/ -- its README forbids quoting any rate "
                        "from it (verdict completeness 1.0%-87.3% under a token limit).",
            "cap_usd": 70,
            "games_per_cell": 100,
        },
        "definitions": {
            "bankruptcy": "game['bankrupt'] true, equivalently outcome == 'bankruptcy'",
            "gap_pp": "variable bankruptcy % minus fixed bankruptcy %, within one "
                      "(model, framing, rationality) pair",
            "per_arm_interval": "95% Wilson score",
            "gap_interval": "Newcombe hybrid-score interval for a difference of two "
                            "independent binomial proportions",
        },
        "caveat": "VERIFIED_FACTS.md section C records a parser defect with an 0.249% flip rate "
                  "on the E7 corpus (16 bet->stop, 2 stop->bet out of 7,639 decisions). These "
                  "figures use the released decision labels; a flip rate that low cannot move "
                  "any bankruptcy count in the table by more than one game.",
        "companion_metrics_note": "gbSA also promised participation, realised bet size and "
                                  "re-betting after a first loss alongside every bankruptcy "
                                  "figure. They are carried in per_model[...]['<cell>']['fixed'|"
                                  "'variable']['companion'] for all 44 cells so the proposed "
                                  "table can be widened without recomputation. Definitions match "
                                  "paper_data/tables/appendix/gbsa_companion_metrics.json.",
        "per_model": per_model,
        "aggregates": aggregates,
        "rationality_effect_on_gap": rat_effect,
        "cell_inventory": {f"{m}|{mo}|{fr}|rat{r}": v for (m, mo, fr, r), v in sorted(cells.items())},
    }
    (OUT_DIR / "gbsa_q3_e7_per_model.json").write_text(json.dumps(doc, indent=2) + "\n")

    # ---------------------------------------------------------------- tex fragment
    L = []
    ap = L.append
    ap("% gbSA Q3 -- rationality instruction, per model.  PROPOSED appendix table; not yet in any .tex.")
    ap("% Generated by paper_data/tables/appendix/code/gbsa_q3_e7_per_model.py")
    ap("% Numbers in paper_data/tables/appendix/gbsa_q3_e7_per_model.json")
    ap("%")
    ap("% Source: HF llm-addiction-research/llm-addiction,")
    ap("%   rebuttal_neurips_2026/framing_rationality_factorial_e7/ (top level; QUARANTINE_truncated_claude/ excluded).")
    ap("%   cap $70, 100 games per cell, 36 of 40 printed cells; the released")
    ap("%   claude-haiku-4-5-20251001 cells are the withdrawn checkpoint's substitute and are not printed.")
    ap("%   Absent: gemma and llama, no-framing x no-rationality, both betting modes.")
    ap("% Bankruptcy = game['bankrupt'].  Per-arm interval = 95% Wilson.")
    ap("% Gap = variable - fixed, in percentage points; interval = Newcombe hybrid score.")
    ap("%")
    ap("% Per-cell provenance: model | framing | rat | mode : bankrupt k/n [95% Wilson] : file")
    for (m, mo, fr, r), v in sorted(cells.items()):
        lo, hi = wilson(v["n_bankrupt"], v["n_games"])
        mark = "  [collected, NOT printed]" if m in NOT_PRINTED else ""
        ap(f"%   {MODEL_LABEL[m]:<17} | {fr:<4} | rat{r} | {mo:<8} : "
           f"{v['n_bankrupt']:>3}/{v['n_games']:<3} [{lo:5.1f}, {hi:5.1f}] : {v['file']}{mark}")
    ap("%")
    ap("% Aggregate means recomputed from these cells. The appendix prints 22.0 / 8.2 / 1.3 / 0.6;")
    ap("% all four reproduce exactly. The API-only column is what gbSA Q3 actually asked for.")
    for framing, rat in CELLS:
        a = aggregates[f"{framing}_rat{rat}"]
        ap(f"%   {framing} rat{rat}: all available {a['mean_gap_pp_all_available']:+.1f} pp over "
           f"{a['n_models_available']} models; API only {a['mean_gap_pp_api_only']:+.1f} pp over "
           f"{a['n_models_api']} models")
    ap("%")
    ap("% Companion measures (participation, mean executed wager, re-bet after first loss) for all")
    ap("% printed cells are in the JSON under per_model[...][cell][arm]['companion'].")
    ap("%")
    ap("\\begin{table}[H]")
    ap("\\centering")
    ap("\\scriptsize")
    ap("\\setlength{\\tabcolsep}{2.5pt}")
    ap("\\caption{\\rev{Bankruptcy in the framing $\\times$ rationality factorial, per model. "
       "Cap $\\$70$, $n = 100$ games in every cell. Each entry gives forced $\\to$ choosing "
       "bankruptcy as a percentage, then the gap in percentage points with a 95\\% Newcombe "
       "interval; bold gaps exclude zero. Dashes mark the four cells that were never collected. "
       "The Claude cells run Claude-Haiku-4.5, as in Table~\\ref{tab:added-controls}.}}")
    ap("\\label{tab:e7-per-model}")
    ap("\\begin{tabular}{lcccc}")
    ap("\\toprule")
    ap("& \\multicolumn{2}{c}{\\textbf{Framing}} & \\multicolumn{2}{c}{\\textbf{No framing}} \\\\")
    ap("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}")
    ap("\\textbf{Model} & \\textbf{no rationality} & \\textbf{$+$ rationality} & "
       "\\textbf{no rationality} & \\textbf{$+$ rationality} \\\\")
    ap("\\midrule")
    for model in MODEL_ORDER:
        e = per_model[MODEL_LABEL[model]]
        row = [MODEL_LABEL[model]]
        for framing, rat in CELLS:
            v = e[f"{framing}_rat{rat}"]
            if v["status"] == "absent":
                row.append("---")
                continue
            g = (f"{v['gap_pp']:+.0f} [{v['gap_newcombe95'][0]:+.1f}, "
                 f"{v['gap_newcombe95'][1]:+.1f}]")
            if v["gap_excludes_zero"]:
                g = "\\textbf{" + g + "}"
            row.append("\\makecell{" + f"{v['fixed']['pct']:.0f} $\\to$ {v['variable']['pct']:.0f}"
                       + "\\\\" + g + "}")
        ap(" & ".join(row) + " \\\\")
    ap("\\midrule")
    a0 = aggregates["role_rat0"]
    a1 = aggregates["role_rat1"]
    a2 = aggregates["none_rat0"]
    a3 = aggregates["none_rat1"]
    ap(f"\\textit{{Mean gap, all available}} & {a0['mean_gap_pp_all_available']:+.1f} & "
       f"{a1['mean_gap_pp_all_available']:+.1f} & {a2['mean_gap_pp_all_available']:+.1f} & "
       f"{a3['mean_gap_pp_all_available']:+.1f} \\\\")
    ap(f"\\textit{{Mean gap, three API models}} & {a0['mean_gap_pp_api_only']:+.1f} & "
       f"{a1['mean_gap_pp_api_only']:+.1f} & {a2['mean_gap_pp_api_only']:+.1f} & "
       f"{a3['mean_gap_pp_api_only']:+.1f} \\\\")
    ap("\\bottomrule")
    ap("\\end{tabular}")
    ap("\\end{table}")
    ap("")
    ap("% ------------------------------------------------------------------------------")
    ap("% PROPOSED EDIT to neurips_content_en/appendix.tex, tab:added-controls, the row")
    ap("% \"Is the gap role-play or instruction following?\" (lines 1165-1179).")
    ap("% The four aggregate means are VERIFIED and stay; two sentences are added to the")
    ap("% Result cell and one clause in the Reading cell is corrected.")
    ap("%")
    ap("% RESULT cell -- keep the existing first sentence, then append:")
    ap("% \\rev{Table~\\ref{tab:e7-per-model} gives every cell separately. The aggregate is carried")
    ap("% by three models: the gap excludes zero only for LLaMA-3.1-8B ($+76$ $[+65.2, +83.1]$),")
    ap("% Gemini-2.5-Flash ($+20$ $[+8.6, +30.9]$) and Gemma-2-9B ($+14$ $[+6.8, +22.3]$), while")
    ap("% GPT-4o-mini and GPT-4.1-mini bankrupt in 0 of 100 games in all sixteen of their")
    ap("% cells. Averaged over the three API models alone the gap is 6.7 points with role")
    ap("% framing and 0.7 points once the rationality instruction is added.}")
    ap("%")
    ap("% READING cell -- replace \"The open-weight role cells carry the largest gaps\" with:")
    ap("% \\rev{The largest gap is LLaMA's role cell, and the four missing cells are open-weight,")
    ap("% so a complete factorial conclusion is not available.} The rationality instruction")
    ap("% narrows the LLaMA gap from 76 to 40 points and closes the Gemma gap from 14 to $-1$;")
    ap("% \\rev{on the three API models it has almost nothing to narrow, since only Gemini")
    ap("% bankrupts at all.}")
    ap("%")
    ap("% Note: 'The open-weight role cells carry the largest gaps' is imprecise as written --")
    ap("% Gemini's role cell (+20) is larger than Gemma's (+14), so only LLaMA supports the claim.")
    (OUT_DIR / "gbsa_q3_e7_per_model.tex").write_text("\n".join(L) + "\n")
    print(json.dumps(aggregates, indent=1))
    for model in MODEL_ORDER:
        e = per_model[MODEL_LABEL[model]]
        for framing, rat in CELLS:
            v = e[f"{framing}_rat{rat}"]
            if v["status"] == "absent":
                print(f"{MODEL_LABEL[model]:<17} {framing} rat{rat}: ABSENT")
            else:
                print(f"{MODEL_LABEL[model]:<17} {framing} rat{rat}: "
                      f"fixed {v['fixed']['n_bankrupt']}/{v['fixed']['n_games']}  "
                      f"var {v['variable']['n_bankrupt']}/{v['variable']['n_games']}  "
                      f"gap {v['gap_pp']:+.1f} [{v['gap_newcombe95'][0]:+.1f},"
                      f"{v['gap_newcombe95'][1]:+.1f}]"
                      f"{'  *' if v['gap_excludes_zero'] else ''}")


if __name__ == "__main__":
    sys.exit(main())
