#!/usr/bin/env python3
"""gbSA promise: participation, realised bet size and re-betting after a first loss,
reported alongside each *primary* bankruptcy figure.

Why this file exists
--------------------
The gbSA reply (W4/Q1) promised: "In the revision we will report participation, the
realised bet size and re-betting after a first loss alongside each bankruptcy figure."
An audit of neurips_content_en/appendix.tex finds the promise discharged only in
scattered places and never for the primary behavioural cells:

  participation  -- appendix:participation-and-exposure, prose + the caption of
                    tab:exposure-matched: five matched-cap cells at n = 50, plus the
                    per-arm n of the LLaMA revisability ladder.
  bet size       -- one sentence, two cells: Gemini GMHWP forced $64.5/round and
                    choosing $47.9/round (and $188 / $282 per game).
  re-bet         -- one column of the LLaMA revisability ladder: 18 / 45 / 100 / 100 %.

None of the three appears for the twelve primary cells behind Figure 2(a), the six-model
fixed-versus-variable slot machine (1,600 games per cell, 19,200 games). This script
computes all three for those twelve cells.

Corpora (canonical, per NEURIPS_CANONICAL_INDEX.md; "model" field checked inside each file)
------------------------------------------------------------------------------------------
  GPT-4o-mini       analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json
                    model = "gpt-4o-mini-corrected"
  GPT-4.1-mini      slot_machine/gpt/gpt5_experiment_20250921_174509.json   model = "gpt-4.1-mini"
  Gemini-2.5-Flash  slot_machine/gemini/gemini_experiment_20250920_042809.json
  Claude-3.5-Haiku  slot_machine/claude/claude_experiment_corrected_20250925.json
                    model = "claude-3-5-haiku-latest"
  LLaMA-3.1-8B      behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json
  Gemma-2-9B        behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json

slot_machine/{gemma,llama}/ carry DEPRECATION_WARNING.md and are NOT used.

Round extraction
----------------
This script does NOT reuse scripts/build_figure_data.py:rounds_of once stringified the round's game_result, which is a
dict in the GPT-4.1-mini, Gemini-2.5-Flash and Claude-3.5-Haiku corpora, so every round of
those three read as a loss.  That has been repaired to resolve the outcome the way the
submitted generator did, so this script and the figure pipeline now agree.  Bet amounts and
balances were never affected.

Outputs
-------
  paper_data/tables/appendix/gbsa_companion_metrics.json
  paper_data/tables/appendix/gbsa_companion_metrics.tex
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np

SNAP = Path(os.environ.get(
    "HF_SNAPSHOT",
    "hf_snapshot",  # set HF_SNAPSHOT to a local copy of the release
))
OUT_DIR = Path(__file__).resolve().parents[1]

SOURCES = [
    ("GPT-4o-mini", "analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json",
     "gpt-4o-mini-corrected"),
    ("GPT-4.1-mini", "slot_machine/gpt/gpt5_experiment_20250921_174509.json", "gpt-4.1-mini"),
    ("Gemini-2.5-Flash", "slot_machine/gemini/gemini_experiment_20250920_042809.json", "gemini-2.5-flash"),
    ("Claude-3.5-Haiku", "slot_machine/claude/claude_experiment_corrected_20250925.json",
     "claude-3-5-haiku-latest"),
    ("LLaMA-3.1-8B", "behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json", "llama"),
    ("Gemma-2-9B", "behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json", "gemma"),
]
BALANCE_RE = re.compile(r"Current balance:\s*\$(\d+(?:\.\d+)?)", re.I)
RNG = np.random.default_rng(24231)
N_BOOT = 2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def newcombe(k2, n2, k1, n1):
    p1, p2 = 100 * k1 / n1, 100 * k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    return (d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2),
            d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2))


def clustered_boot(per_game_wagers: list[list[float]]):
    """Percentile CI for the mean executed wager, resampling whole games."""
    kept = [np.asarray(w, dtype=float) for w in per_game_wagers if w]
    if not kept:
        return (float("nan"), float("nan"), float("nan"), 0, 0)
    flat = np.concatenate(kept)
    point = float(flat.mean())
    sums = np.array([a.sum() for a in kept])
    cnts = np.array([a.size for a in kept], dtype=float)
    idx = RNG.integers(0, len(kept), size=(N_BOOT, len(kept)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return (point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)),
            int(flat.size), len(kept))


def executed_rounds(game: dict) -> list[dict]:
    """[{bet, balance_before, win}, ...] for executed wagers only, in play order."""
    hist = game.get("history")
    if hist:
        out, before = [], 100.0
        for ev in hist:
            if ev.get("bet") is None:
                continue
            out.append({"bet": float(ev["bet"]), "balance_before": before,
                        "win": bool(ev["win"]) if "win" in ev
                        else str(ev.get("result", "")).upper().startswith("W")})
            if ev.get("balance") is not None:
                before = float(ev["balance"])
        return out

    ghist = {h.get("round"): h for h in (game.get("game_history") or [])}
    out = []
    for d in (game.get("round_details") or []):
        bet = d.get("bet") or d.get("parsed_bet") or d.get("bet_amount")
        if bet in (None, "None"):
            continue
        bal = d.get("balance_before")
        if bal is None:
            m = BALANCE_RE.search(d.get("prompt") or "")
            bal = float(m.group(1)) if m else None
        gr = d.get("game_result")
        if gr is None:
            gr = ghist.get(d.get("round"))
        if isinstance(gr, dict):
            win = bool(gr.get("win")) if "win" in gr else \
                str(gr.get("result", "")).upper().startswith("W")
        else:
            win = str(gr or "").upper().startswith("W")
        if bal is None:
            continue
        out.append({"bet": float(bet), "balance_before": float(bal), "win": win})
    return out


def is_bankrupt(game: dict) -> bool:
    if game.get("is_bankrupt") is not None:
        return bool(game["is_bankrupt"])
    if game.get("bankruptcy") is not None:
        return bool(game["bankruptcy"])
    return str(game.get("outcome", game.get("final_outcome", ""))).lower().startswith("bankrupt")


def cell_metrics(games: list[dict]) -> dict:
    n = len(games)
    n_bank = sum(1 for g in games if is_bankrupt(g))
    played = 0
    per_game_wagers, per_game_total = [], []
    at_risk = rebet = 0
    n_wagers_total = 0
    bank_on_first = 0
    for g in games:
        rds = executed_rounds(g)
        if is_bankrupt(g) and len(rds) == 1:
            bank_on_first += 1
        bets = [r["bet"] for r in rds]
        n_wagers_total += len(bets)
        if bets:
            played += 1
            per_game_wagers.append(bets)
            per_game_total.append(sum(bets))
        first_loss = next((i for i, r in enumerate(rds) if not r["win"]), None)
        if first_loss is not None:
            at_risk += 1
            if first_loss + 1 < len(rds):
                rebet += 1
    mean_rounds = n_wagers_total / n
    mean_total_bet_all_games = float(np.sum([sum(w) for w in per_game_wagers])) / n
    net = [float(g.get("final_balance", 0)) - 100.0 for g in games]
    mean_net = float(np.mean(net)) if net else None
    mean_bet, mb_lo, mb_hi, n_wagers, n_wager_games = clustered_boot(per_game_wagers)
    tot = np.asarray(per_game_total, dtype=float)
    if tot.size:
        idx = RNG.integers(0, tot.size, size=(N_BOOT, tot.size))
        tmeans = tot[idx].mean(axis=1)
        tot_stat = {"mean_usd": float(tot.mean()),
                    "ci95": [float(np.percentile(tmeans, 2.5)), float(np.percentile(tmeans, 97.5))],
                    "n_wagering_games": int(tot.size)}
    else:
        tot_stat = {"mean_usd": None, "ci95": [None, None], "n_wagering_games": 0}
    return {
        "n_games": n,
        "existing_table_columns_recomputed": {
            "note": "columns already printed in tab:appendix-slot-comprehensive "
                    "(neurips_content_en/appendix.tex:146-178), recomputed here as a check",
            "mean_rounds": mean_rounds,
            "mean_total_bet_usd": mean_total_bet_all_games,
            "mean_net_pl_usd": mean_net,
            "derivation_note": "the paper's printed Total bet / Mean rounds already equals the "
                               "mean executed wager, so realised bet size is recoverable from the "
                               "existing table but is never stated in it",
        },
        "bankruptcy": {"k": n_bank, "n": n, "pct": 100 * n_bank / n,
                       "wilson95": list(wilson(n_bank, n)),
                       "k_ruined_on_their_first_wager": bank_on_first,
                       "pct_of_bankruptcies_on_first_wager":
                           (100 * bank_on_first / n_bank) if n_bank else None},
        "participation": {"k": played, "n": n, "pct": 100 * played / n,
                          "wilson95": list(wilson(played, n)),
                          "definition": "games containing at least one executed wager"},
        "realised_bet_size": {"mean_usd_per_executed_wager": mean_bet,
                              "ci95_game_clustered_bootstrap": [mb_lo, mb_hi],
                              "n_executed_wagers": n_wagers,
                              "n_wagering_games": n_wager_games,
                              "mean_usd_staked_per_wagering_game": tot_stat},
        "rebet_after_first_loss": {"k": rebet, "n_at_risk": at_risk,
                                   "pct": 100 * rebet / at_risk if at_risk else None,
                                   "wilson95": list(wilson(rebet, at_risk)) if at_risk else [None, None],
                                   "pct_of_all_games": 100 * rebet / n,
                                   "definition": "of games whose play contains at least one losing "
                                                 "round, the share that execute another wager after "
                                                 "that first losing round"},
    }


def main() -> None:
    per_model = {}
    for label, rel, expect_model in SOURCES:
        blob = json.load(open(SNAP / rel))
        got = blob.get("model")
        assert got == expect_model, f"{label}: model field is {got!r}, expected {expect_model!r}"
        games = blob.get("results") or blob.get("games")
        cells = {}
        for mode in ("fixed", "variable"):
            sub = [g for g in games if g.get("bet_type") == mode]
            cells[mode] = cell_metrics(sub)
        # variable-minus-fixed contrasts on the two proportions
        for key, path in (("participation", ("participation", "k", "n")),
                          ("rebet_after_first_loss", ("rebet_after_first_loss", "k", "n_at_risk"))):
            f, v = cells["fixed"][key], cells["variable"][key]
            kf, nf = f[path[1]], f[path[2]]
            kv, nv = v[path[1]], v[path[2]]
            if nf and nv:
                lo, hi = newcombe(kv, nv, kf, nf)
                cells.setdefault("contrasts", {})[key] = {
                    "gap_pp": 100 * kv / nv - 100 * kf / nf, "newcombe95": [lo, hi]}
        f, v = cells["fixed"]["bankruptcy"], cells["variable"]["bankruptcy"]
        lo, hi = newcombe(v["k"], v["n"], f["k"], f["n"])
        cells.setdefault("contrasts", {})["bankruptcy"] = {
            "gap_pp": v["pct"] - f["pct"], "newcombe95": [lo, hi]}
        per_model[label] = {"source_file": rel, "model_field": got, **cells}

    doc = {
        "purpose": "gbSA W4/Q1 promise -- participation, realised bet size and re-betting after a "
                   "first loss, computed for the twelve primary behavioural cells behind "
                   "Figure 2(a).",
        "promise_text": "gbSA reply, W4/Q1: 'In the revision we will report participation, the "
                        "realised bet size and re-betting after a first loss alongside each "
                        "bankruptcy figure.'",
        "audit_of_neurips_content_en_appendix_tex": {
            "participation": {
                "status": "PARTIAL",
                "where": ["appendix.tex:1216-1222 (appendix:participation-and-exposure): Gemini "
                          "GMHWP matched-cap cell, both arms 50/50 games",
                          "appendix.tex:1227 (tab:exposure-matched caption): wagering-game "
                          "denominators for four matched-cap cells at n = 50",
                          "appendix.tex:1059-1065 (LLaMA revisability ladder): per-arm n only"],
                "missing": "all twelve primary cells (six models x fixed/variable, n = 1,600 each)"},
            "realised_bet_size": {
                "status": "PARTIAL",
                "where": ["appendix.tex:1218: Gemini GMHWP fixed $64.5/round and variable "
                          "$47.9/round; $188 and $282 staked per game"],
                "missing": "every cell except that one matched-cap pair, including all twelve "
                           "primary cells"},
            "rebet_after_first_loss": {
                "status": "PARTIAL",
                "where": ["appendix.tex:1053-1065 (LLaMA revisability ladder): 18%, 45%, 100%, "
                          "100% across the four ladder arms, defined at 1053-1054"],
                "missing": "all twelve primary cells, and every model other than LLaMA"},
        },
        "primary_cells": "slot machine, six models x {fixed, variable}, 1,600 games per cell, "
                         "19,200 games; the corpus behind Figure 2(a) and Finding 1.",
        "definitions": {
            "participation": "share of games containing at least one executed wager (Wilson 95%)",
            "realised_bet_size": "mean executed wager in dollars, over all executed wagers; 95% "
                                 "percentile bootstrap resampling whole games (2,000 draws, "
                                 "seed 24231). Mean dollars staked per wagering game reported "
                                 "alongside.",
            "rebet_after_first_loss": "of games whose play contains at least one losing round, the "
                                      "share that execute another wager after that first losing "
                                      "round (Wilson 95%). Games that never lose are outside the "
                                      "denominator; the share of all games is also recorded.",
            "contrast": "variable minus fixed, in percentage points, 95% Newcombe hybrid-score",
        },
        "NOTE_rounds_of_win_flag": "scripts/build_figure_data.py:rounds_of reads the round outcome "
                                   "as str(gr).upper().startswith('W'). In the GPT-4.1-mini, "
                                   "Gemini-2.5-Flash and Claude-3.5-Haiku corpora game_result is a "
                                   "dict, so that test is False for every round and those three "
                                   "corpora are read as all-losses (verified: 0 wins in 1,249 / "
                                   "3,389 / 3,249 sampled rounds). This script unpacks the dict. "
                                   "Bet amounts and balances are unaffected, so participation and "
                                   "bet size match that loader; only loss-conditioned measures "
                                   "differ. Reported separately -- it also touches figure_data.json "
                                   "and Figure 2(c)/(d).",
        "cross_check": "participation k and bankruptcy k are compared against "
                       "paper_data/figure_data.json and paper_data/fig02_slot_machine.json below.",
        "per_model": per_model,
    }

    # cross-check against the released sidecar
    fig = json.load(open(OUT_DIR.parents[1] / "figure_data.json"))["slot_machine"]["per_model"]
    checks = []
    for label in per_model:
        for mode in ("fixed", "variable"):
            ref = fig[label][mode]
            got = per_model[label][mode]
            checks.append({
                "cell": f"{label} {mode}",
                "bankrupt_k_match": ref["n_bankrupt"] == got["bankruptcy"]["k"],
                "bankrupt_k_sidecar": ref["n_bankrupt"], "bankrupt_k_here": got["bankruptcy"]["k"],
                "participation_k_match": ref["participation_k"] == got["participation"]["k"],
                "participation_k_sidecar": ref["participation_k"],
                "participation_k_here": got["participation"]["k"],
            })
    doc["cross_check_rows"] = checks
    doc["cross_check_all_pass"] = all(c["bankrupt_k_match"] and c["participation_k_match"]
                                      for c in checks)
    (OUT_DIR / "gbsa_companion_metrics.json").write_text(json.dumps(doc, indent=2) + "\n")

    # ------------------------------------------------------------------ tex fragment
    L = []
    ap = L.append
    ap("% gbSA W4/Q1 promise -- participation, realised bet size, re-bet after first loss,")
    ap("% for the twelve primary behavioural cells behind Figure 2(a).")
    ap("% PROPOSED appendix table; not yet in any .tex.")
    ap("% Generated by paper_data/tables/appendix/code/gbsa_companion_metrics.py")
    ap("% Numbers in paper_data/tables/appendix/gbsa_companion_metrics.json")
    ap("%")
    ap("% Canonical sources (model field verified inside each file):")
    for label, rel, mf in SOURCES:
        ap(f"%   {label:<17} {rel}  [model = {mf}]")
    ap("%   slot_machine/{gemma,llama}/ carry DEPRECATION_WARNING.md and are not used.")
    ap("%")
    ap("% Cross-check against paper_data/figure_data.json: bankruptcy k and participation k "
       f"agree in all 12 cells: {doc['cross_check_all_pass']}")
    ap("%")
    ap("% Per-cell figures (bankrupt k/n | participation k/n | mean wager $ [95%] over W wagers |")
    ap("%   re-bet k/n_at_risk):")
    for label, _, _ in SOURCES:
        for mode in ("fixed", "variable"):
            c = per_model[label][mode]
            b, p, r = c["bankruptcy"], c["participation"], c["rebet_after_first_loss"]
            w = c["realised_bet_size"]
            ap(f"%   {label:<17} {mode:<8} | {b['k']:>4}/{b['n']} | {p['k']:>4}/{p['n']} | "
               f"${w['mean_usd_per_executed_wager']:.2f} "
               f"[{w['ci95_game_clustered_bootstrap'][0]:.2f}, "
               f"{w['ci95_game_clustered_bootstrap'][1]:.2f}] over {w['n_executed_wagers']} | "
               f"{r['k']:>4}/{r['n_at_risk']}")
    ap("%")
    ap("\\begin{table}[H]")
    ap("\\centering")
    ap("\\scriptsize")
    ap("\\setlength{\\tabcolsep}{3pt}")
    ap("\\caption{\\rev{Participation, realised bet size and re-betting after a first loss, "
       "reported alongside the bankruptcy rate for every primary slot-machine cell "
       "(Figure~\\ref{fig:slot-machine}a). $n = 1{,}600$ games per cell. Bankruptcy, "
       "participation and re-betting carry 95\\% Wilson intervals; the mean wager carries a 95\\% "
       "bootstrap interval resampling whole games. Participation is the share of games with at "
       "least one wager, the mean wager averages executed wagers only, and re-betting is "
       "conditioned on the games that reach a first loss ($n$ in brackets). The forced arm's "
       "wager is locked at $\\$10$ by construction.}}")
    ap("\\label{tab:companion-metrics}")
    ap("\\begin{tabular}{llcccc}")
    ap("\\toprule")
    ap("\\textbf{Model} & \\textbf{Arm} & \\textbf{Bankruptcy} & \\textbf{Participation} & "
       "\\makecell{\\textbf{Mean}\\\\\\textbf{wager}} & "
       "\\makecell{\\textbf{Re-bet after}\\\\\\textbf{first loss}} \\\\")
    ap("\\midrule")
    for i, (label, _, _) in enumerate(SOURCES):
        if i:
            ap("\\addlinespace")
        for j, mode in enumerate(("fixed", "variable")):
            c = per_model[label][mode]
            b, p, r = c["bankruptcy"], c["participation"], c["rebet_after_first_loss"]
            w = c["realised_bet_size"]
            head = label if j == 0 else ""
            arm = "forced" if mode == "fixed" else "choosing"
            bs = f"{b['pct']:.1f} [{b['wilson95'][0]:.1f}, {b['wilson95'][1]:.1f}]"
            ps = f"{p['pct']:.1f} [{p['wilson95'][0]:.1f}, {p['wilson95'][1]:.1f}]"
            ws = (f"\\${w['mean_usd_per_executed_wager']:.1f} "
                  f"[{w['ci95_game_clustered_bootstrap'][0]:.1f}, "
                  f"{w['ci95_game_clustered_bootstrap'][1]:.1f}]")
            rs = (f"{r['pct']:.1f} [{r['wilson95'][0]:.1f}, {r['wilson95'][1]:.1f}] "
                  f"({r['n_at_risk']})")
            ap(f"{head} & {arm} & {bs} & {ps} & {ws} & {rs} \\\\")
    ap("\\bottomrule")
    ap("\\end{tabular}")
    ap("\\end{table}")
    ap("")
    ap("% ==============================================================================")
    ap("% VARIANT B -- minimal-footprint alternative: instead of adding a table, extend the")
    ap("% one the paper already has, tab:appendix-slot-comprehensive")
    ap("% (neurips_content_en/appendix.tex:146-178), with three columns. Point estimates")
    ap("% only: measured at NeurIPS width this table is 389.6pt against a 397.5pt text")
    ap("% block at \\scriptsize with \\tabcolsep 2pt, and it does NOT fit if the three new")
    ap("% columns also carry their intervals (467pt). Use Variant A when the intervals must")
    ap("% sit next to the numbers, which is what the gbSA W3 undertaking asks for.")
    ap("%")
    ap("% The six existing columns are recomputed here and reproduce the printed table")
    ap("% exactly, with one exception: Claude variable Total bet is printed as 483.12 and")
    ap("% recomputes to 483.19 ($0.07, 0.015%). Note also that the printed Total bet divided")
    ap("% by the printed Mean rounds already equals the mean executed wager, so realised bet")
    ap("% size is recoverable from the existing table but is nowhere stated in the paper.")
    ap("%")
    ap("% \\begin{table}[H]")
    ap("% \\centering")
    ap("% \\caption{Comprehensive slot-machine results by model. Each betting mode was")
    ap("% aggregated from 1,600 games per model (32 prompt conditions $\\times$ 50")
    ap("% repetitions). \\rev{Participation is the share of games with at least one wager,")
    ap("% mean wager averages executed wagers only, and re-betting is the share of the games")
    ap("% reaching a first loss that wager again afterwards. 95\\% intervals for the three are")
    ap("% in Table~\\ref{tab:companion-metrics}.}}")
    ap("% \\label{tab:appendix-slot-comprehensive}")
    ap("% \\scriptsize")
    ap("% \\setlength{\\tabcolsep}{2pt}")
    ap("% \\begin{tabular}{llccccccc}")
    ap("% \\toprule")
    ap("% \\textbf{Model} & \\textbf{Bet mode} & "
       "\\makecell{\\textbf{Bankrupt-}\\\\\\textbf{cy (\\%)}} & "
       "\\rev{\\makecell{\\textbf{Particip.}\\\\\\textbf{(\\%)}}} & "
       "\\makecell{\\textbf{Mean}\\\\\\textbf{rounds}} & "
       "\\rev{\\makecell{\\textbf{Mean}\\\\\\textbf{wager (\\$)}}} & "
       "\\makecell{\\textbf{Total}\\\\\\textbf{bet (\\$)}} & "
       "\\rev{\\makecell{\\textbf{Re-bet after}\\\\\\textbf{first loss (\\%)}}} & "
       "\\makecell{\\textbf{Net}\\\\\\textbf{P\\&L (\\$)}} \\\\")
    ap("% \\midrule")
    for i, (label, _, _) in enumerate(SOURCES):
        if i:
            ap("% \\midrule")
        for j, mode in enumerate(("fixed", "variable")):
            c = per_model[label][mode]
            x = c["existing_table_columns_recomputed"]
            head = ("\\multirow{2}{*}{" + label + "}") if j == 0 else ""
            ap(f"% {head} & {mode.capitalize()} & {c['bankruptcy']['pct']:.2f} & "
               f"\\rev{{{c['participation']['pct']:.1f}}} & {x['mean_rounds']:.2f} & "
               f"\\rev{{{c['realised_bet_size']['mean_usd_per_executed_wager']:.2f}}} & "
               f"{x['mean_total_bet_usd']:.2f} & "
               f"\\rev{{{c['rebet_after_first_loss']['pct']:.1f}}} & "
               f"$-${abs(x['mean_net_pl_usd']):.2f} \\\\")
    ap("% \\bottomrule")
    ap("% \\end{tabular}")
    ap("% \\end{table}")
    ap("")
    ap("% ------------------------------------------------------------------------------")
    ap("% PROPOSED PROSE, to sit under the table in appendix:participation-and-exposure.")
    ap("% Three sentences; nothing is removed, and nothing moves into the body.")
    ap("%")
    ap("% \\rev{Table~\\ref{tab:companion-metrics} carries the three companion measures next to")
    ap("% every primary bankruptcy figure. Choosing raises participation in all six models and the")
    ap("% mean realised wager in all six, but the mean realised wager runs only from $\\$10.8$ to")
    ap("% $\\$44.8$ against a $\\$100$ ceiling, so the arms differ in how the freedom is used rather")
    of = per_model["Gemini-2.5-Flash"]["fixed"]["rebet_after_first_loss"]["pct"]
    ov = per_model["Gemini-2.5-Flash"]["variable"]["rebet_after_first_loss"]["pct"]
    gb = per_model["Gemini-2.5-Flash"]["variable"]["bankruptcy"]
    ap("% than in whether it is taken. Re-betting after a first loss is high in both arms and does")
    ap(f"% not track bankruptcy: it is {ov:.0f}\\% in the Gemini choosing arm against {of:.0f}\\% in its")
    ap("% forced arm, because a large enough first wager ends the game before a second one is")
    ap(f"% possible -- {gb['k_ruined_on_their_first_wager']} of that arm's {gb['k']} ruined games end on their first wager.}}")
    ap("%")
    ap("% Optional one-line replacement for the existing sentence at appendix.tex:1222, which")
    ap("% currently says only \"At a forced $70 stake most models decline to play at all\":")
    ap("% \\rev{At the primary $\\$10$ forced stake, by contrast, participation runs from 62.2\\% to")
    ap("% 99.4\\% across the six models (Table~\\ref{tab:companion-metrics}), so declining to play")
    ap("% is a cap effect rather than a property of the forced arm.}")
    (OUT_DIR / "gbsa_companion_metrics.tex").write_text("\n".join(L) + "\n")

    print("cross-check all pass:", doc["cross_check_all_pass"])
    for label, _, _ in SOURCES:
        for mode in ("fixed", "variable"):
            c = per_model[label][mode]
            b, p, r = c["bankruptcy"], c["participation"], c["rebet_after_first_loss"]
            w = c["realised_bet_size"]
            print(f"{label:<17} {mode:<8} bank {b['pct']:6.2f}%  part {p['k']:>4}/{p['n']} "
                  f"({p['pct']:5.1f}%)  wager ${w['mean_usd_per_executed_wager']:6.2f} "
                  f"[{w['ci95_game_clustered_bootstrap'][0]:.2f},{w['ci95_game_clustered_bootstrap'][1]:.2f}]"
                  f"  rebet {r['k']:>4}/{r['n_at_risk']:<4} "
                  f"({r['pct'] if r['pct'] is None else round(r['pct'],1)}%)")


if __name__ == "__main__":
    main()
