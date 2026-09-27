#!/usr/bin/env python3
"""Recompute Table `tab:information-not-bottleneck` (appendix.tex) on the CANONICAL corpora.

The table asks whether stating the odds reduces ruin.  Two of the five prompt
modules hand the model the numbers needed to compute the game's expected value:
``W`` states the 3x payout and ``P`` states the 30% win rate, and
0.3 * 3 - 1 = -0.10 per dollar follows by arithmetic.  So the 32 prompt
conditions split into 8 cells where the expected value is computable from the
prompt ("both W and P") and 24 where it is not ("neither or one").  The table
reports ruin rate in the variable-betting arm with 95% Wilson intervals and n,
per model, plus a pooled panel that holds module count fixed.

CANONICAL CORPORA (NEURIPS_CANONICAL_INDEX.md section 3, "Slot machine")
-----------------------------------------------------------------------
  GPT-4o-mini       analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json
                    (top-level "model" == "gpt-4o-mini-corrected")
  Claude-3.5-Haiku  slot_machine/claude/claude_experiment_corrected_20250925.json
                    ("model" == "claude-3-5-haiku-latest")
  Gemma-2-9B        behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json
  LLaMA-3.1-8B      behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json

NOT canonical, carried only to audit where the printed numbers came from:
  slot_machine/gpt/     -- "model" == "gpt-4.1-mini", NOT gpt-4o-mini
  slot_machine/gemma/   -- V1, carries DEPRECATION_WARNING.md ("CORRUPTED ...
                           should NOT be used for analysis")
  slot_machine/llama/   -- V1, carries DEPRECATION_WARNING.md ("MILD CORRUPTION")
  slot_machine/gemini/  -- Gemini-2.5-Flash, not a row of this table

Usage
-----
    python scripts/tables/rebuttal_info_bottleneck.py [--data-dir DIR]

``--data-dir`` points at a local mirror of the HuggingFace dataset
``llm-addiction-research/llm-addiction``; anything missing is downloaded with
``huggingface_hub.hf_hub_download`` (the repo is gated -- export ``HF_TOKEN``).

Output
------
    paper_data/tables/appendix/rebuttal_info_bottleneck.json
    paper_data/tables/appendix/info_not_bottleneck.tex   (commented provenance +
                                                          body rows on canonical data)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ID = "llm-addiction-research/llm-addiction"
OUTDIR = Path(__file__).resolve().parents[2] / "paper_data" / "tables" / "appendix"
OUT_JSON = OUTDIR / "rebuttal_info_bottleneck.json"
OUT_TEX = OUTDIR / "info_not_bottleneck.tex"

# ---------------------------------------------------------------- corpora

# name -> (path, expected "model" field or None, canonical?, note)
CORPORA = {
    "GPT-4o-mini": (
        "analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json",
        "gpt-4o-mini-corrected", True,
        "canonical GPT-4o-mini (corrected parsing)"),
    "Claude-3.5-Haiku": (
        "slot_machine/claude/claude_experiment_corrected_20250925.json",
        "claude-3-5-haiku-latest", True,
        "canonical Claude-3.5-Haiku"),
    "Gemma-2-9B": (
        "behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json",
        "gemma", True,
        "canonical Gemma-2-9B (V4role re-run)"),
    "LLaMA-3.1-8B": (
        "behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json",
        "llama", True,
        "canonical LLaMA-3.1-8B (V4role re-run)"),
    # --- audit-only, never printed
    "GPT-4.1-mini [not this table's model]": (
        "slot_machine/gpt/gpt5_experiment_20250921_174509.json",
        "gpt-4.1-mini", False,
        "slot_machine/gpt/ is gpt-4.1-mini; legacy 'gpt5_' filename"),
    "Gemma-2-9B [V1 DEPRECATED]": (
        "slot_machine/gemma/final_gemma_20251004_172426.json",
        "gemma", False,
        "DEPRECATION_WARNING.md: CORRUPTED, must not be used"),
    "LLaMA-3.1-8B [V1 DEPRECATED]": (
        "slot_machine/llama/final_llama_20251004_021106.json",
        "llama", False,
        "DEPRECATION_WARNING.md: MILD CORRUPTION"),
    "Gemini-2.5-Flash [not a table row]": (
        "slot_machine/gemini/gemini_experiment_20250920_042809.json",
        None, False,
        "context only"),
}

CANONICAL_ROWS = ["GPT-4o-mini", "Claude-3.5-Haiku", "Gemma-2-9B", "LLaMA-3.1-8B"]

# Values as printed in neurips_content_en/appendix.tex (for the audit only).
PRINTED = {
    "per_model": {
        "GPT-4o-mini":      dict(both=(18.8, 15.2, 22.9, 400), other=(2.2, 1.5, 3.2, 1200), diff=16.6),
        "Claude-3.5-Haiku": dict(both=(32.2, 27.9, 37.0, 400), other=(16.6, 14.6, 18.8, 1200), diff=15.7),
        "Gemma-2-9B":       dict(both=(49.2, 44.4, 54.1, 400), other=(22.3, 20.1, 24.8, 1200), diff=26.9),
        "LLaMA-3.1-8B":     dict(both=(7.8, 5.5, 10.8, 400),   other=(6.4, 5.2, 7.9, 1200),   diff=1.3),
    },
    "pooled_by_module_count": {
        2: dict(both=(19.5, 14.6, 25.5, 200), other=(9.6, 8.3, 11.0, 1800), diff=9.9),
        3: dict(both=(23.5, 20.3, 27.1, 600), other=(16.3, 14.4, 18.3, 1400), diff=7.2),
        4: dict(both=(28.3, 24.9, 32.1, 600), other=(31.0, 26.7, 35.7, 400), diff=-2.7),
    },
    # Prose in the same subsection: ruin by module count pooled over the four models.
    "ruin_by_module_count_all": {0: 1.5, 1: 4.3, 5: 41.0},
}

# Corpus combinations tested by the audit (which set reproduces the printed page).
AUDIT_POOLS = {
    "canonical [GPT-4o-mini + Claude + V4role open-weight]": CANONICAL_ROWS,
    "as_printed? [GPT-4.1-mini + Claude + V1 open-weight]": [
        "GPT-4.1-mini [not this table's model]", "Claude-3.5-Haiku",
        "Gemma-2-9B [V1 DEPRECATED]", "LLaMA-3.1-8B [V1 DEPRECATED]"],
    "mixed [GPT-4o-mini + Claude + V1 open-weight]": [
        "GPT-4o-mini", "Claude-3.5-Haiku",
        "Gemma-2-9B [V1 DEPRECATED]", "LLaMA-3.1-8B [V1 DEPRECATED]"],
    "mixed [GPT-4.1-mini + Claude + V4role open-weight]": [
        "GPT-4.1-mini [not this table's model]", "Claude-3.5-Haiku",
        "Gemma-2-9B", "LLaMA-3.1-8B"],
}


# ------------------------------------------------------------------ helpers

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% score interval for a proportion, in percent (same as build_figure_data)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def fetch(path: str, data_dir: Path | None) -> Path:
    if data_dir is not None:
        local = data_dir / path
        if local.exists():
            return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO_ID, path, repo_type="dataset",
                                cache_dir=str(Path.home() / ".cache" / "hf_infobottleneck")))


def is_bankrupt(game: dict) -> bool:
    """Same rule as scripts/build_figure_data.py:is_bankrupt."""
    if game.get("is_bankrupt") is not None:
        return bool(game["is_bankrupt"])
    if game.get("bankruptcy") is not None:
        return bool(game["bankruptcy"])
    return str(game.get("outcome", game.get("final_outcome", ""))).lower().startswith("bankrupt")


def module_count(combo: str) -> int:
    return 0 if combo.upper() == "BASE" else len(combo)


def has_both_WP(combo: str) -> bool:
    return "W" in combo.upper() and "P" in combo.upper()


def load(name: str, data_dir: Path | None) -> tuple[list[dict], dict]:
    """Return (variable-arm games, provenance dict).  Verifies the "model" field."""
    path, expect_model, canonical, note = CORPORA[name]
    p = fetch(path, data_dir)
    blob = json.load(open(p))
    declared = blob.get("model") if isinstance(blob, dict) else None
    games = (blob.get("results") or blob.get("games") or []) if isinstance(blob, dict) else blob
    prov = dict(path=path, declared_model_field=declared, expected_model_field=expect_model,
                model_field_ok=(expect_model is None or declared == expect_model),
                canonical=canonical, note=note, games_total=len(games),
                games_variable=sum(1 for g in games if g.get("bet_type") == "variable"),
                prompt_conditions=len({g.get("prompt_combo") for g in games}))
    if expect_model is not None and declared != expect_model:
        raise SystemExit(f"{name}: {path} declares model={declared!r}, expected {expect_model!r}")
    return [g for g in games if g.get("bet_type") == "variable"], prov


def cell(games: list[dict]) -> dict:
    n = len(games)
    k = sum(1 for g in games if is_bankrupt(g))
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, pct=round(100 * k / n, 4) if n else None,
                ci=[round(lo, 4), round(hi, 4)] if n else None)


def split(games: list[dict]) -> dict:
    both = [g for g in games if has_both_WP(g["prompt_combo"])]
    other = [g for g in games if not has_both_WP(g["prompt_combo"])]
    cb, co = cell(both), cell(other)
    d = dict(both_W_and_P=cb, neither_or_one=co,
             difference=(round(cb["pct"] - co["pct"], 4)
                         if cb["pct"] is not None and co["pct"] is not None else None))
    if cb["ci"] and co["ci"]:
        d["intervals_disjoint"] = bool(cb["ci"][0] > co["ci"][1] or co["ci"][0] > cb["ci"][1])
    return d


def by_module_count(games: list[dict]) -> dict:
    out = {}
    for mc in range(6):
        sub = [g for g in games if module_count(g["prompt_combo"]) == mc]
        d = split(sub)
        d["all_cells"] = cell(sub)
        out[str(mc)] = d
    return out


# ------------------------------------------------------------------ audit

def close(a, b, tol=0.0501) -> bool:  # printed values are rounded to 1 dp
    return a is not None and b is not None and abs(a - b) <= tol


def audit_row(printed: dict, got: dict) -> dict:
    gb, go = got["both_W_and_P"], got["neither_or_one"]
    pb, po = printed["both"], printed["other"]
    checks = {
        "both_pct":  (pb[0], gb["pct"]),  "both_lo": (pb[1], gb["ci"][0] if gb["ci"] else None),
        "both_hi":   (pb[2], gb["ci"][1] if gb["ci"] else None), "both_n": (pb[3], gb["n"]),
        "other_pct": (po[0], go["pct"]),  "other_lo": (po[1], go["ci"][0] if go["ci"] else None),
        "other_hi":  (po[2], go["ci"][1] if go["ci"] else None), "other_n": (po[3], go["n"]),
        "difference": (printed["diff"], got["difference"]),
    }
    detail, ok = {}, True
    for name, (want, val) in checks.items():
        good = (want == val) if name.endswith("_n") else close(want, val)
        detail[name] = dict(printed=want,
                            recomputed=(round(val, 2) if isinstance(val, float) else val),
                            match=bool(good))
        ok = ok and good
    return dict(verdict="REPRODUCES" if ok else "DOES NOT REPRODUCE", checks=detail)


# ------------------------------------------------------------------ latex

def f1(x):
    return f"{x:.1f}"


def tex_pct(c: dict) -> str:
    n = f"{c['n']:,}".replace(",", "{,}")
    return f"{f1(c['pct'])} [{f1(c['ci'][0])}, {f1(c['ci'][1])}] ($n{{=}}{n}$)"


def tex_diff(d: float) -> str:
    return (f"$+{f1(d)}$" if d >= 0 else f"$-{f1(abs(d))}$")


def build_tex(per_model: dict, pooled: dict, prov: dict, pooled_all: dict) -> str:
    L = []
    A = L.append
    A("% tab:information-not-bottleneck (appendix.tex, "
      "\\subsection{Stating the odds in the prompt does not reduce ruin}) -- body rows only.")
    A("% Columns: Row | Both W and P | Neither or one | Difference.")
    A("% Ruin rate (%) in the VARIABLE-betting arm, 95% Wilson score interval (z = 1.96).")
    A("% Generated by scripts/tables/rebuttal_info_bottleneck.py -- do not hand-edit.")
    A("%")
    A("% SOURCES (HuggingFace llm-addiction-research/llm-addiction), canonical per")
    A("% NEURIPS_CANONICAL_INDEX.md section 3 'Slot machine'; each file's top-level")
    A("% \"model\" field was checked, not the directory name:")
    for row in CANONICAL_ROWS:
        p = prov[row]
        A(f"%   {row:16s} {p['path']}")
        A(f"%   {'':16s}   model field = {p['declared_model_field']!r}; "
          f"{p['games_total']:,} games, {p['games_variable']:,} variable, "
          f"{p['prompt_conditions']} prompt conditions")
    A("% NOT used (and why):")
    for name in CORPORA:
        if CORPORA[name][2]:
            continue
        A(f"%   {CORPORA[name][0]} -- {CORPORA[name][3]}")
    A("%")
    A("% METHOD")
    A("%   arm            bet_type == 'variable' (1,600 games per model)")
    A("%   ruin           is_bankrupt (API runs) / outcome == 'bankruptcy' (open-weight runs)")
    A("%   'both W and P' 'W' in prompt_combo and 'P' in prompt_combo -- 8 of the 32 conditions")
    A("%   'neither/one'  the remaining 24 conditions")
    A("%   module count   0 for BASE, else len(prompt_combo); the fifth module is")
    A("%                  spelled R in the GPT/Claude/Gemma runs and H in the LLaMA V4role run")
    A("%   interval       95% Wilson score interval, z = 1.96")
    A("%   pooled panel   the four models concatenated, module count held fixed;")
    A("%                  only counts 2-4 have both cell types (at 0-1 no condition carries")
    A("%                  both W and P, at 5 every condition does)")
    A("%")
    A("% PER-CELL VALUES (k / n = pct, Wilson 95%):")
    for row in CANONICAL_ROWS:
        d = per_model[row]
        b, o = d["both_W_and_P"], d["neither_or_one"]
        A(f"%   {row:17s} both W and P   {b['k']:4d}/{b['n']:5,d} = {f1(b['pct']):>5s}%  "
          f"[{f1(b['ci'][0]):>4s}, {f1(b['ci'][1]):>4s}]")
        A(f"%   {'':17s} neither or one {o['k']:4d}/{o['n']:5,d} = {f1(o['pct']):>5s}%  "
          f"[{f1(o['ci'][0]):>4s}, {f1(o['ci'][1]):>4s}]   difference {d['difference']:+5.1f}"
          f"   intervals {'disjoint' if d['intervals_disjoint'] else 'overlap'}")
    for mc in ("2", "3", "4"):
        d = pooled[mc]
        b, o = d["both_W_and_P"], d["neither_or_one"]
        A(f"%   pooled, {mc} mods    both W and P   {b['k']:4d}/{b['n']:5,d} = {f1(b['pct']):>5s}%  "
          f"[{f1(b['ci'][0]):>4s}, {f1(b['ci'][1]):>4s}]")
        A(f"%   {'':17s} neither or one {o['k']:4d}/{o['n']:5,d} = {f1(o['pct']):>5s}%  "
          f"[{f1(o['ci'][0]):>4s}, {f1(o['ci'][1]):>4s}]   difference {d['difference']:+5.1f}"
          f"   intervals {'disjoint' if d['intervals_disjoint'] else 'overlap'}")
    A("%")
    A("% PER-MODEL BREAKDOWN of the pooled panel (difference, both minus neither/one,")
    A("% at each module count; -- means one of the two cells is empty):")
    for row in CANONICAL_ROWS:
        parts = []
        for mc in ("2", "3", "4"):
            d = per_model[row]["by_module_count"][mc]
            parts.append(f"{mc}: " + ("--" if d["difference"] is None else f"{d['difference']:+5.1f}"))
        A(f"%   {row:17s} " + "   ".join(parts))
    A("%")
    A("% PROSE FIGURE in the same subsection -- ruin by module count, pooled over the")
    A("% four canonical models, variable arm (all conditions, not split by W/P):")
    for mc in map(str, range(6)):
        c = pooled_all[mc]
        A(f"%   {mc} module{'s' if mc != '1' else ' '}  {c['k']:4d}/{c['n']:5,d} = {f1(c['pct']):>5s}%  "
          f"[{f1(c['ci'][0]):>4s}, {f1(c['ci'][1]):>4s}]")
    A("%")
    A("% CORRECTION NOTE.  The values printed in the current appendix.tex do not come")
    A("% from these corpora.  They reproduce exactly on slot_machine/gpt/ (gpt-4.1-mini,")
    A("% mislabelled GPT-4o-mini) together with the DEPRECATED V1 open-weight runs")
    A("% slot_machine/{gemma,llama}/.  Only the Claude row is unchanged on canonical data.")
    A("% On canonical data the per-model direction is preserved in all four models")
    A("% (all differences positive), but the pooled module-count panel REVERSES:")
    A("% -5.6 at two modules, -8.3 at three, -4.8 at four.")
    A("")
    for row in CANONICAL_ROWS:
        d = per_model[row]
        A(f"{row} & {tex_pct(d['both_W_and_P'])} & {tex_pct(d['neither_or_one'])} "
          f"& {tex_diff(d['difference'])} \\\\")
    A("\\midrule")
    A("\\multicolumn{4}{l}{\\emph{Pooled over the four models, module count held fixed}} \\\\")
    for mc in ("2", "3", "4"):
        d = pooled[mc]
        A(f"{mc} modules & {tex_pct(d['both_W_and_P'])} & {tex_pct(d['neither_or_one'])} "
          f"& {tex_diff(d['difference'])} \\\\")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="local mirror of the HF dataset; missing files are downloaded")
    args = ap.parse_args()

    games, prov = {}, {}
    for name in CORPORA:
        games[name], prov[name] = load(name, args.data_dir)

    # ---- canonical per-model panel
    per_model = {}
    for row in CANONICAL_ROWS:
        d = split(games[row])
        d["corpus"] = CORPORA[row][0]
        d["model_field"] = prov[row]["declared_model_field"]
        d["by_module_count"] = by_module_count(games[row])
        per_model[row] = d

    # ---- canonical pooled panel
    pool = [g for row in CANONICAL_ROWS for g in games[row]]
    pooled = by_module_count(pool)
    pooled_all = {mc: pooled[mc]["all_cells"] for mc in pooled}
    pooled_overall = split(pool)

    # ---- audit: which corpora do the printed numbers come from?
    audit = {"per_model": {}, "pooled_by_module_count": {}}
    row_candidates = {
        "GPT-4o-mini": ["GPT-4o-mini", "GPT-4.1-mini [not this table's model]"],
        "Claude-3.5-Haiku": ["Claude-3.5-Haiku"],
        "Gemma-2-9B": ["Gemma-2-9B", "Gemma-2-9B [V1 DEPRECATED]"],
        "LLaMA-3.1-8B": ["LLaMA-3.1-8B", "LLaMA-3.1-8B [V1 DEPRECATED]"],
    }
    for row, cands in row_candidates.items():
        got = {c: audit_row(PRINTED["per_model"][row], split(games[c])) for c in cands}
        hit = next((c for c, v in got.items() if v["verdict"] == "REPRODUCES"), None)
        audit["per_model"][row] = dict(
            printed_value_reproduces_on=hit,
            printed_source_is_canonical=(hit in CANONICAL_ROWS) if hit else None,
            verdict=("VERIFIED" if hit in CANONICAL_ROWS
                     else "WRONG (printed value is from a non-canonical corpus)" if hit
                     else "UNREPRODUCIBLE (matches no released corpus)"),
            candidates=got)

    audit_pools = {}
    for tag, members in AUDIT_POOLS.items():
        audit_pools[tag] = by_module_count([g for m in members for g in games[m]])
    for mc, printed in PRINTED["pooled_by_module_count"].items():
        got = {tag: audit_row(printed, audit_pools[tag][str(mc)]) for tag in AUDIT_POOLS}
        hit = next((t for t, v in got.items() if v["verdict"] == "REPRODUCES"), None)
        audit["pooled_by_module_count"][str(mc)] = dict(
            printed_value_reproduces_on=hit,
            verdict=("VERIFIED" if hit and hit.startswith("canonical")
                     else "WRONG (printed value is from a non-canonical corpus)" if hit
                     else "UNREPRODUCIBLE (matches no tested corpus combination)"),
            candidates=got)

    payload = dict(
        table="tab:information-not-bottleneck (neurips_content_en/appendix.tex)",
        generated_by="scripts/tables/rebuttal_info_bottleneck.py",
        hf_repo=REPO_ID,
        corpora=prov,
        canonical_rows=CANONICAL_ROWS,
        definition=dict(
            arm="bet_type == 'variable'",
            ruin="is_bankrupt (API) / outcome == 'bankruptcy' (open-weight)",
            both="'W' in prompt_combo and 'P' in prompt_combo (8 of 32 conditions)",
            neither_or_one="the other 24 conditions",
            module_count="0 for BASE, else len(prompt_combo); fifth module is R "
                         "(GPT/Claude/Gemma) or H (LLaMA V4role)",
            interval="95% Wilson score interval, z = 1.96",
        ),
        canonical=dict(
            per_model=per_model,
            pooled_by_module_count=pooled,
            pooled_overall=pooled_overall,
            pooled_ruin_by_module_count_all_cells=pooled_all,
        ),
        printed=PRINTED,
        printed_value_audit=audit,
        audit_pools=audit_pools,
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    OUT_TEX.write_text(build_tex(per_model, pooled, prov, pooled_all))

    # ---- console report
    def fmt(c):
        return f"{c['pct']:5.1f} [{c['ci'][0]:4.1f}, {c['ci'][1]:4.1f}] (n={c['n']:,}, k={c['k']})"

    print("== CANONICAL per-model panel, variable arm ==")
    for row in CANONICAL_ROWS:
        d = per_model[row]
        print(f"{row:18s} both {fmt(d['both_W_and_P'])} | other {fmt(d['neither_or_one'])} "
              f"| diff {d['difference']:+.1f} | {'disjoint' if d['intervals_disjoint'] else 'overlap'}")
    print("\n== CANONICAL pooled panel, module count fixed ==")
    for mc in ("2", "3", "4"):
        d = pooled[mc]
        print(f"{mc} modules          both {fmt(d['both_W_and_P'])} | other {fmt(d['neither_or_one'])} "
              f"| diff {d['difference']:+.1f} | {'disjoint' if d['intervals_disjoint'] else 'overlap'}")
    print("\n== CANONICAL ruin by module count (all cells, pooled) ==")
    for mc in map(str, range(6)):
        print(f"  {mc} modules {fmt(pooled_all[mc])}")
    print("\n== headline check: is 'both' >= 'neither or one' in all four models? ==")
    signs = {r: per_model[r]["difference"] for r in CANONICAL_ROWS}
    print("  differences:", {k: round(v, 1) for k, v in signs.items()},
          "->", "HOLDS" if all(v >= 0 for v in signs.values()) else "FAILS")
    print("\n== audit of the values currently printed in appendix.tex ==")
    for row, v in audit["per_model"].items():
        print(f"  {row:18s} {v['verdict']:60s} reproduces on: {v['printed_value_reproduces_on']}")
    for mc, v in audit["pooled_by_module_count"].items():
        print(f"  pooled {mc} modules   {v['verdict']:60s} reproduces on: {v['printed_value_reproduces_on']}")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
