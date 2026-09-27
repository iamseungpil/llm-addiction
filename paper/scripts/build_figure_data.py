"""Recompute every plotted quantity in the camera-ready from raw games.

Why this file exists
--------------------
Three panels of the submitted paper were not computed from data. Figure 2 panels
(c) and (d) were module-level constants in ``generate_paper_figures.py`` with no
derivation anywhere in either repository, and Figure 3 panel (d) was hand-read
off an older PDF (see the comment that used to sit at line 477 of that file).
This script replaces all three with numbers derived from the released corpora,
and attaches an interval to every quantity the paper plots.

Everything here reads either the public dataset
``llm-addiction-research/llm-addiction`` or the local behavioural mirror, and
writes one JSON that the figure code then draws.  Nothing is hardcoded.

Output: ``paper_data/figure_data.json``
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download  # noqa: E402

REPO_ID = "llm-addiction-research/llm-addiction"
CACHE = Path("/tmp/hfcache_figdata")

# Optional on-disk mirror of the released corpora.  Set ``LLM_ADDICTION_DATA``
# to a checkout of the dataset to read the behavioural games from local disk;
# when that directory is absent every loader falls back to the public
# HuggingFace release named by ``REPO_ID`` above, so the script runs anywhere.
DATA_ROOT = Path(os.environ.get("LLM_ADDICTION_DATA", Path.home() / "llm-addiction-data"))
LOCAL_SM = DATA_ROOT / "behavioral" / "slot_machine"
OUT = Path(__file__).resolve().parent.parent / "paper_data" / "figure_data.json"

BALANCE_RE = re.compile(r"Current balance:\s*\$(\d+(?:\.\d+)?)", re.I)
RNG = np.random.default_rng(24231)
N_BOOT = 2000
MAX_STREAK = 5

# The four API corpora, as named by regenerate_fig2_fig4.py.
API_SOURCES = [
    ("GPT-4o-mini", "analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json"),
    ("GPT-4.1-mini", "slot_machine/gpt/gpt5_experiment_20250921_174509.json"),
    ("Gemini-2.5-Flash", "slot_machine/gemini/gemini_experiment_20250920_042809.json"),
    ("Claude-3.5-Haiku", "slot_machine/claude/claude_experiment_corrected_20250925.json"),
]
OPENWEIGHT_DIRS = [("LLaMA-3.1-8B", "llama_v4_role"), ("Gemma-2-9B", "gemma_v4_role")]

# The GPT-4o-mini cap ablation behind Finding 4.  The variable arm was collected
# in two runs, one per cap pair; the fixed arm is a single file whose bet sizes
# are the caps themselves.
# The variable arm ran twice: a first run that stopped part-way, then a restart
# that skipped the (cap, condition, repetition) cells the first run had finished
# (its experiments_skipped = 600 and 650). Only the union holds 1,600 games per cap.
CAP_VARIABLE = [
    "analysis/fixed_variable_comparison/gpt_variable_max_bet_results/restart_complete_10_30_20251019_171400.json",
    "analysis/fixed_variable_comparison/gpt_variable_max_bet_results/restart_complete_50_70_20251019_162438.json",
    "analysis/fixed_variable_comparison/gpt_variable_max_bet_results/intermediate_20251016_073043.json",
    "analysis/fixed_variable_comparison/gpt_variable_max_bet_results/intermediate_20251016_075750.json",
]
CAP_FIXED = "analysis/fixed_variable_comparison/gpt_fixed_bet_size_results/complete_20251016_010653.json"


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% score interval for a proportion, as a percentage."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def boot_mean_ci(values, n_boot: int = N_BOOT):
    """Percentile interval for a mean, resampling the units it was taken over."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]), 1)
    idx = RNG.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(arr.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), int(arr.size))


# ---------------------------------------------------------------- loading


def fetch(path: str) -> Path:
    return Path(hf_hub_download(REPO_ID, path, repo_type="dataset", cache_dir=str(CACHE)))


def _outcome_win(record) -> bool:
    """Was this round a win, given one round record?

    ``record`` is either the ``game_result`` object the GPT/Gemini/Claude runs
    attach to a round, or the ``game_history`` entry the GPT-4o-mini run keeps
    beside ``round_details``, or a bare outcome string.  Where the record
    carries its own boolean ``win`` that is authoritative; otherwise the ``W``
    / ``L`` letter in ``result`` decides, which is the field the submitted
    generator read.
    """
    if isinstance(record, dict):
        if record.get("win") is not None:
            return bool(record["win"])
        record = record.get("result")
    return str(record or "").upper().startswith("W")


def _round_win(d: dict, idx: int, ghist: list) -> bool:
    """The outcome of round ``idx``, resolved the way the submission resolved it.

    ``regenerate_fig2_fig4.py`` at the submitted commit reads
    ``round_details[idx].game_result.result`` when that object is present, and
    otherwise falls back positionally to ``game_history[idx].result`` -- the
    GPT-4o-mini release is the one that keeps outcomes in ``game_history`` and
    leaves ``game_result`` off the round entirely -- and finally to the round's
    own ``result``/``outcome``.  ``game_result`` is a dict, so it must be
    indexed rather than stringified; stringifying it makes every round of the
    GPT-4.1-mini, Gemini-2.5-Flash and Claude-3.5-Haiku corpora read as a loss.
    """
    gr = d.get("game_result")
    if gr is not None:
        return _outcome_win(gr)
    if idx < len(ghist):
        return _outcome_win(ghist[idx])
    return _outcome_win(d.get("result", d.get("outcome", "")))


def rounds_of(game: dict) -> list[dict]:
    """Normalise one game into [{bet, balance_before, win}, ...].

    The corpora disagree about where the round record lives.  Open-weight runs
    carry ``history`` with the balance *after* the round; the GPT runs carry
    ``round_details`` with ``balance_before`` alongside ``game_history``; the
    Claude run carries only ``round_details``, whose balance has to be read back
    out of the prompt the model was shown.
    """
    hist = game.get("history")
    if hist:
        out, before = [], 100.0
        for ev in hist:
            out.append({"bet": ev.get("bet"), "balance_before": before,
                        "win": bool(ev.get("win")) if "win" in ev
                        else str(ev.get("result", "")).upper().startswith("W")})
            if ev.get("balance") is not None:
                before = float(ev["balance"])
        return out

    details = game.get("round_details") or []
    ghist = game.get("game_history") or []
    out = []
    for idx, d in enumerate(details):
        bet = d.get("bet") or d.get("parsed_bet") or d.get("bet_amount")
        bal = d.get("balance_before")
        if bal is None:
            m = BALANCE_RE.search(d.get("prompt") or "")
            bal = float(m.group(1)) if m else None
        win = _round_win(d, idx, ghist)
        if bet and bal:
            out.append({"bet": bet, "balance_before": float(bal), "win": win})
    return out


def load_games(name: str, source: str, kind: str) -> list[dict]:
    if kind == "hf":
        blob = json.load(open(fetch(source)))
        return blob.get("results") or blob.get("games") or []
    games = []
    if LOCAL_SM.is_dir():
        paths = [str(f) for f in sorted((LOCAL_SM / source).glob("final_*.json"))]
    else:
        paths = [fetch(p) for p in sorted(_hf_listdir(f"behavioral/slot_machine/{source}"))]
    for f in paths:
        blob = json.load(open(f))
        games.extend(blob.get("results") or blob.get("games") or [])
    return games


def _hf_listdir(prefix: str) -> list[str]:
    """The open-weight corpora, listed straight from the release.

    ``LOCAL_SM`` is the optional local mirror under ``$LLM_ADDICTION_DATA``.
    When that path is absent the same files are read from the public dataset
    instead, so the script runs anywhere.
    """
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    return [f for f in files
            if f.startswith(prefix + "/") and Path(f).name.startswith("final_")]


def is_bankrupt(game: dict) -> bool:
    if game.get("is_bankrupt") is not None:
        return bool(game["is_bankrupt"])
    if game.get("bankruptcy") is not None:
        return bool(game["bankruptcy"])
    return str(game.get("outcome", game.get("final_outcome", ""))).lower().startswith("bankrupt")


# ---------------------------------------------------------------- indicators


def game_indicators(rds: list[dict]) -> dict | None:
    """I_BA, I_LC and I_EC for one game, exactly as section 2 defines them."""
    if not rds:
        return None
    ratios = [min(r["bet"] / r["balance_before"], 1.0) for r in rds]
    i_ba = float(np.mean(ratios))
    i_ec = float(np.mean([1.0 if x >= 0.5 else 0.0 for x in ratios]))
    chases = []
    for t in range(len(rds) - 1):
        if not rds[t]["win"]:
            r_t = rds[t]["bet"] / rds[t]["balance_before"]
            r_n = rds[t + 1]["bet"] / rds[t + 1]["balance_before"]
            if r_t > 0:
                chases.append(max(0.0, (r_n - r_t) / r_t))
    return {"I_BA": i_ba, "I_EC": i_ec,
            "I_LC": float(np.mean(chases)) if chases else None,
            "n_rounds": len(rds)}


def streak_increases(rds: list[dict]) -> dict:
    """Bet-ratio increase after a run of k identical outcomes, k = 1..5.

    Under fixed betting the wager is locked, so winning can only raise the
    balance and therefore only lower the ratio.  The post-win entries are
    consequently zero by construction, not by measurement.
    """
    out = defaultdict(list)
    run, last = 0, None
    for t in range(len(rds) - 1):
        kind = "win" if rds[t]["win"] else "loss"
        run = run + 1 if kind == last else 1
        last = kind
        if run > MAX_STREAK:
            continue
        r_t = rds[t]["bet"] / rds[t]["balance_before"]
        r_n = rds[t + 1]["bet"] / rds[t + 1]["balance_before"]
        if r_t > 0:
            out[(kind, run)].append(max(0.0, (r_n - r_t) / r_t))
    return out


# ---------------------------------------------------------------- panels


def build_slot_machine() -> dict:
    per_model, pooled_streaks = {}, {m: defaultdict(list) for m in ("fixed", "variable")}
    for name, src, kind in ([(n, s, "hf") for n, s in API_SOURCES]
                            + [(n, d, "local") for n, d in OPENWEIGHT_DIRS]):
        games = load_games(name, src, kind)
        cells = {}
        for mode in ("fixed", "variable"):
            sub = [g for g in games if g.get("bet_type") == mode]
            k = sum(1 for g in sub if is_bankrupt(g))
            lo, hi = wilson(k, len(sub))
            per_game, played = [], 0
            for g in sub:
                rds = rounds_of(g)
                if rds:
                    played += 1
                ind = game_indicators(rds)
                if ind:
                    per_game.append(ind)
                for key, vals in streak_increases(rds).items():
                    pooled_streaks[mode][key].extend(vals)
            cell = {"n_games": len(sub), "n_bankrupt": k,
                    "bankrupt_pct": 100 * k / len(sub) if sub else None,
                    "bankrupt_ci": [lo, hi],
                    "participation_k": played, "participation_n": len(sub)}
            for ind in ("I_BA", "I_LC", "I_EC"):
                mean, lo_i, hi_i, n = boot_mean_ci([g[ind] for g in per_game if g[ind] is not None])
                cell[ind] = {"mean": mean, "ci": [lo_i, hi_i], "n_games": n}
            cells[mode] = cell
        per_model[name] = cells

    streaks = {}
    for mode, table in pooled_streaks.items():
        streaks[mode] = {}
        for kind in ("win", "loss"):
            series = []
            for k in range(1, MAX_STREAK + 1):
                mean, lo, hi, n = boot_mean_ci(table[(kind, k)])
                series.append({"streak": k, "mean": mean, "ci": [lo, hi], "n_rounds": n})
            streaks[mode][kind] = series
    return {"per_model": per_model, "streaks": streaks}


def build_cap_ablation() -> dict:
    """Finding 4's matched-cap control, recomputed from the released corpora."""
    variable = []
    for p in CAP_VARIABLE:
        variable.extend(json.load(open(fetch(p)))["results"])
    keys = [(g["max_bet"], g["condition_id"], g["repetition"]) for g in variable]
    assert len(keys) == len(set(keys)), "variable matched-cap files overlap"
    fixed_blob = json.load(open(fetch(CAP_FIXED)))
    fixed = fixed_blob["results"]

    rows = []
    for cap in (10, 30, 50, 70):
        f = [g for g in fixed if g.get("bet_size") == cap]
        v = [g for g in variable if g.get("max_bet") == cap]
        fk = sum(1 for g in f if is_bankrupt(g))
        vk = sum(1 for g in v if is_bankrupt(g))
        rows.append({
            "cap": cap,
            "fixed": None if not f else {
                "k": fk, "n": len(f), "pct": 100 * fk / len(f), "ci": list(wilson(fk, len(f)))},
            "variable": {"k": vk, "n": len(v), "pct": 100 * vk / len(v), "ci": list(wilson(vk, len(v)))},
        })
    return {
        "model": fixed_blob.get("model"),
        "rows": rows,
        "round_cap_fixed": fixed_blob["experiment_config"].get("max_rounds"),
        "round_cap_variable": json.load(open(fetch(CAP_VARIABLE[0])))["experiment_config"].get("max_rounds"),
        "note": ("The fixed arm was collected only at bet sizes 30, 50 and 70, so there is no "
                 "fixed cell at a $10 cap. The two arms also differ in their round ceiling, "
                 "which is recorded above and must be stated wherever these rates are shown."),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/build_figure_data.py",
        "hf_repo": REPO_ID,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": 24231,
        "slot_machine": build_slot_machine(),
        "cap_ablation": build_cap_ablation(),
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT}")

    sm = payload["slot_machine"]
    print("\nbankruptcy, fixed then variable")
    for name, cells in sm["per_model"].items():
        f, v = cells["fixed"], cells["variable"]
        print(f"  {name:18s} {f['n_bankrupt']:5d}/{f['n_games']:<5d} "
              f"{f['bankrupt_pct']:6.2f}% [{f['bankrupt_ci'][0]:.2f},{f['bankrupt_ci'][1]:.2f}]"
              f"   {v['n_bankrupt']:5d}/{v['n_games']:<5d} "
              f"{v['bankrupt_pct']:6.2f}% [{v['bankrupt_ci'][0]:.2f},{v['bankrupt_ci'][1]:.2f}]")
    print("\nstreak escalation, mean [95% CI] by run length")
    for mode in ("fixed", "variable"):
        for kind in ("win", "loss"):
            cells = " ".join(f"{s['mean']:.3f}" for s in sm["streaks"][mode][kind])
            ns = ",".join(str(s["n_rounds"]) for s in sm["streaks"][mode][kind])
            print(f"  {mode:9s} post-{kind:5s} {cells}   n=[{ns}]")
    print("\ncap ablation")
    for r in payload["cap_ablation"]["rows"]:
        f = r["fixed"]
        fs = "no fixed arm" if f is None else f"{f['k']:4d}/{f['n']:<5d} {f['pct']:6.2f}%"
        print(f"  ${r['cap']:<3d} fixed {fs}   variable {r['variable']['k']:4d}/"
              f"{r['variable']['n']:<5d} {r['variable']['pct']:6.2f}%")


if __name__ == "__main__":
    main()
