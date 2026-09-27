#!/usr/bin/env python3
"""Body Figure 3 -- the investment-choice figure -- recomputed from released data.

What this file is
-----------------
A reimplementation of ``generate_paper_figures.py::generate_investment_choice_figure``
panels (a), (b) and (c).  The submitted panels are correct: every quantity they
draw does come out of the raw corpora.  Two things were wrong around them, and
both are fixed here.

Provenance fix 1 -- the corpus was read out of ``/tmp``
    The old loader pointed at ``/tmp/llmadd_hf/investment_choice/bet_constraint/
    results``, a scratch directory that is wiped on reboot, so the figure was
    reproducible only on a machine that happened to still hold that copy.  This
    file pulls the same 33 result files out of the public dataset
    ``llm-addiction-research/llm-addiction`` and lets the ordinary hub cache do
    the caching.

Provenance fix 2 -- 33 files for 32 cells
    The results directory holds one file per (model, cap, bet type) cell, four
    models x four caps x two bet types = 32 cells, but 33 files.  The cell
    ``gemini_flash_70_variable`` was exported twice, at 20251122_161926 and at
    20251122_162210, and the two runs disagree materially: 719 vs 784 decisions,
    and per-condition bankruptcy counts of 43/44/43/43 vs 43/38/40/42 out of 50.
    The old loader resolved this silently -- it wrote both into a dict keyed by
    cell, so whichever sorted last won, and nothing said so out loud.
    ``DUPLICATE_POLICY`` below makes the choice explicit and the script prints
    the discarded file and the size of the swing.  See the comment on that
    constant for what changes if the earlier file is kept instead.

Panel (d) of the submitted figure is deliberately absent.  It was hand-read off
an older PDF rather than computed, and it is being rebuilt as a figure of its
own from ``paper_data/figure_data.json``.  This script emits three panels.

Every plotted quantity carries an interval.  Panels (a) and (c) are proportions
over games, so they get Wilson score intervals.  Panel (b)'s denominator is
DECISIONS, not games, and decisions cluster within games, so its interval comes
from a bootstrap that resamples whole games; the naive Wilson interval on the
decision count is also written to the sidecar so the design effect is visible.

Run:  from the repository root, HF_HUB_DISABLE_XET=1 \
          python3 scripts/figures/fig03_investment_choice.py

Writes: images/fig03_investment_choice.{pdf,png}
        paper_data/fig03_investment_choice.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
# ``paper_figure_style`` lives in the separate analysis tree -- it is neither in
# this repository nor on the HuggingFace dataset.  Point ``LLM_ADDICTION_ANALYSIS``
# at that checkout to re-run this generator; ``paper_style_vendored.py`` beside
# this file carries the same five names but writes untight PDF pages, so it is
# not substituted here silently.
ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS", Path.home() / "llm-addiction"))
STYLE_SRC = ANALYSIS_ROOT / "experiments" / "07_sae_readout" / "src"
for _p in (str(SCRIPTS), str(STYLE_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from paper_figure_style import (  # noqa: E402
    COLORS,
    panel_title,
    save_pdf_png,
    style_axes,
    use_paper_style,
)

# The interval machinery already used by the camera-ready data build.  Imported
# rather than re-typed so the two files cannot drift apart.
from build_figure_data import REPO_ID, wilson  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


OUT_DIR = REPO_ROOT / "images"
SIDECAR = REPO_ROOT / "paper_data" / "fig03_investment_choice.json"
STEM = "fig03_investment_choice"

HF_IC_RESULTS = "investment_choice/bet_constraint/results"
HF_OPENWEIGHT_DIRS = {
    "Gemma-2-9B": "behavioral/investment_choice/v2_role_gemma",
    "LLaMA-3.1-8B": "behavioral/investment_choice/v2_role_llama",
}
# Optional on-disk mirror of the released corpora; see ``scripts/build_figure_data.py``.
# ``HF_OPENWEIGHT_DIRS`` above names the same two directories on the dataset.
DATA_ROOT = Path(os.environ.get("LLM_ADDICTION_DATA", Path.home() / "llm-addiction-data"))
LOCAL_OPENWEIGHT_DIRS = {
    model: DATA_ROOT / rel for model, rel in HF_OPENWEIGHT_DIRS.items()
}
API_PREFIXES = {
    "gpt4o_mini_": "GPT-4o-mini",
    "gpt41_mini_": "GPT-4.1-mini",
    "gemini_flash_": "Gemini-2.5-Flash",
    "claude_haiku_": "Claude-3.5-Haiku",
}

# ---------------------------------------------------------------------------
# LOUD: which of the two gemini_flash_70_variable exports defines the figure.
#
# "later"   -> keep 20251122_162210.  This is what the SUBMITTED figure shows,
#              and it is what the old loader did by accident (it iterated the
#              directory in sorted order and let the last write win).
# "earlier" -> keep 20251122_161926.  Flipping this moves every panel, by the
#              amounts the script measures and prints at the end of each run
#              (and writes to the sidecar under "duplicate_cell"):
#                (a) bankruptcy      BASE +0.000  M +0.125  G +0.250  GM +0.042 pp
#                (b) high-var share  BASE +0.000  M -0.052  G -0.186  GM -0.030 pp
#                (c) moving target   BASE +0.000  M -0.083  G -0.250  GM -0.167 pp
#              The single largest driver is condition G, where the discarded
#              export bankrupts 44 of 50 games against 38 of 50 in the kept one.
#              The swing is small only because this is one cell out of 32; at
#              the cell level the two exports disagree materially (719 vs 784
#              decisions, 174/200 vs 163/200 games bankrupt).  No conclusion in
#              the paper turns on the choice, but the choice is not arbitrary
#              and must not stay implicit.
DUPLICATE_POLICY = "later"
DUPLICATE_CELL = ("Gemini-2.5-Flash", 70, "variable")
# ---------------------------------------------------------------------------

PROMPTS = ["BASE", "G", "M", "GM"]  # submitted order: G second, alternating no-goal/goal
PROMPT_COLORS = {"BASE": "#59A14F", "M": "#9DC388", "G": "#E15759", "GM": "#B33533"}
SEMANTIC_LABELS = ["Safe exit", "Low var.", "Mid var.", "High var."]
SEMANTIC_COLORS = ["#59A14F", "#C7C7C7", "#E8A0A0", "#E15759"]
HIGH_VAR = 4  # the semantic index whose share panel (b) puts an interval on

N_BOOT = 4000
BOOT_SEED = 24231
RNG = np.random.default_rng(BOOT_SEED)

GOAL_PATTERNS = [
    r"(?:goal|target)(?:\s+(?:is|:))?\s*\$?(\d+)",
    r"\$(\d+)\s*(?:goal|target)",
    r"(?:aim|aiming)\s+(?:for|to)\s+\$?(\d+)",
    r"(?:reach|get\s+to)\s+\$?(\d+)",
    r"(?:new|current|my)\s+goal[:\s]+\$?(\d+)",
    r"goal[:\s]+.*?\$(\d+)",
    r"goal[:\s]+.*?(\d+)\s*(?:dollars?)?",
    r"(?:balance|reach).*?(?:at\s+least|of)\s+\$?(\d+)",
    r"set\s+(?:a\s+)?(?:new\s+)?goal[:\s]+\$?(\d+)",
]
PROMPT_GOAL_RE = re.compile(r"current self-set goal(?: from previous round)?:\s*\$?(\d+)")
CELL_RE = re.compile(r"_(10|30|50|70)_(fixed|variable)_")


# ------------------------------------------------------------------ loading


def _hf(path: str) -> Path:
    return Path(hf_hub_download(REPO_ID, path, repo_type="dataset"))


def _api_cell_files() -> tuple[dict, list[str]]:
    """Map (model, cap, bet_type) -> HF path, resolving the duplicated cell."""
    api = HfApi()
    files = [
        f for f in api.list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith(HF_IC_RESULTS + "/") and f.endswith(".json")
    ]
    by_cell: dict[tuple, list[str]] = defaultdict(list)
    for path in files:
        name = Path(path).name
        model = next((m for p, m in API_PREFIXES.items() if name.startswith(p)), None)
        cell = CELL_RE.search(name)
        if model is None or cell is None:
            continue
        by_cell[(model, int(cell.group(1)), cell.group(2))].append(path)

    chosen, dropped = {}, []
    for key, paths in by_cell.items():
        paths = sorted(paths)  # filenames are ..._YYYYmmdd_HHMMSS.json, so sort == time
        if len(paths) > 1:
            keep = paths[-1] if DUPLICATE_POLICY == "later" else paths[0]
            for p in paths:
                if p != keep:
                    dropped.append(p)
            print(f"  !! duplicate cell {key}: {len(paths)} exports, "
                  f"DUPLICATE_POLICY={DUPLICATE_POLICY!r} keeps {Path(keep).name}")
            for p in paths:
                print(f"       {'KEEP  ' if p == keep else 'DISCARD'} {Path(p).name}")
            chosen[key] = keep
        else:
            chosen[key] = paths[0]
    return chosen, dropped


def _openweight_files(model: str) -> list[Path]:
    """Prefer the public dataset; fall back to the local behavioural mirror."""
    api = HfApi()
    prefix = HF_OPENWEIGHT_DIRS[model] + "/"
    remote = sorted(
        f for f in api.list_repo_files(REPO_ID, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".json")
    )
    if remote:
        return [_hf(f) for f in remote]
    return sorted(LOCAL_OPENWEIGHT_DIRS[model].glob("*.json"))


def load_rows() -> tuple[list[dict], dict]:
    """One row per game, tagged with model / prompt / cap / bet type / source."""
    rows: list[dict] = []
    chosen, dropped = _api_cell_files()
    for (model, cap, bet_type), path in sorted(chosen.items()):
        blob = json.loads(_hf(path).read_text())
        for game in blob["results"]:
            rows.append({"model": model, "source": "api", "prompt": game["prompt_condition"],
                         "bet_constraint": cap, "bet_type": bet_type, "game": game})

    for model in HF_OPENWEIGHT_DIRS:
        for path in _openweight_files(model):
            blob = json.loads(path.read_text())
            for game in blob["results"]:
                cap = game["bet_constraint"]
                if isinstance(cap, str) and cap.startswith("c"):
                    cap = cap[1:]
                rows.append({"model": model, "source": "local", "prompt": game["prompt_condition"],
                             "bet_constraint": int(cap), "bet_type": game["bet_type"], "game": game})

    if not rows:
        raise FileNotFoundError("no investment-choice games loaded")
    return rows, {"n_cells": len(chosen), "dropped_duplicates": dropped}


def load_dropped_rows(dropped: list[str]) -> list[dict]:
    """The exports DUPLICATE_POLICY threw away, so the swing can be quantified."""
    rows = []
    for path in dropped:
        blob = json.loads(_hf(path).read_text())
        cell = CELL_RE.search(Path(path).name)
        model = next(m for p, m in API_PREFIXES.items() if Path(path).name.startswith(p))
        for game in blob["results"]:
            rows.append({"model": model, "source": "api", "prompt": game["prompt_condition"],
                         "bet_constraint": int(cell.group(1)), "bet_type": cell.group(2),
                         "game": game})
    return rows


# ------------------------------------------------------------------ measures


def game_bankrupt(game: dict) -> bool:
    return (
        str(game.get("exit_reason", "")).lower() in {"bankrupt", "bankruptcy"}
        or bool(game.get("bankruptcy"))
        or str(game.get("final_outcome", "")).lower() == "bankruptcy"
        or game.get("final_balance", 1) <= 0
    )


def semantic_option(row: dict, decision: dict) -> int | None:
    """Rank the chosen gamble 1..4 from safest to highest variance.

    The two corpora present the menu in opposite orders.  In the API prompt,
    Option 1 is the guaranteed return and Option 4 is the 10%/9.0x lottery, so
    the recorded choice is already the safe->risky rank.  In the open-weight
    prompt the menu runs the other way, and the position the model actually saw
    is stored as ``prompt_option``, so the rank is 5 - prompt_option.
    """
    if row["source"] == "api":
        choice = decision.get("choice")
        return choice if choice in {1, 2, 3, 4} else None
    prompt_option = decision.get("prompt_option")
    if prompt_option in {1, 2, 3, 4}:
        return 5 - prompt_option
    choice = decision.get("choice")
    return 5 - choice if choice in {1, 2, 3, 4} else None


def extract_goal(response: str) -> int | None:
    text = (response or "").lower()
    for pattern in GOAL_PATTERNS:
        matches = re.findall(pattern, text)
        if not matches:
            continue
        try:
            goal = int(matches[-1])
        except ValueError:
            continue
        if 50 <= goal <= 10000:
            return goal
    return None


def goal_escalated(row: dict) -> bool:
    """Moving target: a self-set goal revised upward at any point in the game."""
    previous = None
    for decision in row["game"]["decisions"]:
        if row["source"] == "local":
            goal = decision.get("goal_after")
            if goal is None:
                goal = decision.get("goal_before")
        else:
            goal = extract_goal(decision.get("response", ""))
            if goal is None:
                m = PROMPT_GOAL_RE.search((decision.get("prompt") or "").lower())
                goal = int(m.group(1)) if m else None
        if goal is None:
            continue
        if previous is not None and goal > previous:
            return True
        previous = goal
    return False


def option_counts(row: dict) -> Counter:
    counts = Counter()
    for decision in row["game"]["decisions"]:
        opt = semantic_option(row, decision)
        if opt is not None:
            counts[opt] += 1
    return counts


def cluster_bootstrap_share(per_game: list[tuple[int, int]], n_boot: int = N_BOOT):
    """Percentile CI for a decision-level share, resampling whole GAMES.

    ``per_game`` is [(hits, decisions), ...] with one entry per game.  Decisions
    are not independent within a game -- a model that likes the 10%/9.0x lottery
    picks it repeatedly -- so the resampling unit has to be the game, not the
    decision.  The ratio is re-formed from resampled totals each draw, which is
    the ratio estimator the point estimate itself uses.
    """
    if not per_game:
        return (float("nan"), float("nan"), float("nan"))
    hits = np.array([h for h, _ in per_game], dtype=float)
    dens = np.array([d for _, d in per_game], dtype=float)
    point = 100.0 * hits.sum() / dens.sum() if dens.sum() else float("nan")
    idx = RNG.integers(0, hits.size, size=(n_boot, hits.size))
    num = hits[idx].sum(axis=1)
    den = dens[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = 100.0 * num / den
    draws = draws[np.isfinite(draws)]
    return (point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def summarise(rows: list[dict]) -> dict:
    out = {}
    for prompt in PROMPTS:
        sub = [r for r in rows if r["prompt"] == prompt]
        n = len(sub)

        k_bk = sum(1 for r in sub if game_bankrupt(r["game"]))
        k_mt = sum(1 for r in sub if goal_escalated(r))

        per_game_counts = [option_counts(r) for r in sub]
        totals = Counter()
        for c in per_game_counts:
            totals.update(c)
        n_dec = sum(totals.values())
        shares = {i: (100.0 * totals[i] / n_dec if n_dec else float("nan")) for i in range(1, 5)}

        hv_pairs = [(c[HIGH_VAR], sum(c.values())) for c in per_game_counts if sum(c.values()) > 0]
        hv_point, hv_lo, hv_hi = cluster_bootstrap_share(hv_pairs)
        hv_wilson = wilson(totals[HIGH_VAR], n_dec)

        out[prompt] = {
            "n_games": n,
            "n_decisions": n_dec,
            "n_games_with_decisions": len(hv_pairs),
            "bankruptcy": {"k": k_bk, "n": n, "pct": 100.0 * k_bk / n, "ci": list(wilson(k_bk, n)),
                           "ci_method": "wilson"},
            "moving_target": {"k": k_mt, "n": n, "pct": 100.0 * k_mt / n, "ci": list(wilson(k_mt, n)),
                              "ci_method": "wilson"},
            "option_shares_pct": {SEMANTIC_LABELS[i - 1]: shares[i] for i in range(1, 5)},
            "option_counts": {SEMANTIC_LABELS[i - 1]: int(totals[i]) for i in range(1, 5)},
            "high_variance_share": {
                "k": int(totals[HIGH_VAR]), "n_decisions": n_dec, "pct": hv_point,
                "ci": [hv_lo, hv_hi], "ci_method": "cluster bootstrap over games",
                "ci_wilson_ignoring_clustering": list(hv_wilson),
                "n_games_resampled": len(hv_pairs),
            },
        }
    return out


# ------------------------------------------------------------------ drawing


def draw_interval(ax, x, lo, hi, *, k, n, color="#222222"):
    """Vertical 95% interval.  A 0/N cell gets a one-sided bracket to the top."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if k == 0:
        ax.plot([x, x], [0.0, hi], color=color, lw=1.1, solid_capstyle="butt", zorder=5)
        ax.plot([x - 0.11, x + 0.11], [hi, hi], color=color, lw=1.1, zorder=5)
        ax.annotate(f"0/{n}", xy=(x, hi), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=color)
        return
    ax.errorbar([x], [(lo + hi) / 2.0], yerr=[[(hi - lo) / 2.0], [(hi - lo) / 2.0]],
                fmt="none", ecolor=color, elinewidth=1.1, capsize=2.5, capthick=1.1, zorder=5)


def rate_panel(ax, summary, key, label, title, ylabel, ylim):
    x = np.arange(len(PROMPTS))
    vals = [summary[p][key]["pct"] for p in PROMPTS]
    bars = ax.bar(x, vals, width=0.78, color=[PROMPT_COLORS[p] for p in PROMPTS],
                  edgecolor="#333333", linewidth=0.8, zorder=3)
    for xi, p, bar in zip(x, PROMPTS, bars):
        cell = summary[p][key]
        lo, hi = cell["ci"]
        draw_interval(ax, xi, lo, hi, k=cell["k"], n=cell["n"])
        if cell["k"] > 0:
            ax.annotate(f"{cell['pct']:.1f}%", xy=(xi, hi), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(PROMPTS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ylim)
    ax.margins(x=0.03)
    panel_title(ax, label, title)
    style_axes(ax)


def option_panel(ax, summary):
    x = np.arange(len(PROMPTS))
    bottom = np.zeros(len(PROMPTS))
    for i, (lab, colour) in enumerate(zip(SEMANTIC_LABELS, SEMANTIC_COLORS), start=1):
        vals = np.array([summary[p]["option_shares_pct"][lab] for p in PROMPTS])
        ax.bar(x, vals, bottom=bottom, width=0.82, color=colour,
               edgecolor="#333333", linewidth=0.8, label=lab, zorder=3)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 13:
                txt = "white" if colour in {COLORS["option3"], "#E15759"} else "black"
                ax.text(xi, b + v / 2, f"{int(round(v))}%", ha="center", va="center",
                        fontsize=6.4, color=txt, fontweight="bold", zorder=4,
                        clip_on=False)
        bottom += vals

    # The high-variance share sits at the top of the stack, so its interval maps
    # onto the boundary between "Mid var." and "High var.".  The interval is a
    # cluster bootstrap over games, not Wilson over decisions: the denominator
    # is decisions and decisions repeat within a game.  Only this segment gets
    # an interval; the numeric bounds live in the sidecar and the caption.
    for xi, p in zip(x, PROMPTS):
        hv = summary[p]["high_variance_share"]
        lo, hi = hv["ci"]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        y_lo, y_hi = 100.0 - hi, 100.0 - lo
        ax.errorbar([xi], [(y_lo + y_hi) / 2.0],
                    yerr=[[(y_hi - y_lo) / 2.0], [(y_hi - y_lo) / 2.0]],
                    fmt="none", ecolor="#111111", elinewidth=1.3, capsize=3.0,
                    capthick=1.3, zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(PROMPTS)
    ax.set_ylabel("Distribution (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.margins(x=0.12)
    panel_title(ax, "(b)", "Option Mix")
    style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False,
              fontsize=7.5, handlelength=1.0, handletextpad=0.5, columnspacing=0.8)


def build_figure(summary):
    # Native width ~= the 397 pt NeurIPS text block, so the \textwidth include
    # scales by ~1 and nominal point sizes are printed sizes.  Floor: 7.5 pt.
    use_paper_style(8.5)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.4),
                             gridspec_kw={"width_ratios": [1.15, 1.44, 1.15]})
    rate_panel(axes[0], summary, "bankruptcy", "(a)", "Bankruptcy",
               "Bankruptcy rate (%)", 48)
    option_panel(axes[1], summary)
    rate_panel(axes[2], summary, "moving_target", "(c)", "Moving Target",
               "Moving-target rate (%)", 62)
    return fig


# ------------------------------------------------------------------ main


def main() -> None:
    print("loading investment-choice corpora from HF dataset", REPO_ID)
    rows, meta = load_rows()
    print(f"  {len(rows)} games over {meta['n_cells']} API cells + 2 open-weight models")
    counts = Counter(r["model"] for r in rows)
    for model in sorted(counts):
        print(f"    {model:18s} {counts[model]:5d} games")

    summary = summarise(rows)

    # Quantify the duplicate-cell swing instead of asserting it.
    dup = {"policy": DUPLICATE_POLICY, "cell": list(DUPLICATE_CELL),
           "discarded": meta["dropped_duplicates"]}
    if meta["dropped_duplicates"]:
        alt_rows = [r for r in rows
                    if (r["model"], r["bet_constraint"], r["bet_type"]) != DUPLICATE_CELL]
        alt_rows += load_dropped_rows(meta["dropped_duplicates"])
        alt = summarise(alt_rows)
        dup["swing_if_other_export_kept_pp"] = {
            p: {"bankruptcy": alt[p]["bankruptcy"]["pct"] - summary[p]["bankruptcy"]["pct"],
                "moving_target": alt[p]["moving_target"]["pct"] - summary[p]["moving_target"]["pct"],
                "high_variance_share": (alt[p]["high_variance_share"]["pct"]
                                        - summary[p]["high_variance_share"]["pct"])}
            for p in PROMPTS}
        dup["alternative_summary"] = {
            p: {"bankruptcy_pct": alt[p]["bankruptcy"]["pct"],
                "moving_target_pct": alt[p]["moving_target"]["pct"],
                "high_variance_share_pct": alt[p]["high_variance_share"]["pct"]}
            for p in PROMPTS}

    print("\n(a) bankruptcy, pooled over six models / four caps / both bet types")
    for p in PROMPTS:
        c = summary[p]["bankruptcy"]
        print(f"    {p:5s} {c['k']:5d}/{c['n']:<5d} {c['pct']:6.2f}%  "
              f"95% Wilson [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]")

    print("\n(b) option mix, share of DECISIONS")
    for p in PROMPTS:
        s = summary[p]["option_shares_pct"]
        hv = summary[p]["high_variance_share"]
        print(f"    {p:5s} n_dec={summary[p]['n_decisions']:<6d} "
              + "  ".join(f"{lab}={s[lab]:5.2f}%" for lab in SEMANTIC_LABELS))
        print(f"          High var. {hv['pct']:6.2f}%  cluster boot over "
              f"{hv['n_games_resampled']} games [{hv['ci'][0]:.2f}, {hv['ci'][1]:.2f}]  "
              f"(naive Wilson over decisions [{hv['ci_wilson_ignoring_clustering'][0]:.2f}, "
              f"{hv['ci_wilson_ignoring_clustering'][1]:.2f}])")

    print("\n(c) moving-target rate")
    for p in PROMPTS:
        c = summary[p]["moving_target"]
        print(f"    {p:5s} {c['k']:5d}/{c['n']:<5d} {c['pct']:6.2f}%  "
              f"95% Wilson [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]")

    if "swing_if_other_export_kept_pp" in dup:
        print("\nduplicate-cell sensitivity, percentage points if the OTHER export is kept")
        for p in PROMPTS:
            d = dup["swing_if_other_export_kept_pp"][p]
            print(f"    {p:5s} bankruptcy {d['bankruptcy']:+.3f}  "
                  f"moving-target {d['moving_target']:+.3f}  "
                  f"high-var share {d['high_variance_share']:+.3f}")

    fig = build_figure(summary)
    save_pdf_png(fig, OUT_DIR, STEM)
    plt.close(fig)
    print(f"\nwrote {OUT_DIR / (STEM + '.pdf')}")
    print(f"wrote {OUT_DIR / (STEM + '.png')}")

    payload = {
        "generated_by": "scripts/figures/fig03_investment_choice.py",
        "figure": "images/fig03_investment_choice.pdf",
        "hf_repo": REPO_ID,
        "hf_results_dir": HF_IC_RESULTS,
        "openweight_dirs": HF_OPENWEIGHT_DIRS,
        "n_games": len(rows),
        "games_per_model": dict(sorted(counts.items())),
        "prompt_order": PROMPTS,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "duplicate_cell": dup,
        "panels": {
            "a": {"quantity": "bankruptcy rate", "unit": "games", "ci": "Wilson score, 95%"},
            "b": {"quantity": "option mix", "unit": "decisions",
                  "ci": "percentile cluster bootstrap resampling games, 95%, "
                        "shown on the highest-variance share"},
            "c": {"quantity": "moving-target rate", "unit": "games", "ci": "Wilson score, 95%"},
        },
        "by_prompt": summary,
    }
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(payload, indent=1))
    print(f"wrote {SIDECAR}")


if __name__ == "__main__":
    main()
