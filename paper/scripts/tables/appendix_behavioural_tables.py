"""Regenerate the behavioural half of the appendix tables from the released corpora.

Scope
-----
Five elements of ``neurips_content_en/appendix.tex``:

* Table 6  ``tab:appendix-investment-payoff``       -> investment_payoff.tex
* Table 7  ``tab:appendix-slot-comprehensive``      -> slot_comprehensive.tex
* Table 8  ``tab:appendix-investment-comprehensive``-> investment_comprehensive.tex
* Table 9  ``tab:appendix-verbatim-quotes``         -> verbatim_quotes_provenance.tex
* Table 15 ``tab:appendix-behavior-convergence``    -> behaviour_convergence.tex

Each ``.tex`` holds the tabular *body rows only*, so it can be ``\\input`` into
the float that already exists in the paper.  Column order and rounding are the
ones the paper prints.  Interval estimates do not fit inside those column sets,
so they ride along as LaTeX comments at the top of each fragment: Wilson score
intervals for proportions, percentile bootstrap over the resampling unit (the
game) for means.  Comments are inert at compile time and keep every emitted
digit traceable.

Nothing here is a literal read off the paper.  Where a value cannot be derived
from data it is omitted, not carried over.  Two things are deliberately absent:

* the ``I_LC`` column of Table 15, which no definition tried here reproduces
  (see ``ILC_DEFINITIONS_TRIED`` and the console report);
* any regeneration of the *quotes* in Table 9.  No sampling script exists, so
  instead of inventing one this module ships a provenance checker: it parses the
  rows the paper already prints, and asks whether each quoted sentence can be
  located, and located uniquely, in the released corpus at the stated
  (model, task, bet type, prompt, round, repetition).

Run:  from the repository root, \
      HF_HUB_DISABLE_XET=1 python3 scripts/tables/appendix_behavioural_tables.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

# The loaders and the statistics live in build_figure_data; import them rather
# than growing a second copy that can drift.
import build_figure_data as BFD  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

OUT_DIR = REPO / "paper_data" / "tables" / "appendix"
APPENDIX_TEX = REPO / "neurips_content_en" / "appendix.tex"

HF_IC_PREFIX = "investment_choice/bet_constraint/results"
HF_IC_MIRROR = Path("/tmp/llmadd_hf/investment_choice/bet_constraint/results")

# Optional on-disk mirror of the released behavioural corpora; see
# ``scripts/build_figure_data.py``.  The same tree is on the dataset under
# ``behavioral/``.
DATA_ROOT = Path(os.environ.get("LLM_ADDICTION_DATA", Path.home() / "llm-addiction-data"))
BEHAVIOURAL_ROOT = DATA_ROOT / "behavioral"
OPENWEIGHT_KEY = {"Gemma": "gemma", "LLaMA": "llama"}

RNG = np.random.default_rng(24231)
N_BOOT = 2000


# --------------------------------------------------------------- statistics


def boot_ci(values, stat=np.mean, n_boot: int = N_BOOT):
    """Percentile interval for ``stat`` over the unit the values are indexed by."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, arr.size, size=(n_boot, arr.size))
    draws = stat(arr[idx], axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def boot_ratio_ci(numer, denom, n_boot: int = N_BOOT):
    """Percentile interval for a ratio of sums, resampling clusters together.

    Used for the decision-level high-risk share, whose decisions are nested in
    games; the game is the resampling unit, not the decision.
    """
    a = np.asarray(list(numer), dtype=float)
    b = np.asarray(list(denom), dtype=float)
    if a.size < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, a.size, size=(n_boot, a.size))
    sums_b = b[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = 100.0 * a[idx].sum(axis=1) / sums_b
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def pct_interval_comment(label: str, k: int, n: int) -> str:
    """Wilson interval, with the 0/N case flagged as a one-sided upper bracket."""
    lo, hi = BFD.wilson(k, n)
    if k == 0:
        return f"%   {label}: 0/{n} = 0.00%, one-sided 95% upper bound {hi:.2f}%"
    return f"%   {label}: {k}/{n} = {100 * k / n:.2f}%, Wilson 95% [{lo:.2f}, {hi:.2f}]"


# ------------------------------------------------------------------ loading


def ensure_ic_mirror() -> Path:
    """Materialise the HF investment-choice results where the figure code expects them.

    ``generate_paper_figures.py`` reads a local mirror.  Rather than reimplement
    its loader (and risk a different de-duplication rule), fetch the same files
    from the dataset into that path when it is missing, then import the module.
    """
    from huggingface_hub import list_repo_files

    HF_IC_MIRROR.mkdir(parents=True, exist_ok=True)
    remote = [
        f
        for f in list_repo_files(BFD.REPO_ID, repo_type="dataset")
        if f.startswith(HF_IC_PREFIX + "/") and f.endswith(".json")
    ]
    for path in sorted(remote):
        dest = HF_IC_MIRROR / Path(path).name
        if dest.exists():
            continue
        src = hf_hub_download(
            BFD.REPO_ID, path, repo_type="dataset", cache_dir=str(BFD.CACHE)
        )
        shutil.copyfile(src, dest)
    return HF_IC_MIRROR


def slot_games() -> dict[str, list[dict]]:
    sources = [(n, s, "hf") for n, s in BFD.API_SOURCES] + [
        (n, d, "local") for n, d in BFD.OPENWEIGHT_DIRS
    ]
    return {name: BFD.load_games(name, src, kind) for name, src, kind in sources}


def openweight_games(task: str, model_key: str) -> list[dict]:
    patterns = {
        "sm": f"slot_machine/{model_key}_v4_role/final_{model_key}_*.json",
        "ic": f"investment_choice/v2_role_{model_key}/*.json",
        "mw": f"mystery_wheel/{model_key}_v2_role/{model_key}_mysterywheel_*.json",
    }
    games: list[dict] = []
    for path in sorted(BEHAVIOURAL_ROOT.glob(patterns[task])):
        blob = json.loads(path.read_text())
        results = blob.get("results", blob.get("games", []))
        games.extend(results.values() if isinstance(results, dict) else results)
    return games


# ------------------------------------------------- Table 7, slot machine


SLOT_MODEL_ORDER = [
    "GPT-4o-mini",
    "GPT-4.1-mini",
    "Gemini-2.5-Flash",
    "Claude-3.5-Haiku",
    "LLaMA-3.1-8B",
    "Gemma-2-9B",
]


def build_slot_comprehensive(games_by_model: dict[str, list[dict]]) -> tuple[str, dict]:
    lines = [
        "% tab:appendix-slot-comprehensive (appendix Table 7), body rows only.",
        "% Columns: Model | Bet mode | Bankruptcy (%) | Mean rounds | Total bet ($) | Net P&L ($).",
        "% Bankruptcy is games ending bankrupt / games in the cell.  Mean rounds, total bet and",
        "% net P&L are per-game means of total_rounds, total_bet and (total_won - total_bet);",
        "% net P&L equals final_balance - 100 in every cell, which is the ledger identity holding.",
        "% Intervals (not printable inside the paper's column set):",
    ]
    body: list[str] = []
    values: dict = {}

    for model in SLOT_MODEL_ORDER:
        games = games_by_model[model]
        for i, mode in enumerate(("fixed", "variable")):
            sub = [g for g in games if g.get("bet_type") == mode]
            k = sum(1 for g in sub if BFD.is_bankrupt(g))
            n = len(sub)
            rounds = [g["total_rounds"] for g in sub]
            bet = [g["total_bet"] for g in sub]
            pnl = [g["total_won"] - g["total_bet"] for g in sub]
            cell = {
                "n": n,
                "k_bankrupt": k,
                "bankrupt_pct": 100.0 * k / n,
                "bankrupt_ci": list(BFD.wilson(k, n)),
                "mean_rounds": float(np.mean(rounds)),
                "mean_rounds_ci": list(boot_ci(rounds)),
                "total_bet": float(np.mean(bet)),
                "total_bet_ci": list(boot_ci(bet)),
                "net_pnl": float(np.mean(pnl)),
                "net_pnl_ci": list(boot_ci(pnl)),
            }
            values[(model, mode)] = cell

            lines.append(pct_interval_comment(f"{model} {mode} bankruptcy", k, n))
            lines.append(
                f"%   {model} {mode} rounds {cell['mean_rounds']:.2f} "
                f"boot [{cell['mean_rounds_ci'][0]:.2f}, {cell['mean_rounds_ci'][1]:.2f}]; "
                f"total bet {cell['total_bet']:.2f} "
                f"[{cell['total_bet_ci'][0]:.2f}, {cell['total_bet_ci'][1]:.2f}]; "
                f"net P&L {cell['net_pnl']:.2f} "
                f"[{cell['net_pnl_ci'][0]:.2f}, {cell['net_pnl_ci'][1]:.2f}]  (n={n} games)"
            )

            label = "Fixed" if mode == "fixed" else "Variable"
            head = f"\\multirow{{2}}{{*}}{{{model}}}" if i == 0 else ""
            body.append(
                f"{head} & {label} & {cell['bankrupt_pct']:.2f} & "
                f"{cell['mean_rounds']:.2f} & {cell['total_bet']:.2f} & "
                f"{money(cell['net_pnl'])} \\\\"
            )
        if model != SLOT_MODEL_ORDER[-1]:
            body.append("\\midrule")

    return "\n".join(lines + [""] + body) + "\n", values


def money(x: float) -> str:
    """Paper prints negatives as ``$-$1.69``; keep that exact form."""
    return f"$-${abs(x):.2f}" if x < 0 else f"{x:.2f}"


# ----------------------------------------- Table 8, investment choice


IC_MODEL_ORDER = SLOT_MODEL_ORDER
IC_GROUPS = [("No goal", {"BASE", "M"}), ("Goal", {"G", "GM"})]


def build_investment_comprehensive(gpf) -> tuple[str, dict]:
    rows = gpf._load_investment_choice_rows()

    lines = [
        "% tab:appendix-investment-comprehensive (appendix Table 8), body rows only.",
        "% Columns: Model | Prompt group | Bankruptcy (%) | High-risk (%) | Moving-target (%) | n.",
        "% No goal = BASE + M, Goal = G + GM, as the caption states.  Bankruptcy and moving-target",
        "% are game-level rates (n games).  High-risk is the decision-level share of the",
        "% highest-variance option on the common semantic axis, so its interval bootstraps games,",
        "% not decisions.  Definitions imported from generate_paper_figures.py, unchanged.",
        "% Intervals:",
    ]
    body: list[str] = []
    values: dict = {}

    for model in IC_MODEL_ORDER:
        for i, (label, prompts) in enumerate(IC_GROUPS):
            sub = [r for r in rows if r["model"] == model and r["prompt"] in prompts]
            n = len(sub)
            bk = [bool(gpf._game_bankrupt(r["game"])) for r in sub]
            mt = [bool(gpf._goal_escalated(r)) for r in sub]

            hi_per_game, tot_per_game = [], []
            for r in sub:
                hi = tot = 0
                for d in r["game"]["decisions"]:
                    opt = gpf._semantic_option(r, d)
                    if opt is None:
                        continue
                    tot += 1
                    hi += opt == 4
                hi_per_game.append(hi)
                tot_per_game.append(tot)

            k_bk, k_mt = int(sum(bk)), int(sum(mt))
            hr = 100.0 * sum(hi_per_game) / sum(tot_per_game)
            cell = {
                "n": n,
                "bankrupt_pct": 100.0 * k_bk / n,
                "bankrupt_ci": list(BFD.wilson(k_bk, n)),
                "high_risk_pct": hr,
                "high_risk_ci": list(boot_ratio_ci(hi_per_game, tot_per_game)),
                "high_risk_decisions": int(sum(tot_per_game)),
                "moving_target_pct": 100.0 * k_mt / n,
                "moving_target_ci": list(BFD.wilson(k_mt, n)),
            }
            values[(model, label)] = cell

            lines.append(pct_interval_comment(f"{model} {label} bankruptcy", k_bk, n))
            lines.append(pct_interval_comment(f"{model} {label} moving-target", k_mt, n))
            lines.append(
                f"%   {model} {label} high-risk {hr:.1f}% of "
                f"{cell['high_risk_decisions']} decisions, game-clustered bootstrap 95% "
                f"[{cell['high_risk_ci'][0]:.1f}, {cell['high_risk_ci'][1]:.1f}]"
            )

            head = f"\\multirow{{2}}{{*}}{{{model}}}" if i == 0 else ""
            body.append(
                f"{head} & {label} & {cell['bankrupt_pct']:.1f} & "
                f"{cell['high_risk_pct']:.1f} & {cell['moving_target_pct']:.1f} & {n} \\\\"
            )
        if model != IC_MODEL_ORDER[-1]:
            body.append("\\midrule")

    return "\n".join(lines + [""] + body) + "\n", values


# ------------------------------------------- Table 6, payoff structure


OW_OPTION_RE = re.compile(
    r"Option (\d): \(amount you choose\), (\d+)% chance of ([\d.]+)x", re.I
)
OW_SAFE_RE = re.compile(r"Option (\d): \(amount you choose\), 100% returned", re.I)
API_OPTION_RE = re.compile(
    r"Option (\d): Invest \$X, (\d+)% chance of \$X.([\d.]+), (\d+)% chance of \$0", re.I
)
API_SAFE_RE = re.compile(r"Option (\d): Invest \$X, get \$X back \(100%\), game ends", re.I)


def parse_payoff_spec(prompt: str, source: str) -> dict[int, tuple[float, float]] | None:
    """Return {semantic option: (win probability, payout multiple of b)}.

    Semantic axis: 1 = safe exit, 4 = highest variance.  The two batches print the
    ladder in opposite order, which is exactly the remap ``_semantic_option`` does.
    """
    spec: dict[int, tuple[float, float]] = {}
    if source == "local":
        for m in OW_OPTION_RE.finditer(prompt):
            spec[5 - int(m.group(1))] = (int(m.group(2)) / 100.0, float(m.group(3)))
        for m in OW_SAFE_RE.finditer(prompt):
            spec[5 - int(m.group(1))] = (1.0, 1.0)
    else:
        for m in API_OPTION_RE.finditer(prompt):
            spec[int(m.group(1))] = (int(m.group(2)) / 100.0, float(m.group(3)))
        for m in API_SAFE_RE.finditer(prompt):
            spec[int(m.group(1))] = (1.0, 1.0)
    return spec if len(spec) == 4 else None


def moments(p: float, mult: float) -> tuple[float, float, float]:
    """Two-point net return per unit staked: +(mult-1) with prob p, -1 otherwise."""
    win, lose = mult - 1.0, -1.0
    mean = p * win + (1 - p) * lose
    second = p * win**2 + (1 - p) * lose**2
    var = second - mean**2
    return mean, var, float(np.sqrt(max(var, 0.0)))


def build_investment_payoff(gpf) -> tuple[str, dict, dict]:
    rows = gpf._load_investment_choice_rows()
    seen: dict[str, Counter] = {"local": Counter(), "api": Counter()}
    for r in rows:
        for d in r["game"]["decisions"]:
            prompt = d.get("full_prompt") or d.get("prompt") or ""
            spec = parse_payoff_spec(prompt, r["source"])
            if spec:
                seen[r["source"]][
                    tuple(sorted((k, v[0], v[1]) for k, v in spec.items()))
                ] += 1

    if not seen["local"]:
        raise RuntimeError("no corrected-design payoff spec found in the corpus")
    canonical = dict(
        (k, (p, m)) for k, p, m in max(seen["local"].items(), key=lambda kv: kv[1])[0]
    )
    legacy = (
        dict((k, (p, m)) for k, p, m in max(seen["api"].items(), key=lambda kv: kv[1])[0])
        if seen["api"]
        else {}
    )

    names = {1: "safe exit", 2: "low variance", 3: "mid variance", 4: "high variance"}
    lines = [
        "% tab:appendix-investment-payoff (appendix Table 6), body rows only.",
        "% Columns: Option | Win prob. | Payout (x b) | Loss outcome | E[net]/b | Var(net)/b^2 | SD(net)/b.",
        "% Every moment is derived from the payoff definition printed in the round prompts of the",
        "% corrected-design (open-weight) corpus, not typed in: net return is +(payout-1) with",
        "% probability p and -1 otherwise, per unit staked b.",
        f"% Corrected-design spec recovered from {sum(seen['local'].values())} open-weight prompts, "
        f"{len(seen['local'])} distinct spec(s).",
    ]
    for opt in (1, 2, 3, 4):
        p, mult = canonical[opt]
        mean, var, sd = moments(p, mult)
        lines.append(
            f"%   option {opt} ({names[opt]}): p={p:.2f}, payout={mult:.2f}x -> "
            f"E={mean:+.4f}, Var={var:.4f}, SD={sd:.4f}"
        )
    if legacy:
        lines.append(
            f"% Legacy closed-model batch recovered from {sum(seen['api'].values())} API prompts, "
            f"{len(seen['api'])} distinct spec(s):"
        )
        for opt in (1, 2, 3, 4):
            p, mult = legacy[opt]
            mean, var, sd = moments(p, mult)
            lines.append(
                f"%   option {opt} ({names[opt]}): p={p:.2f}, payout={mult:.2f}x -> "
                f"E={mean:+.4f}, Var={var:.4f}, SD={sd:.4f}"
            )

    body: list[str] = []
    computed = {}
    for opt in (1, 2, 3, 4):
        p, mult = canonical[opt]
        mean, var, sd = moments(p, mult)
        computed[opt] = {"p": p, "payout": mult, "mean": mean, "var": var, "sd": sd}
        label = f"{opt} ({names[opt]})"
        if opt in (2, 3):
            label += " "  # the paper pads these two labels by one space
        if opt == 1:
            payout = f"{mult:.2f} (return $b$, end game)"
            loss = "---"
        else:
            payout = f"{mult:.2f}"
            loss = "lose $b$"
        mean_tex = (
            f"$\\phantom{{-}}{mean:.2f}$" if mean >= 0 else f"$-{abs(mean):.2f}$"
        )
        body.append(
            f"{label} & {p:.2f} & {payout} & {loss} & {mean_tex} & "
            f"${var:.2f}$ & ${sd:.2f}$ \\\\"
        )

    return "\n".join(lines + [""] + body) + "\n", computed, legacy


# ------------------------------------- Table 15, indicator convergence

TASK_LABEL = {"sm": "Slot Machine", "ic": "Investment Choice", "mw": "Mystery Wheel"}

# Every I_LC rule this module tried, none of which reproduces the printed column.
ILC_DEFINITIONS_TRIED = [
    "game-level mean of max(0, (r_{t+1}-r_t)/r_t) over post-loss rounds (section 2 formula)",
    "the same with signed rather than clipped relative change",
    "the same with the model's parsed bet instead of the executed bet",
    "the same reading the loss flag from history[] instead of decisions[]",
    "round-level (unclustered) mean of the same quantity",
    "game-level fraction of post-loss rounds whose bet ratio rose",
    "game-level fraction of post-loss rounds whose absolute bet rose",
    "game-level mean of the absolute (not relative) ratio change after a loss",
]


def game_ratio_indicators(game: dict) -> tuple[float, float] | None:
    """I_BA and I_EC for one game of any of the three tasks.

    One rule for all three: the mean of min(bet / balance_before, 1) over every
    decision the game recorded a balance for, counting a non-betting decision as
    a zero ratio, and the share of those decisions at or above half the balance.
    """
    ratios = []
    for d in game.get("decisions", []):
        bet = d.get("bet") if d.get("bet") is not None else d.get("bet_amount")
        bal = d.get("balance_before")
        if bal is None or bal <= 0 or bet is None:
            continue
        ratios.append(min(float(bet) / float(bal), 1.0))
    if not ratios:
        return None
    return float(np.mean(ratios)), float(np.mean([r >= 0.5 for r in ratios]))


def build_behaviour_convergence() -> tuple[str, dict]:
    lines = [
        "% tab:appendix-behavior-convergence (appendix Table 15), body rows only.",
        "% Columns emitted: Model | Task | I_BA | I_EC.",
        "% The paper's I_LC column is NOT emitted.  It could not be reproduced from the released",
        "% corpora under any of the definitions listed in ILC_DEFINITIONS_TRIED; rather than print",
        "% a number whose rule is unknown, the column is left for the float to supply.  The header",
        "% of the float must be narrowed to match, or this fragment will not align.",
        "% I_BA / I_EC: game-level means over the open-weight corpora, one rule for all three",
        "% tasks (see game_ratio_indicators).  Bold marks the largest value within each",
        "% (model, indicator) block, computed from the data.",
        "% Caption caveat the data forces: 'all games' is exact only for slot machine.  In the IC",
        "% and MW corpora the fixed-bet half records bet_amount = None at decision level (the",
        "% imposed stake survives only inside decisions[].outcome), so those two rows are the",
        "% variable-bet half: 800 of 1,600 IC games and 1,600 of 3,200 MW games per model.",
        "% Reproducing the printed digits requires exactly this subset, so it is the subset the",
        "% paper used.",
        "% Intervals (percentile bootstrap over games):",
    ]
    stats: dict = {}
    for model in ("Gemma", "LLaMA"):
        for task in ("sm", "ic", "mw"):
            games = openweight_games(task, OPENWEIGHT_KEY[model])
            per_game = [game_ratio_indicators(g) for g in games]
            per_game = [x for x in per_game if x is not None]
            iba = [x[0] for x in per_game]
            iec = [x[1] for x in per_game]
            stats[(model, task)] = {
                "n_games_total": len(games),
                "n_games_used": len(per_game),
                "I_BA": float(np.mean(iba)),
                "I_BA_ci": list(boot_ci(iba)),
                "I_EC": float(np.mean(iec)),
                "I_EC_ci": list(boot_ci(iec)),
            }
            s = stats[(model, task)]
            lines.append(
                f"%   {model} {TASK_LABEL[task]}: I_BA {s['I_BA']:.3f} "
                f"[{s['I_BA_ci'][0]:.3f}, {s['I_BA_ci'][1]:.3f}], "
                f"I_EC {s['I_EC']:.3f} [{s['I_EC_ci'][0]:.3f}, {s['I_EC_ci'][1]:.3f}] "
                f"({s['n_games_used']} of {s['n_games_total']} games scored)"
            )

    body: list[str] = []
    for model in ("Gemma", "LLaMA"):  # the float draws no rule between the two blocks
        best = {
            ind: max(("sm", "ic", "mw"), key=lambda t: stats[(model, t)][ind])
            for ind in ("I_BA", "I_EC")
        }
        for task in ("sm", "ic", "mw"):
            s = stats[(model, task)]
            cells = []
            for ind in ("I_BA", "I_EC"):
                txt = f"{s[ind]:.3f}"
                cells.append(f"\\textbf{{{txt}}}" if best[ind] == task else txt)
            body.append(f"{model} & {TASK_LABEL[task]} & " + " & ".join(cells) + " \\\\")

    return "\n".join(lines + [""] + body) + "\n", stats


# ----------------------------- Table 9, verbatim quote provenance check


QUOTE_ROW_RE = re.compile(
    r"^(?P<prov>[A-Za-z0-9\-.]+(?:-[A-Za-z0-9.]+)*,\s*(?:SM|IC),[^&]*?)&\s*``(?P<quote>.*?)''\s*\\\\",
    re.S,
)


def parse_quote_rows(tex: str) -> list[dict]:
    """Pull (provenance, quote) pairs out of the verbatim-quote float.

    The paper is read-only here; this only reads it, so the checker always tests
    the rows the paper actually prints rather than a copy that can drift.
    """
    start = tex.index("\\label{tab:appendix-verbatim-quotes}")
    end = tex.index("\\end{table}", start)
    block = tex[start:end]
    rows = []
    for raw in block.split("\\\\"):
        if "``" not in raw:
            continue
        head, _, rest = raw.partition("&")
        if "``" not in rest:
            continue
        prov = detex(head).strip()
        quote = rest[rest.index("``") + 2 :]
        quote = quote[: quote.rindex("''")] if "''" in quote else quote
        if not prov or "," not in prov:
            continue
        rows.append({"provenance": prov, "quote": normalise(detex(quote))})
    return rows


def detex(s: str) -> str:
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\$", "$").replace("\\%", "%").replace("\\&", "&")
    s = s.replace("\\times", "x").replace("~", " ")
    s = re.sub(r"\[\d+pt\]", " ", s)
    return s


def normalise(s: str) -> str:
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "---").replace("\u00d7", "x")
    return re.sub(r"\s+", " ", s).strip()


def parse_provenance(prov: str) -> dict:
    """Split the six promised fields out of the printed provenance string."""
    parts = [p.strip() for p in prov.split(",")]
    out: dict = {"raw": prov, "model": parts[0], "extra": []}
    for p in parts[1:]:
        if p in ("SM", "IC"):
            out["task"] = p
        elif p in ("var", "fixed", "variable"):
            out["bet_type"] = "variable" if p == "var" else "fixed"
        elif re.fullmatch(r"r\d+", p):
            out["round"] = int(p[1:])
        elif re.fullmatch(r"(rep|trial)\s*\d+", p):
            out["repetition"] = int(re.search(r"\d+", p).group())
            out["repetition_label"] = p.split()[0]
        elif re.fullmatch(r"[GMHWP]+", p):
            out["prompt"] = p
        else:
            out["extra"].append(p)
    return out


def iter_slot_responses(games_by_model: dict[str, list[dict]]):
    for model, games in games_by_model.items():
        for g in games:
            details = g.get("round_details") or g.get("decisions") or []
            for d in details:
                text = d.get("gpt_response_full") or d.get("response") or ""
                if not text:
                    continue
                yield {
                    "model": model,
                    "task": "SM",
                    "bet_type": g.get("bet_type"),
                    "prompt": g.get("prompt_combo"),
                    "round": d.get("round"),
                    "repetition": g.get("repetition"),
                    "text": normalise(text),
                }


def iter_ic_responses(rows: list[dict]):
    for r in rows:
        game = r["game"]
        for d in game["decisions"]:
            text = d.get("response") or ""
            if not text:
                continue
            yield {
                "model": r["model"],
                "task": "IC",
                "bet_type": r["bet_type"],
                "prompt": r["prompt"],
                "round": d.get("round"),
                "repetition": game.get("trial", game.get("game_id")),
                "cap": r["bet_constraint"],
                "balance": d.get("balance_before"),
                "text": normalise(text),
            }


def locate(quote: str, corpus: list[dict]) -> tuple[list[dict], str]:
    """Exact substring first; fall back to ordered fragments across an elision."""
    hits = [c for c in corpus if quote in c["text"]]
    if hits:
        return hits, "verbatim"
    frags = [f.strip(" .,;") for f in quote.split("---") if len(f.strip()) > 15]
    if len(frags) > 1:
        hits = [c for c in corpus if all(f in c["text"] for f in frags)]
        if hits:
            return hits, "elided (fragments joined by ---)"
    return [], "not found"


def coords(entry: dict) -> tuple:
    return (
        entry["model"],
        entry["task"],
        entry["bet_type"],
        entry["prompt"],
        entry["round"],
        entry["repetition"],
    )


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$.]+", s.lower()) if len(w) > 2}


def coverage(quote: str, text: str) -> float:
    """Share of the quote's content words present in the candidate response."""
    q = content_words(quote)
    return len(q & content_words(text)) / len(q) if q else 0.0


def build_verbatim_provenance(games_by_model, ic_rows) -> tuple[str, list[dict]]:
    tex = APPENDIX_TEX.read_text()
    rows = parse_quote_rows(tex)

    sm_corpus = list(iter_slot_responses(games_by_model))
    ic_corpus = list(iter_ic_responses(ic_rows))
    by_coord = {}
    for entry in sm_corpus + ic_corpus:
        by_coord.setdefault(coords(entry), []).append(entry)

    results = []
    for row in rows:
        p = parse_provenance(row["provenance"])
        corpus = sm_corpus if p.get("task") == "SM" else ic_corpus
        stated = (
            p["model"],
            p.get("task"),
            p.get("bet_type"),
            p.get("prompt"),
            p.get("round"),
            p.get("repetition"),
        )
        at_stated = by_coord.get(stated, [])
        hits, mode = locate(row["quote"], corpus)
        matching = [h for h in hits if coords(h) == stated]

        if hits and matching and len(hits) == 1:
            verdict = "verbatim, unique in corpus, at stated coordinates"
            detail = "1 match"
        elif hits and matching:
            verdict = (
                f"at stated coordinates, but the sentence recurs in {len(hits) - 1} "
                "other responses"
            )
            detail = f"{len(hits)} matches, {mode}"
        elif hits:
            got = sorted({coords(h) for h in hits}, key=str)[:3]
            verdict = "verbatim elsewhere; printed coordinates do not hold"
            detail = f"{len(hits)} match(es) at " + "; ".join(
                f"{m}/{t}/{b}/{pr}/r{rd}/rep {rp}" for m, t, b, pr, rd, rp in got
            )
        elif at_stated:
            cov = max(coverage(row["quote"], e["text"]) for e in at_stated)
            verdict = (
                f"stated response exists but the sentence is not verbatim "
                f"({cov:.0%} of its content words present)"
            )
            detail = "paraphrase or stitched excerpt"
        else:
            verdict = "not locatable"
            detail = (
                f"no response at the stated coordinates; searched "
                f"{len(corpus)} {p.get('task', '?')} responses"
            )
        results.append(
            {**row, **p, "verdict": verdict, "detail": detail, "n_hits": len(hits)}
        )

    lines = [
        "% Provenance check for tab:appendix-verbatim-quotes (appendix Table 9), body rows only.",
        "% There is no sampling script for that table and none is invented here.  This is the",
        "% audit the caption's promise implies: for each printed row, take the six fields",
        "% (model / task / bet type / prompt / round / repetition), search the released corpus for",
        "% the quoted sentence, and report whether it is there, whether it is there exactly once,",
        "% and whether it sits at the stated coordinates.",
        f"% Corpus searched: {len(sm_corpus)} slot-machine responses, {len(ic_corpus)} "
        "investment-choice responses.",
        "% Columns: # | Provenance as printed | Located | Verdict.",
        "",
    ]
    body = []
    for i, r in enumerate(results, 1):
        prov = r["provenance"].replace("$", "\\$").replace("&", "\\&").replace("%", "\\%")
        detail = r["detail"].replace("$", "\\$").replace("&", "\\&").replace("%", "\\%")
        body.append(
            f"{i} & {prov} & {r['n_hits']} & {r['verdict']} ({detail}) \\\\"
        )
    return "\n".join(lines + body) + "\n", results


# ------------------------------------------------------------------- main


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_ic_mirror()
    import generate_paper_figures as gpf  # imported only for its loaders

    print("loading slot-machine corpora ...", flush=True)
    games_by_model = slot_games()

    print("Table 7  comprehensive slot machine")
    slot_tex, slot_vals = build_slot_comprehensive(games_by_model)
    (OUT_DIR / "slot_comprehensive.tex").write_text(slot_tex)
    for model in SLOT_MODEL_ORDER:
        for mode in ("fixed", "variable"):
            c = slot_vals[(model, mode)]
            print(
                f"   {model:18s} {mode:9s} bk {c['bankrupt_pct']:6.2f}  "
                f"rounds {c['mean_rounds']:6.2f}  bet {c['total_bet']:7.2f}  "
                f"pnl {c['net_pnl']:8.2f}"
            )

    print("\nTable 8  comprehensive investment choice")
    ic_tex, ic_vals = build_investment_comprehensive(gpf)
    (OUT_DIR / "investment_comprehensive.tex").write_text(ic_tex)
    for model in IC_MODEL_ORDER:
        for label, _ in IC_GROUPS:
            c = ic_vals[(model, label)]
            print(
                f"   {model:18s} {label:8s} bk {c['bankrupt_pct']:5.1f}  "
                f"high-risk {c['high_risk_pct']:5.1f}  moving-target "
                f"{c['moving_target_pct']:5.1f}  n {c['n']}"
            )

    print("\nTable 6  investment-choice payoff structure")
    pay_tex, pay_vals, legacy = build_investment_payoff(gpf)
    (OUT_DIR / "investment_payoff.tex").write_text(pay_tex)
    for opt, v in pay_vals.items():
        print(
            f"   option {opt}: p={v['p']:.2f} payout={v['payout']:.2f}x  "
            f"E[net]/b={v['mean']:+.4f}  Var={v['var']:.4f}  SD={v['sd']:.4f}"
        )
    if legacy:
        p, m = legacy[3]
        mean, var, sd = moments(p, m)
        print(
            f"   legacy closed-model batch, mid-variance cell: p={p:.2f} payout={m:.2f}x "
            f"-> E[net]/b={mean:+.4f}, Var={var:.4f}, SD={sd:.4f}"
        )

    print("\nTable 15  behavioural indicator convergence")
    conv_tex, conv_vals = build_behaviour_convergence()
    (OUT_DIR / "behaviour_convergence.tex").write_text(conv_tex)
    for (model, task), s in conv_vals.items():
        print(
            f"   {model:6s} {TASK_LABEL[task]:18s} I_BA {s['I_BA']:.3f}  "
            f"I_EC {s['I_EC']:.3f}  (n={s['n_games_used']}/{s['n_games_total']})"
        )
    print("   I_LC omitted; definitions tried and rejected:")
    for d in ILC_DEFINITIONS_TRIED:
        print(f"     - {d}")

    print("\nTable 9  verbatim-quote provenance check")
    quote_tex, quote_res = build_verbatim_provenance(
        games_by_model, gpf._load_investment_choice_rows()
    )
    (OUT_DIR / "verbatim_quotes_provenance.tex").write_text(quote_tex)
    for i, r in enumerate(quote_res, 1):
        print(f"   {i:2d}. {r['provenance']}")
        print(f"       -> {r['verdict']} [{r['detail']}]")

    print("\nwrote:")
    for name in sorted(p.name for p in OUT_DIR.glob("*.tex")):
        print(f"   {OUT_DIR / name}")


if __name__ == "__main__":
    main()
