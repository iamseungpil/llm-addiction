"""Regenerate Figure 2 (slot machine) and Figure 4 (investment choice).

Both slot-machine panels are regenerated from canonical raw data. The GPT-4o-
mini slot-machine release uses the corrected parsing export, where realized
round outcomes live in ``game_history`` instead of ``round_details.game_result``.
The loader below aligns those two structures so the round-level irrationality
metrics can be reconstructed for all six models.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

import sys

# The unreleased ``paper_figure_style`` helper lives in the separate analysis
# tree, not in this repository and not on the HuggingFace dataset.  Point
# ``LLM_ADDICTION_ANALYSIS`` at that checkout to re-run this generator; a
# five-name subset is vendored at ``scripts/figures/paper_style_vendored.py``
# for the generators that were ported to it.
ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS", Path.home() / "llm-addiction"))
STYLE_SRC = ANALYSIS_ROOT / "experiments" / "07_sae_readout" / "src"
if str(STYLE_SRC) not in sys.path:
    sys.path.insert(0, str(STYLE_SRC))

from paper_figure_style import COLORS, annotate_bars, panel_title, save_pdf_png, style_axes, use_paper_style


# Optional on-disk mirror of the released corpora; see ``scripts/build_figure_data.py``.
DATA_ROOT = Path(os.environ.get("LLM_ADDICTION_DATA", Path.home() / "llm-addiction-data"))
LOCAL_SM = DATA_ROOT / "behavioral" / "slot_machine"
OUT_DIR = Path(__file__).resolve().parent / "images"
HF_CACHE = Path("/tmp/api_sm_hf")
HF_CACHE.mkdir(exist_ok=True, parents=True)
BALANCE_RE = re.compile(r"Current balance:\s*\$(\d+(?:\.\d+)?)", re.I)

API_SOURCES = [
    ("GPT-4o-mini", "analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json"),
    ("GPT-4.1-mini", "slot_machine/gpt/gpt5_experiment_20250921_174509.json"),
    ("Gemini-2.5-Flash", "slot_machine/gemini/gemini_experiment_20250920_042809.json"),
    ("Claude-3.5-Haiku", "slot_machine/claude/claude_experiment_corrected_20250925.json"),
]


def _slot_display_name(name):
    return (
        name.replace("GPT-4o-mini", "GPT-4o-\nmini")
        .replace("GPT-4.1-mini", "GPT-4.1-\nmini")
        .replace("Gemini-2.5-Flash", "Gemini-2.5-\nFlash")
        .replace("Claude-3.5-Haiku", "Claude-3.5-\nHaiku")
        .replace("LLaMA-3.1-8B", "LLaMA-3.1-\n8B")
        .replace("Gemma-2-9B", "Gemma-2-\n9B")
    )


def compute_sm_metrics_from_games(games, get_bk, get_bet, get_bal, get_result, get_details=None):
    by = {"fixed": {"games": 0, "bk": 0, "iba": [], "ilc": [], "iec": []},
          "variable": {"games": 0, "bk": 0, "iba": [], "ilc": [], "iec": []}}
    for g in games:
        bt = g.get("bet_type")
        if bt not in by:
            continue
        by[bt]["games"] += 1
        if get_bk(g):
            by[bt]["bk"] += 1
        decisions = get_details(g) if get_details is not None else g.get("decisions", [])
        iba_r, iec_r, ratios = [], [], []
        for dd in decisions:
            bet = get_bet(dd)
            bal = get_bal(dd)
            if bet is None or not bal or bet <= 0:
                continue
            r = min(bet / bal, 1.0)
            iba_r.append(r)
            iec_r.append(1 if r >= 0.5 else 0)
            res = get_result(dd)
            is_loss = str(res).upper() in ("L", "LOSS", "LOSE")
            ratios.append((r, is_loss))
        if iba_r:
            by[bt]["iba"].append(np.mean(iba_r))
        if iec_r:
            by[bt]["iec"].append(np.mean(iec_r))
        lc = []
        for i in range(1, len(ratios)):
            pr, pl = ratios[i - 1]
            r, _ = ratios[i]
            if pl and pr > 0:
                lc.append(max(0.0, (r - pr) / pr))
        if lc:
            by[bt]["ilc"].append(np.mean(lc))
    for bt, d in by.items():
        d["bk_rate"] = 100.0 * d["bk"] / d["games"] if d["games"] else 0.0
        d["iba_mean"] = float(np.mean(d["iba"])) if d["iba"] else 0.0
        d["ilc_mean"] = float(np.mean(d["ilc"])) if d["ilc"] else 0.0
        d["iec_mean"] = float(np.mean(d["iec"])) if d["iec"] else 0.0
    return by


def load_local_openweight(model_dir_name):
    files = sorted((LOCAL_SM / model_dir_name).glob("final_*.json"))
    all_games = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        all_games.extend(d.get("results", d.get("games", [])))
    return compute_sm_metrics_from_games(
        all_games,
        get_bk=lambda g: bool(g.get("bankruptcy")) or str(g.get("outcome", g.get("final_outcome", ""))).lower() in ("bankrupt", "bankruptcy"),
        get_bet=lambda dd: dd.get("bet") or dd.get("parsed_bet") or dd.get("bet_amount"),
        get_bal=lambda dd: dd.get("balance_before"),
        get_result=lambda dd: dd.get("result", ""),
        get_details=lambda g: g.get("decisions", []),
    )


def load_api_from_hf(repo_path):
    local = HF_CACHE / Path(repo_path).name
    if not local.exists():
        hf_hub_download(
            repo_id="llm-addiction-research/llm-addiction",
            filename=repo_path, repo_type="dataset",
            local_dir=str(HF_CACHE), local_dir_use_symlinks=False,
        )
        # hf_hub_download may place under subdirs; search
    for candidate in HF_CACHE.rglob(Path(repo_path).name):
        return candidate
    return None


def load_api_metrics(repo_path):
    p = load_api_from_hf(repo_path)
    if p is None or not p.exists():
        return None
    with open(p) as fh:
        d = json.load(fh)
    games = d.get("results", d.get("games", []))
    return compute_sm_metrics_from_games(
        games,
        get_bk=lambda g: bool(g.get("is_bankrupt") or g.get("bankruptcy"))
                        or str(g.get("outcome", g.get("final_outcome", ""))).lower() in ("bankrupt", "bankruptcy"),
        get_bet=lambda dd: dd.get("bet_amount"),
        get_bal=lambda dd: dd.get("balance_before") or (
            float(BALANCE_RE.search(dd.get("prompt", "")).group(1))
            if BALANCE_RE.search(dd.get("prompt", ""))
            else None
        ),
        get_result=lambda dd: dd.get("resolved_result"),
        get_details=lambda g: [
            {
                **dd,
                "resolved_result": (
                    (dd.get("game_result") or {}).get("result")
                    if dd.get("game_result") is not None
                    else (
                        g.get("game_history", [])[idx].get("result")
                        if idx < len(g.get("game_history", []))
                        else dd.get("result", dd.get("outcome", ""))
                    )
                ),
            }
            for idx, dd in enumerate(g.get("round_details", []))
        ],
    )


def build_model_metrics():
    models = {}
    # Open-weight from local canonical
    print("Loading LLaMA SM from local canonical...")
    models["LLaMA-3.1-8B"] = load_local_openweight("llama_v4_role")
    print("Loading Gemma SM from local canonical...")
    models["Gemma-2-9B"] = load_local_openweight("gemma_v4_role")
    # API from HF
    for name, path in API_SOURCES:
        print(f"Loading {name} from HF...")
        try:
            m = load_api_metrics(path)
            if m is not None:
                models[name] = m
        except Exception as e:
            print(f"  Failed {name}: {e}")
    return models


def generate_fig2(models):
    """Slot machine: (a) BK rate by bet type per model; (b) six-model mean metrics."""
    order = ["GPT-4o-mini", "GPT-4.1-mini", "Gemini-2.5-Flash", "Claude-3.5-Haiku", "LLaMA-3.1-8B", "Gemma-2-9B"]
    # Filter to those available
    order = [m for m in order if m in models]

    use_paper_style(9.2)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.15, 2.95))

    # (a) BK rate by bet type
    x = np.arange(len(order))
    width = 0.37
    fixed = [models[m]["fixed"]["bk_rate"] for m in order]
    variable = [models[m]["variable"]["bk_rate"] for m in order]
    b1 = ax_a.bar(x - width / 2, fixed, width, label="Fixed", color=COLORS["fixed"])
    b2 = ax_a.bar(x + width / 2, variable, width, label="Variable", color=COLORS["variable"])
    annotate_bars(ax_a, b1, fmt="{:.1f}", suffix="%")
    annotate_bars(ax_a, b2, fmt="{:.1f}", suffix="%")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([_slot_display_name(m) for m in order])
    ax_a.set_ylabel("Bankruptcy rate (%)")
    ax_a.set_ylim(0, 80)
    panel_title(ax_a, "(a)", "Bankruptcy Rate by Bet Type")
    style_axes(ax_a)
    ax_a.legend(loc="upper left")

    metrics = ["I_BA", "I_LC", "I_EC"]
    fixed_agg = np.array([
        np.mean([models[m]["fixed"]["iba_mean"] for m in order]),
        np.mean([models[m]["fixed"]["ilc_mean"] for m in order]),
        np.mean([models[m]["fixed"]["iec_mean"] for m in order]),
    ])
    var_agg = np.array([
        np.mean([models[m]["variable"]["iba_mean"] for m in order]),
        np.mean([models[m]["variable"]["ilc_mean"] for m in order]),
        np.mean([models[m]["variable"]["iec_mean"] for m in order]),
    ])
    x2 = np.arange(len(metrics))
    b3 = ax_b.bar(x2 - width / 2, fixed_agg, width, label="Fixed", color=COLORS["fixed"])
    b4 = ax_b.bar(x2 + width / 2, var_agg, width, label="Variable", color=COLORS["variable"])
    annotate_bars(ax_b, b3, fmt="{:.3f}", size=8.2)
    annotate_bars(ax_b, b4, fmt="{:.3f}", size=8.2)
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels([r"$I_{BA}$", r"$I_{LC}$", r"$I_{EC}$"])
    ax_b.set_ylabel("Metric value")
    ax_b.set_ylim(0, 0.75)
    panel_title(ax_b, "(b)", "Irrationality Metrics by Bet Type")
    style_axes(ax_b)
    ax_b.legend(loc="upper left")

    save_pdf_png(fig, OUT_DIR, "slot_machine_analysis2")
    plt.close(fig)
    return order, fixed, variable, fixed_agg, var_agg


def main():
    models = build_model_metrics()
    print("\n--- Model metrics summary ---")
    for m, v in models.items():
        print(f"  {m}: fixed BK={v['fixed']['bk_rate']:.2f}%, variable BK={v['variable']['bk_rate']:.2f}%")
    order, fixed, variable, fixed_agg, var_agg = generate_fig2(models)
    print("\n--- Figure 2 summary ---")
    print(f"Order: {order}")
    print(f"Fixed BK rates: {[f'{v:.2f}' for v in fixed]}")
    print(f"Variable BK rates: {[f'{v:.2f}' for v in variable]}")
    print(f"Fixed I_BA,I_LC,I_EC: {[f'{v:.3f}' for v in fixed_agg]}")
    print(f"Variable I_BA,I_LC,I_EC: {[f'{v:.3f}' for v in var_agg]}")


if __name__ == "__main__":
    main()
