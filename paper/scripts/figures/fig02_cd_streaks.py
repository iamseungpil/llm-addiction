#!/usr/bin/env python3
"""Figure 2 panels (c) and (d): recompute the plotted streak series from the raw games.

Quantity: section 2's bet-to-balance ratio increase, max(0, (r_{t+1} - r_t) / r_t), averaged over
decision rounds that follow a run of identical outcomes.  Streak lengths 1-4 use runs of exactly k
outcomes; the last bin is "5 or more".  Corpus: the six slot-machine corpora listed in CORPORA,
19,200 games.  The open-weight runs are `slot_machine/llama/` and `slot_machine/gemma/`, which is
the corpus these two panels were computed on; panels (a) and (b) use the role-prompt runs
(`*_v4_role/`).  Sample sizes over runs of at least one outcome: fixed win 7,293 / loss 16,244,
variable win 21,891 / loss 48,573.

Run:  HF_TOKEN=... HF_HUB_DISABLE_XET=1 python3 scripts/figures/fig02_cd_streaks.py
Writes paper_data/fig02_cd_streaks.json and exits non-zero if a plotted value does not reproduce.
"""
import json, os, statistics as st, sys
from collections import defaultdict
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "llm-addiction-research/llm-addiction"
CORPORA = {
    "GPT-4o-mini": "analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json",
    "GPT-4.1-mini": "slot_machine/gpt/gpt5_experiment_20250921_174509.json",
    "Gemini-2.5-Flash": "slot_machine/gemini/gemini_experiment_20250920_042809.json",
    "Claude-3.5-Haiku": "slot_machine/claude/claude_experiment_corrected_20250925.json",
    "LLaMA-3.1-8B": "slot_machine/llama/final_llama_20251004_021106.json",
    "Gemma-2-9B": "slot_machine/gemma/final_gemma_20251004_172426.json",
}
PLOTTED = {
    ("fixed", "win"): [0.07, 0.09, 0.01, 0.02, 0.00],
    ("variable", "win"): [0.23, 0.25, 0.25, 0.24, 0.58],
    ("fixed", "loss"): [0.24, 0.25, 0.21, 0.19, 0.29],
    ("variable", "loss"): [0.67, 0.45, 0.28, 0.37, 0.37],
}
CAPTION_N = {("fixed", "win"): 7293, ("fixed", "loss"): 16244,
             ("variable", "win"): 21891, ("variable", "loss"): 48573}


def rounds(game):
    h = game.get("history") or game.get("game_history")
    if h is None:
        h = [r["game_result"] for r in game.get("round_details", [])
             if isinstance(r.get("game_result"), dict)]
    bal, out = 100, []
    for e in h:
        win = bool(e["win"]) if e.get("win") is not None else str(e["result"]).startswith("W")
        out.append((e["bet"] / bal if bal > 0 else 0.0, win))
        bal = e["balance"]
    return out


def main():
    acc, n_any = defaultdict(list), defaultdict(int)
    for path in CORPORA.values():
        d = json.load(open(hf_hub_download(REPO, path, repo_type="dataset",
                                           token=os.environ.get("HF_TOKEN"))))
        for g in d.get("results") or d.get("games") or d:
            rs = rounds(g)
            for t in range(len(rs) - 1):
                r, w = rs[t]
                if r <= 0:
                    continue
                run, j = 1, t - 1
                while j >= 0 and rs[j][1] == w:
                    run, j = run + 1, j - 1
                key = (g["bet_type"], "win" if w else "loss")
                acc[key + (min(run, 5),)].append(max(0.0, (rs[t + 1][0] - r) / r))
                n_any[key] += 1
    out, bad = {}, 0
    for key, target in PLOTTED.items():
        got = [st.mean(acc[key + (k,)]) for k in range(1, 6)]
        ok = all(abs(round(g, 2) - t) < 0.0051 for g, t in zip(got, target)) and n_any[key] == CAPTION_N[key]
        bad += not ok
        out["_".join(key)] = {"recomputed": got, "plotted": target, "n_rounds": n_any[key],
                              "n_by_bin": [len(acc[key + (k,)]) for k in range(1, 6)], "reproduces": ok}
        print(key, [round(g, 3) for g in got], target, n_any[key], "OK" if ok else "MISMATCH")
    out["multipliers_streak_1"] = {"post_win": out["variable_win"]["recomputed"][0] / out["fixed_win"]["recomputed"][0],
                                   "post_loss": out["variable_loss"]["recomputed"][0] / out["fixed_loss"]["recomputed"][0]}
    print("multipliers", out["multipliers_streak_1"])
    Path(__file__).resolve().parents[2].joinpath("paper_data", "fig02_cd_streaks.json").write_text(json.dumps(out, indent=2))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
