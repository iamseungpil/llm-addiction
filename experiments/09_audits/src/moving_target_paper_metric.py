"""Reproduce the paper's Figure 4(c) moving-target rate, then add the stricter definition.

The submitted paper defines the moving-target rate in §2 as the fraction of games where the model
"raises its self-set goal AFTER MEETING IT". The figure code
(paper_neurips_2026/figures/body/fig04_investment_choice/code/generate_paper_figures.py,
function `_goal_escalated`) returns True as soon as any extracted goal exceeds the previous one.
It never tests whether the balance reached the previous goal. So the published number answers a
weaker question than the definition promised.

This script does two things, on the same corpus and with the same extractor as the figure, so the
comparison is like-for-like:

  PAPER      exactly `_goal_escalated`: any upward revision, over ALL games in the condition.
  STRICT     the same walk, but a revision only counts when the balance had already reached the
             standing goal.

`_extract_goal_from_response` and the file selection are copied from the figure script so that no
part of the difference comes from a change of instrument or corpus.

Corpus, as the figure uses it:
  API models     investment_choice/bet_constraint/results/*.json  (one canonical file per
                 model x cap x bet type; later timestamps win)
  open weights   behavioral/investment_choice/v2_role_{gemma,llama}/*.json, which store
                 goal_before / goal_after directly instead of free text.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import re
from pathlib import Path

PROMPTS = ["BASE", "G", "M", "GM"]

# Verbatim from generate_paper_figures.py:205-229.
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
PROMPT_GOAL = re.compile(r"current self-set goal(?: from previous round)?:\s*\$?(\d+)")


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


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def walk(game: dict, source: str) -> dict:
    """One pass over a game. `previous_goal` follows the figure script exactly."""
    previous_goal: int | None = None
    max_balance = 0
    escalated = False
    escalated_after_reaching = False
    any_goal = False

    for dec in game.get("decisions") or game.get("history") or []:
        bb = dec.get("balance_before")
        if bb is not None:
            max_balance = max(max_balance, bb)

        if source == "local":
            goal = dec.get("goal_after")
            if goal is None:
                goal = dec.get("goal_before")
        else:
            goal = extract_goal(dec.get("response", ""))
            if goal is None:
                m = PROMPT_GOAL.search((dec.get("prompt") or "").lower())
                goal = int(m.group(1)) if m else None

        if goal is not None:
            any_goal = True
            if previous_goal is not None and goal > previous_goal:
                escalated = True
                if max_balance >= previous_goal:
                    escalated_after_reaching = True
            previous_goal = goal

        ba = dec.get("balance_after")
        if ba is not None:
            max_balance = max(max_balance, ba)

    return {"any_goal": any_goal, "paper": escalated, "strict": escalated_after_reaching}


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []

    api_files: dict[tuple[str, int, str], Path] = {}
    for path in sorted((root / "investment_choice/bet_constraint/results").glob("*.json")):
        name = path.name
        model = next((m for pre, m in (
            ("gpt4o_mini_", "GPT-4o-mini"), ("gpt41_mini_", "GPT-4.1-mini"),
            ("gemini_flash_", "Gemini-2.5-Flash"), ("claude_haiku_", "Claude-3.5-Haiku"))
            if name.startswith(pre)), None)
        if model is None:
            continue
        m = re.search(r"_(10|30|50|70)_(fixed|variable)_", name)
        if m is None:
            continue
        api_files[(model, int(m.group(1)), m.group(2))] = path

    for (model, _cap, _bet), path in sorted(api_files.items()):
        for game in json.loads(path.read_text())["results"]:
            rows.append({"model": model, "source": "api",
                         "prompt": game["prompt_condition"], "game": game})

    for model, sub in (("Gemma-2-9B", "v2_role_gemma"), ("LLaMA-3.1-8B", "v2_role_llama")):
        for path in sorted((root / "behavioral/investment_choice" / sub).glob("*.json")):
            for game in json.loads(path.read_text())["results"]:
                rows.append({"model": model, "source": "local",
                             "prompt": game["prompt_condition"], "game": game})
    return rows


def main() -> None:
    default_root = next(iter(sorted(glob.glob(str(
        Path.home() / ".cache/huggingface/hub/datasets--llm-addiction-research--llm-addiction"
        / "snapshots/*/investment_choice/bet_constraint")))), None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(default_root).parents[1]) if default_root else "")
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/moving_target_paper_metric.json")
    args = ap.parse_args()

    root = Path(args.root)
    rows = load_rows(root)
    if not rows:
        raise SystemExit(f"no rows loaded from {root}")

    print(f"{len(rows)} games loaded "
          f"({sum(r['source'] == 'api' for r in rows)} api, {sum(r['source'] == 'local' for r in rows)} open-weight)\n")

    scored = [(r, walk(r["game"], r["source"])) for r in rows]

    print("Condition level, all models pooled. PAPER is Figure 4(c) as published.")
    print(f"{'cond':<6}{'games':>7}{'anyGoal%':>10}{'PAPER (any raise)':>26}{'STRICT (raise after reaching)':>34}")
    report: dict = {}
    for prompt in PROMPTS:
        sub = [s for r, s in scored if r["prompt"] == prompt]
        n = len(sub)
        if not n:
            continue
        p_k = sum(s["paper"] for s in sub)
        s_k = sum(s["strict"] for s in sub)
        ag = 100 * sum(s["any_goal"] for s in sub) / n
        lp, hp = wilson(p_k, n)
        ls, hs = wilson(s_k, n)
        print(f"{prompt:<6}{n:>7}{ag:>9.1f}%{100*p_k/n:>14.1f} [{lp:.1f},{hp:.1f}]"
              f"{100*s_k/n:>18.1f} [{ls:.1f},{hs:.1f}]")
        report[prompt] = {"games": n, "any_goal_pct": ag,
                          "paper_pct": 100 * p_k / n, "paper_ci": [lp, hp],
                          "strict_pct": 100 * s_k / n, "strict_ci": [ls, hs]}

    print("\nPer model, goal conditions (G, GM) versus no-goal (BASE, M).")
    by_model: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r, s in scored:
        arm = "goal" if r["prompt"] in {"G", "GM"} else "nogoal"
        c = by_model[r["model"]]
        c[f"{arm}_n"] += 1
        c[f"{arm}_paper"] += int(s["paper"])
        c[f"{arm}_strict"] += int(s["strict"])
    for model in sorted(by_model):
        c = by_model[model]
        line = f"  {model:<18}"
        for arm, label in (("goal", "goal"), ("nogoal", "no-goal")):
            n = c[f"{arm}_n"]
            if not n:
                continue
            line += (f"{label} n={n:<5} PAPER {100*c[f'{arm}_paper']/n:>5.1f}%"
                     f"  STRICT {100*c[f'{arm}_strict']/n:>5.1f}%   ")
        print(line)
        report[f"MODEL|{model}"] = dict(c)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
