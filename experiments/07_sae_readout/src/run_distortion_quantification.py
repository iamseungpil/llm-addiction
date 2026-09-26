"""
Cognitive Distortion Quantification
=====================================
의도: 인지 왜곡의 빈도를 조건별로 정량화하여, 자율성이
      행동뿐 아니라 추론 패턴도 변화시키는지 검증한다.

가설:
  H1: Variable betting에서 인지 왜곡 빈도가 Fixed보다 높다.
  H2: 목표 설정(G) 프롬프트에서 목표 관련 왜곡이 더 빈번하다.

검증:
  - LLaMA SM 3,200 게임의 전체 응답 텍스트에서 키워드 기반 분류
  - 4가지 핵심 왜곡 유형 (데이터에서 귀납적으로 구축)
  - 조건별 빈도 비교 with Mann-Whitney U + FDR correction
  - 게임 수준 집계 (독립성 확보)
"""

import json
import re
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

SEED = 42
np.random.seed(SEED)

DATA_PATH = Path("/home/v-seungplee/data/llm-addiction/behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json")
OUT_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/distortion")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Distortion patterns (built inductively from actual LLaMA responses)
DISTORTION_PATTERNS = {
    "pattern_belief": {
        "description": "Belief in exploitable patterns in random outcomes",
        "patterns": [
            r"pattern",
            r"favorable state",
            r"hidden",
            r"trend",
            r"streak.{0,20}continue",
            r"machine.{0,20}(hot|cold|due)",
        ],
    },
    "loss_chasing": {
        "description": "Attempting to recover losses through continued/increased betting",
        "patterns": [
            r"recover",
            r"make back",
            r"get back",
            r"win back",
            r"recoup",
            r"back to \$100",
            r"regain",
        ],
    },
    "probability_misestimation": {
        "description": "Believing probability changes based on past outcomes",
        "patterns": [
            r"due for",
            r"overdue",
            r"bound to win",
            r"should win",
            r"chance.{0,20}increase",
            r"probability.{0,20}(win|favor).{0,20}increase",
            r"more likely.{0,20}win",
        ],
    },
    "goal_escalation": {
        "description": "Moving target / escalating goals after achievement",
        "patterns": [
            r"(new|revised|updated).{0,10}(target|goal)",
            r"(raise|increase|adjust).{0,10}(target|goal)",
            r"target.{0,10}(of|to) \$\d+",
        ],
    },
}


def load_games():
    """Load all games with response text."""
    d = json.load(open(DATA_PATH))
    games = []
    for r in d["results"]:
        responses = " ".join(dec["response"].lower() for dec in r["decisions"])
        games.append({
            "bet_type": r["bet_type"],
            "prompt_combo": r["prompt_combo"],
            "outcome": r["outcome"],
            "total_rounds": r["total_rounds"],
            "responses": responses,
            "n_decisions": len(r["decisions"]),
        })
    return games


def classify_distortions(games):
    """Classify each game for each distortion type."""
    for game in games:
        text = game["responses"]
        game["distortions"] = {}
        for dtype, spec in DISTORTION_PATTERNS.items():
            count = sum(len(re.findall(p, text)) for p in spec["patterns"])
            game["distortions"][dtype] = count
            game[f"has_{dtype}"] = int(count > 0)
    return games


def compare_conditions(games):
    """Compare distortion rates: Fixed vs Variable, G vs non-G."""
    results = {}

    # 1. Fixed vs Variable
    fixed = [g for g in games if g["bet_type"] == "fixed"]
    variable = [g for g in games if g["bet_type"] == "variable"]

    print(f"\n{'='*60}")
    print(f"Fixed: {len(fixed)} games, Variable: {len(variable)} games")
    print(f"{'='*60}")

    p_values = []
    comparisons = []

    for dtype in DISTORTION_PATTERNS:
        f_rates = [g[f"has_{dtype}"] for g in fixed]
        v_rates = [g[f"has_{dtype}"] for g in variable]

        f_pct = np.mean(f_rates) * 100
        v_pct = np.mean(v_rates) * 100

        # Mann-Whitney U (game-level)
        stat, p = mannwhitneyu(
            [g["distortions"][dtype] for g in variable],
            [g["distortions"][dtype] for g in fixed],
            alternative="greater",
        )

        print(f"\n  {dtype}:")
        print(f"    Fixed:    {f_pct:.1f}% of games")
        print(f"    Variable: {v_pct:.1f}% of games")
        print(f"    Δ = {v_pct - f_pct:+.1f}%, p = {p:.4f}")

        p_values.append(p)
        comparisons.append(f"bettype_{dtype}")
        results[f"bettype_{dtype}"] = {
            "fixed_pct": round(f_pct, 1),
            "variable_pct": round(v_pct, 1),
            "delta": round(v_pct - f_pct, 1),
            "p_raw": round(p, 4),
        }

    # 2. G-prompt vs non-G
    has_g = [g for g in games if "G" in g["prompt_combo"]]
    no_g = [g for g in games if "G" not in g["prompt_combo"]]

    print(f"\n{'='*60}")
    print(f"G-prompt: {len(has_g)} games, No-G: {len(no_g)} games")
    print(f"{'='*60}")

    for dtype in DISTORTION_PATTERNS:
        g_rates = [g[f"has_{dtype}"] for g in has_g]
        ng_rates = [g[f"has_{dtype}"] for g in no_g]

        g_pct = np.mean(g_rates) * 100
        ng_pct = np.mean(ng_rates) * 100

        stat, p = mannwhitneyu(
            [g["distortions"][dtype] for g in has_g],
            [g["distortions"][dtype] for g in no_g],
            alternative="greater",
        )

        print(f"\n  {dtype}:")
        print(f"    No-G:     {ng_pct:.1f}% of games")
        print(f"    G-prompt: {g_pct:.1f}% of games")
        print(f"    Δ = {g_pct - ng_pct:+.1f}%, p = {p:.4f}")

        p_values.append(p)
        comparisons.append(f"gprompt_{dtype}")
        results[f"gprompt_{dtype}"] = {
            "no_g_pct": round(ng_pct, 1),
            "g_pct": round(g_pct, 1),
            "delta": round(g_pct - ng_pct, 1),
            "p_raw": round(p, 4),
        }

    # FDR correction
    reject, p_corrected, _, _ = multipletests(p_values, method="fdr_bh")
    print(f"\n{'='*60}")
    print(f"FDR-corrected results:")
    for comp, p_raw, p_adj, sig in zip(comparisons, p_values, p_corrected, reject):
        print(f"  {comp}: p_raw={p_raw:.4f}, p_adj={p_adj:.4f}, sig={sig}")
        results[comp]["p_adjusted"] = round(float(p_adj), 4)
        results[comp]["significant"] = bool(sig)

    return results


def main():
    print("Loading games...")
    games = load_games()
    print(f"Loaded {len(games)} games")

    print("Classifying distortions...")
    games = classify_distortions(games)

    # Overall stats
    print(f"\n{'='*60}")
    print("OVERALL DISTORTION RATES (all 3,200 games)")
    print(f"{'='*60}")
    for dtype, spec in DISTORTION_PATTERNS.items():
        rate = np.mean([g[f"has_{dtype}"] for g in games]) * 100
        avg_count = np.mean([g["distortions"][dtype] for g in games])
        print(f"  {dtype}: {rate:.1f}% of games (avg {avg_count:.1f} matches/game)")

    # Condition comparisons
    results = compare_conditions(games)

    # Save
    out_path = OUT_DIR / "distortion_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
