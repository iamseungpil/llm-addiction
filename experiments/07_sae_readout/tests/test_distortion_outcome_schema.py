#!/usr/bin/env python3
"""Regression test for the three round-outcome schemas in the six-model corpora.

The corpora disagree about where a round's W/L is written, and a loader that knows
only one spelling silently reports zero post-loss decisions for the others rather
than failing. This test pins all three spellings against
``run_multimodel_distortion_analysis.load_games``.

  open-weight V4role  decisions[i]["result"] ("W"/"L") and decisions[i]["win"] (bool)
  four API exports    round_details[i]["game_result"]["result"] (a DICT, never str() it)
  GPT-4o-mini         no per-step outcome at all; game_history[i]["result"] one level up

Synthetic games only, so it runs anywhere with no dataset present.

    python3 experiments/07_sae_readout/tests/test_distortion_outcome_schema.py
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "run_multimodel_distortion_analysis.py"

# The task spec is a 30% win rate; every corpus must land in this band.
WIN_RATE_BAND = (0.25, 0.35)

N_GAMES = 40
N_ROUNDS = 60


def _load_module():
    spec = importlib.util.spec_from_file_location("distortion_mod", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["distortion_mod"] = module
    spec.loader.exec_module(module)
    return module


def _rounds(rng: random.Random) -> list[tuple[int, int, bool, int]]:
    balance, out = 100, []
    for round_no in range(1, N_ROUNDS + 1):
        bet = 10
        win = rng.random() < 0.30
        balance = balance - bet + (3 * bet if win else 0)
        out.append((round_no, bet, win, balance))
    return out


# Every response carries a distortion keyword so no step is dropped by extract_response.
RESPONSE = "the machine feels due for a hit"


def _game_api(rounds):
    return {
        "bet_type": "variable", "prompt_combo": "GM", "repetition": 0,
        "total_rounds": len(rounds), "final_balance": rounds[-1][3],
        "round_details": [
            {"round": r, "response": RESPONSE, "decision": "continue", "bet_amount": b,
             "game_result": {"round": r, "bet": b, "result": "W" if w else "L", "balance": bal}}
            for r, b, w, bal in rounds
        ],
    }


def _game_v4role(rounds):
    return {
        "bet_type": "fixed", "prompt_combo": "GM", "repetition": 0,
        "total_rounds": len(rounds), "final_balance": rounds[-1][3],
        "decisions": [
            {"round": r, "response": RESPONSE, "decision": "continue", "bet": b,
             "result": "W" if w else "L", "win": w,
             "balance_before": bal - (3 * b if w else 0) + b, "balance": bal}
            for r, b, w, bal in rounds
        ],
    }


def _game_gpt4o_fixed_parsing(rounds):
    """round_details carries the prompt/response/bet; the W/L lives in game_history."""
    return {
        "bet_type": "variable", "prompt_combo": "GM", "repetition": 0,
        "total_rounds": len(rounds), "final_balance": rounds[-1][3],
        "round_details": [
            {"round": r, "prompt": "Current balance: $100", "gpt_response_full": RESPONSE,
             "decision": "continue", "bet_amount": b, "parsing_info": "ok"}
            for r, b, w, bal in rounds
        ],
        "game_history": [
            {"round": r, "bet": b, "result": "W" if w else "L", "balance": bal}
            for r, b, w, bal in rounds
        ],
    }


CORPORA = {
    "api_round_details_game_result": _game_api,
    "v4role_decisions": _game_v4role,
    "gpt4o_mini_fixed_parsing": _game_gpt4o_fixed_parsing,
}


def main() -> int:
    module = _load_module()
    rng = random.Random(7)
    tmp = Path(tempfile.mkdtemp(prefix="distortion_schema_"))
    failures: list[str] = []

    print(f"{'corpus':32s} {'decisions':>9s} {'post_loss':>9s} {'post_win':>8s} {'win_rate':>8s}")
    for name, builder in CORPORA.items():
        payload = {"results": [builder(_rounds(rng)) for _ in range(N_GAMES)], "timestamp": "n/a"}
        path = tmp / f"{name}.json"
        path.write_text(json.dumps(payload))

        source = module.DataSource(
            model_key=name, display_name=name, path=path, provenance="synthetic"
        )
        _, decisions = module.load_games(source)
        post_loss = sum(1 for d in decisions if d["post_loss"])
        post_win = sum(1 for d in decisions if d["post_win"])
        resolved = post_loss + post_win

        win_rate = post_win / resolved if resolved else float("nan")
        print(f"{name:32s} {len(decisions):9d} {post_loss:9d} {post_win:8d} {win_rate:8.3f}")

        if not resolved:
            failures.append(f"{name}: no round outcome resolved (schema not handled)")
        elif not (WIN_RATE_BAND[0] <= win_rate <= WIN_RATE_BAND[1]):
            failures.append(f"{name}: win rate {win_rate:.3f} outside {WIN_RATE_BAND}")

    print()
    if failures:
        for line in failures:
            print(f"FAIL  {line}")
        return 1
    print("PASS  all three outcome schemas resolve, every win rate inside "
          f"[{WIN_RATE_BAND[0]}, {WIN_RATE_BAND[1]}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
