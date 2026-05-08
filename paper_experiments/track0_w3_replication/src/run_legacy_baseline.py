"""Track 0 W3 parity baseline shim — runs the legacy GPT-4o-mini experiments verbatim.

Track 0 W3 Plan v5.2 §3.5.1 + §8: the rebuttal hinges on byte-for-byte protocol
parity with the original GPT-4o-mini fixed-bet and variable-bet experiments. This
shim invokes the legacy classes EXACTLY as they were written — no patches, no
overrides on prompt or sampling — except for two attribute redirects:

  1. `self.results_dir` is pointed at our `--output_dir` so the legacy class drops
     its JSON inside the parity directory we own.
  2. `self.bet_sizes` (fixed) / `self.max_bets` (variable) is narrowed to a single
     cap so each invocation matches one Track 0 cell.

The legacy class itself is imported untouched from `legacy/gpt_*_experiment/src/`.
We then loop `args.n_games` games at `prompt_combo="BASE"` (the only combo Track 0
uses) and emit one Track-0-shaped payload that `parity_check.py` can compare against
the v6 cells one-for-one.

Usage:
    python run_legacy_baseline.py --cap 70 --mode variable --n_games 200 \
        --output_dir /scratch/x3415a02/data/llm-addiction/track0_w3/parity_legacy_baseline/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# Paper-canonical sm cap-ablation source — moved out of legacy/ into
# paper_experiments/sm_cap_ablation/ on 2026-05-08 for findability. The single
# directory now hosts both experiment classes + improved_gpt_parsing.
SM_CAP_ABLATION_ROOT = "/home/v-seungplee/llm-addiction/paper_experiments/sm_cap_ablation/src"
sys.path.insert(0, SM_CAP_ABLATION_ROOT)

from gpt_fixed_bet_size_experiment import GPTFixedBetSizeExperiment  # noqa: E402
from gpt_variable_max_bet_experiment import GPTVariableMaxBetExperiment  # noqa: E402


def _track0_payload_from_legacy_results(
    legacy_results: list,
    *,
    cap: int,
    mode: str,
    n_games: int,
    timestamp: str,
) -> dict:
    """Translate legacy result dicts into the Track-0 payload schema used by
    `run_track0_api.py` so `parity_check.py` can consume both shapes uniformly."""
    out = []
    for i, r in enumerate(legacy_results):
        bankrupt = bool(r["is_bankrupt"])
        voluntary = bool(r["voluntary_stop"])
        outcome = "bankruptcy" if bankrupt else ("voluntary_stop" if voluntary else "max_rounds")
        history = []
        for h in r.get("game_history", []):
            history.append({
                "round": h["round"],
                "bet": h["bet"],
                "result": h["result"],
                "balance": h["balance"],
                "win": h.get("win", h["result"] == "W"),
                "winnings": int(h["bet"] * 3.0) if h.get("win", h["result"] == "W") else 0,
            })
        out.append({
            "game_id": i,
            "model": "gpt-4o-mini",
            "model_id": "gpt-4o-mini",
            "provider": "openai",
            "cap": cap,
            "mode": mode,
            "prompt_combo": r.get("prompt_combo", "BASE"),
            "outcome": outcome,
            "bankrupt": bankrupt,
            "final_balance": int(r["final_balance"]),
            "total_rounds": int(r["total_rounds"]),
            "total_bet": int(r["total_bet"]),
            "total_won": int(r["total_won"]),
            "history": history,
            "rounds": [],  # legacy dataclass stores round-level prompts under round_details; omit to keep payload small
            "seed": None,
        })
    return {
        "track": "0_w3_parity_legacy_baseline",
        "model": "gpt-4o-mini",
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "cap": cap,
        "mode": mode,
        "n_games": n_games,
        "smoke": False,
        "config_snapshot": {
            "generation": {
                "initial_balance": 100,
                "win_rate": 0.30,
                "payout": 3.0,
                "max_rounds_fixed": 100,
                "max_rounds_variable": 50,
            },
            "source": "legacy verbatim",
        },
        "timestamp": timestamp,
        "results": out,
    }


def _run_fixed(args: argparse.Namespace, output_dir: Path, timestamp: str) -> None:
    # Round-2 C1: bypass GPTFixedBetSizeExperiment.__init__ — its hardcoded
    # `Path('/home/ubuntu/llm_addiction/...').mkdir(...)` call fails on this
    # host, and even if it succeeded the constructor sets `self.log_file`
    # eagerly, so a post-hoc `self.logs_dir =` redirect leaves `self.log_file`
    # pointing at the unwritable original path. The `__new__` pattern
    # (mirroring tests/test_protocol_parity.py:171) attaches every attribute
    # `run_single_game` and `log()` actually read — verified against
    # legacy/gpt_fixed_bet_size_experiment/src/gpt_fixed_bet_size_experiment.py:108-144.
    exp = GPTFixedBetSizeExperiment.__new__(GPTFixedBetSizeExperiment)
    exp.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY"))
    exp.model_name = "gpt-4o-mini"
    exp.win_rate = 0.3
    exp.payout = 3.0
    exp.max_rounds = 100
    exp.bet_sizes = [args.cap]
    exp.results_dir = output_dir
    exp.logs_dir = output_dir
    exp.log_file = output_dir / f"legacy_log_{timestamp}.log"
    exp.results = []
    exp.current_experiment = 0
    # Force prompt_combo = BASE; the legacy class otherwise iterates 32 combos.
    legacy_results = []
    for rep in range(args.n_games):
        exp.current_experiment += 1
        result = exp.run_single_game(
            bet_size=args.cap,
            prompt_combo="BASE",
            condition_id=0,
            repetition=rep,
        )
        legacy_results.append(result)
    payload = _track0_payload_from_legacy_results(
        legacy_results, cap=args.cap, mode="fixed", n_games=args.n_games, timestamp=timestamp,
    )
    fname = f"legacy_gpt-4o-mini_cap{args.cap}_fixed_{timestamp}.json"
    out_path = output_dir / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[track0/legacy_baseline] wrote {out_path}")


def _run_variable(args: argparse.Namespace, output_dir: Path, timestamp: str) -> None:
    # Round-2 C1: same __new__ bypass as fixed. Variable runner's own
    # __init__ (legacy/gpt_variable_max_bet_experiment/src/gpt_variable_max_bet_experiment.py:108-144)
    # has the same hardcoded /home/ubuntu mkdir + eager log_file bug, plus
    # uses `max_bets` (not `bet_sizes`) and max_rounds=50 (not 100).
    exp = GPTVariableMaxBetExperiment.__new__(GPTVariableMaxBetExperiment)
    exp.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GPT_API_KEY"))
    exp.model_name = "gpt-4o-mini"
    exp.win_rate = 0.3
    exp.payout = 3.0
    exp.max_rounds = 50
    exp.max_bets = [args.cap]
    exp.results_dir = output_dir
    exp.logs_dir = output_dir
    exp.log_file = output_dir / f"legacy_log_{timestamp}.log"
    exp.results = []
    exp.current_experiment = 0
    legacy_results = []
    for rep in range(args.n_games):
        exp.current_experiment += 1
        result = exp.run_single_game(
            max_bet=args.cap,
            prompt_combo="BASE",
            condition_id=0,
            repetition=rep,
        )
        legacy_results.append(result)
    payload = _track0_payload_from_legacy_results(
        legacy_results, cap=args.cap, mode="variable", n_games=args.n_games, timestamp=timestamp,
    )
    fname = f"legacy_gpt-4o-mini_cap{args.cap}_variable_{timestamp}.json"
    out_path = output_dir / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[track0/legacy_baseline] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track 0 W3 parity baseline: run legacy GPT-4o-mini code verbatim",
    )
    parser.add_argument("--cap", type=int, required=True, choices=[10, 30, 50, 70])
    parser.add_argument("--mode", required=True, choices=["fixed", "variable"])
    parser.add_argument("--n_games", type=int, default=200)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GPT_API_KEY")):
        raise RuntimeError("OPENAI_API_KEY (or GPT_API_KEY) must be set for parity baseline runs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Seed Python's RNG for the slot machine; the legacy classes use the global
    # `random` module, so seeding here gives us reproducible W/L draws.
    random.seed(42)

    print(f"[track0/legacy_baseline] cap={args.cap} mode={args.mode} n_games={args.n_games}")
    t0 = time.time()
    if args.mode == "fixed":
        _run_fixed(args, output_dir, timestamp)
    else:
        _run_variable(args, output_dir, timestamp)
    print(f"[track0/legacy_baseline] elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
