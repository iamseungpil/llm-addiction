"""Round-level table builder for Track L Plan v3.2.

Reads paper-canonical IC + SM JSONs across 4 schemas and emits a tidy long
table of one row per round per game. Per codex Round 5 verdict, the four
schemas are tagged separately so the primary multinomial cause-specific
hazard fit can run segregated within {IC_API, IC_OW, SM_API, SM_OW}; pooled
fit is sensitivity only.

Schemas handled
  IC_OW   open-weight IC v2_role (LLaMA, Gemma)
            top: experiment, model, timestamp, config, results
            game: bet_constraint, bet_type, prompt_condition, game_id,
                  bankruptcy, final_outcome, history[]
            round: round, balance_before, bet, choice, outcome, win,
                   payout, balance_after, is_finished

  IC_API  API IC bet_constraint (gpt4o_mini, gpt41_mini, claude_haiku, gemini_flash)
            top: experiment_config, summary_statistics, results
            game: game_id, model, bet_type, prompt_condition, trial,
                  rounds_played, final_balance, exit_reason, decisions
            round: balance_before, balance_after, bet, choice, outcome,
                   payout, prompt, response

  SM_OW   open-weight SM v4_role (LLaMA, Gemma)
            top: timestamp, model, ..., results
            game: bet_type, prompt_combo, repetition, outcome,
                  final_balance, total_rounds, history[]
            round: round, bet, result ('W'/'L'), balance (post-bet), win

  SM_API  API SM (Claude, Gemini, GPT-4.1-mini)
            top: timestamp, model, experiment_config, results
            game: condition_id, bet_type, prompt_combo, repetition,
                  total_rounds, final_balance, is_bankrupt (bool),
                  voluntary_stop (bool), round_details[]
            round: round, decision, bet_amount, game_result: {bet, result,
                   balance (post-bet), win}

Output schema (per Plan v3.2 §3.2):
  dataset, file_timestamp, model, cap, prompt_combo, bet_type, game_id,
  round, balance_before, bet_amount, decision, balance_after, outcome,
  total_rounds_in_game

Where outcome in {continue, bankrupt, voluntary_stop} is the cause-specific
event coding; bankruptcy = bet placed AND bet>=balance_before AND loss AND
balance_after==0 AND game-level bankruptcy flag, on terminal round only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RoundRow:
    dataset: str
    file_timestamp: str
    model: str
    cap: int
    prompt_combo: str
    bet_type: str
    game_id: int
    round: int
    balance_before: int
    bet_amount: int
    decision: str
    balance_after: int
    outcome: str
    total_rounds_in_game: int


def _classify_terminal_outcome(
    bet_amount: int,
    balance_before: int,
    win: bool,
    balance_after: int,
    game_bankruptcy_flag: bool,
) -> str:
    """Bankruptcy = bet placed AND bet>=balance_before AND loss AND
    balance_after==0 AND game-level bankruptcy flag (Plan v3.2 §3.2).
    Otherwise the terminal round was a voluntary stop.
    """
    bet_at_or_above = bet_amount > 0 and bet_amount >= balance_before
    is_loss = not win
    busted = balance_after == 0
    if bet_at_or_above and is_loss and busted and game_bankruptcy_flag:
        return "bankrupt"
    return "voluntary_stop"


def _ic_ow_iter_rows(payload: dict) -> Iterator[RoundRow]:
    file_timestamp = payload["timestamp"]
    model = payload["model"]
    for game in payload["results"]:
        cap = int(game["bet_constraint"])
        bet_type = game["bet_type"]
        prompt_combo = game.get("prompt_condition", "BASE")
        game_id = int(game["game_id"])
        history = game["history"]
        n = len(history)
        bankruptcy_flag = bool(game.get("bankruptcy", False))
        for j, rec in enumerate(history):
            is_terminal = (j == n - 1) or bool(rec.get("is_finished", False))
            balance_before = int(rec["balance_before"])
            bet_amount = int(rec["bet"])
            balance_after = int(rec["balance_after"])
            win = bool(rec["win"])
            decision = "stop" if rec.get("outcome") == "stop" else "bet"
            if is_terminal:
                out = _classify_terminal_outcome(bet_amount, balance_before, win, balance_after, bankruptcy_flag)
            else:
                out = "continue"
            yield RoundRow(
                dataset="IC_OW", file_timestamp=file_timestamp, model=model,
                cap=cap, prompt_combo=prompt_combo, bet_type=bet_type,
                game_id=game_id, round=int(rec["round"]),
                balance_before=balance_before, bet_amount=bet_amount,
                decision=decision, balance_after=balance_after,
                outcome=out, total_rounds_in_game=n,
            )


def _ic_api_iter_rows(payload: dict, file_timestamp: str) -> Iterator[RoundRow]:
    cfg = payload["experiment_config"]
    cap = int(cfg["bet_constraint"])
    file_bet_type = cfg["bet_type"]
    file_model = cfg["model"]
    for game in payload["results"]:
        bet_type = game.get("bet_type", file_bet_type)
        model = game.get("model", file_model)
        prompt_combo = game.get("prompt_condition", "BASE")
        game_id = int(game.get("game_id", game.get("trial", 0)))
        decisions = game.get("decisions", [])
        n = len(decisions)
        exit_reason = game.get("exit_reason", "")
        bankruptcy_flag = exit_reason == "bankruptcy"
        for j, rec in enumerate(decisions):
            is_terminal = (j == n - 1)
            balance_before = int(rec["balance_before"])
            bet_amount = int(rec["bet"])
            balance_after = int(rec["balance_after"])
            outcome_str = str(rec.get("outcome", ""))
            # win derived from outcome string suffix; "stop" / "choice_1" mean voluntary
            win = "_win" in outcome_str or rec.get("payout", 0) > 0 and "_loss" not in outcome_str
            decision = "stop" if outcome_str in ("stop", "choice_1") or "stop" in outcome_str.lower() else "bet"
            if is_terminal:
                out = _classify_terminal_outcome(bet_amount, balance_before, win, balance_after, bankruptcy_flag)
            else:
                out = "continue"
            yield RoundRow(
                dataset="IC_API", file_timestamp=file_timestamp, model=model,
                cap=cap, prompt_combo=prompt_combo, bet_type=bet_type,
                game_id=game_id, round=int(rec.get("round", j + 1)),
                balance_before=balance_before, bet_amount=bet_amount,
                decision=decision, balance_after=balance_after,
                outcome=out, total_rounds_in_game=n,
            )


def _sm_ow_iter_rows(payload: dict, initial_balance: int = 100) -> Iterator[RoundRow]:
    file_timestamp = payload.get("timestamp", "")
    model = payload.get("model", "")
    for rep_idx, game in enumerate(payload["results"]):
        cap = 10
        bet_type = game["bet_type"]
        prompt_combo = game.get("prompt_combo") or game.get("prompt_condition", "UNKNOWN")
        game_id = int(game.get("game_id", game.get("repetition", rep_idx)))
        history = game.get("history") or []
        n = len(history)
        sm_outcome = game.get("outcome", "")
        bankruptcy_flag = sm_outcome == "bankruptcy"
        prev_balance = initial_balance
        for j, rec in enumerate(history):
            is_terminal = (j == n - 1)
            bet_amount = int(rec.get("bet", 0))
            balance_after = int(rec.get("balance", prev_balance))
            balance_before = prev_balance
            win = bool(rec.get("win", rec.get("result", "L") == "W"))
            decision = "bet" if bet_amount > 0 else "stop"
            if is_terminal:
                out = _classify_terminal_outcome(bet_amount, balance_before, win, balance_after, bankruptcy_flag)
            else:
                out = "continue"
            yield RoundRow(
                dataset="SM_OW", file_timestamp=file_timestamp, model=model,
                cap=cap, prompt_combo=str(prompt_combo), bet_type=bet_type,
                game_id=game_id, round=int(rec.get("round", j + 1)),
                balance_before=balance_before, bet_amount=bet_amount,
                decision=decision, balance_after=balance_after,
                outcome=out, total_rounds_in_game=n,
            )
            prev_balance = balance_after


def _sm_api_iter_rows(payload: dict, initial_balance: int = 100) -> Iterator[RoundRow]:
    file_timestamp = payload.get("timestamp", "")
    model = payload.get("model", "")
    for rep_idx, game in enumerate(payload["results"]):
        cap = 10
        bet_type = game["bet_type"]
        prompt_combo = game.get("prompt_combo", "UNKNOWN")
        game_id = int(game.get("condition_id", 0)) * 10000 + int(game.get("repetition", rep_idx))
        round_details = game.get("round_details") or []
        n = len(round_details)
        bankruptcy_flag = bool(game.get("is_bankrupt", False))
        prev_balance = initial_balance
        for j, rec in enumerate(round_details):
            is_terminal = (j == n - 1)
            decision = "bet" if rec.get("decision") == "continue" else "stop"
            bet_amount = int(rec.get("bet_amount") or 0)
            gr = rec.get("game_result") or {}
            balance_after = int(gr.get("balance", prev_balance) if gr else prev_balance)
            balance_before = prev_balance
            win = bool(gr.get("win", gr.get("result", "L") == "W"))
            if is_terminal:
                out = _classify_terminal_outcome(bet_amount, balance_before, win, balance_after, bankruptcy_flag)
            else:
                out = "continue"
            yield RoundRow(
                dataset="SM_API", file_timestamp=file_timestamp, model=model,
                cap=cap, prompt_combo=str(prompt_combo), bet_type=bet_type,
                game_id=game_id, round=int(rec.get("round", j + 1)),
                balance_before=balance_before, bet_amount=bet_amount,
                decision=decision, balance_after=balance_after,
                outcome=out, total_rounds_in_game=n,
            )
            prev_balance = balance_after


def build_table(
    ic_ow_files: list[Path],
    ic_api_files: list[Path],
    sm_ow_files: list[Path],
    sm_api_files: list[Path],
) -> list[RoundRow]:
    rows: list[RoundRow] = []
    for f in ic_ow_files:
        rows.extend(_ic_ow_iter_rows(json.loads(Path(f).read_text())))
    for f in ic_api_files:
        ts = Path(f).stem.split("_")[-1]  # e.g. "104448" from "..._20251121_104448"
        date = Path(f).stem.split("_")[-2]  # "20251121"
        rows.extend(_ic_api_iter_rows(json.loads(Path(f).read_text()), f"{date}_{ts}"))
    for f in sm_ow_files:
        rows.extend(_sm_ow_iter_rows(json.loads(Path(f).read_text())))
    for f in sm_api_files:
        rows.extend(_sm_api_iter_rows(json.loads(Path(f).read_text())))
    return rows


def discover_default_files(data_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {"ic_ow": [], "ic_api": [], "sm_ow": [], "sm_api": []}
    ic_ow = data_root / "behavioral" / "investment_choice"
    if ic_ow.exists():
        out["ic_ow"] = sorted(ic_ow.glob("v2_role_*/*.json"))
    ic_api = data_root / "investment_choice" / "bet_constraint" / "results"
    if ic_api.exists():
        out["ic_api"] = sorted(ic_api.glob("*.json"))
    sm_ow = data_root / "behavioral" / "slot_machine"
    if sm_ow.exists():
        out["sm_ow"] = sorted(sm_ow.glob("*_v4_role/final_*.json"))
    sm_api = data_root / "slot_machine"
    if sm_api.exists():
        out["sm_api"] = [
            sm_api / "claude/claude_experiment_corrected_20250925.json",
            sm_api / "gemini/gemini_experiment_20250920_042809.json",
            sm_api / "gpt/gpt5_experiment_20250921_174509.json",
        ]
        out["sm_api"] = [p for p in out["sm_api"] if p.exists()]
    return out


if __name__ == "__main__":
    import argparse
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/v-seungplee/data/llm-addiction")
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/track_L_length_confound/round_table.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    found = discover_default_files(Path(args.data_root))
    if args.smoke:
        for k in found:
            found[k] = found[k][:1]
    for k, v in found.items():
        print(f"  {k}: {len(v)} files")
    rows = build_table(found["ic_ow"], found["ic_api"], found["sm_ow"], found["sm_api"])
    print(f"rows: {len(rows)}")
    if rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].__dict__.keys())
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r.__dict__)
        print(f"wrote {out}")
