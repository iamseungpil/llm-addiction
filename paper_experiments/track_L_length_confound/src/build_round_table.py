"""Round-level table builder for Track L Plan v3.1.

Reads paper-canonical IC cap-variation + SM v4_role JSONs and emits a tidy
long table of one row per round per game with the bankruptcy / voluntary-stop
competing-event coding required by the multinomial cause-specific hazard fit.

Schema (per Plan v3.1 §3.2):
  dataset           : "IC" | "SM"
  file_timestamp    : str  -- from JSON top-level `timestamp`
  model             : str  -- "llama" | "gemma"
  cap               : int  -- {10, 30, 50, 70} for IC; {10} for SM
  prompt_combo      : str  -- BASE / G / M / GM (IC) | encoded SM cond
  bet_type          : str  -- "variable" | "fixed"
  game_id           : int  -- per-file game index
  round             : int  -- 1-based
  balance_before    : int
  bet_amount        : int
  decision          : str  -- "bet" | "stop"  (parser tag)
  balance_after     : int
  outcome           : str  -- "continue" | "bankrupt" | "voluntary_stop"
  total_rounds_in_game : int
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


def _classify_outcome(
    is_terminal_round: bool,
    bet_amount: int,
    balance_before: int,
    win: bool,
    balance_after: int,
    game_bankruptcy_flag: bool,
    game_final_outcome: str,
) -> str:
    """Plan v3 §3.2: bankruptcy = bet placed AND bet >= balance_before AND loss
    AND balance_after == 0. Otherwise voluntary_stop terminates the game.
    Otherwise continue.
    """
    if not is_terminal_round:
        return "continue"
    bet_at_or_above_balance = bet_amount > 0 and bet_amount >= balance_before
    is_loss = not win
    busted = balance_after == 0
    if bet_at_or_above_balance and is_loss and busted and game_bankruptcy_flag:
        return "bankrupt"
    if game_final_outcome in ("voluntary_stop", "stop"):
        return "voluntary_stop"
    return "voluntary_stop"


def _ic_iter_rows(payload: dict) -> Iterator[RoundRow]:
    file_timestamp = payload["timestamp"]
    model = payload["model"]
    for game in payload["results"]:
        cap = int(game["bet_constraint"])
        bet_type = game["bet_type"]
        prompt_combo = game.get("prompt_condition", "BASE")
        game_id = game["game_id"]
        history = game["history"]
        n_rounds = len(history)
        bankruptcy_flag = bool(game.get("bankruptcy", False))
        final_outcome = game.get("final_outcome", "voluntary_stop")
        for j, rec in enumerate(history):
            is_terminal = (j == n_rounds - 1) or bool(rec.get("is_finished", False))
            decision = "stop" if rec.get("outcome") == "stop" else "bet"
            outcome_label = _classify_outcome(
                is_terminal_round=is_terminal,
                bet_amount=int(rec["bet"]),
                balance_before=int(rec["balance_before"]),
                win=bool(rec["win"]),
                balance_after=int(rec["balance_after"]),
                game_bankruptcy_flag=bankruptcy_flag,
                game_final_outcome=final_outcome,
            )
            yield RoundRow(
                dataset="IC",
                file_timestamp=file_timestamp,
                model=model,
                cap=cap,
                prompt_combo=prompt_combo,
                bet_type=bet_type,
                game_id=game_id,
                round=int(rec["round"]),
                balance_before=int(rec["balance_before"]),
                bet_amount=int(rec["bet"]),
                decision=decision,
                balance_after=int(rec["balance_after"]),
                outcome=outcome_label,
                total_rounds_in_game=n_rounds,
            )


def _sm_iter_rows(payload: dict, initial_balance: int = 100) -> Iterator[RoundRow]:
    """SM v4_role schema (different from IC):
    - Game-level: {bet_type, prompt_combo, repetition, outcome, final_balance,
                   total_rounds, total_bet, total_won, history[], decisions[]}
    - history[j]: {round, bet, result ('W'/'L'), balance (post-bet), win}
    - outcome at game level uses "bankruptcy" (NOT "bankrupt") and "voluntary_stop"
    - No balance_before / balance_after on rounds — derive from prev round's balance
    - No game_id — use repetition index
    - cap is fixed at $10 for SM v4_role panel (per Plan v3.1 §3.1)
    """
    file_timestamp = payload.get("timestamp", "")
    model = payload.get("model", "")
    for rep_idx, game in enumerate(payload["results"]):
        cap = 10
        bet_type = game["bet_type"]
        prompt_combo = game.get("prompt_combo") or game.get("prompt_condition", "UNKNOWN")
        game_id = int(game.get("game_id", game.get("repetition", rep_idx)))
        history = game.get("history") or []
        n_rounds = len(history)
        sm_outcome = game.get("outcome", "")
        bankruptcy_flag = sm_outcome == "bankruptcy"
        final_outcome = "bankrupt" if bankruptcy_flag else (
            "voluntary_stop" if sm_outcome in ("voluntary_stop", "stop") else sm_outcome
        )
        prev_balance = initial_balance
        for j, rec in enumerate(history):
            is_terminal = (j == n_rounds - 1)
            bet_amount = int(rec.get("bet", 0))
            balance_after = int(rec.get("balance", prev_balance))
            balance_before = prev_balance
            win = bool(rec.get("win", rec.get("result", "L") == "W"))
            decision = "bet" if bet_amount > 0 else "stop"
            outcome_label = _classify_outcome(
                is_terminal_round=is_terminal,
                bet_amount=bet_amount,
                balance_before=balance_before,
                win=win,
                balance_after=balance_after,
                game_bankruptcy_flag=bankruptcy_flag,
                game_final_outcome=final_outcome,
            )
            yield RoundRow(
                dataset="SM",
                file_timestamp=file_timestamp,
                model=model,
                cap=cap,
                prompt_combo=str(prompt_combo),
                bet_type=bet_type,
                game_id=game_id,
                round=int(rec.get("round", j + 1)),
                balance_before=balance_before,
                bet_amount=bet_amount,
                decision=decision,
                balance_after=balance_after,
                outcome=outcome_label,
                total_rounds_in_game=n_rounds,
            )
            prev_balance = balance_after


def build_table(ic_files: list[Path], sm_files: list[Path]) -> list[RoundRow]:
    rows: list[RoundRow] = []
    for f in ic_files:
        payload = json.loads(Path(f).read_text())
        rows.extend(_ic_iter_rows(payload))
    for f in sm_files:
        payload = json.loads(Path(f).read_text())
        rows.extend(_sm_iter_rows(payload))
    return rows


def discover_default_files(data_root: Path) -> tuple[list[Path], list[Path]]:
    ic_root = data_root / "investment_choice"
    sm_root = data_root / "slot_machine"
    ic = sorted(ic_root.glob("v2_role_*/*.json")) if ic_root.exists() else []
    sm = sorted(sm_root.glob("*_v4_role/final_*.json")) if sm_root.exists() else []
    return ic, sm


if __name__ == "__main__":
    import argparse
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/v-seungplee/data/llm-addiction/behavioral")
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/track_L_length_confound/round_table.csv")
    ap.add_argument("--smoke", action="store_true", help="single IC file only")
    args = ap.parse_args()

    ic, sm = discover_default_files(Path(args.data_root))
    if args.smoke:
        ic, sm = ic[:1], []
    print(f"IC files: {len(ic)}  SM files: {len(sm)}")
    rows = build_table(ic, sm)
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
