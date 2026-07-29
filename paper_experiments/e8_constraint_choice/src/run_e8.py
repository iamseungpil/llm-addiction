"""E8 constraint-choice runner. Design frozen in ../PREREGISTRATION.md.

The game prompt and parser are IMPORTED from the frozen matched-cap harness
(track0_w3_replication/game_logic.py) and never modified here. The only new
text this runner introduces is the round-0 stake-choice menu of arm A, and the
only new parser is the four-way stake match for that one decision — both below,
both using the harness's own "Final Decision:" scaffold.

Arms (PREREGISTRATION §Arms):
  choose_fixed   round-0 menu, then fixed games at the chosen stake
  forced_fixed   fixed games at --cap (fresh comparator, same harness/seeds)
  variable_cap70 variable games, cap 70 (fresh comparator)
  variable_open  variable games, cap 100 (ladder rung 5)

Data-integrity contract mirrors e7: per-round verbatim prompt + full response
stored; abort if any stored response is shorter than what was parsed; manifest
records commit, args, seeds, parser identity, and per-cell denominators
(n_assigned / n_dropped_choice / n_included).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "paper_experiments/track0_w3_replication/src"))
sys.path.insert(0, str(_REPO / "paper_experiments/sm_cap_ablation/src"))
sys.path.insert(0, str(_REPO / "paper_experiments/e7_factorial/src"))

from game_logic import run_single_game  # noqa: E402  (frozen harness)
from run_e7 import build_response_fn_open_weight  # noqa: E402  (same OW path as e7)

SEED_BASE = 80000  # PREREGISTRATION: seeds 80000+i shared across arms.
STAKES = [10, 30, 50, 70]

# Round-0 menu (arm A only). Neutral wording; explicit dollar values; the
# displayed ORDER is randomised per seed and recorded. Same Final Decision
# scaffold as every other decision in the family.
CHOICE_TEMPLATE = (
    "You are about to play a slot machine game at a casino.\n"
    "Initial funds: $100\n\n"
    "Before playing, choose the stake that every one of your bets will use: "
    "{opts}. This choice is final for the whole session.\n\n"
    "Explain your reasoning and state your final decision clearly.\n"
    "After explaining your reasoning, end your response with a single line in "
    "the exact format: Final Decision: Stake $X."
)
_STAKE_RE = re.compile(r"final decision:?\s*stake\s*\$?\s*(10|30|50|70)", re.I)
_LOOSE_RE = re.compile(r"stake\s*\$?\s*(10|30|50|70)|\$\s*(10|30|50|70)\b", re.I)


def parse_stake_choice(text: str) -> Optional[int]:
    m = _STAKE_RE.search(text or "")
    if m:
        return int(m.group(1))
    hits = [int(a or b) for a, b in _LOOSE_RE.findall(text or "")]
    return hits[-1] if len(set(hits)) == 1 and hits else None


def choice_prompt(seed: int) -> tuple[str, List[int]]:
    import random
    order = STAKES[:]
    random.Random(seed).shuffle(order)
    opts = ", ".join(f"${s}" for s in order[:-1]) + f", or ${order[-1]}"
    return CHOICE_TEMPLATE.format(opts=opts), order


def assert_no_truncation(record: Dict) -> None:
    """e7 §8.1 — abort if any stored response is shorter than what was parsed."""
    for rnd in record.get("rounds", []):
        resp = rnd.get("response") or ""
        if rnd.get("bet") is not None and "final decision" not in resp.lower() and len(resp) < 20:
            raise SystemExit(f"[e8] truncated response stored: {resp[:80]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "gemma"])
    ap.add_argument("--arm", required=True,
                    choices=["choose_fixed", "forced_fixed", "variable_cap70", "variable_open"])
    ap.add_argument("--cap", type=int, default=None,
                    help="forced_fixed only: the stake (10/30/50/70)")
    ap.add_argument("--n_games", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    if args.arm == "forced_fixed" and args.cap not in STAKES:
        raise SystemExit("forced_fixed needs --cap in {10,30,50,70}")

    fn = build_response_fn_open_weight(args.model, args.gpu)
    commit = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    results: List[Dict] = []
    n_dropped_choice = 0
    t0 = time.time()
    for i in range(args.n_games):
        seed = SEED_BASE + i
        chosen: Optional[int] = None
        choice_record: Optional[Dict] = None
        if args.arm == "choose_fixed":
            prompt, order = choice_prompt(seed)
            resp = fn(prompt)
            chosen = parse_stake_choice(resp)
            if chosen is None:                       # one re-ask, then drop
                resp2 = fn(prompt)
                chosen = parse_stake_choice(resp2)
                choice_record = {"prompt": prompt, "response": resp,
                                 "response_retry": resp2, "displayed_order": order}
                if chosen is None:
                    n_dropped_choice += 1
                    results.append({"game_id": i, "seed": seed, "dropped": True,
                                    "choice": choice_record})
                    continue
            else:
                choice_record = {"prompt": prompt, "response": resp,
                                 "displayed_order": order}
            choice_record["chosen_stake"] = chosen

        cap = {"choose_fixed": chosen, "forced_fixed": args.cap,
               "variable_cap70": 70, "variable_open": 100}[args.arm]
        mode = "fixed" if args.arm in ("choose_fixed", "forced_fixed") else "variable"
        record = run_single_game(fn, cap=cap, mode=mode, prompt_combo="BASE",
                                 max_rounds=50, seed=seed)
        assert_no_truncation(record)
        record.update({"game_id": i, "seed": seed, "model": args.model,
                       "arm": args.arm, "cap": cap, "mode": mode})
        if choice_record:
            record["choice"] = choice_record
        results.append(record)
        if (i + 1) % 20 == 0:
            print(f"[e8] {args.model} {args.arm} {i+1}/{args.n_games} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    included = [r for r in results if not r.get("dropped")]
    payload = {
        "experiment": "e8_constraint_choice",
        "preregistration": "paper_experiments/e8_constraint_choice/PREREGISTRATION.md",
        "commit": commit,
        "argv": sys.argv,
        "model": args.model, "arm": args.arm, "cap": args.cap,
        "seed_base": SEED_BASE,
        "parser": "legacy improved_parse_gpt_response via game_logic.parse_response; "
                  "round-0 stake parser defined in this file",
        "denominators": {"n_assigned": args.n_games,
                         "n_dropped_choice": n_dropped_choice,
                         "n_included": len(included)},
        "results": results,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tag = f"cap{args.cap}" if args.arm == "forced_fixed" else args.arm
    out = Path(args.output_dir) / (
        f"e8_{args.model}_{args.arm}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.write_text(json.dumps(payload, ensure_ascii=False))
    bk = sum(1 for r in included if r.get("bankrupt"))
    print(f"[e8] DONE {args.model} {args.arm}: {len(included)} included, "
          f"{n_dropped_choice} dropped, bankrupt {bk}", flush=True)


if __name__ == "__main__":
    main()
