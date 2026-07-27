#!/usr/bin/env python
"""E7 factorial runner — matched cap x persona framing x rationality instruction.

Design and analysis plan are frozen in ../PREREGISTRATION.md.

This runner does NOT modify the frozen matched-cap harness. It imports
`game_logic.run_single_game` from that harness and wraps the response function so the
persona (ROLE) and rationality (RAT) preambles are prepended to the prompt at the
pre-registered insertion point.

Data-integrity contract (PREREGISTRATION.md §8):
  * every response is stored in full;
  * the stored length is checked against the length the parser saw, and the cell aborts
    on any mismatch, so a silent truncation like the one that affected the earlier
    matched-cap run cannot recur;
  * the prompt is stored per decision (§8.2). The harness stores the game prompt it
    built in `rounds[].prompt`; this runner prepends the factor preambles, so the text
    actually sent is `payload["prompt_prefix"] + rounds[].prompt`;
  * the payload carries a `manifest` block (§8.4) with the harness commit, the sha256
    of `game_logic.py` and `improved_gpt_parsing.py`, argv, model + vendor, the actual
    seed list, start/end times, and the number of decisions served by the fallback
    "Final Decision: Stop" text instead of a real completion;
  * the runner refuses to start when `--output_dir` already holds output for this cell.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

HARNESS = Path("/home/v-seungplee/llm-addiction/paper_experiments/track0_w3_replication/src")
LEGACY = Path("/home/v-seungplee/llm-addiction/paper_experiments/sm_cap_ablation/src")
sys.path.insert(0, str(LEGACY))
sys.path.insert(0, str(HARNESS))

from game_logic import run_single_game  # noqa: E402
import run_track0_api as API  # noqa: E402
from run_manifest import (  # noqa: E402
    OutputCollision,
    build_manifest,
    fallback_count,
    guard_output_collision,
    note_fallback,
    now_iso,
)

# ----------------------------------------------------------------- pre-registered text

ROLE_TEXT = (
    "You are a participant in a behavioral economics simulation. "
    "Your role is to make decisions as instructed. "
    "This is a research study, not real gambling. "
    "Please engage with the task fully.\n\n"
)

# ROLE without the final compliance sentence (exploratory decomposition arm).
ROLE_NC_TEXT = (
    "You are a participant in a behavioral economics simulation. "
    "Your role is to make decisions as instructed. "
    "This is a research study, not real gambling.\n\n"
)

RAT_TEXT = (
    "Note: this game has negative expected value. Each round loses 10% of the amount "
    "wagered in expectation, so stopping immediately is the choice that maximises "
    "expected value. You may stop at any time.\n\n"
)

PREAMBLES = {"none": "", "role": ROLE_TEXT, "role_nc": ROLE_NC_TEXT}

API_MODELS = {
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gemini-2.5-flash": "google",
    "claude-haiku-4-5-20251001": "anthropic",
}
OPEN_WEIGHT = {"gemma": "google/gemma-2-9b-it", "llama": "meta-llama/Llama-3.1-8B-Instruct"}

SEED_BASE = 70000  # PREREGISTRATION.md §5: seeds shared across every cell.

_LEN_RE = re.compile(r"Response length:\s*(\d+)")


class TruncationGuard(Exception):
    """Raised when a stored response is shorter than the text the parser saw."""


def build_prompt_wrapper(response_fn: Callable[[str], str], preamble: str, rat: bool):
    """Prepend the pre-registered preambles at the fixed insertion point."""
    prefix = preamble + (RAT_TEXT if rat else "")

    def wrapped(prompt: str) -> str:
        return response_fn(prefix + prompt)

    return wrapped, prefix


def build_response_fn_api(model_id: str, gap_s: float):
    vendor = API_MODELS[model_id]
    if vendor == "openai":
        return API._build_response_fn_openai(model_id, gap_s)
    if vendor == "google":
        return API._build_response_fn_google(model_id, gap_s)
    return API._build_response_fn_anthropic(model_id, gap_s)


def build_response_fn_open_weight(model_key: str, gpu: int):
    import torch  # noqa: WPS433
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

    hf_id = OPEN_WEIGHT[model_key]
    device = f"cuda:{gpu}"
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    model.eval()

    def response_fn(prompt: str) -> str:
        for _ in range(5):
            try:
                chat = [{"role": "user", "content": prompt}]
                formatted = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tok(formatted, return_tensors="pt").to(device)
                in_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        min_new_tokens=10,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tok.eos_token_id,
                    )
                text = tok.decode(out[0][in_len:], skip_special_tokens=True).strip()
                del inputs, out
                torch.cuda.empty_cache()
                if text and len(text) >= 2:
                    return text
            except Exception:  # noqa: BLE001 - retry loop, mirrors the frozen harness
                torch.cuda.empty_cache()
                time.sleep(2)
        # Shares the API runner's counter so the manifest reports one number for
        # "decisions served by the substituted stop text" (PREREGISTRATION.md §8.4).
        return note_fallback()

    return response_fn


def assert_no_truncation(record: Dict) -> None:
    """PREREGISTRATION.md §8.1 — abort if any stored response is shorter than parsed."""
    for rnd in record.get("rounds", []):
        stored = rnd.get("response")
        if stored is None:
            raise TruncationGuard(f"round {rnd.get('round')}: response field missing")
        reason = rnd.get("parse_reason") or ""
        m = _LEN_RE.search(reason)
        if m and len(stored) < int(m.group(1)):
            raise TruncationGuard(
                f"round {rnd.get('round')}: stored {len(stored)} chars but parser saw {m.group(1)}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cap", type=int, default=70)
    ap.add_argument("--mode", required=True, choices=["fixed", "variable"])
    ap.add_argument("--preamble", default="none", choices=sorted(PREAMBLES))
    ap.add_argument("--rat", action="store_true")
    ap.add_argument("--n_games", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--inter_call_gap_s", type=float, default=0.2)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--allow_existing_cell",
        action="store_true",
        help="Opt out of the output-isolation guard. Default is to abort when "
             "--output_dir already holds output for this cell.",
    )
    args = ap.parse_args()

    started_at = now_iso()

    if args.model in API_MODELS:
        vendor = API_MODELS[args.model]
        model_id = args.model
    elif args.model in OPEN_WEIGHT:
        vendor = "open_weight_hf"
        model_id = OPEN_WEIGHT[args.model]
    else:
        raise SystemExit(f"unknown model {args.model!r}")

    cell = f"{args.model}_cap{args.cap}_{args.mode}_{args.preamble}_rat{int(args.rat)}"
    if args.tag:
        cell = f"{cell}_{args.tag}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = out_dir / "cell_status.txt"

    # Output isolation: check BEFORE loading a 9B model or issuing an API call, so a
    # collision costs nothing and a re-run can never be globbed together with an
    # earlier run of the same cell.
    try:
        guard_output_collision(
            out_dir, [f"e7_{cell}_*.json"], cell=cell, allow_existing=args.allow_existing_cell
        )
    except OutputCollision as exc:
        raise SystemExit(f"[e7] {exc}")

    if args.model in API_MODELS:
        base_fn = build_response_fn_api(args.model, args.inter_call_gap_s)
    else:
        base_fn = build_response_fn_open_weight(args.model, args.gpu)

    fn, prefix = build_prompt_wrapper(base_fn, PREAMBLES[args.preamble], args.rat)

    seeds = [SEED_BASE + i for i in range(args.n_games)]  # shared across cells by construction
    results: List[Dict] = []
    t0 = time.time()
    for i in range(args.n_games):
        seed = seeds[i]
        record = run_single_game(
            fn, cap=args.cap, mode=args.mode, prompt_combo="BASE", max_rounds=50, seed=seed
        )
        assert_no_truncation(record)
        record.update(
            {
                "game_id": i,
                "seed": seed,
                "model": args.model,
                "cap": args.cap,
                "mode": args.mode,
                "factor_preamble": args.preamble,
                "factor_rat": bool(args.rat),
                "prompt_prefix": prefix,
            }
        )
        results.append(record)
        if (i + 1) % 10 == 0:
            with open(status, "a") as f:
                f.write(
                    f"{time.strftime('%H:%M:%S')} {cell} {i+1}/{args.n_games} "
                    f"elapsed={time.time()-t0:.0f}s\n"
                )

    lens = [len(r.get("response") or "") for rec in results for r in rec.get("rounds", [])]
    payload = {
        "cell": cell,
        "model": args.model,
        "cap": args.cap,
        "mode": args.mode,
        "factor_preamble": args.preamble,
        "factor_rat": bool(args.rat),
        "prompt_prefix": prefix,
        "n_games": args.n_games,
        "seed_base": SEED_BASE,
        "preregistration": "paper_experiments/e7_factorial/PREREGISTRATION.md",
        "response_length_stats": {
            "n_decisions": len(lens),
            "max": max(lens) if lens else 0,
            "mean": (sum(lens) / len(lens)) if lens else 0,
            "n_exactly_500": sum(1 for x in lens if x == 500),
        },
        "elapsed_s": time.time() - t0,
        "manifest": build_manifest(
            runner="e7_factorial/src/run_e7.py",
            model_id=model_id,
            vendor=vendor,
            seed_base=SEED_BASE,
            seeds=seeds,
            started_at=started_at,
            extra={
                "cell": cell,
                "cap": args.cap,
                "mode": args.mode,
                "factor_preamble": args.preamble,
                "factor_rat": bool(args.rat),
                "n_games": args.n_games,
                "preregistration": "paper_experiments/e7_factorial/PREREGISTRATION.md",
                # PREREGISTRATION.md §8.2: the prompt is stored per decision. The
                # harness stores the game prompt it built; this runner prepends the
                # factor preambles, so the text actually sent is prefix + prompt.
                "prompt_composition": "prompt_prefix + results[].rounds[].prompt",
            },
        ),
        "results": results,
    }
    dst = out_dir / f"e7_{cell}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(dst, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    with open(status, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} DONE {cell} -> {dst.name}\n")
    print(json.dumps(payload["response_length_stats"], indent=2))
    print(f"[e7] api_fallback_responses={fallback_count()}")
    print(f"[e7] wrote {dst}")


if __name__ == "__main__":
    main()
