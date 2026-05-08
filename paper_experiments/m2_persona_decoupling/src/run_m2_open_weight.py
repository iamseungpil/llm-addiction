"""Track A1 / M2 launcher for open-weight (Gemma-2-9b, LLaMA-3.1-8b) framing runs.

Reuses the canonical track0_w3_replication.src.game_logic.{run_single_game,
SlotMachineGame, create_prompt, parse_response} — does NOT reimplement game logic.
The framing prefix (first_person vs role_play_gambler) is injected at the
response_fn boundary so the slot-machine state machine, parser, and round loop
remain identical to Track 0; only the preamble swaps.

Pre-registered grid lives in `configs/m2_config.yaml`; CLI flags pick a single
cell (model × condition × framing × task) for parallel scheduling on AMLT.
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
from typing import Callable

import torch
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse canonical track0 game logic.
# Append (not insert) so M2's own modules win name collisions with track0 (sanity_checks).
TRACK0_SRC = HERE.parents[1] / "track0_w3_replication" / "src"
if str(TRACK0_SRC) not in sys.path:
    sys.path.append(str(TRACK0_SRC))

from game_logic import run_single_game  # noqa: E402
from prompts import build_prompt  # noqa: E402
import prompts as prompts_mod  # noqa: E402


OPEN_WEIGHT_HF_IDS = {
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}

CONDITIONS = ("BASE", "+G", "+M", "+GM")
FRAMINGS = ("first_person", "role_play_gambler")
TASKS = ("SM", "IC", "MW")


def _load_cfg() -> dict:
    cfg_path = HERE.parent / "configs" / "m2_config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _build_response_fn(
    model,
    tokenizer,
    device: str,
    condition: str,
    framing: str,
    seed_slot: dict,
) -> Callable[[str], str]:
    # Wraps `model.generate` and rewrites the prompt to inject the M2 framing
    # prefix. The track0 create_prompt path emitted include_role_instruction=False
    # in run_single_game (see _make_run_single_game_caller) so the prefix here owns
    # the preamble exclusively — no duplicate ROLE_INSTRUCTION under first_person.
    #
    # Determinism: model.generate uses the torch global RNG; seeding only the
    # game RNG (via run_single_game) is not enough. The outer loop writes the
    # current game's seed into `seed_slot["seed"]` before invoking
    # response_fn, and we re-seed torch + cuda here on every call so identical
    # game_seed values produce identical Gemma rationales across runs.
    def response_fn(prompt_body: str) -> str:
        prompt = _wrap_with_framing(prompt_body, condition, framing)
        game_seed_for_this_round = int(seed_slot["seed"])
        torch.manual_seed(game_seed_for_this_round)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(game_seed_for_this_round)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                chat = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(formatted, return_tensors="pt").to(device)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        min_new_tokens=10,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = out[0][input_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                del inputs, out
                torch.cuda.empty_cache()
                if text and len(text) >= 2:
                    return text
            except torch.cuda.OutOfMemoryError:
                print("[m2/open_weight] CUDA OOM; clearing cache and retrying", file=sys.stderr)
                torch.cuda.empty_cache()
                time.sleep(1.0)
            except (RuntimeError, ValueError) as e:
                print(f"[m2/open_weight] generation retry ({type(e).__name__}): {e}", file=sys.stderr)
                torch.cuda.empty_cache()
                time.sleep(0.5)
        return "Final Decision: Stop"
    return response_fn


def _wrap_with_framing(prompt_body: str, condition: str, framing: str) -> str:
    if framing == "first_person":
        prefix = prompts_mod.first_person_prefix(condition)
    elif framing == "role_play_gambler":
        prefix = prompts_mod.role_play_gambler_prefix(condition)
    else:
        raise ValueError(f"unknown framing {framing}")
    return prefix + prompt_body


def _condition_to_combo(condition: str) -> str:
    if condition == "BASE":
        return "BASE"
    if condition == "+G":
        return "G"
    if condition == "+M":
        return "M"
    if condition == "+GM":
        return "GM"
    raise ValueError(f"unknown condition {condition}")


def _load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name not in OPEN_WEIGHT_HF_IDS:
        raise ValueError(f"unknown open-weight model: {model_name}")
    hf_id = OPEN_WEIGHT_HF_IDS[model_name]

    os.environ["TORCH_COMPILE"] = "0"

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tok


def main() -> None:
    cfg = _load_cfg()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(OPEN_WEIGHT_HF_IDS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
    parser.add_argument("--framing", required=True, choices=list(FRAMINGS))
    parser.add_argument("--task", default="SM", choices=list(TASKS))
    parser.add_argument("--n_games", type=int, default=None,
                        help="Override config n_games (M2 default = config.stage_1.n_games_per_cell)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Force n_games=5; CI-only.")
    args = parser.parse_args()

    # IC and MW are documented in the choices list (Plan §2.5 robustness grid)
    # but are not yet wired into the runner — the runner currently launches SM
    # only. Silently relabelling SM play as IC/MW would corrupt analysis, so
    # we hard-fail at argparse time instead.
    if args.task in ("IC", "MW"):
        raise NotImplementedError(
            f"Task {args.task} not yet wired (Plan §2.5 fallback). Stage-1 launches SM only."
        )

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    gen = cfg["generation"]
    n_games = 5 if args.smoke else (args.n_games or cfg["stage_1"]["n_games_per_cell"])
    out_dir = Path(args.output_dir or cfg["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_base = gen["seed_base"]
    random.seed(seed_base)

    print(f"[m2/open_weight] model={args.model} cond={args.condition} framing={args.framing} task={args.task} n_games={n_games}")
    model, tokenizer = _load_model(args.model)
    device = "cuda:0"
    # Mutable per-game seed slot — written by the outer loop, read inside
    # response_fn before each model.generate so torch's global RNG is
    # deterministic per game_seed (see C4 fix in module docstring).
    seed_slot: dict = {"seed": seed_base}
    response_fn = _build_response_fn(model, tokenizer, device, args.condition, args.framing, seed_slot)

    prompt_combo = _condition_to_combo(args.condition)
    results = []
    for i in tqdm(range(n_games), desc=f"{args.model}/{args.condition}/{args.framing}/{args.task}"):
        game_seed = seed_base + i
        seed_slot["seed"] = game_seed
        record = run_single_game(
            response_fn=response_fn,
            cap=None,                       # M2 uses legacy SM (no matched-cap manipulation)
            mode="variable",                # SM legacy default; framing × condition is the manipulation
            initial_balance=gen["initial_balance"],
            win_rate=gen["win_rate"],
            payout=gen["payout"],
            max_rounds=gen["max_rounds"],
            prompt_combo=prompt_combo,
            include_role_instruction=False,  # framing prefix is injected by response_fn wrapper
            seed=game_seed,
        )
        record["game_id"] = i
        record["model"] = args.model
        record["condition"] = args.condition
        record["framing"] = args.framing
        record["task"] = args.task
        record["seed"] = game_seed
        results.append(record)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = cfg["output"]["filename_pattern"].format(
        model=args.model,
        condition=args.condition.replace("+", "p"),  # filename-safe
        framing=args.framing,
        task=args.task,
        timestamp=timestamp,
    )
    payload = {
        "track": "A1_m2_persona_decoupling",
        "model": args.model,
        "condition": args.condition,
        "framing": args.framing,
        "task": args.task,
        "n_games": n_games,
        "smoke": args.smoke,
        "config_snapshot": {"generation": gen, "stage_1_n_games_per_cell": cfg["stage_1"]["n_games_per_cell"]},
        "timestamp": timestamp,
        "results": results,
    }
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[m2/open_weight] wrote {out_path}")


if __name__ == "__main__":
    main()
