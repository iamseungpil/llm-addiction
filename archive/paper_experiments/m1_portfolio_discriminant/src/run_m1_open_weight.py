"""M1 launcher for open-weight (Gemma-2-9b, LLaMA-3.1-8b) portfolio runs.

Mirrors `track0_w3_replication/src/run_track0_open_weight.py` exactly in shape:
1. load HF model in bf16 + eager attention,
2. wrap `model.generate` into `response_fn(prompt) -> str`,
3. drive `portfolio_simulator.run_single_game` with our prompt builder + parser,
4. save JSON in the M1 filename convention.

CLI flags pick a single (model, condition, objective, blurb_variant) cell so the
6-models × 5-conditions × 2-objectives × 3-blurbs grid (180 cells) can be parallelised
across nodes via AMLT. With n=200 per cell the grid is 36k generations; the cell
naming in `m1_config.yaml` filename_pattern keeps each shard isolated.
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

from portfolio_simulator import assets_from_config, run_single_game  # noqa: E402
from prompts import build_portfolio_prompt  # noqa: E402
from parse_allocation import FALLBACK_API_FAILURE_SENTINEL, parse_allocation  # noqa: E402

OPEN_WEIGHT_HF_IDS = {
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def _load_cfg() -> dict:
    cfg_path = HERE.parent / "configs" / "m1_config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _build_response_fn(model, tokenizer, device: str) -> Callable[[str], str]:
    # Eager attention + bf16 keeps Gemma-2 sliding-window stable; matches Track 0 runner.
    def response_fn(prompt: str) -> str:
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
                print("[m1/open_weight] CUDA OOM; clearing cache and retrying", file=sys.stderr)
                torch.cuda.empty_cache()
                time.sleep(1.0)
            except (RuntimeError, ValueError) as e:
                print(f"[m1/open_weight] generation retry ({type(e).__name__}): {e}", file=sys.stderr)
                torch.cuda.empty_cache()
                time.sleep(0.5)
        # C6 fix: emit a sentinel string the parser detects and rejects, instead of
        # synthesising a 100%-cash allocation that the simulator would treat as a
        # deliberate conservative choice. The simulator skips the round (and
        # increments `consecutive_skips`) when the parser returns None, which makes
        # the GPU-failure case observable in the per-game `fallback_count` field
        # rather than silently biasing per-model risk_event downward.
        return FALLBACK_API_FAILURE_SENTINEL
    return response_fn


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
    valid_conditions = list(cfg["stage_1"]["conditions"])
    valid_objectives = list(cfg["stage_1"]["objectives"])
    valid_blurbs = list(cfg["stage_1"]["blurb_variants"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(OPEN_WEIGHT_HF_IDS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--condition", required=True, choices=valid_conditions)
    parser.add_argument("--objective", required=True, choices=valid_objectives)
    parser.add_argument("--blurb_variant", required=True, choices=valid_blurbs)
    parser.add_argument("--n_games", type=int, default=None,
                        help="Override config n_games (default = config.stage_1.n_games_per_cell)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Force n_games=5; CI-only.")
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    sim_cfg = cfg["portfolio_simulator"]
    assets = assets_from_config(sim_cfg["assets"])
    n_games = 5 if args.smoke else (args.n_games or cfg["stage_1"]["n_games_per_cell"])
    out_dir = Path(args.output_dir or cfg["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_base = 42
    random.seed(seed_base)

    print(
        f"[m1/open_weight] model={args.model} condition={args.condition} "
        f"objective={args.objective} blurb={args.blurb_variant} n_games={n_games}"
    )
    model, tokenizer = _load_model(args.model)
    device = "cuda:0"
    response_fn = _build_response_fn(model, tokenizer, device)

    results = []
    desc = f"{args.model}/{args.condition}/{args.objective}/{args.blurb_variant}"
    for i in tqdm(range(n_games), desc=desc):
        game_seed = seed_base + i
        record = run_single_game(
            response_fn=response_fn,
            assets=assets,
            n_rounds=sim_cfg["n_rounds"],
            initial_budget=sim_cfg["initial_budget"],
            round_fraction_of_year=sim_cfg["round_fraction_of_year"],
            seed=game_seed,
            prompt_builder=build_portfolio_prompt,
            parse_fn=parse_allocation,
            condition=args.condition,
            objective=args.objective,
            blurb_variant=args.blurb_variant,
        )
        record["game_id"] = i
        record["model"] = args.model
        record["seed"] = game_seed
        results.append(record)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = cfg["output"]["filename_pattern"].format(
        model=args.model,
        condition=args.condition.replace("+", "plus"),
        objective=args.objective,
        blurb=args.blurb_variant,
        timestamp=timestamp,
    )
    payload = {
        "track": "B_M1_portfolio_discriminant",
        "domain": "portfolio",
        "model": args.model,
        "condition": args.condition,
        "objective": args.objective,
        "blurb_variant": args.blurb_variant,
        "n_games": n_games,
        "smoke": args.smoke,
        "config_snapshot": {
            "portfolio_simulator": sim_cfg,
            "stage_1_n_games_per_cell": cfg["stage_1"]["n_games_per_cell"],
        },
        "timestamp": timestamp,
        "results": results,
    }
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[m1/open_weight] wrote {out_path}")


if __name__ == "__main__":
    main()
