"""Track 0 launcher for open-weight (Gemma-2-9b, LLaMA-3.1-8b) matched-cap runs.

Reuses prompt + parser from `game_logic`. The runner here is responsible only for:
1. loading the HF model in bf16 (per CLAUDE.md),
2. wrapping `model.generate` into a `response_fn(prompt) -> str` callback that
   `game_logic.run_single_game` consumes,
3. saving JSON in the CLAUDE.md filename convention extended for cap/mode.

Pre-registered grid lives in `configs/track0_config.yaml`; CLI flags pick a single
cell (one model, one cap, one mode) for parallel scheduling on AMLT.
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

# Allow running as `python run_track0_open_weight.py ...` from the src/ dir.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Parity-critical: `game_logic` imports `improved_parse_gpt_response`, which lives in
# /home/v-seungplee/llm-addiction/legacy/. Insert the legacy directory BEFORE the
# `from game_logic import ...` line so the parser resolves to the legacy reference.
sys.path.insert(0, str(HERE.parent.parent / "sm_cap_ablation" / "src"))

from game_logic import run_single_game  # noqa: E402

OPEN_WEIGHT_HF_IDS = {
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def _load_cfg() -> dict:
    cfg_path = HERE.parent / "configs" / "track0_config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _build_response_fn(model, tokenizer, device: str) -> Callable[[str], str]:
    # Eager attention + bf16 keeps Gemma-2 sliding-window stable per existing SM runner.
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
                print("[track0/open_weight] CUDA OOM; clearing cache and retrying", file=sys.stderr)
                torch.cuda.empty_cache()
                time.sleep(1.0)
            except (RuntimeError, ValueError) as e:
                print(f"[track0/open_weight] generation retry ({type(e).__name__}): {e}", file=sys.stderr)
                torch.cuda.empty_cache()
                time.sleep(0.5)
        return "Final Decision: Stop"
    return response_fn


def _load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name not in OPEN_WEIGHT_HF_IDS:
        raise ValueError(f"unknown open-weight model: {model_name}")
    hf_id = OPEN_WEIGHT_HF_IDS[model_name]

    # Disable torch.compile for Gemma-2 sliding-window per existing SM runner.
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
    valid_caps = list(cfg["stage_1"]["caps"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(OPEN_WEIGHT_HF_IDS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cap", type=int, required=True, choices=valid_caps)
    parser.add_argument("--mode", required=True, choices=["fixed", "variable"])
    parser.add_argument("--n_games", type=int, default=None,
                        help="Override config n_games (Track 0 default = config.stage_1.n_games_per_cell)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Force n_games=5; CI-only.")
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    gen = cfg["generation"]
    # Per-mode max_rounds: legacy fixed-bet runner uses 100, variable runner uses 50.
    # Ref: legacy/gpt_fixed_bet_size_experiment/src/gpt_fixed_bet_size_experiment.py:120
    #      legacy/gpt_variable_max_bet_experiment/src/gpt_variable_max_bet_experiment.py:120
    max_rounds_fixed = int(cfg["generation"]["max_rounds_fixed"])
    max_rounds_variable = int(cfg["generation"]["max_rounds_variable"])
    n_games = 5 if args.smoke else (args.n_games or cfg["stage_1"]["n_games_per_cell"])
    out_dir = Path(args.output_dir or cfg["output"]["base_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_base = gen["seed_base"]
    random.seed(seed_base)

    print(f"[track0/open_weight] model={args.model} cap={args.cap} mode={args.mode} n_games={n_games}")
    model, tokenizer = _load_model(args.model)
    device = "cuda:0"
    response_fn = _build_response_fn(model, tokenizer, device)

    results = []
    for i in tqdm(range(n_games), desc=f"{args.model}/cap{args.cap}/{args.mode}"):
        game_seed = seed_base + i
        record = run_single_game(
            response_fn=response_fn,
            cap=args.cap,
            mode=args.mode,
            initial_balance=gen["initial_balance"],
            win_rate=gen["win_rate"],
            payout=gen["payout"],
            max_rounds=max_rounds_fixed if args.mode == "fixed" else max_rounds_variable,
            seed=game_seed,
        )
        record["game_id"] = i
        record["model"] = args.model
        record["seed"] = game_seed
        results.append(record)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = cfg["output"]["filename_pattern"].format(
        model=args.model, cap=args.cap, mode=args.mode, timestamp=timestamp
    )
    payload = {
        "track": "0_w3_replication",
        "model": args.model,
        "cap": args.cap,
        "mode": args.mode,
        "n_games": n_games,
        "smoke": args.smoke,
        "config_snapshot": {"generation": gen, "stage_1_n_games_per_cell": cfg["stage_1"]["n_games_per_cell"]},
        "timestamp": timestamp,
        "results": results,
    }
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[track0/open_weight] wrote {out_path}")


if __name__ == "__main__":
    main()
