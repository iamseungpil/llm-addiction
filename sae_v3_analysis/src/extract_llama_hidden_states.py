#!/usr/bin/env python3
"""
Extract LLaMA hidden states at decision points ONLY.
For IC and SM paradigms, saves hidden states at key layers.
~4800 forward passes total → ~10 min on A100.

Output:
  IC: {output_dir}/ic_hidden_states_dp.npz
  SM: {output_dir}/sm_hidden_states_dp.npz
Each contains: hidden_states (n_games, n_layers, 4096), metadata arrays
"""

import os, sys, json, gc, math, torch, numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from prompt_reconstruction import reconstruct_sm_prompt

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HIDDEN_DIM = 4096
LAYERS = [8, 12, 22, 25, 30]  # Key layers for Analysis 2, 3

IC_DATA = Path("/home/v-seungplee/data/llm-addiction/behavioral/investment_choice/v2_role_llama")
SM_DATA = Path("/home/v-seungplee/data/llm-addiction/behavioral/slot_machine/llama_v4_role")
OUTPUT_DIR = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ic_decision_points():
    """Load IC decision-point prompts (last round of each game)."""
    records = []
    files = sorted(IC_DATA.glob("llama_investment_*.json"))
    game_id = 0
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        for game in data["results"]:
            game_id += 1
            decs = game.get("decisions", [])
            valid = [d for d in decs if not d.get("skipped", False)]
            if not valid:
                continue
            last = valid[-1]
            records.append({
                "game_id": game_id,
                "prompt": last["full_prompt"],
                "game_outcome": "bankruptcy" if game.get("bankruptcy") else
                                "voluntary_stop" if game.get("stopped_voluntarily") else "max_rounds",
                "bet_type": game.get("bet_type", "fixed"),
                "balance": last.get("balance_before", 100),
                "prompt_condition": game.get("prompt_condition", "BASE"),
                "round_num": last.get("round", 1),
            })
    log(f"IC: {len(records)} decision points from {game_id} games")
    log(f"  BK: {sum(1 for r in records if r['game_outcome']=='bankruptcy')}")
    return records


def load_sm_decision_points():
    """Load SM decision-point prompts (last valid decision of each game)."""
    records = []
    files = sorted(SM_DATA.glob("final_*.json"))
    game_id = 0
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        for game in data["results"]:
            game_id += 1
            decs = game.get("decisions", [])
            history = game.get("history", [])
            bet_type = game.get("bet_type", "fixed")
            combo = game.get("prompt_combo", "BASE")
            outcome = game.get("outcome", "")

            valid = [d for d in decs if d.get("action") != "skip"]
            if not valid:
                continue
            last = valid[-1]
            last_idx = len(valid) - 1

            # Count bet actions before last for history slice
            hist_idx = sum(1 for d in valid[:last_idx] if d.get("action") == "bet")
            history_slice = history[:hist_idx]

            prompt = reconstruct_sm_prompt(
                prompt_combo=combo,
                bet_type=bet_type,
                balance=last.get("balance_before", 100),
                history_slice=history_slice,
            )
            records.append({
                "game_id": game_id,
                "prompt": prompt,
                "game_outcome": outcome,
                "bet_type": bet_type,
                "balance": last.get("balance_before", 100),
                "prompt_condition": combo,
                "round_num": last.get("round", 1),
            })
    log(f"SM: {len(records)} decision points from {game_id} games")
    log(f"  BK: {sum(1 for r in records if r['game_outcome']=='bankruptcy')}")
    return records


def extract_hidden_states(records, model, tokenizer, device):
    """Run forward pass and extract hidden states at decision points."""
    N = len(records)
    n_layers = len(LAYERS)
    hidden = np.zeros((N, n_layers, HIDDEN_DIM), dtype=np.float32)
    valid = np.ones(N, dtype=bool)

    for i in tqdm(range(N), desc="Forward passes"):
        try:
            chat = [{"role": "user", "content": records[i]["prompt"]}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)

            with torch.no_grad():
                outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)

            for j, layer in enumerate(LAYERS):
                h = outputs.hidden_states[layer + 1][:, -1, :]
                hidden[i, j, :] = h.float().cpu().numpy().squeeze()

        except Exception as e:
            log(f"  Error at {i}: {e}")
            valid[i] = False

    return hidden, valid


def save_result(paradigm, records, hidden, valid):
    out_dir = OUTPUT_DIR / paradigm / "llama"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hidden_states_dp.npz"

    # Build metadata arrays
    game_ids = np.array([r["game_id"] for r in records])
    game_outcomes = np.array([r["game_outcome"] for r in records])
    bet_types = np.array([r["bet_type"] for r in records])
    balances = np.array([r["balance"] for r in records], dtype=np.float32)
    prompt_conditions = np.array([r["prompt_condition"] for r in records])
    round_nums = np.array([r["round_num"] for r in records])

    np.savez(out_file,
             hidden_states=hidden,
             valid_mask=valid,
             layers=np.array(LAYERS),
             game_ids=game_ids,
             game_outcomes=game_outcomes,
             bet_types=bet_types,
             balances=balances,
             prompt_conditions=prompt_conditions,
             round_nums=round_nums)
    log(f"Saved {out_file} ({hidden.shape})")


def main():
    log("=" * 70)
    log("LLaMA Hidden State Extraction (Decision Points Only)")
    log(f"Layers: {LAYERS}")
    log("=" * 70)

    device = "cuda:0"

    # Load data
    ic_records = load_ic_decision_points()
    sm_records = load_sm_decision_points()

    # Load model once
    log(f"\nLoading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    log("Model loaded")

    # Extract IC
    log("\n--- IC Hidden States ---")
    ic_hidden, ic_valid = extract_hidden_states(ic_records, model, tokenizer, device)
    save_result("investment_choice", ic_records, ic_hidden, ic_valid)
    log(f"IC: {ic_valid.sum()}/{len(ic_valid)} valid")

    # Extract SM
    log("\n--- SM Hidden States ---")
    sm_hidden, sm_valid = extract_hidden_states(sm_records, model, tokenizer, device)
    save_result("slot_machine", sm_records, sm_hidden, sm_valid)
    log(f"SM: {sm_valid.sum()}/{len(sm_valid)} valid")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    log("\n=== Done ===")
    log(f"IC: {ic_hidden.shape}, SM: {sm_hidden.shape}")


if __name__ == "__main__":
    main()
