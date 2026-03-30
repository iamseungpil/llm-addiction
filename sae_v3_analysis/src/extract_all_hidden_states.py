#!/usr/bin/env python3
"""
Extract ALL missing hidden states for V12 steering experiments.
Supports LLaMA (MW) and Gemma (SM/IC/MW) — all 4 missing combos.

Usage:
  python extract_all_hidden_states.py --model gemma   # Gemma SM+IC+MW
  python extract_all_hidden_states.py --model llama   # LLaMA MW only
  python extract_all_hidden_states.py --model all     # Everything missing
"""
import os, sys, json, gc, torch, numpy as np, argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from prompt_reconstruction import reconstruct_sm_prompt

LAYERS = [8, 12, 22, 25, 30]
OUTPUT_DIR = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
BEHAVIORAL = Path("/home/v-seungplee/data/llm-addiction/behavioral")

MODELS = {
    "llama": {"name": "meta-llama/Llama-3.1-8B-Instruct", "hidden_dim": 4096,
              "sm_data": BEHAVIORAL / "slot_machine/llama_v4_role",
              "ic_data": BEHAVIORAL / "investment_choice/v2_role_llama",
              "mw_data": BEHAVIORAL / "mystery_wheel/llama_v2_role"},
    "gemma": {"name": "google/gemma-2-9b-it", "hidden_dim": 3584,
              "sm_data": BEHAVIORAL / "slot_machine/gemma_v4_role",
              "ic_data": BEHAVIORAL / "investment_choice/v2_role_gemma",
              "mw_data": BEHAVIORAL / "mystery_wheel/gemma_v2_role"},
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# Data Loaders (extract last decision point per game)
# ============================================================

def load_sm_decision_points(data_dir):
    """Load SM last decision-point prompts."""
    records = []
    files = sorted(data_dir.glob("final_*.json")) + sorted(data_dir.glob("*slot*.json"))
    game_id = 0
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        results = data.get("results", data.get("games", []))
        for game in results:
            game_id += 1
            decs = game.get("decisions", [])
            valid = [d for d in decs if d.get("action") != "skip" and not d.get("skipped", False)]
            if not valid:
                continue
            last = valid[-1]
            # Use full_prompt if available, else reconstruct
            prompt = last.get("full_prompt")
            if not prompt:
                history = game.get("history", [])
                last_idx = len(valid) - 1
                hist_idx = sum(1 for d in valid[:last_idx] if d.get("action") == "bet")
                prompt = reconstruct_sm_prompt(
                    prompt_combo=game.get("prompt_combo", game.get("prompt_condition", "BASE")),
                    bet_type=game.get("bet_type", "fixed"),
                    balance=last.get("balance_before", 100),
                    history_slice=history[:hist_idx],
                )
            outcome = game.get("outcome", "")
            if not outcome:
                outcome = "bankruptcy" if game.get("bankruptcy") else \
                          "voluntary_stop" if game.get("stopped_voluntarily") else "max_rounds"
            records.append({
                "game_id": game_id, "prompt": prompt, "game_outcome": outcome,
                "bet_type": game.get("bet_type", "fixed"),
                "balance": last.get("balance_before", 100),
                "prompt_condition": game.get("prompt_combo", game.get("prompt_condition", "BASE")),
                "round_num": last.get("round", 1),
            })
    log(f"SM: {len(records)} decision points, BK={sum(1 for r in records if r['game_outcome']=='bankruptcy')}")
    return records


def load_ic_decision_points(data_dir):
    """Load IC last decision-point prompts."""
    records = []
    files = sorted(data_dir.glob("*investment*.json")) + sorted(data_dir.glob("*ic*.json"))
    game_id = 0
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        results = data.get("results", data.get("games", []))
        for game in results:
            game_id += 1
            decs = game.get("decisions", [])
            valid = [d for d in decs if not d.get("skipped", False)]
            if not valid:
                continue
            last = valid[-1]
            prompt = last.get("full_prompt", "")
            if not prompt:
                continue
            outcome = "bankruptcy" if game.get("bankruptcy") else \
                      "voluntary_stop" if game.get("stopped_voluntarily") else "max_rounds"
            records.append({
                "game_id": game_id, "prompt": prompt, "game_outcome": outcome,
                "bet_type": game.get("bet_type", "fixed"),
                "balance": last.get("balance_before", 100),
                "prompt_condition": game.get("prompt_condition", "BASE"),
                "round_num": last.get("round", 1),
            })
    log(f"IC: {len(records)} decision points, BK={sum(1 for r in records if r['game_outcome']=='bankruptcy')}")
    return records


def load_mw_decision_points(data_dir):
    """Load MW last decision-point prompts."""
    records = []
    files = sorted(data_dir.glob("*mystery*.json")) + sorted(data_dir.glob("*mw*.json"))
    game_id = 0
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        results = data.get("results", data.get("games", []))
        for game in results:
            game_id += 1
            decs = game.get("decisions", [])
            valid = [d for d in decs if not d.get("skipped", False)]
            if not valid:
                continue
            last = valid[-1]
            prompt = last.get("full_prompt", "")
            if not prompt:
                continue
            outcome = "bankruptcy" if game.get("bankruptcy") else \
                      "voluntary_stop" if game.get("stopped_voluntarily") else "max_rounds"
            records.append({
                "game_id": game_id, "prompt": prompt, "game_outcome": outcome,
                "bet_type": game.get("bet_type", "fixed"),
                "balance": last.get("balance_before", 100),
                "prompt_condition": game.get("prompt_condition", "BASE"),
                "round_num": last.get("round", 1),
            })
    log(f"MW: {len(records)} decision points, BK={sum(1 for r in records if r['game_outcome']=='bankruptcy')}")
    return records


# ============================================================
# Extraction Engine
# ============================================================

def extract_hidden_states(records, model, tokenizer, device, layers, hidden_dim):
    """Forward pass to extract hidden states at decision points."""
    N = len(records)
    n_layers = len(layers)
    hidden = np.zeros((N, n_layers, hidden_dim), dtype=np.float32)
    valid = np.ones(N, dtype=bool)

    for i in tqdm(range(N), desc="Forward passes"):
        try:
            chat = [{"role": "user", "content": records[i]["prompt"]}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)

            with torch.no_grad():
                outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)

            for j, layer in enumerate(layers):
                h = outputs.hidden_states[layer + 1][:, -1, :]
                hidden[i, j, :] = h.float().cpu().numpy().squeeze()

        except Exception as e:
            log(f"  Error at {i}: {e}")
            valid[i] = False

    return hidden, valid


def save_result(model_name, paradigm, records, hidden, valid, layers):
    out_dir = OUTPUT_DIR / paradigm / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hidden_states_dp.npz"

    np.savez(out_file,
             hidden_states=hidden,
             valid_mask=valid,
             layers=np.array(layers),
             game_ids=np.array([r["game_id"] for r in records]),
             game_outcomes=np.array([r["game_outcome"] for r in records]),
             bet_types=np.array([r["bet_type"] for r in records]),
             balances=np.array([r["balance"] for r in records], dtype=np.float32),
             prompt_conditions=np.array([r["prompt_condition"] for r in records]),
             round_nums=np.array([r["round_num"] for r in records]))
    log(f"Saved {out_file} ({hidden.shape})")
    n_bk = sum(1 for r in records if r["game_outcome"] == "bankruptcy")
    n_valid = valid.sum()
    log(f"  Valid: {n_valid}/{len(records)}, BK: {n_bk}")
    return str(out_file)


# ============================================================
# Main
# ============================================================

def run_extraction(model_key, device="cuda:0"):
    cfg = MODELS[model_key]
    hidden_dim = cfg["hidden_dim"]
    layers = LAYERS if model_key == "llama" else [8, 12, 22, 25, 30]  # same layers for now

    # Check what's missing
    tasks = []
    for paradigm, loader, data_dir in [
        ("slot_machine", load_sm_decision_points, cfg["sm_data"]),
        ("investment_choice", load_ic_decision_points, cfg["ic_data"]),
        ("mystery_wheel", load_mw_decision_points, cfg["mw_data"]),
    ]:
        out_file = OUTPUT_DIR / paradigm / model_key / "hidden_states_dp.npz"
        if out_file.exists():
            log(f"SKIP {model_key} {paradigm}: already exists at {out_file}")
        else:
            tasks.append((paradigm, loader, data_dir))

    if not tasks:
        log(f"Nothing to extract for {model_key}")
        return

    # Load model
    log(f"\nLoading {cfg['name']}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"], torch_dtype=torch.bfloat16, device_map=device,
        low_cpu_mem_usage=True, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    log(f"Model loaded ({hidden_dim}-dim)")

    # Adjust layers for Gemma (42 layers vs LLaMA 32)
    n_model_layers = model.config.num_hidden_layers
    valid_layers = [l for l in layers if l < n_model_layers]
    if len(valid_layers) < len(layers):
        log(f"  Adjusted layers {layers} → {valid_layers} (model has {n_model_layers} layers)")
    layers = valid_layers

    # Extract each missing paradigm
    results = {}
    for paradigm, loader, data_dir in tasks:
        log(f"\n{'='*60}")
        log(f"Extracting: {model_key} {paradigm}")
        log(f"{'='*60}")

        records = loader(data_dir)
        if len(records) == 0:
            log(f"  WARNING: No records found in {data_dir}")
            continue

        hidden, valid = extract_hidden_states(records, model, tokenizer, device, layers, hidden_dim)
        out_path = save_result(model_key, paradigm, records, hidden, valid, layers)
        results[paradigm] = out_path

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llama", "gemma", "all"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    log("=" * 70)
    log("Hidden State Extraction for V12 Steering")
    log(f"Model: {args.model}, Device: {args.device}")
    log("=" * 70)

    if args.model == "all":
        for m in ["gemma", "llama"]:
            run_extraction(m, args.device)
    else:
        run_extraction(args.model, args.device)

    log("\nDONE. All missing hidden states extracted.")

    # Verification: check all 6 combos
    log("\n=== VERIFICATION ===")
    for paradigm in ["slot_machine", "investment_choice", "mystery_wheel"]:
        for model_key in ["llama", "gemma"]:
            f = OUTPUT_DIR / paradigm / model_key / "hidden_states_dp.npz"
            if f.exists():
                d = np.load(f, allow_pickle=True)
                n_bk = sum(d["game_outcomes"] == "bankruptcy")
                log(f"  ✅ {model_key} {paradigm}: {d['hidden_states'].shape}, BK={n_bk}")
            else:
                log(f"  ❌ {model_key} {paradigm}: MISSING")


if __name__ == "__main__":
    main()
