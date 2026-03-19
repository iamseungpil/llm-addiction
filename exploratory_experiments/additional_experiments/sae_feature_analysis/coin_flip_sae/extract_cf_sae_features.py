#!/usr/bin/env python3
"""
Coin Flip SAE Feature Extraction (Gemma only)
==============================================

Extracts SAE features from Gemma coin flip experiment data.
Matches V2 extraction pipeline (chat_template, sae_lens Gemma SAE).

Data: 950 games (c10 variable, 19/32 conditions, Gemma-2-9b-it)
Key: 0% bankruptcy, 100% voluntary stop (transparent probability → rational behavior)

Usage:
    python extract_cf_sae_features.py --device cuda:0
"""

import argparse
import json
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# Config
# ===========================================================================
DATA_FILE = Path("/home/jovyan/beomi/llm-addiction-data/coin_flip/gemma_coinflip_checkpoint_950.json")
OUTPUT_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v2/coin_flip/gemma")
MODEL_ID = "google/gemma-2-9b-it"
LAYERS = list(range(42))  # 0-41


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ===========================================================================
# Data Loading
# ===========================================================================
def load_games() -> List[Dict]:
    """Load coin flip checkpoint data."""
    logger.info(f"Loading {DATA_FILE}")
    with open(DATA_FILE) as f:
        data = json.load(f)

    logger.info(f"Completed: {data['completed']}/{data['total']}")
    results = data["results"]

    games = []
    for g in results:
        decisions = g.get("decisions", [])
        if not decisions:
            continue

        last_dec = decisions[-1]
        full_prompt = last_dec.get("full_prompt", "")
        if not full_prompt:
            continue

        # Determine outcome
        if g.get("bankruptcy", False):
            outcome = "bankruptcy"
        elif g.get("stopped_voluntarily", False):
            outcome = "voluntary_stop"
        elif g.get("max_rounds_reached", False):
            outcome = "max_rounds"
        else:
            outcome = "voluntary_stop"  # default for coin flip (all voluntary)

        games.append({
            "prompt": full_prompt,
            "outcome": outcome,
            "bet_type": g.get("bet_type", "variable"),
            "bet_constraint": str(g.get("bet_constraint", "10")),
            "prompt_condition": g.get("prompt_condition", "unknown"),
            "final_balance": g.get("final_balance", 100),
            "rounds_completed": g.get("rounds_completed", len(decisions)),
            "game_id": g.get("game_id", len(games)),
        })

    logger.info(f"Loaded {len(games)} games")
    return games


# ===========================================================================
# Phase A: Hidden State Extraction
# ===========================================================================
def extract_hidden_states(games: List[Dict], device: str) -> np.ndarray:
    """Extract last-token hidden states from all layers."""
    n_layers = len(LAYERS)
    n_games = len(games)

    logger.info(f"=== Phase A: Hidden State Extraction (GEMMA) ===")
    logger.info(f"Model: {MODEL_ID}, Games: {n_games}, Layers: {n_layers}")

    gpu_id = int(device.split(":")[1]) if ":" in device else 0
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
        device_map={"": gpu_id}, low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    d_model = model.config.hidden_size
    logger.info(f"d_model={d_model}, allocating ({n_games}, {n_layers}, {d_model}) float32")
    hidden_all = np.zeros((n_games, n_layers, d_model), dtype=np.float32)

    for i in tqdm(range(n_games), desc="Forward passes"):
        prompt = games[i]["prompt"]

        # Apply chat_template for Gemma (instruction-tuned)
        chat = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)

        for j, layer in enumerate(LAYERS):
            h = outputs.hidden_states[layer + 1][0, -1, :]
            hidden_all[i, j, :] = h.float().cpu().numpy()

    del model, tokenizer
    clear_gpu()
    logger.info(f"Phase A complete: {hidden_all.shape}")
    return hidden_all


# ===========================================================================
# Phase B: SAE Encoding + Save
# ===========================================================================
def encode_and_save(hidden_all: np.ndarray, games: List[Dict], device: str):
    """Phase B: Encode with GemmaScope SAE and save NPZ."""
    n_games = len(games)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Metadata arrays
    outcomes = np.array([g["outcome"] for g in games])
    game_ids = np.array([g["game_id"] for g in games], dtype=np.int64)
    bet_types = np.array([g["bet_type"] for g in games])
    bet_constraints = np.array([g["bet_constraint"] for g in games])
    prompt_conditions = np.array([g["prompt_condition"] for g in games])
    final_balances = np.array([g["final_balance"] for g in games], dtype=np.float64)
    rounds_completed = np.array([g["rounds_completed"] for g in games], dtype=np.int64)

    logger.info(f"=== Phase B: SAE Encoding (GEMMA, GemmaScope) ===")
    logger.info(f"Output: {OUTPUT_DIR}")

    from sae_lens import SAE

    for j, layer in enumerate(LAYERS):
        out_file = OUTPUT_DIR / f"layer_{layer}_features.npz"
        if out_file.exists():
            logger.info(f"Layer {layer}: exists, skipping")
            continue

        h_layer = hidden_all[:, j, :]
        h_tensor = torch.tensor(h_layer, dtype=torch.float32, device=device)

        try:
            sae = SAE.from_pretrained(
                release="gemma-scope-9b-pt-res-canonical",
                sae_id=f"layer_{layer}/width_131k/canonical",
                device=device,
            )
        except Exception as e:
            logger.error(f"Layer {layer}: SAE load failed: {e}")
            continue

        chunks = []
        for s in range(0, n_games, 256):
            e = min(s + 256, n_games)
            with torch.no_grad():
                chunks.append(sae.encode(h_tensor[s:e]).cpu().numpy())
        features = np.concatenate(chunks, axis=0)
        n_feat = features.shape[1]
        del sae

        np.savez_compressed(
            out_file,
            features=features,
            outcomes=outcomes,
            game_ids=game_ids,
            bet_types=bet_types,
            bet_constraints=bet_constraints,
            prompt_conditions=prompt_conditions,
            final_balances=final_balances,
            rounds_completed=rounds_completed,
            layer=layer,
            model_type="gemma",
            paradigm="coin_flip",
            timestamp=ts,
        )

        sparsity = np.mean(features != 0)
        logger.info(f"Layer {layer}: {n_games}x{n_feat}, sparsity={sparsity:.4f}, saved")

        del h_tensor, features, chunks
        clear_gpu()

    # Summary
    from collections import Counter
    summary = {
        "version": "V2",
        "paradigm": "coin_flip",
        "model": MODEL_ID,
        "sae_source": "sae_lens gemma-scope-9b-pt-res-canonical",
        "chat_template_applied": True,
        "n_games": n_games,
        "n_layers": len(LAYERS),
        "outcomes": dict(Counter(outcomes)),
        "conditions": sorted(set(prompt_conditions)),
        "timestamp": ts,
    }
    with open(OUTPUT_DIR / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Coin Flip SAE Feature Extraction")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Coin Flip SAE Feature Extraction (Gemma)")
    logger.info("=" * 70)

    games = load_games()
    if not games:
        logger.error("No games loaded!")
        return

    from collections import Counter
    logger.info(f"Outcomes: {dict(Counter(g['outcome'] for g in games))}")
    logger.info(f"Conditions: {len(set(g['prompt_condition'] for g in games))} unique")

    t0 = datetime.now()
    hidden_all = extract_hidden_states(games, args.device)
    t_a = (datetime.now() - t0).total_seconds()
    logger.info(f"Phase A: {t_a:.0f}s ({t_a/60:.1f} min)")

    t0 = datetime.now()
    encode_and_save(hidden_all, games, args.device)
    t_b = (datetime.now() - t0).total_seconds()
    logger.info(f"Phase B: {t_b:.0f}s ({t_b/60:.1f} min)")

    logger.info(f"TOTAL: {(t_a + t_b)/60:.1f} min")


if __name__ == "__main__":
    main()
