#!/usr/bin/env python3
"""
V2 Investment Choice SAE Feature Extraction (LLaMA + Gemma)
============================================================

Fixes from V1:
1. Gemma: applies chat_template (V1 used raw prompt — mismatch with experiment)
2. LLaMA: uses fnlp SAE direct (matching LlamaScopeWorking, not sae_lens)
3. Gemma: includes ALL constraints (c10/c30/c50, 1200 games — V1 only had c30/c50)
4. LLaMA: all 7 data files (700 games, c10-c70 × fixed/variable)
5. Stores prompt_condition and prompt_option for downstream analysis

Note: IC prompts are stored from the experiment (full_prompt/prompt field),
so NO decision-point fix is needed (unlike slot machine reconstruction).

Usage:
    python extract_ic_sae_features_v2.py --model gemma --device cuda:1
    python extract_ic_sae_features_v2.py --model llama --device cuda:1
"""

import argparse
import json
import logging
import math
import os
import gc
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# Data Paths
# ===========================================================================
GEMMA_DATA_DIR = Path("/home/jovyan/beomi/llm-addiction-data/investment_choice/gemma_parser_fixed_v2")
GEMMA_FILES = {
    "c10": "gemma_investment_c10_20260223_181530.json",
    "c30": "gemma_investment_c30_20260223_181532.json",
    "c50": "gemma_investment_c50_20260224_001943.json",
}

LLAMA_DATA_DIR = Path("/home/jovyan/beomi/llm-addiction-data/investment_choice/results")
LLAMA_FILES = [
    "llama_10_fixed_20260222_175459.json",
    "llama_10_variable_20260222_175127.json",
    "llama_30_fixed_20260222_182843.json",
    "llama_30_variable_20260222_185256.json",
    "llama_50_fixed_20260222_185630.json",
    "llama_70_fixed_20260222_192251.json",
    "llama_70_variable_20260222_181727.json",
]

OUTPUT_BASE = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v2/investment_choice")

MODEL_IDS = {
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.1-8B",
}

LAYERS = {
    "gemma": list(range(42)),   # 0-41
    "llama": list(range(32)),   # 0-31
}


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ===========================================================================
# Data Loading
# ===========================================================================
def load_gemma_games() -> List[Dict]:
    """Load all Gemma IC games (c10/c30/c50, 1200 games)."""
    games = []
    for constraint, filename in GEMMA_FILES.items():
        filepath = GEMMA_DATA_DIR / filename
        logger.info(f"Loading {filepath}")
        with open(filepath) as f:
            data = json.load(f)

        for g in data["results"]:
            decisions = g.get("decisions", [])
            if not decisions:
                continue
            last_dec = decisions[-1]
            full_prompt = last_dec.get("full_prompt", "")
            if not full_prompt:
                continue

            outcome = "bankruptcy" if g.get("bankruptcy", False) else "voluntary_stop"
            games.append({
                "prompt": full_prompt,
                "outcome": outcome,
                "bet_type": g.get("bet_type", "unknown"),
                "bet_constraint": constraint.replace("c", ""),
                "prompt_condition": g.get("prompt_condition", "unknown"),
                "final_balance": g.get("final_balance", 0),
                "rounds_completed": len(decisions),
                "choice_last": last_dec.get("choice", 0),
                "prompt_option_last": last_dec.get("prompt_option", 0),
                "game_id": g.get("game_id", len(games)),
            })

    logger.info(f"Gemma total: {len(games)} games")
    return games


def load_llama_games() -> List[Dict]:
    """Load all LLaMA IC games (c10-c70 × fixed/variable, ~700 games)."""
    games = []
    for filename in LLAMA_FILES:
        filepath = LLAMA_DATA_DIR / filename
        if not filepath.exists():
            logger.warning(f"Not found: {filepath}")
            continue

        logger.info(f"Loading {filepath}")
        with open(filepath) as f:
            data = json.load(f)

        # Parse constraint from filename: llama_30_fixed_... → "30"
        parts = filename.split("_")
        constraint = parts[1]  # "10", "30", "50", "70"

        for g in data["results"]:
            decisions = g.get("decisions", [])
            if not decisions:
                continue
            last_dec = decisions[-1]
            prompt_text = last_dec.get("prompt", "")
            if not prompt_text:
                continue

            # LLaMA uses exit_reason, not bankruptcy flag
            exit_reason = g.get("exit_reason", "")
            outcome = "bankruptcy" if exit_reason == "bankrupt" else "voluntary_stop"

            games.append({
                "prompt": prompt_text,
                "outcome": outcome,
                "bet_type": g.get("bet_type", "unknown"),
                "bet_constraint": constraint,
                "prompt_condition": g.get("prompt_condition", "unknown"),
                "final_balance": g.get("final_balance", 0),
                "rounds_completed": len(decisions),
                "choice_last": last_dec.get("choice", 0),
                "prompt_option_last": 0,  # LLaMA data doesn't store this
                "game_id": g.get("game_id", len(games)),
            })

    logger.info(f"LLaMA total: {len(games)} games")
    return games


# ===========================================================================
# Phase A: Hidden State Extraction
# ===========================================================================
def extract_hidden_states(games: List[Dict], model_type: str, device: str) -> np.ndarray:
    """Extract last-token hidden states from all layers in one forward pass per game."""
    model_id = MODEL_IDS[model_type]
    layers = LAYERS[model_type]
    n_layers = len(layers)

    logger.info(f"=== Phase A: Hidden State Extraction ({model_type.upper()}) ===")
    logger.info(f"Model: {model_id}, Games: {len(games)}, Layers: {n_layers}")

    gpu_id = int(device.split(":")[1]) if ":" in device else 0
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16,
        device_map={"": gpu_id}, low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    d_model = model.config.hidden_size
    n_games = len(games)
    logger.info(f"d_model={d_model}, allocating ({n_games}, {n_layers}, {d_model}) float32")
    hidden_all = np.zeros((n_games, n_layers, d_model), dtype=np.float32)

    for i in tqdm(range(n_games), desc="Forward passes"):
        prompt = games[i]["prompt"]

        # Apply chat_template for Gemma (instruction-tuned)
        if model_type == "gemma":
            chat = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(device)

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)

        for j, layer in enumerate(layers):
            h = outputs.hidden_states[layer + 1][0, -1, :]
            hidden_all[i, j, :] = h.float().cpu().numpy()

    del model, tokenizer
    clear_gpu()
    logger.info(f"Phase A complete: {hidden_all.shape}")
    return hidden_all


# ===========================================================================
# Phase B: SAE Encoding + Save
# ===========================================================================
def load_fnlp_sae(layer: int, device: str):
    """Load fnlp LlamaScope SAE (matching LlamaScopeWorking)."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    ckpt = hf_hub_download(
        repo_id="fnlp/Llama3_1-8B-Base-LXR-8x",
        filename=f"Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors",
    )
    hp_path = hf_hub_download(
        repo_id="fnlp/Llama3_1-8B-Base-LXR-8x",
        filename=f"Llama3_1-8B-Base-L{layer}R-8x/hyperparams.json",
    )
    with open(hp_path) as f:
        hp = json.load(f)

    weights = load_file(ckpt, device="cpu")
    norm_factor = math.sqrt(hp["d_model"]) / hp["dataset_average_activation_norm"]["in"]
    W_E = weights["encoder.weight"].T.float().to(device)
    b_E = weights["encoder.bias"].float().to(device)

    return {"W_E": W_E, "b_E": b_E, "norm_factor": norm_factor}


def fnlp_encode(h: torch.Tensor, params: dict) -> torch.Tensor:
    """Encode: ReLU((h.float() * norm_factor) @ W_E + b_E)"""
    x = h.float() * params["norm_factor"]
    return torch.relu(x @ params["W_E"] + params["b_E"])


def encode_and_save(hidden_all: np.ndarray, games: List[Dict], model_type: str, device: str):
    """Phase B: Encode hidden states with correct SAE per model and save NPZ."""
    layers = LAYERS[model_type]
    n_games = len(games)
    output_dir = OUTPUT_BASE / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Metadata arrays
    outcomes = np.array([g["outcome"] for g in games])
    game_ids = np.array([g["game_id"] for g in games], dtype=np.int64)
    bet_types = np.array([g["bet_type"] for g in games])
    bet_constraints = np.array([g["bet_constraint"] for g in games])
    prompt_conditions = np.array([g["prompt_condition"] for g in games])
    final_balances = np.array([g["final_balance"] for g in games], dtype=np.float64)
    rounds_completed = np.array([g["rounds_completed"] for g in games], dtype=np.int64)
    choices_last = np.array([g["choice_last"] for g in games], dtype=np.int64)
    prompt_options_last = np.array([g["prompt_option_last"] for g in games], dtype=np.int64)

    logger.info(f"=== Phase B: SAE Encoding ({model_type.upper()}) ===")
    logger.info(f"Output: {output_dir}")

    for j, layer in enumerate(layers):
        out_file = output_dir / f"layer_{layer}_features.npz"
        if out_file.exists():
            logger.info(f"Layer {layer}: exists, skipping")
            continue

        h_layer = hidden_all[:, j, :]  # (n_games, d_model)
        h_tensor = torch.tensor(h_layer, dtype=torch.float32, device=device)

        if model_type == "llama":
            try:
                sae_params = load_fnlp_sae(layer, device)
            except Exception as e:
                logger.error(f"Layer {layer}: fnlp SAE load failed: {e}")
                continue

            chunks = []
            for s in range(0, n_games, 256):
                e = min(s + 256, n_games)
                with torch.no_grad():
                    chunks.append(fnlp_encode(h_tensor[s:e], sae_params).cpu().numpy())
            features = np.concatenate(chunks, axis=0)
            n_feat = features.shape[1]
            del sae_params

        else:  # gemma
            from sae_lens import SAE
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
            choices_last=choices_last,
            prompt_options_last=prompt_options_last,
            layer=layer,
            model_type=model_type,
            timestamp=ts,
        )

        sparsity = np.mean(features != 0)
        logger.info(f"Layer {layer}: {n_games}x{n_feat}, sparsity={sparsity:.4f}, saved")

        del h_tensor, features, chunks
        clear_gpu()

    # Summary
    summary = {
        "version": "V2",
        "model": MODEL_IDS[model_type],
        "model_type": model_type,
        "sae_source": "fnlp_direct" if model_type == "llama" else "sae_lens",
        "chat_template_applied": model_type == "gemma",
        "n_games": n_games,
        "n_layers": len(layers),
        "n_bankruptcy": int((outcomes == "bankruptcy").sum()),
        "n_voluntary": int((outcomes == "voluntary_stop").sum()),
        "bet_constraints": sorted(set(bet_constraints)),
        "timestamp": ts,
    }
    with open(output_dir / "extraction_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="V2 IC SAE Feature Extraction")
    parser.add_argument("--model", required=True, choices=["llama", "gemma"])
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"V2 Investment Choice SAE Extraction: {args.model.upper()}")
    logger.info("=" * 70)

    if args.model == "gemma":
        games = load_gemma_games()
    else:
        games = load_llama_games()

    if not games:
        logger.error("No games loaded!")
        return

    from collections import Counter
    logger.info(f"Outcomes: {dict(Counter(g['outcome'] for g in games))}")
    logger.info(f"Bet types: {dict(Counter(g['bet_type'] for g in games))}")
    logger.info(f"Constraints: {dict(Counter(g['bet_constraint'] for g in games))}")

    t0 = datetime.now()
    hidden_all = extract_hidden_states(games, args.model, args.device)
    t_a = (datetime.now() - t0).total_seconds()
    logger.info(f"Phase A: {t_a:.0f}s ({t_a/60:.1f} min)")

    t0 = datetime.now()
    encode_and_save(hidden_all, games, args.model, args.device)
    t_b = (datetime.now() - t0).total_seconds()
    logger.info(f"Phase B: {t_b:.0f}s ({t_b/60:.1f} min)")

    logger.info(f"TOTAL: {(t_a + t_b)/60:.1f} min")


if __name__ == "__main__":
    main()
