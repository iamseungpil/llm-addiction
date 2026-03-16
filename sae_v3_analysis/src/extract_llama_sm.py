#!/usr/bin/env python3
"""
Extract LLaMA SM (Slot Machine) V4role SAE features (all rounds).

Structurally identical to extract_llama_ic.py, adapted for SM data format.

SM-specific differences from IC:
  - SM decisions do NOT contain saved prompts; prompts are reconstructed via
    prompt_reconstruction.reconstruct_sm_prompt().
  - Game outcome comes from `game["outcome"]` string field
    ("bankruptcy", "voluntary_stop", "max_rounds").
  - Prompt condition is `game["prompt_combo"]` (e.g. "BASE", "GM", "GMRWP").
  - bet_constraint is "N/A" (SM does not use IC-style constraints).
  - Decisions use `action` ("bet"/"stop"/"skip") instead of `choice`.
  - History entries map 1:1 with bet decisions (needed for prompt reconstruction).

Pipeline:
  Phase A: Forward pass through LLaMA-3.1-8B-Instruct -> hidden states
  Phase B: fnlp/LlamaScope SAE encoding -> sparse COO NPZ

Output format identical to Gemma V3 (extract_all_rounds.py):
  sae_features_L{0-31}.npz with sparse COO + metadata arrays

Usage:
    python extract_llama_sm.py --gpu 0
    python extract_llama_sm.py --gpu 1 --phase-a-only
    python extract_llama_sm.py --gpu 0 --layers 10,11,12,15,16
"""

import os
import sys
import json
import math
import gc
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
import argparse
import logging

from prompt_reconstruction import reconstruct_sm_prompt

# ============================================================
# Constants
# ============================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
N_LAYERS = 32
N_SAE_FEATURES = 32768  # fnlp LlamaScope width
HIDDEN_DIM = 4096  # LLaMA-3.1-8B hidden dimension
RANDOM_SEED = 42

DATA_DIR = Path("/home/jovyan/beomi/llm-addiction-data/slot_machine/experiment_0_llama_v4_role")
OUTPUT_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v3/slot_machine/llama")


@dataclass
class RoundRecord:
    game_id: int
    round_num: int
    prompt: str
    balance_before: float
    choice: Optional[int]
    bet_amount: Optional[float]
    game_outcome: str  # "bankruptcy", "voluntary_stop", "max_rounds"
    bet_type: str
    bet_constraint: str
    prompt_condition: str
    is_last_round: bool
    paradigm: str = "sm"


# ============================================================
# Data Loading
# ============================================================

def load_sm_rounds(data_dir: Path, logger: logging.Logger) -> List[RoundRecord]:
    """Load all SM V4role rounds, reconstructing prompts from game state.

    SM data format (from final_llama_YYYYMMDD.json):
      - results[].outcome: "voluntary_stop" | "bankruptcy" | "max_rounds"
      - results[].prompt_combo: "BASE", "G", "GM", "GMRWP", etc.
      - results[].bet_type: "fixed" or "variable"
      - results[].decisions[]: action, bet, balance_before, balance_after, ...
      - results[].history[]: round, bet, result, balance, win (slot outcomes)

    Prompts are NOT stored in SM decisions; they are reconstructed using
    prompt_reconstruction.reconstruct_sm_prompt() with the game state
    available at each decision point.
    """
    rounds = []
    game_counter = 0

    # Find data files (final_*.json, exclude checkpoints)
    json_files = sorted(data_dir.glob("final_*.json"))
    if not json_files:
        # Fallback: try any JSON that is not a checkpoint
        json_files = sorted(
            f for f in data_dir.glob("*.json")
            if "checkpoint" not in f.name
        )

    logger.info(f"Found {len(json_files)} data files: {[f.name for f in json_files]}")

    for filepath in json_files:
        with open(filepath) as f:
            data = json.load(f)

        results = data.get("results", [])
        logger.info(f"  {filepath.name}: {len(results)} games")

        for game in results:
            game_counter += 1
            gid = game_counter

            decisions = game.get("decisions", [])
            history = game.get("history", [])
            bet_type = game.get("bet_type", "fixed")
            prompt_combo = game.get("prompt_combo", "BASE")
            game_outcome = game.get("outcome", "")

            # Filter out skip decisions (no valid model response)
            valid_decisions = [d for d in decisions if d.get("action") != "skip"]

            if not valid_decisions:
                continue

            # Track history index: bet decisions map 1:1 to history entries
            hist_idx = 0
            n_valid = len(valid_decisions)

            for i, dec in enumerate(valid_decisions):
                action = dec.get("action", "")
                balance = dec.get("balance_before", 100)

                # History slice: entries BEFORE this decision (what model saw)
                history_slice = history[:hist_idx]

                # Reconstruct the prompt the model saw
                prompt = reconstruct_sm_prompt(
                    prompt_combo=prompt_combo,
                    bet_type=bet_type,
                    balance=balance,
                    history_slice=history_slice,
                )

                # Map action to choice: 1 = bet, 2 = stop
                choice = 1 if action == "bet" else 2

                rounds.append(RoundRecord(
                    game_id=gid,
                    round_num=dec.get("round", i + 1),
                    prompt=prompt,
                    balance_before=balance,
                    choice=choice,
                    bet_amount=dec.get("bet") if action == "bet" else None,
                    game_outcome=game_outcome,
                    bet_type=bet_type,
                    bet_constraint="N/A",
                    prompt_condition=prompt_combo,
                    is_last_round=(i == n_valid - 1),
                    paradigm="sm",
                ))

                # Advance history index after bet decisions
                if action == "bet":
                    hist_idx += 1

    logger.info(f"Loaded {len(rounds)} rounds from {game_counter} games")
    n_bk = len(set(r.game_id for r in rounds if r.game_outcome == "bankruptcy"))
    logger.info(f"  Bankruptcies: {n_bk} games")
    if game_counter > 0:
        logger.info(f"  Avg rounds/game: {len(rounds)/game_counter:.1f}")
    return rounds


# ============================================================
# Sparse COO conversion
# ============================================================

def dense_to_sparse_coo(features: np.ndarray):
    """Convert dense feature matrix to sparse COO arrays."""
    rows, cols = np.nonzero(features)
    values = features[rows, cols].astype(np.float32)
    return rows.astype(np.int32), cols.astype(np.int32), values


# ============================================================
# Phase A: Hidden State Extraction
# ============================================================

def phase_a_extract(
    rounds: List[RoundRecord],
    layers: List[int],
    device: str,
    logger: logging.Logger,
    checkpoint_dir: Optional[Path] = None,
) -> tuple:
    """Extract hidden states for all rounds via single forward pass per round.

    Returns:
        hidden_all: (n_rounds, n_layers, hidden_dim) float32 array
        valid_mask: (n_rounds,) bool array
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    n_rounds = len(rounds)
    n_layers_req = len(layers)

    mem_gb = n_rounds * n_layers_req * HIDDEN_DIM * 4 / 1e9
    logger.info(f"Phase A: {n_rounds} rounds, {n_layers_req} layers, dim={HIDDEN_DIM}")
    logger.info(f"Memory estimate: {mem_gb:.2f} GB")

    # Check for Phase A checkpoint
    if checkpoint_dir:
        ckpt_file = checkpoint_dir / "phase_a_hidden_states.npz"
        meta_file = checkpoint_dir / "phase_a_metadata.json"
        if ckpt_file.exists() and meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            if not meta.get("partial", True) and meta.get("n_rounds") == n_rounds:
                logger.info(f"Loading Phase A checkpoint ({n_rounds} rounds)")
                data = np.load(ckpt_file)
                return data["hidden_states"], data["valid_mask"]
            else:
                logger.info("Partial/mismatched checkpoint found, re-extracting")

    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    gpu_id = int(device.split(":")[1]) if ":" in device else 0
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": gpu_id},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    logger.info("Model loaded")

    # Pre-allocate
    hidden_all = np.zeros((n_rounds, n_layers_req, HIDDEN_DIM), dtype=np.float32)
    valid_mask = np.ones(n_rounds, dtype=bool)
    n_skipped = 0

    for i in tqdm(range(n_rounds), desc="Forward passes"):
        try:
            prompt = rounds[i].prompt

            # Apply LLaMA Instruct chat template
            chat = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=2048
            ).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"], output_hidden_states=True
                )

            # Extract last-token hidden state for each layer
            for j, layer in enumerate(layers):
                h = outputs.hidden_states[layer + 1][:, -1, :]
                hidden_all[i, j, :] = h.float().cpu().numpy().squeeze()

        except Exception as e:
            logger.error(f"Error at round {i} (game {rounds[i].game_id}, "
                         f"round {rounds[i].round_num}): {e}")
            valid_mask[i] = False
            n_skipped += 1

        # Checkpoint every 2000 rounds
        if checkpoint_dir and i > 0 and i % 2000 == 0:
            _save_checkpoint(checkpoint_dir, hidden_all, valid_mask,
                             n_rounds, layers, n_skipped, logger, partial=True)

    logger.info(f"Phase A complete: {n_rounds - n_skipped}/{n_rounds} valid ({n_skipped} skipped)")

    # Unload model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model unloaded")

    # Save final checkpoint
    if checkpoint_dir:
        _save_checkpoint(checkpoint_dir, hidden_all, valid_mask,
                         n_rounds, layers, n_skipped, logger, partial=False)

    return hidden_all, valid_mask


def _save_checkpoint(checkpoint_dir, hidden_all, valid_mask, n_rounds, layers,
                     n_skipped, logger, partial=False):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoint_dir / "phase_a_hidden_states.npz"
    label = "partial" if partial else "final"
    logger.info(f"Saving Phase A {label} checkpoint")
    np.savez_compressed(ckpt_file, hidden_states=hidden_all, valid_mask=valid_mask)
    meta = {
        "n_rounds": n_rounds, "n_layers": len(layers), "layers": layers,
        "hidden_dim": HIDDEN_DIM, "n_skipped": n_skipped,
        "partial": partial, "timestamp": datetime.now().isoformat(),
    }
    with open(checkpoint_dir / "phase_a_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Phase B: SAE Encoding + Sparse Save
# ============================================================

def load_fnlp_sae(layer: int, device: str, logger) -> dict:
    """Load fnlp LlamaScope SAE (ReLU encoding, NOT sae_lens)."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id="fnlp/Llama3_1-8B-Base-LXR-8x",
        filename=f"Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors",
    )
    hp_path = hf_hub_download(
        repo_id="fnlp/Llama3_1-8B-Base-LXR-8x",
        filename=f"Llama3_1-8B-Base-L{layer}R-8x/hyperparams.json",
    )
    with open(hp_path) as f:
        hp = json.load(f)

    dataset_norm = hp["dataset_average_activation_norm"]["in"]
    d_model = hp["d_model"]
    norm_factor = math.sqrt(d_model) / dataset_norm

    weights = load_file(ckpt_path, device="cpu")
    W_E = weights["encoder.weight"].T.float().to(device)  # (4096, 32768)
    b_E = weights["encoder.bias"].float().to(device)       # (32768,)

    logger.info(f"fnlp SAE L{layer}: norm_factor={norm_factor:.6f}, W_E={W_E.shape}")
    return {"W_E": W_E, "b_E": b_E, "norm_factor": norm_factor}


def fnlp_encode(h: torch.Tensor, sae_params: dict) -> torch.Tensor:
    """ReLU((h.float() * norm_factor) @ W_E + b_E)"""
    x = h.float() * sae_params["norm_factor"]
    return torch.relu(x @ sae_params["W_E"] + sae_params["b_E"])


def phase_b_encode(
    hidden_all: np.ndarray,
    valid_mask: np.ndarray,
    rounds: List[RoundRecord],
    layers: List[int],
    output_dir: Path,
    device: str,
    logger: logging.Logger,
):
    """Encode hidden states through fnlp SAE and save as sparse COO."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_rounds = len(rounds)

    for j, layer in enumerate(layers):
        sae_file = output_dir / f"sae_features_L{layer}.npz"
        if sae_file.exists():
            logger.info(f"Layer {layer}: already exists, skipping")
            continue

        logger.info(f"Encoding layer {layer}...")

        h_layer = hidden_all[:, j, :]  # (n_rounds, 4096)
        h_tensor = torch.tensor(h_layer, dtype=torch.float32, device=device)

        try:
            sae_params = load_fnlp_sae(layer, device, logger)
        except Exception as e:
            logger.error(f"Failed to load SAE for layer {layer}: {e}")
            del h_tensor
            torch.cuda.empty_cache()
            continue

        # Batch encode
        chunk_size = 256
        all_features = []
        for start in range(0, n_rounds, chunk_size):
            end = min(start + chunk_size, n_rounds)
            with torch.no_grad():
                feats = fnlp_encode(h_tensor[start:end], sae_params)
            all_features.append(feats.cpu().numpy())

        del sae_params, h_tensor
        torch.cuda.empty_cache()
        gc.collect()

        features_dense = np.concatenate(all_features, axis=0)
        del all_features

        # Convert to sparse COO
        row_indices, col_indices, values = dense_to_sparse_coo(features_dense)
        nnz = len(values)
        avg_nnz = nnz / n_rounds if n_rounds > 0 else 0

        logger.info(
            f"Layer {layer}: {n_rounds} rounds, {nnz} nnz "
            f"(avg {avg_nnz:.0f}/round, sparsity {1 - avg_nnz / N_SAE_FEATURES:.4%})"
        )

        # Build metadata arrays
        game_ids = np.array([r.game_id for r in rounds], dtype=np.int64)
        round_nums = np.array([r.round_num for r in rounds], dtype=np.int32)
        game_outcomes = np.array([r.game_outcome for r in rounds])
        bet_types = np.array([r.bet_type for r in rounds])
        bet_constraints = np.array([r.bet_constraint for r in rounds])
        prompt_conditions = np.array([r.prompt_condition for r in rounds])
        balances = np.array([r.balance_before for r in rounds], dtype=np.float64)
        is_last_round = np.array([r.is_last_round for r in rounds], dtype=bool)
        paradigms = np.array([r.paradigm for r in rounds])

        np.savez_compressed(
            sae_file,
            row_indices=row_indices,
            col_indices=col_indices,
            values=values,
            shape=np.array([n_rounds, N_SAE_FEATURES], dtype=np.int64),
            game_ids=game_ids,
            round_nums=round_nums,
            game_outcomes=game_outcomes,
            bet_types=bet_types,
            bet_constraints=bet_constraints,
            prompt_conditions=prompt_conditions,
            balances=balances,
            is_last_round=is_last_round,
            valid_mask=valid_mask,
            paradigms=paradigms,
            layer=layer,
        )

        # Metadata JSON sidecar
        meta = {
            "layer": layer,
            "n_rounds": n_rounds,
            "n_features": N_SAE_FEATURES,
            "nnz_total": int(nnz),
            "avg_nnz_per_round": float(avg_nnz),
            "sparsity": float(1 - avg_nnz / N_SAE_FEATURES),
            "n_games": len(set(r.game_id for r in rounds)),
            "n_bankruptcy": sum(1 for r in rounds if r.game_outcome == "bankruptcy" and r.is_last_round),
            "model": MODEL_NAME,
            "sae_source": "fnlp_direct",
            "storage_format": "sparse_coo",
            "timestamp": datetime.now().isoformat(),
        }
        with open(sae_file.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

        del features_dense, row_indices, col_indices, values
        gc.collect()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Extract LLaMA SM V4role SAE features")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--layers", type=str, default="all",
                        help="Comma-separated layers or 'all' for 0-31")
    parser.add_argument("--phase-a-only", action="store_true",
                        help="Only extract hidden states (skip SAE encoding)")
    args = parser.parse_args()

    # Setup
    device = f"cuda:{args.gpu}"
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    if args.layers == "all":
        layers = list(range(N_LAYERS))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    # Logger
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extraction_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("LLaMA SM V4role SAE Feature Extraction")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"SAE: fnlp/LlamaScope ({N_SAE_FEATURES} features/layer)")
    logger.info(f"Layers: {layers}")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load data
    rounds = load_sm_rounds(data_dir, logger)
    if not rounds:
        logger.error("No rounds loaded!")
        return

    # Phase A: Hidden states
    logger.info("\n" + "=" * 70)
    logger.info("PHASE A: Hidden State Extraction")
    logger.info("=" * 70)
    hidden_all, valid_mask = phase_a_extract(
        rounds, layers, device, logger, checkpoint_dir
    )

    if args.phase_a_only:
        logger.info("Phase A only mode -- done")
        return

    # Phase B: SAE encoding
    logger.info("\n" + "=" * 70)
    logger.info("PHASE B: SAE Encoding (fnlp/LlamaScope)")
    logger.info("=" * 70)
    phase_b_encode(hidden_all, valid_mask, rounds, layers, output_dir, device, logger)

    # Save extraction summary
    summary = {
        "model": MODEL_NAME,
        "sae": "fnlp/Llama3_1-8B-Base-LXR-8x",
        "paradigm": "sm",
        "n_games": len(set(r.game_id for r in rounds)),
        "n_rounds": len(rounds),
        "n_layers": len(layers),
        "layers": layers,
        "n_bankruptcy": len(set(r.game_id for r in rounds if r.game_outcome == "bankruptcy")),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
