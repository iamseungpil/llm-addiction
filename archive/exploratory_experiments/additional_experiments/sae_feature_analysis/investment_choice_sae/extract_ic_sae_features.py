"""
Investment Choice SAE Feature Extraction (Gemma 42 Layers)

Extract GemmaScope SAE features from c30/c50 investment choice experiment data.
- Phase A: Load Gemma model, extract hidden states from all 42 layers
- Phase B: Encode hidden states through GemmaScope SAE (131K features per layer)

Output: layer_{0..41}_features.npz files with game-level metadata.

Usage:
    CUDA_VISIBLE_DEVICES=1 python extract_ic_sae_features.py --gpu 0
"""

import argparse
import json
import logging
import os
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_WIDTH = "131k"
D_SAE = 131072
N_LAYERS = 42

DATA_DIR = Path("/home/jovyan/beomi/llm-addiction-data/investment_choice/gemma_parser_fixed_v2")
OUTPUT_DIR = Path("/home/jovyan/beomi/llm-addiction-data/investment_choice_sae/gemma_42layers")

C30_FILE = "gemma_investment_c30_20260223_181532.json"
C50_FILE = "gemma_investment_c50_20260224_001943.json"


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_games() -> List[Dict[str, Any]]:
    """Load c30 and c50 game data, extract metadata and last-decision prompts."""
    games = []

    for filename, constraint in [(C30_FILE, "30"), (C50_FILE, "50")]:
        filepath = DATA_DIR / filename
        logger.info(f"Loading {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        results = data["results"]
        logger.info(f"  c{constraint}: {len(results)} games")

        for g in results:
            decisions = g.get("decisions", [])
            if not decisions:
                logger.warning(f"Game {g.get('game_id')} has no decisions, skipping")
                continue

            last_decision = decisions[-1]
            full_prompt = last_decision.get("full_prompt", "")
            if not full_prompt:
                logger.warning(f"Game {g.get('game_id')} has no full_prompt, skipping")
                continue

            # Determine outcome
            if g.get("bankruptcy", False):
                outcome = "bankruptcy"
            else:
                outcome = "voluntary_stop"

            games.append({
                "game_id": g.get("game_id", len(games)),
                "full_prompt": full_prompt,
                "outcome": outcome,
                "bet_type": g.get("bet_type", "unknown"),
                "bet_constraint": constraint,
                "final_balance": g.get("final_balance", 0),
                "rounds_completed": g.get("rounds_completed", 0),
                "choice_last": last_decision.get("choice", 0),
            })

    logger.info(f"Total games loaded: {len(games)}")

    # Print summary
    from collections import Counter
    outcomes = Counter(g["outcome"] for g in games)
    bet_types = Counter(g["bet_type"] for g in games)
    constraints = Counter(g["bet_constraint"] for g in games)
    logger.info(f"  Outcomes: {dict(outcomes)}")
    logger.info(f"  Bet types: {dict(bet_types)}")
    logger.info(f"  Constraints: {dict(constraints)}")

    return games


def extract_hidden_states(games: List[Dict], device: torch.device) -> np.ndarray:
    """
    Phase A: Load Gemma model and extract hidden states from all 42 layers.

    Returns:
        hidden_states_all: (N_games, N_layers, d_model) float32 array
    """
    logger.info(f"=== Phase A: Hidden State Extraction ===")
    logger.info(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Get hidden dim from model config
    d_model = model.config.hidden_size
    n_games = len(games)
    logger.info(f"Model loaded. d_model={d_model}, n_games={n_games}")
    logger.info(f"Allocating hidden states array: ({n_games}, {N_LAYERS}, {d_model}) float32")

    # Pre-allocate RAM array for all hidden states
    hidden_states_all = np.zeros((n_games, N_LAYERS, d_model), dtype=np.float32)

    for i, game in enumerate(tqdm(games, desc="Extracting hidden states")):
        prompt = game["full_prompt"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # outputs.hidden_states: tuple of (n_layers+1,) tensors of shape (1, seq_len, d_model)
        # Index 0 = embeddings, index 1..42 = layer 0..41
        for layer in range(N_LAYERS):
            last_token = outputs.hidden_states[layer + 1][0, -1, :]  # (d_model,)
            hidden_states_all[i, layer, :] = last_token.float().cpu().numpy()

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{n_games} games")

    # Unload model
    del model, tokenizer
    clear_gpu_memory()
    logger.info(f"Phase A complete. Hidden states shape: {hidden_states_all.shape}")

    return hidden_states_all


def encode_and_save(
    hidden_states_all: np.ndarray,
    games: List[Dict],
    device: torch.device,
):
    """
    Phase B: Encode hidden states through GemmaScope SAE and save per-layer NPZ files.
    """
    logger.info(f"=== Phase B: SAE Encoding ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_games = len(games)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare metadata arrays (same for all layers)
    outcomes = np.array([g["outcome"] for g in games])
    game_ids = np.array([g["game_id"] for g in games], dtype=np.int64)
    bet_types = np.array([g["bet_type"] for g in games])
    bet_constraints = np.array([g["bet_constraint"] for g in games])
    final_balances = np.array([g["final_balance"] for g in games], dtype=np.float64)
    rounds_completed = np.array([g["rounds_completed"] for g in games], dtype=np.int64)
    choices_last = np.array([g["choice_last"] for g in games], dtype=np.int64)

    for layer in range(N_LAYERS):
        logger.info(f"--- Layer {layer}/{N_LAYERS - 1} ---")

        # Load SAE for this layer
        sae_id = f"layer_{layer}/width_{SAE_WIDTH}/canonical"
        logger.info(f"  Loading SAE: {SAE_RELEASE} / {sae_id}")

        sae, _, _ = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=sae_id,
            device=str(device),
        )

        # Encode all games for this layer
        features = np.zeros((n_games, D_SAE), dtype=np.float32)

        # Process in batches to manage GPU memory
        batch_size = 64
        hidden_layer = hidden_states_all[:, layer, :]  # (n_games, d_model)

        for start in range(0, n_games, batch_size):
            end = min(start + batch_size, n_games)
            batch = torch.tensor(hidden_layer[start:end], dtype=torch.float32, device=device)

            with torch.no_grad():
                encoded = sae.encode(batch)  # (batch_size, D_SAE)

            features[start:end] = encoded.cpu().numpy()

        # Sparsity check
        sparsity = np.mean(features != 0)
        logger.info(f"  Sparsity (non-zero ratio): {sparsity:.4f}")

        # Save NPZ
        output_file = OUTPUT_DIR / f"layer_{layer}_features.npz"
        np.savez_compressed(
            output_file,
            features=features,
            outcomes=outcomes,
            game_ids=game_ids,
            bet_types=bet_types,
            bet_constraints=bet_constraints,
            final_balances=final_balances,
            rounds_completed=rounds_completed,
            choices_last=choices_last,
            layer=layer,
            model_type="gemma",
            timestamp=timestamp,
        )

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved {output_file} ({file_size_mb:.1f} MB)")

        # Unload SAE
        del sae, features
        clear_gpu_memory()

    logger.info(f"Phase B complete. {N_LAYERS} layer files saved to {OUTPUT_DIR}")

    # Save summary JSON
    summary = {
        "timestamp": timestamp,
        "model": MODEL_ID,
        "sae_release": SAE_RELEASE,
        "sae_width": SAE_WIDTH,
        "n_games": n_games,
        "n_layers": N_LAYERS,
        "d_sae": D_SAE,
        "output_dir": str(OUTPUT_DIR),
        "source_files": {
            "c30": str(DATA_DIR / C30_FILE),
            "c50": str(DATA_DIR / C50_FILE),
        },
        "outcomes_distribution": {
            "bankruptcy": int(np.sum(outcomes == "bankruptcy")),
            "voluntary_stop": int(np.sum(outcomes == "voluntary_stop")),
        },
    }
    summary_file = OUTPUT_DIR / "extraction_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="IC SAE Feature Extraction (Gemma 42 Layers)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Load game data
    games = load_games()
    if len(games) == 0:
        logger.error("No games loaded, exiting")
        return

    # Phase A: Extract hidden states
    hidden_states_all = extract_hidden_states(games, device)

    # Phase B: SAE encoding + save
    encode_and_save(hidden_states_all, games, device)

    logger.info("=== All phases complete! ===")


if __name__ == "__main__":
    main()
