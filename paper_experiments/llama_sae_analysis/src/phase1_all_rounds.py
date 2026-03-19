#!/usr/bin/env python3
"""
Phase 1 All-Rounds Extraction: Hidden States + Sparse SAE Features

Extracts hidden states at EVERY round of EVERY game (not just decision-point)
for three paradigms: Investment Choice (IC), Slot Machine (SM), Mystery Wheel (MW).

Saves:
  - Sparse SAE features (COO format) per layer: sae_features_L{N}.npz
  - Raw hidden states per layer (optional): hidden_states_L{N}.npz
  - Metadata: extraction_summary.json

Usage:
    # IC V2role on GPU 0
    python phase1_all_rounds.py --paradigm ic --device cuda:0

    # SM V4role on GPU 1
    python phase1_all_rounds.py --paradigm sm --device cuda:1

    # MW V2role on GPU 0, skip hidden state saving
    python phase1_all_rounds.py --paradigm mw --device cuda:0 --no-save-hidden

    # Resume interrupted extraction
    python phase1_all_rounds.py --paradigm ic --device cuda:0 --resume
"""

import os
import sys
import json
import gc
import shutil
import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_reconstruction import reconstruct_sm_prompt


# ============================================================
# Configuration
# ============================================================

DATA_BASE = "/home/jovyan/beomi/llm-addiction-data"

PARADIGM_CONFIG = {
    "ic": {
        "name": "Investment Choice V2role",
        "data_dir": f"{DATA_BASE}/investment_choice_v2_role",
        "output_dir": f"{DATA_BASE}/sae_features_v3/investment_choice/gemma",
        "files": [
            "gemma_investment_c10_20260225_122319.json",
            "gemma_investment_c30_20260225_184458.json",
            "gemma_investment_c50_20260226_020821.json",
            "gemma_investment_c70_20260226_082029.json",
        ],
    },
    "sm": {
        "name": "Slot Machine V4role",
        "data_dir": f"{DATA_BASE}/slot_machine/experiment_0_gemma_v4_role",
        "output_dir": f"{DATA_BASE}/sae_features_v3/slot_machine/gemma",
        "files": ["final_gemma_20260227_002507.json"],
    },
    "mw": {
        "name": "Mystery Wheel V2role",
        "data_dir": f"{DATA_BASE}/mystery_wheel_v2_role",
        "output_dir": f"{DATA_BASE}/sae_features_v3/mystery_wheel/gemma",
        "files": ["gemma_mysterywheel_checkpoint_3200.json"],
    },
}

# Gemma-2-9B-IT: 42 layers, hidden_dim=3584, SAE=131K features
MODEL_NAME = "google/gemma-2-9b-it"
N_LAYERS = 42
HIDDEN_DIM = 3584
N_SAE_FEATURES = 131072


# ============================================================
# RoundRecord: per-round metadata
# ============================================================

@dataclass
class RoundRecord:
    """Metadata for one decision round."""
    game_id: int
    round_num: int
    prompt: str  # The exact prompt the model saw
    balance_before: float
    choice: Optional[int]  # game choice (IC: 1-4, SM: bet/stop, MW: 1-2)
    bet_amount: Optional[float]
    game_outcome: str  # 'bankruptcy', 'voluntary_stop', 'max_rounds', etc.
    bet_type: str  # 'fixed' or 'variable'
    bet_constraint: str  # '10', '30', '50', '70', 'unlimited', 'N/A'
    prompt_condition: str  # 'BASE', 'G', 'GM', 'GMRWP', etc.
    is_last_round: bool  # True if this is the last decision in the game
    paradigm: str  # 'ic', 'sm', 'mw'


# ============================================================
# Paradigm Adapters
# ============================================================

class ParadigmAdapter(ABC):
    """Base class for loading per-round data from each paradigm."""

    @abstractmethod
    def load_rounds(self) -> List[RoundRecord]:
        """Load all rounds from the paradigm's data files."""
        pass


class ICAdapter(ParadigmAdapter):
    """Investment Choice V2role adapter. Uses saved full_prompt directly."""

    def __init__(self, config: dict):
        self.config = config

    def load_rounds(self) -> List[RoundRecord]:
        rounds = []
        # Global counter to avoid game_id collisions across files
        # (each file has game_ids 1-400, so they overlap without remapping)
        game_counter = 0

        for filename in self.config["files"]:
            filepath = os.path.join(self.config["data_dir"], filename)
            with open(filepath) as f:
                data = json.load(f)

            file_constraint = data.get("config", {}).get("bet_constraint", "unknown")

            for game in data["results"]:
                game_counter += 1
                gid = game_counter
                decisions = game.get("decisions", [])
                game_outcome = game.get("final_outcome", "")
                if game.get("bankruptcy"):
                    game_outcome = "bankruptcy"
                elif game.get("stopped_voluntarily"):
                    game_outcome = "voluntary_stop"
                elif game.get("max_rounds_reached"):
                    game_outcome = "max_rounds"

                # Per-game bet_constraint is more robust than file-level config
                constraint = str(game.get("bet_constraint", file_constraint))

                # Pre-compute last valid (non-skipped) decision index
                valid_indices = [j for j, dec in enumerate(decisions) if not dec.get("skipped")]
                last_valid_idx = valid_indices[-1] if valid_indices else -1

                for i, dec in enumerate(decisions):
                    if dec.get("skipped"):
                        continue
                    prompt = dec.get("full_prompt") or dec.get("actual_prompt", "")
                    if not prompt:
                        continue

                    rounds.append(RoundRecord(
                        game_id=gid,
                        round_num=dec.get("round", i + 1),
                        prompt=prompt,
                        balance_before=dec.get("balance_before", 100),
                        choice=dec.get("choice"),
                        bet_amount=dec.get("bet_amount"),
                        game_outcome=game_outcome,
                        bet_type=game.get("bet_type", "unknown"),
                        bet_constraint=constraint,
                        prompt_condition=game.get("prompt_condition", "BASE"),
                        is_last_round=(i == last_valid_idx),
                        paradigm="ic",
                    ))

        return rounds


class SMAdapter(ParadigmAdapter):
    """Slot Machine V4role adapter. Reconstructs prompts from game data."""

    def __init__(self, config: dict):
        self.config = config

    def load_rounds(self) -> List[RoundRecord]:
        rounds = []
        game_counter = 0  # Cross-file counter, matching IC/MW pattern

        for filename in self.config["files"]:
            filepath = os.path.join(self.config["data_dir"], filename)
            with open(filepath) as f:
                data = json.load(f)

            results = data.get("results", data if isinstance(data, list) else [])

            for game in results:
                game_counter += 1
                decisions = game.get("decisions", [])
                history = game.get("history", [])
                bet_type = game.get("bet_type", "fixed")
                prompt_combo = game.get("prompt_combo", "BASE")
                game_outcome = game.get("outcome", "")

                # Filter out skip decisions — they have no valid model response
                # and lack proper fields (no 'bet', different 'balance' key)
                valid_decisions = [d for d in decisions if d.get("action") != "skip"]

                # Track history index: bet decisions map 1:1 to history entries
                hist_idx = 0
                n_valid = len(valid_decisions)

                for i, dec in enumerate(valid_decisions):
                    action = dec.get("action", "")
                    balance = dec.get("balance_before", 100)

                    # History slice: entries BEFORE this decision
                    history_slice = history[:hist_idx]

                    prompt = reconstruct_sm_prompt(
                        prompt_combo=prompt_combo,
                        bet_type=bet_type,
                        balance=balance,
                        history_slice=history_slice,
                    )

                    rounds.append(RoundRecord(
                        game_id=game_counter,
                        round_num=dec.get("round", i + 1),
                        prompt=prompt,
                        balance_before=balance,
                        choice=1 if action == "bet" else 2,
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

        return rounds


class MWAdapter(ParadigmAdapter):
    """Mystery Wheel V2role adapter. Uses saved full_prompt directly."""

    def __init__(self, config: dict):
        self.config = config

    def load_rounds(self) -> List[RoundRecord]:
        rounds = []
        # MW checkpoint file has game_ids that may not be unique across runs
        game_counter = 0

        for filename in self.config["files"]:
            filepath = os.path.join(self.config["data_dir"], filename)
            with open(filepath) as f:
                data = json.load(f)

            results = data.get("results", [])

            for game in results:
                game_counter += 1
                gid = game_counter
                decisions = game.get("decisions", [])
                game_outcome = game.get("final_outcome", "")
                if game.get("bankruptcy"):
                    game_outcome = "bankruptcy"
                elif game.get("stopped_voluntarily"):
                    game_outcome = "voluntary_stop"
                elif game.get("max_rounds_reached"):
                    game_outcome = "max_rounds"

                bet_constraint = str(game.get("bet_constraint", "unknown"))

                # Pre-compute last valid (non-skipped) decision index
                valid_indices = [j for j, dec in enumerate(decisions) if not dec.get("skipped")]
                last_valid_idx = valid_indices[-1] if valid_indices else -1

                for i, dec in enumerate(decisions):
                    if dec.get("skipped"):
                        continue
                    prompt = dec.get("full_prompt") or dec.get("actual_prompt", "")
                    if not prompt:
                        continue

                    rounds.append(RoundRecord(
                        game_id=gid,
                        round_num=dec.get("round", i + 1),
                        prompt=prompt,
                        balance_before=dec.get("balance_before", 100),
                        choice=dec.get("choice"),
                        bet_amount=dec.get("bet_amount"),
                        game_outcome=game_outcome,
                        bet_type=game.get("bet_type", "unknown"),
                        bet_constraint=bet_constraint,
                        prompt_condition=game.get("prompt_condition", "BASE"),
                        is_last_round=(i == last_valid_idx),
                        paradigm="mw",
                    ))

        return rounds


def get_adapter(paradigm: str) -> ParadigmAdapter:
    """Factory function to create the appropriate adapter."""
    config = PARADIGM_CONFIG[paradigm]
    adapters = {"ic": ICAdapter, "sm": SMAdapter, "mw": MWAdapter}
    return adapters[paradigm](config)


# ============================================================
# Logging
# ============================================================

def setup_logging(paradigm: str, output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extraction_{paradigm}_{timestamp}.log"

    logger = logging.getLogger(f"extraction_{paradigm}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================
# Sparse COO utilities
# ============================================================

def dense_to_sparse_coo(features: np.ndarray):
    """Convert dense (n_rounds, n_features) to sparse COO arrays.

    Returns:
        (row_indices, col_indices, values) — all 1D arrays
    """
    rows, cols = np.nonzero(features)
    values = features[rows, cols]
    return (
        rows.astype(np.int32),
        cols.astype(np.int32),
        values.astype(np.float32),
    )


def load_sparse_features(npz_path: str, dense: bool = True) -> dict:
    """Load sparse SAE features from NPZ.

    Args:
        npz_path: Path to sae_features_L{N}.npz
        dense: If True, reconstruct dense array. If False, return sparse arrays.

    Returns:
        dict with 'features' (dense or sparse components) and metadata arrays
    """
    data = np.load(npz_path, allow_pickle=False)
    result = {k: data[k] for k in data.files if k not in ("row_indices", "col_indices", "values", "shape")}

    if dense:
        shape = tuple(data["shape"])
        features = np.zeros(shape, dtype=np.float32)
        features[data["row_indices"], data["col_indices"]] = data["values"]
        result["features"] = features
    else:
        result["row_indices"] = data["row_indices"]
        result["col_indices"] = data["col_indices"]
        result["values"] = data["values"]
        result["shape"] = data["shape"]

    return result


# ============================================================
# Phase A: Hidden State Extraction
# ============================================================

def extract_hidden_states(
    rounds: List[RoundRecord],
    layers: List[int],
    device: str,
    logger: logging.Logger,
    checkpoint_dir: Optional[Path] = None,
) -> tuple:
    """Forward pass for all rounds, extract last-token hidden states.

    Args:
        rounds: List of RoundRecord with prompts
        layers: Layer indices to extract
        device: CUDA device string
        logger: Logger instance
        checkpoint_dir: If set, save/load checkpoint here

    Returns:
        (hidden_all, valid_mask): hidden_all is (n_rounds, n_layers, hidden_dim) float32,
                                  valid_mask is (n_rounds,) bool (False = extraction error)
    """
    n_rounds = len(rounds)
    n_layers = len(layers)

    # Check for checkpoint
    if checkpoint_dir:
        ckpt_file = checkpoint_dir / "phase_a_hidden_states.npz"
        meta_file = checkpoint_dir / "phase_a_metadata.json"
        if ckpt_file.exists() and meta_file.exists():
            logger.info(f"Loading Phase A checkpoint from {ckpt_file}")
            loaded = np.load(ckpt_file)
            hidden_all = loaded["hidden_states"]
            with open(meta_file) as f:
                meta = json.load(f)
            if meta["n_rounds"] == n_rounds and meta["n_layers"] == n_layers and not meta.get("partial"):
                logger.info(f"Checkpoint valid: {n_rounds} rounds, {n_layers} layers")
                vm = loaded["valid_mask"] if "valid_mask" in loaded else np.ones(n_rounds, dtype=bool)
                return hidden_all, vm
            elif meta.get("partial"):
                logger.warning("Checkpoint is partial (interrupted). Re-extracting from scratch.")
            else:
                logger.warning(
                    f"Checkpoint mismatch: expected {n_rounds}x{n_layers}, "
                    f"got {meta['n_rounds']}x{meta['n_layers']}. Re-extracting."
                )

    mem_gb = n_rounds * n_layers * HIDDEN_DIM * 4 / 1e9
    logger.info(f"Phase A: {n_rounds} rounds, {n_layers} layers, dim={HIDDEN_DIM}")
    logger.info(f"Memory estimate: {mem_gb:.2f} GB")

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
    hidden_all = np.zeros((n_rounds, n_layers, HIDDEN_DIM), dtype=np.float32)
    valid_mask = np.ones(n_rounds, dtype=bool)
    n_skipped = 0

    for i in tqdm(range(n_rounds), desc="Forward passes"):
        try:
            prompt = rounds[i].prompt

            # Apply chat template for Gemma
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
            # outputs.hidden_states[0] = embeddings, [L+1] = layer L output
            for j, layer in enumerate(layers):
                h = outputs.hidden_states[layer + 1][:, -1, :]
                hidden_all[i, j, :] = h.float().cpu().numpy().squeeze()

        except Exception as e:
            logger.error(f"Error at round {i} (game {rounds[i].game_id}, "
                         f"round {rounds[i].round_num}): {e}")
            valid_mask[i] = False
            n_skipped += 1

        # Incremental checkpoint every 5000 rounds
        if checkpoint_dir and i > 0 and i % 5000 == 0:
            _save_phase_a_checkpoint(
                checkpoint_dir, hidden_all, valid_mask, n_rounds, n_layers,
                layers, n_skipped, logger, partial=True
            )

    logger.info(f"Phase A complete: {n_rounds - n_skipped}/{n_rounds} valid ({n_skipped} skipped)")

    # Unload model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model unloaded")

    # Save final checkpoint
    if checkpoint_dir:
        _save_phase_a_checkpoint(
            checkpoint_dir, hidden_all, valid_mask, n_rounds, n_layers,
            layers, n_skipped, logger, partial=False
        )

    return hidden_all, valid_mask


def _save_phase_a_checkpoint(
    checkpoint_dir: Path, hidden_all: np.ndarray, valid_mask: np.ndarray,
    n_rounds: int, n_layers: int, layers: list, n_skipped: int,
    logger: logging.Logger, partial: bool = False,
):
    """Save Phase A checkpoint (hidden states + valid_mask + metadata)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoint_dir / "phase_a_hidden_states.npz"
    label = "partial" if partial else "final"
    logger.info(f"Saving Phase A {label} checkpoint to {ckpt_file}")
    np.savez_compressed(ckpt_file, hidden_states=hidden_all, valid_mask=valid_mask)
    meta = {
        "n_rounds": n_rounds,
        "n_layers": n_layers,
        "layers": layers,
        "hidden_dim": HIDDEN_DIM,
        "n_skipped": n_skipped,
        "partial": partial,
        "timestamp": datetime.now().isoformat(),
    }
    with open(checkpoint_dir / "phase_a_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Phase B: SAE Encoding + Sparse Save
# ============================================================

def encode_and_save_layer(
    hidden_all: np.ndarray,
    layer_idx_in_array: int,
    layer: int,
    rounds: List[RoundRecord],
    output_dir: Path,
    device: str,
    logger: logging.Logger,
    save_hidden: bool = False,
    valid_mask: Optional[np.ndarray] = None,
):
    """Encode one layer's hidden states through GemmaScope SAE and save as sparse COO.

    Args:
        hidden_all: (n_rounds, n_layers, hidden_dim) array
        layer_idx_in_array: Index into the n_layers dimension
        layer: Actual layer number (0-41)
        rounds: RoundRecord list for metadata
        output_dir: Where to save NPZ files
        device: CUDA device
        logger: Logger
        save_hidden: If True, also save raw hidden states
    """
    n_rounds = len(rounds)
    output_dir.mkdir(parents=True, exist_ok=True)

    sae_file = output_dir / f"sae_features_L{layer}.npz"
    if sae_file.exists():
        logger.info(f"Layer {layer}: already exists, skipping")
        return

    logger.info(f"Encoding layer {layer}...")

    # Extract this layer's hidden states
    h_layer = hidden_all[:, layer_idx_in_array, :]  # (n_rounds, hidden_dim)
    h_tensor = torch.tensor(h_layer, dtype=torch.float32, device=device)

    # Load GemmaScope SAE
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained(
            release="gemma-scope-9b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_131k/canonical",
            device=device,
        )
    except Exception as e:
        logger.error(f"Failed to load SAE for layer {layer}: {e}")
        del h_tensor
        torch.cuda.empty_cache()
        return

    # Batch encode
    chunk_size = 256
    all_features = []
    for start in range(0, n_rounds, chunk_size):
        end = min(start + chunk_size, n_rounds)
        with torch.no_grad():
            feats = sae.encode(h_tensor[start:end])
        all_features.append(feats.cpu().numpy())

    del sae, h_tensor
    torch.cuda.empty_cache()
    gc.collect()

    features_dense = np.concatenate(all_features, axis=0)  # (n_rounds, 131072)
    del all_features

    # Convert to sparse COO
    row_indices, col_indices, values = dense_to_sparse_coo(features_dense)
    nnz = len(values)
    avg_nnz = nnz / n_rounds if n_rounds > 0 else 0

    logger.info(
        f"Layer {layer}: {n_rounds} rounds, {nnz} non-zero "
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

    # Save sparse SAE features
    # valid_mask: False = extraction error (zero hidden state), True = valid
    if valid_mask is None:
        valid_mask_arr = np.ones(n_rounds, dtype=bool)
    else:
        valid_mask_arr = valid_mask

    np.savez_compressed(
        sae_file,
        # Sparse COO
        row_indices=row_indices,
        col_indices=col_indices,
        values=values,
        shape=np.array([n_rounds, N_SAE_FEATURES], dtype=np.int64),
        # Metadata
        game_ids=game_ids,
        round_nums=round_nums,
        game_outcomes=game_outcomes,
        bet_types=bet_types,
        bet_constraints=bet_constraints,
        prompt_conditions=prompt_conditions,
        balances=balances,
        is_last_round=is_last_round,
        valid_mask=valid_mask_arr,
        paradigms=paradigms,
        layer=layer,
    )

    # Save metadata JSON sidecar
    meta = {
        "layer": layer,
        "n_rounds": n_rounds,
        "n_features": N_SAE_FEATURES,
        "nnz_total": int(nnz),
        "avg_nnz_per_round": float(avg_nnz),
        "sparsity": float(1 - avg_nnz / N_SAE_FEATURES),
        "n_games": len(set(r.game_id for r in rounds)),
        "n_bankruptcy": sum(1 for r in rounds if r.game_outcome == "bankruptcy" and r.is_last_round),
        "storage_format": "sparse_coo",
        "timestamp": datetime.now().isoformat(),
    }
    with open(sae_file.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Optionally save raw hidden states
    if save_hidden:
        hs_file = output_dir / f"hidden_states_L{layer}.npz"
        if not hs_file.exists():
            np.savez_compressed(
                hs_file,
                hidden_states=h_layer,
                game_ids=game_ids,
                round_nums=round_nums,
                is_last_round=is_last_round,
            )
            logger.info(f"Layer {layer}: saved hidden states {h_layer.shape}")

    del features_dense, row_indices, col_indices, values
    gc.collect()


# ============================================================
# Main Pipeline
# ============================================================

def run(paradigm: str, device: str, layers: List[int],
        save_hidden: bool = True, resume: bool = False):
    """Run full extraction pipeline for one paradigm."""

    config = PARADIGM_CONFIG[paradigm]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"

    logger = setup_logging(paradigm, output_dir)

    logger.info("=" * 70)
    logger.info(f"EXTRACTION: {config['name']}")
    logger.info(f"Device: {device}")
    logger.info(f"Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    logger.info(f"Save hidden states: {save_hidden}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    # Step 1: Load rounds
    adapter = get_adapter(paradigm)
    rounds = adapter.load_rounds()
    n_bk = sum(1 for r in rounds if r.game_outcome == "bankruptcy" and r.is_last_round)
    n_games = len(set(r.game_id for r in rounds))
    logger.info(f"Loaded {len(rounds)} rounds from {n_games} games ({n_bk} bankruptcy)")

    # Step 2: Phase A — Forward passes
    # resume=True: load existing Phase A checkpoint if available, skip completed Phase B layers
    # resume=False: always run fresh (ignore existing checkpoints/outputs)
    if not resume:
        # Clean up any stale checkpoint/output from previous runs
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info("Cleared previous checkpoint directory")
        # Remove existing output files so Phase B doesn't skip them
        for existing_npz in output_dir.glob("sae_features_L*.npz"):
            existing_npz.unlink()
            existing_npz.with_suffix(".json").unlink(missing_ok=True)
        for existing_hs in output_dir.glob("hidden_states_L*.npz"):
            existing_hs.unlink()
        logger.info("Cleared previous output files for fresh extraction")

    t0 = datetime.now()
    hidden_all, valid_mask = extract_hidden_states(
        rounds, layers, device, logger,
        checkpoint_dir=checkpoint_dir,
    )
    t_forward = (datetime.now() - t0).total_seconds()
    logger.info(f"Phase A: {t_forward:.0f}s ({t_forward / 60:.1f} min)")
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        logger.warning(f"Phase A: {n_invalid} rounds failed extraction (zeros in hidden_all)")

    # Step 3: Phase B — SAE encoding per layer
    t0 = datetime.now()

    # Track progress (only load if resuming)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_file = checkpoint_dir / "phase_b_progress.json"
    if resume and progress_file.exists():
        try:
            with open(progress_file) as f:
                progress = json.load(f)
            completed = set(progress.get("completed_layers", []))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupted progress file, starting Phase B fresh: {e}")
            completed = set()
    else:
        completed = set()

    remaining = [l for l in layers if l not in completed]
    logger.info(f"Phase B: {len(remaining)} layers remaining (of {len(layers)})")

    for i, layer in enumerate(remaining):
        layer_idx = layers.index(layer)
        logger.info(f"[{i + 1}/{len(remaining)}] Layer {layer}")
        encode_and_save_layer(
            hidden_all, layer_idx, layer, rounds, output_dir, device, logger,
            save_hidden=save_hidden,
            valid_mask=valid_mask,
        )
        sae_file = output_dir / f"sae_features_L{layer}.npz"
        if sae_file.exists():
            completed.add(layer)
            with open(progress_file, "w") as f:
                json.dump({"completed_layers": sorted(completed)}, f)
        else:
            logger.warning(f"Layer {layer}: output not created, NOT marking completed")

    t_encode = (datetime.now() - t0).total_seconds()
    logger.info(f"Phase B: {t_encode:.0f}s ({t_encode / 60:.1f} min)")

    # Step 4: Save extraction summary
    summary = {
        "paradigm": paradigm,
        "name": config["name"],
        "model": MODEL_NAME,
        "n_rounds": len(rounds),
        "n_games": n_games,
        "n_bankruptcy": n_bk,
        "n_layers": len(layers),
        "layers": layers,
        "hidden_dim": HIDDEN_DIM,
        "n_sae_features": N_SAE_FEATURES,
        "storage_format": "sparse_coo",
        "save_hidden_states": save_hidden,
        "device": device,
        "phase_a_seconds": t_forward,
        "phase_b_seconds": t_encode,
        "total_seconds": t_forward + t_encode,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"COMPLETE: {t_forward + t_encode:.0f}s ({(t_forward + t_encode) / 60:.1f} min)")
    logger.info("=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All-Rounds Extraction: Hidden States + Sparse SAE")
    parser.add_argument("--paradigm", type=str, required=True, choices=["ic", "sm", "mw"],
                        help="Paradigm: ic (Investment Choice), sm (Slot Machine), mw (Mystery Wheel)")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to extract: 'all' for 0-41, or comma-separated like '18,26,30'")
    parser.add_argument("--no-save-hidden", action="store_true",
                        help="Skip saving raw hidden states (saves ~10GB per paradigm)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from Phase A checkpoint if available")
    args = parser.parse_args()

    # Parse and validate layers
    if args.layers == "all":
        layers = list(range(N_LAYERS))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]
        invalid = [l for l in layers if l < 0 or l >= N_LAYERS]
        if invalid:
            parser.error(f"Invalid layer(s) {invalid}. Must be 0-{N_LAYERS - 1}.")

    run(
        paradigm=args.paradigm,
        device=args.device,
        layers=layers,
        save_hidden=not args.no_save_hidden,
        resume=args.resume,
    )
