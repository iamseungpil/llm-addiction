#!/usr/bin/env python3
"""
FIXED VERSION: Re-extract SAE feature activation means for ALL features
using Experiment 1 (6,400 multiround games) prompts.

Key fixes:
1. ✅ Added SAE encoding (hidden states → SAE features)
2. ✅ Output format matches pathway analysis requirements
3. ✅ Checkpoint support for crash recovery
4. ✅ Test mode for quick validation

Runtime: ~2-3 hours on A100 for full run
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add path for SAE loader
sys.path.append("/home/ubuntu/llm_addiction/causal_feature_discovery/src")
from llama_scope_working import LlamaScopeWorking


def load_experiments() -> List[Dict]:
    """Load the two Experiment 1 result files (main + missing)."""
    results_dir = Path("/data/llm_addiction/results")
    main_file = results_dir / "exp1_multiround_intermediate_20250819_140040.json"
    missing_file = results_dir / "exp1_missing_complete_20250820_090040.json"

    all_experiments: List[Dict] = []
    for file_path in (main_file, missing_file):
        print(f"Loading {file_path}...")
        with open(file_path, "r") as f:
            data = json.load(f)

        experiments = data.get("results", []) if isinstance(data, dict) else []
        all_experiments.extend(experiments)
        print(f"  Added {len(experiments)} rows")
        del data, experiments
        gc.collect()

    print(f"Total experiments loaded: {len(all_experiments)}")
    return all_experiments


def load_causal_features() -> List[Dict]:
    """Load reparsed causal features list."""
    causal_file = Path(
        "/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_list_REPARSED.json"
    )
    print(f"Loading causal features from {causal_file}...")
    with open(causal_file, "r") as f:
        data = json.load(f)
    features = data["features"]
    print(f"Loaded {len(features)} causal features")
    return features


def load_model():
    """Load Llama-3.1-8B."""
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    print("✅ Model loaded")
    return model, tokenizer


def load_saes(target_layers: List[int]) -> Dict[int, LlamaScopeWorking]:
    """Load SAE models for all target layers."""
    print(f"Loading SAEs for {len(target_layers)} layers...")
    saes = {}
    for layer in tqdm(target_layers, desc="Loading SAEs"):
        try:
            sae = LlamaScopeWorking(layer=layer, device="cuda")
            saes[layer] = sae
        except Exception as e:
            print(f"⚠️  Failed to load SAE for layer {layer}: {e}")
            continue
    print(f"✅ Loaded {len(saes)} SAEs")
    return saes


def extract_sae_features(
    model, tokenizer, saes: Dict[int, LlamaScopeWorking], prompt: str, target_layers: List[int]
) -> Dict[int, np.ndarray]:
    """Run one prompt and return SAE feature activations for target layers.

    Returns:
        Dict mapping layer -> SAE feature vector [32,768]
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    features: Dict[int, np.ndarray] = {}
    for layer in target_layers:
        if layer not in saes:
            continue
        if layer >= len(hidden_states):
            continue

        # Get hidden state
        layer_hidden = hidden_states[layer][0, -1, :]  # [4,096]

        # ✅ CRITICAL FIX: Encode through SAE
        sae = saes[layer]
        with torch.no_grad():
            # Encode to get feature activations
            feature_acts = sae.encode(layer_hidden.unsqueeze(0).float())  # [1, 32768]
            features[layer] = feature_acts[0].cpu().numpy()  # [32768]

    return features


def accumulate_means(
    sums: Dict[int, np.ndarray],
    counts: Dict[int, int],
    features: Dict[int, np.ndarray],
):
    """Update running sums/counts in-place."""
    for layer, vec in features.items():
        if layer not in sums:
            sums[layer] = np.zeros_like(vec, dtype=np.float64)
            counts[layer] = 0
        sums[layer] += vec.astype(np.float64)
        counts[layer] += 1


def save_checkpoint(
    checkpoint_path: Path,
    safe_sums: Dict[int, np.ndarray],
    safe_counts: Dict[int, int],
    risky_sums: Dict[int, np.ndarray],
    risky_counts: Dict[int, int],
    processed: int,
):
    """Save checkpoint for crash recovery."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "processed": processed,
        "safe_sums": {k: v.tolist() for k, v in safe_sums.items()},
        "safe_counts": safe_counts,
        "risky_sums": {k: v.tolist() for k, v in risky_sums.items()},
        "risky_counts": risky_counts,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)
    print(f"💾 Checkpoint saved: {processed} experiments processed")


def load_checkpoint(checkpoint_path: Path):
    """Load checkpoint if exists."""
    if not checkpoint_path.exists():
        return None, None, None, None, 0

    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "r") as f:
        checkpoint = json.load(f)

    safe_sums = {int(k): np.array(v, dtype=np.float64) for k, v in checkpoint["safe_sums"].items()}
    risky_sums = {int(k): np.array(v, dtype=np.float64) for k, v in checkpoint["risky_sums"].items()}
    safe_counts = {int(k): v for k, v in checkpoint["safe_counts"].items()}
    risky_counts = {int(k): v for k, v in checkpoint["risky_counts"].items()}
    processed = checkpoint["processed"]

    print(f"✅ Checkpoint loaded: {processed} experiments already processed")
    return safe_sums, safe_counts, risky_sums, risky_counts, processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of experiments (for testing)")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="Save checkpoint every N experiments")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    target_layers = list(range(1, 32))  # layers 1-31
    causal_features = load_causal_features()

    # Load checkpoint if resuming
    checkpoint_path = Path("/tmp/feature_means_checkpoint.json")
    if args.resume:
        safe_sums, safe_counts, risky_sums, risky_counts, start_idx = load_checkpoint(checkpoint_path)
    else:
        safe_sums, safe_counts, risky_sums, risky_counts, start_idx = {}, {}, {}, {}, 0

    experiments = load_experiments()
    if args.limit:
        experiments = experiments[: args.limit]
        print(f"⚠️  TEST MODE: Limited to {args.limit} experiments")

    model, tokenizer = load_model()
    saes = load_saes(target_layers)

    processed = start_idx
    skipped = 0

    for idx, exp in enumerate(tqdm(experiments[start_idx:], desc="Processing experiments", initial=start_idx)):
        try:
            is_bankrupt = exp.get("is_bankrupt", False)
            rounds = exp.get("round_features", [])
            if not rounds:
                skipped += 1
                continue

            prompt = rounds[-1].get("prompt", "")
            if not prompt:
                skipped += 1
                continue

            # Extract SAE features
            feats = extract_sae_features(model, tokenizer, saes, prompt, target_layers)

            # Accumulate
            if is_bankrupt:
                accumulate_means(risky_sums, risky_counts, feats)
            else:
                accumulate_means(safe_sums, safe_counts, feats)

            processed += 1

            # Periodic cleanup
            if processed % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Checkpoint
            if processed % args.checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_path, safe_sums, safe_counts, risky_sums, risky_counts, processed
                )

        except Exception as exc:
            print(f"⚠️  Error processing experiment {idx}: {exc}")
            skipped += 1
            continue

    print(f"\n✅ Processed {processed} experiments (skipped {skipped})")
    print(f"Safe experiments per layer: ~{safe_counts.get(1, 0)}")
    print(f"Risky experiments per layer: ~{risky_counts.get(1, 0)}")

    # Compute means
    print("\nComputing feature means...")
    feature_means = {}

    for feat in tqdm(causal_features, desc="Extracting feature means"):
        layer = feat["layer"]
        feature_id = feat["feature_id"]
        feature_name = f"L{layer}-{feature_id}"

        if layer not in safe_sums or layer not in risky_sums:
            continue

        safe_count = safe_counts.get(layer, 0)
        risky_count = risky_counts.get(layer, 0)

        if safe_count == 0 or risky_count == 0:
            continue

        # Extract single feature value from accumulated arrays
        safe_mean_array = safe_sums[layer] / safe_count
        risky_mean_array = risky_sums[layer] / risky_count

        if feature_id < len(safe_mean_array):
            feature_means[feature_name] = {
                "safe_mean": float(safe_mean_array[feature_id]),
                "risky_mean": float(risky_mean_array[feature_id]),
            }

    print(f"\n✅ Extracted means for {len(feature_means)} features")

    # Save in pathway analysis format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis")
        / f"feature_means_lookup_REPARSED_{timestamp}.json"
    )

    output = {
        "timestamp": timestamp,
        "source": "Experiment 1 full re-extraction with SAE encoding (FIXED)",
        "total_experiments_processed": processed,
        "total_features": len(feature_means),
        "missing_features": [
            f"{f['layer']}-{f['feature_id']}" for f in causal_features if f"L{f['layer']}-{f['feature_id']}" not in feature_means
        ],
        "feature_means": feature_means,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved to: {output_path}")
    print(f"📊 Coverage: {len(feature_means)}/{len(causal_features)} features ({100*len(feature_means)/len(causal_features):.1f}%)")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("🗑️  Checkpoint cleaned up")


if __name__ == "__main__":
    main()
