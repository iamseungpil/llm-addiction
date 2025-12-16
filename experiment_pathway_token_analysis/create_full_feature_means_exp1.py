#!/usr/bin/env python3
"""
Re-extract activation means for ALL features (no significance filter)
using Experiment 1 (6,400 multiround games) prompts.

Why:
- Reparsing in Experiment 2 surfaced many causal features not present in the
  significance-filtered Experiment 1 outputs. Pathway analysis needs means
  for every feature, not only the previously "significant" ones.

What this script does:
1) Load all 6,400 Experiment 1 game transcripts (safe vs bankrupt labels)
2) Re-run Llama-3.1-8B to collect last-token hidden states for layers 1–31
3) Compute per-layer per-feature means separately for safe and bankrupt games
4) Save a lookup JSON covering all 31 × 4,096 features (no filtering)

Output:
  /data/llm_addiction/experiment_1_L1_31_extraction/L1_31_feature_means_FULL_<timestamp>.json

Runtime/Resources:
- Heavy: runs the base model over 6,400 prompts, ~2–3 hours on one A100 (estimate).
- GPU memory: fits easily (single forward per prompt), but uses many forwards.
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_model():
    """Load Llama-3.1-8B with hidden states."""
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    print("✅ Model loaded")
    return model, tokenizer


def extract_layer_hidden(
    model, tokenizer, prompt: str, target_layers: Iterable[int]
) -> Dict[int, np.ndarray]:
    """Run one prompt and return last-token hidden states for target layers."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    features: Dict[int, np.ndarray] = {}
    for layer in target_layers:
        if layer < len(hidden_states):
            # Hidden states are 0-indexed; layer numbering follows same indexing here.
            layer_hidden = hidden_states[layer][0, -1, :]  # shape [hidden_size]
            features[layer] = layer_hidden.cpu().numpy()
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


def compute_means_for_group(
    sums: Dict[int, np.ndarray],
    counts: Dict[int, int],
) -> Dict[str, List[float]]:
    """Turn sums/counts into mean lists."""
    means: Dict[str, List[float]] = {}
    for layer, sum_vec in sums.items():
        count = counts.get(layer, 0)
        if count == 0:
            continue
        means[f"L{layer}"] = (sum_vec / count).tolist()
    return means


def main():
    target_layers = list(range(1, 32))  # layers 1–31 inclusive

    experiments = load_experiments()
    model, tokenizer = load_model()

    safe_sums: Dict[int, np.ndarray] = {}
    risky_sums: Dict[int, np.ndarray] = {}
    safe_counts: Dict[int, int] = {}
    risky_counts: Dict[int, int] = {}

    processed = 0
    for exp in tqdm(experiments, desc="Processing experiments"):
        try:
            is_bankrupt = exp.get("is_bankrupt", False)
            rounds = exp.get("round_features", [])
            if not rounds:
                continue
            prompt = rounds[-1].get("prompt", "")
            if not prompt:
                continue

            feats = extract_layer_hidden(model, tokenizer, prompt, target_layers)

            if is_bankrupt:
                accumulate_means(risky_sums, risky_counts, feats)
            else:
                accumulate_means(safe_sums, safe_counts, feats)

            processed += 1
            if processed % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as exc:  # pragma: no cover - best-effort loop
            print(f"⚠️  Error processing experiment: {exc}")
            continue

    print(f"\nProcessed {processed} experiments")
    print(f"Safe counts (per layer): {sum(safe_counts.values())}")
    print(f"Risky counts (per layer): {sum(risky_counts.values())}")

    safe_means = compute_means_for_group(safe_sums, safe_counts)
    risky_means = compute_means_for_group(risky_sums, risky_counts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path("/data/llm_addiction/experiment_1_L1_31_extraction")
        / f"L1_31_feature_means_FULL_{timestamp}.json"
    )

    output = {
        "timestamp": timestamp,
        "source": "Experiment 1 multiround (exp1_multiround_intermediate_20250819_140040.json + missing)",
        "total_experiments_processed": processed,
        "layers": {},
    }

    # Merge safe/risky into layer structure
    for layer in target_layers:
        layer_key = f"L{layer}"
        if layer_key in safe_means or layer_key in risky_means:
            output["layers"][layer_key] = {
                "safe_count": safe_counts.get(layer, 0),
                "risky_count": risky_counts.get(layer, 0),
                "safe_mean": safe_means.get(layer_key, []),
                "risky_mean": risky_means.get(layer_key, []),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved full feature means to: {output_path}")
    print("Use this file to refresh feature_means_lookup for pathway analysis.")


if __name__ == "__main__":
    # Ensure the causal_feature_discovery path is available if needed later.
    sys.path.append("/home/ubuntu/llm_addiction/causal_feature_discovery/src")
    main()
