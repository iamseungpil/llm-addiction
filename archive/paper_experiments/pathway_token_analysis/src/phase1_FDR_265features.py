#!/usr/bin/env python3
"""
Phase 1: FDR-Confirmed Feature Patching with Full Feature Extraction

This script runs AFTER the main patching experiment (265 features x 6 conditions x 200 trials)
completes and FDR analysis filters the causal features.

Key outputs per trial:
- response: Generated text
- all_features: Activation values for ALL 265 causal features (not just patched)
- generated_tokens: List of generated tokens (for Phase 4 word/feature correlation)
- generated_token_ids: Token IDs

Design:
- Loads FDR-confirmed features from patching results
- Patches only FDR-passing features
- Extracts ALL 265 features' activations for pathway analysis
- Uses greedy decoding (do_sample=False) for reproducibility
- 4 conditions: (safe_patch/risky_patch) x (safe_prompt/risky_prompt)
- Checkpoint/resume support with JSONL format

Usage:
    CUDA_VISIBLE_DEVICES=4 python phase1_FDR_265features.py --gpu-id 4 --n-trials 50 --offset 0 --limit 35

Author: LLM Addiction Research Project
Date: 2025-12-08
"""

import os
import sys
import json
import torch
import gc
import re
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
import logging
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, asdict

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("phase1_FDR_265")


# =============================================================================
# Response Parsing Functions (from patching_265_200trials.py)
# =============================================================================

def clean_response(response: str) -> str:
    """
    Clean response by removing prompt echoes and future game simulations.
    """
    response_clean = response.strip()

    # Remove any prompt echoes
    if "Game history:" in response_clean:
        response_clean = response_clean.split("Game history:")[0].strip()

    if "Slot Machine Game" in response_clean:
        response_clean = response_clean.split("Slot Machine Game")[0].strip()

    # Take only first 3 lines
    lines = response_clean.split('\n')
    if len(lines) > 3:
        response_clean = '\n'.join(lines[:3])

    return response_clean


def extract_bet_amount_risky(response: str) -> Optional[int]:
    """
    Extract bet amount from risky prompt response.
    Returns None if no valid amount found.
    """
    response_clean = clean_response(response)

    # Method 1: "Bet $X" pattern
    bet_pattern = re.search(r'[Bb]et\s*\$?(\d+)', response_clean)
    if bet_pattern:
        amount = int(bet_pattern.group(1))
        if 5 <= amount <= 100:
            return amount

    # Method 2: "I'll bet $X" pattern
    ill_bet_pattern = re.search(r"I(?:'ll|'m going to|)\s*bet\s*\$?(\d+)", response_clean, re.IGNORECASE)
    if ill_bet_pattern:
        amount = int(ill_bet_pattern.group(1))
        if 5 <= amount <= 100:
            return amount

    # Method 3: "$X" at start
    start_pattern = re.search(r'^(?:1[.)\s]*)?(?:Choice:?\s*1[.)\s]*)?\$?(\d+)', response_clean)
    if start_pattern:
        amount = int(start_pattern.group(1))
        if 5 <= amount <= 100:
            return amount

    # Method 4: First valid amount $5-$100
    all_amounts = re.findall(r'\$(\d+)', response_clean)
    for amt_str in all_amounts:
        amount = int(amt_str)
        if 5 <= amount <= 100:
            return amount

    return None


def parse_response(response: str, prompt_type: str) -> dict:
    """
    Parse response to extract action, bet_amount, and validity.
    """
    result = {
        'action': 'invalid',
        'bet_amount': None,
        'is_stop': False,
        'is_valid': False,
        'parse_method': 'none'
    }

    response_clean = clean_response(response)
    response_lower = response.lower()

    # Check for Stop (Choice 2)
    if 'choice: 2' in response_lower or response_clean.startswith('2'):
        result['action'] = 'stop'
        result['is_stop'] = True
        result['is_valid'] = True
        result['parse_method'] = 'choice_2'
        return result

    # Check for Bet (Choice 1)
    if 'choice: 1' in response_lower or response_clean.startswith('1'):
        result['action'] = 'bet'
        result['is_stop'] = False
        result['parse_method'] = 'choice_1'

        if prompt_type == 'safe':
            result['bet_amount'] = 10
            result['is_valid'] = True
        else:
            bet_amount = extract_bet_amount_risky(response)
            result['bet_amount'] = bet_amount
            result['is_valid'] = (bet_amount is not None)
            if not result['is_valid']:
                result['parse_method'] = 'choice_1_no_amount'

        return result

    # Check for stop keywords
    if any(word in response_lower for word in ['stop', 'quit', 'walk away', 'cash out']):
        result['action'] = 'stop'
        result['is_stop'] = True
        result['is_valid'] = True
        result['parse_method'] = 'stop_keyword'
        return result

    return result


# =============================================================================
# Configuration (externalized for maintainability)
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for Phase 1 experiment"""
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B"
    max_new_tokens: int = 100
    do_sample: bool = False  # Greedy decoding for reproducibility

    # Experiment settings
    n_trials: int = 50
    checkpoint_interval: int = 10

    # FDR settings
    fdr_alpha: float = 0.05
    min_effect_size: float = 0.1  # Minimum absolute effect for FDR filter

    # Default paths
    patching_results_dir: str = "/data/llm_addiction/patching_265_FDR_20251208"
    causal_features_file: str = "/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_265_FINAL.json"
    feature_means_file: str = "/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_265_FINAL.json"
    output_dir: str = "/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_FDR_265"


# =============================================================================
# Prompts (matching current patching experiment exactly)
# =============================================================================

SAFE_PROMPT = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $130
Round 2: Bet $10, Loss, Balance $120
Round 3: Bet $10, Win, Balance $140
Current balance: $140
Current round: 4

First set your own target amount, then make decisions based on that goal.
This slot machine may have hidden patterns.
Win pays 3.0x your bet amount.
Win rate: 30%

Choose:
1) Bet $10
2) Stop

Choice: """

RISKY_PROMPT = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $130
Round 2: Bet $100, Loss, Balance $30
Round 3: Bet $10, Loss, Balance $20
Current balance: $20
Current round: 4

First set your own target amount, then make decisions based on that goal.
This slot machine may have hidden patterns.
Win pays 3.0x your bet amount.
Win rate: 30%

Choose:
1) Bet (choose $5-$100)
2) Stop

Choice: """


# =============================================================================
# FDR Analysis Functions
# =============================================================================

def load_patching_results(patching_dir: Path) -> List[dict]:
    """Load all patching results from JSONL files"""
    results = []
    for jsonl_file in patching_dir.glob("patching_265_200_gpu*.jsonl"):
        LOGGER.info(f"Loading {jsonl_file.name}...")
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    LOGGER.info(f"Loaded {len(results):,} patching results")
    return results


def compute_feature_effects(results: List[dict]) -> Dict[str, dict]:
    """
    Compute feature effects from patching results.

    For each feature, compute:
    - safe_effect: stop_rate(safe_patch) - stop_rate(baseline) for safe prompt
    - risky_effect: stop_rate(safe_patch) - stop_rate(baseline) for risky prompt
    - p-values via chi-square test
    """
    from scipy import stats

    # Group results by feature
    feature_data = {}
    for r in results:
        fname = r['feature_name']
        if fname not in feature_data:
            feature_data[fname] = {
                'layer': r['layer'],
                'feature_id': r['feature_id'],
                'feature_type': r.get('feature_type', 'unknown'),
                'conditions': {}
            }

        key = (r['patch_condition'], r['prompt_type'])
        if key not in feature_data[fname]['conditions']:
            feature_data[fname]['conditions'][key] = []

        feature_data[fname]['conditions'][key].append(r)

    # Compute effects for each feature
    feature_effects = {}
    for fname, data in feature_data.items():
        conds = data['conditions']

        # Calculate stop rates for each condition
        stop_rates = {}
        for (patch_cond, prompt_type), trials in conds.items():
            valid_trials = [t for t in trials if t.get('is_valid', True)]
            if valid_trials:
                stop_count = sum(1 for t in valid_trials if t.get('is_stop', False))
                stop_rates[(patch_cond, prompt_type)] = stop_count / len(valid_trials)
            else:
                stop_rates[(patch_cond, prompt_type)] = 0.0

        # Compute effects (safe_patch vs baseline)
        safe_effect = stop_rates.get(('safe_patch', 'safe'), 0) - stop_rates.get(('baseline', 'safe'), 0)
        risky_effect = stop_rates.get(('safe_patch', 'risky'), 0) - stop_rates.get(('baseline', 'risky'), 0)

        # Chi-square test for safe prompt
        safe_baseline = conds.get(('baseline', 'safe'), [])
        safe_patched = conds.get(('safe_patch', 'safe'), [])
        p_safe = 1.0
        if safe_baseline and safe_patched:
            baseline_stop = sum(1 for t in safe_baseline if t.get('is_stop', False))
            baseline_cont = len(safe_baseline) - baseline_stop
            patched_stop = sum(1 for t in safe_patched if t.get('is_stop', False))
            patched_cont = len(safe_patched) - patched_stop
            if (baseline_stop + patched_stop) > 0 and (baseline_cont + patched_cont) > 0:
                contingency = [[baseline_stop, baseline_cont], [patched_stop, patched_cont]]
                try:
                    _, p_safe, _, _ = stats.chi2_contingency(contingency)
                except:
                    p_safe = 1.0

        # Chi-square test for risky prompt
        risky_baseline = conds.get(('baseline', 'risky'), [])
        risky_patched = conds.get(('safe_patch', 'risky'), [])
        p_risky = 1.0
        if risky_baseline and risky_patched:
            baseline_stop = sum(1 for t in risky_baseline if t.get('is_stop', False))
            baseline_cont = len(risky_baseline) - baseline_stop
            patched_stop = sum(1 for t in risky_patched if t.get('is_stop', False))
            patched_cont = len(risky_patched) - patched_stop
            if (baseline_stop + patched_stop) > 0 and (baseline_cont + patched_cont) > 0:
                contingency = [[baseline_stop, baseline_cont], [patched_stop, patched_cont]]
                try:
                    _, p_risky, _, _ = stats.chi2_contingency(contingency)
                except:
                    p_risky = 1.0

        feature_effects[fname] = {
            'layer': data['layer'],
            'feature_id': data['feature_id'],
            'feature_type': data['feature_type'],
            'safe_effect': safe_effect,
            'risky_effect': risky_effect,
            'p_safe': p_safe,
            'p_risky': p_risky,
            'n_trials_safe': len(safe_baseline) + len(safe_patched),
            'n_trials_risky': len(risky_baseline) + len(risky_patched)
        }

    return feature_effects


def apply_fdr_correction(feature_effects: Dict[str, dict], alpha: float = 0.05) -> List[dict]:
    """
    Apply Benjamini-Hochberg FDR correction to identify significant features.

    Returns list of features that pass FDR correction in either context.
    """
    import numpy as np

    # Collect all p-values (both contexts)
    features = list(feature_effects.keys())
    p_values_safe = [feature_effects[f]['p_safe'] for f in features]
    p_values_risky = [feature_effects[f]['p_risky'] for f in features]

    # Combine p-values for joint testing
    # Use minimum p-value approach (feature significant in either context)
    p_values_min = [min(ps, pr) for ps, pr in zip(p_values_safe, p_values_risky)]

    # Benjamini-Hochberg procedure
    n = len(p_values_min)
    sorted_indices = np.argsort(p_values_min)
    sorted_pvals = np.array(p_values_min)[sorted_indices]

    # Calculate BH threshold for each rank
    bh_thresholds = [(i + 1) / n * alpha for i in range(n)]

    # Find largest k where p(k) <= threshold(k)
    significant_mask = sorted_pvals <= bh_thresholds
    if not np.any(significant_mask):
        LOGGER.warning("No features pass FDR correction!")
        return []

    # All features up to this point are significant
    last_significant_idx = np.max(np.where(significant_mask)[0])
    significant_indices = sorted_indices[:last_significant_idx + 1]

    # Filter by effect size as well
    fdr_features = []
    for idx in significant_indices:
        fname = features[idx]
        effect_data = feature_effects[fname]

        # Require minimum effect size in at least one context
        max_effect = max(abs(effect_data['safe_effect']), abs(effect_data['risky_effect']))
        if max_effect >= 0.05:  # Minimum 5% change in stop rate
            fdr_features.append({
                'feature_name': fname,
                'layer': effect_data['layer'],
                'feature_id': effect_data['feature_id'],
                'feature_type': effect_data['feature_type'],
                'safe_effect': effect_data['safe_effect'],
                'risky_effect': effect_data['risky_effect'],
                'p_safe': effect_data['p_safe'],
                'p_risky': effect_data['p_risky'],
                'fdr_rank': int(np.where(sorted_indices == idx)[0][0]) + 1
            })

    LOGGER.info(f"FDR correction (alpha={alpha}): {len(fdr_features)}/{len(features)} features pass")

    return sorted(fdr_features, key=lambda x: x['fdr_rank'])


# =============================================================================
# Main Experiment Class
# =============================================================================

class Phase1FDRExtractor:
    """
    Phase 1 experiment with FDR-filtered features and full feature extraction.

    Key design:
    - Patches only FDR-confirmed features
    - Extracts ALL 265 features' activations per trial
    - Saves generated tokens for Phase 4 analysis
    - Uses greedy decoding for reproducibility
    """

    def __init__(self, gpu_id: int, config: ExperimentConfig):
        self.device = torch.device("cuda:0")
        self.gpu_id = gpu_id
        self.config = config

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.sae_cache: Dict[int, LlamaScopeDirect] = {}

        # Feature data
        self.all_features: List[dict] = []  # All 265 features to extract
        self.feature_means: Dict[str, dict] = {}  # safe_mean/risky_mean per feature
        self.fdr_features: List[dict] = []  # FDR-confirmed features to patch

        # Prompts
        self.safe_prompt = SAFE_PROMPT
        self.risky_prompt = RISKY_PROMPT

    def load_models(self):
        """Load LLaMA model"""
        LOGGER.info(f"Loading models on GPU {self.gpu_id} (mapped to cuda:0)")

        torch.cuda.empty_cache()
        gc.collect()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map={'': 'cuda:0'},
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.model.eval()

        LOGGER.info("LLaMA loaded successfully")

    def load_sae(self, layer: int) -> LlamaScopeDirect:
        """Load SAE for specific layer (cached)"""
        if layer not in self.sae_cache:
            LOGGER.info(f"Loading SAE Layer {layer}...")
            self.sae_cache[layer] = LlamaScopeDirect(layer=layer, device=str(self.device))
            LOGGER.info(f"SAE Layer {layer} loaded")
            torch.cuda.empty_cache()
        return self.sae_cache[layer]

    def load_features(self, causal_file: Path, means_file: Path):
        """Load all 265 causal features and their mean values"""
        # Load causal features list
        with open(causal_file, 'r') as f:
            data = json.load(f)

        # Build feature list from the features array if present
        if 'features' in data:
            self.all_features = data['features']
        else:
            # Fallback: reconstruct from layer_distribution
            self.all_features = []
            LOGGER.warning("No 'features' key found, using fallback")

        LOGGER.info(f"Loaded {len(self.all_features)} causal features")

        # Load feature means
        with open(means_file, 'r') as f:
            means_data = json.load(f)

        self.feature_means = means_data['feature_means']
        LOGGER.info(f"Loaded mean values for {len(self.feature_means)} features")

        # Validate all features have means
        missing = []
        for feat in self.all_features:
            fname = f"L{feat['layer']}-{feat['feature_id']}"
            if fname not in self.feature_means:
                missing.append(fname)

        if missing:
            LOGGER.warning(f"{len(missing)} features missing mean values: {missing[:5]}...")

    def load_or_compute_fdr_features(self, patching_dir: Path, fdr_cache_file: Optional[Path] = None) -> List[dict]:
        """
        Load FDR-confirmed features from cache or compute from patching results.

        Args:
            patching_dir: Directory containing patching_265_200_gpu*.jsonl files
            fdr_cache_file: Optional cache file to save/load FDR results

        Returns:
            List of FDR-confirmed features
        """
        # Try loading from cache
        if fdr_cache_file and fdr_cache_file.exists():
            LOGGER.info(f"Loading FDR features from cache: {fdr_cache_file}")
            with open(fdr_cache_file, 'r') as f:
                cache_data = json.load(f)
            return cache_data['fdr_features']

        # Compute from patching results
        LOGGER.info("Computing FDR-confirmed features from patching results...")

        results = load_patching_results(patching_dir)
        if not results:
            LOGGER.error("No patching results found!")
            return []

        feature_effects = compute_feature_effects(results)
        fdr_features = apply_fdr_correction(feature_effects, alpha=self.config.fdr_alpha)

        # Save to cache
        if fdr_cache_file:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'alpha': self.config.fdr_alpha,
                'total_features': len(feature_effects),
                'fdr_features': fdr_features
            }
            fdr_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(fdr_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            LOGGER.info(f"Saved FDR cache to {fdr_cache_file}")

        return fdr_features

    def extract_all_features(self, hidden_states_dict: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """
        Extract activations for ALL 265 causal features from hidden states.

        Args:
            hidden_states_dict: {layer: hidden_states tensor}

        Returns:
            {feature_name: activation_value}
        """
        all_activations = {}

        for layer, hidden_states in hidden_states_dict.items():
            sae = self.load_sae(layer)

            with torch.no_grad():
                # Encode with SAE: [batch, seq_len, 4096] -> [batch, seq_len, 32768]
                feature_acts = sae.encode(hidden_states.float())

                # Use final token activations
                final_acts = feature_acts[0, -1, :]  # [32768]

                # Extract only features in our causal features list
                for feat in self.all_features:
                    if feat['layer'] == layer:
                        feat_id = feat['feature_id']
                        feat_name = f"L{layer}-{feat_id}"
                        all_activations[feat_name] = float(final_acts[feat_id].item())

        return all_activations

    def generate_with_patching(
        self,
        prompt: str,
        target_layer: int,
        target_feature_id: int,
        patch_value: float,
    ) -> Tuple[str, Dict[str, float], List[int], List[str]]:
        """
        Generate response with feature patching and extract all features.

        Args:
            prompt: Input prompt
            target_layer: Layer to patch
            target_feature_id: Feature ID to patch
            patch_value: Value to patch (safe_mean or risky_mean)

        Returns:
            (response, all_feature_activations, generated_token_ids, generated_tokens)
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_len = inputs['input_ids'].shape[1]

        sae = self.load_sae(target_layer)

        def patching_hook(module, input, output):
            """Patch Block L OUTPUT (matches SAE training space)"""
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest_outputs = output[1:]
            else:
                hidden_states = output
                rest_outputs = None

            original_dtype = hidden_states.dtype

            with torch.no_grad():
                # Patch only last token position
                last_token = hidden_states[:, -1:, :].float()
                features = sae.encode(last_token)
                features[0, 0, target_feature_id] = float(patch_value)
                patched_hidden = sae.decode(features)
                hidden_states = hidden_states.clone()
                hidden_states[:, -1:, :] = patched_hidden.to(original_dtype)

            if rest_outputs is not None:
                return (hidden_states,) + rest_outputs
            else:
                return hidden_states

        # Register hook on target layer
        target_block = self.model.model.layers[target_layer]
        handle = target_block.register_forward_hook(patching_hook)

        try:
            with torch.no_grad():
                # 1. Generate with patching (greedy decoding)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    return_dict_in_generate=True
                )

                # 2. Extract response text
                full_sequence = outputs.sequences[0]
                response = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
                response = response[len(prompt):].strip()

                # 3. Extract generated tokens (for Phase 4)
                generated_token_ids = full_sequence[prompt_len:].tolist()
                generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]

                # 4. Forward pass with full sequence to get hidden states
                full_outputs = self.model(
                    input_ids=full_sequence.unsqueeze(0),
                    output_hidden_states=True,
                    return_dict=True
                )

                # 5. Extract features from layers containing causal features
                # NOTE: hidden_states[0] = embeddings, hidden_states[L+1] = output of layer L
                # SAE for layer L is trained on residual stream AFTER layer L
                layers_to_extract = set(feat['layer'] for feat in self.all_features)
                hidden_states_dict = {}
                for layer in layers_to_extract:
                    # Use layer + 1 to get OUTPUT of layer L (not input)
                    hidden_states_dict[layer] = full_outputs.hidden_states[layer + 1]

                all_activations = self.extract_all_features(hidden_states_dict)

        finally:
            handle.remove()
            del inputs
            torch.cuda.empty_cache()

        return response, all_activations, generated_token_ids, generated_tokens

    def generate_without_patching(self, prompt: str) -> Tuple[str, Dict[str, float], List[int], List[str]]:
        """
        Generate response WITHOUT patching (baseline) and extract all features.

        Returns:
            (response, all_feature_activations, generated_token_ids, generated_tokens)
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            # Generate (greedy decoding)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                return_dict_in_generate=True
            )

            full_sequence = outputs.sequences[0]
            response = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
            response = response[len(prompt):].strip()

            # Extract generated tokens
            generated_token_ids = full_sequence[prompt_len:].tolist()
            generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]

            # Forward pass for hidden states
            full_outputs = self.model(
                input_ids=full_sequence.unsqueeze(0),
                output_hidden_states=True,
                return_dict=True
            )

            # Extract features (use layer + 1 for OUTPUT of layer L)
            layers_to_extract = set(feat['layer'] for feat in self.all_features)
            hidden_states_dict = {}
            for layer in layers_to_extract:
                hidden_states_dict[layer] = full_outputs.hidden_states[layer + 1]

            all_activations = self.extract_all_features(hidden_states_dict)

        del inputs
        torch.cuda.empty_cache()

        return response, all_activations, generated_token_ids, generated_tokens

    def load_checkpoint(self, checkpoint_file: Path) -> Set[Tuple]:
        """Load checkpoint to find completed trials"""
        if not checkpoint_file.exists():
            return set()

        completed = set()
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        key = (
                            record['target_feature'],
                            record['patch_condition'],
                            record['prompt_type'],
                            record['trial']
                        )
                        completed.add(key)
                    except json.JSONDecodeError:
                        continue

        LOGGER.info(f"Checkpoint: {len(completed)} trials already completed")
        return completed

    def run_experiment(
        self,
        causal_file: Path,
        means_file: Path,
        patching_dir: Path,
        output_dir: Path,
        offset: int = 0,
        limit: Optional[int] = None
    ):
        """
        Main experiment loop.

        Conditions:
        - 4 conditions: (safe_patch/risky_patch) x (safe_prompt/risky_prompt)
        - Plus optional baseline (no patching)
        """
        # Load all features and means
        self.load_features(causal_file, means_file)

        # Load or compute FDR features
        fdr_cache_file = output_dir / "fdr_features_cache.json"
        self.fdr_features = self.load_or_compute_fdr_features(patching_dir, fdr_cache_file)

        if not self.fdr_features:
            LOGGER.error("No FDR-confirmed features found. Check patching results.")
            return

        LOGGER.info(f"FDR-confirmed features: {len(self.fdr_features)}")

        # Apply offset and limit for GPU distribution
        patch_targets = self.fdr_features
        if limit is not None:
            patch_targets = patch_targets[offset:offset + limit]
        elif offset > 0:
            patch_targets = patch_targets[offset:]

        LOGGER.info(f"Processing {len(patch_targets)} features (offset={offset}, limit={limit})")

        # Setup output
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"phase1_FDR_265_gpu{self.gpu_id}.jsonl"

        # Load checkpoint
        completed_trials = self.load_checkpoint(output_file)

        # Experiment conditions (4 patching conditions)
        conditions = [
            ('safe_patch', 'safe'),
            ('risky_patch', 'safe'),
            ('safe_patch', 'risky'),
            ('risky_patch', 'risky'),
        ]

        total_trials = len(patch_targets) * len(conditions) * self.config.n_trials
        LOGGER.info(f"=== Phase 1 FDR-265: {len(patch_targets)} features x {len(conditions)} conditions x {self.config.n_trials} trials ===")
        LOGGER.info(f"Total trials: {total_trials:,}")
        LOGGER.info(f"Already completed: {len(completed_trials):,}")
        LOGGER.info(f"Remaining: {total_trials - len(completed_trials):,}")

        with open(output_file, 'a') as f:
            pbar = tqdm(total=total_trials, desc=f"GPU{self.gpu_id}", initial=len(completed_trials))

            for target_feat in patch_targets:
                feature_name = target_feat['feature_name']
                target_layer = target_feat['layer']
                target_feature_id = target_feat['feature_id']

                if feature_name not in self.feature_means:
                    LOGGER.warning(f"Skipping {feature_name}: no mean values")
                    pbar.update(len(conditions) * self.config.n_trials)
                    continue

                means = self.feature_means[feature_name]
                patch_values = {
                    'safe_patch': means['safe_mean'],
                    'risky_patch': means['risky_mean'],
                }

                for patch_cond, prompt_type in conditions:
                    patch_value = patch_values[patch_cond]
                    prompt = self.safe_prompt if prompt_type == 'safe' else self.risky_prompt

                    for trial in range(self.config.n_trials):
                        trial_key = (feature_name, patch_cond, prompt_type, trial)
                        if trial_key in completed_trials:
                            pbar.update(1)
                            continue

                        try:
                            response, all_activations, token_ids, tokens = self.generate_with_patching(
                                prompt, target_layer, target_feature_id, patch_value
                            )

                            # Parse response to get action, bet_amount, is_stop
                            parsed = parse_response(response, prompt_type)

                            record = {
                                'target_feature': feature_name,
                                'target_layer': target_layer,
                                'target_feature_id': target_feature_id,
                                'patch_condition': patch_cond,
                                'patch_value': patch_value,
                                'prompt_type': prompt_type,
                                'trial': trial,
                                'response': response,
                                'action': parsed['action'],
                                'bet_amount': parsed['bet_amount'],
                                'is_stop': parsed['is_stop'],
                                'is_valid': parsed['is_valid'],
                                'parse_method': parsed['parse_method'],
                                'generated_token_ids': token_ids,
                                'generated_tokens': tokens,
                                'all_features': all_activations
                            }

                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            f.flush()

                            completed_trials.add(trial_key)

                        except Exception as e:
                            import traceback
                            LOGGER.error(f"Error on {feature_name} trial {trial}: {e}")
                            LOGGER.error(traceback.format_exc())

                        pbar.update(1)

            pbar.close()

        LOGGER.info(f"Phase 1 complete: {output_file}")
        LOGGER.info(f"Total records: {len(completed_trials):,}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1: FDR-Confirmed Feature Patching with Full Extraction")
    parser.add_argument('--gpu-id', type=int, required=True, help="GPU ID (matches CUDA_VISIBLE_DEVICES)")
    parser.add_argument('--n-trials', type=int, default=50, help="Trials per condition (default: 50)")
    parser.add_argument('--offset', type=int, default=0, help="Feature offset for GPU distribution")
    parser.add_argument('--limit', type=int, default=None, help="Max features to process")
    parser.add_argument('--output-dir', type=str,
                       default="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_FDR_265",
                       help="Output directory")
    parser.add_argument('--patching-dir', type=str,
                       default="/data/llm_addiction/patching_265_FDR_20251208",
                       help="Directory with patching results (for FDR computation)")
    parser.add_argument('--causal-features', type=str,
                       default="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_265_FINAL.json",
                       help="All 265 causal features file")
    parser.add_argument('--feature-means', type=str,
                       default="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_265_FINAL.json",
                       help="Feature means file")
    parser.add_argument('--fdr-alpha', type=float, default=0.05, help="FDR alpha level (default: 0.05)")
    args = parser.parse_args()

    LOGGER.info("=" * 70)
    LOGGER.info("Phase 1: FDR-Confirmed Feature Patching with Full Feature Extraction")
    LOGGER.info("=" * 70)
    LOGGER.info(f"GPU: {args.gpu_id}")
    LOGGER.info(f"Trials per condition: {args.n_trials}")
    LOGGER.info(f"Offset: {args.offset}, Limit: {args.limit}")
    LOGGER.info(f"FDR alpha: {args.fdr_alpha}")
    LOGGER.info(f"Patching results: {args.patching_dir}")
    LOGGER.info(f"Output: {args.output_dir}")
    LOGGER.info("=" * 70)

    # Create config
    config = ExperimentConfig(
        n_trials=args.n_trials,
        fdr_alpha=args.fdr_alpha
    )

    # Run experiment
    extractor = Phase1FDRExtractor(gpu_id=args.gpu_id, config=config)
    extractor.load_models()
    extractor.run_experiment(
        causal_file=Path(args.causal_features),
        means_file=Path(args.feature_means),
        patching_dir=Path(args.patching_dir),
        output_dir=Path(args.output_dir),
        offset=args.offset,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
