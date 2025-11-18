#!/usr/bin/env python3
"""
Step 1: Extract SAE Activations from Experiment 2 Responses

Purpose:
- Exp2 response logs have text but NO activations
- We need activations to split high/low groups for word analysis
- Extract activations for 2,787 causal features

Method:
- Load response text from Exp2 logs
- Re-run forward pass + SAE encode
- Save activations for each (feature, trial)

Output:
- activations_{layer}.npz: feature activations for each response
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import gc

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM


class ActivationExtractor:
    """
    Extract SAE activations from Exp2 response texts
    """

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}'

        # Paths
        self.exp2_response_dir = Path("/data/llm_addiction/experiment_2_multilayer_patching/response_logs")
        self.output_dir = Path("/data/llm_addiction/experiment_3_activation_cache")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.model = None
        self.tokenizer = None
        self.sae_cache = {}

        print("Loading models...")
        self._load_models()

    def _load_models(self):
        """Load LLaMA"""
        # Use base Llama for tokenizer and model
        base_model_name = 'meta-llama/Llama-3.1-8B'

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )

        print("  Models loaded")

    def load_sae(self, layer):
        """Load SAE for specific layer"""
        if layer in self.sae_cache:
            return self.sae_cache[layer]

        print(f"  Loading SAE Layer {layer}...")
        sae = LlamaScopeDirect(
            device=self.device,
            layer=layer,
            model_path='fnlp/Llama3_1-8B-Base-LXR-8x'
        )

        self.sae_cache[layer] = sae
        return sae

    def extract_activation(self, prompt_text, layer, feature_id):
        """
        Extract single feature activation from prompt

        Args:
            prompt_text: str
            layer: int
            feature_id: int

        Returns:
            activation: float
        """

        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)

            # Forward pass
            outputs = self.model(
                inputs.input_ids,
                output_hidden_states=True
            )

            # Get hidden state
            hidden = outputs.hidden_states[layer][:, -1, :]

            # SAE encode
            sae = self.load_sae(layer)
            features = sae.encode(hidden)  # (1, 32768)

            # Get specific feature
            activation = features[0, feature_id].item()

        return activation

    def process_layer_features(self, layer):
        """
        Process all features in a specific layer

        For each feature:
        1. Load all Exp2 responses for that feature
        2. Extract activations
        3. Save

        Output format:
        {
          'L25-1234': {
            'activations': [0.5, 0.3, 0.8, ...],  # 180 values (6 conditions × 30 trials)
            'conditions': ['safe_baseline', ...],
            'trials': [0, 1, 2, ...],
            'responses': ['text1', 'text2', ...]
          }
        }
        """

        print(f"\n{'='*80}")
        print(f"Processing Layer {layer}")
        print(f"{'='*80}\n")

        # Find all features in this layer
        safe_features = pd.read_csv("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv")
        risky_features = pd.read_csv("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv")

        layer_features = []
        for _, row in safe_features.iterrows():
            if row['feature'].startswith(f'L{layer}-'):
                layer_features.append(row['feature'])
        for _, row in risky_features.iterrows():
            if row['feature'].startswith(f'L{layer}-'):
                layer_features.append(row['feature'])

        print(f"Found {len(layer_features)} features in Layer {layer}")

        # Load Exp2 response logs
        print("Loading Exp2 response logs...")
        feature_responses = {}

        log_files = sorted(self.exp2_response_dir.glob(f"*L{layer}*.json"))

        for log_file in tqdm(log_files, desc=f"Loading L{layer} logs"):
            with open(log_file, 'r') as f:
                data = json.load(f)

            for record in data:
                feature = record['feature']

                if feature not in layer_features:
                    continue

                if feature not in feature_responses:
                    feature_responses[feature] = []

                feature_responses[feature].append({
                    'condition': record['condition'],
                    'trial': record['trial'],
                    'response': record['response']
                })

        print(f"Loaded responses for {len(feature_responses)} features")

        # Extract activations
        results = {}

        for feature in tqdm(layer_features, desc=f"Extracting L{layer} activations"):
            if feature not in feature_responses:
                print(f"  Warning: {feature} not in response logs")
                continue

            responses = feature_responses[feature]

            # Parse feature
            feat_layer, feat_id = self._parse_feature(feature)

            # Extract activation for each response
            activations = []
            conditions = []
            trials = []
            response_texts = []

            for resp_data in responses:
                # Reconstruct prompt (we need the PROMPT, not the response!)
                # Problem: Response logs only have response, not prompt!

                # We need to reconstruct the prompt
                prompt = self._reconstruct_prompt(resp_data['condition'])

                # Extract activation
                act = self.extract_activation(prompt, feat_layer, feat_id)

                activations.append(act)
                conditions.append(resp_data['condition'])
                trials.append(resp_data['trial'])
                response_texts.append(resp_data['response'])

            results[feature] = {
                'activations': activations,
                'conditions': conditions,
                'trials': trials,
                'responses': response_texts
            }

            # Memory cleanup
            if len(results) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Save
        output_file = self.output_dir / f'activations_L{layer}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f)

        print(f"\n✅ Saved: {output_file}")
        print(f"Features processed: {len(results)}")

        return results

    def _parse_feature(self, feature_str):
        """Parse 'L25-1234' -> (25, 1234)"""
        parts = feature_str.split('-')
        layer = int(parts[0][1:])
        feat_id = int(parts[1])
        return layer, feat_id

    def _reconstruct_prompt(self, condition):
        """
        Reconstruct prompt from condition name

        Conditions:
        - safe_baseline, safe_with_safe_patch, safe_with_risky_patch
        - risky_baseline, risky_with_safe_patch, risky_with_risky_patch
        """

        safe_prompt = """Slot Machine Game

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

        risky_prompt = """Slot Machine Game

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

        if 'safe' in condition:
            return safe_prompt
        else:
            return risky_prompt


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4, help='GPU ID')
    parser.add_argument('--layer', type=int, required=True, help='Layer to process')
    args = parser.parse_args()

    extractor = ActivationExtractor(gpu_id=args.gpu)
    extractor.process_layer_features(args.layer)


if __name__ == "__main__":
    main()
