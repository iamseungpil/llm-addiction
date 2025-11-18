#!/usr/bin/env python3
"""
Gradient-based Pathway Tracking for 2,787 Causal Features
Feature-centric backward Jacobian tracking (Anthropic 2025 method)

Based on: Circuit Tracing: Revealing Computational Graphs in Language Models
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import gc

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM


class GradientPathwayTracker:
    """
    Anthropic Attribution Graphs 방식
    Feature-centric backward Jacobian tracking
    """

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}'

        # Paths
        self.safe_features_csv = Path("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv")
        self.risky_features_csv = Path("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv")

        self.results_dir = Path("/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load causal features
        print("Loading 2,787 causal features...")
        self.safe_features = pd.read_csv(self.safe_features_csv)
        self.risky_features = pd.read_csv(self.risky_features_csv)

        print(f"  Safe features: {len(self.safe_features)}")
        print(f"  Risky features: {len(self.risky_features)}")

        self.all_features = self._prepare_features()
        print(f"  Total: {len(self.all_features)}")

        # Prompts (from Experiment 2)
        self.safe_prompt = """Slot Machine Game

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

        self.risky_prompt = """Slot Machine Game

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

        # Models
        self.model = None
        self.tokenizer = None
        self.sae_cache = {}

        # Load models
        self._load_models()

    def _prepare_features(self):
        """Prepare 2,787 features list"""
        all_features = []

        # Safe features
        for _, row in self.safe_features.iterrows():
            layer, feat_id = self._parse_feature(row['feature'])
            all_features.append({
                'feature': row['feature'],
                'type': 'safe',
                'layer': layer,
                'feature_id': feat_id
            })

        # Risky features
        for _, row in self.risky_features.iterrows():
            layer, feat_id = self._parse_feature(row['feature'])
            all_features.append({
                'feature': row['feature'],
                'type': 'risky',
                'layer': layer,
                'feature_id': feat_id
            })

        # Sort by layer
        all_features.sort(key=lambda x: (x['layer'], x['feature_id']))

        return all_features

    def _parse_feature(self, feature_str):
        """Parse 'L25-1234' -> (25, 1234)"""
        parts = feature_str.split('-')
        layer = int(parts[0][1:])
        feat_id = int(parts[1])
        return layer, feat_id

    def _load_models(self):
        """Load LLaMA and SAEs"""
        print("\nLoading models...")

        # LLaMA
        model_name = 'fnlp/Llama3_1-8B-Base-LXR-8x'
        print(f"  Loading LLaMA: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )

        print("  Models loaded successfully")

    def load_sae(self, layer):
        """Load SAE for specific layer (with caching)"""
        if layer in self.sae_cache:
            return self.sae_cache[layer]

        print(f"  Loading SAE for Layer {layer}...")

        sae = LlamaScopeDirect(
            device=self.device,
            layer=layer,
            model_path='fnlp/Llama3_1-8B-Base-LXR-8x'
        )

        self.sae_cache[layer] = sae
        return sae

    def track_feature_pathway(self, target_feature, prompt, prompt_name):
        """
        특정 feature의 upstream pathway 추적

        Args:
            target_feature: {'layer': 31, 'feature_id': 10692, 'feature': 'L31-10692'}
            prompt: safe_prompt or risky_prompt
            prompt_name: 'safe' or 'risky'

        Returns:
            pathway: List of upstream contributors with gradients
        """

        target_layer = target_feature['layer']
        target_id = target_feature['feature_id']

        # Clear gradients
        if hasattr(self.model, 'zero_grad'):
            self.model.zero_grad()

        try:
            # 1. Forward pass with gradient tracking
            with torch.enable_grad():
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

                # Get hidden states
                outputs = self.model(
                    inputs.input_ids,
                    output_hidden_states=True,
                    use_cache=False  # Important for gradients
                )

                # 2. Extract ALL layer SAE features (L1 to target_layer)
                all_layer_features = {}

                for layer in range(1, target_layer + 1):
                    hidden = outputs.hidden_states[layer][:, -1, :]  # Last token

                    # SAE encode (requires_grad=True!)
                    sae = self.load_sae(layer)
                    features = sae.encode(hidden)  # (32768,)

                    # Enable gradient
                    if layer < target_layer:
                        features.requires_grad_(True)

                    all_layer_features[layer] = features

                # 3. Target feature activation
                target_activation = all_layer_features[target_layer][0, target_id]

            # 4. Backward pass (compute gradients)
            target_activation.backward()

            # 5. Extract upstream contributors
            pathway = []

            for source_layer in range(1, target_layer):
                if source_layer not in all_layer_features:
                    continue

                grad = all_layer_features[source_layer].grad

                if grad is None:
                    continue

                # Strong contributors: |gradient| > 0.1
                grad_abs = torch.abs(grad[0])
                strong_indices = torch.where(grad_abs > 0.1)[0]

                for src_idx in strong_indices:
                    pathway.append({
                        'source': f'L{source_layer}-{src_idx.item()}',
                        'target': f'L{target_layer}-{target_id}',
                        'gradient': grad[0, src_idx].item(),
                        'source_layer': source_layer,
                        'source_id': src_idx.item(),
                        'target_layer': target_layer,
                        'target_id': target_id,
                        'prompt': prompt_name
                    })

            # Sort by gradient magnitude
            pathway.sort(key=lambda x: abs(x['gradient']), reverse=True)

            # Clear memory
            del outputs, all_layer_features
            torch.cuda.empty_cache()

            return pathway

        except Exception as e:
            print(f"  Error tracking {target_feature['feature']} on {prompt_name}: {e}")
            return []

    def analyze_all_features(self, checkpoint_interval=50):
        """2,787 features 모두 분석"""

        results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print(f"\n{'='*80}")
        print(f"Starting Gradient Pathway Tracking")
        print(f"Total features: {len(self.all_features)}")
        print(f"GPU: {self.gpu_id}")
        print(f"{'='*80}\n")

        for idx, feat_info in enumerate(tqdm(self.all_features, desc="Tracking pathways")):

            # Safe prompt pathway
            safe_pathway = self.track_feature_pathway(
                target_feature=feat_info,
                prompt=self.safe_prompt,
                prompt_name='safe'
            )

            # Risky prompt pathway
            risky_pathway = self.track_feature_pathway(
                target_feature=feat_info,
                prompt=self.risky_prompt,
                prompt_name='risky'
            )

            result = {
                'feature': feat_info['feature'],
                'type': feat_info['type'],
                'layer': feat_info['layer'],
                'feature_id': feat_info['feature_id'],
                'safe_pathway': safe_pathway[:20],  # Top 20 contributors
                'risky_pathway': risky_pathway[:20],
                'safe_upstream_count': len(safe_pathway),
                'risky_upstream_count': len(risky_pathway),
                'safe_avg_gradient': np.mean([abs(p['gradient']) for p in safe_pathway]) if safe_pathway else 0,
                'risky_avg_gradient': np.mean([abs(p['gradient']) for p in risky_pathway]) if risky_pathway else 0
            }

            results.append(result)

            # Checkpoint
            if (idx + 1) % checkpoint_interval == 0:
                checkpoint_file = self.results_dir / f'checkpoint_{timestamp}_{idx+1}.json'
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n✅ Checkpoint saved: {idx+1}/{len(self.all_features)}")

            # Memory cleanup
            if (idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Final save
        final_file = self.results_dir / f'gradient_pathways_{timestamp}_final.json'
        with open(final_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✅ Analysis complete!")
        print(f"Results saved: {final_file}")
        print(f"Total features analyzed: {len(results)}")
        print(f"{'='*80}\n")

        return results, final_file


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4, help='GPU ID')
    args = parser.parse_args()

    # Run tracker
    tracker = GradientPathwayTracker(gpu_id=args.gpu)
    results, output_file = tracker.analyze_all_features()

    # Quick stats
    print("\n=== Quick Statistics ===")

    safe_feats = [r for r in results if r['type'] == 'safe']
    risky_feats = [r for r in results if r['type'] == 'risky']

    print(f"Safe features: {len(safe_feats)}")
    print(f"  Avg upstream (safe prompt): {np.mean([f['safe_upstream_count'] for f in safe_feats]):.1f}")
    print(f"  Avg upstream (risky prompt): {np.mean([f['risky_upstream_count'] for f in safe_feats]):.1f}")

    print(f"\nRisky features: {len(risky_feats)}")
    print(f"  Avg upstream (safe prompt): {np.mean([f['safe_upstream_count'] for f in risky_feats]):.1f}")
    print(f"  Avg upstream (risky prompt): {np.mean([f['risky_upstream_count'] for f in risky_feats]):.1f}")

    # Middle layer risky features
    middle_risky = [f for f in risky_feats if 9 <= f['layer'] <= 17]
    print(f"\nMiddle-layer risky features (L9-L17): {len(middle_risky)}")
    if middle_risky:
        print(f"  Avg upstream count: {np.mean([f['safe_upstream_count'] + f['risky_upstream_count'] for f in middle_risky]) / 2:.1f}")


if __name__ == "__main__":
    main()
