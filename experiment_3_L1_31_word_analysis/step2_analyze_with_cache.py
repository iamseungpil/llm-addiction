#!/usr/bin/env python3
"""
Experiment 3 Step 2: Feature-Word Analysis using Cached Hidden States
Load cached hidden states and analyze word patterns for all features
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import re

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect

class CachedFeatureWordAnalysis:
    """Analyze feature-word patterns using cached hidden states"""

    def __init__(self, gpu_id: int = 0, layer_range=(1, 31)):
        self.device = f'cuda:{gpu_id}'
        self.layer_start, self.layer_end = layer_range

        self.cache_dir = Path('/data/llm_addiction/experiment_3_hidden_cache')
        self.results_dir = Path('/data/llm_addiction/experiment_3_L1_31_word_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.sae_cache = {}

        print(f"🎯 Analyzing layers {layer_range[0]}-{layer_range[1]}")

    def load_sae(self, layer: int):
        """Load SAE for layer on-demand"""
        if layer not in self.sae_cache:
            print(f"  Loading SAE Layer {layer}...")
            self.sae_cache[layer] = LlamaScopeDirect(layer=layer)
            torch.cuda.empty_cache()
        return self.sae_cache[layer]

    def load_l1_31_features(self):
        """Load 87,012 significant features from L1-31 extraction"""
        print("📂 Loading L1-31 significant features...")

        l1_31_file = '/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json'

        with open(l1_31_file, 'r') as f:
            l1_31_data = json.load(f)

        all_features = []

        for layer in range(self.layer_start, self.layer_end + 1):
            layer_key = str(layer)
            if layer_key not in l1_31_data['layer_results']:
                continue

            layer_data = l1_31_data['layer_results'][layer_key]
            sig_features = layer_data['significant_features']

            for feat in sig_features:
                feature_idx = feat['feature_idx']
                cohens_d = feat.get('cohen_d', 0.0)
                feature_type = 'risky' if cohens_d > 0 else 'safe'

                all_features.append({
                    'feature': f'L{layer}-{feature_idx}',
                    'layer': layer,
                    'feature_id': feature_idx,
                    'type': feature_type,
                    'cohens_d': cohens_d,
                    'p_value': feat.get('p_value', 1.0)
                })

        print(f"✅ Loaded {len(all_features)} features")
        return all_features

    def load_exp1_data(self):
        """Load 6,400 Exp1 response texts"""
        print("📂 Loading Exp1 response texts...")

        file1 = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
        file2 = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'

        with open(file1, 'r') as f:
            data1 = json.load(f)

        with open(file2, 'r') as f:
            data2 = json.load(f)

        exp1_responses = []

        for entry in data1.get('results', []):
            if 'round_features' in entry and entry['round_features']:
                last_round = entry['round_features'][-1]
                exp1_responses.append(last_round.get('response', ''))

        for entry in data2.get('results', []):
            if 'round_features' in entry and entry['round_features']:
                last_round = entry['round_features'][-1]
                exp1_responses.append(last_round.get('response', ''))

        print(f"✅ Loaded {len(exp1_responses)} response texts")
        return exp1_responses

    def tokenize_response(self, response: str) -> list:
        """Tokenize response into words"""
        response_lower = response.lower()
        response_clean = re.sub(r'[^a-z0-9\s가-힣]', ' ', response_lower)
        words = response_clean.split()
        words = [w for w in words if len(w) > 2]
        return words

    def analyze_feature(self, feature_info: dict, cached_hidden: np.ndarray,
                       response_texts: list) -> dict:
        """Analyze word patterns for single feature"""

        layer = feature_info['layer']
        feature_id = feature_info['feature_id']
        feature_name = feature_info['feature']

        # Load SAE
        sae = self.load_sae(layer)

        # Encode cached hidden states with SAE
        all_activations = []
        all_words = []

        for i, (hidden_state, response_text) in enumerate(zip(cached_hidden, response_texts)):
            if not response_text:
                continue

            try:
                # SAE encode (FAST - no forward pass!)
                hidden_tensor = torch.from_numpy(hidden_state).float().to(self.device)
                features = sae.encode(hidden_tensor)
                activation = features[feature_id].cpu().item()

                # Tokenize response
                words = self.tokenize_response(response_text)

                all_activations.append(activation)
                all_words.append(words)

            except Exception as e:
                if i < 5:
                    print(f"⚠️ Error: {e}")
                continue

        if len(all_activations) < 100:
            return None

        # Split into high/low activation groups
        activations_arr = np.array(all_activations)
        median = np.median(activations_arr)

        high_words = []
        low_words = []

        for activation, words in zip(all_activations, all_words):
            if activation >= median:
                high_words.extend(words)
            else:
                low_words.extend(words)

        # Count word frequencies
        high_counter = Counter(high_words)
        low_counter = Counter(low_words)

        high_total = sum(high_counter.values())
        low_total = sum(low_counter.values())

        distinctive_high = []

        for word in set(high_words + low_words):
            high_freq = high_counter[word] / high_total if high_total > 0 else 0
            low_freq = low_counter[word] / low_total if low_total > 0 else 0
            diff = high_freq - low_freq

            if diff > 0.005:  # At least 0.5% difference
                distinctive_high.append((word, high_freq, low_freq, diff))

        distinctive_high.sort(key=lambda x: x[3], reverse=True)

        return {
            'feature': feature_name,
            'layer': layer,
            'feature_id': feature_id,
            'type': feature_info['type'],
            'cohens_d': feature_info['cohens_d'],
            'n_responses': len(all_activations),
            'median_activation': float(median),
            'high_activation_words': [
                {'word': w, 'high_freq': hf, 'low_freq': lf, 'diff': d}
                for w, hf, lf, d in distinctive_high[:20]
            ]
        }

    def run(self):
        """Main analysis loop"""
        print("="*80)
        print("EXPERIMENT 3 STEP 2: FEATURE-WORD ANALYSIS (CACHED)")
        print("="*80)

        # Load data
        all_features = self.load_l1_31_features()
        response_texts = self.load_exp1_data()

        # Group features by layer
        features_by_layer = {}
        for feat in all_features:
            layer = feat['layer']
            if layer not in features_by_layer:
                features_by_layer[layer] = []
            features_by_layer[layer].append(feat)

        all_results = []
        skipped = 0

        # Process layer by layer
        for layer in tqdm(range(self.layer_start, self.layer_end + 1),
                         desc="Processing layers"):

            if layer not in features_by_layer:
                print(f"\n⚠️ No features for Layer {layer}")
                continue

            # Load cached hidden states for this layer
            cache_file = self.cache_dir / f'layer_{layer}_hidden_states.npz'

            if not cache_file.exists():
                print(f"\n❌ Cache missing for Layer {layer}: {cache_file}")
                continue

            print(f"\n📂 Loading Layer {layer} cache...")
            cached_data = np.load(cache_file)
            cached_hidden = cached_data['hidden_states']
            print(f"  ✅ Loaded {len(cached_hidden)} hidden states")

            # Analyze all features for this layer
            layer_features = features_by_layer[layer]
            print(f"  Analyzing {len(layer_features)} features...")

            for feature in tqdm(layer_features, desc=f"  L{layer} features", leave=False):
                try:
                    result = self.analyze_feature(feature, cached_hidden, response_texts)

                    if result is not None:
                        all_results.append(result)
                    else:
                        skipped += 1

                except Exception as e:
                    print(f"\n❌ Error: {feature['feature']}: {e}")
                    skipped += 1

            # Save checkpoint after each layer
            self.save_checkpoint(all_results, f"L{self.layer_start}_{layer}")

            print(f"  ✅ Layer {layer} complete: {len(all_results)} total features analyzed")

        # Final save
        self.save_final(all_results)

        print("\n" + "="*80)
        print("📊 ANALYSIS COMPLETE")
        print(f"   Features analyzed: {len(all_results)}")
        print(f"   Features skipped: {skipped}")
        print("="*80)

    def save_checkpoint(self, results: list, suffix: str):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.results_dir / f'checkpoint_{suffix}_{len(results)}_{timestamp}.json'

        with open(checkpoint_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'layers': f'{self.layer_start}-{self.layer_end}',
                'n_features': len(results),
                'results': results
            }, f, indent=2)

    def save_final(self, results: list):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_file = self.results_dir / f'final_L{self.layer_start}_{self.layer_end}_{timestamp}.json'

        with open(final_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'layers': f'{self.layer_start}-{self.layer_end}',
                'n_features': len(results),
                'results': results
            }, f, indent=2)

        print(f"\n💾 Final results: {final_file.name}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--layer-start', type=int, default=1)
    parser.add_argument('--layer-end', type=int, default=31)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    analysis = CachedFeatureWordAnalysis(
        gpu_id=0,
        layer_range=(args.layer_start, args.layer_end)
    )
    analysis.run()
