#!/usr/bin/env python3
"""
Experiment 3 Extended: Feature-Word Co-occurrence Analysis for ALL L1-31 Features
Analyze word patterns in responses for all 87,012 significant features from L1-31 extraction

Data sources:
1. 87,012 significant features from L1-31 extraction
2. 6,400 Exp1 responses (bankruptcy vs voluntary stop groups)

Method:
- Extract SAE features for all responses
- Group responses by feature activation (high/low)
- Analyze word frequency differences between groups
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
from transformers import AutoTokenizer, AutoModelForCausalLM

class ExtendedFeatureWordAnalysis:
    def __init__(self, gpu_id: int = 0, layer_start: int = 1, layer_end: int = 31):
        """Initialize extended feature-word analysis for L1-31"""
        self.device = f'cuda:{gpu_id}'
        self.gpu_id = gpu_id
        self.layer_start = layer_start
        self.layer_end = layer_end

        self.results_dir = Path('/data/llm_addiction/experiment_3_L1_31_word_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.sae_cache = {}

        print(f"🎯 Analyzing Layers {layer_start}-{layer_end}")

    def load_model(self):
        """Load LLaMA model"""
        print("🚀 Loading LLaMA model...")

        torch.cuda.empty_cache()

        model_name = "meta-llama/Llama-3.1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={'': self.gpu_id},
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.model.eval()

        print("✅ LLaMA loaded successfully")

    def load_sae(self, layer: int):
        """Load SAE for layer on-demand"""
        if layer not in self.sae_cache:
            print(f"🔧 Loading SAE Layer {layer}...")
            self.sae_cache[layer] = LlamaScopeDirect(layer=layer)
            torch.cuda.empty_cache()
        return self.sae_cache[layer]

    def load_l1_31_features(self):
        """Load 87,012 significant features from L1-31 extraction"""
        print("📂 Loading L1-31 significant features (87,012 features)...")

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

                # Classify as safe/risky based on Cohen's d direction
                # Positive d: bankrupt > safe (risky feature)
                # Negative d: safe > bankrupt (safe feature)
                feature_type = 'risky' if cohens_d > 0 else 'safe'

                all_features.append({
                    'feature': f'L{layer}-{feature_idx}',
                    'layer': layer,
                    'feature_id': feature_idx,
                    'type': feature_type,
                    'cohens_d': cohens_d,
                    'p_value': feat.get('p_value', 1.0),
                    'bankrupt_mean': feat.get('bankrupt_mean', 0.0),
                    'safe_mean': feat.get('safe_mean', 0.0)
                })

        print(f"✅ Loaded {len(all_features)} features across layers {self.layer_start}-{self.layer_end}")

        # Print summary by layer
        layer_counts = {}
        for feat in all_features:
            layer = feat['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        print(f"\nFeatures per layer:")
        for layer in sorted(layer_counts.keys()):
            print(f"  Layer {layer}: {layer_counts[layer]} features")

        return all_features

    def load_exp1_data(self):
        """Load 6,400 Exp1 responses"""
        print("📂 Loading Exp1 data (6,400 games)...")

        file1 = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
        file2 = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'

        with open(file1, 'r') as f:
            data1 = json.load(f)

        with open(file2, 'r') as f:
            data2 = json.load(f)

        # Extract last round responses
        exp1_responses = []

        for entry in data1.get('results', []):
            # Get last round response
            if 'round_features' in entry and entry['round_features']:
                last_round = entry['round_features'][-1]
                exp1_responses.append({
                    'prompt': last_round.get('prompt', ''),
                    'response': last_round.get('response', ''),
                    'is_bankrupt': entry.get('is_bankrupt', False),
                    'voluntary_stop': entry.get('voluntary_stop', False),
                    'bet': last_round.get('bet_amount', 0),
                    'decision': last_round.get('decision', '')
                })

        for entry in data2.get('results', []):
            if 'round_features' in entry and entry['round_features']:
                last_round = entry['round_features'][-1]
                exp1_responses.append({
                    'prompt': last_round.get('prompt', ''),
                    'response': last_round.get('response', ''),
                    'is_bankrupt': entry.get('is_bankrupt', False),
                    'voluntary_stop': entry.get('voluntary_stop', False),
                    'bet': last_round.get('bet_amount', 0),
                    'decision': last_round.get('decision', '')
                })

        print(f"✅ Loaded {len(exp1_responses)} Exp1 responses")
        return exp1_responses

    def extract_features(self, prompt: str, layer: int) -> np.ndarray:
        """Extract SAE features from prompt"""
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(self.device)

        # Hook to capture hidden state
        captured_hidden = None

        def hook_fn(module, args, output):
            nonlocal captured_hidden
            hidden = output[0] if isinstance(output, tuple) else output
            captured_hidden = hidden.detach()

        hook = self.model.model.layers[layer].register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                _ = self.model(**inputs)

            # Encode with SAE
            sae = self.load_sae(layer)
            features = sae.encode(captured_hidden)  # (1, seq_len, n_features)

            # Take last token
            last_token_features = features[0, -1, :].cpu().numpy()

            return last_token_features

        finally:
            hook.remove()

    def tokenize_response(self, response: str) -> list:
        """Tokenize response into words"""
        # Clean and tokenize
        response_lower = response.lower()
        response_clean = re.sub(r'[^a-z0-9\s가-힣]', ' ', response_lower)
        words = response_clean.split()

        # Filter out very short words
        words = [w for w in words if len(w) > 2]

        return words

    def analyze_feature(self, feature_info: dict, exp1_responses: list) -> dict:
        """Analyze word patterns for single feature across Exp1 data"""
        layer = feature_info['layer']
        feature_id = feature_info['feature_id']
        feature_name = feature_info['feature']

        # Extract features for all Exp1 responses
        all_activations = []
        all_words = []

        for i, resp in enumerate(exp1_responses):
            prompt = resp['prompt']
            response = resp['response']

            if not prompt or not response:
                continue

            try:
                # Extract features
                features = self.extract_features(prompt, layer)
                activation = features[feature_id]

                # Tokenize response
                words = self.tokenize_response(response)

                all_activations.append(activation)
                all_words.append(words)

            except Exception as e:
                if i < 10:  # Only print first 10 errors
                    print(f"⚠️  Error processing response {i}: {e}")
                continue

        if len(all_activations) < 100:
            print(f"⚠️  Warning: Only {len(all_activations)} valid responses for {feature_name}")
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

        # Find distinctive words
        high_total = sum(high_counter.values())
        low_total = sum(low_counter.values())

        distinctive_high = []
        distinctive_low = []

        for word in set(high_words + low_words):
            high_freq = high_counter[word] / high_total if high_total > 0 else 0
            low_freq = low_counter[word] / low_total if low_total > 0 else 0

            diff = high_freq - low_freq

            if abs(diff) > 0.005:  # At least 0.5% difference (lowered threshold)
                if diff > 0:
                    distinctive_high.append((word, high_freq, low_freq, diff))
                else:
                    distinctive_low.append((word, high_freq, low_freq, abs(diff)))

        # Sort by difference
        distinctive_high.sort(key=lambda x: x[3], reverse=True)
        distinctive_low.sort(key=lambda x: x[3], reverse=True)

        return {
            'feature': feature_name,
            'layer': layer,
            'feature_id': feature_id,
            'type': feature_info['type'],
            'cohens_d': feature_info['cohens_d'],
            'n_responses': len(all_activations),
            'median_activation': float(median),
            'mean_activation': float(np.mean(activations_arr)),
            'std_activation': float(np.std(activations_arr)),
            'high_activation_words': [
                {'word': w, 'high_freq': hf, 'low_freq': lf, 'diff': d}
                for w, hf, lf, d in distinctive_high[:20]
            ],
            'low_activation_words': [
                {'word': w, 'high_freq': hf, 'low_freq': lf, 'diff': d}
                for w, hf, lf, d in distinctive_low[:20]
            ]
        }

    def run(self):
        """Main analysis loop"""
        print("=" * 80)
        print("🚀 EXPERIMENT 3 EXTENDED: L1-31 FEATURE-WORD ANALYSIS")
        print(f"   GPU: {self.gpu_id}")
        print(f"   Layers: {self.layer_start}-{self.layer_end}")
        print(f"   Analyzing ~87,012 features with 6,400 responses")
        print("=" * 80)

        self.load_model()

        # Load data
        all_features = self.load_l1_31_features()
        exp1_responses = self.load_exp1_data()

        # Analyze each feature
        all_results = []
        skipped = 0

        for i, feature in enumerate(tqdm(all_features, desc="Analyzing features")):
            try:
                result = self.analyze_feature(feature, exp1_responses)

                if result is not None:
                    all_results.append(result)
                else:
                    skipped += 1

                # Save checkpoint every 500 features
                if len(all_results) % 500 == 0 and len(all_results) > 0:
                    self.save_checkpoint(all_results)
                    print(f"\n✅ Checkpoint: {len(all_results)} features analyzed, {skipped} skipped")

            except Exception as e:
                print(f"\n❌ Error analyzing {feature['feature']}: {e}")
                skipped += 1
                continue

        # Final save
        self.save_final(all_results)

        print("\n" + "=" * 80)
        print("📊 ANALYSIS COMPLETE")
        print(f"   Features analyzed: {len(all_results)}")
        print(f"   Features skipped: {skipped}")
        print("=" * 80)

    def save_checkpoint(self, results: list):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.results_dir / f'checkpoint_L{self.layer_start}_{self.layer_end}_{len(results)}_{timestamp}.json'

        with open(checkpoint_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'gpu': self.gpu_id,
                'layers': f'{self.layer_start}-{self.layer_end}',
                'n_features': len(results),
                'results': results
            }, f, indent=2)

        print(f"\n💾 Checkpoint saved: {checkpoint_file.name}")

    def save_final(self, results: list):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_file = self.results_dir / f'final_L{self.layer_start}_{self.layer_end}_{timestamp}.json'

        summary = {
            'timestamp': timestamp,
            'gpu': self.gpu_id,
            'layers': f'{self.layer_start}-{self.layer_end}',
            'n_features': len(results),
            'results': results
        }

        with open(final_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Final results saved: {final_file.name}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--layer-start', type=int, default=1, help='Start layer')
    parser.add_argument('--layer-end', type=int, default=31, help='End layer')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    exp = ExtendedFeatureWordAnalysis(
        gpu_id=0,  # After setting CUDA_VISIBLE_DEVICES, always use 0
        layer_start=args.layer_start,
        layer_end=args.layer_end
    )
    exp.run()
