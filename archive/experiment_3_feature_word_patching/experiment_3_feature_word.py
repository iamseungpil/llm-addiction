#!/usr/bin/env python3
"""
Experiment 3: Feature-Word Co-occurrence Analysis
Analyze word patterns in responses when features are active/inactive

Data sources:
1. 441 causal features from Exp5 (multiround patching responses)
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

class FeatureWordAnalysis:
    def __init__(self, gpu_id: int = 0):
        """Initialize feature-word analysis"""
        self.device = 'cuda:0'
        self.gpu_id = gpu_id

        self.results_dir = Path('/data/llm_addiction/experiment_3_feature_word')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.sae_cache = {}

    def load_model(self):
        """Load LLaMA model"""
        print("🚀 Loading LLaMA model")

        torch.cuda.empty_cache()

        model_name = "meta-llama/Llama-3.1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={'': 0},
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
                    'outcome': entry.get('outcome', ''),
                    'bet': last_round.get('bet_amount', 0),
                    'action': last_round.get('decision', '')
                })

        for entry in data2.get('results', []):
            if 'round_features' in entry and entry['round_features']:
                last_round = entry['round_features'][-1]
                exp1_responses.append({
                    'prompt': last_round.get('prompt', ''),
                    'response': last_round.get('response', ''),
                    'outcome': entry.get('outcome', ''),
                    'bet': last_round.get('bet_amount', 0),
                    'action': last_round.get('decision', '')
                })

        print(f"✅ Loaded {len(exp1_responses)} Exp1 responses")
        return exp1_responses

    def load_exp5_data(self):
        """Load 441 causal feature responses from Exp5"""
        print("📂 Loading Exp5 multiround patching responses...")

        exp5_dir = Path('/data/llm_addiction/results')
        response_logs = list(exp5_dir.glob('exp2_response_log_*.json'))

        all_responses = []

        for log_file in response_logs:
            with open(log_file, 'r') as f:
                data = json.load(f)
                all_responses.extend(data)

        print(f"✅ Loaded {len(all_responses)} Exp5 patching responses")
        return all_responses

    def load_causal_features(self):
        """Load 441 causal features from Exp2 analysis"""
        print("📂 Loading 441 causal features...")

        csv_file = '/home/ubuntu/llm_addiction/analysis/exp2_feature_group_summary.csv'

        import csv
        causal_features = []

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['classified_as'] in ['safe', 'risky']:
                    # Parse feature name (e.g., "L25-14826")
                    feature_name = row['feature']
                    layer = int(feature_name.split('-')[0][1:])
                    feature_id = int(feature_name.split('-')[1])

                    causal_features.append({
                        'feature': feature_name,
                        'layer': layer,
                        'feature_id': feature_id,
                        'type': row['classified_as']
                    })

        print(f"✅ Loaded {len(causal_features)} causal features")
        return causal_features

    def extract_features(self, prompt: str, layer: int) -> np.ndarray:
        """Extract SAE features from prompt"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

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

        print(f"\n🔍 Analyzing {feature_name}...")

        # Extract features for all Exp1 responses
        all_activations = []
        all_words = []

        for i, resp in enumerate(tqdm(exp1_responses, desc=f"Processing {feature_name}")):
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
                print(f"⚠️  Error processing response {i}: {e}")
                continue

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

            if abs(diff) > 0.01:  # At least 1% difference
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
            'n_responses': len(all_activations),
            'median_activation': float(median),
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
        print("🚀 EXPERIMENT 3: FEATURE-WORD CO-OCCURRENCE ANALYSIS")
        print(f"   GPU: {self.gpu_id}")
        print(f"   Analyzing 441 causal features with 6,400 responses")
        print("=" * 80)

        self.load_model()

        # Load data
        causal_features = self.load_causal_features()
        exp1_responses = self.load_exp1_data()

        # Analyze each feature
        all_results = []

        for feature in causal_features:
            try:
                result = self.analyze_feature(feature, exp1_responses)
                all_results.append(result)

                # Save checkpoint every 20 features
                if len(all_results) % 20 == 0:
                    self.save_checkpoint(all_results)

            except Exception as e:
                print(f"❌ Error analyzing {feature['feature']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Final save
        self.save_final(all_results)

        print("\n" + "=" * 80)
        print("📊 ANALYSIS COMPLETE")
        print(f"   Features analyzed: {len(all_results)}")
        print("=" * 80)

    def save_checkpoint(self, results: list):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.results_dir / f'checkpoint_{len(results)}_{timestamp}.json'

        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_file}")

    def save_final(self, results: list):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_file = self.results_dir / f'final_feature_word_{timestamp}.json'

        summary = {
            'timestamp': timestamp,
            'gpu': self.gpu_id,
            'n_features': len(results),
            'results': results
        }

        with open(final_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Final results saved: {final_file}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    exp = FeatureWordAnalysis(gpu_id=args.gpu)
    exp.run()
