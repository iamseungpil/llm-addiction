"""
Feature-Word Association Analysis
Analyze which tokens/words activate 441 causal features most strongly
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import random
sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking

class FeatureWordAssociation:
    def __init__(self, device='cuda:0', sample_size=640):
        self.device = device
        self.sample_size = sample_size

        print("Loading 441 causal features...")
        csv_file = '/home/ubuntu/llm_addiction/analysis/exp2_feature_group_summary.csv'
        df = pd.read_csv(csv_file)
        causal_df = df[df['classified_as'].isin(['safe', 'risky'])]

        self.causal_features = []
        for _, row in causal_df.iterrows():
            feature_string = row['feature']
            layer = int(feature_string.split('-')[0][1:])
            feature_id = int(feature_string.split('-')[1])
            self.causal_features.append({
                'layer': layer,
                'feature_id': feature_id,
                'name': feature_string,
                'type': row['classified_as']
            })
        print(f"✅ Loaded {len(self.causal_features)} causal features")

        # Group by layer for efficient SAE loading
        self.features_by_layer = defaultdict(list)
        for feat in self.causal_features:
            self.features_by_layer[feat['layer']].append(feat)

        print("\nLoading LLaMA model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            cache_dir='/data/.cache/huggingface'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            cache_dir='/data/.cache/huggingface'
        )
        self.model.eval()
        print("✅ LLaMA model loaded")

        # Don't preload SAEs - load on-demand to save memory
        self.current_sae = None
        self.current_sae_layer = None

        # Token-feature activation tracking
        # token_activations[token_id] = {feature_name: [activations]}
        self.token_activations = defaultdict(lambda: defaultdict(list))

    def sample_experiments(self, json_file, n_samples):
        """Sample experiments from large JSON file"""
        print(f"\nLoading experiment indices from {json_file}...")

        # First pass: count total experiments
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            experiments = data.get('results', data)
        else:
            experiments = data

        total = len(experiments)
        print(f"Total experiments: {total}")

        # Sample indices
        if n_samples >= total:
            sampled_indices = list(range(total))
        else:
            sampled_indices = sorted(random.sample(range(total), n_samples))

        print(f"Sampled {len(sampled_indices)} experiments")

        # Return sampled experiments
        sampled = [experiments[i] for i in sampled_indices]
        del data  # Free memory
        return sampled

    def load_sae(self, layer):
        """Load SAE for specific layer (on-demand)"""
        if self.current_sae_layer == layer:
            return self.current_sae

        # Clear previous SAE
        if self.current_sae is not None:
            del self.current_sae
            torch.cuda.empty_cache()

        self.current_sae = LlamaScopeWorking(layer, device=self.device)
        self.current_sae_layer = layer
        return self.current_sae

    def extract_features_from_response(self, response_text, layer):
        """Extract feature activations for a single response"""
        if not response_text or len(response_text.strip()) == 0:
            return None

        # Tokenize
        tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
        if len(tokens) == 0:
            return None

        # Convert to tensor
        input_ids = torch.tensor([tokens]).to(self.device)

        # Get hidden states
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[layer]  # [1, seq_len, 4096]

        # Load SAE on-demand
        sae = self.load_sae(layer)

        # Extract features for each token
        token_features = []

        for token_idx in range(hidden_states.shape[1]):
            hidden = hidden_states[:, token_idx:token_idx+1, :].float()
            features = sae.encode(hidden)  # [1, 1, 32768]
            token_features.append(features[0, 0].cpu().numpy())

        return {
            'tokens': tokens,
            'features': np.array(token_features)  # [seq_len, 32768]
        }

    def process_experiment(self, experiment):
        """Process one experiment and record token-feature associations"""
        # Try both 'round_features' and 'rounds' keys
        rounds = experiment.get('round_features', experiment.get('rounds', []))
        if not rounds:
            return

        # Process each round's response
        for round_data in rounds:
            response = round_data.get('response', '').strip()
            if not response:
                continue

            # Process each layer that has causal features
            for layer in self.features_by_layer.keys():
                result = self.extract_features_from_response(response, layer)
                if result is None:
                    continue

                tokens = result['tokens']
                features = result['features']  # [seq_len, 32768]

                # Record activations for causal features in this layer
                for feat_info in self.features_by_layer[layer]:
                    feature_id = feat_info['feature_id']
                    feature_name = feat_info['name']

                    # Get activations for this feature across all tokens
                    activations = features[:, feature_id]  # [seq_len]

                    # Record each token's activation
                    for token_id, activation in zip(tokens, activations):
                        self.token_activations[token_id][feature_name].append(float(activation))

    def analyze_associations(self):
        """Analyze token-feature associations"""
        print("\n📊 Analyzing token-feature associations...")

        results = {}
        for feat_info in tqdm(self.causal_features, desc="Processing features"):
            feature_name = feat_info['name']

            # Collect all token activations for this feature
            token_stats = []
            for token_id, feature_acts in self.token_activations.items():
                if feature_name in feature_acts:
                    activations = feature_acts[feature_name]
                    token_stats.append({
                        'token_id': token_id,
                        'token_text': self.tokenizer.decode([token_id]),
                        'mean_activation': np.mean(activations),
                        'max_activation': np.max(activations),
                        'frequency': len(activations)
                    })

            # Sort by mean activation
            token_stats.sort(key=lambda x: x['mean_activation'], reverse=True)

            # Store top 20
            results[feature_name] = {
                'feature_info': feat_info,
                'top_tokens': token_stats[:20]
            }

        return results

    def run_experiment(self):
        """Main experiment flow"""
        print("\n" + "="*80)
        print("🔬 Feature-Word Association Analysis")
        print("="*80)

        # Sample experiments
        json_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
        experiments = self.sample_experiments(json_file, self.sample_size)

        # Process experiments
        print(f"\n📊 Processing {len(experiments)} experiments...")
        for i, exp in enumerate(tqdm(experiments, desc="Experiments")):
            self.process_experiment(exp)

            # Clear cache periodically
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

        # Analyze associations
        results = self.analyze_associations()

        # Save results
        output_file = '/data/llm_addiction/analysis/feature_word_associations.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {output_file}")
        print(f"   Total features analyzed: {len(results)}")
        print(f"   Total unique tokens tracked: {len(self.token_activations)}")

        # Print sample results
        print("\n📋 Sample results (first 3 features):")
        for i, (feat_name, data) in enumerate(list(results.items())[:3]):
            print(f"\n  {feat_name} ({data['feature_info']['type']}):")
            for j, token_info in enumerate(data['top_tokens'][:5]):
                print(f"    {j+1}. '{token_info['token_text']}' - "
                      f"mean: {token_info['mean_activation']:.3f}, "
                      f"freq: {token_info['frequency']}")

        return output_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--sample-size', type=int, default=320, help='Number of experiments to sample')
    args = parser.parse_args()

    random.seed(42)
    device = f'cuda:{args.gpu}'
    experiment = FeatureWordAssociation(device=device, sample_size=args.sample_size)
    output_file = experiment.run_experiment()
    print("\n🎉 Feature-word association analysis complete!")
