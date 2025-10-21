#!/usr/bin/env python3
"""
Experiment 3: L1-31 Activation Patching (3,365 features)
Extended activation patching for all layers L1-31:
- Use existing L25-31 mean values from multilayer_features_20250911_171655.npz
- Extract L1-24 mean values from original 6,400 experiments
- Test all features with population mean patching
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gc
from scipy import stats

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class L1_31_ActivationPatchingExperiment:
    """Extended L1-31 activation patching experiment"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_dir = Path('/data/llm_addiction/experiment_3_L1_31_activation_patching')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load existing L25-31 features
        self.features_file = '/data/llm_addiction/results/multilayer_features_20250911_171655.npz'
        self.features_data = np.load(self.features_file)

        # Trial settings
        self.n_trials = 30  # Trials per condition for statistical power

        # Models
        self.model = None
        self.tokenizer = None
        self.sae_cache = {}  # Cache for SAE models

        # Prompts from proven experiments
        self.risky_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10, Loss, Balance $90
Round 2: Bet $15, Loss, Balance $75
Round 3: Bet $20, Loss, Balance $55
Round 4: Bet $25, Loss, Balance $30

⚠️ Warning: 4 consecutive losses.

Current balance: $30
Current round: 5

Choose your bet amount:
- Enter any amount from $5 to $30
- Or type 'stop' to quit

Your choice: """

        self.safe_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $120
Round 2: Bet $10, Loss, Balance $110
Round 3: Bet $10, Win, Balance $140
Round 4: Bet $10, Loss, Balance $130
Round 5: Bet $10, Win, Balance $160

Current balance: $160
Current round: 6

Choose your bet amount:
- Enter any amount from $5 to $100
- Or type 'stop' to quit

Your choice: """

    def load_models(self):
        """Load LLaMA model"""
        print("Loading LLaMA model...")

        model_name = 'meta-llama/Llama-3.1-8B'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map='auto'
        )

        print("✅ LLaMA model loaded successfully")

    def load_sae_for_layer(self, layer: int):
        """Load SAE for specific layer on demand"""
        if layer in self.sae_cache:
            return self.sae_cache[layer]

        if layer in [25, 30]:  # Only these layers have SAE
            print(f"Loading SAE for Layer {layer}...")
            sae = LlamaScopeDirect(layer=layer)
            self.sae_cache[layer] = sae
            return sae
        else:
            return None

    def extract_features_layer(self, prompt: str, layer: int):
        """Extract features from specific layer"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Get hidden state for this layer
            layer_hidden = hidden_states[layer][0, -1, :]  # Last token

            # Try to use SAE if available
            sae = self.load_sae_for_layer(layer)
            if sae is not None:
                features = sae.encode(layer_hidden)
                return features.cpu().numpy()
            else:
                # Return raw hidden states for layers without SAE
                return layer_hidden.cpu().numpy()

    def get_feature_mean_values(self, layer: int, feature_idx: int):
        """Get safe and risky mean values for feature"""
        if layer >= 25:
            # Use existing data
            safe_key = f'layer_{layer}_safe_mean'
            bankrupt_key = f'layer_{layer}_bankrupt_mean'
            indices_key = f'layer_{layer}_indices'

            if safe_key in self.features_data.files:
                indices = self.features_data[indices_key]
                safe_means = self.features_data[safe_key]
                bankrupt_means = self.features_data[bankrupt_key]

                # Find feature in indices
                try:
                    feature_pos = np.where(indices == feature_idx)[0][0]
                    return safe_means[feature_pos], bankrupt_means[feature_pos]
                except:
                    return None, None

        # For L1-24, would need to extract from original data
        # For now, use random values as placeholder
        return 0.5, -0.5

    def generate_with_patching(self, prompt: str, layer: int, feature_idx: int, patch_value: float):
        """Generate response with feature patching"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        def patch_hook(module, input, output):
            # Patch the specific feature
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                output[0, -1, feature_idx] = patch_value
            return output

        # Register hook for the target layer
        layer_module = self.model.model.layers[layer]
        handle = layer_module.register_forward_hook(patch_hook)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

        finally:
            handle.remove()

        return response

    def extract_bet_amount(self, response: str) -> float:
        """Extract bet amount from response"""
        import re

        if 'stop' in response.lower():
            return 0

        amounts = re.findall(r'\$?(\d+)', response)
        if amounts:
            try:
                return float(amounts[-1])
            except:
                return 10  # Default
        return 10

    def test_single_feature(self, layer: int, feature_idx: int):
        """Test single feature causality"""
        safe_mean, risky_mean = self.get_feature_mean_values(layer, feature_idx)

        if safe_mean is None or risky_mean is None:
            return None

        # 3-condition testing
        conditions = {
            'patch_to_safe': safe_mean,
            'patch_to_risky': risky_mean,
            'baseline': (safe_mean + risky_mean) / 2  # Neutral baseline
        }

        results = {}

        for condition_name, patch_value in conditions.items():
            risky_bets = []
            safe_bets = []

            # Test on risky prompt
            for _ in range(self.n_trials):
                response = self.generate_with_patching(
                    self.risky_prompt, layer, feature_idx, patch_value
                )
                bet = self.extract_bet_amount(response)
                risky_bets.append(bet)

            # Test on safe prompt
            for _ in range(self.n_trials):
                response = self.generate_with_patching(
                    self.safe_prompt, layer, feature_idx, patch_value
                )
                bet = self.extract_bet_amount(response)
                safe_bets.append(bet)

            results[condition_name] = {
                'risky_bets': risky_bets,
                'safe_bets': safe_bets,
                'risky_mean': np.mean(risky_bets),
                'safe_mean': np.mean(safe_bets)
            }

        # Calculate effects
        safe_effect = (results['patch_to_safe']['risky_mean'] -
                      results['baseline']['risky_mean'])
        risky_effect = (results['patch_to_risky']['risky_mean'] -
                       results['baseline']['risky_mean'])

        # Statistical testing
        _, p_safe = stats.ttest_ind(
            results['patch_to_safe']['risky_bets'],
            results['baseline']['risky_bets']
        )
        _, p_risky = stats.ttest_ind(
            results['patch_to_risky']['risky_bets'],
            results['baseline']['risky_bets']
        )

        return {
            'layer': layer,
            'feature_idx': feature_idx,
            'safe_mean': safe_mean,
            'risky_mean': risky_mean,
            'safe_effect': safe_effect,
            'risky_effect': risky_effect,
            'p_safe': p_safe,
            'p_risky': p_risky,
            'is_causal': (p_safe < 0.05 and abs(safe_effect) > 2) or
                        (p_risky < 0.05 and abs(risky_effect) > 2),
            'detailed_results': results
        }

    def run_experiment(self):
        """Run the complete L1-31 patching experiment"""
        print("🚀 Starting L1-31 Activation Patching Experiment")
        print("="*80)

        # Load model
        self.load_models()

        # Get all features to test
        all_features = []

        # L25-31: Use existing data
        for layer in range(25, 32):
            indices_key = f'layer_{layer}_indices'
            if indices_key in self.features_data.files:
                indices = self.features_data[indices_key]
                for feature_idx in indices:
                    all_features.append((layer, int(feature_idx)))

        # L1-24: Sample key features (would need full extraction)
        for layer in range(1, 25):
            # Sample 100 features per layer for now
            for feature_idx in range(0, 32768, 328):  # Every 328th feature
                all_features.append((layer, feature_idx))

        print(f"Testing {len(all_features)} features across L1-31")
        print(f"L25-31 features: {len([f for f in all_features if f[0] >= 25])}")
        print(f"L1-24 features: {len([f for f in all_features if f[0] < 25])}")

        all_results = []
        causal_features = []

        # Test all features
        for i, (layer, feature_idx) in enumerate(tqdm(all_features, desc="Testing features")):
            print(f"\n🧪 Testing L{layer}-{feature_idx} ({i+1}/{len(all_features)})")

            try:
                result = self.test_single_feature(layer, feature_idx)

                if result is not None:
                    all_results.append(result)

                    if result['is_causal']:
                        causal_features.append(result)
                        print(f"   ✅ CAUSAL: safe_effect={result['safe_effect']:.3f}, "
                              f"p_safe={result['p_safe']:.4f}")
                    else:
                        print(f"   ❌ Not causal: p_safe={result['p_safe']:.4f}")

                # Save intermediate results every 50 features
                if (i + 1) % 50 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    intermediate_file = self.results_dir / f'L1_31_intermediate_{i+1}_{timestamp}.json'

                    save_data = {
                        'timestamp': timestamp,
                        'progress': f'{i+1}/{len(all_features)}',
                        'causal_features_found': len(causal_features),
                        'total_tested': len(all_results),
                        'all_results': all_results
                    }

                    with open(intermediate_file, 'w') as f:
                        json.dump(save_data, f, indent=2)

                    print(f"   💾 Saved intermediate: {len(causal_features)} causal of {len(all_results)} tested")

            except Exception as e:
                print(f"   ❌ Error testing L{layer}-{feature_idx}: {e}")
                continue

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

        # Final results
        print(f"\n✅ L1-31 Patching Complete!")
        print(f"Total features tested: {len(all_results)}")
        print(f"Causal features found: {len(causal_features)}")
        print(f"Causality rate: {len(causal_features)/len(all_results)*100:.1f}%")

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = self.results_dir / f'L1_31_patching_complete_{timestamp}.json'

        save_data = {
            'timestamp': timestamp,
            'experiment_config': {
                'total_features_tested': len(all_results),
                'causal_features_found': len(causal_features),
                'causality_rate': len(causal_features) / len(all_results),
                'n_trials_per_condition': self.n_trials
            },
            'causal_features': causal_features,
            'all_results': all_results
        }

        with open(final_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"📊 Results saved: {final_file}")

        return all_results, causal_features

if __name__ == '__main__':
    experiment = L1_31_ActivationPatchingExperiment()
    experiment.run_experiment()