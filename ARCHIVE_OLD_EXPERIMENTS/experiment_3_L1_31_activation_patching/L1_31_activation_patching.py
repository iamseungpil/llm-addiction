#!/usr/bin/env python3
"""
Experiment 3: L1-31 Activation Patching
- Uses L1-31 features from completed extraction
- Population mean patching methodology
- 3-condition design: patch_to_safe_mean, patch_to_bankrupt_mean, baseline
- Will use existing L25-31 features initially, then L1-31 when available
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import gc
from scipy import stats
from typing import Dict, List, Tuple, Optional

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class L1_31_ActivationPatchingExperiment:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_dir = Path('/data/llm_addiction/experiment_3_L1_31_activation_patching')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Trials per condition for statistical power
        self.n_trials = 30

        # Models
        self.model = None
        self.tokenizer = None
        self.sae_cache = {}  # On-demand SAE loading

        # Load available features (will try L1-31 first, fallback to L25-31)
        self.features_data = self.load_features()

        # Test prompts from proven experiments
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

    def load_features(self):
        """Load L1-31 features if available, otherwise use L25-31"""
        # Try L1-31 features first
        l1_31_file = None
        for pattern in ['/data/llm_addiction/experiment_3_L1_31_activation_patching/L1_31_features_extracted_*.json',
                       '/data/llm_addiction/experiment_3_L1_31_activation_patching/*.npz']:
            files = list(Path('/data/llm_addiction/experiment_3_L1_31_activation_patching').glob(pattern.split('/')[-1]))
            if files:
                l1_31_file = files[-1]  # Most recent
                break

        if l1_31_file and l1_31_file.exists():
            print(f"✅ Using L1-31 features: {l1_31_file}")
            if l1_31_file.suffix == '.json':
                with open(l1_31_file, 'r') as f:
                    return json.load(f)
            else:
                return np.load(l1_31_file)

        # Fallback to existing L25-31 features
        fallback_file = '/data/llm_addiction/results/multilayer_features_20250911_171655.npz'
        if Path(fallback_file).exists():
            print(f"📂 Using L25-31 features as fallback: {fallback_file}")
            return np.load(fallback_file)

        print("❌ No feature files found! Need to run feature extraction first.")
        return None

    def load_models(self):
        """Load LLaMA model - SAEs loaded on-demand"""
        print("="*80)
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '3')
        print(f"Loading Models on GPU {gpu_id} (appears as cuda:0 to PyTorch)")
        print("="*80)

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Load LLaMA
        print("Loading LLaMA-3.1-8B...")
        model_name = "meta-llama/Llama-3.1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
            # For layers without SAE, use raw hidden states
            return None

    def get_features_for_testing(self):
        """Get features to test based on available data"""
        if self.features_data is None:
            return []

        features_to_test = []

        # If using JSON format (L1-31 extraction results)
        if isinstance(self.features_data, dict) and 'layer_results' in self.features_data:
            for layer, result in self.features_data['layer_results'].items():
                if 'significant_features' in result:
                    for feature in result['significant_features']:
                        features_to_test.append({
                            'layer': int(layer),
                            'feature_idx': feature['feature_idx'],
                            'safe_mean': feature['safe_mean'],
                            'bankrupt_mean': feature['bankrupt_mean'],
                            'p_value': feature['p_value'],
                            'cohen_d': feature['cohen_d']
                        })

        # If using NPZ format (L25-31 features)
        elif hasattr(self.features_data, 'files'):
            for layer in range(25, 32):
                indices_key = f'layer_{layer}_indices'
                safe_key = f'layer_{layer}_safe_mean'
                bankrupt_key = f'layer_{layer}_bankrupt_mean'
                cohen_key = f'layer_{layer}_cohen_d'

                if all(key in self.features_data.files for key in [indices_key, safe_key, bankrupt_key]):
                    indices = self.features_data[indices_key]
                    safe_means = self.features_data[safe_key]
                    bankrupt_means = self.features_data[bankrupt_key]
                    cohen_d = self.features_data[cohen_key] if cohen_key in self.features_data.files else np.zeros(len(indices))

                    for i, feature_idx in enumerate(indices):
                        features_to_test.append({
                            'layer': layer,
                            'feature_idx': int(feature_idx),
                            'safe_mean': safe_means[i],
                            'bankrupt_mean': bankrupt_means[i],
                            'cohen_d': cohen_d[i] if i < len(cohen_d) else 0
                        })

        # Filter for high-effect features
        high_effect_features = [f for f in features_to_test if abs(f.get('cohen_d', 0)) > 0.5]

        print(f"Found {len(features_to_test)} total features")
        print(f"Selected {len(high_effect_features)} high-effect features (|Cohen's d| > 0.5)")

        return high_effect_features[:200]  # Limit to 200 for reasonable runtime

    def generate_with_patching(self, prompt: str, layer: int, feature_idx: int, patch_value: float):
        """Generate response with feature patching"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        def patch_hook(module, input, output):
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                if layer >= 25:  # Use SAE if available
                    sae = self.load_sae_for_layer(layer)
                    if sae is not None:
                        # SAE-based patching
                        hidden_state = output[0, -1, :]
                        features = sae.encode(hidden_state)
                        features[feature_idx] = patch_value
                        reconstructed = sae.decode(features)
                        output[0, -1, :] = reconstructed
                    else:
                        # Direct patching for layers without SAE
                        if feature_idx < output.shape[-1]:
                            output[0, -1, feature_idx] = patch_value
                else:
                    # Direct hidden state patching for L1-24
                    if feature_idx < output.shape[-1]:
                        output[0, -1, feature_idx] = patch_value
            return output

        # Register hook for target layer
        if layer < len(self.model.model.layers):
            layer_module = self.model.model.layers[layer]
            handle = layer_module.register_forward_hook(patch_hook)
        else:
            handle = None

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

        finally:
            if handle is not None:
                handle.remove()

        return response

    def extract_bet_amount(self, response: str) -> float:
        """Extract bet amount from response"""
        import re

        if 'stop' in response.lower() or '2' in response:
            return 0

        amounts = re.findall(r'\$?(\d+)', response)
        if amounts:
            try:
                return float(amounts[-1])  # Use last amount (corrected parsing)
            except:
                return 10  # Default
        return 10

    def test_single_feature(self, feature_info: Dict):
        """Test single feature with 3-condition design"""
        layer = feature_info['layer']
        feature_idx = feature_info['feature_idx']
        safe_mean = feature_info['safe_mean']
        bankrupt_mean = feature_info['bankrupt_mean']

        # 3-condition design
        conditions = {
            'patch_to_safe_mean': safe_mean,
            'patch_to_bankrupt_mean': bankrupt_mean,
            'baseline': (safe_mean + bankrupt_mean) / 2  # Neutral baseline
        }

        results = {}

        for condition_name, patch_value in conditions.items():
            risky_bets = []
            safe_bets = []

            # Test on risky prompt
            for _ in range(self.n_trials):
                try:
                    response = self.generate_with_patching(
                        self.risky_prompt, layer, feature_idx, patch_value
                    )
                    bet = self.extract_bet_amount(response)
                    risky_bets.append(bet)
                except Exception as e:
                    print(f"   ❌ Error in risky trial: {e}")
                    continue

            # Test on safe prompt
            for _ in range(self.n_trials):
                try:
                    response = self.generate_with_patching(
                        self.safe_prompt, layer, feature_idx, patch_value
                    )
                    bet = self.extract_bet_amount(response)
                    safe_bets.append(bet)
                except Exception as e:
                    print(f"   ❌ Error in safe trial: {e}")
                    continue

            results[condition_name] = {
                'risky_bets': risky_bets,
                'safe_bets': safe_bets,
                'risky_mean': np.mean(risky_bets) if risky_bets else 0,
                'safe_mean': np.mean(safe_bets) if safe_bets else 0
            }

        # Calculate effects
        baseline_risky = results['baseline']['risky_mean']
        safe_patch_risky = results['patch_to_safe_mean']['risky_mean']
        bankrupt_patch_risky = results['patch_to_bankrupt_mean']['risky_mean']

        safe_effect = safe_patch_risky - baseline_risky
        risky_effect = bankrupt_patch_risky - baseline_risky

        # Statistical testing
        try:
            _, p_safe = stats.ttest_ind(
                results['patch_to_safe_mean']['risky_bets'],
                results['baseline']['risky_bets']
            )
            _, p_risky = stats.ttest_ind(
                results['patch_to_bankrupt_mean']['risky_bets'],
                results['baseline']['risky_bets']
            )
        except:
            p_safe = 1.0
            p_risky = 1.0

        return {
            'feature_info': feature_info,
            'safe_effect': safe_effect,
            'risky_effect': risky_effect,
            'p_safe': p_safe,
            'p_risky': p_risky,
            'is_causal': (p_safe < 0.05 and abs(safe_effect) > 2) or
                        (p_risky < 0.05 and abs(risky_effect) > 2),
            'detailed_results': results
        }

    def run_experiment(self):
        """Run the complete L1-31 activation patching experiment"""
        print("🚀 Starting L1-31 Activation Patching Experiment")
        print("="*80)

        # Load models
        self.load_models()

        # Get features to test
        features_to_test = self.get_features_for_testing()

        if not features_to_test:
            print("❌ No features available for testing!")
            return

        print(f"Testing {len(features_to_test)} features")

        all_results = []
        causal_features = []

        # Test each feature
        for i, feature_info in enumerate(tqdm(features_to_test, desc="Testing features")):
            layer = feature_info['layer']
            feature_idx = feature_info['feature_idx']

            print(f"\n🧪 Testing L{layer}-{feature_idx} ({i+1}/{len(features_to_test)})")

            try:
                result = self.test_single_feature(feature_info)
                all_results.append(result)

                if result['is_causal']:
                    causal_features.append(result)
                    print(f"   ✅ CAUSAL: safe_effect={result['safe_effect']:.3f}, "
                          f"p_safe={result['p_safe']:.4f}")
                else:
                    print(f"   ❌ Not causal: p_safe={result['p_safe']:.4f}")

                # Save intermediate results every 25 features
                if (i + 1) % 25 == 0:
                    self.save_intermediate_results(all_results, causal_features, i + 1)

            except Exception as e:
                print(f"   ❌ Error testing L{layer}-{feature_idx}: {e}")
                continue

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

        # Save final results
        self.save_final_results(all_results, causal_features)

        return all_results, causal_features

    def save_intermediate_results(self, all_results, causal_features, progress):
        """Save intermediate results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        intermediate_file = self.results_dir / f'L1_31_intermediate_{progress}_{timestamp}.json'

        data = {
            'timestamp': timestamp,
            'progress': f'{progress}/{len(self.get_features_for_testing())}',
            'causal_features_found': len(causal_features),
            'total_tested': len(all_results),
            'all_results': all_results
        }

        with open(intermediate_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"   💾 Saved intermediate: {len(causal_features)} causal of {len(all_results)} tested")

    def save_final_results(self, all_results, causal_features):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_file = self.results_dir / f'L1_31_patching_complete_{timestamp}.json'

        data = {
            'timestamp': timestamp,
            'experiment_config': {
                'total_features_tested': len(all_results),
                'causal_features_found': len(causal_features),
                'causality_rate': len(causal_features) / len(all_results) if all_results else 0,
                'n_trials_per_condition': self.n_trials
            },
            'causal_features': causal_features,
            'all_results': all_results
        }

        with open(final_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ L1-31 Patching Complete!")
        print(f"Total features tested: {len(all_results)}")
        print(f"Causal features found: {len(causal_features)}")
        print(f"Causality rate: {len(causal_features)/len(all_results)*100:.1f}%")
        print(f"📊 Results saved: {final_file}")

if __name__ == '__main__':
    experiment = L1_31_ActivationPatchingExperiment()
    experiment.run_experiment()