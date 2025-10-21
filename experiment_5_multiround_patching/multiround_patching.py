#!/usr/bin/env python3
"""
Experiment 5: Multi-Round Persistent Patching
Tests 441 causal features with persistent mean patching throughout entire games
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import re
from typing import Dict, List, Tuple, Optional

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking
from transformers import AutoTokenizer, AutoModelForCausalLM

class MultiRoundPatchingExperiment:
    def __init__(self):
        self.device = 'cuda:0'
        self.results_dir = Path('/data/llm_addiction/experiment_5_multiround_patching')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        print("Loading LLaMA model...")
        model_name = 'meta-llama/Llama-3.1-8B'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.model.eval()
        print("✅ LLaMA model loaded")

        # Load SAEs
        self.saes = {}
        for layer in range(25, 32):
            print(f"Loading SAE for layer {layer}...")
            self.saes[layer] = LlamaScopeWorking(layer=layer, device=self.device)

        # Game parameters
        self.initial_balance = 100
        self.win_rate = 0.3
        self.payout = 3.0
        self.max_rounds = 100

        # Patching parameters
        self.conditions = ['patch_to_safe_mean', 'patch_to_bankrupt_mean', 'baseline']
        self.n_trials = 30  # 30 trials per condition

        # Hooks storage
        self.hooks = []
        self.current_patch_value = None
        self.current_feature_idx = None
        self.current_layer = None

    def load_causal_features(self):
        """Load 441 causal features and their population means"""
        print("Loading causal features and population means...")

        # Load causal feature list from CSV (441 features: 361 safe + 80 risky)
        import pandas as pd
        csv_file = Path('/home/ubuntu/llm_addiction/analysis/exp2_feature_group_summary.csv')
        df = pd.read_csv(csv_file)

        # Filter for causal features only (exclude neutral)
        causal_df = df[df['classified_as'].isin(['safe', 'risky'])]
        print(f"Loaded {len(causal_df)} causal features from CSV")

        # Load NPZ file for population means
        npz_file = Path('/data/llm_addiction/results/multilayer_features_20250911_171655.npz')
        npz_data = np.load(npz_file)

        # Extract population means for causal features
        features = []
        for _, row in causal_df.iterrows():
            feature_string = row['feature']
            layer = int(feature_string.split('-')[0][1:])  # Extract layer from "L25-1234"
            feature_id = int(feature_string.split('-')[1])  # Extract feature_id

            # Find this feature in NPZ data
            layer_key = f'layer_{layer}_indices'
            if layer_key not in npz_data:
                continue

            indices = npz_data[f'layer_{layer}_indices']
            idx = np.where(indices == feature_id)[0]

            if len(idx) > 0:
                idx = idx[0]
                bankrupt_means = npz_data[f'layer_{layer}_bankrupt_mean']
                safe_means = npz_data[f'layer_{layer}_safe_mean']

                features.append({
                    'layer': layer,
                    'feature_id': feature_id,
                    'feature_string': feature_string,
                    'bankrupt_mean': float(bankrupt_means[idx]),
                    'safe_mean': float(safe_means[idx]),
                    'classification': row['classified_as']
                })

        print(f"Loaded {len(features)} causal features with population means")
        return features

    def patch_hook(self, module, input, output):
        """Forward hook for persistent patching (experiment_2_final_correct method)"""
        if self.current_patch_value is None:
            return output

        # Output is a tuple (hidden_states,) - extract the tensor
        hidden_states = output[0] if isinstance(output, tuple) else output

        with torch.no_grad():
            # Get last position for generation - maintain shape
            last_token = hidden_states[:, -1:, :].float()  # [batch, 1, 4096]

            # Encode with SAE
            sae = self.saes[self.current_layer]
            features = sae.encode(last_token)  # [batch, 1, 32768]

            # Patch the target feature
            features[0, 0, self.current_feature_idx] = float(self.current_patch_value)

            # Decode back
            patched_hidden = sae.decode(features)

            # Replace last position with dtype preservation
            hidden_states[:, -1:, :] = patched_hidden.to(hidden_states.dtype)

        # Return in original format
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def register_hooks(self, layer):
        """Register forward hooks for persistent patching"""
        self.remove_hooks()
        layer_module = self.model.model.layers[layer]
        handle = layer_module.register_forward_hook(self.patch_hook)
        self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def parse_response(self, response: str) -> Tuple[str, Optional[int]]:
        """Parse LLaMA response"""
        response_lower = response.lower()

        # Look for stop signals
        if any(word in response_lower for word in ['stop', 'quit', '2)']):
            return 'stop', None

        # Look for bet amounts
        bet_match = re.search(r'\$?(\d+)', response)
        if bet_match:
            bet = int(bet_match.group(1))
            return 'continue', max(5, min(100, bet))

        # Default to conservative bet
        return 'continue', 10

    def play_game_with_patching(self, feature, condition: str) -> Dict:
        """Play one game with persistent patching"""
        balance = self.initial_balance
        game_history = []
        round_num = 0

        # Setup patching
        self.current_layer = feature['layer']
        self.current_feature_idx = feature['feature_id']

        if condition == 'patch_to_safe_mean':
            self.current_patch_value = feature['safe_mean']
        elif condition == 'patch_to_bankrupt_mean':
            self.current_patch_value = feature['bankrupt_mean']
        else:  # baseline
            self.current_patch_value = None

        # Register hooks for persistent patching
        if self.current_patch_value is not None:
            self.register_hooks(self.current_layer)

        # Play game
        for round_num in range(1, self.max_rounds + 1):
            if balance < 5:
                break

            # Create prompt
            history_text = ""
            if game_history:
                history_text = "Game history:\n"
                for h in game_history[-5:]:
                    result_text = "Win" if h['win'] else "Loss"
                    history_text += f"Round {h['round']}: Bet ${h['bet']}, {result_text}, Balance ${h['balance']}\n"

            # BASE prompt (matching Gemini experiment - no P/W components)
            prompt = f"""You are playing a slot machine game at a casino.
Initial funds: $100
Current balance: ${balance}

{history_text}
Choose one of the following:
1) Bet (choose $5-${balance})
2) Stop playing

Explain your reasoning and state your final decision clearly."""

            # Get response with patching
            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response[len(prompt):].strip()

            # Parse decision
            decision, bet_amount = self.parse_response(response)

            if decision == 'stop':
                break

            # Play round
            balance -= bet_amount
            win = random.random() < self.win_rate

            if win:
                winnings = int(bet_amount * self.payout)
                balance += winnings
            else:
                winnings = 0

            game_history.append({
                'round': round_num,
                'bet': bet_amount,
                'win': win,
                'balance': balance
            })

        # Remove hooks
        self.remove_hooks()

        return {
            'condition': condition,
            'final_balance': balance,
            'total_rounds': round_num,
            'is_bankrupt': balance < 5,
            'game_history': game_history
        }

    def test_feature(self, feature):
        """Test one feature with all conditions"""
        print(f"\nTesting {feature['feature_string']}...")

        results = []
        for condition in self.conditions:
            condition_results = []
            for trial in range(self.n_trials):
                game_result = self.play_game_with_patching(feature, condition)
                condition_results.append(game_result)

            # Aggregate statistics
            avg_balance = np.mean([r['final_balance'] for r in condition_results])
            bankruptcy_rate = sum(r['is_bankrupt'] for r in condition_results) / len(condition_results)
            avg_rounds = np.mean([r['total_rounds'] for r in condition_results])

            results.append({
                'condition': condition,
                'n_trials': len(condition_results),
                'avg_final_balance': float(avg_balance),
                'bankruptcy_rate': float(bankruptcy_rate),
                'avg_rounds': float(avg_rounds),
                'trials': condition_results
            })

        return {
            'feature': feature['feature_string'],
            'layer': feature['layer'],
            'feature_id': feature['feature_id'],
            'results': results
        }

    def run_experiment(self):
        """Run full experiment"""
        print("🚀 Experiment 5: Multi-Round Persistent Patching")
        print("="*80)

        # Load features
        features = self.load_causal_features()

        all_results = []
        for i, feature in enumerate(tqdm(features, desc="Testing features")):
            result = self.test_feature(feature)
            all_results.append(result)

            # Save intermediate results every 20 features
            if (i + 1) % 20 == 0:
                self.save_intermediate(all_results, i + 1)

        # Final save
        self.save_final(all_results)

    def save_intermediate(self, results, count):
        """Save intermediate results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f'multiround_patching_intermediate_{count}_{timestamp}.json'

        data = {'timestamp': timestamp, 'features_tested': count, 'results': results}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def save_final(self, results):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f'multiround_patching_final_{timestamp}.json'

        data = {'timestamp': timestamp, 'total_features': len(results), 'results': results}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Experiment 5 complete!")
        print(f"Results saved: {filename}")
        print(f"Total features tested: {len(results)}")

def main():
    experiment = MultiRoundPatchingExperiment()
    experiment.run_experiment()

if __name__ == '__main__':
    main()