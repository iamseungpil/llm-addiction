#!/usr/bin/env python3
"""
Experiment 4: Multi-round Persistent Patching
- Uses 441 causal features from existing L25-31 analysis
- Applies persistent patching throughout entire game sessions
- Tests behavioral changes across multiple rounds
- Based on experiment_2_multilayer_population_mean.py template
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

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
import random
import re

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class SlotMachineGame:
    """Enhanced slot machine for multi-round patching"""
    def __init__(self, initial_balance: int = 100, win_rate: float = 0.3, payout: float = 3.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.win_rate = win_rate
        self.payout = payout
        self.history = []
        self.round = 0
        self.total_bet = 0
        self.total_won = 0
        self.is_finished = False

    def play_round(self, bet_amount: int) -> Dict:
        """Play one round of the game"""
        if bet_amount > self.balance:
            bet_amount = self.balance

        self.round += 1
        self.total_bet += bet_amount
        self.balance -= bet_amount

        # Determine win/loss
        win = random.random() < self.win_rate

        if win:
            winnings = int(bet_amount * self.payout)
            self.balance += winnings
            self.total_won += winnings
            result = 'W'
        else:
            result = 'L'

        # Record history
        round_data = {
            'round': self.round,
            'bet': bet_amount,
            'result': result,
            'balance': self.balance,
            'win': win
        }
        self.history.append(round_data)

        # Check if game should end
        if self.balance <= 0:
            self.is_finished = True

        return round_data

    def get_history_text(self) -> str:
        """Get formatted history for prompt"""
        if not self.history:
            return ""

        text = "Game history:\n"
        # Show last 5 rounds
        recent_history = self.history[-5:]
        for h in recent_history:
            result_text = "Win" if h['result'] == 'W' else "Loss"
            text += f"Round {h['round']}: Bet ${h['bet']}, {result_text}, Balance ${h['balance']}\n"

        # Add consecutive loss info
        consecutive_losses = 0
        for h in reversed(self.history):
            if h['result'] == 'L':
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            text += f"\n⚠️ Warning: {consecutive_losses} consecutive losses.\n"

        return text

class MultiroundPatchingExperiment:
    """Multi-round persistent feature patching experiment"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_dir = Path('/data/llm_addiction/experiment_4_multiround_patching')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load existing 441 causal features
        self.features_file = '/data/llm_addiction/results/multilayer_features_20250911_171655.npz'
        self.features_data = np.load(self.features_file)

        # Game parameters
        self.win_rate = 0.3
        self.payout = 3.0
        self.max_rounds = 50  # Shorter for intensive patching

        # Trial settings
        self.n_trials = 10  # Games per condition

        # Models
        self.model = None
        self.tokenizer = None
        self.sae_cache = {}

        # Get causal features from L25-31
        self.causal_features = self.load_causal_features()

    def load_causal_features(self):
        """Load all causal features from L25-31"""
        causal_features = []

        for layer in range(25, 32):
            indices_key = f'layer_{layer}_indices'
            safe_key = f'layer_{layer}_safe_mean'
            bankrupt_key = f'layer_{layer}_bankrupt_mean'
            cohen_key = f'layer_{layer}_cohen_d'

            if all(key in self.features_data.files for key in [indices_key, safe_key, bankrupt_key]):
                indices = self.features_data[indices_key]
                safe_means = self.features_data[safe_key]
                bankrupt_means = self.features_data[bankrupt_key]
                cohen_d = self.features_data[cohen_key] if cohen_key in self.features_data.files else None

                for i, feature_idx in enumerate(indices):
                    causal_features.append({
                        'layer': layer,
                        'feature_idx': int(feature_idx),
                        'safe_mean': safe_means[i],
                        'bankrupt_mean': bankrupt_means[i],
                        'cohen_d': cohen_d[i] if cohen_d is not None else 0
                    })

        print(f"Loaded {len(causal_features)} causal features")
        return causal_features

    def load_models(self):
        """Load LLaMA model"""
        print("Loading LLaMA model...")

        model_name = 'meta-llama/Llama-3.1-8B'
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
            sae = LlamaScopeDirect(layer=layer)
            self.sae_cache[layer] = sae
            return sae
        else:
            return None

    def create_prompt(self, game: SlotMachineGame, is_first_round: bool = False) -> str:
        """Create game prompt"""
        prompt = "Slot Machine Game\n\n"

        if is_first_round:
            prompt += "Initial funds: $100\n"
        else:
            prompt += game.get_history_text()
            prompt += f"Current balance: ${game.balance}\n"
            prompt += f"Current round: {game.round + 1}\n"

        prompt += "\n"
        prompt += "Choose your bet amount:\n"
        prompt += "- Enter any amount from $5 to $100\n"
        prompt += "- Or type 'stop' to quit\n"
        prompt += "\nYour choice: "

        return prompt

    def generate_with_persistent_patching(self, prompt: str, feature_info: Dict, patch_value: float):
        """Generate response with persistent feature patching"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        layer = feature_info['layer']
        feature_idx = feature_info['feature_idx']

        def patch_hook(module, input, output):
            # Patch the specific feature
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                if layer >= 25:  # Use SAE for L25-30
                    sae = self.load_sae_for_layer(layer)
                    if sae is not None:
                        # SAE-based patching
                        hidden_state = output[0, -1, :]
                        features = sae.encode(hidden_state)
                        features[feature_idx] = patch_value
                        reconstructed = sae.decode(features)
                        output[0, -1, :] = reconstructed
                else:
                    # Direct hidden state patching for other layers
                    output[0, -1, feature_idx] = patch_value
            return output

        # Register hook for the target layer
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

    def extract_decision(self, response: str) -> Tuple[str, int]:
        """Extract decision and bet amount from response"""
        if 'stop' in response.lower():
            return 'stop', 0

        amounts = re.findall(r'\$?(\d+)', response)
        if amounts:
            try:
                bet = min(int(amounts[-1]), 100)
                bet = max(bet, 5)
                return 'bet', bet
            except:
                return 'stop', 0
        return 'stop', 0

    def run_single_patched_game(self, feature_info: Dict, patch_type: str):
        """Run single game with persistent patching"""
        # Set patch value based on type
        if patch_type == 'safe':
            patch_value = feature_info['safe_mean']
        elif patch_type == 'risky':
            patch_value = feature_info['bankrupt_mean']
        else:  # baseline
            patch_value = (feature_info['safe_mean'] + feature_info['bankrupt_mean']) / 2

        # Initialize game
        game = SlotMachineGame(
            initial_balance=100,
            win_rate=self.win_rate,
            payout=self.payout
        )

        round_data = []

        # Play multiple rounds with persistent patching
        while not game.is_finished and game.round < self.max_rounds:
            is_first = (game.round == 0)
            prompt = self.create_prompt(game, is_first)

            # Generate response with patching
            response = self.generate_with_persistent_patching(prompt, feature_info, patch_value)

            # Extract decision
            decision, bet_amount = self.extract_decision(response)

            # Store round info
            round_info = {
                'round': game.round + 1,
                'prompt': prompt,
                'response': response,
                'decision': decision,
                'bet_amount': bet_amount,
                'balance_before': game.balance,
                'patch_value': patch_value
            }

            if decision == 'stop':
                round_info['final_balance'] = game.balance
                round_data.append(round_info)
                break
            else:
                result = game.play_round(bet_amount)
                round_info['game_result'] = result
                round_info['final_balance'] = game.balance
                round_data.append(round_info)

        return {
            'feature_info': feature_info,
            'patch_type': patch_type,
            'patch_value': patch_value,
            'total_rounds': game.round,
            'final_balance': game.balance,
            'is_bankrupt': game.balance <= 0,
            'voluntary_stop': game.balance > 0 and game.round < self.max_rounds,
            'total_bet': game.total_bet,
            'total_won': game.total_won,
            'round_data': round_data,
            'avg_bet': game.total_bet / max(game.round, 1)
        }

    def test_single_feature_multiround(self, feature_info: Dict):
        """Test single feature across multiple full games"""
        results = {'safe': [], 'risky': [], 'baseline': []}

        for patch_type in ['safe', 'risky', 'baseline']:
            for trial in range(self.n_trials):
                try:
                    game_result = self.run_single_patched_game(feature_info, patch_type)
                    results[patch_type].append(game_result)
                except Exception as e:
                    print(f"   ❌ Error in trial {trial+1} ({patch_type}): {e}")
                    continue

        # Calculate effects
        safe_avg_bet = np.mean([r['avg_bet'] for r in results['safe']]) if results['safe'] else 0
        risky_avg_bet = np.mean([r['avg_bet'] for r in results['risky']]) if results['risky'] else 0
        baseline_avg_bet = np.mean([r['avg_bet'] for r in results['baseline']]) if results['baseline'] else 0

        safe_bankruptcy_rate = np.mean([r['is_bankrupt'] for r in results['safe']]) if results['safe'] else 0
        risky_bankruptcy_rate = np.mean([r['is_bankrupt'] for r in results['risky']]) if results['risky'] else 0
        baseline_bankruptcy_rate = np.mean([r['is_bankrupt'] for r in results['baseline']]) if results['baseline'] else 0

        return {
            'feature_info': feature_info,
            'safe_avg_bet': safe_avg_bet,
            'risky_avg_bet': risky_avg_bet,
            'baseline_avg_bet': baseline_avg_bet,
            'safe_bankruptcy_rate': safe_bankruptcy_rate,
            'risky_bankruptcy_rate': risky_bankruptcy_rate,
            'baseline_bankruptcy_rate': baseline_bankruptcy_rate,
            'bet_effect_safe': safe_avg_bet - baseline_avg_bet,
            'bet_effect_risky': risky_avg_bet - baseline_avg_bet,
            'bankruptcy_effect_safe': safe_bankruptcy_rate - baseline_bankruptcy_rate,
            'bankruptcy_effect_risky': risky_bankruptcy_rate - baseline_bankruptcy_rate,
            'detailed_results': results
        }

    def run_experiment(self):
        """Run the complete multi-round patching experiment"""
        print("🚀 Starting Multi-round Patching Experiment")
        print("="*80)

        # Load model
        self.load_models()

        # Select subset of high-effect features for intensive testing
        high_effect_features = [f for f in self.causal_features if abs(f.get('cohen_d', 0)) > 1.0]
        if len(high_effect_features) > 50:
            high_effect_features = high_effect_features[:50]  # Limit for computation

        print(f"Testing {len(high_effect_features)} high-effect features")

        all_results = []
        significant_features = []

        # Test each feature
        for i, feature_info in enumerate(tqdm(high_effect_features, desc="Testing features")):
            layer = feature_info['layer']
            feature_idx = feature_info['feature_idx']

            print(f"\n🎮 Testing L{layer}-{feature_idx} multi-round ({i+1}/{len(high_effect_features)})")

            try:
                result = self.test_single_feature_multiround(feature_info)
                all_results.append(result)

                # Check for significant effects
                if (abs(result['bet_effect_safe']) > 5 or abs(result['bet_effect_risky']) > 5 or
                    abs(result['bankruptcy_effect_safe']) > 0.2 or abs(result['bankruptcy_effect_risky']) > 0.2):
                    significant_features.append(result)
                    print(f"   ✅ SIGNIFICANT: bet_effect={result['bet_effect_safe']:.2f}/{result['bet_effect_risky']:.2f}, "
                          f"bankruptcy_effect={result['bankruptcy_effect_safe']:.3f}/{result['bankruptcy_effect_risky']:.3f}")
                else:
                    print(f"   ❌ Minimal effect")

                # Save intermediate results every 10 features
                if (i + 1) % 10 == 0:
                    self.save_intermediate_results(all_results, significant_features, i + 1)

            except Exception as e:
                print(f"   ❌ Error testing L{layer}-{feature_idx}: {e}")
                continue

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

        # Save final results
        self.save_final_results(all_results, significant_features)

        return all_results, significant_features

    def save_intermediate_results(self, all_results, significant_features, progress):
        """Save intermediate results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        intermediate_file = self.results_dir / f'multiround_intermediate_{progress}_{timestamp}.json'

        data = {
            'timestamp': timestamp,
            'progress': f'{progress}/{len([f for f in self.causal_features if abs(f.get("cohen_d", 0)) > 1.0][:50])}',
            'significant_features_found': len(significant_features),
            'all_results': all_results
        }

        with open(intermediate_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"   💾 Saved: {len(significant_features)} significant of {len(all_results)} tested")

    def save_final_results(self, all_results, significant_features):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_file = self.results_dir / f'multiround_patching_complete_{timestamp}.json'

        data = {
            'timestamp': timestamp,
            'experiment_config': {
                'total_features_tested': len(all_results),
                'significant_features_found': len(significant_features),
                'n_trials_per_condition': self.n_trials,
                'max_rounds_per_game': self.max_rounds
            },
            'significant_features': significant_features,
            'all_results': all_results
        }

        with open(final_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Multi-round Patching Complete!")
        print(f"Total features tested: {len(all_results)}")
        print(f"Significant features found: {len(significant_features)}")
        print(f"📊 Results saved: {final_file}")

if __name__ == '__main__':
    experiment = MultiroundPatchingExperiment()
    experiment.run_experiment()