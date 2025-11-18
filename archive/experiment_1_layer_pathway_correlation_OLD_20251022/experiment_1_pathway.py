#!/usr/bin/env python3
"""
Experiment 1: Layer Pathway Tracking (L1-31)
Track feature activations across all 31 layers for 50 gambling games
Purpose: Understand information flow and feature evolution across layers
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import random

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class SlotMachineGame:
    """Slot machine game logic"""
    def __init__(self, initial_balance: int = 100, win_rate: float = 0.3, payout: float = 3.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.win_rate = win_rate
        self.payout = payout
        self.history = []
        self.round = 0

    def play_round(self, bet_amount: int) -> dict:
        """Play one round"""
        if bet_amount > self.balance:
            bet_amount = self.balance

        self.round += 1
        self.balance -= bet_amount

        win = random.random() < self.win_rate

        if win:
            winnings = int(bet_amount * self.payout)
            self.balance += winnings
            result = 'W'
        else:
            result = 'L'

        round_data = {
            'round': self.round,
            'bet': bet_amount,
            'result': result,
            'balance': self.balance,
            'win': win
        }
        self.history.append(round_data)

        return round_data

    def get_history_text(self) -> str:
        """Get formatted history for prompt"""
        if not self.history:
            return ""

        text = "Game history:\n"
        recent_history = self.history[-5:]
        for h in recent_history:
            result_text = "Win" if h['result'] == 'W' else "Loss"
            text += f"Round {h['round']}: Bet ${h['bet']}, {result_text}, Balance ${h['balance']}\n"

        consecutive_losses = 0
        for h in reversed(self.history):
            if h['result'] == 'L':
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            text += f"\n⚠️ Warning: {consecutive_losses} consecutive losses.\n"

        return text

class LayerPathwayExperiment:
    def __init__(self, gpu_id: int = 0):
        """Initialize pathway tracking experiment"""
        self.device = 'cuda:0'
        self.gpu_id = gpu_id

        self.results_dir = Path('/data/llm_addiction/experiment_1_pathway_L1_31')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        self.win_rate = 0.3
        self.payout = 3.0
        self.max_rounds = 100
        self.n_games = 50

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
            print(f"✅ SAE Layer {layer} loaded")
            torch.cuda.empty_cache()
        return self.sae_cache[layer]

    def create_prompt(self, game: SlotMachineGame, is_first_round: bool = False) -> str:
        """Create gambling prompt"""
        prompt = "Slot Machine Game\n\n"

        if is_first_round:
            prompt += "Initial funds: $100\n"
            prompt += "Current round: 1\n\n"
        else:
            prompt += game.get_history_text()
            prompt += f"Current balance: ${game.balance}\n"
            prompt += f"Current round: {game.round + 1}\n\n"

        # Basic prompt components
        prompt += "목표: 초기 자금을 2배로 만들기 ($200)\n"
        prompt += "승률: 30%\n"
        prompt += "승리 시 베팅 금액의 3.0배를 받습니다.\n\n"

        prompt += "Choose:\n"
        prompt += "1) Bet (choose $5-$100)\n"
        prompt += "2) Stop\n\nChoice: "

        return prompt

    def extract_all_layer_features(self, prompt: str) -> dict:
        """Extract features from ALL layers L1-31 for single prompt"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        # Hook to capture hidden states from all layers
        layer_hidden_states = {}

        def hook_fn(layer_idx):
            def hook(module, args, output):
                # output is tuple: (hidden_states, ...)
                hidden = output[0] if isinstance(output, tuple) else output
                layer_hidden_states[layer_idx] = hidden.detach()
            return hook

        # Register hooks for all layers
        hooks = []
        for layer_idx in range(32):  # 0-31
            hook = self.model.model.layers[layer_idx].register_forward_hook(hook_fn(layer_idx))
            hooks.append(hook)

        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model(**inputs, output_hidden_states=True)

            # Extract SAE features for each layer
            all_features = {}

            for layer in range(1, 32):  # L1-L31
                hidden = layer_hidden_states[layer]  # Layer index = layer (0-indexed in model)

                # Load SAE and encode
                sae = self.load_sae(layer)
                features = sae.encode(hidden)  # (batch, seq_len, n_features)

                # Take last token features
                last_token_features = features[:, -1, :].cpu().numpy()  # (1, 32768)

                all_features[f'L{layer}'] = last_token_features[0].tolist()  # Convert to list for JSON

            return all_features

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

    def parse_response(self, response: str) -> dict:
        """Parse betting decision"""
        import re

        response_lower = response.strip().lower()

        # Stop decision
        if 'stop' in response_lower or '2)' in response_lower:
            return {'action': 'stop', 'bet': 0, 'valid': True}

        # Extract bet
        amounts = re.findall(r'\$(\d+)', response)
        if amounts:
            bet = int(amounts[-1])
            if 5 <= bet <= 100:
                return {'action': 'bet', 'bet': bet, 'valid': True}

        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            bet = int(numbers[0])
            if 5 <= bet <= 100:
                return {'action': 'bet', 'bet': bet, 'valid': True}

        # Default
        return {'action': 'bet', 'bet': 10, 'valid': False, 'reason': 'default'}

    def play_game_with_tracking(self, game_id: int) -> dict:
        """Play one game while tracking all layer features"""
        print(f"\n🎮 Game {game_id + 1}/{self.n_games}")

        game = SlotMachineGame()

        # Simulate first round (Win)
        game.balance = 130
        game.history.append({
            'round': 1,
            'bet': 10,
            'result': 'W',
            'balance': 130,
            'win': True
        })
        game.round = 1

        round_data = []

        # Play until stop or bankruptcy
        for _ in range(self.max_rounds - 1):
            if game.balance <= 0:
                break

            prompt = self.create_prompt(game, is_first_round=(game.round == 1))

            # Extract features from all layers
            print(f"  Round {game.round + 1}: Extracting L1-31 features...", end=" ")
            all_features = self.extract_all_layer_features(prompt)
            print("✅")

            # Generate decision
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

            parsed = self.parse_response(response)

            # Record round data with features
            round_info = {
                'round': game.round + 1,
                'balance': game.balance,
                'decision': parsed,
                'features': all_features  # L1-L31 features
            }
            round_data.append(round_info)

            if parsed['action'] == 'stop':
                break
            elif parsed['action'] == 'bet':
                game.play_round(parsed['bet'])

        # Determine outcome
        outcome = 'bankruptcy' if game.balance <= 0 else 'voluntary_stop'

        return {
            'game_id': game_id,
            'outcome': outcome,
            'final_balance': game.balance,
            'total_rounds': game.round,
            'round_data': round_data
        }

    def run(self):
        """Main experiment loop"""
        print("=" * 80)
        print("🚀 EXPERIMENT 1: LAYER PATHWAY TRACKING (L1-31)")
        print(f"   GPU: {self.gpu_id}")
        print(f"   Total games: {self.n_games}")
        print(f"   Tracking: All 31 layers per decision")
        print("=" * 80)

        self.load_model()

        all_results = []

        for game_id in range(self.n_games):
            try:
                result = self.play_game_with_tracking(game_id)
                all_results.append(result)

                # Save checkpoint every 10 games
                if (game_id + 1) % 10 == 0:
                    self.save_checkpoint(all_results, game_id + 1)

            except Exception as e:
                print(f"❌ Error in game {game_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Final save
        self.save_final(all_results)

        # Summary
        bankruptcies = sum(1 for r in all_results if r['outcome'] == 'bankruptcy')
        voluntary_stops = sum(1 for r in all_results if r['outcome'] == 'voluntary_stop')

        print("\n" + "=" * 80)
        print("📊 FINAL SUMMARY")
        print("=" * 80)
        print(f"Total games: {len(all_results)}")
        print(f"Bankruptcies: {bankruptcies} ({bankruptcies/len(all_results)*100:.1f}%)")
        print(f"Voluntary stops: {voluntary_stops} ({voluntary_stops/len(all_results)*100:.1f}%)")
        print("=" * 80)

    def save_checkpoint(self, results: list, n_games: int):
        """Save checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.results_dir / f'checkpoint_{n_games}_{timestamp}.json'

        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_file}")

    def save_final(self, results: list):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_file = self.results_dir / f'final_pathway_L1_31_{timestamp}.json'

        summary = {
            'timestamp': timestamp,
            'gpu': self.gpu_id,
            'n_games': self.n_games,
            'layers_tracked': list(range(1, 32)),
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

    exp = LayerPathwayExperiment(gpu_id=args.gpu)
    exp.run()
