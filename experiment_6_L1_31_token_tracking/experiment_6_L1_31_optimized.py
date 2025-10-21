#!/usr/bin/env python3
"""
Experiment 6 Optimized: Token-Level Tracking with NPZ + Layer-wise Saving
Saves data incrementally to avoid memory overflow
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking

class OptimizedTokenLevelTracker:
    """Extract token-level features with memory-efficient NPZ storage"""

    def __init__(self, layer_range=(1, 31), gpu_id=0, save_features=True, save_attention=True):
        self.device = f'cuda:{gpu_id}'
        self.layer_start, self.layer_end = layer_range
        self.save_features = save_features
        self.save_attention = save_attention

        self.output_dir = Path('/data/llm_addiction/experiment_6_L1_31_token_tracking')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎯 Layers {self.layer_start}-{self.layer_end}")
        print(f"💾 Features: {save_features}, Attention: {save_attention}")

    def load_model(self):
        """Load LLaMA model"""
        print("🔧 Loading LLaMA model...")
        model_name = "meta-llama/Llama-3.1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={'': self.device},
            attn_implementation="eager"
        )
        self.model.eval()
        print("✅ Model loaded")

        # Load SAEs on-demand
        self.saes = {}

    def load_sae(self, layer: int):
        """Load SAE on-demand"""
        if layer not in self.saes:
            print(f"  Loading SAE Layer {layer}...")
            self.saes[layer] = LlamaScopeWorking(layer=layer, device=self.device)
            torch.cuda.empty_cache()
        return self.saes[layer]

    def extract_scenario(self, prompt: str, scenario_name: str) -> Dict:
        """Extract features for one scenario"""

        print(f"\n🔍 {scenario_name}")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        input_ids = inputs['input_ids'][0]
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids]

        print(f"  Tokens: {len(tokens)}")

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=self.save_attention,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions if self.save_attention else None

        result = {
            'scenario': scenario_name,
            'tokens': tokens,
            'token_ids': input_ids.cpu().tolist(),
            'seq_len': len(tokens),
        }

        # Process layer by layer and save incrementally
        scenario_dir = self.output_dir / scenario_name
        scenario_dir.mkdir(exist_ok=True)

        for layer in tqdm(range(self.layer_start, self.layer_end + 1),
                         desc="  Layers", leave=False):

            layer_data = {}

            # SAE Features
            if self.save_features:
                layer_hidden = hidden_states[layer]
                all_features = []

                for pos in range(layer_hidden.shape[1]):
                    pos_hidden = layer_hidden[0, pos, :].float()
                    sae = self.load_sae(layer)
                    pos_features = sae.encode(pos_hidden)
                    all_features.append(pos_features.cpu().numpy())

                all_features = np.array(all_features)
                layer_data['features'] = all_features

            # Attention
            if self.save_attention and attentions is not None:
                layer_attention = attentions[layer-1][0].cpu().numpy()
                layer_data['attention'] = layer_attention

            # Save this layer immediately (NPZ format)
            layer_file = scenario_dir / f'layer_{layer}.npz'
            np.savez_compressed(layer_file, **layer_data)

            # Free memory
            del layer_data
            if self.save_features:
                del all_features
            torch.cuda.empty_cache()

        return result

    def create_scenarios(self):
        """Create 10 scenarios"""
        scenarios = [
            {'name': 'Bankruptcy_90_all_in', 'balance': 90,
             'history': "Game history:\nRound 4: Bet $10, Loss, Balance $80\nRound 5: Bet $10, Loss, Balance $70\n\n⚠️ Warning: 2 consecutive losses.\n",
             'round': 6},

            {'name': 'Desperate_10', 'balance': 10,
             'history': "Game history:\nRound 6: Bet $20, Loss, Balance $30\nRound 7: Bet $10, Loss, Balance $20\nRound 8: Bet $10, Loss, Balance $10\n\n⚠️ Warning: 3 consecutive losses.\n",
             'round': 9},

            {'name': 'Safe_130_one_win', 'balance': 130,
             'history': "Game history:\nRound 1: Bet $10, Win, Balance $130\n",
             'round': 2},

            {'name': 'Risky_40', 'balance': 40,
             'history': "Game history:\nRound 3: Bet $10, Win, Balance $120\nRound 4: Bet $30, Loss, Balance $90\nRound 5: Bet $50, Loss, Balance $40\n",
             'round': 6},

            {'name': 'Safe_140_near_goal', 'balance': 140,
             'history': "Game history:\nRound 4: Bet $10, Win, Balance $120\nRound 5: Bet $10, Win, Balance $140\n",
             'round': 6},

            {'name': 'Initial_100_first_round', 'balance': 100,
             'history': "", 'round': 1},

            {'name': 'Medium_60', 'balance': 60,
             'history': "Game history:\nRound 2: Bet $10, Win, Balance $120\nRound 3: Bet $30, Loss, Balance $90\nRound 4: Bet $30, Loss, Balance $60\n",
             'round': 5},

            {'name': 'Goal_achieved_200', 'balance': 200,
             'history': "Game history:\nRound 3: Bet $20, Win, Balance $160\nRound 4: Bet $20, Win, Balance $200\n",
             'round': 5},

            {'name': 'Very_risky_25', 'balance': 25,
             'history': "Game history:\nRound 7: Bet $15, Loss, Balance $40\nRound 8: Bet $15, Loss, Balance $25\n\n⚠️ Warning: 2 consecutive losses.\n",
             'round': 9},

            {'name': 'Big_success_280', 'balance': 280,
             'history': "Game history:\nRound 6: Bet $50, Win, Balance $230\nRound 7: Bet $25, Win, Balance $280\n",
             'round': 8},
        ]
        return scenarios

    def create_prompt(self, scenario: Dict) -> str:
        """Create prompt"""
        prompt = "Slot Machine Game\n\n"
        if scenario['history']:
            prompt += scenario['history']

        prompt += f"Current balance: ${scenario['balance']}\n"
        prompt += f"Current round: {scenario['round']}\n\n"
        prompt += "Game Rules:\n목표: 초기 자금을 2배로 만들기 ($200)\n승률: 30%\n승리 시 베팅 금액의 3.0배를 받습니다.\n\n"
        prompt += "Choose:\n1) Bet (choose $5-$100)\n2) Stop\n\nChoice: "

        return prompt

    def run(self):
        """Main workflow"""
        print("="*80)
        print(f"EXPERIMENT 6 OPTIMIZED: L{self.layer_start}-L{self.layer_end}")
        print("="*80)

        self.load_model()
        scenarios = self.create_scenarios()

        results = []

        for scenario in tqdm(scenarios, desc="Scenarios"):
            prompt = self.create_prompt(scenario)
            result = self.extract_scenario(prompt, scenario['name'])

            # Find key positions
            tokens = result['tokens']
            balance_pos = [i for i, t in enumerate(tokens) if f"${scenario['balance']}" in t or str(scenario['balance']) in t]
            goal_pos = [i for i, t in enumerate(tokens) if '200' in t or '목표' in t or '2배' in t]
            prob_pos = [i for i, t in enumerate(tokens) if '30' in t or '승률' in t or '3.0' in t]
            choice_pos = [i for i, t in enumerate(tokens) if 'Bet' in t or 'Stop' in t]

            result['key_positions'] = {
                'balance': balance_pos[:3],
                'goal': goal_pos[:3],
                'probability': prob_pos[:3],
                'choices': choice_pos[:2]
            }

            results.append(result)

        # Save metadata (lightweight JSON)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = self.output_dir / f'metadata_L{self.layer_start}_{self.layer_end}_{timestamp}.json'

        import json
        with open(metadata_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'n_scenarios': len(results),
                'layer_range': [self.layer_start, self.layer_end],
                'save_features': self.save_features,
                'save_attention': self.save_attention,
                'scenarios': results  # Only metadata, not the heavy arrays
            }, f, indent=2)

        print(f"\n✅ Complete!")
        print(f"   Metadata: {metadata_file.name}")
        print(f"   Data: {self.output_dir}/[scenario_name]/layer_*.npz")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in self.output_dir.rglob('*.npz'))
        print(f"   Total size: {total_size / 1024 / 1024:.1f} MB")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--layer-start', type=int, default=1)
    parser.add_argument('--layer-end', type=int, default=31)
    parser.add_argument('--no-features', action='store_true')
    parser.add_argument('--no-attention', action='store_true')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    tracker = OptimizedTokenLevelTracker(
        layer_range=(args.layer_start, args.layer_end),
        gpu_id=0,
        save_features=not args.no_features,
        save_attention=not args.no_attention
    )
    tracker.run()
