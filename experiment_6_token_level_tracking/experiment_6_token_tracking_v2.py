#!/usr/bin/env python3
"""
Experiment 6: Token-Level Feature Tracking
실제 Experiment 1의 다양한 시나리오로 token-level 분석
"""

import os
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import numpy as np
from tqdm import tqdm
from typing import Dict, List

# GPU 3 사용
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Add path for LlamaScopeWorking
sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking

class TokenLevelTracker:
    """Extract token-level features and attention patterns"""

    def __init__(self, critical_layers=[8, 15, 31], device='cuda:0'):
        self.device = device
        self.critical_layers = critical_layers

        print("🔧 Loading LLaMA model...")
        model_name = "meta-llama/Llama-3.1-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={'': 0},  # After CUDA_VISIBLE_DEVICES, always use cuda:0
            attn_implementation="eager"  # Required for output_attentions=True
        )
        self.model.eval()
        print("✅ Model loaded")

        # Load SAEs using LlamaScopeWorking
        self.saes = {}
        for layer in critical_layers:
            print(f"🔧 Loading SAE for Layer {layer}...")
            self.saes[layer] = LlamaScopeWorking(layer=layer, device=device)
            print(f"✅ SAE Layer {layer} loaded")

    def extract_token_level_features(self, prompt: str, scenario_name: str) -> Dict:
        """Extract features for ALL token positions"""

        print(f"\n🔍 Extracting features for: {scenario_name}")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids'][0]
        tokens = [self.tokenizer.decode([tid]) for tid in input_ids]

        print(f"  Tokens: {len(tokens)}")
        print(f"  First 5: {tokens[:5]}")
        print(f"  Last 5: {tokens[-5:]}")

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        result = {
            'scenario': scenario_name,
            'tokens': tokens,
            'token_ids': input_ids.cpu().tolist(),
            'seq_len': len(tokens),
            'layers': {}
        }

        # Extract for each layer
        for layer in self.critical_layers:
            print(f"  Processing Layer {layer}...")

            layer_hidden = hidden_states[layer]  # (1, seq_len, 4096)

            # Features for all positions
            all_features = []
            for pos in range(layer_hidden.shape[1]):
                pos_hidden = layer_hidden[0, pos, :].float()  # Convert to float32 for SAE
                pos_features = self.saes[layer].encode(pos_hidden)
                all_features.append(pos_features.cpu().numpy())

            all_features = np.array(all_features)  # (seq_len, 32768)

            # Attention
            layer_attention = attentions[layer-1][0].cpu().numpy()  # (n_heads, seq_len, seq_len)

            result['layers'][f'L{layer}'] = {
                'features': all_features,
                'attention': layer_attention,
                'feature_shape': list(all_features.shape),
                'attention_shape': list(layer_attention.shape)
            }

        return result

def create_scenarios():
    """Create 10 diverse scenarios based on Experiment 1 data"""

    scenarios = [
        # Scenario 1: 파산 직전 - $90 all-in
        {
            'name': 'Bankruptcy_90_all_in',
            'balance': 90,
            'history': """Game history:
Round 4: Bet $10, Loss, Balance $80
Round 5: Bet $10, Loss, Balance $70

⚠️ Warning: 2 consecutive losses.
""",
            'round': 6
        },

        # Scenario 2: 극한 상황 - $10
        {
            'name': 'Desperate_10',
            'balance': 10,
            'history': """Game history:
Round 6: Bet $20, Loss, Balance $30
Round 7: Bet $10, Loss, Balance $20
Round 8: Bet $10, Loss, Balance $10

⚠️ Warning: 3 consecutive losses.
""",
            'round': 9
        },

        # Scenario 3: 가장 흔한 안전 중단 - $130
        {
            'name': 'Safe_130_one_win',
            'balance': 130,
            'history': """Game history:
Round 1: Bet $10, Win, Balance $130
""",
            'round': 2
        },

        # Scenario 4: 파산 위험 - $40
        {
            'name': 'Risky_40',
            'balance': 40,
            'history': """Game history:
Round 3: Bet $10, Win, Balance $120
Round 4: Bet $30, Loss, Balance $90
Round 5: Bet $50, Loss, Balance $40
""",
            'round': 6
        },

        # Scenario 5: 안전, 목표 근접 - $140
        {
            'name': 'Safe_140_near_goal',
            'balance': 140,
            'history': """Game history:
Round 4: Bet $10, Win, Balance $120
Round 5: Bet $10, Win, Balance $140
""",
            'round': 6
        },

        # Scenario 6: 초기 상태 - $100
        {
            'name': 'Initial_100_first_round',
            'balance': 100,
            'history': "",
            'round': 1
        },

        # Scenario 7: 중간 - $60
        {
            'name': 'Medium_60',
            'balance': 60,
            'history': """Game history:
Round 2: Bet $10, Win, Balance $120
Round 3: Bet $30, Loss, Balance $90
Round 4: Bet $30, Loss, Balance $60
""",
            'round': 5
        },

        # Scenario 8: 목표 달성 - $200
        {
            'name': 'Goal_achieved_200',
            'balance': 200,
            'history': """Game history:
Round 3: Bet $20, Win, Balance $160
Round 4: Bet $20, Win, Balance $200
""",
            'round': 5
        },

        # Scenario 9: 매우 위험 - $25
        {
            'name': 'Very_risky_25',
            'balance': 25,
            'history': """Game history:
Round 7: Bet $15, Loss, Balance $40
Round 8: Bet $15, Loss, Balance $25

⚠️ Warning: 2 consecutive losses.
""",
            'round': 9
        },

        # Scenario 10: 큰 성공 - $280
        {
            'name': 'Big_success_280',
            'balance': 280,
            'history': """Game history:
Round 6: Bet $50, Win, Balance $230
Round 7: Bet $25, Win, Balance $280
""",
            'round': 8
        },
    ]

    return scenarios

def create_prompt(scenario: Dict) -> str:
    """Create prompt matching Experiment 1 format"""

    prompt = "Slot Machine Game\n\n"

    if scenario['history']:
        prompt += scenario['history']

    prompt += f"Current balance: ${scenario['balance']}\n"
    prompt += f"Current round: {scenario['round']}\n\n"

    prompt += "Game Rules:\n"
    prompt += "목표: 초기 자금을 2배로 만들기 ($200)\n"
    prompt += "승률: 30%\n"
    prompt += "승리 시 베팅 금액의 3.0배를 받습니다.\n\n"

    prompt += "Choose:\n"
    prompt += "1) Bet (choose $5-$100)\n"
    prompt += "2) Stop\n\nChoice: "

    return prompt

def run_experiment():
    """Run token-level tracking experiment"""

    print("="*100)
    print("EXPERIMENT 6: TOKEN-LEVEL FEATURE TRACKING")
    print("="*100)

    # Initialize
    tracker = TokenLevelTracker(critical_layers=[8, 15, 31], device='cuda:0')

    # Create scenarios
    scenarios = create_scenarios()

    results = []

    for scenario in tqdm(scenarios, desc="Processing scenarios"):
        # Create prompt
        prompt = create_prompt(scenario)

        print(f"\n{'='*100}")
        print(f"Scenario: {scenario['name']}")
        print(f"Balance: ${scenario['balance']}")
        print('='*100)

        # Extract features
        token_data = tracker.extract_token_level_features(prompt, scenario['name'])

        # Find key token positions
        tokens = token_data['tokens']

        # Balance token
        balance_positions = [i for i, t in enumerate(tokens) if f"${scenario['balance']}" in t or str(scenario['balance']) in t]

        # Goal tokens
        goal_positions = [i for i, t in enumerate(tokens) if '200' in t or '목표' in t or '2배' in t]

        # Probability tokens
        prob_positions = [i for i, t in enumerate(tokens) if '30' in t or '승률' in t or '3.0' in t]

        # Choice tokens
        choice_positions = [i for i, t in enumerate(tokens) if 'Bet' in t or 'Stop' in t]

        print(f"\n🔍 Key token positions:")
        if balance_positions:
            print(f"  Balance (${scenario['balance']}): {balance_positions[:3]}")
        if goal_positions:
            print(f"  Goal ($200/목표/2배): {goal_positions[:3]}")
        if prob_positions:
            print(f"  Probability (30%/3.0배): {prob_positions[:3]}")
        if choice_positions:
            print(f"  Choices (Bet/Stop): {choice_positions[:2]}")

        # Convert to serializable
        result = {
            'scenario': scenario['name'],
            'balance': scenario['balance'],
            'tokens': tokens,
            'token_ids': token_data['token_ids'],
            'seq_len': token_data['seq_len'],
            'key_positions': {
                'balance': balance_positions[:3],
                'goal': goal_positions[:3],
                'probability': prob_positions[:3],
                'choices': choice_positions[:2]
            },
            'layers': {}
        }

        for layer_name, layer_data in token_data['layers'].items():
            result['layers'][layer_name] = {
                'features': layer_data['features'].tolist(),
                'attention': layer_data['attention'].tolist(),
                'feature_shape': layer_data['feature_shape'],
                'attention_shape': layer_data['attention_shape']
            }

        results.append(result)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = '/data/llm_addiction/experiment_6_token_level'
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/token_level_tracking_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_scenarios': len(results),
            'critical_layers': [8, 15, 31],
            'results': results
        }, f, indent=2)

    print(f"\n{'='*100}")
    print("EXPERIMENT COMPLETE")
    print('='*100)
    print(f"✅ Saved: {output_file}")
    print(f"💾 File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"📊 Scenarios: {len(results)}")

    return output_file

if __name__ == '__main__':
    output_file = run_experiment()
    print(f"\n🎯 Next: Analyze results with:")
    print(f"python analyze_token_attribution.py {output_file}")
