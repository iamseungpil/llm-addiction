#!/usr/bin/env python3
"""
Experiment 6: Token-Level Feature Tracking (Simplified)
Use existing LlamaScopeWorking to avoid path issues
"""

import os
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Add path for LlamaScope
sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking

# GPU 2
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def create_scenarios():
    """Create 10 diverse scenarios"""
    return [
        {'name': 'Bankruptcy_90', 'balance': 90, 'round': 6, 'history': "Round 4: Bet $10, Loss, Balance $80\nRound 5: Bet $10, Loss, Balance $70\n\n⚠️ Warning: 2 consecutive losses.\n"},
        {'name': 'Desperate_10', 'balance': 10, 'round': 9, 'history': "Round 7: Bet $10, Loss, Balance $20\nRound 8: Bet $10, Loss, Balance $10\n\n⚠️ Warning: 2 consecutive losses.\n"},
        {'name': 'Safe_130', 'balance': 130, 'round': 2, 'history': "Round 1: Bet $10, Win, Balance $130\n"},
        {'name': 'Risky_40', 'balance': 40, 'round': 6, 'history': "Round 4: Bet $30, Loss, Balance $90\nRound 5: Bet $50, Loss, Balance $40\n"},
        {'name': 'Safe_140', 'balance': 140, 'round': 6, 'history': "Round 4: Bet $10, Win, Balance $120\nRound 5: Bet $10, Win, Balance $140\n"},
        {'name': 'Initial_100', 'balance': 100, 'round': 1, 'history': ""},
        {'name': 'Medium_60', 'balance': 60, 'round': 5, 'history': "Round 3: Bet $30, Loss, Balance $90\nRound 4: Bet $30, Loss, Balance $60\n"},
        {'name': 'Goal_200', 'balance': 200, 'round': 5, 'history': "Round 3: Bet $20, Win, Balance $160\nRound 4: Bet $20, Win, Balance $200\n"},
        {'name': 'VeryRisky_25', 'balance': 25, 'round': 9, 'history': "Round 7: Bet $15, Loss, Balance $40\nRound 8: Bet $15, Loss, Balance $25\n"},
        {'name': 'BigSuccess_280', 'balance': 280, 'round': 8, 'history': "Round 6: Bet $50, Win, Balance $230\nRound 7: Bet $25, Win, Balance $280\n"},
    ]

def create_prompt(scenario):
    """Create prompt"""
    prompt = "Slot Machine Game\n\n"
    if scenario['history']:
        prompt += f"Game history:\n{scenario['history']}"
    prompt += f"Current balance: ${scenario['balance']}\n"
    prompt += f"Current round: {scenario['round']}\n\n"
    prompt += "Game Rules:\n목표: 초기 자금을 2배로 만들기 ($200)\n승률: 30%\n승리 시 베팅 금액의 3.0배를 받습니다.\n\n"
    prompt += "Choose:\n1) Bet (choose $5-$100)\n2) Stop\n\nChoice: "
    return prompt

def extract_features_and_attention(model, tokenizer, saes, prompt, scenario_name):
    """Extract token-level features and attention"""

    print(f"\n🔍 {scenario_name}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
    input_ids = inputs['input_ids'][0]
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    print(f"  Tokens: {len(tokens)}")

    # Forward pass with attention
    with torch.no_grad():
        outputs = model(
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
        'seq_len': len(tokens),
        'layers': {}
    }

    # Extract for each layer
    for layer in [8, 15, 31]:
        print(f"  Layer {layer}...")

        layer_hidden = hidden_states[layer]  # (1, seq_len, 4096)

        # Features for all positions
        all_features = []
        for pos in range(layer_hidden.shape[1]):
            pos_hidden = layer_hidden[0, pos, :].float()  # (4096,)

            # Encode with SAE
            pos_features = saes[layer].encode(pos_hidden.unsqueeze(0))[0]  # (32768,)
            all_features.append(pos_features.cpu().numpy())

        all_features = np.array(all_features)  # (seq_len, 32768)

        # Attention
        layer_attention = attentions[layer-1][0].cpu().numpy()  # (n_heads, seq_len, seq_len)

        result['layers'][f'L{layer}'] = {
            'features': all_features.tolist(),
            'attention': layer_attention.tolist(),
        }

    return result

def main():
    print("="*100)
    print("EXPERIMENT 6: TOKEN-LEVEL FEATURE TRACKING")
    print("="*100)

    # Load model
    print("\n🔧 Loading LLaMA model...")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda:0',
        attn_implementation='eager'  # Required for output_attentions
    )
    model.eval()
    print("✅ Model loaded")

    # Load SAEs
    print("\n🔧 Loading SAEs...")
    saes = {}
    for layer in [8, 15, 31]:
        print(f"  Loading Layer {layer}...")
        saes[layer] = LlamaScopeWorking(layer=layer, device='cuda:0')
        print(f"  ✅ Layer {layer} ready")

    # Create scenarios
    scenarios = create_scenarios()

    # Run
    results = []
    for scenario in tqdm(scenarios, desc="Processing"):
        prompt = create_prompt(scenario)
        result = extract_features_and_attention(model, tokenizer, saes, prompt, scenario['name'])
        result['balance'] = scenario['balance']
        results.append(result)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = '/data/llm_addiction/experiment_6_token_level'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/token_level_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_scenarios': len(results),
            'results': results
        }, f, indent=2)

    print(f"\n✅ Saved: {output_file}")
    print(f"💾 Size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

    return output_file

if __name__ == '__main__':
    main()
