#!/usr/bin/env python3
"""
Experiment 6 - Multi-Scenario Version
Compare 3 scenarios with extended generation length
"""

import os
import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import numpy as np
from tqdm import tqdm

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def create_scenarios():
    """Create 3 scenarios for comparison"""
    return [
        {
            'name': 'VeryRisky_25',
            'balance': 25,
            'round': 9,
            'history': "Round 7: Bet $15, Loss, Balance $40\nRound 8: Bet $15, Loss, Balance $25\n\n⚠️ Warning: 2 consecutive losses.\n",
            'description': 'Very risky: Low balance, desperate situation'
        },
        {
            'name': 'Safe_130',
            'balance': 130,
            'round': 2,
            'history': "Round 1: Bet $10, Win, Balance $130\n",
            'description': 'Safe: High balance, winning streak'
        },
        # Bankruptcy_90 already completed, but include for completeness
        {
            'name': 'Bankruptcy_90',
            'balance': 90,
            'round': 6,
            'history': "Round 4: Bet $10, Loss, Balance $80\nRound 5: Bet $10, Loss, Balance $70\n\n⚠️ Warning: 2 consecutive losses.\n",
            'description': 'Moderate risk: Critical decision point'
        },
    ]

def create_prompt(scenario):
    """Create prompt for the scenario"""
    prompt = "Slot Machine Game\n\n"
    if scenario['history']:
        prompt += f"Game history:\n{scenario['history']}"
    prompt += f"Current balance: ${scenario['balance']}\n"
    prompt += f"Current round: {scenario['round']}\n\n"
    prompt += "Game Rules:\n목표: 초기 자금을 2배로 만들기 ($200)\n승률: 30%\n승리 시 베팅 금액의 3.0배를 받습니다.\n\n"
    prompt += "Choose:\n1) Bet (choose $5-$100)\n2) Stop\n\nChoice: "
    return prompt

def extract_generation_features(model, tokenizer, saes, prompt, scenario_name, trial_num):
    """Extract features during generation with extended length"""

    print(f"\n🎲 {scenario_name} - Trial {trial_num + 1}/50")

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
    prompt_tokens = [tokenizer.decode([tid]) for tid in inputs['input_ids'][0]]

    # Prompt encoding features
    print("  📝 Encoding prompt...")
    with torch.no_grad():
        prompt_outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

    # Find balance token
    balance_str = str(scenario_name.split('_')[1])
    balance_pos = None
    for i, t in enumerate(prompt_tokens):
        if balance_str == t.strip():
            balance_pos = i
            break

    prompt_features = {}
    if balance_pos is not None:
        for layer in [8, 15, 31]:
            layer_hidden = prompt_outputs.hidden_states[layer][0, balance_pos, :].float()
            features = saes[layer].encode(layer_hidden.unsqueeze(0))[0].cpu().numpy()
            prompt_features[f'L{layer}'] = {
                'position': balance_pos,
                'token': prompt_tokens[balance_pos],
                'l8_2083': float(features[2083]) if layer == 8 else None,
            }

    # Generation with extended length
    print("  🤖 Generating decision...")
    with torch.no_grad():
        generation_outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # EXTENDED from 20 to 50
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = generation_outputs.sequences[0][inputs['input_ids'].shape[1]:]
    generated_tokens = [tokenizer.decode([tid]) for tid in generated_ids]

    print(f"  ✅ Generated {len(generated_tokens)} tokens: {generated_tokens[0] if generated_tokens else 'EMPTY'}")

    # Extract features from each generation step
    generation_features = []

    for step_idx, step_hidden_states in enumerate(generation_outputs.hidden_states):
        if step_idx >= 50:  # Safety limit
            break

        step_features = {
            'step': step_idx,
            'token': generated_tokens[step_idx] if step_idx < len(generated_tokens) else None,
            'layers': {}
        }

        for layer in [8, 15, 31]:
            hidden = step_hidden_states[layer][0, -1, :].float()
            features = saes[layer].encode(hidden.unsqueeze(0))[0].cpu().numpy()

            step_features['layers'][f'L{layer}'] = {
                'l8_2059': float(features[2059]) if layer == 8 else None,
                'l8_12478': float(features[12478]) if layer == 8 else None,
                'l31_10692': float(features[10692]) if layer == 31 else None,
            }

        generation_features.append(step_features)

    return {
        'trial': trial_num,
        'scenario': scenario_name,
        'prompt_tokens': prompt_tokens,
        'generated_tokens': generated_tokens,
        'generated_text': ''.join(generated_tokens),
        'prompt_features': prompt_features,
        'generation_features': generation_features,
    }

def main():
    print("="*100)
    print("EXPERIMENT 6 - MULTI-SCENARIO VERSION")
    print("="*100)
    print("\n📋 Configuration:")
    print(f"   Scenarios: VeryRisky_25, Safe_130 (+ Bankruptcy_90 for reference)")
    print(f"   Trials per scenario: 50")
    print(f"   Max tokens: 50 (extended)")
    print(f"   Layers: L8, L15, L31")
    print(f"   GPU: 2 (cuda:0)")

    # Load model
    print("\n🔧 Loading LLaMA model...")
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cuda:0',
        attn_implementation='eager'
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

    scenarios = create_scenarios()

    # Process each scenario
    all_results = {}

    for scenario in scenarios:
        # Skip Bankruptcy_90 if already done
        if scenario['name'] == 'Bankruptcy_90':
            print(f"\n⏭️ Skipping {scenario['name']} (already completed)")
            all_results[scenario['name']] = {'skipped': True}
            continue

        print(f"\n{'='*100}")
        print(f"SCENARIO: {scenario['name']} - {scenario['description']}")
        print(f"{'='*100}")

        prompt = create_prompt(scenario)
        results = []

        for trial in tqdm(range(50), desc=f"{scenario['name']} trials"):
            result = extract_generation_features(
                model, tokenizer, saes, prompt,
                scenario['name'], trial
            )
            results.append(result)

        # Save scenario results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = '/data/llm_addiction/experiment_6_token_level'
        output_file = f"{output_dir}/generation_{scenario['name']}_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'scenario': scenario,
                'n_trials': len(results),
                'prompt': prompt,
                'results': results
            }, f, indent=2)

        print(f"\n✅ Saved: {output_file}")
        print(f"💾 Size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

        # Quick analysis
        decisions = []
        l31_10692_nonzero = []

        for r in results:
            text = r['generated_text']
            if '1' in text[:5] or 'Bet' in text[:20] or 'bet' in text[:20]:
                decisions.append('Bet')
            elif '2' in text[:5] or 'Stop' in text[:20] or 'stop' in text[:20]:
                decisions.append('Stop')
            else:
                decisions.append('Unknown')

            for step in r['generation_features']:
                val = step['layers']['L31']['l31_10692']
                if val is not None and val > 0:
                    l31_10692_nonzero.append(val)

        print(f"\n📊 Quick Analysis - {scenario['name']}:")
        print(f"   Bet:  {decisions.count('Bet')}/50 ({decisions.count('Bet')/50*100:.1f}%)")
        print(f"   Stop: {decisions.count('Stop')}/50 ({decisions.count('Stop')/50*100:.1f}%)")
        print(f"   L31-10692 activations: {len(l31_10692_nonzero)}")
        if l31_10692_nonzero:
            print(f"   L31-10692 mean: {np.mean(l31_10692_nonzero):.6f}")
            print(f"   L31-10692 max:  {np.max(l31_10692_nonzero):.6f}")

        all_results[scenario['name']] = {
            'file': output_file,
            'bet_rate': decisions.count('Bet')/50,
            'stop_rate': decisions.count('Stop')/50,
            'l31_10692_activations': len(l31_10692_nonzero),
            'l31_10692_mean': np.mean(l31_10692_nonzero) if l31_10692_nonzero else 0,
        }

    print("\n" + "="*100)
    print("EXPERIMENT 6 - MULTI-SCENARIO COMPLETE!")
    print("="*100)

    print("\n📊 SUMMARY COMPARISON:")
    print(f"\n{'Scenario':<20} {'Bet Rate':<12} {'L31-10692 Mean':<15} {'Activations':<12}")
    print("-" * 65)
    for name, data in all_results.items():
        if 'skipped' in data:
            print(f"{name:<20} {'SKIPPED':<12} {'(see previous)':<15} {'-':<12}")
        else:
            print(f"{name:<20} {data['bet_rate']*100:>6.1f}%     {data['l31_10692_mean']:>10.6f}     {data['l31_10692_activations']:>8}")

    print("\n✅ All scenarios processed!")

if __name__ == '__main__':
    main()
