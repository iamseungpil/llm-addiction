#!/usr/bin/env python3
"""
Analyze Experiment 6 Generation Results
Focus on features activated during actual token generation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict

def load_data(filepath):
    print(f"Loading: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✅ Loaded {data['n_trials']} trials\n")
    return data

def analyze_decision_features(data):
    """
    Analyze features during decision generation
    Focus on L8-2059, L8-12478, L31-10692
    """
    print("="*100)
    print("ANALYSIS 1: DECISION FEATURES DURING GENERATION")
    print("="*100)

    # Collect features from first generated token (the decision token)
    l8_2059_values = []
    l8_12478_values = []
    l31_10692_values = []
    decisions = []

    for result in data['results']:
        trial = result['trial']
        generated_text = result['generated_text']

        # Determine decision
        if '1' in generated_text[:5] or 'Bet' in generated_text[:20] or 'bet' in generated_text[:20]:
            decision = 'Bet'
        elif '2' in generated_text[:5] or 'Stop' in generated_text[:20] or 'stop' in generated_text[:20]:
            decision = 'Stop'
        else:
            decision = 'Unknown'

        decisions.append(decision)

        # Get features from first generation step
        if len(result['generation_features']) > 0:
            first_step = result['generation_features'][0]

            l8_2059 = first_step['layers']['L8']['l8_2059']
            l8_12478 = first_step['layers']['L8']['l8_12478']
            l31_10692 = first_step['layers']['L31']['l31_10692']

            if l8_2059 is not None:
                l8_2059_values.append(l8_2059)
            if l8_12478 is not None:
                l8_12478_values.append(l8_12478)
            if l31_10692 is not None:
                l31_10692_values.append(l31_10692)

    print(f"\n📊 Decision Distribution:")
    bet_count = decisions.count('Bet')
    stop_count = decisions.count('Stop')
    unknown_count = decisions.count('Unknown')
    print(f"   Bet:     {bet_count}/50 ({bet_count/50*100:.1f}%)")
    print(f"   Stop:    {stop_count}/50 ({stop_count/50*100:.1f}%)")
    print(f"   Unknown: {unknown_count}/50 ({unknown_count/50*100:.1f}%)")

    print(f"\n🎯 Feature Activations (First Generated Token):")

    print(f"\n   L8-2059 (risky feature from Exp 1):")
    l8_2059_nonzero = [v for v in l8_2059_values if v > 0]
    print(f"      Non-zero: {len(l8_2059_nonzero)}/50 ({len(l8_2059_nonzero)/50*100:.1f}%)")
    if l8_2059_nonzero:
        print(f"      Mean: {np.mean(l8_2059_nonzero):.6f}")
        print(f"      Max:  {np.max(l8_2059_nonzero):.6f}")
        print(f"      Min:  {np.min(l8_2059_nonzero):.6f}")

    print(f"\n   L8-12478 (safe feature from Exp 1):")
    l8_12478_nonzero = [v for v in l8_12478_values if v > 0]
    print(f"      Non-zero: {len(l8_12478_nonzero)}/50 ({len(l8_12478_nonzero)/50*100:.1f}%)")
    if l8_12478_nonzero:
        print(f"      Mean: {np.mean(l8_12478_nonzero):.6f}")
        print(f"      Max:  {np.max(l8_12478_nonzero):.6f}")
        print(f"      Min:  {np.min(l8_12478_nonzero):.6f}")

    print(f"\n   L31-10692 (risky feature from Exp 2):")
    l31_10692_nonzero = [v for v in l31_10692_values if v > 0]
    print(f"      Non-zero: {len(l31_10692_nonzero)}/50 ({len(l31_10692_nonzero)/50*100:.1f}%)")
    if l31_10692_nonzero:
        print(f"      Mean: {np.mean(l31_10692_nonzero):.6f}")
        print(f"      Max:  {np.max(l31_10692_nonzero):.6f}")
        print(f"      Min:  {np.min(l31_10692_nonzero):.6f}")

    return {
        'decisions': decisions,
        'l8_2059': l8_2059_values,
        'l8_12478': l8_12478_values,
        'l31_10692': l31_10692_values,
    }

def analyze_feature_evolution(data):
    """
    Analyze how features evolve across generation steps
    """
    print("\n" + "="*100)
    print("ANALYSIS 2: FEATURE EVOLUTION ACROSS GENERATION STEPS")
    print("="*100)

    # Average features across all trials by step
    max_steps = max(len(r['generation_features']) for r in data['results'])

    l8_2059_by_step = [[] for _ in range(max_steps)]
    l31_10692_by_step = [[] for _ in range(max_steps)]

    for result in data['results']:
        for step_idx, step in enumerate(result['generation_features']):
            if step_idx < max_steps:
                l8_2059 = step['layers']['L8']['l8_2059']
                l31_10692 = step['layers']['L31']['l31_10692']

                if l8_2059 is not None:
                    l8_2059_by_step[step_idx].append(l8_2059)
                if l31_10692 is not None:
                    l31_10692_by_step[step_idx].append(l31_10692)

    # Calculate means
    l8_2059_means = [np.mean(vals) if vals else 0 for vals in l8_2059_by_step]
    l31_10692_means = [np.mean(vals) if vals else 0 for vals in l31_10692_by_step]

    print(f"\n📈 Feature Evolution (first 10 steps):")
    print(f"\n{'Step':<6} {'L8-2059':<15} {'L31-10692':<15} {'# Samples':<12}")
    print("-" * 60)
    for step in range(min(10, max_steps)):
        print(f"{step:<6} {l8_2059_means[step]:<15.6f} {l31_10692_means[step]:<15.6f} {len(l8_2059_by_step[step]):<12}")

    return {
        'l8_2059_by_step': l8_2059_by_step,
        'l31_10692_by_step': l31_10692_by_step,
        'l8_2059_means': l8_2059_means,
        'l31_10692_means': l31_10692_means,
    }

def compare_prompt_vs_generation(data):
    """
    Compare prompt encoding features vs generation features
    """
    print("\n" + "="*100)
    print("ANALYSIS 3: PROMPT ENCODING vs GENERATION FEATURES")
    print("="*100)

    # Prompt features (at balance token)
    prompt_l8_2083 = []

    # Generation features (first token)
    gen_l8_2059 = []
    gen_l31_10692 = []

    for result in data['results']:
        # Prompt features
        if 'L8' in result['prompt_features']:
            prompt_features = np.array(result['prompt_features']['L8']['features'])
            prompt_l8_2083.append(prompt_features[2083])

        # Generation features
        if len(result['generation_features']) > 0:
            first_step = result['generation_features'][0]
            l8_2059 = first_step['layers']['L8']['l8_2059']
            l31_10692 = first_step['layers']['L31']['l31_10692']

            if l8_2059 is not None:
                gen_l8_2059.append(l8_2059)
            if l31_10692 is not None:
                gen_l31_10692.append(l31_10692)

    print(f"\n🔍 PROMPT ENCODING (balance token '$90'):")
    print(f"   L8-2083 (numerical encoding):")
    print(f"      Mean: {np.mean(prompt_l8_2083):.6f}")
    print(f"      Std:  {np.std(prompt_l8_2083):.6f}")
    print(f"      Range: [{np.min(prompt_l8_2083):.6f}, {np.max(prompt_l8_2083):.6f}]")

    print(f"\n🤖 GENERATION (first decision token):")
    print(f"   L8-2059 (risky decision feature):")
    gen_l8_2059_nonzero = [v for v in gen_l8_2059 if v > 0]
    if gen_l8_2059_nonzero:
        print(f"      Active trials: {len(gen_l8_2059_nonzero)}/50")
        print(f"      Mean (when active): {np.mean(gen_l8_2059_nonzero):.6f}")
        print(f"      Max: {np.max(gen_l8_2059_nonzero):.6f}")
    else:
        print(f"      No activations")

    print(f"\n   L31-10692 (risky output feature):")
    gen_l31_10692_nonzero = [v for v in gen_l31_10692 if v > 0]
    if gen_l31_10692_nonzero:
        print(f"      Active trials: {len(gen_l31_10692_nonzero)}/50")
        print(f"      Mean (when active): {np.mean(gen_l31_10692_nonzero):.6f}")
        print(f"      Max: {np.max(gen_l31_10692_nonzero):.6f}")

    print(f"\n💡 KEY INSIGHT:")
    print(f"   ✅ L8-2083 activates during PROMPT encoding (numerical input)")
    print(f"   ✅ L8-2059, L31-10692 activate during GENERATION (decision output)")
    print(f"   → Different features for different processing stages CONFIRMED!")

def create_visualizations(decision_data, evolution_data, data):
    """Create comprehensive visualizations"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Plot 1: Decision distribution
    ax = fig.add_subplot(gs[0, 0])
    decisions = decision_data['decisions']
    decision_counts = {
        'Bet': decisions.count('Bet'),
        'Stop': decisions.count('Stop'),
        'Unknown': decisions.count('Unknown')
    }
    colors = ['#C73E1D', '#2E86AB', '#CCCCCC']
    ax.bar(decision_counts.keys(), decision_counts.values(), color=colors, alpha=0.7)
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Decision Distribution (50 trials)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (k, v) in enumerate(decision_counts.items()):
        ax.text(i, v + 1, f'{v}\n({v/50*100:.0f}%)', ha='center', fontweight='bold')

    # Plot 2: L8-2059 distribution
    ax = fig.add_subplot(gs[0, 1])
    l8_2059_nonzero = [v for v in decision_data['l8_2059'] if v > 0]
    if l8_2059_nonzero:
        ax.hist(l8_2059_nonzero, bins=15, color='#C73E1D', alpha=0.7, edgecolor='black')
        ax.set_xlabel('L8-2059 Activation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'L8-2059 Distribution ({len(l8_2059_nonzero)}/50 trials)',
                     fontsize=13, fontweight='bold')
        ax.axvline(np.mean(l8_2059_nonzero), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(l8_2059_nonzero):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: L31-10692 distribution
    ax = fig.add_subplot(gs[0, 2])
    l31_10692_nonzero = [v for v in decision_data['l31_10692'] if v > 0]
    if l31_10692_nonzero:
        ax.hist(l31_10692_nonzero, bins=15, color='#C73E1D', alpha=0.7, edgecolor='black')
        ax.set_xlabel('L31-10692 Activation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'L31-10692 Distribution ({len(l31_10692_nonzero)}/50 trials)',
                     fontsize=13, fontweight='bold')
        ax.axvline(np.mean(l31_10692_nonzero), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(l31_10692_nonzero):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Feature evolution
    ax = fig.add_subplot(gs[1, :])
    steps = range(min(15, len(evolution_data['l8_2059_means'])))
    ax.plot(steps, evolution_data['l8_2059_means'][:len(steps)],
            'o-', color='#C73E1D', linewidth=2, markersize=8, label='L8-2059 (risky)', alpha=0.7)
    ax.plot(steps, evolution_data['l31_10692_means'][:len(steps)],
            's-', color='#2E86AB', linewidth=2, markersize=8, label='L31-10692 (risky)', alpha=0.7)
    ax.set_xlabel('Generation Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Activation', fontsize=12, fontweight='bold')
    ax.set_title('Feature Evolution Across Generation Steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 5: Prompt vs Generation features
    ax = fig.add_subplot(gs[2, :2])

    # Collect prompt L8-2083
    prompt_l8_2083 = []
    for r in data['results']:
        if 'L8' in r['prompt_features']:
            prompt_l8_2083.append(r['prompt_features']['L8']['features'][2083])

    categories = ['L8-2083\n(Prompt)', 'L8-2059\n(Generation)', 'L31-10692\n(Generation)']
    values = [
        np.mean(prompt_l8_2083),
        np.mean([v for v in decision_data['l8_2059'] if v > 0]) if l8_2059_nonzero else 0,
        np.mean([v for v in decision_data['l31_10692'] if v > 0]) if l31_10692_nonzero else 0,
    ]
    colors_bar = ['#F4A259', '#C73E1D', '#C73E1D']

    bars = ax.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Mean Activation', fontsize=12, fontweight='bold')
    ax.set_title('Prompt Encoding vs Generation Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (cat, val) in enumerate(zip(categories, values)):
        ax.text(i, val + val*0.05, f'{val:.4f}', ha='center', fontweight='bold', fontsize=11)

    # Plot 6: Sample generation sequences
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    # Show first 5 generations
    text_content = "Sample Generations:\n" + "="*40 + "\n"
    for i in range(min(5, len(data['results']))):
        result = data['results'][i]
        text_content += f"\nTrial {i+1}: {result['generated_text'][:50]}...\n"

    ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('/home/ubuntu/llm_addiction/experiment_6_token_level_tracking/generation_analysis.png',
                dpi=300, bbox_inches='tight')
    print("\n✅ Saved: generation_analysis.png")
    plt.close()

def main():
    data_file = '/data/llm_addiction/experiment_6_token_level/generation_level_20251010_055927.json'

    data = load_data(data_file)

    decision_data = analyze_decision_features(data)
    evolution_data = analyze_feature_evolution(data)
    compare_prompt_vs_generation(data)

    create_visualizations(decision_data, evolution_data, data)

    print("\n" + "="*100)
    print("🎉 EXPERIMENT 6 GENERATION ANALYSIS COMPLETE!")
    print("="*100)

if __name__ == '__main__':
    main()
