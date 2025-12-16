#!/usr/bin/env python3
"""
Create Irrationality Index Graph by Prompt and Betting Type
Similar to average_effects_analysis.png but for irrationality metrics
NO HALLUCINATION - All data from actual experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Data paths
STATS_FILE = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/investment_choice_stats.json')
OUTPUT_PNG = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/figures/irrationality_by_condition.png')
OUTPUT_PDF = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/figures/irrationality_by_condition.pdf')

# Load Investment Choice results to analyze by prompt
RESULTS_DIR = Path('/data/llm_addiction/investment_choice_experiment/results')

def load_detailed_results():
    """Load all games with prompt conditions"""
    all_games = []

    for result_file in RESULTS_DIR.glob('*.json'):
        with open(result_file) as f:
            data = json.load(f)

        model = data['experiment_config']['model']
        bet_type = data['experiment_config']['bet_type']

        for game in data['results']:
            game['model'] = model
            game['bet_type'] = bet_type
            all_games.append(game)

    return all_games

def calculate_option4_rate_by_condition(games):
    """Calculate Option 4 selection rate grouped by model, bet_type, prompt"""
    stats = defaultdict(lambda: {'option4': 0, 'total': 0})

    for game in games:
        model = game['model']
        bet_type = game['bet_type']
        prompt = game.get('prompt_condition', 'BASE')

        key = (model, bet_type, prompt)

        for decision in game.get('decisions', []):
            choice = decision.get('choice')
            if choice is not None:
                stats[key]['total'] += 1
                if choice == 4:
                    stats[key]['option4'] += 1

    # Calculate rates
    results = {}
    for key, counts in stats.items():
        if counts['total'] > 0:
            rate = (counts['option4'] / counts['total']) * 100
            results[key] = rate
        else:
            results[key] = 0.0

    return results

def create_irrationality_graph():
    """Create graph showing Option 4 rate (irrationality proxy) by condition"""
    print("Loading detailed results...")
    games = load_detailed_results()
    print(f"Loaded {len(games)} games")

    print("\nCalculating Option 4 rates by condition...")
    rates = calculate_option4_rate_by_condition(games)

    # Models and prompts
    models = ['gpt4o_mini', 'gpt41_mini', 'gemini_flash', 'claude_haiku']
    model_labels = ['GPT-4o-mini', 'GPT-4.1-mini', 'Gemini-2.5-Flash', 'Claude-3.5-Haiku']
    prompts = ['BASE', 'G', 'M', 'GM']
    bet_types = ['fixed', 'variable']

    # Create figure - 4x1 layout
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    x = np.arange(len(prompts))
    width = 0.35

    for idx, (model, model_label) in enumerate(zip(models, model_labels)):
        ax = axes[idx]

        fixed_values = [rates.get((model, 'fixed', prompt), 0) for prompt in prompts]
        variable_values = [rates.get((model, 'variable', prompt), 0) for prompt in prompts]

        bars1 = ax.bar(x - width/2, fixed_values, width, label='Fixed Betting',
                      color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, variable_values, width, label='Variable Betting',
                      color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_ylabel('Option 4 Selection Rate (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prompt Condition', fontsize=14, fontweight='bold')
        ax.set_title(model_label, fontsize=16, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(prompts, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 105)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 2:  # Only show label if bar is visible
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Option 4 Selection Rate (Irrationality Indicator) by Prompt and Betting Type',
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save both PNG and PDF
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
    print(f"\n✅ Figures saved:")
    print(f"   PNG: {OUTPUT_PNG}")
    print(f"   PDF: {OUTPUT_PDF}")

    # Print statistics
    print("\n" + "="*80)
    print("OPTION 4 SELECTION RATES BY CONDITION")
    print("="*80)

    for model, model_label in zip(models, model_labels):
        print(f"\n{model_label}:")
        for bet_type in bet_types:
            print(f"  {bet_type.capitalize()}:")
            for prompt in prompts:
                rate = rates.get((model, bet_type, prompt), 0)
                print(f"    {prompt}: {rate:.2f}%")

    print("="*80)

    return rates

if __name__ == '__main__':
    print("="*80)
    print("Creating Irrationality Index Graph")
    print("="*80)
    rates = create_irrationality_graph()
    print("\n✅ Done!")
