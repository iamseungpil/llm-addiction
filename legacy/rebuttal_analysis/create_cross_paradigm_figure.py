#!/usr/bin/env python3
"""
Create 3-panel Cross-Paradigm Analysis figure
Style matched to Causal Feature Effects figure (Image #3)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Style settings - matching Causal Feature Effects style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.grid'] = False

# Colors
FIXED_COLOR = '#5DADE2'   # Light blue
VARIABLE_COLOR = '#E74C3C'  # Red/coral


def load_slot_machine_data():
    """
    Slot machine data from paper (19,200 games, 6 models)
    Values extracted from the original figure
    """
    return {
        'GPT-4o': {'fixed': 0.0, 'variable': 21.3},
        'GPT-4.1': {'fixed': 0.0, 'variable': 6.3},
        'Gemini': {'fixed': 3.1, 'variable': 48.1},
        'Claude': {'fixed': 0.0, 'variable': 20.5},
        'LLaMA': {'fixed': 0.1, 'variable': 7.1},
        'Gemma': {'fixed': 12.8, 'variable': 29.1},
    }


def load_investment_choice_data():
    """Load investment choice extended CoT data"""
    results_dir = '/data/llm_addiction/investment_choice_extended_cot/results'
    all_games = []

    for f in sorted(os.listdir(results_dir)):
        if f.endswith('.json'):
            with open(os.path.join(results_dir, f)) as fp:
                data = json.load(fp)
                config = data['experiment_config']
                for game in data['results']:
                    game['model'] = config['model']
                    game['bet_type'] = config['bet_type']
                    all_games.append(game)

    return all_games


def calculate_investment_stats(all_games):
    """Calculate bankruptcy and option stats by model and bet type"""
    models = ['gpt4o_mini', 'gpt41_mini', 'gemini_flash', 'claude_haiku']
    model_labels = ['GPT-4o', 'GPT-4.1', 'Gemini', 'Claude']

    # Bankruptcy by model
    bankruptcy_by_model = {}
    for model, label in zip(models, model_labels):
        bankruptcy_by_model[label] = {}
        for bt in ['fixed', 'variable']:
            games = [g for g in all_games if g['model'] == model and g['bet_type'] == bt]
            if games:
                bankrupt = sum(1 for g in games if g['final_balance'] <= 0)
                bankruptcy_by_model[label][bt] = 100 * bankrupt / len(games)
            else:
                bankruptcy_by_model[label][bt] = 0

    # Option distribution
    option_dist = {'fixed': defaultdict(int), 'variable': defaultdict(int)}
    option_totals = {'fixed': 0, 'variable': 0}

    for g in all_games:
        bt = g['bet_type']
        for d in g.get('decisions', []):
            option_dist[bt][d['choice']] += 1
            option_totals[bt] += 1

    option_pcts = {}
    for bt in ['fixed', 'variable']:
        option_pcts[bt] = {}
        for opt in [1, 2, 3, 4]:
            if option_totals[bt] > 0:
                option_pcts[bt][opt] = 100 * option_dist[bt][opt] / option_totals[bt]
            else:
                option_pcts[bt][opt] = 0

    # Calculate within-game bet escalation rate (bet % of balance)
    escalation_count = {'fixed': 0, 'variable': 0}
    total_games = {'fixed': 0, 'variable': 0}

    for g in all_games:
        bt = g['bet_type']
        decisions = g.get('decisions', [])
        if len(decisions) >= 3:
            total_games[bt] += 1

            # Calculate bet % for first and last decisions
            first_d = decisions[0]
            last_d = decisions[-1]

            if first_d['balance_before'] > 0 and last_d['balance_before'] > 0:
                first_pct = first_d['bet'] / first_d['balance_before']
                last_pct = last_d['bet'] / last_d['balance_before']

                if last_pct > first_pct:
                    escalation_count[bt] += 1

    bet_escalation = {}
    for bt in ['fixed', 'variable']:
        if total_games[bt] > 0:
            bet_escalation[bt] = 100 * escalation_count[bt] / total_games[bt]
        else:
            bet_escalation[bt] = 0

    return bankruptcy_by_model, option_pcts, bet_escalation


def create_figure():
    """Create 4-panel figure with unequal widths"""
    # Load data
    slot_data = load_slot_machine_data()
    inv_games = load_investment_choice_data()
    inv_bankruptcy, option_pcts, bet_escalation = calculate_investment_stats(inv_games)

    # Create figure with gridspec for unequal column widths
    # (a) and (b) are wider, (c) and (d) are narrower
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1.3, 0.7, 0.7], wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # ============ Panel (a): Slot Machine ============
    ax = axes[0]
    models_slot = ['GPT-4o', 'GPT-4.1', 'Gemini', 'Claude', 'LLaMA', 'Gemma']
    x = np.arange(len(models_slot))
    width = 0.35

    fixed_vals = [slot_data[m]['fixed'] for m in models_slot]
    variable_vals = [slot_data[m]['variable'] for m in models_slot]

    bars1 = ax.bar(x - width/2, fixed_vals, width, label='Fixed',
                   color=FIXED_COLOR, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, variable_vals, width, label='Variable',
                   color=VARIABLE_COLOR, edgecolor='black', linewidth=1)

    ax.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('(a) Slot Machine',
                 fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models_slot, fontsize=10)
    ax.set_ylim(0, 60)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black')

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ============ Panel (b): Investment Choice by Model ============
    ax = axes[1]
    models_inv = ['GPT-4o', 'GPT-4.1', 'Gemini', 'Claude']
    x = np.arange(len(models_inv))

    fixed_vals = [inv_bankruptcy[m]['fixed'] for m in models_inv]
    variable_vals = [inv_bankruptcy[m]['variable'] for m in models_inv]

    bars1 = ax.bar(x - width/2, fixed_vals, width, label='Fixed',
                   color=FIXED_COLOR, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, variable_vals, width, label='Variable',
                   color=VARIABLE_COLOR, edgecolor='black', linewidth=1)

    ax.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('(b) Investment Choice',
                 fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models_inv, fontsize=10)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='black')

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ============ Panel (c): Option 4 Selection ============
    ax = axes[2]

    # Get Option 4 selection rates
    opt4_fixed = option_pcts['fixed'][4]
    opt4_variable = option_pcts['variable'][4]

    # Bar positions
    bar_width = 0.5
    x = np.arange(2)

    bars = ax.bar(x, [opt4_fixed, opt4_variable], bar_width,
                  color=[FIXED_COLOR, VARIABLE_COLOR], edgecolor='black', linewidth=1)

    ax.set_ylabel('Selection Rate (%)', fontweight='bold')
    ax.set_ylim(0, 50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('(c) Option 4 Selection',
                 fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Fixed', 'Variable'], fontsize=10)

    # Add value labels
    for bar, val in zip(bars, [opt4_fixed, opt4_variable]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ============ Panel (d): Bet Escalation ============
    ax = axes[3]

    # Get within-game bet escalation rates
    fixed_esc = bet_escalation['fixed']
    variable_esc = bet_escalation['variable']

    # Bar positions
    bar_width = 0.5
    x = np.arange(2)

    bars = ax.bar(x, [fixed_esc, variable_esc], bar_width,
                  color=[FIXED_COLOR, VARIABLE_COLOR], edgecolor='black', linewidth=1)

    ax.set_ylabel('Escalation Rate (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title('(d) Bet Escalation',
                 fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Fixed', 'Variable'], fontsize=10)

    # Add value labels
    for bar, val in zip(bars, [fixed_esc, variable_esc]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Main title
    fig.suptitle('Cross-Paradigm Analysis: Betting Type Effects on Bankruptcy',
                 fontsize=18, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def main():
    print("Creating Cross-Paradigm Analysis figure...")

    fig = create_figure()

    # Save figure
    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f'{output_dir}/cross_paradigm_3panel.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/cross_paradigm_3panel.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"\nFigures saved to:")
    print(f"  - {output_dir}/cross_paradigm_3panel.png")
    print(f"  - {output_dir}/cross_paradigm_3panel.pdf")
    plt.close()


if __name__ == '__main__':
    main()
