#!/usr/bin/env python3
"""Generate 4x2 streak analysis figure with REAL Claude data - no proxy."""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Model data sources
MODEL_PATHS = {
    'gpt4mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'gpt41mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'gemini25flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
}

MODEL_DISPLAY_NAMES = {
    'gpt4mini': 'GPT-4o-mini',
    'gpt41mini': 'GPT-4.1-mini',
    'gemini25flash': 'Gemini-2.5-Flash',
    'claude': 'Claude-3.5-Sonnet',
}

def extract_game_history(exp, model):
    """Extract game history from experiment data"""
    if model == 'gpt4mini':
        return exp.get('game_history', [])
    elif model in ['gpt41mini', 'gemini25flash', 'claude']:
        round_details = exp.get('round_details', [])
        history = []
        for rd in round_details:
            if isinstance(rd, dict) and 'game_result' in rd:
                game_result = rd['game_result']
                if isinstance(game_result, dict):
                    history.append(game_result)
        return history
    return []

def identify_streaks(history):
    """Identify win/loss streaks in game history"""
    if not history:
        return []

    streaks = []
    current_streak_type = None
    current_streak_length = 0

    for i, round_data in enumerate(history):
        is_win = round_data.get('win', False)

        if current_streak_type is None:
            # First round
            current_streak_type = 'win' if is_win else 'loss'
            current_streak_length = 1
        elif (is_win and current_streak_type == 'win') or (not is_win and current_streak_type == 'loss'):
            # Streak continues
            current_streak_length += 1
        else:
            # Streak breaks
            streaks.append({
                'type': current_streak_type,
                'length': current_streak_length,
                'end_index': i - 1
            })
            current_streak_type = 'win' if is_win else 'loss'
            current_streak_length = 1

    # Add final streak
    if current_streak_length > 0:
        streaks.append({
            'type': current_streak_type,
            'length': current_streak_length,
            'end_index': len(history) - 1
        })

    return streaks

def calculate_streak_metrics(exp, model):
    """Calculate streak-based metrics for an experiment"""
    history = extract_game_history(exp, model)
    if len(history) < 2:
        return []

    streaks = identify_streaks(history)
    streak_metrics = []

    for i, streak in enumerate(streaks):
        if streak['length'] > 5:  # Limit to streaks of length 1-5
            continue

        # Check if there's a next round after streak (for continuation)
        next_round_idx = streak['end_index'] + 1
        if next_round_idx < len(history):
            continued = True  # Player continued after streak
        else:
            continued = False  # Game ended or player stopped

        # Check for bet increase (if there's a next bet)
        bet_increased = False
        if next_round_idx < len(history):
            current_bet = history[streak['end_index']].get('bet', 0)
            next_bet = history[next_round_idx].get('bet', 0)
            if next_bet > current_bet:
                bet_increased = True

        streak_metrics.append({
            'type': streak['type'],
            'length': streak['length'],
            'continued': continued,
            'bet_increased': bet_increased
        })

    return streak_metrics

def load_and_analyze_model(model_key):
    """Load and analyze streak data for one model"""
    print(f"Analyzing {model_key}...")

    try:
        with open(MODEL_PATHS[model_key], 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {model_key}: {e}")
        return None

    experiments = data.get('results', [])
    print(f"  Loaded {len(experiments)} experiments")

    # Collect streak metrics
    streak_stats = defaultdict(lambda: {'continued': 0, 'bet_increased': 0, 'total': 0})

    for exp in experiments:
        streak_metrics = calculate_streak_metrics(exp, model_key)
        for metric in streak_metrics:
            key = f"{metric['length']}-{metric['type']}"
            streak_stats[key]['total'] += 1
            if metric['continued']:
                streak_stats[key]['continued'] += 1
            if metric['bet_increased']:
                streak_stats[key]['bet_increased'] += 1

    # Calculate rates
    model_results = {}
    for streak_key, stats in streak_stats.items():
        if stats['total'] > 0:
            model_results[streak_key] = {
                'continuation_rate': stats['continued'] / stats['total'],
                'bet_increase_rate': stats['bet_increased'] / stats['total'],
                'occurrences': stats['total']
            }

    print(f"  Found {len(model_results)} streak types")
    return model_results

def create_4x2_streak_analysis_corrected():
    """Create corrected 4x2 streak analysis figure with real data"""
    models = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']
    streak_lengths = range(1, 6)

    # Load and analyze all models
    all_model_data = {}
    for model in models:
        model_data = load_and_analyze_model(model)
        if model_data:
            all_model_data[model] = model_data

    # Create 4x2 figure
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))

    for row, model_key in enumerate(models):
        if model_key not in all_model_data:
            print(f"Warning: No data for {model_key}")
            continue

        model_name = MODEL_DISPLAY_NAMES[model_key]
        model_data = all_model_data[model_key]

        # Extract data for plotting
        win_bet_increase = []
        loss_bet_increase = []
        win_continuation = []
        loss_continuation = []

        for length in streak_lengths:
            win_key = f"{length}-win"
            loss_key = f"{length}-loss"

            if win_key in model_data:
                win_bet_increase.append(model_data[win_key]['bet_increase_rate'])
                win_continuation.append(model_data[win_key]['continuation_rate'])
            else:
                win_bet_increase.append(0)
                win_continuation.append(0)

            if loss_key in model_data:
                loss_bet_increase.append(model_data[loss_key]['bet_increase_rate'])
                loss_continuation.append(model_data[loss_key]['continuation_rate'])
            else:
                loss_bet_increase.append(0)
                loss_continuation.append(0)

        # Left column: Bet Increase Rate
        ax1 = axes[row, 0]
        x = np.array(streak_lengths)
        width = 0.35

        bars1 = ax1.bar(x - width/2, win_bet_increase, width,
                       label='Win Streak', color='#4CAF50', alpha=0.8)
        bars2 = ax1.bar(x + width/2, loss_bet_increase, width,
                       label='Loss Streak', color='#F44336', alpha=0.8)

        ax1.set_ylabel('Bet Increase Rate', fontsize=12)
        ax1.set_xticks(streak_lengths)
        ax1.grid(axis='y', alpha=0.3)

        if row == 0:
            ax1.legend(fontsize=10)
            ax1.set_title('Average Bet Increase Rate', fontsize=14, fontweight='bold')

        if row == len(models) - 1:
            ax1.set_xlabel('Streak Length', fontsize=12)

        # Right column: Continuation Rate
        ax2 = axes[row, 1]

        bars3 = ax2.bar(x - width/2, win_continuation, width,
                       label='Win Streak', color='#4CAF50', alpha=0.8)
        bars4 = ax2.bar(x + width/2, loss_continuation, width,
                       label='Loss Streak', color='#F44336', alpha=0.8)

        ax2.set_ylabel('Continuation Rate', fontsize=12)
        ax2.set_xticks(streak_lengths)
        ax2.grid(axis='y', alpha=0.3)

        if row == 0:
            ax2.legend(fontsize=10)
            ax2.set_title('Average Continuation Rate', fontsize=14, fontweight='bold')

        if row == len(models) - 1:
            ax2.set_xlabel('Streak Length', fontsize=12)

        # Add model name
        ax1.text(0.02, 0.95, model_name, transform=ax1.transAxes,
                fontsize=13, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Adjust tick sizes
        ax1.tick_params(axis='both', labelsize=10)
        ax2.tick_params(axis='both', labelsize=10)

    plt.suptitle('CORRECTED Streak Analysis by Model (Real Data)', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save corrected figure
    png_path = '/home/ubuntu/llm_addiction/writing/figures/appendix/streak_analysis_4x2_CORRECTED.png'
    pdf_path = '/home/ubuntu/llm_addiction/writing/figures/appendix/streak_analysis_4x2_CORRECTED.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"\n✅ CORRECTED 4x2 Streak Analysis Figure saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"\n🔍 Data Sources Used:")
    for model, path in MODEL_PATHS.items():
        print(f"  {model}: {path}")

def main():
    print("=" * 80)
    print("GENERATING CORRECTED 4x2 STREAK ANALYSIS (NO PROXY DATA)")
    print("=" * 80)
    create_4x2_streak_analysis_corrected()
    print("\n" + "=" * 80)
    print("✅ CORRECTED ANALYSIS COMPLETE - ALL MODELS USE REAL DATA")
    print("=" * 80)

if __name__ == '__main__':
    main()