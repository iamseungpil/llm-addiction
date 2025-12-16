#!/usr/bin/env python3
"""
Create streak analysis figure for slot machine experiment.
Uses FINAL files only (no checkpoints).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Data files - FINAL files only
DATA_FILES = {
    'GPT-4o': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'GPT-4.1': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'Gemini': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'Claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
    'LLaMA': '/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json',
    'Gemma': '/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json',
}


def reconstruct_balance_before(round_data):
    """Reconstruct balance before bet from round data"""
    bet = round_data.get('bet', 0)
    balance_after = round_data.get('balance', 100)
    win = round_data.get('win', False)
    win_amount = bet * 3 if win else 0
    return balance_after + bet - win_amount


def extract_game_history(exp, model):
    """Extract game history based on model-specific data structure"""
    # LLaMA and Gemma use 'history'
    if model in ['LLaMA', 'Gemma']:
        history = exp.get('history', [])
        if history:
            return history
        return exp.get('game_history', [])

    # GPT-4o uses 'game_history'
    elif model == 'GPT-4o':
        game_history = exp.get('game_history', [])
        if game_history:
            return game_history
        # Fallback to round_details
        round_details = exp.get('round_details', [])
        if round_details:
            extracted = []
            for rd in round_details:
                if isinstance(rd, dict) and 'game_result' in rd:
                    gr = rd['game_result']
                    if isinstance(gr, dict):
                        extracted.append(gr)
            return extracted
        return []

    # GPT-4.1, Gemini, Claude use 'round_details' with 'game_result'
    elif model in ['GPT-4.1', 'Gemini', 'Claude']:
        round_details = exp.get('round_details', [])
        if round_details:
            extracted = []
            for rd in round_details:
                if isinstance(rd, dict) and 'game_result' in rd:
                    gr = rd['game_result']
                    if isinstance(gr, dict):
                        extracted.append(gr)
            return extracted
        return []

    # Default fallback
    return exp.get('history', exp.get('game_history', []))


def load_all_data():
    """Load all slot machine data from final files only"""
    all_experiments = []

    for model, path in DATA_FILES.items():
        try:
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict) and 'results' in data:
                    results = data['results']
                elif isinstance(data, list):
                    results = data
                else:
                    results = []

                for exp in results:
                    exp['model'] = model
                    all_experiments.append(exp)

                print(f"Loaded {len(results)} experiments for {model}")
        except Exception as e:
            print(f"Error loading {model}: {e}")

    return all_experiments


def analyze_streak_behavior(all_experiments):
    """Analyze bet increase and continuation rates by streak"""

    # Only variable betting
    variable_exps = [e for e in all_experiments if e.get('bet_type') == 'variable']

    # Track by streak
    win_streak_increase = defaultdict(list)  # streak_length -> [0/1 for bet increased]
    loss_streak_increase = defaultdict(list)
    win_streak_continue = defaultdict(list)  # streak_length -> [0/1 for continued]
    loss_streak_continue = defaultdict(list)

    for exp in variable_exps:
        model = exp.get('model', 'unknown')
        history = extract_game_history(exp, model)
        if not history or len(history) < 2:
            continue

        for i in range(len(history)):
            rd = history[i]
            if not isinstance(rd, dict):
                continue

            # Calculate current streak length
            streak_type = 'win' if rd.get('win', False) else 'loss'
            streak_len = 1
            for j in range(i - 1, -1, -1):
                if not isinstance(history[j], dict):
                    break
                if history[j].get('win', False) == rd.get('win', False):
                    streak_len += 1
                else:
                    break

            streak_len = min(streak_len, 5)  # Cap at 5

            # Check balance after this round
            balance = rd.get('balance', 0)
            if balance <= 0:
                continue  # Can't continue if bankrupt

            # Did they continue?
            if i + 1 < len(history):
                continued = 1
                next_rd = history[i + 1]

                # Did bet increase?
                prev_bet = rd.get('bet', 10)
                curr_bet = next_rd.get('bet', 10) if isinstance(next_rd, dict) else 10

                if isinstance(prev_bet, (int, float)) and isinstance(curr_bet, (int, float)) and prev_bet > 0:
                    bet_increased = 1 if curr_bet > prev_bet else 0
                else:
                    bet_increased = 0

                if streak_type == 'win':
                    win_streak_increase[streak_len].append(bet_increased)
                    win_streak_continue[streak_len].append(continued)
                else:
                    loss_streak_increase[streak_len].append(bet_increased)
                    loss_streak_continue[streak_len].append(continued)
            else:
                # Last round - they stopped
                if streak_type == 'win':
                    win_streak_continue[streak_len].append(0)
                else:
                    loss_streak_continue[streak_len].append(0)

    return {
        'win_increase': win_streak_increase,
        'loss_increase': loss_streak_increase,
        'win_continue': win_streak_continue,
        'loss_continue': loss_streak_continue,
    }


def create_figure(streak_data):
    """Create 2-panel streak analysis figure"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plt.subplots_adjust(wspace=0.4)

    streaks = [1, 2, 3, 4, 5]
    x = np.arange(len(streaks))
    width = 0.35

    # Panel (a): Bet Increase Rate
    ax = axes[0]

    win_rates = []
    loss_rates = []
    for s in streaks:
        win_data = streak_data['win_increase'].get(s, [])
        loss_data = streak_data['loss_increase'].get(s, [])
        win_rates.append(np.mean(win_data) if win_data else 0)
        loss_rates.append(np.mean(loss_data) if loss_data else 0)

    bars1 = ax.bar(x - width/2, win_rates, width, label='After Win', color='#27ae60', edgecolor='black')
    bars2 = ax.bar(x + width/2, loss_rates, width, label='After Loss', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Bet Increase Rate')
    ax.set_title('(a) Bet Increase Rate by Streak Length', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in streaks])
    ax.set_xlabel('Consecutive Streak Length')
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars1, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, rate in zip(bars2, loss_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel (b): Continuation Rate
    ax = axes[1]

    win_cont = []
    loss_cont = []
    for s in streaks:
        win_data = streak_data['win_continue'].get(s, [])
        loss_data = streak_data['loss_continue'].get(s, [])
        win_cont.append(np.mean(win_data) * 100 if win_data else 0)
        loss_cont.append(np.mean(loss_data) * 100 if loss_data else 0)

    bars1 = ax.bar(x - width/2, win_cont, width, label='After Win', color='#27ae60', edgecolor='black')
    bars2 = ax.bar(x + width/2, loss_cont, width, label='After Loss', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Continuation Rate (%)')
    ax.set_title('(b) Continuation Rate by Streak Length', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in streaks])
    ax.set_xlabel('Consecutive Streak Length')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars1, win_cont):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, rate in zip(bars2, loss_cont):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig, win_rates, loss_rates, win_cont, loss_cont


def main():
    print("=" * 60)
    print("STREAK ANALYSIS FOR SLOT MACHINE EXPERIMENT")
    print("(Using FINAL files only - no checkpoints)")
    print("=" * 60)

    print("\nLoading data...")
    all_experiments = load_all_data()

    total = len(all_experiments)
    variable = len([e for e in all_experiments if e.get('bet_type') == 'variable'])
    print(f"\nTotal experiments: {total}")
    print(f"Variable betting: {variable}")

    print("\nAnalyzing streak behavior...")
    streak_data = analyze_streak_behavior(all_experiments)

    print("\nCreating figure...")
    fig, win_inc, loss_inc, win_cont, loss_cont = create_figure(streak_data)

    # Save
    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    fig.savefig(f'{output_dir}/streak_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/streak_analysis.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nFigures saved to {output_dir}/streak_analysis.png")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nBet Increase Rate (Win Streak 1→5):")
    print(f"  {win_inc[0]:.3f} → {win_inc[4]:.3f}")
    print("\nBet Increase Rate (Loss Streak 1→5):")
    print(f"  {loss_inc[0]:.3f} → {loss_inc[4]:.3f}")
    print("\nContinuation Rate (Win Streak 1→5):")
    print(f"  {win_cont[0]:.1f}% → {win_cont[4]:.1f}%")
    print("\nContinuation Rate (Loss Streak 1→5):")
    print(f"  {loss_cont[0]:.1f}% → {loss_cont[4]:.1f}%")


if __name__ == '__main__':
    main()
