#!/usr/bin/env python3
"""
Create 1x2 streak analysis figure (Fixed vs Variable)
Using Section 2 formula: I_Chasing = max(0, (r_{t+1} - r_t) / r_t)
where r_t = bet_t / balance_t

Data: 19,200 games (6 models × 3,200 each)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13

# Data files - 6 models, each with 3,200 experiments
DATA_FILES = {
    'GPT-4o': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'GPT-4.1': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'Gemini': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'Claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
    'LLaMA': '/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json',
    'Gemma': '/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json',
}


def extract_game_history(exp, model):
    """Extract game history based on model-specific data structure"""
    if model in ['LLaMA', 'Gemma']:
        history = exp.get('history', [])
        if history:
            return history
        return exp.get('game_history', [])
    elif model == 'GPT-4o':
        game_history = exp.get('game_history', [])
        if game_history:
            return game_history
        round_details = exp.get('round_details', [])
        if round_details:
            return [rd.get('game_result', {}) for rd in round_details if isinstance(rd, dict)]
        return []
    else:
        round_details = exp.get('round_details', [])
        if round_details:
            return [rd.get('game_result', {}) for rd in round_details if isinstance(rd, dict)]
        return []


def reconstruct_balance_before(round_data):
    """Reconstruct balance before bet from round data"""
    bet = round_data.get('bet', 0)
    balance_after = round_data.get('balance', 100)
    win = round_data.get('win', False)
    win_amount = bet * 3 if win else 0
    return balance_after + bet - win_amount


def load_all_data():
    """Load all slot machine data"""
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


def calculate_ichasing(all_experiments):
    """Calculate I_Chasing values using Section 2 method"""

    results = {
        'fixed_win': defaultdict(list),
        'fixed_loss': defaultdict(list),
        'variable_win': defaultdict(list),
        'variable_loss': defaultdict(list),
    }

    for exp in all_experiments:
        bet_type = exp.get('bet_type', 'unknown')
        if bet_type not in ['fixed', 'variable']:
            continue

        model = exp.get('model', 'unknown')
        history = extract_game_history(exp, model)
        if not history or len(history) < 2:
            continue

        for i in range(len(history) - 1):
            rd = history[i]
            next_rd = history[i + 1]
            if not isinstance(rd, dict) or not isinstance(next_rd, dict):
                continue

            # Current round result
            is_win = rd.get('win', False)
            streak_type = 'win' if is_win else 'loss'

            # Calculate streak length
            streak_len = 1
            for j in range(i - 1, -1, -1):
                if not isinstance(history[j], dict):
                    break
                if history[j].get('win', False) == is_win:
                    streak_len += 1
                else:
                    break
            streak_len = min(streak_len, 5)

            # Get bet amounts
            curr_bet = rd.get('bet', 10)
            next_bet = next_rd.get('bet', 10)

            # Calculate balance before current bet
            balance_before = reconstruct_balance_before(rd)
            balance_after = rd.get('balance', 100)

            if balance_before <= 0 or balance_after <= 0:
                continue

            # Calculate betting ratios
            r_t = curr_bet / balance_before
            r_t1 = next_bet / balance_after

            # I_Chasing = max(0, (r_{t+1} - r_t) / r_t)
            if r_t > 0:
                ichasing = max(0, (r_t1 - r_t) / r_t)
            else:
                ichasing = 0

            key = f'{bet_type}_{streak_type}'
            results[key][streak_len].append(ichasing)

    return results


def create_figure(results):
    """Create 1x2 comparison figure"""

    streaks = [1, 2, 3, 4, 5]

    # Calculate means
    fixed_win = [np.mean(results['fixed_win'].get(s, [0])) for s in streaks]
    fixed_loss = [np.mean(results['fixed_loss'].get(s, [0])) for s in streaks]
    variable_win = [np.mean(results['variable_win'].get(s, [0])) for s in streaks]
    variable_loss = [np.mean(results['variable_loss'].get(s, [0])) for s in streaks]

    # Get n values
    fixed_win_n = sum(len(results['fixed_win'].get(s, [])) for s in streaks)
    fixed_loss_n = sum(len(results['fixed_loss'].get(s, [])) for s in streaks)
    variable_win_n = sum(len(results['variable_win'].get(s, [])) for s in streaks)
    variable_loss_n = sum(len(results['variable_loss'].get(s, [])) for s in streaks)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(wspace=0.4)

    x = np.arange(len(streaks))
    width = 0.35

    fig.suptitle(r'Betting Ratio Increase by Streak Length',
                 fontsize=18, fontweight='bold', y=0.94)

    # Panel (a): After Win
    ax = axes[0]
    bars1 = ax.bar(x - width/2, fixed_win, width, label='Fixed', color='#27ae60', edgecolor='black')
    bars2 = ax.bar(x + width/2, variable_win, width, label='Variable', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Betting Ratio Increase')
    ax.set_title('(a) After Win', pad=10, fontsize=17, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in streaks])
    ax.set_xlabel('Consecutive Streak Length')
    ax.set_ylim(0, 0.7)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, fixed_win):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, color='green')
    for bar, val in zip(bars2, variable_win):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, color='red')

    # Panel (b): After Loss
    ax = axes[1]
    bars1 = ax.bar(x - width/2, fixed_loss, width, label='Fixed', color='#27ae60', edgecolor='black')
    bars2 = ax.bar(x + width/2, variable_loss, width, label='Variable', color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Betting Ratio Increase')
    ax.set_title('(b) After Loss', pad=10, fontsize=17, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in streaks])
    ax.set_xlabel('Consecutive Streak Length')
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, fixed_loss):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, color='green')
    for bar, val in zip(bars2, variable_loss):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=12, color='red')

    plt.tight_layout()

    return fig, {
        'fixed_win': fixed_win,
        'fixed_loss': fixed_loss,
        'variable_win': variable_win,
        'variable_loss': variable_loss,
        'fixed_win_n': fixed_win_n,
        'fixed_loss_n': fixed_loss_n,
        'variable_win_n': variable_win_n,
        'variable_loss_n': variable_loss_n,
    }


def main():
    print("=" * 70)
    print("STREAK ANALYSIS: Fixed vs Variable Betting")
    print("Using Section 2 Method: I_Chasing = max(0, (r_{t+1} - r_t) / r_t)")
    print("=" * 70)

    print("\nLoading data...")
    all_experiments = load_all_data()

    total = len(all_experiments)
    fixed = len([e for e in all_experiments if e.get('bet_type') == 'fixed'])
    variable = len([e for e in all_experiments if e.get('bet_type') == 'variable'])

    print(f"\nTotal experiments: {total}")
    print(f"Fixed betting: {fixed}")
    print(f"Variable betting: {variable}")

    print("\nCalculating I_Chasing values...")
    results = calculate_ichasing(all_experiments)

    print("\nCreating figure...")
    fig, stats = create_figure(results)

    # Save figures
    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    fig.savefig(f'{output_dir}/streak_analysis_1x2_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/streak_analysis_1x2_comparison.pdf', bbox_inches='tight', facecolor='white')

    # Copy to iclr2026/images
    import shutil
    shutil.copy(f'{output_dir}/streak_analysis_1x2_comparison.pdf',
                '/home/ubuntu/llm_addiction/rebuttal_analysis/iclr2026/images/')

    plt.close()

    print(f"\nFigures saved:")
    print(f"  {output_dir}/streak_analysis_1x2_comparison.png")
    print(f"  {output_dir}/streak_analysis_1x2_comparison.pdf")
    print(f"  /home/ubuntu/llm_addiction/rebuttal_analysis/iclr2026/images/streak_analysis_1x2_comparison.pdf")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nValues (mean I_Chasing by streak length):")
    print(f"  Fixed Win:     {[f'{v:.2f}' for v in stats['fixed_win']]}")
    print(f"  Fixed Loss:    {[f'{v:.2f}' for v in stats['fixed_loss']]}")
    print(f"  Variable Win:  {[f'{v:.2f}' for v in stats['variable_win']]}")
    print(f"  Variable Loss: {[f'{v:.2f}' for v in stats['variable_loss']]}")

    print("\nSample sizes:")
    print(f"  Fixed Win:     n={stats['fixed_win_n']:,}")
    print(f"  Fixed Loss:    n={stats['fixed_loss_n']:,}")
    print(f"  Variable Win:  n={stats['variable_win_n']:,}")
    print(f"  Variable Loss: n={stats['variable_loss_n']:,}")

    # Caption values
    fixed_win_avg = np.mean(stats['fixed_win'])
    variable_win_avg = np.mean(stats['variable_win'])
    fixed_loss_avg = np.mean(stats['fixed_loss'])
    variable_loss_avg = np.mean(stats['variable_loss'])

    print("\nFor caption:")
    print(f"  After Win:  Variable/Fixed = {variable_win_avg/fixed_win_avg:.1f}x ({variable_win_avg:.2f} vs {fixed_win_avg:.2f})")
    print(f"  After Loss: Variable/Fixed = {variable_loss_avg/fixed_loss_avg:.1f}x ({variable_loss_avg:.2f} vs {fixed_loss_avg:.2f})")


if __name__ == '__main__':
    main()
