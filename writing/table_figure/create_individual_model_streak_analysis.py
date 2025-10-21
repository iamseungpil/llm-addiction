#!/usr/bin/env python3
"""Generate individual model streak analysis comparison figure."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/writing/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def calculate_streak_metrics(games, streak_length, streak_type, model_name):
    """Calculate bet increase rate and continuation rate for given streak patterns"""
    bet_increases = []
    continuations = []

    for game in games:
        if 'round_details' not in game:
            continue

        # Handle different data structures
        if model_name == 'GPT-4o-mini':
            # GPT-4o-mini: round_details is a list of rounds
            if not isinstance(game['round_details'], list) or len(game['round_details']) < streak_length + 1:
                continue
            rounds = game['round_details']
        else:
            # Other models: round_details contains game_result
            if not isinstance(game['round_details'], list) or len(game['round_details']) < streak_length + 1:
                continue
            rounds = game['round_details']

        # Find consecutive streaks of the specified type and length
        i = 0
        while i <= len(rounds) - streak_length - 1:
            # Check if we have a streak of the required length
            streak_matches = True
            for j in range(streak_length):
                round_result = rounds[i + j]

                # Extract win/loss information based on model type
                if model_name == 'GPT-4o-mini':
                    # For GPT-4o-mini, calculate win/loss from balance change
                    balance_before = round_result.get('balance_before', 100)
                    bet = round_result.get('bet_amount', 0) or 0

                    # Look at next round's balance to determine win/loss
                    if i + j + 1 < len(rounds):
                        next_balance_before = rounds[i + j + 1].get('balance_before', balance_before)
                        # If balance increased more than bet amount, it's a win
                        won = (next_balance_before > balance_before + bet)
                    else:
                        # Last round, check final balance
                        final_balance = game.get('final_balance', balance_before)
                        won = (final_balance > balance_before + bet)
                else:
                    # Other models: use game_result
                    if 'game_result' not in round_result:
                        streak_matches = False
                        break
                    game_result = round_result['game_result']
                    won = game_result.get('won', game_result.get('win', False)) if game_result else False

                if streak_type == 'win' and not won:
                    streak_matches = False
                    break
                elif streak_type == 'loss' and won:
                    streak_matches = False
                    break

            if streak_matches and i + streak_length < len(rounds):
                # We found a streak, now analyze the next round
                current_round = rounds[i + streak_length - 1]
                next_round = rounds[i + streak_length]

                # Calculate bet increase rate
                current_bet = current_round.get('bet_amount', 0) or 0
                next_bet = next_round.get('bet_amount', 0) or 0

                if current_bet > 0 and next_bet > 0:
                    bet_change = (next_bet - current_bet) / current_bet
                    if bet_change > 0:  # Only count increases
                        bet_increases.append(1)  # Count as an increase
                    else:
                        bet_increases.append(0)  # Count as no increase

                # Check if player continued (didn't quit)
                if next_round.get('decision') == 'continue':
                    continuations.append(1)
                else:
                    continuations.append(0)

                i += streak_length  # Move past this streak
            else:
                i += 1

    bet_increase_rate = np.mean(bet_increases) if bet_increases else 0
    continuation_rate = np.mean(continuations) if continuations else 0

    return bet_increase_rate, continuation_rate

# Load data from all 4 models
model_files = {
    'GPT-4o-mini': '/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250911_071013.json',
    'GPT-4.1-mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'Gemini-2.5-Flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'Claude-3.5-Haiku': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json'
}

# Data structures for results
streak_lengths = range(1, 6)  # 1-5 streak lengths
model_results = {}

print("Processing individual model streak analysis...")

# Process each model
for model_name, file_path in model_files.items():
    print(f"Processing {model_name}...")

    try:
        with open(file_path, 'r') as f:
            model_data = json.load(f)

        # Extract games from the data structure
        if isinstance(model_data, list):
            games = model_data
        elif 'results' in model_data:
            games = model_data['results']
        else:
            games = []

        print(f"Found {len(games)} games for {model_name}")

        # Initialize results for this model
        model_results[model_name] = {
            'win_bet_increase': [],
            'loss_bet_increase': [],
            'win_continuation': [],
            'loss_continuation': []
        }

        # Calculate metrics for each streak length
        for length in streak_lengths:
            # Win streaks
            win_bet_inc, win_cont = calculate_streak_metrics(games, length, 'win', model_name)
            model_results[model_name]['win_bet_increase'].append(win_bet_inc)
            model_results[model_name]['win_continuation'].append(win_cont)

            # Loss streaks
            loss_bet_inc, loss_cont = calculate_streak_metrics(games, length, 'loss', model_name)
            model_results[model_name]['loss_bet_increase'].append(loss_bet_inc)
            model_results[model_name]['loss_continuation'].append(loss_cont)

        print(f"Completed {model_name}")

    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue

# Create 2x4 figure (2 metrics × 4 models)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Individual Model Streak Analysis (Averaged Across Conditions)', fontsize=28, fontweight='bold', y=0.98)

# Colors
colors = {'win': '#2ca02c', 'loss': '#d62728'}  # Green for Win, Red for Loss
x = np.array(streak_lengths)
width = 0.35

model_names = list(model_results.keys())

# Top row: Bet Increase Rate
for i, model_name in enumerate(model_names):
    ax = axes[0, i]

    win_values = model_results[model_name]['win_bet_increase']
    loss_values = model_results[model_name]['loss_bet_increase']

    ax.bar(x - width/2, win_values, width, label='Win Streak',
           color=colors['win'], alpha=0.8)
    ax.bar(x + width/2, loss_values, width, label='Loss Streak',
           color=colors['loss'], alpha=0.8)

    ax.set_title(f'{model_name}', fontsize=24, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Bet Increase Rate', fontsize=20, fontweight='bold')
    ax.set_xticks(streak_lengths)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.7)

    if i == 0:
        ax.legend(fontsize=18)

# Bottom row: Continuation Rate
for i, model_name in enumerate(model_names):
    ax = axes[1, i]

    win_values = model_results[model_name]['win_continuation']
    loss_values = model_results[model_name]['loss_continuation']

    ax.bar(x - width/2, win_values, width, label='Win Streak',
           color=colors['win'], alpha=0.8)
    ax.bar(x + width/2, loss_values, width, label='Loss Streak',
           color=colors['loss'], alpha=0.8)

    if i == 0:
        ax.set_ylabel('Continuation Rate', fontsize=20, fontweight='bold')
    ax.set_xlabel('Streak Length', fontsize=20, fontweight='bold')
    ax.set_xticks(streak_lengths)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.5, 1.05)

plt.tight_layout()

# Save files
png_path = OUTPUT_DIR / 'individual_model_streak_analysis.png'
pdf_path = OUTPUT_DIR / 'individual_model_streak_analysis.pdf'

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.close()

print(f"\n✓ Created individual model streak analysis figure")
print(f"✓ Data source: 4 models with individual breakdowns")
print(f"✓ PNG saved to: {png_path}")
print(f"✓ PDF saved to: {pdf_path}")

# Print summary statistics by model
print("\n=== MODEL-SPECIFIC SUMMARY STATISTICS ===")
for model_name in model_names:
    print(f"\n{model_name}:")
    for i, length in enumerate(streak_lengths):
        win_bet = model_results[model_name]['win_bet_increase'][i]
        loss_bet = model_results[model_name]['loss_bet_increase'][i]
        win_cont = model_results[model_name]['win_continuation'][i]
        loss_cont = model_results[model_name]['loss_continuation'][i]

        print(f"  Streak {length}: Win Bet={win_bet:.3f}, Loss Bet={loss_bet:.3f}, "
              f"Win Cont={win_cont:.3f}, Loss Cont={loss_cont:.3f}")