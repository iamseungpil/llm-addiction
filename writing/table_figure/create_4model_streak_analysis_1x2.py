import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')

def calculate_streak_metrics(games, streak_length, streak_type, model_name=''):
    """Calculate bet increase rate and continuation rate for given streak patterns"""
    bet_increases = []
    continuations = []

    for game in games:
        if 'round_details' not in game:
            continue

        # Handle different data structures
        if model_name == 'gpt4mini':
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
                if model_name == 'gpt4mini':
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
    'GPT-4o-mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'GPT-4.1-mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'Gemini-2.5-Flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'Claude-3.5-Sonnet': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json'
}

# Data structures for aggregated results
streak_lengths = range(1, 6)  # 1-5 streak lengths
win_bet_increase = {length: [] for length in streak_lengths}
loss_bet_increase = {length: [] for length in streak_lengths}
win_continuation = {length: [] for length in streak_lengths}
loss_continuation = {length: [] for length in streak_lengths}

print("Processing 4-model streak analysis...")

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
            # Try to find games in the data structure
            games = []
            for key, value in model_data.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict) and 'rounds' in value[0]:
                        games = value
                        break

        print(f"Found {len(games)} games for {model_name}")

        # Map model name to internal identifier
        model_id = 'gpt4mini' if 'GPT-4o-mini' in model_name else ''

        # Calculate metrics for each streak length
        for length in streak_lengths:
            # Win streaks
            win_bet_inc, win_cont = calculate_streak_metrics(games, length, 'win', model_id)
            win_bet_increase[length].append(win_bet_inc)
            win_continuation[length].append(win_cont)

            # Loss streaks
            loss_bet_inc, loss_cont = calculate_streak_metrics(games, length, 'loss', model_id)
            loss_bet_increase[length].append(loss_bet_inc)
            loss_continuation[length].append(loss_cont)

        print(f"Completed {model_name}")

    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue

# Calculate averages across all 4 models
win_bet_avg = [np.mean(win_bet_increase[l]) for l in streak_lengths]
loss_bet_avg = [np.mean(loss_bet_increase[l]) for l in streak_lengths]
win_cont_avg = [np.mean(win_continuation[l]) for l in streak_lengths]
loss_cont_avg = [np.mean(loss_continuation[l]) for l in streak_lengths]

# Calculate standard errors for error bars
win_bet_se = [np.std(win_bet_increase[l]) / np.sqrt(len(win_bet_increase[l])) if len(win_bet_increase[l]) > 1 else 0 for l in streak_lengths]
loss_bet_se = [np.std(loss_bet_increase[l]) / np.sqrt(len(loss_bet_increase[l])) if len(loss_bet_increase[l]) > 1 else 0 for l in streak_lengths]
win_cont_se = [np.std(win_continuation[l]) / np.sqrt(len(win_continuation[l])) if len(win_continuation[l]) > 1 else 0 for l in streak_lengths]
loss_cont_se = [np.std(loss_continuation[l]) / np.sqrt(len(loss_continuation[l])) if len(loss_continuation[l]) > 1 else 0 for l in streak_lengths]

# Create 1x2 figure
fig, axes = plt.subplots(1, 2, figsize=(18, 5.0))

# Left panel: Bet Increase Rate
x = np.array(streak_lengths)
width = 0.35

axes[0].bar(x - width/2, win_bet_avg, width, yerr=win_bet_se,
            label='Win Streak', color='#2ca02c', alpha=0.8, capsize=5)
axes[0].bar(x + width/2, loss_bet_avg, width, yerr=loss_bet_se,
            label='Loss Streak', color='#d62728', alpha=0.8, capsize=5)

axes[0].set_xlabel('Streak Length', fontsize=26, fontweight='bold')
axes[0].set_ylabel('Bet Increase Rate', fontsize=26, fontweight='bold')
axes[0].set_title('(a) Bet Increase Rate', fontsize=28, fontweight='bold')
axes[0].set_xticks(streak_lengths)
legend = axes[0].legend(fontsize=20, loc='upper left',
                       facecolor='white', edgecolor='gray',
                       framealpha=1.0, shadow=False)
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_boxstyle("round,pad=0.5")
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='both', which='major', labelsize=22)

# Right panel: Continuation Rate
axes[1].bar(x - width/2, win_cont_avg, width, yerr=win_cont_se,
            label='Win Streak', color='#2ca02c', alpha=0.8, capsize=5)
axes[1].bar(x + width/2, loss_cont_avg, width, yerr=loss_cont_se,
            label='Loss Streak', color='#d62728', alpha=0.8, capsize=5)

axes[1].set_xlabel('Streak Length', fontsize=26, fontweight='bold')
axes[1].set_ylabel('Continuation Rate', fontsize=26, fontweight='bold')
axes[1].set_title('(b) Continuation Rate', fontsize=28, fontweight='bold')
axes[1].set_xticks(streak_lengths)
axes[1].set_yticks(np.arange(0, 1.2, 0.2))
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='both', which='major', labelsize=22)

# Add centered color explanation between charts (vertical layout)
# fig.text(0.470, 0.55, 'Win Streak',
#          horizontalalignment='left', verticalalignment='center',
#          fontsize=16, fontweight='bold',
#          color='black', transform=fig.transFigure)
# fig.text(0.470, 0.45, 'Loss Streak',
#          horizontalalignment='left', verticalalignment='center',
#          fontsize=16, fontweight='bold',
#          color='black', transform=fig.transFigure)

# Add colored squares
# fig.text(0.450, 0.55, '■', horizontalalignment='center', verticalalignment='center',
#          fontsize=14, color='#2ca02c', transform=fig.transFigure)
# fig.text(0.450, 0.45, '■', horizontalalignment='center', verticalalignment='center',
#          fontsize=14, color='#d62728', transform=fig.transFigure)

plt.suptitle('Streak Analysis (Averaged Across Models)', fontsize=32, fontweight='bold', y=1.02)
plt.tight_layout(w_pad=20.0)
plt.savefig('/home/ubuntu/llm_addiction/writing/figures/4model_streak_analysis_1x2.png',
            dpi=300, bbox_inches='tight')
plt.savefig('/home/ubuntu/llm_addiction/writing/figures/4model_streak_analysis_1x2.pdf',
            format='pdf', bbox_inches='tight')
plt.close()

print("✓ Created 4-model streak analysis figure")
print(f"✓ Data source: 4 models (GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Sonnet)")
print(f"✓ Streak lengths: 1-5")
print(f"✓ Saved to: /home/ubuntu/llm_addiction/writing/figures/4model_streak_analysis_1x2.png")

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
for i, length in enumerate(streak_lengths):
    print(f"Streak Length {length}:")
    print(f"  Win Bet Increase: {win_bet_avg[i]:.3f} ± {win_bet_se[i]:.3f}")
    print(f"  Loss Bet Increase: {loss_bet_avg[i]:.3f} ± {loss_bet_se[i]:.3f}")
    print(f"  Win Continuation: {win_cont_avg[i]:.3f} ± {win_cont_se[i]:.3f}")
    print(f"  Loss Continuation: {loss_cont_avg[i]:.3f} ± {loss_cont_se[i]:.3f}")