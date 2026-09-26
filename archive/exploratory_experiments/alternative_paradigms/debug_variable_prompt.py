#!/usr/bin/env python3
"""Debug variable betting prompts"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from blackjack.game_logic import BlackjackGame
from blackjack.run_experiment import BlackjackExperiment

# Create experiment instance
exp = BlackjackExperiment(model_name='llama', gpu_id=0, bet_type='variable')

# Create game
game = BlackjackGame(initial_chips=100, min_bet=10, max_bet=100, bet_type='variable')

print("=" * 80)
print("VARIABLE BETTING PROMPTS")
print("=" * 80)

# Test Phase 1: Continue or Stop
print("\n### Phase 1: Continue or Stop? ###\n")
prompt1 = exp.build_prompt(game, components='BASE', phase='continue_stop')
print(prompt1)
print("\n" + "=" * 80)

# Test Phase 2: Bet amount
print("\n### Phase 2: Bet amount ###\n")
prompt2 = exp.build_prompt(game, components='BASE', phase='bet_amount')
print(prompt2)
print("\n" + "=" * 80)

# Simulate some history
game.round_num = 2
game.chips = 85
game.history = [
    {'round': 1, 'bet': 20, 'player_hand': 'K♠ 9♦ (19)', 'dealer_hand': '7♣ 8♥ (15)', 'outcome': 'win', 'payout': 20, 'chips': 120},
    {'round': 2, 'bet': 35, 'player_hand': '10♥ 6♣ (16)', 'dealer_hand': '9♣ K♦ (19)', 'outcome': 'lose', 'payout': 0, 'chips': 85}
]

print("\n### Phase 1 with History ###\n")
prompt3 = exp.build_prompt(game, components='BASE', phase='continue_stop')
print(prompt3)
print("\n" + "=" * 80)

print("\n### Phase 2 with History ###\n")
prompt4 = exp.build_prompt(game, components='BASE', phase='bet_amount')
print(prompt4)
print("\n" + "=" * 80)
