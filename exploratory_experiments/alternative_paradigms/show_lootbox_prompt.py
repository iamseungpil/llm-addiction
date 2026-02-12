#!/usr/bin/env python3
"""Show example of simplified Lootbox prompt"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lootbox.game_logic import LootBoxGame
from lootbox.run_experiment import LootBoxExperiment

def show_prompt_examples():
    """Generate and display example prompts"""

    experiment = LootBoxExperiment(model_name='llama', gpu_id=0)

    print("=" * 80)
    print("수정된 LOOTBOX 프롬프트 (블랙잭 스타일)")
    print("=" * 80)

    # Example 1: Variable bet, first round
    print("\n" + "=" * 80)
    print("예시 1: Variable Bet - 첫 라운드 (BASE 컴포넌트)")
    print("=" * 80)
    game = LootBoxGame(initial_gems=1000, bet_type='variable')
    prompt = experiment.build_prompt(game, 'variable', 'BASE')
    print(prompt)
    print(f"\n모델이 출력해야 할 것: 숫자 (예: '100', '150') 또는 'Stop'")

    # Example 2: Fixed bet, first round
    print("\n" + "=" * 80)
    print("예시 2: Fixed Bet - 첫 라운드 (BASE 컴포넌트)")
    print("=" * 80)
    game = LootBoxGame(initial_gems=1000, bet_type='fixed')
    prompt = experiment.build_prompt(game, 'fixed', 'BASE')
    print(prompt)
    print(f"\n모델이 출력해야 할 것: 숫자 (예: '100') 또는 'Stop'")

    # Example 3: After a few rounds with history
    print("\n" + "=" * 80)
    print("예시 3: Variable Bet - 히스토리가 있는 경우 (GMHWP 컴포넌트)")
    print("=" * 80)
    game = LootBoxGame(initial_gems=1000, bet_type='variable')
    # Simulate a few rounds
    game.open_variable_box(100)  # Round 1
    game.open_variable_box(150)  # Round 2
    game.open_variable_box(200)  # Round 3
    prompt = experiment.build_prompt(game, 'variable', 'GMHWP', current_goal='Get 2 legendary items')
    print(prompt)
    print(f"\n모델이 출력해야 할 것: 숫자 (예: '100', '250') 또는 'Stop'")

    print("\n" + "=" * 80)
    print("블랙잭과의 비교")
    print("=" * 80)
    print("\n블랙잭 프롬프트 패턴:")
    print('  "Round X: Chips=$Y, Player bets $"')
    print('  → 모델: "10" 또는 "Stop"')
    print("\nLootbox 프롬프트 패턴:")
    print('  "Round X: Gems=Y, Player spends "')
    print('  → 모델: "100" 또는 "Stop"')
    print("\n✓ 동일한 completion 스타일!")
    print("=" * 80)

if __name__ == '__main__':
    show_prompt_examples()
