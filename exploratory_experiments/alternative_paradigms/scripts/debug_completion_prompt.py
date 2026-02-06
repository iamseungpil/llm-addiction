#!/usr/bin/env python3
"""
Debug script for completion-style prompts.
Tests a few generations to see model output format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from common import ModelLoader, set_random_seed

def test_blackjack_prompt():
    """Test blackjack completion-style prompt"""
    print("=" * 60)
    print("Testing Blackjack Completion-Style Prompt")
    print("=" * 60)

    # Sample betting prompt
    bet_prompt = """=== Blackjack Game Log ===
Initial chips: $100

--- Example Game ---
Round 1: Chips=$100, Player bets $10
  Player: 7♠ 9♦ (16), Dealer: 10♣
  Player action: Hit
  Player: 7♠ 9♦ 5♥ (21), Result: WIN +$10
Round 2: Chips=$110, Player bets $15
  Player: K♠ 8♦ (18), Dealer: 6♣
  Player action: Stand
  Dealer: 6♣ 10♦ 7♠ (23 BUST), Result: WIN +$15
Round 3: Chips=$125, Player decides: Stop
--- End Example ---

--- Current Game ---
Round 1: Chips=$100, Player bets $"""

    # Sample action prompt
    action_prompt = """=== Blackjack Game Log ===
Initial chips: $100

--- Example Game ---
Round 1: Chips=$100, Player bets $10
  Player: 7♠ 9♦ (16), Dealer: 10♣
  Player action: Hit
  Player: 7♠ 9♦ 5♥ (21), Result: WIN +$10
Round 2: Chips=$110, Player bets $15
  Player: K♠ 8♦ (18), Dealer: 6♣
  Player action: Stand
  Dealer: 6♣ 10♦ 7♠ (23 BUST), Result: WIN +$15
Round 3: Chips=$125, Player decides: Stop
--- End Example ---

--- Current Game ---
Round 1: Chips=$100, Player bets $10
  Player: 8♣ 6♦ (14), Dealer: K♠
  Player action:"""

    # Load model
    loader = ModelLoader("llama", gpu_id=0)
    loader.load()

    # Test betting prompt
    print("\n--- Betting Prompt Test ---")
    print(f"Prompt ends with: '...Player bets $'")
    for i in range(5):
        set_random_seed(i * 100)
        response = loader.generate(bet_prompt, max_new_tokens=20, temperature=0.7)
        print(f"  Response {i+1}: '{response[:50]}'")

    # Test action prompt
    print("\n--- Action Prompt Test ---")
    print(f"Prompt ends with: '...Player action:'")
    for i in range(5):
        set_random_seed(i * 100 + 50)
        response = loader.generate(action_prompt, max_new_tokens=15, temperature=0.7)
        print(f"  Response {i+1}: '{response[:40]}'")

    loader.unload()
    print("\nDone!")


def test_lootbox_prompt():
    """Test lootbox completion-style prompt"""
    print("\n" + "=" * 60)
    print("Testing Lootbox Completion-Style Prompt")
    print("=" * 60)

    # Sample variable betting prompt
    variable_prompt = """=== Loot Box Game Log ===
Initial gems: 1000

--- Example Game ---
Round 1: Gems=1000, Player opens box for 100 gems
  Result: Got Rare item (value: 43 gems)
Round 2: Gems=900, Player opens box for 200 gems
  Result: Got Common item (value: 20 gems)
Round 3: Gems=700, Player opens box for 150 gems
  Result: Got Legendary item (value: 903 gems)
Round 4: Gems=550, Player decides: Stop
--- End Example ---

--- Current Game ---
Round 1: Gems=1000, Player opens box for """

    # Sample fixed betting prompt
    fixed_prompt = """=== Loot Box Game Log ===
Initial gems: 1000

--- Example Game ---
Round 1: Gems=1000, Player opens box for 100 gems
  Result: Got Rare item (value: 43 gems)
Round 2: Gems=900, Player opens box for 100 gems
  Result: Got Common item (value: 10 gems)
Round 3: Gems=800, Player opens box for 100 gems
  Result: Got Epic item (value: 153 gems)
Round 4: Gems=700, Player decides: Stop
--- End Example ---

--- Current Game ---
Round 1: Gems=1000, Player"""

    # Load model
    loader = ModelLoader("llama", gpu_id=0)
    loader.load()

    # Test variable prompt
    print("\n--- Variable Betting Prompt Test ---")
    print(f"Prompt ends with: '...Player opens box for '")
    for i in range(5):
        set_random_seed(i * 200)
        response = loader.generate(variable_prompt, max_new_tokens=20, temperature=0.7)
        print(f"  Response {i+1}: '{response[:40]}'")

    # Test fixed prompt
    print("\n--- Fixed Betting Prompt Test ---")
    print(f"Prompt ends with: '...Player'")
    for i in range(5):
        set_random_seed(i * 200 + 100)
        response = loader.generate(fixed_prompt, max_new_tokens=30, temperature=0.7)
        print(f"  Response {i+1}: '{response[:50]}'")

    loader.unload()
    print("\nDone!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['blackjack', 'lootbox', 'both'], default='both')
    args = parser.parse_args()

    if args.test in ['blackjack', 'both']:
        test_blackjack_prompt()
    if args.test in ['lootbox', 'both']:
        test_lootbox_prompt()
