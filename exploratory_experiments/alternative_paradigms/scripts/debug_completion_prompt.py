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


# Lootbox test removed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['blackjack'], default='blackjack')
    args = parser.parse_args()

    if args.test == 'blackjack':
        test_blackjack_prompt()
