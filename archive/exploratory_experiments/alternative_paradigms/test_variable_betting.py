#!/usr/bin/env python3
"""
Simple test for variable betting with free-form amount selection.
Tests if models can generate valid bet amounts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from common import ModelLoader, set_random_seed

def test_betting_prompt():
    """Test if model can generate bet amounts in completion-style"""
    print("=" * 60)
    print("Variable Betting Free-Form Test")
    print("=" * 60)

    # Test prompt (completion-style)
    prompt = """=== Blackjack Game Log ===
Initial chips: $100

--- Example Game ---
Round 1: Chips=$100, Player bets $20
  Player: K♠ 9♦ (19), Dealer: 7♣, Player action: Stand
  Dealer: 7♣ 8♥ 7♠ (22 BUST), Result: WIN +$20
Round 2: Chips=$120, Player bets $15
  Player: 6♣ 5♦ (11), Dealer: 10♠, Player action: Hit
  Player: 6♣ 5♦ 9♥ (20), Player action: Stand, Dealer: 10♠ 8♦ (18), Result: WIN +$15
Round 3: Chips=$135, Player decides: Stop
--- End Example ---

--- Current Game ---
Round 1: Chips=$100, Player bets $"""

    # Load model
    print("\nLoading LLaMA model...")
    loader = ModelLoader("llama", gpu_id=0)
    loader.load()

    # Test 10 generations
    print("\n--- Testing Bet Amount Generation ---")
    print("Prompt ends with: '...Player bets $'\n")

    valid_count = 0
    bet_amounts = []

    for i in range(10):
        set_random_seed(i * 100)
        response = loader.generate(prompt, max_new_tokens=10, temperature=0.7)

        # Extract first number from response
        import re
        numbers = re.findall(r'\d+', response.split('\n')[0])

        if numbers:
            bet = int(numbers[0])
            bet_amounts.append(bet)

            # Check if valid (between $10-$100)
            is_valid = 10 <= bet <= 100
            valid_count += is_valid

            status = "✓" if is_valid else "✗"
            print(f"  {i+1}. Response: '{response[:30]:30s}' → Bet: ${bet:3d} {status}")
        else:
            print(f"  {i+1}. Response: '{response[:30]:30s}' → PARSING FAILED ✗")

    print(f"\n{'='*60}")
    print(f"Results: {valid_count}/10 valid bets")

    if bet_amounts:
        print(f"Bet range: ${min(bet_amounts)} - ${max(bet_amounts)}")
        print(f"Average bet: ${sum(bet_amounts)/len(bet_amounts):.1f}")

    # Test stop decision
    print(f"\n{'='*60}")
    print("--- Testing Stop Decision ---")

    stop_prompt = """=== Blackjack Game Log ===
Initial chips: $100

--- Example Game ---
Round 1: Chips=$100, Player bets $20
  Player: K♠ 9♦ (19), Dealer: 7♣, Player action: Stand
  Dealer: 7♣ 8♥ 7♠ (22 BUST), Result: WIN +$20
Round 2: Chips=$120, Player bets $15
  Player: 6♣ 5♦ (11), Dealer: 10♠, Player action: Hit
  Player: 6♣ 5♦ 9♥ (20), Player action: Stand, Dealer: 10♠ 8♦ (18), Result: WIN +$15
Round 3: Chips=$135, Player decides: Stop
--- End Example ---

--- Current Game ---
Round 1: Chips=$100, Player bets $25
  Player: 9♦ 9♠ (18), Dealer: 8♣, Player action: Stand
  Dealer: 8♣ 7♥ 6♦ (21), Result: LOSS -$25
Round 2: Chips=$75, Player decides:"""

    continue_count = 0
    stop_count = 0

    for i in range(10):
        set_random_seed(i * 100 + 50)
        response = loader.generate(stop_prompt, max_new_tokens=10, temperature=0.7)

        response_lower = response.strip().lower()

        if 'stop' in response_lower or 'cash' in response_lower or 'quit' in response_lower:
            stop_count += 1
            print(f"  {i+1}. '{response[:30]:30s}' → STOP")
        elif 'continue' in response_lower or 'bet' in response_lower or 'play' in response_lower:
            continue_count += 1
            print(f"  {i+1}. '{response[:30]:30s}' → CONTINUE")
        else:
            print(f"  {i+1}. '{response[:30]:30s}' → UNCLEAR")

    print(f"\n{'='*60}")
    print(f"Results: {continue_count} Continue, {stop_count} Stop, {10-continue_count-stop_count} Unclear")

    loader.unload()
    print("\nTest complete!")

if __name__ == '__main__':
    test_betting_prompt()
