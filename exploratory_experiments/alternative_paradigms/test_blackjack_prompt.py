#!/usr/bin/env python3
"""
Test Blackjack prompt generation to verify no biasing labels.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from blackjack.game_logic import BlackjackGame
from blackjack.run_experiment import BlackjackExperiment

def test_prompt_generation():
    """Test that prompts don't contain biasing labels"""

    print("=" * 80)
    print("Testing Blackjack Prompt Generation (No Biasing Labels)")
    print("=" * 80)

    # Create experiment instance
    exp = BlackjackExperiment(model_name='llama', gpu_id=0, bet_type='variable')

    # Create game instance
    game = BlackjackGame(
        initial_chips=100,
        min_bet=10,
        max_bet=100,
        bet_type='variable'
    )

    # Test 1: Initial betting phase prompt
    print("\n" + "=" * 80)
    print("TEST 1: Initial Betting Phase (Variable Betting)")
    print("=" * 80)
    prompt = exp.build_prompt(game, components='BASE', is_betting_phase=True)
    print(prompt)

    # Check for biasing labels
    biasing_words = ['safe', 'aggressive', 'cautious', 'risky', 'conservative', 'bold']
    found_bias = []
    for word in biasing_words:
        if word.lower() in prompt.lower():
            found_bias.append(word)

    if found_bias:
        print(f"\n⚠️  WARNING: Found biasing labels: {found_bias}")
    else:
        print(f"\n✅ PASS: No biasing labels found")

    # Test 2: Fixed betting
    print("\n" + "=" * 80)
    print("TEST 2: Fixed Betting Mode")
    print("=" * 80)
    exp_fixed = BlackjackExperiment(model_name='llama', gpu_id=0, bet_type='fixed')
    game_fixed = BlackjackGame(
        initial_chips=100,
        min_bet=10,
        max_bet=20,
        bet_type='fixed'
    )
    prompt_fixed = exp_fixed.build_prompt(game_fixed, components='BASE', is_betting_phase=True)
    print(prompt_fixed)

    found_bias_fixed = []
    for word in biasing_words:
        if word.lower() in prompt_fixed.lower():
            found_bias_fixed.append(word)

    if found_bias_fixed:
        print(f"\n⚠️  WARNING: Found biasing labels: {found_bias_fixed}")
    else:
        print(f"\n✅ PASS: No biasing labels found")

    # Test 3: Low chips scenario
    print("\n" + "=" * 80)
    print("TEST 3: Low Chips Scenario (Chips=$30)")
    print("=" * 80)
    game_low = BlackjackGame(
        initial_chips=100,
        min_bet=10,
        max_bet=100,
        bet_type='variable'
    )
    game_low.chips = 30
    prompt_low = exp.build_prompt(game_low, components='BASE', is_betting_phase=True)
    print(prompt_low)

    found_bias_low = []
    for word in biasing_words:
        if word.lower() in prompt_low.lower():
            found_bias_low.append(word)

    if found_bias_low:
        print(f"\n⚠️  WARNING: Found biasing labels: {found_bias_low}")
    else:
        print(f"\n✅ PASS: No biasing labels found")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_pass = not found_bias and not found_bias_fixed and not found_bias_low

    if all_pass:
        print("✅ All tests passed - No biasing labels found")
        print("\nPrompt Format:")
        print("  - Option 1: Stop playing, keep chips, game ends")
        print("  - Option 2/3/4: Bet amounts with neutral descriptions")
        print("  - No labels like 'safe', 'aggressive', 'cautious', etc.")
    else:
        print("❌ Some tests failed - Biasing labels detected")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(test_prompt_generation())
