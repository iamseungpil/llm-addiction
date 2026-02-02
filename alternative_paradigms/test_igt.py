#!/usr/bin/env python3
"""
Quick test script for IGT implementation

Tests:
1. Game logic (deck rewards/losses)
2. Prompt generation
3. Response parsing
4. Result saving
"""

import sys
import random
import numpy as np
from pathlib import Path

# Directly import game_logic module to avoid torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location("igt_game_logic", str(Path(__file__).parent / "src" / "igt" / "game_logic.py"))
igt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(igt_module)
IowaGamblingTask = igt_module.IowaGamblingTask


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def parse_choice(response: str, valid_choices: list) -> str:
    """Parse choice from response"""
    response = response.upper().strip()

    # First check for explicit patterns
    import re
    for choice in valid_choices:
        # Pattern: "Choice: X", "Deck X", "Select X", etc.
        pattern = rf'(?:choice|deck|select|option)[:\s]+{choice}\b'
        if re.search(pattern, response, re.IGNORECASE):
            return choice

    # Then check for standalone letter
    for choice in valid_choices:
        # Pattern: standalone letter (with word boundary)
        pattern = rf'\b{choice}\b'
        if re.search(pattern, response[:50]):
            return choice

    return None


def test_game_logic():
    """Test IGT game logic"""
    print("=" * 70)
    print("TEST 1: Game Logic")
    print("=" * 70)

    set_random_seed(42)
    game = IowaGamblingTask(initial_balance=2000, num_trials=20)

    print(f"Initial balance: ${game.balance}")
    print(f"\nPlaying 20 trials with predetermined sequence:")
    print("Deck A (x5), Deck B (x5), Deck C (x5), Deck D (x5)\n")

    deck_sequence = ['A'] * 5 + ['B'] * 5 + ['C'] * 5 + ['D'] * 5

    for i, deck in enumerate(deck_sequence):
        result = game.play_trial(deck)
        net = result['net']
        print(f"Trial {i+1}: Deck {deck} → Reward ${result['reward']}, Loss ${result['loss']}, Net ${net:+d}, Balance ${result['balance']}")

    print(f"\nFinal balance: ${game.balance}")
    print(f"Net score: {game.get_net_score()}")
    print(f"Deck preferences: {game.get_deck_preference_summary()}")

    # Check expected behavior
    assert game.trial == 20, "Should complete 20 trials"
    assert game.deck_counts == {'A': 5, 'B': 5, 'C': 5, 'D': 5}, "Deck counts mismatch"
    print("\n✅ Game logic test PASSED")


def test_prompt_generation():
    """Test prompt generation"""
    print("\n" + "=" * 70)
    print("TEST 2: Prompt Generation")
    print("=" * 70)

    set_random_seed(42)
    game = IowaGamblingTask(initial_balance=2000, num_trials=100)

    # Play a few trials
    game.play_trial('A')
    game.play_trial('B')
    game.play_trial('C')

    # Check history text
    history_text = game.get_history_text(max_display=10)
    print("\nGenerated history text:")
    print(history_text)

    assert "Trial 1" in history_text, "Should show trial history"
    assert "Deck A" in history_text, "Should show deck choices"

    # Check deck summary
    summary_text = game.get_deck_summary()
    print("\nGenerated deck summary:")
    print(summary_text)

    assert "Deck A" in summary_text, "Should show deck A"
    assert "Selected 1 time" in summary_text, "Should show selection count"

    print("\n✅ Prompt generation test PASSED")


def test_response_parsing():
    """Test response parsing"""
    print("\n" + "=" * 70)
    print("TEST 3: Response Parsing")
    print("=" * 70)

    test_cases = [
        ("I choose deck A", "A"),
        ("Deck B", "B"),
        ("I'll select C", "C"),
        ("Choice: D", "D"),
        ("Let me try deck A this time", "A"),
        ("DECK C", "C"),
        ("Random text without choice", None),
    ]

    print("\nTesting response parsing:")
    for response, expected in test_cases:
        result = parse_choice(response, ['A', 'B', 'C', 'D'])
        status = "✓" if result == expected else "✗"
        print(f"{status} '{response[:30]}...' → {result} (expected: {expected})")

        if result != expected:
            print(f"  ERROR: Expected {expected}, got {result}")
            raise AssertionError(f"Parsing failed for: {response}")

    print("\n✅ Response parsing test PASSED")


def test_learning_curve():
    """Test learning curve calculation"""
    print("\n" + "=" * 70)
    print("TEST 4: Learning Curve")
    print("=" * 70)

    set_random_seed(42)
    game = IowaGamblingTask(initial_balance=2000, num_trials=100)

    # Simulate learning: start with A/B, shift to C/D
    for i in range(100):
        if i < 40:
            # Early trials: prefer disadvantageous
            deck = 'A' if i % 2 == 0 else 'B'
        else:
            # Later trials: prefer advantageous
            deck = 'C' if i % 2 == 0 else 'D'

        game.play_trial(deck)

    learning_curve = game.get_learning_curve(block_size=20)

    print("\nLearning curve (5 blocks of 20 trials):")
    print(f"{'Block':<8} {'Trials':<12} {'Adv %':<10} {'Disadv %':<12} {'Net Score':<10}")
    print("-" * 60)

    for block in learning_curve:
        print(f"{block['block']:<8} {block['trials']:<12} {block['advantageous_pct']:<10.1f} {block['disadvantageous_pct']:<12.1f} {block['net_score']:<10}")

    # Check expected learning pattern
    assert learning_curve[0]['net_score'] < 0, "Block 1 should prefer disadvantageous"
    assert learning_curve[-1]['net_score'] > 0, "Block 5 should prefer advantageous"

    print("\n✅ Learning curve test PASSED")


def test_final_result():
    """Test final result structure"""
    print("\n" + "=" * 70)
    print("TEST 5: Final Result Structure")
    print("=" * 70)

    set_random_seed(42)
    game = IowaGamblingTask(initial_balance=2000, num_trials=20)

    # Play 20 trials
    for i in range(20):
        deck = ['A', 'B', 'C', 'D'][i % 4]
        game.play_trial(deck)

    result = game.get_game_result()

    print("\nFinal result keys:")
    for key in result.keys():
        if key != 'history' and key != 'learning_curve':
            print(f"  - {key}: {result[key]}")

    # Check required fields
    required_fields = [
        'trials_completed', 'final_balance', 'profit_loss',
        'total_won', 'total_lost', 'deck_counts', 'deck_preferences',
        'net_score', 'learning_curve', 'history'
    ]

    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    print("\n✅ Final result structure test PASSED")


def main():
    """Run all tests"""
    print("\n" + "🧪 IGT IMPLEMENTATION VALIDATION" + "\n")

    try:
        test_game_logic()
        test_prompt_generation()
        test_response_parsing()
        test_learning_curve()
        test_final_result()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nIGT implementation is working correctly.")
        print("Ready for full experiment execution.")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
