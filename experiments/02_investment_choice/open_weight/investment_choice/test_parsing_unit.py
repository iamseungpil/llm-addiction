#!/usr/bin/env python3
"""
Unit tests for the fixed parsing logic.

Tests the parse_choice_variable() function directly without running full games.
"""

def test_parsing():
    """Test parsing logic with various response formats."""

    # Mock constraint and balance
    constraint = "30"
    balance = 100

    # Create a minimal mock object for the experiment
    class MockExperiment:
        def __init__(self):
            self.bet_constraint = constraint
            self.model_loader = type('obj', (object,), {'config': {}})()

    exp = MockExperiment()

    # Import the parsing method (need to add path first)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from investment_choice.run_experiment import InvestmentChoiceExperiment

    # Create real experiment object to access parse_choice_variable
    real_exp = InvestmentChoiceExperiment('llama', 0, 'variable', str(constraint))

    print("="*70)
    print("PARSING UNIT TESTS")
    print("="*70)

    test_cases = [
        # (response, expected_choice, expected_bet, expected_valid, description)
        ("2", 2, None, False, "Single digit triggers retry (not immediate fallback)"),
        ("2, $15", 2, 15, True, "Ideal format: choice, amount"),
        ("2, 15", 2, 15, True, "Without dollar sign"),
        ("2,$20", 2, 20, True, "No space after comma"),
        ("2 $10", 2, 10, True, "Space instead of comma"),
        ("3", 3, None, False, "Single digit 3 triggers retry"),
        ("1", 1, None, False, "Single digit 1 triggers retry"),
        ("4, $25", 4, 25, True, "Option 4 with amount"),
        ("Option 2, $18", 2, 18, True, "With 'Option' prefix"),
        ("I choose Option 3, $12", 3, 12, True, "Verbose response"),
        ("2\nYou got $0", 2, None, False, "Continuation text after choice (triggers retry)"),
        ("3, $50", 3, 30, True, "Amount > constraint should be capped at 30"),
        ("2, $0", 2, 1, True, "$0 becomes $1 (min bet)"),
    ]

    print(f"\nConstraint: c{constraint}, Balance: ${balance}")
    print(f"Expected fallback: $3 (10% of {constraint})")
    print(f"\n{'Response':<30} {'Expected':<20} {'Got':<20} {'Status':<10}")
    print("-"*80)

    passed = 0
    failed = 0

    for response, exp_choice, exp_bet, exp_valid, description in test_cases:
        parsed = real_exp.parse_choice_variable(response, balance)
        got_choice = parsed.get('choice')
        got_bet = parsed.get('bet_amount')
        got_valid = parsed.get('valid')
        reason = parsed.get('reason', '')

        # Check if result matches expectation
        if got_choice == exp_choice and got_bet == exp_bet and got_valid == exp_valid:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1

        # Truncate response for display
        resp_display = response[:28] + ".." if len(response) > 30 else response
        bet_str = f"${exp_bet}" if exp_bet is not None else "None"
        valid_str = "✓" if exp_valid else "✗"
        exp_display = f"c{exp_choice} {bet_str} [{valid_str}]"

        bet_str_got = f"${got_bet}" if got_bet is not None else "None"
        valid_str_got = "✓" if got_valid else "✗"
        got_display = f"c{got_choice} {bet_str_got} [{valid_str_got}]"

        print(f"{resp_display:<30} {exp_display:<20} {got_display:<20} {status:<10}")

        # Show reason if failed
        if status == "✗ FAIL":
            print(f"  → {description}")
            print(f"  → Reason: {reason}")

    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*70)

    if failed == 0:
        print("✅ All tests passed! Parsing fix is working correctly.")
    else:
        print(f"⚠️  {failed} test(s) failed. Review parsing logic.")

    return failed == 0


if __name__ == '__main__':
    success = test_parsing()
    exit(0 if success else 1)
