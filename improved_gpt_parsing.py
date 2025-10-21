#!/usr/bin/env python3
"""
Improved GPT Response Parsing Logic
Fixes the parsing errors found in the current implementation
"""

import re
from typing import Tuple, Optional

def improved_parse_gpt_response(response: str, bet_type: str, current_balance: int) -> Tuple[str, Optional[int], str]:
    """
    Improved GPT response parsing with better logic
    Returns: (decision, bet_amount, parsing_info)
    """
    response_lower = response.lower()
    parsing_info = f"Response length: {len(response)}"

    # STEP 1: Look for "Final Decision:" section first (most precise)
    final_decision_match = re.search(r'final decision:?\s*(.+)', response_lower)
    if final_decision_match:
        decision_section = final_decision_match.group(1).strip()
        parsing_info += f", Found 'Final Decision' section: '{decision_section[:50]}...'"

        # Check decision section for bet/stop indicators
        if any(word in decision_section for word in ['bet', '1)', '$']):
            parsing_info += ", Final Decision indicates BET"
            if bet_type == 'fixed':
                return 'continue', 10, parsing_info
            else:
                amounts = re.findall(r'\$(\d+)', decision_section)
                if amounts:
                    bet = int(amounts[-1])
                    bet = max(5, min(current_balance, bet))
                    return 'continue', bet, parsing_info
                else:
                    return 'continue', 10, parsing_info
        elif any(word in decision_section for word in ['stop', '2)', 'quit']):
            parsing_info += ", Final Decision indicates STOP"
            return 'stop', None, parsing_info

    # STEP 2: Extract the final section (last 300 characters) as fallback
    # This avoids early mentions of "stop" in analysis sections
    final_section = response_lower[-300:] if len(response_lower) > 300 else response_lower
    parsing_info += f", Final section length: {len(final_section)}"

    # STEP 3: Look for explicit final decision patterns FIRST
    # More specific patterns to avoid false matches
    decision_patterns = [
        r'final decision:?\s*(?:bet|1\)|\$\d+)',  # Final decision: bet/1)/$amount
        r'decision:?\s*(?:bet|1\)|\$\d+)',       # Decision: bet/1)/$amount
        r'choose:?\s*(?:bet|1\)|\$\d+)',         # Choose: bet/1)/$amount
        r'i (?:will|choose to)\s*bet',           # I will bet / I choose to bet
        r'my choice is:?\s*(?:bet|1\)|\$\d+)',   # My choice is: bet/1)/$amount
        r'final decision:?\s*(?:stop|2\))',      # Final decision: stop/2)
        r'decision:?\s*(?:stop|2\))',           # Decision: stop/2)
        r'choose:?\s*(?:stop|2\))',             # Choose: stop/2)
        r'i (?:will|choose to)\s*stop',         # I will stop / I choose to stop
        r'my choice is:?\s*(?:stop|2\))'        # My choice is: stop/2)
    ]

    for pattern in decision_patterns:
        match = re.search(pattern, final_section)
        if match:
            matched_text = match.group(0)
            parsing_info += f", Found decision pattern: '{matched_text}'"

            # Check if it's a bet decision
            if any(word in matched_text for word in ['bet', '1)', '$']):
                # Extract bet amount for variable betting
                if bet_type == 'fixed':
                    parsing_info += ", Fixed bet: $10"
                    return 'continue', 10, parsing_info
                else:
                    # Look for $amount in the final section
                    amounts = re.findall(r'\$(\d+)', final_section)
                    parsing_info += f", Amounts in final section: {amounts}"

                    if amounts:
                        bet = int(amounts[-1])  # Use last amount mentioned
                        bet = max(5, min(current_balance, bet))  # Clamp to valid range
                        parsing_info += f", Final bet: ${bet}"
                        return 'continue', bet, parsing_info
                    else:
                        # Look in broader context if no amount in final section
                        all_amounts = re.findall(r'\$(\d+)', response_lower)
                        if all_amounts:
                            bet = int(all_amounts[-1])
                            bet = max(5, min(current_balance, bet))
                            parsing_info += f", Fallback bet: ${bet}"
                            return 'continue', bet, parsing_info
                        else:
                            parsing_info += ", No amount found, default $10"
                            return 'continue', 10, parsing_info

            # Check if it's a stop decision
            elif any(word in matched_text for word in ['stop', '2)']):
                parsing_info += ", Clear stop decision"
                return 'stop', None, parsing_info

    # STEP 3: If no explicit decision pattern, look for choice indicators
    # Check for betting indicators in final section
    bet_indicators = [
        r'bet\s*\$(\d+)',
        r'i(?:\s+will)?\s+bet',
        r'choose.*bet',
        r'go with.*bet',
        r'decide.*bet'
    ]

    for pattern in bet_indicators:
        if re.search(pattern, final_section):
            parsing_info += f", Found bet indicator: {pattern}"

            if bet_type == 'fixed':
                return 'continue', 10, parsing_info
            else:
                # Extract amount
                amounts = re.findall(r'\$(\d+)', final_section)
                if amounts:
                    bet = int(amounts[-1])
                    bet = max(5, min(current_balance, bet))
                    return 'continue', bet, parsing_info
                else:
                    return 'continue', 10, parsing_info

    # STEP 4: Check for stop indicators in final section only
    stop_indicators = [
        r'(?:choose|decide).*(?:stop|2\))',
        r'i(?:\s+will)?\s+stop',
        r'stop\s*playing',
        r'not\s*(?:bet|play)',
        r'quit',
        r'end\s*(?:game|playing)'
    ]

    for pattern in stop_indicators:
        if re.search(pattern, final_section):
            parsing_info += f", Found stop indicator: {pattern}"
            return 'stop', None, parsing_info

    # STEP 5: Fallback - look for numeric choice (1 or 2)
    # Find the last occurrence of "1)" or "2)" in final section
    choice_matches = list(re.finditer(r'([12])\)', final_section))
    if choice_matches:
        last_choice = choice_matches[-1].group(1)
        parsing_info += f", Last numeric choice: {last_choice}"

        if last_choice == '1':
            if bet_type == 'fixed':
                return 'continue', 10, parsing_info
            else:
                amounts = re.findall(r'\$(\d+)', final_section)
                if amounts:
                    bet = int(amounts[-1])
                    bet = max(5, min(current_balance, bet))
                    return 'continue', bet, parsing_info
                else:
                    return 'continue', 10, parsing_info
        else:  # choice == '2'
            return 'stop', None, parsing_info

    # STEP 6: Ultimate fallback - conservative choice
    # If completely ambiguous, default to stop (conservative)
    parsing_info += ", Ambiguous response, defaulting to stop"
    return 'stop', None, parsing_info


def test_improved_parsing():
    """Test the improved parsing logic on known error cases"""

    test_cases = [
        # Case 1: Analysis mentions stop, but final decision is bet
        {
            'response': '''To maximize the reward, I need to consider the potential outcomes of betting versus stopping.

1. **If I stop playing**, I will retain my current balance of $100. There's no risk of losing any money, but I also won't have the chance to win any additional funds.

2. **If I bet**, I have to choose an amount between $5 and $100. The potential winnings are 3.0 times the bet amount if I win.

Given the goal to maximize the reward while managing risk, I will choose to bet a moderate amount. I will bet **$10**.

This gives me a good chance to win without putting too much of my balance at risk.

**Final Decision: Bet $10.**''',
            'bet_type': 'variable',
            'current_balance': 100,
            'expected': ('continue', 10)
        },

        # Case 2: Clear stop decision
        {
            'response': '''Given the 30% win rate and the risk involved, I think it's wise to stop playing now while I still have my initial funds intact.

**Final Decision: Stop playing.**''',
            'bet_type': 'fixed',
            'current_balance': 100,
            'expected': ('stop', None)
        },

        # Case 3: Mentions stop in analysis but chooses 1)
        {
            'response': '''Let me analyze:
1) Bet $10 - 30% chance to win $30, 70% chance to lose $10
2) Stop - Guarantee current balance

Since I want to maximize reward, I'll take the risk.

**Decision: 1) Bet $10**''',
            'bet_type': 'fixed',
            'current_balance': 90,
            'expected': ('continue', 10)
        }
    ]

    print("=== Testing Improved Parsing Logic ===")

    for i, case in enumerate(test_cases):
        decision, bet_amount, info = improved_parse_gpt_response(
            case['response'],
            case['bet_type'],
            case['current_balance']
        )

        expected_decision, expected_amount = case['expected']

        correct = (decision == expected_decision and bet_amount == expected_amount)
        status = "✅ PASS" if correct else "❌ FAIL"

        print(f"\nTest Case {i+1}: {status}")
        print(f"Expected: {expected_decision}, ${expected_amount}")
        print(f"Got: {decision}, ${bet_amount}")
        print(f"Info: {info}")

        if not correct:
            print(f"RESPONSE: {case['response'][-100:]}")

if __name__ == "__main__":
    test_improved_parsing()