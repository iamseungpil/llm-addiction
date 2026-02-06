#!/usr/bin/env python3
"""
Test script to isolate and debug the parse_response function
"""

import sys
import re

def parse_response(response: str) -> dict:
    """Parse betting response - EXACT copy from experiment file"""
    # Multiple patterns to catch different formats
    patterns = [
        r'\$(\d+)(?:\.|,|!|\s|$)',  # $25
        r'(\d+)\s*dollars?',        # 25 dollars
        r'bet\s*(\d+)',            # bet 25
        r'(\d+)(?=\s*$)',          # 25 at end
    ]
    
    amounts = []
    for pattern in patterns:
        amounts.extend(re.findall(pattern, response, re.IGNORECASE))
    
    if amounts:
        bet = int(amounts[-1])  # Take last amount found
        return {
            'bet': bet,
            'action': 'stop' if bet == 0 else 'bet',
            'valid': True,
            'response': response
        }
    
    # Check for stop/quit words
    stop_words = ['stop', 'quit', 'fold', 'pass', 'done']
    if any(word in response.lower() for word in stop_words):
        return {
            'bet': 0,
            'action': 'stop', 
            'valid': True,
            'response': response
        }
    
    # Invalid response
    return {
        'bet': 10,  # Default fallback
        'action': 'bet',
        'valid': False,
        'response': response
    }

def test_individual_responses():
    """Test the exact responses mentioned in the issue"""
    test_responses = [
        " $5\n",
        " 15\n", 
        " 10\n",
        "$25",
        "I'll bet $15",
        "bet 20",
        "25 dollars",
        "stop",
        "I choose to quit",
        "Game simulation goes here...\nFinal bet: $30",
        "",
        "nonsense response",
    ]
    
    print("INDIVIDUAL PARSING TESTS:")
    print("=" * 50)
    
    for i, response in enumerate(test_responses):
        result = parse_response(response)
        print(f"Test {i+1}: '{response.strip()}'")
        print(f"  Result: bet={result['bet']}, valid={result['valid']}, action={result['action']}")
        print()

def simulate_experimental_loop():
    """Simulate how parse_response is called in the experimental loop"""
    print("EXPERIMENTAL LOOP SIMULATION:")
    print("=" * 50)
    
    # Test conditions like in the real experiment
    test_responses = [" $5\n", " 15\n", " 10\n"] * 10  # 30 responses like n_trials=30
    
    valid_trials = 0
    invalid_trials = 0
    bets = []
    
    for trial, response in enumerate(test_responses):
        try:
            parsed = parse_response(response)
            
            # This is the exact logic from the experiment
            if not parsed.get('valid', True):
                invalid_trials += 1
                continue
            
            bets.append(parsed['bet'])
            valid_trials += 1
            
        except Exception as e:
            print(f"  Error in trial {trial}: {e}")
            invalid_trials += 1
            continue
    
    print(f"Final counts: valid_trials={valid_trials}, invalid_trials={invalid_trials}")
    print(f"Bets collected: {bets[:10]}...")  # Show first 10
    
    # Check if this matches the reported issue
    if valid_trials == 0 and invalid_trials > 0:
        print("🚨 FOUND THE BUG: All trials marked invalid!")
    elif valid_trials == len(test_responses):
        print("✅ All trials parsed correctly")
    else:
        print(f"⚠️  Mixed results: {valid_trials} valid, {invalid_trials} invalid")

def debug_edge_cases():
    """Test edge cases that might cause silent failures"""
    print("EDGE CASE DEBUGGING:")
    print("=" * 50)
    
    edge_cases = [
        "I'll bet $5 on this round.",  # Multiple potential matches
        "My goal is $200 but I'll bet $5",  # Should take LAST amount ($5)
        "$200 $150 $100 $5",  # Multiple amounts
        "Round 1: $10, Round 2: $15, Round 3: $5",  # Should take $5
        "$0",  # Zero bet
        "I bet 0 dollars",  # Zero in words
        "I'll bet $5.50",  # Decimal (should fail current regex)
        "I'll bet $5.00",  # Decimal with .00
        "bet: $5",  # Colon separator
        "Amount: 5",  # No dollar sign
    ]
    
    for case in edge_cases:
        result = parse_response(case)
        print(f"'{case}' → bet={result['bet']}, valid={result['valid']}")

if __name__ == "__main__":
    test_individual_responses()
    simulate_experimental_loop()
    debug_edge_cases()