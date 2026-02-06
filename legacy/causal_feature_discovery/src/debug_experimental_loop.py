#!/usr/bin/env python3
"""
Debug the exact experimental loop logic to identify the parsing disconnect
"""

def parse_response(response: str) -> dict:
    """Parse betting response - EXACT copy from experiment file"""
    import re
    
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

def simulate_exact_experimental_logic():
    """Simulate the EXACT experimental loop logic"""
    print("EXACT EXPERIMENTAL LOOP SIMULATION:")
    print("=" * 60)
    
    # Mock responses that should parse successfully
    test_responses = [" $5\n", " 15\n", " 10\n", "$25", "I'll bet $20"] * 6  # 30 responses
    
    # Exact loop logic from experiment
    bets = []
    stops = 0
    invalids = 0
    n_trials = 30
    
    for trial in range(n_trials):
        try:
            # Simulate response generation (this is where real failures might occur)
            if trial < len(test_responses):
                response = test_responses[trial]
            else:
                response = " $10\n"  # Default response
                
            print(f"Trial {trial}: response='{response.strip()}'")
            
            parsed = parse_response(response)
            print(f"  Parsed: {parsed}")
            
            # 🔍 THIS IS THE CRITICAL CHECK
            if not parsed.get('valid', True):
                print(f"  ❌ INVALID: Adding to invalids count")
                invalids += 1
                continue
            
            print(f"  ✅ VALID: Adding bet={parsed['bet']} to bets list")
            bets.append(parsed['bet'])
            if parsed['action'] == 'stop':
                stops += 1
                
        except Exception as e:
            print(f"  💥 EXCEPTION in trial {trial}: {e}")
            invalids += 1
            continue
    
    # Final calculation
    valid_trials = len(bets)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(f"valid_trials = len(bets) = {valid_trials}")
    print(f"invalid_trials = {invalids}")
    print(f"stops = {stops}")
    print(f"bets = {bets[:5]}... (first 5)")
    
    # Check for the reported issue
    if valid_trials == 0 and invalids > 0:
        print("\n🚨 BUG REPRODUCED: All trials marked invalid!")
        return False
    elif valid_trials > 0:
        print("\n✅ Parsing working correctly")
        return True
    else:
        print("\n⚠️  No trials processed at all")
        return None

def test_edge_case_validity_check():
    """Test the specific validity check logic"""
    print("\nTESTING VALIDITY CHECK LOGIC:")
    print("=" * 60)
    
    test_cases = [
        {'bet': 5, 'valid': True, 'action': 'bet'},
        {'bet': 0, 'valid': True, 'action': 'stop'},
        {'bet': 10, 'valid': False, 'action': 'bet'},  # This should be invalid
        {'bet': 15, 'action': 'bet'},  # Missing 'valid' key
        {'bet': 20, 'valid': None, 'action': 'bet'},  # None valid
    ]
    
    for i, parsed in enumerate(test_cases):
        print(f"Case {i+1}: {parsed}")
        
        # This is the exact check from the experiment
        is_invalid = not parsed.get('valid', True)
        print(f"  not parsed.get('valid', True) = {is_invalid}")
        
        if is_invalid:
            print("  ❌ Would be marked INVALID")
        else:
            print("  ✅ Would be marked VALID")
        print()

def test_response_generation_simulation():
    """Test what might go wrong in response generation"""
    print("\nTESTING RESPONSE GENERATION ISSUES:")
    print("=" * 60)
    
    # Simulate potential issues in generate_with_patching
    problematic_responses = [
        None,  # Could generate_with_patching return None?
        "",    # Empty response
        " ",   # Whitespace only
        "CUDA out of memory error occurred",  # Error message
        "Token limit exceeded",  # Another error
        "The model failed to generate a valid response",
    ]
    
    for i, response in enumerate(problematic_responses):
        print(f"Problematic response {i+1}: {repr(response)}")
        
        try:
            if response is None:
                print("  💥 None response would cause TypeError")
                continue
                
            parsed = parse_response(response)
            print(f"  Parsed: {parsed}")
            
            is_valid = parsed.get('valid', True)
            print(f"  Would be marked: {'VALID' if is_valid else 'INVALID'}")
            
        except Exception as e:
            print(f"  💥 Exception: {e}")
        print()

if __name__ == "__main__":
    success = simulate_exact_experimental_logic()
    test_edge_case_validity_check()
    test_response_generation_simulation()
    
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY:")
    if success:
        print("✅ The parse_response function itself works correctly")
        print("🔍 The issue is likely in:")
        print("   1. generate_with_patching() returning invalid responses")
        print("   2. Exception handling swallowing valid responses") 
        print("   3. Model generation failures due to GPU/memory issues")
    else:
        print("❌ Found a bug in the experimental loop logic")