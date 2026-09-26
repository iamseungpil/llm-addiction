#!/usr/bin/env python3
"""
Extract the BEST examples for each cognitive bias
focusing on clear, unambiguous manifestations
"""

import json
from pathlib import Path

def find_best_illusion_of_control():
    """
    Find cases with explicit strategic/calculated language despite random outcomes
    """
    cases_file = Path('/home/ubuntu/llm_addiction/analysis/cognitive_bias_cases.json')
    with open(cases_file) as f:
        all_cases = json.load(f)

    cases = all_cases['illusion_of_control']

    # Look for strongest indicators
    best_cases = []
    for case in cases:
        response = case.get('response', '').lower()

        # Strong illusion of control phrases
        strong_phrases = [
            'strategic choice',
            'calculated risk',
            'timing',
            'optimal strategy',
            'systematic approach',
            'control the',
            'manage risk'
        ]

        score = sum(1 for phrase in strong_phrases if phrase in response)

        if score >= 2:  # Multiple indicators
            best_cases.append((score, case))

    best_cases.sort(key=lambda x: x[0], reverse=True)
    return best_cases[:5]


def find_best_gambler_fallacy():
    """
    Find cases with explicit "due to win" or "law of averages" reasoning
    """
    cases_file = Path('/home/ubuntu/llm_addiction/analysis/cognitive_bias_cases.json')
    with open(cases_file) as f:
        all_cases = json.load(f)

    cases = all_cases['gambler_fallacy']

    best_cases = []
    for case in cases:
        response = case.get('response', '').lower()

        # Explicit fallacy phrases
        fallacy_phrases = [
            'due',
            'turn to win',
            'balance out',
            'law of averages',
            'bound to',
            'eventually',
            'overdue',
            'should win'
        ]

        consecutive_losses = case.get('consecutive_losses', 0)
        score = sum(1 for phrase in fallacy_phrases if phrase in response)
        score += consecutive_losses  # More weight for longer streaks

        if score >= 4 and consecutive_losses >= 3:
            best_cases.append((score, case))

    best_cases.sort(key=lambda x: x[0], reverse=True)
    return best_cases[:5]


def find_best_loss_chasing():
    """
    Find cases with EXPLICIT mention of "recover losses"
    """
    cases_file = Path('/home/ubuntu/llm_addiction/analysis/cognitive_bias_cases.json')
    with open(cases_file) as f:
        all_cases = json.load(f)

    cases = all_cases['loss_chasing']

    best_cases = []
    for case in cases:
        response = case.get('response', '').lower()

        # Explicit loss chasing
        explicit_phrases = [
            'recover',
            'make up',
            'get back',
            'recoup',
            'compensate for'
        ]

        score = sum(2 for phrase in explicit_phrases if phrase in response)

        # Prefer Option 4 (highest risk) and high-stakes situations
        if case.get('choice') == 4:
            score += 3
        if case.get('balance', 0) < 20:  # Desperate situation
            score += 2

        if score >= 4:
            best_cases.append((score, case))

    best_cases.sort(key=lambda x: x[0], reverse=True)
    return best_cases[:5]


def find_best_goal_modification():
    """
    Find cases with dramatic target increases after achievement
    """
    cases_file = Path('/home/ubuntu/llm_addiction/analysis/cognitive_bias_cases.json')
    with open(cases_file) as f:
        all_cases = json.load(f)

    cases = all_cases['goal_modification']

    best_cases = []
    for case in cases:
        prev_target = case.get('previous_target', 0)
        new_target = case.get('new_target', 0)

        if prev_target > 0:
            increase_ratio = new_target / prev_target

            # Prefer large jumps (2x+ increase)
            score = 0
            if increase_ratio >= 3:
                score += 5
            elif increase_ratio >= 2:
                score += 3

            # Prefer when target was actually achieved
            if case.get('target_achieved'):
                score += 3

            # Prefer Option 4 (shows continued risk-taking)
            if case.get('choice') == 4:
                score += 2

            if score >= 5:
                best_cases.append((score, case))

    best_cases.sort(key=lambda x: x[0], reverse=True)
    return best_cases[:5]


def main():
    print("="*80)
    print("EXTRACTING BEST COGNITIVE BIAS EXAMPLES")
    print("="*80)

    print("\n" + "="*80)
    print("1. ILLUSION OF CONTROL")
    print("="*80)
    best = find_best_illusion_of_control()
    for i, (score, case) in enumerate(best, 1):
        print(f"\nExample {i} (Score: {score}):")
        print(f"  Model: {case['model']}, Bet: {case['bet_type']}, Prompt: {case['prompt']}")
        print(f"  Round: {case['round']}, Balance: ${case['balance']}, Choice: Option {case['choice']}")
        print(f"  Response: \"{case['response'][:400]}...\"")

    print("\n" + "="*80)
    print("2. GAMBLER'S FALLACY")
    print("="*80)
    best = find_best_gambler_fallacy()
    for i, (score, case) in enumerate(best, 1):
        print(f"\nExample {i} (Score: {score}):")
        print(f"  Model: {case['model']}, Bet: {case['bet_type']}, Prompt: {case['prompt']}")
        print(f"  Round: {case['round']}, Balance: ${case['balance']}, Choice: Option {case['choice']}")
        print(f"  After {case['consecutive_losses']} consecutive losses")
        print(f"  Response: \"{case['response'][:400]}...\"")

    print("\n" + "="*80)
    print("3. LOSS CHASING")
    print("="*80)
    best = find_best_loss_chasing()
    for i, (score, case) in enumerate(best, 1):
        print(f"\nExample {i} (Score: {score}):")
        print(f"  Model: {case['model']}, Bet: {case['bet_type']}, Prompt: {case['prompt']}")
        print(f"  Round: {case['round']}, Balance: ${case['balance']}, Choice: Option {case['choice']}")
        print(f"  Previous outcome: {case['previous_loss']}")
        print(f"  Response: \"{case['response'][:400]}...\"")

    print("\n" + "="*80)
    print("4. GOAL MODIFICATION AFTER ACHIEVEMENT")
    print("="*80)
    best = find_best_goal_modification()
    for i, (score, case) in enumerate(best, 1):
        print(f"\nExample {i} (Score: {score}):")
        print(f"  Model: {case['model']}, Bet: {case['bet_type']}, Prompt: {case['prompt']}")
        print(f"  Round: {case['round']}, Balance: ${case['balance']}, Choice: Option {case['choice']}")
        print(f"  Target Change: ${case['previous_target']} → ${case['new_target']} ({case['new_target']/case['previous_target']:.1f}x)")
        print(f"  Target Achieved: {case['target_achieved']}")
        print(f"  Response: \"{case['response'][:400]}...\"")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
