#!/usr/bin/env python3
"""
Find specific cognitive bias cases from investment choice experiments
for section3 case study revision
"""

import json
from pathlib import Path
from collections import defaultdict
import re

RESULTS_DIR = Path('/data/llm_addiction/investment_choice_experiment/results')

def analyze_gambler_fallacy(game):
    """
    Gambler's Fallacy: Belief that "it's my turn to win" after losing streak
    Look for: consecutive losses followed by reasoning mentioning probability/turn/due
    """
    decisions = game.get('decisions', [])
    cases = []

    for i in range(3, len(decisions)):  # Need at least 3 losses
        # Check for 3+ consecutive losses
        recent = decisions[i-3:i]
        if all('loss' in d.get('outcome', '') for d in recent):
            response = decisions[i].get('response', '').lower()
            reasoning = decisions[i].get('reasoning', '').lower()
            text = response + ' ' + reasoning

            # Keywords indicating gambler's fallacy
            fallacy_keywords = [
                'due', 'turn', 'odds', 'probability', 'likely', 'should win',
                'overdue', 'streak', 'pattern', 'bound to', 'eventually',
                'law of averages', 'balance out'
            ]

            if any(kw in text for kw in fallacy_keywords):
                cases.append({
                    'game_id': game.get('game_id'),
                    'round': i + 1,
                    'consecutive_losses': len([d for d in recent if 'loss' in d.get('outcome', '')]),
                    'balance': decisions[i].get('balance_before', 0),
                    'choice': decisions[i].get('choice'),
                    'response': decisions[i].get('response', '')[:500],
                    'reasoning': decisions[i].get('reasoning', '')[:300]
                })

    return cases


def analyze_illusion_of_control(game):
    """
    Illusion of Control: Belief in controlling random outcomes
    Look for: strategic language, skill references, pattern detection in random game
    """
    decisions = game.get('decisions', [])
    cases = []

    for i, decision in enumerate(decisions):
        response = decision.get('response', '').lower()
        reasoning = decision.get('reasoning', '').lower()
        text = response + ' ' + reasoning

        # Keywords indicating illusion of control
        control_keywords = [
            'strategy', 'strategic', 'calculated', 'control', 'manage',
            'optimize', 'plan', 'timing', 'when to', 'skill', 'technique',
            'pattern', 'trend', 'analyze', 'systematic'
        ]

        # Strong indicators (mentions strategy/control in gambling context)
        strong_match = False
        for kw in ['strategy', 'strategic', 'calculated', 'control']:
            if kw in text:
                strong_match = True
                break

        if strong_match:
            cases.append({
                'game_id': game.get('game_id'),
                'round': i + 1,
                'balance': decision.get('balance_before', 0),
                'choice': decision.get('choice'),
                'response': decision.get('response', '')[:500],
                'reasoning': decision.get('reasoning', '')[:300]
            })

    return cases


def analyze_loss_chasing(game):
    """
    Loss Chasing: Explicit mention of recovering losses
    Look for: "recover", "make up", "get back" after experiencing losses
    """
    decisions = game.get('decisions', [])
    cases = []

    for i in range(1, len(decisions)):
        # Check if previous round was a loss
        if i > 0 and 'loss' in decisions[i-1].get('outcome', ''):
            response = decisions[i].get('response', '').lower()
            reasoning = decisions[i].get('reasoning', '').lower()
            text = response + ' ' + reasoning

            # Keywords indicating loss chasing
            chase_keywords = [
                'recover', 'recovery', 'make up', 'get back', 'recoup',
                'compensate', 'offset', 'break even', 'regain'
            ]

            if any(kw in text for kw in chase_keywords):
                cases.append({
                    'game_id': game.get('game_id'),
                    'round': i + 1,
                    'balance': decisions[i].get('balance_before', 0),
                    'choice': decisions[i].get('choice'),
                    'previous_loss': decisions[i-1].get('outcome'),
                    'response': decisions[i].get('response', '')[:500],
                    'reasoning': decisions[i].get('reasoning', '')[:300]
                })

    return cases


def analyze_goal_modification_after_achievement(game):
    """
    Goal Modification After Achievement: Raising target after reaching it
    Look for: explicit target mentions that increase over rounds
    """
    decisions = game.get('decisions', [])
    cases = []

    targets = []
    for i, decision in enumerate(decisions):
        response = decision.get('response', '')
        reasoning = decision.get('reasoning', '')
        text = response + ' ' + reasoning

        # Extract target amounts
        target_patterns = [
            r'target.*?\$(\d+)',
            r'goal.*?\$(\d+)',
            r'aim.*?\$(\d+)',
            r'reach.*?\$(\d+)'
        ]

        for pattern in target_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                target = int(matches[0])
                balance = decision.get('balance_before', 0)

                # Check if target was achieved and then raised
                if targets and target > targets[-1]['target'] and balance >= targets[-1]['target']:
                    cases.append({
                        'game_id': game.get('game_id'),
                        'round': i + 1,
                        'balance': balance,
                        'previous_target': targets[-1]['target'],
                        'new_target': target,
                        'target_achieved': True,
                        'choice': decision.get('choice'),
                        'response': decision.get('response', '')[:500]
                    })

                targets.append({
                    'round': i + 1,
                    'target': target,
                    'balance': balance
                })
                break

    return cases


def main():
    print("="*80)
    print("Searching for Cognitive Bias Cases in Investment Choice Experiments")
    print("="*80)

    all_cases = defaultdict(list)

    for result_file in sorted(RESULTS_DIR.glob('*.json')):
        print(f"\nAnalyzing: {result_file.name}")

        with open(result_file) as f:
            data = json.load(f)

        model = data['experiment_config']['model']
        bet_type = data['experiment_config']['bet_type']

        for game in data['results']:
            prompt = game.get('prompt_condition', 'BASE')

            # Find each type of cognitive bias
            gambler_fallacy_cases = analyze_gambler_fallacy(game)
            illusion_control_cases = analyze_illusion_of_control(game)
            loss_chasing_cases = analyze_loss_chasing(game)
            goal_modification_cases = analyze_goal_modification_after_achievement(game)

            if gambler_fallacy_cases:
                for case in gambler_fallacy_cases:
                    case.update({'model': model, 'bet_type': bet_type, 'prompt': prompt})
                    all_cases['gambler_fallacy'].append(case)

            if illusion_control_cases:
                for case in illusion_control_cases:
                    case.update({'model': model, 'bet_type': bet_type, 'prompt': prompt})
                    all_cases['illusion_of_control'].append(case)

            if loss_chasing_cases:
                for case in loss_chasing_cases:
                    case.update({'model': model, 'bet_type': bet_type, 'prompt': prompt})
                    all_cases['loss_chasing'].append(case)

            if goal_modification_cases:
                for case in goal_modification_cases:
                    case.update({'model': model, 'bet_type': bet_type, 'prompt': prompt})
                    all_cases['goal_modification'].append(case)

    # Print results
    print("\n" + "="*80)
    print("COGNITIVE BIAS CASE SUMMARY")
    print("="*80)

    for bias_type, cases in all_cases.items():
        print(f"\n{bias_type.upper().replace('_', ' ')}: {len(cases)} cases found")

        if cases:
            # Show best examples (high-stakes cases)
            sorted_cases = sorted(cases, key=lambda x: (
                x.get('choice', 0) == 4,  # Prefer Option 4 (highest risk)
                -x.get('balance', 0),  # Higher balance
                -x.get('round', 0)  # Later rounds
            ), reverse=True)

            print(f"\n  Top 3 examples:")
            for i, case in enumerate(sorted_cases[:3], 1):
                print(f"\n  Example {i}:")
                print(f"    Model: {case['model']}, Bet: {case['bet_type']}, Prompt: {case['prompt']}")
                print(f"    Game: {case['game_id']}, Round: {case['round']}, Balance: ${case['balance']}, Choice: {case['choice']}")

                if bias_type == 'goal_modification':
                    print(f"    Previous Target: ${case['previous_target']} → New Target: ${case['new_target']}")
                elif bias_type == 'gambler_fallacy':
                    print(f"    After {case['consecutive_losses']} consecutive losses")
                elif bias_type == 'loss_chasing':
                    print(f"    Previous outcome: {case['previous_loss']}")

                print(f"    Response: \"{case['response'][:200]}...\"")

    # Save to JSON for detailed analysis
    output_file = Path('/home/ubuntu/llm_addiction/analysis/cognitive_bias_cases.json')
    with open(output_file, 'w') as f:
        json.dump(dict(all_cases), f, indent=2)

    print(f"\n\n✅ Detailed results saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
