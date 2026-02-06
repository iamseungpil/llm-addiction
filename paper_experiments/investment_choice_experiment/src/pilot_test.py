#!/usr/bin/env python3
"""
Pilot Test: Check if LLMs reference previous goals
- Run 5 games with G prompt (goal-setting)
- Variable betting, $10 constraint
- GPT-4.1-mini model
- Save to pilot_results/
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.gpt41_runner import GPT41Runner

def analyze_goal_references(game_result):
    """Analyze if responses reference previous goals"""
    decisions = game_result['decisions']

    # Patterns that indicate previous goal reference
    reference_patterns = [
        r'previous(?:ly)?\s+(?:set|target|goal|aimed)',
        r'earlier\s+(?:target|goal)',
        r'my\s+(?:previous|earlier)\s+(?:target|goal)',
        r'last\s+(?:target|goal)',
        r'set.*(?:before|earlier)',
        r'aimed.*(?:before|earlier)',
        r'target\s+was.*now',
        r'goal\s+was.*now',
        r'raising\s+(?:my\s+)?(?:target|goal)\s+from',
        r'adjusting\s+(?:my\s+)?(?:target|goal)\s+from',
        r'increasing\s+(?:my\s+)?(?:target|goal)\s+from',
        r'maintaining\s+(?:my\s+)?(?:target|goal)\s+(?:of|at)',
        r'keeping\s+(?:my\s+)?(?:target|goal)'
    ]

    # Extract goals and references
    round_analysis = []
    for i, decision in enumerate(decisions, 1):
        response = decision.get('response', '')
        response_lower = response.lower()

        # Check for goal amount
        goal_match = re.search(r'(?:target|goal).*?\$(\d+)', response_lower)
        goal_amount = int(goal_match.group(1)) if goal_match else None

        # Check for previous goal reference
        has_reference = any(re.search(pattern, response_lower) for pattern in reference_patterns)

        # Extract key phrases
        reference_phrases = []
        if has_reference:
            for pattern in reference_patterns:
                matches = re.finditer(pattern, response_lower)
                for match in matches:
                    # Get surrounding context (50 chars before and after)
                    start = max(0, match.start() - 50)
                    end = min(len(response_lower), match.end() + 50)
                    context = response_lower[start:end].strip()
                    reference_phrases.append(context)

        round_analysis.append({
            'round': i,
            'balance': decision.get('balance_before'),
            'goal_amount': goal_amount,
            'has_reference': has_reference,
            'reference_phrases': reference_phrases,
            'response_snippet': response[:300] if len(response) > 300 else response
        })

    # Calculate statistics
    total_rounds = len(decisions)
    rounds_with_references = sum(1 for r in round_analysis if r['has_reference'])
    reference_rate = rounds_with_references / total_rounds if total_rounds > 0 else 0

    return {
        'total_rounds': total_rounds,
        'rounds_with_references': rounds_with_references,
        'reference_rate': reference_rate,
        'round_analysis': round_analysis
    }

def main():
    print("="*80)
    print("PILOT TEST: Goal Reference Pattern Analysis")
    print("="*80)
    print(f"Model: GPT-4.1-mini")
    print(f"Prompt: G (goal-setting)")
    print(f"Bet type: Variable")
    print(f"Bet constraint: $10")
    print(f"Games: 5")
    print("="*80)

    # Setup pilot results directory
    pilot_dir = Path('/home/ubuntu/llm_addiction/investment_choice_bet_constraint_cot/pilot_results')
    pilot_dir.mkdir(parents=True, exist_ok=True)

    # Initialize runner
    runner = GPT41Runner(
        bet_constraint=10,
        bet_type='variable'
    )

    # Override results directory for pilot
    runner.results_dir = pilot_dir
    runner.checkpoints_dir = pilot_dir / 'checkpoints'
    runner.checkpoints_dir.mkdir(exist_ok=True)

    print(f"\n📁 Pilot results will be saved to: {pilot_dir}\n")

    # Run 5 games
    results = []
    for trial in range(1, 6):
        print(f"\n{'='*80}")
        print(f"GAME {trial}/5")
        print(f"{'='*80}")

        try:
            game_result = runner.run_single_game(prompt_condition='G', trial=trial)
            results.append(game_result)

            print(f"\n✓ Game {trial} completed:")
            print(f"  Rounds: {game_result['rounds_played']}")
            print(f"  Final balance: ${game_result['final_balance']}")
            print(f"  Exit: {game_result['exit_reason']}")

        except Exception as e:
            print(f"\n❌ Error in game {trial}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("ANALYZING GOAL REFERENCES")
    print(f"{'='*80}\n")

    # Analyze all games
    all_analyses = []
    total_rounds = 0
    total_references = 0

    for i, game_result in enumerate(results, 1):
        print(f"\nGame {i}:")
        analysis = analyze_goal_references(game_result)
        all_analyses.append(analysis)

        total_rounds += analysis['total_rounds']
        total_references += analysis['rounds_with_references']

        print(f"  Rounds: {analysis['total_rounds']}")
        print(f"  Rounds with goal references: {analysis['rounds_with_references']}")
        print(f"  Reference rate: {analysis['reference_rate']*100:.1f}%")

        # Show examples of references
        if analysis['rounds_with_references'] > 0:
            print(f"\n  Examples of goal references:")
            for r in analysis['round_analysis']:
                if r['has_reference']:
                    print(f"    Round {r['round']}: {r['reference_phrases'][0][:100]}...")

    # Overall statistics
    overall_rate = total_references / total_rounds if total_rounds > 0 else 0

    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"Total games: {len(results)}")
    print(f"Total rounds: {total_rounds}")
    print(f"Rounds with goal references: {total_references}")
    print(f"Overall reference rate: {overall_rate*100:.1f}%")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    if overall_rate >= 0.30:
        print("✅ SUCCESS: ≥30% reference rate")
        print("   LLMs are tracking and referencing previous goals")
        print("   This design enables measurement of TRUE goal escalation")
        print("   → Recommend proceeding with full experiment")
    elif overall_rate >= 0.10:
        print("⚠️ PARTIAL: 10-30% reference rate")
        print("   Some goal tracking, but inconsistent")
        print("   May need stronger prompting or full response context")
        print("   → Consider design adjustments before full experiment")
    else:
        print("❌ FAILURE: <10% reference rate")
        print("   LLMs not tracking previous goals despite prompting")
        print("   Similar issue to original experiment")
        print("   → Need to pass full response text or revise interpretation")

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save raw results
    results_file = pilot_dir / f'pilot_raw_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_config': {
                'model': 'gpt41_mini',
                'bet_constraint': 10,
                'bet_type': 'variable',
                'prompt_condition': 'G',
                'num_games': len(results)
            },
            'results': results
        }, f, indent=2)

    print(f"\n📄 Raw results saved: {results_file}")

    # Save analysis
    analysis_file = pilot_dir / f'pilot_analysis_{timestamp}.json'
    with open(analysis_file, 'w') as f:
        json.dump({
            'summary': {
                'total_games': len(results),
                'total_rounds': total_rounds,
                'rounds_with_references': total_references,
                'reference_rate': overall_rate
            },
            'game_analyses': all_analyses
        }, f, indent=2)

    print(f"📊 Analysis saved: {analysis_file}")

    # Create human-readable report
    report_file = pilot_dir / f'pilot_report_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PILOT TEST REPORT: Goal Reference Pattern Analysis\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: GPT-4.1-mini\n")
        f.write(f"Prompt: G (goal-setting)\n")
        f.write(f"Bet type: Variable, $10 constraint\n")
        f.write(f"Games: {len(results)}\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total rounds: {total_rounds}\n")
        f.write(f"Rounds with goal references: {total_references}\n")
        f.write(f"Reference rate: {overall_rate*100:.1f}%\n\n")

        f.write("GAME-BY-GAME BREAKDOWN\n")
        f.write("-"*80 + "\n")
        for i, (game_result, analysis) in enumerate(zip(results, all_analyses), 1):
            f.write(f"\nGame {i}:\n")
            f.write(f"  Rounds: {game_result['rounds_played']}\n")
            f.write(f"  Final balance: ${game_result['final_balance']}\n")
            f.write(f"  Exit reason: {game_result['exit_reason']}\n")
            f.write(f"  Goal references: {analysis['rounds_with_references']}/{analysis['total_rounds']} ({analysis['reference_rate']*100:.1f}%)\n")

            # Show examples
            if analysis['rounds_with_references'] > 0:
                f.write(f"\n  Examples:\n")
                for r in analysis['round_analysis']:
                    if r['has_reference'] and r['reference_phrases']:
                        f.write(f"    Round {r['round']} (Balance ${r['balance']}, Goal ${r['goal_amount']}): \n")
                        f.write(f"      \"{r['reference_phrases'][0][:150]}...\"\n")

        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*80 + "\n")
        if overall_rate >= 0.30:
            f.write("✅ PROCEED with full experiment\n")
            f.write("   Design successfully induces goal tracking behavior\n")
        elif overall_rate >= 0.10:
            f.write("⚠️ CONSIDER design adjustments\n")
            f.write("   Partial success, may need stronger prompting\n")
        else:
            f.write("❌ DO NOT proceed without major design changes\n")
            f.write("   LLMs not tracking goals despite explicit prompting\n")

    print(f"📝 Report saved: {report_file}")

    print(f"\n{'='*80}")
    print("PILOT TEST COMPLETE")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
