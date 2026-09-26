#!/usr/bin/env python3
"""
Extended Investment Choice Experiment Runner
- 100 rounds (extended from 10)
- Constraints: $10, $30, $50, $70, unlimited
- Bet types: fixed, variable
- Models: Claude, GPT-4o, GPT-4.1, Gemini
- CoT prompting with goal tracking

Usage:
    python run_experiment.py --model claude --constraint 30 --bet_type variable
    python run_experiment.py --model gpt4o --constraint unlimited --bet_type variable
    python run_experiment.py --model gemini --constraint 50 --bet_type fixed
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Extended Investment Choice Experiment')
    parser.add_argument('--model', type=str, required=True,
                       choices=['claude', 'gpt4o', 'gpt41', 'gemini'],
                       help='Model to use')
    parser.add_argument('--constraint', type=str, required=True,
                       help='Bet constraint (10, 30, 50, 70, or unlimited)')
    parser.add_argument('--bet_type', type=str, required=True,
                       choices=['fixed', 'variable'],
                       help='Betting type')
    parser.add_argument('--trials', type=int, default=50,
                       help='Trials per condition (default: 50)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint file to resume from')

    args = parser.parse_args()

    # Parse constraint
    if args.constraint.lower() == 'unlimited':
        bet_constraint = 'unlimited'
    else:
        try:
            bet_constraint = int(args.constraint)
            if bet_constraint not in [10, 30, 50, 70]:
                print(f"⚠️ Warning: Non-standard constraint {bet_constraint}")
        except ValueError:
            print(f"❌ Invalid constraint: {args.constraint}")
            print("   Valid options: 10, 30, 50, 70, unlimited")
            sys.exit(1)

    # Select runner
    if args.model == 'claude':
        from models.claude_runner import ClaudeRunner
        runner = ClaudeRunner(bet_constraint, args.bet_type)
    elif args.model == 'gpt4o':
        from models.gpt4o_runner import GPT4oRunner
        runner = GPT4oRunner(bet_constraint, args.bet_type)
    elif args.model == 'gpt41':
        from models.gpt41_runner import GPT41Runner
        runner = GPT41Runner(bet_constraint, args.bet_type)
    elif args.model == 'gemini':
        from models.gemini_runner import GeminiRunner
        runner = GeminiRunner(bet_constraint, args.bet_type)
    else:
        print(f"❌ Unknown model: {args.model}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("EXTENDED INVESTMENT CHOICE EXPERIMENT")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Constraint: {bet_constraint}")
    print(f"Bet Type: {args.bet_type}")
    print(f"Max Rounds: 100")
    print(f"Trials per condition: {args.trials}")
    print("=" * 70 + "\n")

    # Run experiment
    runner.run_experiment(trials_per_condition=args.trials, resume_from=args.resume)


if __name__ == '__main__':
    main()
