#!/usr/bin/env python3
"""
Main Runner for Extended Investment Choice Experiment (100 rounds)

Usage:
    python run_all_experiments.py --model gpt4o --constraint 10 --bet_type fixed
    python run_all_experiments.py --model claude --constraint unlimited --bet_type variable
    python run_all_experiments.py --model all --constraint all --bet_type both

Models: gpt4o, gpt41, claude, gemini, all
Constraints: 10, 30, 50, 70, unlimited, all
Bet types: fixed, variable, both
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load .env file if exists
def load_env():
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()

from models.gpt4o_runner import GPT4oRunner
from models.gpt41_runner import GPT41Runner
from models.claude_runner import ClaudeRunner
from models.gemini_runner import GeminiRunner


def run_experiment(model_name: str, constraint: Union[int, str], bet_type: str,
                   trials: int = 50, resume_file: Optional[str] = None):
    """
    Run experiment for specified model, constraint, and bet type

    Args:
        model_name: 'gpt4o', 'gpt41', 'claude', or 'gemini'
        constraint: Bet cap amount (10, 30, 50, 70) or 'unlimited'
        bet_type: 'fixed' or 'variable'
        trials: Number of trials per condition (default 50)
        resume_file: Optional path to existing results JSON to resume
    """
    constraint_str = f"${constraint}" if isinstance(constraint, int) else constraint

    print(f"\n{'='*80}")
    print(f"🚀 Starting Extended Investment Choice Experiment (100 rounds)")
    print(f"   Model: {model_name}")
    print(f"   Constraint: {constraint_str}")
    print(f"   Bet Type: {bet_type}")
    print(f"   Max Rounds: 100")
    print(f"   Trials per condition: {trials}")
    print(f"{'='*80}\n")

    # Select runner
    runners = {
        'gpt4o': GPT4oRunner,
        'gpt41': GPT41Runner,
        'claude': ClaudeRunner,
        'gemini': GeminiRunner
    }

    if model_name not in runners:
        raise ValueError(f"Invalid model: {model_name}. Choose from: {list(runners.keys())}")

    # Initialize and run
    runner = runners[model_name](constraint, bet_type)
    runner.run_experiment(trials_per_condition=trials, resume_from=resume_file)

    print(f"\n{'='*80}")
    print(f"✅ Experiment completed for {model_name} ({constraint_str}, {bet_type})")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run Extended Investment Choice Experiment (100 rounds)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GPT-4o-mini with $10 constraint and fixed betting
  python run_all_experiments.py --model gpt4o --constraint 10 --bet_type fixed

  # Run Claude with unlimited constraint and variable betting
  python run_all_experiments.py --model claude --constraint unlimited --bet_type variable

  # Run all models with all constraints and both bet types
  python run_all_experiments.py --model all --constraint all --bet_type both

  # Run with custom number of trials
  python run_all_experiments.py --model gpt4o --constraint 50 --bet_type fixed --trials 25

  # Resume from checkpoint
  python run_all_experiments.py --model claude --constraint 30 --bet_type variable --resume path/to/checkpoint.json
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['gpt4o', 'gpt41', 'claude', 'gemini', 'all'],
        help='Model to use (or "all" for all models)'
    )

    parser.add_argument(
        '--constraint',
        type=str,
        required=True,
        help='Bet constraint (10, 30, 50, 70, unlimited, or "all")'
    )

    parser.add_argument(
        '--bet_type',
        type=str,
        required=True,
        choices=['fixed', 'variable', 'both'],
        help='Betting type (or "both" for both types)'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of trials per condition (default: 50)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to existing checkpoint JSON to resume from'
    )

    args = parser.parse_args()

    # Determine which models to run
    if args.model == 'all':
        models = ['gpt4o', 'gpt41', 'claude', 'gemini']
    else:
        models = [args.model]

    # Determine which constraints to run
    valid_int_constraints = [10, 30, 50, 70]
    if args.constraint == 'all':
        constraints = valid_int_constraints + ['unlimited']
    elif args.constraint.lower() == 'unlimited':
        constraints = ['unlimited']
    else:
        try:
            constraint_val = int(args.constraint)
            if constraint_val not in valid_int_constraints:
                print(f"⚠️ Warning: Non-standard constraint ${constraint_val}")
            constraints = [constraint_val]
        except ValueError:
            print(f"❌ Error: Invalid constraint '{args.constraint}'. Must be 10, 30, 50, 70, unlimited, or all")
            return

    # Determine which bet types to run
    if args.bet_type == 'both':
        bet_types = ['fixed', 'variable']
    else:
        bet_types = [args.bet_type]

    # Run experiments
    total_experiments = len(models) * len(constraints) * len(bet_types)

    if args.resume and total_experiments != 1:
        raise ValueError("--resume can only be used when running a single model/constraint/bet_type combination")

    print(f"\n📊 Extended Investment Choice Experiment (100 rounds)")
    print(f"   Total experiments to run: {total_experiments}")
    print(f"   Models: {', '.join(models)}")
    print(f"   Constraints: {', '.join([f'${c}' if isinstance(c, int) else c for c in constraints])}")
    print(f"   Bet types: {', '.join(bet_types)}")
    print(f"   Games per experiment: {args.trials * 4} (4 conditions × {args.trials} trials)")
    print(f"   Total games: {total_experiments * args.trials * 4}\n")

    current = 0
    for model in models:
        for constraint in constraints:
            for bet_type in bet_types:
                current += 1
                constraint_str = f"${constraint}" if isinstance(constraint, int) else constraint
                print(f"\n{'#'*80}")
                print(f"# Experiment {current}/{total_experiments}: {model} - {constraint_str} - {bet_type}")
                print(f"{'#'*80}")

                try:
                    resume_file = args.resume if args.resume else None
                    run_experiment(model, constraint, bet_type, args.trials, resume_file)
                except Exception as e:
                    print(f"\n❌ Error in {model} ({constraint_str}, {bet_type}): {e}")
                    print("Continuing to next experiment...\n")
                    continue

    print(f"\n{'='*80}")
    print(f"🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
