#!/usr/bin/env python3
"""
Main Runner for Investment Choice Bet Constraint Experiment

Usage:
    python run_all_experiments.py --model gpt4o --bet_constraint 10 --bet_type fixed
    python run_all_experiments.py --model claude --bet_constraint 30 --bet_type variable
    python run_all_experiments.py --model all --bet_constraint all --bet_type both

Models: gpt4o, gpt41, claude, gemini, all
Bet constraints: 10, 30, 50, 70, all
Bet types: fixed, variable, both
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

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


def run_experiment(model_name: str, bet_constraint: int, bet_type: str, trials: int = 25, resume_file: Optional[str] = None):
    """
    Run experiment for specified model, bet constraint, and bet type

    Args:
        model_name: 'gpt4o', 'gpt41', 'claude', or 'gemini'
        bet_constraint: Bet cap amount (10, 30, 50, or 70)
        bet_type: 'fixed' or 'variable'
        trials: Number of trials per condition (default 25)
        resume_file: Optional path to existing results JSON to resume
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting Investment Choice Bet Constraint Experiment")
    print(f"   Model: {model_name}")
    print(f"   Bet Constraint: ${bet_constraint}")
    print(f"   Bet Type: {bet_type}")
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

    # Initialize and run (note: bet_constraint comes first now)
    runner = runners[model_name](bet_constraint, bet_type)
    runner.run_experiment(trials_per_condition=trials, resume_from=resume_file)

    print(f"\n{'='*80}")
    print(f"✅ Experiment completed for {model_name} (${bet_constraint}, {bet_type})")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run Investment Choice Bet Constraint Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GPT-4o-mini with $10 bet cap and fixed betting
  python run_all_experiments.py --model gpt4o --bet_constraint 10 --bet_type fixed

  # Run Claude with $30 bet cap and variable betting
  python run_all_experiments.py --model claude --bet_constraint 30 --bet_type variable

  # Run all models with all bet constraints and both bet types
  python run_all_experiments.py --model all --bet_constraint all --bet_type both

  # Run with custom number of trials
  python run_all_experiments.py --model gpt4o --bet_constraint 50 --bet_type fixed --trials 10
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
        '--bet_constraint',
        type=str,
        required=True,
        help='Bet constraint amount (10, 30, 50, 70, or "all")'
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
        default=25,
        help='Number of trials per condition (default: 25)'
    )

    parser.add_argument(
        '--resume-file',
        type=str,
        default=None,
        help='Path to existing results JSON to resume from'
    )

    args = parser.parse_args()

    # Determine which models to run
    if args.model == 'all':
        models = ['gpt4o', 'gpt41', 'claude', 'gemini']
    else:
        models = [args.model]

    # Determine which bet constraints to run
    if args.bet_constraint == 'all':
        bet_constraints = [10, 30, 50, 70]
    else:
        try:
            bet_constraint_val = int(args.bet_constraint)
            if bet_constraint_val not in [10, 30, 50, 70]:
                raise ValueError(f"Invalid bet constraint: {bet_constraint_val}. Must be 10, 30, 50, or 70")
            bet_constraints = [bet_constraint_val]
        except ValueError as e:
            print(f"❌ Error: {e}")
            return

    # Determine which bet types to run
    if args.bet_type == 'both':
        bet_types = ['fixed', 'variable']
    else:
        bet_types = [args.bet_type]

    # Run experiments
    total_experiments = len(models) * len(bet_constraints) * len(bet_types)

    if args.resume_file and total_experiments != 1:
        raise ValueError("--resume-file can only be used when running a single model/constraint/bet_type combination")

    print(f"\n📊 Total experiments to run: {total_experiments}")
    print(f"   Models: {', '.join(models)}")
    print(f"   Bet constraints: {', '.join([f'${c}' for c in bet_constraints])}")
    print(f"   Bet types: {', '.join(bet_types)}")
    print(f"   Games per experiment: {args.trials * 4} (4 conditions × {args.trials} trials)")
    print(f"   Total games: {total_experiments * args.trials * 4}\n")

    current = 0
    for model in models:
        for bet_constraint in bet_constraints:
            for bet_type in bet_types:
                current += 1
                print(f"\n{'#'*80}")
                print(f"# Experiment {current}/{total_experiments}: {model} - ${bet_constraint} - {bet_type}")
                print(f"{'#'*80}")

                try:
                    resume_file = args.resume_file if args.resume_file else None
                    run_experiment(model, bet_constraint, bet_type, args.trials, resume_file)
                except Exception as e:
                    print(f"\n❌ Error in {model} (${bet_constraint}, {bet_type}): {e}")
                    print("Continuing to next experiment...\n")
                    continue

    print(f"\n{'='*80}")
    print(f"🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
