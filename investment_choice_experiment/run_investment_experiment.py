#!/usr/bin/env python3
"""
Main Runner for Investment Choice Experiment

Usage:
    python run_investment_experiment.py --model gpt4o --bet_type fixed
    python run_investment_experiment.py --model claude --bet_type variable
    python run_investment_experiment.py --model all --bet_type both

Models: gpt4o, gpt41, claude, gemini, all
Bet types: fixed, variable, both
"""

import argparse
import os
from pathlib import Path

# Load .env file if exists
def load_env():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()

from models import GPT4oRunner, GPT41Runner, ClaudeRunner, GeminiRunner


def run_experiment(model_name: str, bet_type: str, trials: int = 50):
    """
    Run experiment for specified model and bet type

    Args:
        model_name: 'gpt4o', 'gpt41', 'claude', or 'gemini'
        bet_type: 'fixed' or 'variable'
        trials: Number of trials per condition (default 50)
    """
    print(f"\n{'='*80}")
    print(f"🚀 Starting Investment Choice Experiment")
    print(f"   Model: {model_name}")
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

    # Initialize and run
    runner = runners[model_name](bet_type)
    runner.run_experiment(trials_per_condition=trials)

    print(f"\n{'='*80}")
    print(f"✅ Experiment completed for {model_name} ({bet_type})")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run Investment Choice Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GPT-4o-mini with fixed betting
  python run_investment_experiment.py --model gpt4o --bet_type fixed

  # Run Claude with variable betting
  python run_investment_experiment.py --model claude --bet_type variable

  # Run all models with both bet types
  python run_investment_experiment.py --model all --bet_type both

  # Run with custom number of trials
  python run_investment_experiment.py --model gpt4o --bet_type fixed --trials 10
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

    args = parser.parse_args()

    # Determine which models and bet types to run
    if args.model == 'all':
        models = ['gpt4o', 'gpt41', 'claude', 'gemini']
    else:
        models = [args.model]

    if args.bet_type == 'both':
        bet_types = ['fixed', 'variable']
    else:
        bet_types = [args.bet_type]

    # Run experiments
    total_experiments = len(models) * len(bet_types)
    print(f"\n📊 Total experiments to run: {total_experiments}")
    print(f"   Models: {', '.join(models)}")
    print(f"   Bet types: {', '.join(bet_types)}")
    print(f"   Games per experiment: {args.trials * 4} (4 conditions × {args.trials} trials)")
    print(f"   Total games: {total_experiments * args.trials * 4}\n")

    current = 0
    for model in models:
        for bet_type in bet_types:
            current += 1
            print(f"\n{'#'*80}")
            print(f"# Experiment {current}/{total_experiments}: {model} - {bet_type}")
            print(f"{'#'*80}")

            try:
                run_experiment(model, bet_type, args.trials)
            except Exception as e:
                print(f"\n❌ Error in {model} ({bet_type}): {e}")
                print("Continuing to next experiment...\n")
                continue

    print(f"\n{'='*80}")
    print(f"🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
