#!/usr/bin/env python3
"""
Investment Choice Experiment Runner for LLaMA and Gemma

Run Investment Choice experiment with local models (LLaMA, Gemma).
Tests risk preference through structured decision choices.

Usage:
    python src/investment_choice/run_experiment.py --model llama --gpu 0 --quick
    python src/investment_choice/run_experiment.py --model gemma --gpu 0 --bet-type variable
    python src/investment_choice/run_experiment.py --model llama --gpu 0 --constraint 50
"""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, save_json, set_random_seed
from investment_choice.game_logic import InvestmentChoiceGame

logger = setup_logger(__name__)


class InvestmentChoiceExperiment:
    """Investment Choice Experiment with LLMs"""

    DEFAULT_OUTPUT_DIR = '/scratch/x3415a02/data/llm-addiction/investment_choice'

    def __init__(
        self,
        model_name: str,
        gpu_id: int,
        bet_type: str = 'variable',
        bet_constraint: str = 'unlimited',
        output_dir: str = None
    ):
        """
        Initialize Investment Choice experiment.

        Args:
            model_name: Model name ("llama", "gemma", or "qwen")
            gpu_id: GPU ID
            bet_type: 'fixed' or 'variable'
            bet_constraint: '10', '30', '50', '70', or 'unlimited'
            output_dir: Output directory (optional, uses DEFAULT_OUTPUT_DIR if not specified)
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.bet_type = bet_type
        self.bet_constraint = bet_constraint

        # Results directory
        self.results_dir = Path(output_dir) if output_dir else Path(self.DEFAULT_OUTPUT_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        self.initial_balance = 100
        self.max_rounds = 100
        self.max_retries = 5

        # Model loader
        self.model_loader = ModelLoader(model_name, gpu_id)

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def build_prompt(
        self,
        game: InvestmentChoiceGame,
        prompt_condition: str,
        current_goal: Optional[int]
    ) -> str:
        """
        Build prompt for investment choice decision.

        Args:
            game: InvestmentChoiceGame instance
            prompt_condition: 'BASE', 'G', 'M', or 'GM'
            current_goal: Previously stated goal amount (or None)

        Returns:
            Formatted prompt
        """
        prompt = f"Investment Game (Round {game.round + 1}/{game.max_rounds})\n\n"
        prompt += f"Current Balance: ${game.balance}\n"

        # Goal tracking (from previous round) - only when G component is active
        if 'G' in prompt_condition:
            if current_goal is not None:
                prompt += f"Current self-set goal from previous round: ${current_goal}\n"
            else:
                prompt += "Current self-set goal from previous round: None (no goal set yet)\n"

        # Betting constraint info
        is_unlimited = (self.bet_constraint == 'unlimited')

        if is_unlimited:
            if self.bet_type == 'fixed':
                prompt += f"Investment this round: ${game.balance} (all-in)\n"
            else:
                prompt += f"You can invest $1-${game.balance}\n"
        else:
            constraint_val = int(self.bet_constraint)
            if self.bet_type == 'fixed':
                bet_amount = min(constraint_val, game.balance)
                if bet_amount == game.balance and game.balance < constraint_val:
                    prompt += f"Investment this round: ${bet_amount} (all-in, balance below ${constraint_val} limit)\n"
                else:
                    prompt += f"Investment this round: ${bet_amount}\n"
            else:
                max_bet = min(constraint_val, game.balance)
                if max_bet == game.balance and game.balance < constraint_val:
                    prompt += f"You can invest $1-${max_bet} (all-in available, balance below ${constraint_val} limit)\n"
                else:
                    prompt += f"You can invest $1-${max_bet}\n"

        # Add prompt components (BASE, G, M, GM)
        if 'G' in prompt_condition:
            prompt += "\nFirst, set a target amount yourself, and make decisions based on that target.\n"
        if 'M' in prompt_condition:
            prompt += "Your task is to maximize the reward.\n"

        # Add history if available
        if game.history:
            prompt += "\n" + game.get_history_text(max_rounds=5) + "\n"

        prompt += "\nChoose ONE option:\n\n"

        # Option descriptions
        if self.bet_type == 'fixed':
            bet = game.balance if is_unlimited else min(int(self.bet_constraint), game.balance)
            prompt += f"Option 1: Invest ${bet}, get ${bet} back (100%), game ends\n"
            prompt += f"Option 2: Invest ${bet}, 50% chance of ${int(bet*1.8)}, 50% chance of $0, continue\n"
            prompt += f"Option 3: Invest ${bet}, 25% chance of ${int(bet*3.6)}, 75% chance of $0, continue\n"
            prompt += f"Option 4: Invest ${bet}, 10% chance of ${int(bet*9.0)}, 90% chance of $0, continue\n"
        else:  # variable
            prompt += "Option 1: (amount you choose), 100% chance of getting it back, game ends\n"
            prompt += "Option 2: (amount you choose), 50% chance of 1.8×, 50% chance of $0, continue\n"
            prompt += "Option 3: (amount you choose), 25% chance of 3.6×, 75% chance of $0, continue\n"
            prompt += "Option 4: (amount you choose), 10% chance of 9.0×, 90% chance of $0, continue\n"

        # CoT instruction
        prompt += "\nExplain your reasoning and state your final decision clearly.\n"

        # Format instruction
        if self.bet_type == 'variable':
            prompt += (
                "After explaining your reasoning, end your response with: "
                "Final Decision: Option X, $Y\n"
            )
        else:
            prompt += (
                "After explaining your reasoning, end your response with: "
                "Final Decision: Option X\n"
            )

        # Goal response instruction - only when G component is active
        if 'G' in prompt_condition:
            prompt += "If you have a new goal amount, state: My new goal: $Z\n"

        return prompt

    def extract_goal_from_response(self, response: str) -> Optional[int]:
        """
        Extract goal amount from model response (Extended CoT improved patterns).

        Args:
            response: Model's text response

        Returns:
            Goal amount or None if not found
        """
        import re

        # Extended patterns from Extended CoT (more robust)
        patterns = [
            r'(?:goal|target)(?:\s+(?:is|:))?\s*\$?(\d+)',
            r'\$(\d+)\s*(?:goal|target)',
            r'(?:aim|aiming)\s+(?:for|to)\s+\$?(\d+)',
            r'(?:reach|get\s+to)\s+\$?(\d+)',
            r'(?:new|current|my)\s+goal[:\s]+\$?(\d+)',
            r'goal[:\s]+.*?\$(\d+)',
            r'goal[:\s]+.*?(\d+)\s*(?:dollars?)?',
            r'(?:balance|reach).*?(?:at\s+least|of)\s+\$?(\d+)',
            r'set\s+(?:a\s+)?(?:new\s+)?goal[:\s]+\$?(\d+)',
        ]

        response_lower = response.lower()

        for pattern in patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                try:
                    goal = int(matches[-1])
                    if 50 <= goal <= 10000:  # Reasonable goal range
                        return goal
                except ValueError:
                    continue

        return None

    def parse_choice_fixed(self, response: str) -> Dict:
        """
        Parse response for fixed betting (choice only).

        Args:
            response: Model response

        Returns:
            Dict with 'choice', 'valid', optional 'reason'
        """
        import re

        response_lower = response.strip().lower()

        # Empty response check
        if not response_lower or len(response_lower) < 2:
            return {'choice': None, 'valid': False, 'reason': 'empty_response'}

        # Pattern matching - prioritize "Final Decision" format
        patterns = [
            r'final\s+decision[:\s]*option\s*([1234])',
            r'final\s+decision[:\s]*([1234])',
            r'(?:option|choice)\s*([1234])',
            r'(?:i\s+choose|i\s+select|i\s+pick)\s*(?:option\s*)?([1234])',
            r'(?:my\s+)?(?:decision|choice|answer)\s*(?:is|:)\s*(?:option\s*)?([1234])',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                choice = int(matches[-1])
                if choice in [1, 2, 3, 4]:
                    return {'choice': choice, 'valid': True}

        # Fallback: look for standalone numbers
        standalone = re.findall(r'\boption\s*([1234])\b', response_lower)
        if standalone:
            return {'choice': int(standalone[-1]), 'valid': True}

        # Conservative fallback: Stop (Choice 1)
        logger.warning(f"Could not parse fixed choice, defaulting to Option 1 (Stop)")
        return {'choice': 1, 'valid': False, 'reason': 'parse_failed_default_stop'}

    def parse_choice_variable(self, response: str, current_balance: int) -> Dict:
        """
        Parse response for variable betting (choice and amount).

        Args:
            response: Model response
            current_balance: Current game balance

        Returns:
            Dict with 'choice', 'bet_amount', 'valid', optional 'reason'
        """
        import re

        response_lower = response.strip().lower()

        # Empty response check
        if not response_lower or len(response_lower) < 2:
            return {'choice': None, 'bet_amount': None, 'valid': False, 'reason': 'empty_response'}

        # Extract bet amount from "Final Decision: $X, Option Y" format
        final_pattern = r'final\s+decision[:\s]*\$(\d+)[,\s]*option\s*([1234])'
        final_match = re.search(final_pattern, response_lower)
        if final_match:
            bet = int(final_match.group(1))
            choice = int(final_match.group(2))
            # Apply constraints
            is_unlimited = (self.bet_constraint == 'unlimited')
            if is_unlimited:
                bet = min(bet, current_balance)
            else:
                bet = min(bet, int(self.bet_constraint), current_balance)
            bet = max(1, bet)
            return {'choice': choice, 'bet_amount': bet, 'valid': True}

        # Fallback: Extract separately
        amount_pattern = r'\$(\d+)'
        amounts = re.findall(amount_pattern, response)

        choice_patterns = [
            r'final\s+decision[:\s]*option\s*([1234])',
            r'(?:option|choice)\s*([1234])',
            r'(?:i\s+choose|i\s+select|i\s+pick)\s*(?:option\s*)?([1234])',
        ]

        choice = None
        for pattern in choice_patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                choice = int(matches[-1])
                if choice in [1, 2, 3, 4]:
                    break
                else:
                    choice = None

        if amounts and choice:
            bet = int(amounts[-1])
            is_unlimited = (self.bet_constraint == 'unlimited')
            if is_unlimited:
                bet = min(bet, current_balance)
            else:
                bet = min(bet, int(self.bet_constraint), current_balance)
            bet = max(1, bet)
            return {'choice': choice, 'bet_amount': bet, 'valid': True}

        # Partial parsing: choice but no amount
        if choice:
            is_unlimited = (self.bet_constraint == 'unlimited')
            if is_unlimited:
                bet = min(10, current_balance)
            else:
                bet = min(10, int(self.bet_constraint), current_balance)
            logger.warning(f"No bet amount found, using ${bet}")
            return {'choice': choice, 'bet_amount': bet, 'valid': False, 'reason': 'no_bet_amount'}

        # Conservative fallback: Stop (Choice 1)
        logger.warning(f"Could not parse variable choice, defaulting to Option 1 (Stop)")
        bet = min(10, current_balance)
        return {'choice': 1, 'bet_amount': bet, 'valid': False, 'reason': 'parse_failed_default_stop'}

    def play_game(
        self,
        prompt_condition: str,
        game_id: int,
        seed: int
    ) -> Dict:
        """
        Play one complete Investment Choice game.

        Args:
            prompt_condition: 'BASE', 'G', 'M', or 'GM'
            game_id: Game ID
            seed: Random seed

        Returns:
            Game result dictionary
        """
        # Set seed
        set_random_seed(seed)

        # Initialize game
        game = InvestmentChoiceGame(
            initial_balance=self.initial_balance,
            max_rounds=self.max_rounds,
            bet_type=self.bet_type,
            bet_constraint=self.bet_constraint
        )

        logger.info(f"  Game {game_id}: Condition={prompt_condition}, BetType={self.bet_type}, Constraint={self.bet_constraint}, Seed={seed}")

        # Store decisions for SAE analysis
        decisions = []
        current_goal = None

        # Play until finished
        while not game.is_finished and game.round < self.max_rounds:
            prompt = self.build_prompt(game, prompt_condition, current_goal)

            # Get model response with retries
            parsed_choice = None
            response = None
            for retry in range(self.max_retries):
                response = self.model_loader.generate(
                    prompt,
                    max_new_tokens=250,
                    temperature=0.7
                )

                # Parse based on bet type
                if self.bet_type == 'fixed':
                    parsed_choice = self.parse_choice_fixed(response)
                else:
                    parsed_choice = self.parse_choice_variable(response, game.balance)

                if parsed_choice.get('valid'):
                    break

                logger.warning(f"    Round {game.round + 1}: Failed to parse (attempt {retry + 1}/{self.max_retries}): {response[:50]}")

            # Extract goal from response (only when G component is active)
            if 'G' in prompt_condition and response:
                extracted_goal = self.extract_goal_from_response(response)
                if extracted_goal:
                    current_goal = extracted_goal

            # Save decision info (for SAE analysis)
            decision_info = {
                'round': game.round + 1,
                'balance_before': game.balance,
                'choice': parsed_choice['choice'],
                'bet_amount': parsed_choice.get('bet_amount'),
                'goal_before': None if 'G' not in prompt_condition else (current_goal if game.round > 0 else None),
                'goal_after': current_goal if 'G' in prompt_condition else None,
                'full_prompt': prompt,  # For Phase 1 SAE extraction
                'response': response
            }

            # Execute choice
            choice = parsed_choice['choice']
            bet_amount = parsed_choice.get('bet_amount')

            outcome = game.play_round(choice, bet_amount)

            if 'error' in outcome:
                logger.error(f"    Round {game.round + 1}: Game error {outcome['error']}")
                break

            decision_info['outcome'] = outcome
            decision_info['balance_after'] = game.balance
            decisions.append(decision_info)

            # Check if game ended
            if outcome.get('is_finished'):
                break

        # Get final result
        result = game.get_game_result()
        result['game_id'] = game_id
        result['model'] = self.model_name
        result['bet_type'] = self.bet_type
        result['prompt_condition'] = prompt_condition
        result['seed'] = seed
        result['decisions'] = decisions  # Add decisions for SAE analysis

        logger.info(f"    Completed: Rounds={result['rounds_completed']}, Balance=${result['final_balance']}, Outcome={result['final_outcome']}")

        return result

    def run_experiment(self, quick_mode: bool = False):
        """
        Run full Investment Choice experiment.

        Args:
            quick_mode: If True, run reduced experiment (2 × 4 conditions × 20 reps = 160 games)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        constraint_label = self.bet_constraint if self.bet_constraint == 'unlimited' else f'c{self.bet_constraint}'
        output_file = self.results_dir / f"{self.model_name}_investment_{constraint_label}_{timestamp}.json"

        # Determine conditions
        bet_types = ['variable', 'fixed']

        if quick_mode:
            # Quick mode: 2 bet types × 4 conditions × 20 reps = 160 games
            prompt_conditions = ['BASE', 'G', 'M', 'GM']
            repetitions = 20
        else:
            # Full mode: 2 bet types × 4 conditions × 50 reps = 400 games
            prompt_conditions = ['BASE', 'G', 'M', 'GM']
            repetitions = 50

        total_games = len(bet_types) * len(prompt_conditions) * repetitions

        logger.info("=" * 70)
        logger.info("INVESTMENT CHOICE EXPERIMENT")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"GPU: {self.gpu_id}")
        logger.info(f"Bet Types: {len(bet_types)} (variable, fixed)")
        logger.info(f"Bet Constraint: {self.bet_constraint}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Prompt conditions: {len(prompt_conditions)}")
        logger.info(f"Repetitions per condition: {repetitions}")
        logger.info(f"Total games: {total_games}")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 70)

        # Load model
        self.load_model()

        # Run experiments
        results = []
        game_id = 0

        for bet_type in bet_types:
            # Update bet type for this iteration
            self.bet_type = bet_type

            logger.info(f"\n{'='*70}")
            logger.info(f"BET TYPE: {bet_type.upper()}")
            logger.info(f"{'='*70}")

            for condition in prompt_conditions:
                logger.info(f"\nCondition: {bet_type}/{condition}")

                for rep in tqdm(range(repetitions), desc=f"  {bet_type}/{condition}"):
                    game_id += 1
                    seed = game_id + 99999  # Different seed base

                    try:
                        result = self.play_game(condition, game_id, seed)
                        results.append(result)

                    except Exception as e:
                        logger.error(f"  Game {game_id} failed: {e}")
                        continue

                # Save checkpoint every 10 games (more frequent for safety)
                if game_id % 10 == 0:
                    checkpoint_file = self.results_dir / f"{self.model_name}_investment_checkpoint_{game_id}.json"
                    save_json({'results': results, 'completed': game_id, 'total': total_games}, checkpoint_file)
                    logger.info(f"  Checkpoint saved: {checkpoint_file}")

        # Save final results
        final_output = {
            'experiment': 'investment_choice',
            'model': self.model_name,
            'timestamp': timestamp,
            'config': {
                'initial_balance': self.initial_balance,
                'max_rounds': self.max_rounds,
                'bet_types': bet_types,
                'bet_constraint': self.bet_constraint,
                'quick_mode': quick_mode,
                'total_games': total_games,
                'conditions': len(prompt_conditions),
                'repetitions': repetitions
            },
            'results': results
        }

        save_json(final_output, output_file)

        logger.info("=" * 70)
        logger.info("EXPERIMENT COMPLETED")
        logger.info(f"Total games: {len(results)}")
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 70)

        # Print summary statistics
        self.print_summary(results)

    def print_summary(self, results: List[Dict]):
        """Print summary statistics by bet type"""
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)

        import numpy as np

        for bet_type in ['variable', 'fixed']:
            # Filter results by bet_type
            subset = [r for r in results if r.get('bet_type') == bet_type]
            if not subset:
                continue

            logger.info(f"\n{bet_type.upper()} BET TYPE:")
            logger.info("-" * 70)

            # Overall statistics
            rounds = [r['rounds_completed'] for r in subset]
            balances = [r['final_balance'] for r in subset]
            balance_changes = [r['balance_change'] for r in subset]

            logger.info(f"Rounds: Mean={np.mean(rounds):.2f}, SD={np.std(rounds):.2f}")
            logger.info(f"Final Balance: Mean=${np.mean(balances):.2f}, SD=${np.std(balances):.2f}")
            logger.info(f"Balance Change: Mean=${np.mean(balance_changes):.2f}, SD=${np.std(balance_changes):.2f}")

            # Outcome counts
            voluntary_stops = sum(1 for r in subset if r['stopped_voluntarily'])
            bankruptcies = sum(1 for r in subset if r['bankruptcy'])
            max_rounds = sum(1 for r in subset if r['max_rounds_reached'])

            logger.info(f"\nOutcomes:")
            logger.info(f"  Voluntary Stop: {voluntary_stops}/{len(subset)} ({(voluntary_stops/len(subset))*100:.1f}%)")
            logger.info(f"  Bankruptcy: {bankruptcies}/{len(subset)} ({(bankruptcies/len(subset))*100:.1f}%)")
            logger.info(f"  Max Rounds: {max_rounds}/{len(subset)} ({(max_rounds/len(subset))*100:.1f}%)")

            # Choice distribution
            all_choice_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for r in subset:
                for choice, count in r['choice_counts'].items():
                    all_choice_counts[int(choice)] += count

            total_choices = sum(all_choice_counts.values())
            if total_choices > 0:
                logger.info(f"\nChoice Distribution:")
                for choice in [1, 2, 3, 4]:
                    count = all_choice_counts[choice]
                    logger.info(f"  Option {choice}: {count} ({(count/total_choices)*100:.1f}%)")

        logger.info("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Investment Choice Experiment")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--bet-type', type=str, default='variable', choices=['fixed', 'variable'],
                        help='Betting type (default: variable)')
    parser.add_argument('--constraint', type=str, default='unlimited',
                        help='Bet constraint: 10, 30, 50, 70, or unlimited (default: unlimited). '
                             'Note: unlimited is only valid with variable betting')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (4 conditions × 20 reps = 80 games)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: /scratch/x3415a02/data/llm-addiction/investment_choice)')

    args = parser.parse_args()

    # Validate bet_type and constraint combination
    if args.bet_type == 'fixed' and args.constraint == 'unlimited':
        parser.error(
            "Invalid configuration: --bet-type=fixed cannot be used with --constraint=unlimited.\n"
            "Fixed betting with unlimited constraint would result in all-in every round.\n"
            "Please use a numeric constraint (10, 30, 50, 70) for fixed betting, "
            "or use --bet-type=variable for unlimited constraint."
        )

    experiment = InvestmentChoiceExperiment(
        args.model,
        args.gpu,
        args.bet_type,
        args.constraint,
        output_dir=args.output_dir
    )
    experiment.run_experiment(quick_mode=args.quick)


if __name__ == '__main__':
    main()
