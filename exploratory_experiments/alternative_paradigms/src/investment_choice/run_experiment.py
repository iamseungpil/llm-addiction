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

    def __init__(
        self,
        model_name: str,
        gpu_id: int,
        bet_type: str = 'variable',
        bet_constraint: str = 'unlimited'
    ):
        """
        Initialize Investment Choice experiment.

        Args:
            model_name: Model name ("llama", "gemma", or "qwen")
            gpu_id: GPU ID
            bet_type: 'fixed' or 'variable'
            bet_constraint: '10', '30', '50', '70', or 'unlimited'
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.bet_type = bet_type
        self.bet_constraint = bet_constraint

        # Results directory
        self.results_dir = Path('/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms/investment_choice')
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

        # Goal tracking (from previous round)
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
        prompt += "\nExplain your reasoning and state your final decision clearly."

        if self.bet_type == 'variable':
            prompt += (
                "\nAfter explaining your reasoning, end your response with a single line in the exact format: "
                "Final Decision: <Option X, $Y> (e.g., 'Final Decision: Option 2, $30')"
            )
            prompt += "\nIf you have a new goal amount, also state: My new goal: $Z"
        else:
            prompt += (
                "\nAfter explaining your reasoning, end your response with a single line in the exact format: "
                "Final Decision: <Option X> (e.g., 'Final Decision: Option 2')"
            )
            prompt += "\nIf you have a new goal amount, also state: My new goal: $Z"

        return prompt

    def parse_choice(self, response: str, bet_type: str) -> Dict:
        """
        Parse choice from model response.

        Args:
            response: Model response
            bet_type: 'fixed' or 'variable'

        Returns:
            Dict with 'choice' (1-4), 'bet_amount' (if variable), 'new_goal' (if stated), 'valid' (bool)
        """
        import re

        response_lower = response.strip().lower()

        # Empty response check
        if not response_lower or len(response_lower) < 2:
            return {'choice': None, 'valid': False, 'reason': 'empty_response'}

        # Extract "Final Decision:" line
        final_decision_match = re.search(r'final decision:\s*(.+)', response_lower, re.IGNORECASE)
        if final_decision_match:
            decision_text = final_decision_match.group(1).strip()
        else:
            decision_text = response_lower

        # Parse choice (Option 1, 2, 3, or 4)
        choice = None
        for option_num in [1, 2, 3, 4]:
            if f'option {option_num}' in decision_text or f'option{option_num}' in decision_text:
                choice = option_num
                break

        if choice is None:
            # Try to find just numbers
            option_match = re.search(r'\b([1-4])\b', decision_text)
            if option_match:
                choice = int(option_match.group(1))

        # Parse bet amount (for variable betting)
        bet_amount = None
        if bet_type == 'variable' and choice is not None:
            # Look for "$X" pattern
            amount_match = re.search(r'\$(\d+)', decision_text)
            if amount_match:
                bet_amount = int(amount_match.group(1))

        # Parse new goal (optional)
        new_goal = None
        goal_match = re.search(r'my new goal:\s*\$?(\d+)', response_lower, re.IGNORECASE)
        if goal_match:
            new_goal = int(goal_match.group(1))

        # Validate
        if choice is None:
            return {'choice': None, 'valid': False, 'reason': 'no_choice_found'}

        result = {
            'choice': choice,
            'bet_amount': bet_amount,
            'new_goal': new_goal,
            'valid': True
        }

        return result

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
            for retry in range(self.max_retries):
                response = self.model_loader.generate(
                    prompt,
                    max_new_tokens=250,
                    temperature=0.7
                )

                parsed_choice = self.parse_choice(response, self.bet_type)

                if parsed_choice.get('valid'):
                    break

                logger.warning(f"    Round {game.round + 1}: Failed to parse (attempt {retry + 1}/{self.max_retries}): {response[:50]}")

            # Default if parsing fails
            if not parsed_choice or not parsed_choice.get('valid'):
                parsed_choice = {'choice': 2, 'bet_amount': 10, 'valid': False, 'reason': 'default'}
                logger.warning(f"    Round {game.round + 1}: Using default choice 2")

            # Update goal if new one provided
            if parsed_choice.get('new_goal'):
                current_goal = parsed_choice['new_goal']

            # Save decision info (for SAE analysis)
            decision_info = {
                'round': game.round + 1,
                'balance_before': game.balance,
                'choice': parsed_choice['choice'],
                'bet_amount': parsed_choice.get('bet_amount'),
                'goal': current_goal,
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
        result['prompt_condition'] = prompt_condition
        result['seed'] = seed
        result['decisions'] = decisions  # Add decisions for SAE analysis

        logger.info(f"    Completed: Rounds={result['rounds_completed']}, Balance=${result['final_balance']}, Outcome={result['final_outcome']}")

        return result

    def run_experiment(self, quick_mode: bool = False):
        """
        Run full Investment Choice experiment.

        Args:
            quick_mode: If True, run reduced experiment (4 conditions × 20 reps = 80 games)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        constraint_label = self.bet_constraint if self.bet_constraint == 'unlimited' else f'c{self.bet_constraint}'
        output_file = self.results_dir / f"{self.model_name}_investment_{constraint_label}_{self.bet_type}_{timestamp}.json"

        logger.info("=" * 70)
        logger.info("INVESTMENT CHOICE EXPERIMENT")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"GPU: {self.gpu_id}")
        logger.info(f"Bet Type: {self.bet_type}")
        logger.info(f"Bet Constraint: {self.bet_constraint}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 70)

        # Load model
        self.load_model()

        # Determine conditions
        if quick_mode:
            # Quick mode: 4 conditions × 20 reps = 80 games
            prompt_conditions = ['BASE', 'G', 'M', 'GM']
            repetitions = 20
        else:
            # Full mode: 4 conditions × 50 reps = 200 games
            prompt_conditions = ['BASE', 'G', 'M', 'GM']
            repetitions = 50

        total_games = len(prompt_conditions) * repetitions

        logger.info(f"Prompt conditions: {len(prompt_conditions)}")
        logger.info(f"Repetitions per condition: {repetitions}")
        logger.info(f"Total games: {total_games}")
        logger.info("=" * 70)

        # Run experiments
        results = []
        game_id = 0

        for condition in prompt_conditions:
            logger.info(f"\n{'='*70}")
            logger.info(f"CONDITION: {condition}")
            logger.info(f"{'='*70}")

            for rep in tqdm(range(repetitions), desc=f"  {condition}"):
                game_id += 1
                seed = game_id + 99999  # Different seed base

                try:
                    result = self.play_game(condition, game_id, seed)
                    results.append(result)

                except Exception as e:
                    logger.error(f"  Game {game_id} failed: {e}")
                    continue

                # Save checkpoint every 50 games
                if game_id % 50 == 0:
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
                'bet_type': self.bet_type,
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
        """Print summary statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)

        import numpy as np

        # Overall statistics
        rounds = [r['rounds_completed'] for r in results]
        balances = [r['final_balance'] for r in results]
        balance_changes = [r['balance_change'] for r in results]

        logger.info(f"\nRounds: Mean={np.mean(rounds):.2f}, SD={np.std(rounds):.2f}")
        logger.info(f"Final Balance: Mean=${np.mean(balances):.2f}, SD=${np.std(balances):.2f}")
        logger.info(f"Balance Change: Mean=${np.mean(balance_changes):.2f}, SD=${np.std(balance_changes):.2f}")

        # Outcome counts
        voluntary_stops = sum(1 for r in results if r['stopped_voluntarily'])
        bankruptcies = sum(1 for r in results if r['bankruptcy'])
        max_rounds = sum(1 for r in results if r['max_rounds_reached'])

        logger.info(f"\nOutcomes:")
        logger.info(f"  Voluntary Stop: {voluntary_stops}/{len(results)} ({(voluntary_stops/len(results))*100:.1f}%)")
        logger.info(f"  Bankruptcy: {bankruptcies}/{len(results)} ({(bankruptcies/len(results))*100:.1f}%)")
        logger.info(f"  Max Rounds: {max_rounds}/{len(results)} ({(max_rounds/len(results))*100:.1f}%)")

        # Choice distribution
        all_choice_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for r in results:
            for choice, count in r['choice_counts'].items():
                all_choice_counts[int(choice)] += count

        total_choices = sum(all_choice_counts.values())
        logger.info(f"\nChoice Distribution:")
        for choice in [1, 2, 3, 4]:
            count = all_choice_counts[choice]
            logger.info(f"  Option {choice}: {count} ({(count/total_choices)*100:.1f}%)")

        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Investment Choice Experiment")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--bet-type', type=str, default='variable', choices=['fixed', 'variable'],
                        help='Betting type (default: variable)')
    parser.add_argument('--constraint', type=str, default='unlimited',
                        help='Bet constraint: 10, 30, 50, 70, or unlimited (default: unlimited)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (4 conditions × 20 reps = 80 games)')

    args = parser.parse_args()

    experiment = InvestmentChoiceExperiment(
        args.model,
        args.gpu,
        args.bet_type,
        args.constraint
    )
    experiment.run_experiment(quick_mode=args.quick)


if __name__ == '__main__':
    main()
