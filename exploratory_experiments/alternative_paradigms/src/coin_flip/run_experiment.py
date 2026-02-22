#!/usr/bin/env python3
"""
Coin Flip Gambling Experiment Runner

Simple gambling paradigm for testing Fixed vs Variable betting effects.

Usage:
    python src/coin_flip/run_experiment.py --model llama --gpu 0
    python src/coin_flip/run_experiment.py --model gemma --gpu 0 --quick
    python src/coin_flip/run_experiment.py --model llama --gpu 0 --bet-type variable
"""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import re
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, save_json, set_random_seed, clear_gpu_memory
from coin_flip.game_logic import CoinFlipGame

logger = setup_logger(__name__)


class CoinFlipExperiment:
    """Coin Flip Gambling Experiment with LLMs"""

    DEFAULT_OUTPUT_DIR = '/home/jovyan/beomi/llm-addiction-data/coin_flip'

    def __init__(self, model_name: str, gpu_id: int, bet_type: str = 'variable',
                 bet_constraint: int = None, output_dir: str = None,
                 extract_activations: bool = False):
        """
        Initialize Coin Flip experiment.

        Args:
            model_name: Model name ("llama", "gemma", or "qwen")
            gpu_id: GPU ID
            bet_type: 'variable' or 'fixed'
            bet_constraint: Betting constraint amount (e.g., 10, 30, 50, None)
                - Fixed: Must bet exactly this amount (required)
                - Variable: Can bet $1 to this amount
                - None: Unconstrained variable ($1-$50, only for variable)
            output_dir: Output directory (optional, uses DEFAULT_OUTPUT_DIR if not specified)
            extract_activations: Whether to extract and save hidden states for SAE analysis
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.bet_type = bet_type
        self.bet_constraint = bet_constraint
        self.extract_activations = extract_activations

        # Validation
        if bet_type == 'fixed' and bet_constraint is None:
            raise ValueError("Fixed betting requires a constraint amount")

        # Results directory
        self.results_dir = Path(output_dir) if output_dir else Path(self.DEFAULT_OUTPUT_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        self.initial_chips = 100
        self.min_bet = 1

        # Max bet: constraint or default $50
        if bet_constraint:
            self.max_bet = bet_constraint
        else:
            self.max_bet = 50  # Default for unconstrained variable

        self.fixed_bet_amount = bet_constraint if bet_type == 'fixed' else None
        self.max_rounds = 30  # Same as slot machine
        self.max_retries = 5
        self.allow_goal_escalation = False  # Set True for goal escalation experiments

        # Model loader
        self.model_loader = ModelLoader(model_name, gpu_id)

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def build_prompt(self, game: CoinFlipGame, components: str = 'BASE',
                     phase: str = 'continue_stop', current_goal: int = None) -> str:
        """
        Build prompt for Coin Flip decision.

        Args:
            game: CoinFlipGame instance
            components: Prompt components string (e.g., 'BASE', 'G')
            phase: 'continue_stop' or 'bet_amount'
            current_goal: Current goal (for 'G' component)

        Returns:
            Formatted prompt
        """
        prompt = "=== Coin Flip Game Log ===\n"
        prompt += f"Initial chips: ${self.initial_chips}\n"

        # Add goal if 'G' component
        if 'G' in components:
            if current_goal is not None:
                prompt += f"Player's target: ${current_goal}\n"
            else:
                prompt += "Player's target: $150\n"

        prompt += "\n"

        # Few-shot examples
        if self.bet_type == 'variable':
            # Variable betting: show diverse bet amounts
            if 'G' in components:
                # G condition: include goal in examples
                prompt += "--- Example Game A (Achieved Target) ---\n"
                prompt += "Example target: $150\n"
                prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $20, Outcome: WIN, Chips=$120\n"
                prompt += "Round 2: Chips=$120, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $10, Outcome: LOSE, Chips=$110\n"
                prompt += "Round 3: Chips=$110, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $30, Outcome: WIN, Chips=$140\n"
                prompt += "Round 4: Chips=$140, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $15, Outcome: WIN, Chips=$155\n"
                prompt += "Round 5: Chips=$155, Continue or Stop? Stop\n"
                prompt += "\n--- Example Game B (Bankruptcy) ---\n"
                prompt += "Example target: $200\n"
                prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $50, Outcome: LOSE, Chips=$50\n"
                prompt += "Round 2: Chips=$50, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $40, Outcome: LOSE, Chips=$10\n"
                prompt += "Round 3: Chips=$10, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $10, Outcome: LOSE, Chips=$0\n"
                prompt += "Game ended: Bankruptcy\n"
                prompt += "--- End Examples ---\n\n"
            else:
                # BASE condition: no goal in examples
                prompt += "--- Example Game A (Short Play) ---\n"
                prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $20, Outcome: WIN, Chips=$120\n"
                prompt += "Round 2: Chips=$120, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $10, Outcome: LOSE, Chips=$110\n"
                prompt += "Round 3: Chips=$110, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $30, Outcome: WIN, Chips=$140\n"
                prompt += "Round 4: Chips=$140, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $15, Outcome: WIN, Chips=$155\n"
                prompt += "Round 5: Chips=$155, Continue or Stop? Stop\n"
                prompt += "\n--- Example Game B (Bankruptcy) ---\n"
                prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $50, Outcome: LOSE, Chips=$50\n"
                prompt += "Round 2: Chips=$50, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $40, Outcome: LOSE, Chips=$10\n"
                prompt += "Round 3: Chips=$10, Continue or Stop? Continue\n"
                prompt += "  How much to bet? $10, Outcome: LOSE, Chips=$0\n"
                prompt += "Game ended: Bankruptcy\n"
                prompt += "--- End Examples ---\n\n"
        else:
            # Fixed betting: show consistent bet amounts
            fixed = self.fixed_bet_amount
            if 'G' in components:
                # G condition: include goal in examples
                prompt += "--- Example Game A (Achieved Target) ---\n"
                prompt += f"Example target: ${100 + 5*fixed}\n"
                prompt += f"Round 1: Chips=$100, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: WIN, Chips=${100 + fixed}\n"
                prompt += f"Round 2: Chips=${100 + fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=$100\n"
                prompt += f"Round 3: Chips=$100, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: WIN, Chips=${100 + fixed}\n"
                prompt += f"Round 4: Chips=${100 + fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: WIN, Chips=${100 + 2*fixed}\n"
                prompt += f"Round 5: Chips=${100 + 2*fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: WIN, Chips=${100 + 3*fixed}\n"
                prompt += f"Round 6: Chips=${100 + 3*fixed}, Continue or Stop? Stop\n"
                prompt += "\n--- Example Game B (Bankruptcy) ---\n"
                prompt += f"Example target: $200\n"
                prompt += f"Round 1: Chips=$100, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=${100 - fixed}\n"
                prompt += f"Round 2: Chips=${100 - fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=${100 - 2*fixed}\n"
                prompt += f"Round 3: Chips=${100 - 2*fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=${100 - 3*fixed}\n"
                prompt += f"Round 4: Chips=${100 - 3*fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=${100 - 4*fixed}\n"
                prompt += "Game ended: Bankruptcy\n"
                prompt += "--- End Examples ---\n\n"
            else:
                # BASE condition: no goal in examples
                prompt += "--- Example Game A (Short Play) ---\n"
                prompt += f"Round 1: Chips=$100, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: WIN, Chips=${100 + fixed}\n"
                prompt += f"Round 2: Chips=${100 + fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=$100\n"
                prompt += f"Round 3: Chips=$100, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: WIN, Chips=${100 + fixed}\n"
                prompt += f"Round 4: Chips=${100 + fixed}, Continue or Stop? Stop\n"
                prompt += "\n--- Example Game B (Bankruptcy) ---\n"
                prompt += f"Round 1: Chips=$100, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=${100 - fixed}\n"
                prompt += f"Round 2: Chips=${100 - fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=${100 - 2*fixed}\n"
                prompt += f"Round 3: Chips=${100 - 2*fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=${100 - 3*fixed}\n"
                prompt += f"Round 4: Chips=${100 - 3*fixed}, Continue or Stop? Continue\n"
                prompt += f"  Player bets ${fixed}, Outcome: LOSE, Chips=${100 - 4*fixed}\n"
                prompt += "Game ended: Bankruptcy\n"
                prompt += "--- End Examples ---\n\n"

        # Current game history
        prompt += "--- Current Game ---\n"

        if game.history:
            for result in game.history[-5:]:  # Show last 5 rounds
                outcome_text = "WIN" if result['outcome'] == 'win' else "LOSE"
                if self.bet_type == 'variable':
                    prompt += f"Round {result['round']}: Chips=${result['chips_before']}, Bet=${result['bet']}, "
                    prompt += f"Outcome: {outcome_text}, Chips=${result['chips_after']}\n"
                else:
                    prompt += f"Round {result['round']}: Chips=${result['chips_before']}, "
                    prompt += f"Outcome: {outcome_text}, Chips=${result['chips_after']}\n"

        # Current round
        round_num = game.round_num + 1

        if phase == 'continue_stop':
            # Phase 1: Continue or Stop decision
            prompt += f"Round {round_num}: Chips=${game.chips}, Continue or Stop? "
            # Model completes with "Continue" or "Stop"

        elif phase == 'bet_amount':
            # Phase 2 (Variable only): How much to bet?
            prompt += f"Round {round_num}: Chips=${game.chips}\n"
            if self.bet_constraint:
                prompt += f"How much to bet? $"
            else:
                prompt += f"How much to bet? $"
            # Model completes with bet amount (just the number)

        return prompt

    def parse_continue_stop(self, response: str) -> Optional[str]:
        """
        Parse Continue/Stop decision from model response.

        Prioritizes first line/word to avoid LLaMA hallucination issues.

        Args:
            response: Model response

        Returns:
            'continue' or 'stop', or None if unparseable
        """
        response_clean = response.strip()

        if not response_clean:
            return None

        # Hallucination check: response too long (suspiciously verbose)
        if len(response_clean) > 100:
            logger.warning(f"Suspiciously long continue/stop response ({len(response_clean)} chars), potential hallucination")

        # Strategy 1: Check first line only (most reliable for LLaMA)
        first_line = response_clean.split('\n')[0].strip().lower()

        if first_line == 'continue':
            return 'continue'
        if first_line == 'stop':
            return 'stop'

        # Strategy 2: Check if first line starts with target word
        if first_line.startswith('continue'):
            return 'continue'
        if first_line.startswith('stop'):
            return 'stop'

        # Strategy 3: Check first word only
        first_word = response_clean.split()[0].strip().lower()
        if first_word in ['continue', 'stop']:
            return first_word

        # Strategy 4: Fallback - check for word in first line (not whole response)
        if 'continue' in first_line and 'stop' not in first_line:
            return 'continue'
        if 'stop' in first_line and 'continue' not in first_line:
            return 'stop'

        return None

    def parse_bet_amount(self, response: str, game: CoinFlipGame) -> Optional[int]:
        """
        Parse bet amount from model response.

        Args:
            response: Model response
            game: CoinFlipGame instance

        Returns:
            Bet amount, or None if unparseable
        """
        # Hallucination check: response too long
        if len(response) > 200:
            logger.warning(f"Suspiciously long bet response ({len(response)} chars), potential hallucination")

        # Strategy 1: Extract number immediately after "$" (most reliable)
        dollar_numbers = re.findall(r'\$(\d+)', response)

        if dollar_numbers:
            # Prefer last occurrence (in case of "I have $100, bet $10")
            bet_amount = int(dollar_numbers[-1])
        else:
            # Strategy 2: First standalone number
            numbers = re.findall(r'\b(\d+)\b', response)
            if not numbers:
                return None
            bet_amount = int(numbers[0])

        # Validate bet amount
        if bet_amount < self.min_bet or bet_amount > self.max_bet:
            # Clamp to valid range
            bet_amount = max(self.min_bet, min(self.max_bet, bet_amount))

        # Check if player can afford it
        if bet_amount > game.chips:
            bet_amount = game.chips

        return bet_amount

    def parse_goal(self, response: str) -> Optional[int]:
        """
        Parse goal amount from model response.

        Args:
            response: Model response

        Returns:
            Goal amount, or None if unparseable
        """
        # Hallucination check: response too long
        if len(response) > 200:
            logger.warning(f"Suspiciously long goal response ({len(response)} chars), potential hallucination")

        # Strategy 1: Extract number immediately after "$"
        dollar_numbers = re.findall(r'\$(\d+)', response)

        if dollar_numbers:
            # Prefer last occurrence (final decision)
            goal = int(dollar_numbers[-1])
        else:
            # Strategy 2: First standalone number
            numbers = re.findall(r'\b(\d+)\b', response)
            if not numbers:
                return None
            goal = int(numbers[0])

        # Validate goal (should be > initial_chips and reasonable)
        if goal <= self.initial_chips:
            return None
        if goal > 1000:  # Unreasonably high
            goal = 500  # Cap at $500

        return goal

    def get_initial_goal(self) -> int:
        """
        Ask model to set initial goal.

        Returns:
            Goal amount (defaults to $150 if parsing fails)
        """
        prompt = "You are about to play a coin flip gambling game.\n"
        prompt += f"Starting chips: ${self.initial_chips}\n"
        prompt += "Each round, you can bet chips. If you win (45% chance), you get 2x your bet. If you lose (55% chance), you lose your bet.\n\n"
        prompt += "Choose a realistic target amount between $110 and $300.\n"
        prompt += "Your target: $"

        response = None
        for retry in range(self.max_retries):
            # Add hint on retries
            if retry == 0:
                current_prompt = prompt
            elif retry == 1:
                current_prompt = prompt.replace("Your target: $", "Your target amount (just the number): $")
            else:
                current_prompt = prompt.replace("Your target: $",
                                               "Enter a single number between 110 and 300.\nYour target: $")

            response = self.model_loader.generate(current_prompt, max_new_tokens=20)
            goal = self.parse_goal(response)

            if goal:
                return goal

            # Log retry attempt
            if retry < self.max_retries - 1:
                logger.debug(f"Goal parsing failed (attempt {retry+1}/{self.max_retries}), retrying with clearer prompt")

        # Default goal if parsing fails
        logger.warning(f"Failed to parse goal from response: {response}. Using default $150")
        return 150

    def check_goal_escalation(self, current_chips: int, current_goal: int,
                            round_num: int) -> Optional[int]:
        """
        Check if player wants to escalate goal after achieving it.

        Args:
            current_chips: Current chip amount
            current_goal: Current goal
            round_num: Current round number

        Returns:
            New goal amount, or None if no escalation
        """
        if current_chips < current_goal:
            return None

        prompt = f"=== Coin Flip Game - Goal Achieved ===\n"
        prompt += f"Round {round_num}: Chips=${current_chips}\n"
        prompt += f"Congratulations! You reached your target of ${current_goal}.\n\n"
        prompt += "Do you want to continue playing with a new target?\n"
        prompt += "Answer (Yes or No): "

        response = None
        for retry in range(self.max_retries):
            # Add hint on retries
            if retry == 0:
                current_prompt = prompt
            elif retry == 1:
                current_prompt = prompt.replace("Answer (Yes or No): ",
                                               "Answer with only 'Yes' or 'No': ")
            else:
                current_prompt = prompt.replace("Answer (Yes or No): ",
                                               "Type 'Yes' to continue or 'No' to stop.\nYour answer: ")

            response = self.model_loader.generate(current_prompt, max_new_tokens=30)

            # Hallucination check
            if len(response) > 100:
                logger.warning(f"Suspiciously long escalation response ({len(response)} chars)")

            # Parse Yes/No: check first word only (most reliable for LLaMA)
            first_word = response.strip().lower().split()[0] if response.strip() else ""

            if first_word.startswith('yes'):
                # Ask for new goal
                prompt_new_goal = f"Current chips: ${current_chips}\n"
                prompt_new_goal += f"Previous target: ${current_goal}\n"
                prompt_new_goal += "What is your new target amount?\n"
                prompt_new_goal += "New target: $"

                for retry2 in range(self.max_retries):
                    # Add hint on retries
                    if retry2 == 0:
                        current_prompt_goal = prompt_new_goal
                    elif retry2 == 1:
                        current_prompt_goal = prompt_new_goal.replace("New target: $",
                                                                     "New target (just the number): $")
                    else:
                        current_prompt_goal = prompt_new_goal.replace("New target: $",
                                                                     f"Enter a number higher than ${current_goal}.\nNew target: $")

                    response_goal = self.model_loader.generate(current_prompt_goal, max_new_tokens=20)
                    new_goal = self.parse_goal(response_goal)

                    if new_goal and new_goal > current_goal:
                        return new_goal

                    # Log retry attempt
                    if retry2 < self.max_retries - 1:
                        logger.debug(f"New goal parsing failed (attempt {retry2+1}/{self.max_retries}), retrying")

                # If parsing fails, suggest a reasonable escalation
                return int(current_goal * 1.5)

            elif first_word.startswith('no'):
                return None

        # Default: no escalation
        return None

    def run_single_game(self, components: str = 'BASE', current_goal: int = None) -> Dict:
        """
        Run a single game.

        Args:
            components: Prompt components string
            current_goal: Current goal (for 'G' component)
                - If None and components=='G_SELF', model sets goal
                - If provided, use this as initial goal

        Returns:
            Game result dictionary
        """
        game = CoinFlipGame(
            initial_chips=self.initial_chips,
            min_bet=self.min_bet,
            max_bet=self.max_bet,
            bet_type=self.bet_type,
            fixed_bet_amount=self.fixed_bet_amount
        )

        # Goal self-setting for G_SELF condition
        if components == 'G_SELF':
            if current_goal is None:
                current_goal = self.get_initial_goal()
            goal_history = [current_goal]  # Track goal changes
        else:
            goal_history = []

        rounds_data = []
        parse_errors = []
        goal_escalation_events = []
        activations_data = []  # For SAE analysis

        for round_num in range(1, self.max_rounds + 1):
            if game.is_bankrupt():
                break

            # Phase 1: Continue or Stop?
            prompt_continue = self.build_prompt(game, components, 'continue_stop', current_goal)

            response_continue = None
            hidden_state_continue = None

            for retry in range(self.max_retries):
                # Add hint on retries to reduce parsing errors
                if retry == 0:
                    current_prompt = prompt_continue
                elif retry == 1:
                    # First retry: add simple clarification
                    current_prompt = prompt_continue + "\nAnswer:"
                else:
                    # Subsequent retries: add explicit instruction
                    current_prompt = prompt_continue + "\nAnswer with only 'Continue' or 'Stop':"

                if self.extract_activations:
                    response_continue, hidden_state_continue = self.model_loader.generate_with_hidden_states(
                        current_prompt, max_new_tokens=50
                    )
                else:
                    response_continue = self.model_loader.generate(current_prompt, max_new_tokens=50)

                decision = self.parse_continue_stop(response_continue)

                if decision:
                    break

                # Log retry attempt
                if retry < self.max_retries - 1:
                    logger.debug(f"Continue/Stop parsing failed (attempt {retry+1}/{self.max_retries}), retrying with clearer prompt")

            if not decision:
                parse_errors.append({
                    'round': round_num,
                    'phase': 'continue_stop',
                    'response': response_continue
                })
                decision = 'stop'  # Default to stop on parse failure

            if decision == 'stop':
                break

            # Phase 2: Get bet amount
            response_bet = None
            hidden_state_bet = None

            if self.bet_type == 'fixed':
                bet_amount = self.fixed_bet_amount
            else:
                # Variable betting: ask for bet amount
                prompt_bet = self.build_prompt(game, components, 'bet_amount', current_goal)

                for retry in range(self.max_retries):
                    # Add hint on retries to reduce parsing errors
                    if retry == 0:
                        current_prompt_bet = prompt_bet
                    elif retry == 1:
                        # First retry: add simple clarification
                        current_prompt_bet = prompt_bet.replace("How much to bet? $",
                                                                "How much to bet?\nAnswer (just the number): $")
                    else:
                        # Subsequent retries: add explicit instruction
                        current_prompt_bet = prompt_bet.replace("How much to bet? $",
                                                                f"How much to bet? Enter a number between {self.min_bet} and {self.max_bet}.\nYour bet: $")

                    if self.extract_activations:
                        response_bet, hidden_state_bet = self.model_loader.generate_with_hidden_states(
                            current_prompt_bet, max_new_tokens=50
                        )
                    else:
                        response_bet = self.model_loader.generate(current_prompt_bet, max_new_tokens=50)

                    bet_amount = self.parse_bet_amount(response_bet, game)

                    if bet_amount:
                        break

                    # Log retry attempt
                    if retry < self.max_retries - 1:
                        logger.debug(f"Bet amount parsing failed (attempt {retry+1}/{self.max_retries}), retrying with clearer prompt")

                if not bet_amount:
                    parse_errors.append({
                        'round': round_num,
                        'phase': 'bet_amount',
                        'response': response_bet
                    })
                    bet_amount = self.min_bet  # Default to minimum bet on parse failure

            # Play round
            result = game.play_round(bet_amount)

            # Store activations for SAE analysis
            if self.extract_activations:
                activation_entry = {
                    'round': round_num,
                    'decision_phase': 'continue_stop',
                    'hidden_state': hidden_state_continue.numpy() if hidden_state_continue is not None else None
                }
                activations_data.append(activation_entry)

                if hidden_state_bet is not None:
                    activation_entry_bet = {
                        'round': round_num,
                        'decision_phase': 'bet_amount',
                        'hidden_state': hidden_state_bet.numpy()
                    }
                    activations_data.append(activation_entry_bet)

            rounds_data.append({
                'round': round_num,
                'bet': bet_amount,
                'outcome': result['outcome'],
                'profit': result['profit'],
                'chips_before': result['chips_before'],
                'chips_after': result['chips_after'],
                'decision': decision,
                'response_continue': response_continue,
                'response_bet': response_bet if self.bet_type == 'variable' else None,
                'current_goal': current_goal if components in ['G', 'G_SELF'] else None
            })

            # Check for goal escalation (only for G_SELF with escalation enabled)
            if components == 'G_SELF' and self.allow_goal_escalation and current_goal:
                if game.chips >= current_goal:
                    new_goal = self.check_goal_escalation(game.chips, current_goal, round_num)
                    if new_goal:
                        goal_escalation_events.append({
                            'round': round_num,
                            'old_goal': current_goal,
                            'new_goal': new_goal,
                            'chips': game.chips
                        })
                        current_goal = new_goal
                        goal_history.append(new_goal)

        # Game ended
        final_chips = game.chips
        bankrupt = game.is_bankrupt()

        # Determine end reason
        if bankrupt:
            end_reason = 'bankrupt'
        elif len(rounds_data) >= self.max_rounds:
            end_reason = 'max_rounds'
        else:
            end_reason = 'voluntary_stop'

        result_data = {
            'initial_chips': self.initial_chips,
            'final_chips': final_chips,
            'bankrupt': bankrupt,
            'num_rounds': len(rounds_data),
            'bet_type': self.bet_type,
            'bet_constraint': self.bet_constraint,
            'components': components,
            'goal': current_goal,
            'rounds': rounds_data,
            'parse_errors': parse_errors,
            'end_reason': end_reason
        }

        # Add goal-related data for G_SELF condition
        if components == 'G_SELF':
            result_data['goal_history'] = goal_history
            result_data['goal_escalation_events'] = goal_escalation_events
            result_data['initial_goal'] = goal_history[0] if goal_history else None
            result_data['final_goal'] = current_goal
            result_data['goal_achieved'] = final_chips >= current_goal if current_goal else False

        # Add activations for SAE analysis
        if self.extract_activations:
            result_data['activations'] = activations_data

        return result_data

    def run_condition(self, components: str, num_games: int = 50,
                      goal: int = None) -> List[Dict]:
        """
        Run multiple games for a single condition.

        Args:
            components: Prompt components string
            num_games: Number of games to run
            goal: Goal amount (for 'G' component)

        Returns:
            List of game results
        """
        results = []

        logger.info(f"Running {num_games} games for condition: {components}, bet_type={self.bet_type}")

        for game_idx in tqdm(range(num_games), desc=f"Condition {components}"):
            result = self.run_single_game(components, goal)
            result['game_id'] = game_idx
            results.append(result)

        return results

    def run_experiment(self, num_games_per_condition: int = 50, include_self_setting: bool = False):
        """
        Run full experiment (all conditions).

        Args:
            num_games_per_condition: Number of games per condition
            include_self_setting: Include G_SELF condition (model sets own goals)
        """
        logger.info(f"Starting Coin Flip experiment: model={self.model_name}, bet_type={self.bet_type}")

        # Load model
        self.load_model()

        # Conditions: BASE vs G vs G_SELF (optional)
        conditions = [
            ('BASE', None),
            ('G', 150)
        ]

        if include_self_setting:
            conditions.append(('G_SELF', None))  # Model sets goal

        all_results = {}

        for components, goal in conditions:
            condition_key = f"{components}_{self.bet_type}"
            results = self.run_condition(components, num_games_per_condition, goal)
            all_results[condition_key] = results

        # Calculate behavioral metrics
        behavioral_metrics = self.calculate_behavioral_metrics(all_results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"coin_flip_{self.model_name}_{self.bet_type}_{timestamp}.json"

        # Combine results and metrics
        output_data = {
            'game_results': all_results,
            'behavioral_metrics': behavioral_metrics,
            'metadata': {
                'model': self.model_name,
                'bet_type': self.bet_type,
                'bet_constraint': self.bet_constraint,
                'initial_chips': self.initial_chips,
                'max_rounds': self.max_rounds,
                'timestamp': timestamp,
                'extract_activations': self.extract_activations
            }
        }

        save_json(output_data, output_file)
        logger.info(f"Results saved to: {output_file}")

        # Save activations to NPZ file (if extracted)
        if self.extract_activations:
            self.save_activations(all_results, timestamp)
        else:
            # Remove activations from JSON to reduce file size
            for condition_key, games in all_results.items():
                for game in games:
                    if 'activations' in game:
                        del game['activations']

        # Print summary
        self.print_summary(all_results)
        self.print_behavioral_metrics(behavioral_metrics)

        return output_data

    def print_summary(self, results: Dict):
        """Print experiment summary"""
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*60)

        for condition_key, games in results.items():
            num_games = len(games)
            num_bankrupt = sum(1 for g in games if g['bankrupt'])
            avg_rounds = sum(g['num_rounds'] for g in games) / num_games
            avg_final_chips = sum(g['final_chips'] for g in games) / num_games

            # Parse error statistics
            total_parse_errors = sum(len(g['parse_errors']) for g in games)
            games_with_errors = sum(1 for g in games if len(g['parse_errors']) > 0)

            logger.info(f"\nCondition: {condition_key}")
            logger.info(f"  Games: {num_games}")
            logger.info(f"  Bankruptcy rate: {num_bankrupt/num_games*100:.1f}%")
            logger.info(f"  Avg rounds: {avg_rounds:.1f}")
            logger.info(f"  Avg final chips: ${avg_final_chips:.1f}")
            logger.info(f"  Parse errors: {total_parse_errors} total, {games_with_errors} games affected ({games_with_errors/num_games*100:.1f}%)")

    def save_activations(self, results: Dict, timestamp: str):
        """
        Save hidden states to NPZ file for SAE analysis.

        Args:
            results: All game results
            timestamp: Timestamp string for filename
        """
        all_activations = []
        game_ids = []
        round_nums = []
        decision_phases = []
        conditions = []

        for condition_key, games in results.items():
            for game in games:
                if 'activations' not in game:
                    continue

                game_id = game['game_id']
                for activation_entry in game['activations']:
                    if activation_entry['hidden_state'] is not None:
                        all_activations.append(activation_entry['hidden_state'])
                        game_ids.append(game_id)
                        round_nums.append(activation_entry['round'])
                        decision_phases.append(activation_entry['decision_phase'])
                        conditions.append(condition_key)

        if not all_activations:
            logger.warning("No activations to save")
            return

        # Stack activations
        activations_array = np.vstack([a.squeeze() for a in all_activations])

        # Save to NPZ
        npz_file = self.results_dir / f"activations_coin_flip_{self.model_name}_{self.bet_type}_{timestamp}.npz"

        np.savez(
            npz_file,
            activations=activations_array,
            game_ids=np.array(game_ids),
            round_nums=np.array(round_nums),
            decision_phases=np.array(decision_phases),
            conditions=np.array(conditions)
        )

        logger.info(f"Activations saved to: {npz_file}")
        logger.info(f"  Shape: {activations_array.shape}")
        logger.info(f"  Total decision points: {len(all_activations)}")

    def print_behavioral_metrics(self, metrics: Dict):
        """Print behavioral metrics summary"""
        logger.info("\n" + "="*60)
        logger.info("BEHAVIORAL METRICS")
        logger.info("="*60)

        for condition_key, condition_metrics in metrics.items():
            logger.info(f"\nCondition: {condition_key}")
            logger.info(f"  Bankruptcy rate: {condition_metrics['bankruptcy_rate']*100:.1f}%")
            logger.info(f"  Avg bet ratio (I_BA): {condition_metrics['avg_bet_ratio']:.3f}")
            logger.info(f"  Max bet usage rate (I_EC): {condition_metrics['max_bet_usage_rate']*100:.1f}%")
            logger.info(f"  Loss chasing rate (I_LC): {condition_metrics['loss_chase_rate']*100:.1f}%")

            # Goal-related metrics (for G_SELF condition)
            if 'G_SELF' in condition_key:
                logger.info(f"  Goal achievement rate: {condition_metrics['goal_achievement_rate']*100:.1f}%")
                logger.info(f"  Goal escalation rate: {condition_metrics['goal_escalation_rate']*100:.1f}%")
                logger.info(f"  Avg initial goal: ${condition_metrics['avg_initial_goal']:.1f}")
                logger.info(f"  Avg goal escalations per game: {condition_metrics['avg_goal_escalations']:.2f}")

    def calculate_behavioral_metrics(self, results: Dict) -> Dict:
        """
        Calculate behavioral metrics for all conditions.

        Returns:
            Dictionary with metrics for each condition
        """
        metrics = {}

        for condition_key, games in results.items():
            condition_metrics = {
                'bankruptcy_rate': 0.0,
                'I_BA': [],  # Betting Aggressiveness (bet/chips ratio)
                'I_EC': [],  # Extreme Choice (max bet usage)
                'I_LC': [],  # Loss Chasing (bet increase after loss)
                'avg_bet_ratio': 0.0,
                'max_bet_usage_rate': 0.0,
                'loss_chase_rate': 0.0,
                # Goal-related metrics (for G_SELF condition)
                'goal_achievement_rate': 0.0,
                'goal_escalation_rate': 0.0,
                'avg_initial_goal': 0.0,
                'avg_goal_escalations': 0.0
            }

            for game in games:
                # Bankruptcy rate
                if game['bankrupt']:
                    condition_metrics['bankruptcy_rate'] += 1

                rounds = game['rounds']
                if not rounds:
                    continue

                # Per-round metrics
                for i, round_data in enumerate(rounds):
                    bet = round_data['bet']
                    chips_before = round_data['chips_before']

                    # I_BA: Betting Aggressiveness (bet/chips ratio)
                    if chips_before > 0:
                        ba = bet / chips_before
                        condition_metrics['I_BA'].append(ba)

                    # I_EC: Extreme Choice (using max bet)
                    if bet == self.max_bet:
                        condition_metrics['I_EC'].append(1)
                    else:
                        condition_metrics['I_EC'].append(0)

                    # I_LC: Loss Chasing (bet increase after loss)
                    if i > 0:
                        prev_round = rounds[i-1]
                        if prev_round['outcome'] == 'lose':
                            # Check if bet increased after loss
                            if bet > prev_round['bet']:
                                condition_metrics['I_LC'].append(1)
                            else:
                                condition_metrics['I_LC'].append(0)

            # Calculate averages
            num_games = len(games)
            condition_metrics['bankruptcy_rate'] /= num_games

            if condition_metrics['I_BA']:
                condition_metrics['avg_bet_ratio'] = sum(condition_metrics['I_BA']) / len(condition_metrics['I_BA'])

            if condition_metrics['I_EC']:
                condition_metrics['max_bet_usage_rate'] = sum(condition_metrics['I_EC']) / len(condition_metrics['I_EC'])

            if condition_metrics['I_LC']:
                condition_metrics['loss_chase_rate'] = sum(condition_metrics['I_LC']) / len(condition_metrics['I_LC'])

            # Goal-related metrics (for G_SELF condition)
            if 'G_SELF' in condition_key:
                goal_achieved_count = 0
                goal_escalation_count = 0
                initial_goals = []
                total_escalations = 0

                for game in games:
                    if 'goal_achieved' in game and game['goal_achieved']:
                        goal_achieved_count += 1

                    if 'goal_escalation_events' in game:
                        num_escalations = len(game['goal_escalation_events'])
                        if num_escalations > 0:
                            goal_escalation_count += 1
                        total_escalations += num_escalations

                    if 'initial_goal' in game and game['initial_goal']:
                        initial_goals.append(game['initial_goal'])

                condition_metrics['goal_achievement_rate'] = goal_achieved_count / num_games
                condition_metrics['goal_escalation_rate'] = goal_escalation_count / num_games
                condition_metrics['avg_goal_escalations'] = total_escalations / num_games

                if initial_goals:
                    condition_metrics['avg_initial_goal'] = sum(initial_goals) / len(initial_goals)

            metrics[condition_key] = condition_metrics

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Coin Flip Gambling Experiment")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help="Model name")
    parser.add_argument('--gpu', type=int, required=True, help="GPU ID")
    parser.add_argument('--bet-type', type=str, default='variable', choices=['fixed', 'variable'],
                        help="Betting type (default: variable)")
    parser.add_argument('--bet-constraint', type=int, default=None,
                        help="Betting constraint amount (required for fixed, optional for variable)")
    parser.add_argument('--num-games', type=int, default=50,
                        help="Number of games per condition (default: 50)")
    parser.add_argument('--quick', action='store_true',
                        help="Quick test mode (10 games per condition)")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory (optional)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--goal-self-setting', action='store_true',
                        help="Include G_SELF condition (model sets own goals)")
    parser.add_argument('--allow-goal-escalation', action='store_true',
                        help="Allow goal escalation after achievement (only for G_SELF)")
    parser.add_argument('--extract-activations', action='store_true',
                        help="Extract and save hidden states for SAE analysis")

    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Quick mode
    if args.quick:
        num_games = 10
        logger.info("Quick test mode: 10 games per condition")
    else:
        num_games = args.num_games

    # Set default bet constraint if not specified
    if args.bet_type == 'fixed' and args.bet_constraint is None:
        args.bet_constraint = 10  # Default $10 for fixed betting
        logger.info(f"Using default fixed bet amount: ${args.bet_constraint}")

    # Run experiment
    experiment = CoinFlipExperiment(
        model_name=args.model,
        gpu_id=args.gpu,
        bet_type=args.bet_type,
        bet_constraint=args.bet_constraint,
        output_dir=args.output_dir,
        extract_activations=args.extract_activations
    )

    # Set goal escalation flag
    experiment.allow_goal_escalation = args.allow_goal_escalation

    experiment.run_experiment(
        num_games_per_condition=num_games,
        include_self_setting=args.goal_self_setting
    )

    # Clear GPU memory
    clear_gpu_memory()


if __name__ == '__main__':
    main()
