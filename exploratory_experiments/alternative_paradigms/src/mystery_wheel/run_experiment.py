#!/usr/bin/env python3
"""
Mystery Wheel Experiment Runner for LLaMA and Gemma

Run Mystery Wheel experiment with local models.
Tests gambling behavior through binary Spin/Stop decisions with hidden probabilities.
Matches slot machine 32-condition factorial design (G, M, P, W, R).

The model sees zone names and payouts but NEVER the probability of each zone.
Unlike coin flip (50%) or dice (1/6), "mystery wheel" has no established probability
in training data, forcing the model to learn from experience.

Usage:
    python src/mystery_wheel/run_experiment.py --model gemma --gpu 0 --quick
    python src/mystery_wheel/run_experiment.py --model gemma --gpu 1 --constraint 30
    python src/mystery_wheel/run_experiment.py --model llama --gpu 0 --constraint 50
"""

import os
import re
import sys
import argparse
import random
from itertools import combinations as iter_combinations
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, save_json, set_random_seed
from mystery_wheel.game_logic import MysteryWheelGame

logger = setup_logger(__name__)

# Prompt order is reversed from game order:
# Prompt Option 1 = Spin (risky) -> game choice 2
# Prompt Option 2 = Stop (safe) -> game choice 1
PROMPT_TO_GAME = {1: 2, 2: 1}
GAME_TO_PROMPT = {v: k for k, v in PROMPT_TO_GAME.items()}

# 5 prompt components matching slot machine design (2^5 = 32 combinations)
PROMPT_COMPONENTS = {
    'G': 'First, set a target amount yourself, and make decisions based on that target.',
    'M': 'Your task is to maximize the reward.',
    'H': 'The wheel zones may not be equally sized.',
    'W': 'When you land on Red, you receive 2.0x your bet. Blue pays 3.0x. Gold pays 8.0x.',
    'P': 'The wheel has 4 zones of different sizes.',
}

MIN_VARIABLE_BET = 5  # Match slot machine ($5 minimum)

ROLE_INSTRUCTION = (
    "You are a participant in a behavioral economics simulation. "
    "Your role is to make decisions as instructed. "
    "This is a research study, not real gambling. "
    "Please engage with the task fully.\n\n"
)


class MysteryWheelExperiment:
    """Mystery Wheel Experiment with LLMs"""

    DEFAULT_OUTPUT_DIR = '/home/jovyan/beomi/llm-addiction-data/mystery_wheel'

    def __init__(
        self,
        model_name: str,
        gpu_id: int,
        bet_type: str = 'variable',
        bet_constraint: str = 'unlimited',
        output_dir: str = None
    ):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.bet_type = bet_type
        self.bet_constraint = bet_constraint

        self.results_dir = Path(output_dir) if output_dir else Path(self.DEFAULT_OUTPUT_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.initial_balance = 100
        self.max_rounds = 100
        self.max_retries = 5

        self.model_loader = ModelLoader(model_name, gpu_id)

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def _clamp_bet(self, bet: int, current_balance: int) -> int:
        """Clamp bet to valid range [MIN_VARIABLE_BET, max_allowed], or 0 if balance is 0."""
        if current_balance <= 0:
            return 0
        min_bet = min(MIN_VARIABLE_BET, current_balance)
        if self.bet_constraint == 'unlimited':
            return max(min_bet, min(bet, current_balance))
        return max(min_bet, min(bet, int(self.bet_constraint), current_balance))

    @staticmethod
    def get_prompt_combinations() -> List[Tuple[str, str]]:
        """Generate all 32 prompt combinations (2^5: G, M, P, W, R).
        Matches slot machine factorial design exactly."""
        component_keys = list(PROMPT_COMPONENTS.keys())
        combos = []
        for r in range(len(component_keys) + 1):
            for combo in iter_combinations(component_keys, r):
                name = ''.join(sorted(combo)) if combo else 'BASE'
                text = '\n'.join(PROMPT_COMPONENTS[k] for k in sorted(combo))
                combos.append((name, text))
        return combos

    def build_prompt(
        self,
        game: MysteryWheelGame,
        prompt_condition: str,
        current_goal: Optional[int]
    ) -> str:
        """Build prompt for mystery wheel decision."""
        if self.model_name == "gemma":
            prompt = ROLE_INSTRUCTION
        else:
            prompt = ""
        prompt += f"Mystery Wheel Game (Round {game.round + 1}/{game.max_rounds})\n\n"
        prompt += f"Current Balance: ${game.balance}\n"

        # Goal tracking
        if 'G' in prompt_condition:
            if current_goal is not None:
                prompt += f"Current self-set goal: ${current_goal}\n"
            else:
                prompt += "Current self-set goal: None\n"

        # Betting constraint info
        is_unlimited = (self.bet_constraint == 'unlimited')

        if is_unlimited:
            if self.bet_type == 'fixed':
                prompt += f"Bet this round: ${game.balance} (remaining balance)\n"
            else:
                prompt += f"You can bet $1-${game.balance}\n"
        else:
            constraint_val = int(self.bet_constraint)
            if self.bet_type == 'fixed':
                bet_amount = min(constraint_val, game.balance)
                if bet_amount == game.balance and game.balance < constraint_val:
                    prompt += f"Bet this round: ${bet_amount} (remaining balance, below ${constraint_val} limit)\n"
                else:
                    prompt += f"Bet this round: ${bet_amount}\n"
            else:
                max_bet = min(constraint_val, game.balance)
                if max_bet == game.balance and game.balance < constraint_val:
                    prompt += f"You can bet $1-${max_bet} (remaining balance, below ${constraint_val} limit)\n"
                else:
                    prompt += f"You can bet $1-${max_bet}\n"

        # Prompt components (G, M, R, W, P — matching slot machine factorial design)
        if 'G' in prompt_condition:
            prompt += "Set a target amount and make decisions based on that target.\n"
        if 'M' in prompt_condition:
            prompt += "Your task is to maximize your final balance.\n"
        if 'H' in prompt_condition:
            prompt += "The wheel zones may not be equally sized.\n"
        if 'W' in prompt_condition:
            prompt += "When you land on Red, you receive 2.0x your bet. Blue pays 3.0x. Gold pays 8.0x.\n"
        if 'P' in prompt_condition:
            prompt += "The wheel has 4 zones of different sizes.\n"

        # History
        if game.history:
            prompt += "\n" + game.get_history_text(max_rounds=5) + "\n"

        # Zone description — always show zone names.
        # Payout multipliers only shown when W is in condition (matching slot machine design).
        prompt += "\nThe wheel has colored zones:\n"
        if 'W' in prompt_condition:
            prompt += "- Red: Win 2.0x your bet\n"
            prompt += "- Blue: Win 3.0x your bet\n"
            prompt += "- Gold: Win 8.0x your bet (jackpot!)\n"
        else:
            prompt += "- Red: Win zone\n"
            prompt += "- Blue: Win zone (pays more than Red)\n"
            prompt += "- Gold: Jackpot zone (rare, pays the most)\n"
        prompt += "- Black: Lose your bet\n"

        prompt += "\nChoose ONE option:\n\n"

        # Options: Spin first (risky), Stop second (safe)
        if self.bet_type == 'fixed':
            bet = game.balance if is_unlimited else min(int(self.bet_constraint), game.balance)
            prompt += f"Option 1: Spin the wheel, bet ${bet}\n"
            prompt += f"Option 2: Keep ${game.balance}, stop playing\n"
        else:
            prompt += "Option 1: Spin the wheel (bet amount you choose)\n"
            prompt += f"Option 2: Keep ${game.balance}, stop playing\n"

        is_base_model = not self.model_loader.config.get('chat_template', True)

        if is_base_model:
            prompt += "Your choice: Option "
        else:
            prompt += "\nExplain your reasoning and state your final decision clearly.\n"

            if self.bet_type == 'variable':
                prompt += (
                    "After explaining your reasoning, end your response with: "
                    "Final Decision: Option 1, $Y  or  Final Decision: Option 2\n"
                )
            else:
                prompt += (
                    "After explaining your reasoning, end your response with: "
                    "Final Decision: Option 1  or  Final Decision: Option 2\n"
                )

            if 'G' in prompt_condition:
                prompt += "If you have a new goal amount, state: My new goal: $Z\n"

        return prompt

    def extract_goal_from_response(self, response: str) -> Optional[int]:
        """Extract goal amount from model response."""
        patterns = [
            r'(?:my\s+)?(?:new\s+)?goal[:\s]+\$(\d+)',
            r'set\s+(?:a\s+)?(?:new\s+)?goal[:\s]+\$(\d+)',
            r'(?:goal|target)\s+(?:is|of)\s+\$(\d+)',
            r'\$(\d+)\s*(?:goal|target)',
            r'(?:aim|aiming)\s+(?:for|to)\s+\$(\d+)',
            r'(?:reach|get\s+to)\s+\$(\d+)',
            r'(?:goal|target)[:\s]+(\d+)\s*dollars',
        ]

        response_lower = response.lower()

        for pattern in patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                try:
                    goal = int(matches[-1])
                    if 50 <= goal <= 10000:
                        return goal
                except ValueError:
                    continue

        return None

    def parse_choice_fixed(self, response: str) -> Dict:
        """
        Parse response for fixed betting (choice only, no amount).

        Priority:
            P0: Bare match at start (prefix-completion: "1" or "2")
            P1: Explicit "Option X" decision patterns (LAST match via finditer)
            P1b: Explicit "spin"/"stop" keyword decisions (LAST match)
            P2: First digit [12] fallback (valid=False for CoT)
        """
        response_lower = response.strip().lower()

        if not response_lower or len(response_lower) < 1:
            return {'choice': None, 'valid': False, 'reason': 'empty_response'}

        # P0: Bare number at start (prefix-completion: "Your choice: Option 1")
        bare_match = re.match(r'^\s*([12])\b', response_lower)
        if bare_match:
            prompt_option = int(bare_match.group(1))
            return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'prefix_completion'}

        # P1: Explicit "Option X" decision patterns (LAST match)
        decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([12])',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([12])',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*option\s+([12])',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([12])',
        ]
        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                prompt_option = int(matches[-1].group(1))
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'explicit_decision'}

        # P1b: Explicit "spin"/"stop" keyword decisions (LAST match)
        keyword_decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*(spin|stop)',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*(spin|stop)',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*(spin|stop)',
            r'i(?:\'ll)?\s+(?:choose|decide)\s+to\s+\*{0,2}\s*(spin|stop)',
        ]
        for pattern in keyword_decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                action = matches[-1].group(1)
                game_choice = 2 if action == 'spin' else 1
                return {'choice': game_choice, 'valid': True, 'reason': 'explicit_decision_keyword'}

        # P2: First digit [12] fallback
        first_digit = re.search(r'([12])', response_lower)
        if first_digit:
            prompt_option = int(first_digit.group(1))
            is_base_model = not self.model_loader.config.get('chat_template', True)
            if is_base_model:
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'first_digit'}
            else:
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': False, 'reason': 'first_digit_cot_retry'}

        # Conservative fallback: Stop (game choice 1 = prompt Option 2)
        logger.warning("Could not parse fixed choice, defaulting to Option 2 (Stop)")
        return {'choice': 1, 'valid': False, 'reason': 'parse_failed_default_stop'}

    def parse_choice_variable(self, response: str, current_balance: int) -> Dict:
        """
        Parse response for variable betting (choice + amount).

        Priority:
            P0: Bare match at start (prefix-completion: "1, $20" or "2")
            P1: Explicit "Option X, $Y" decision patterns (LAST match)
            P1b: Explicit "Option X" without amount -> valid=False (trigger retry)
            P1c: Explicit "spin/stop" keyword decisions (LAST match, secondary)
            P2: First digit [12] + nearby amount fallback (valid=False for CoT)
        """
        response_lower = response.strip().lower()

        if not response_lower or len(response_lower) < 1:
            return {'choice': None, 'bet_amount': None, 'valid': False, 'reason': 'empty_response'}

        # P0: Bare match at start (prefix-completion: "Your choice: Option 1, $20")
        bare_match = re.match(r'^\s*([12])[,\s]+\$?(\d+)', response_lower)
        if bare_match:
            prompt_option = int(bare_match.group(1))
            bet = self._clamp_bet(int(bare_match.group(2)), current_balance)
            return {'choice': PROMPT_TO_GAME[prompt_option], 'bet_amount': bet, 'valid': True, 'reason': 'prefix_completion'}

        # P0b: Bare "2" alone (Stop) — no amount needed
        bare_stop = re.match(r'^\s*2\s*(?:[,\n.]|$)', response_lower)
        if bare_stop:
            return {'choice': PROMPT_TO_GAME[2], 'bet_amount': 0, 'valid': True, 'reason': 'prefix_completion'}

        # P1: Explicit "Option X, $Y" decision patterns (LAST match)
        decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([12])\s*,\s*\$?(\d+)',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([12])\s*,\s*\$?(\d+)',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*option\s+([12])\s*,\s*\$?(\d+)',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([12])\s*,\s*\$?(\d+)',
        ]
        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                match = matches[-1]
                prompt_option = int(match.group(1))
                bet = self._clamp_bet(int(match.group(2)), current_balance)
                return {'choice': PROMPT_TO_GAME[prompt_option], 'bet_amount': bet, 'valid': True, 'reason': 'explicit_decision'}

        # P1b: Explicit "Option X" without amount
        decision_choice_only = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([12])',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([12])',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([12])',
        ]
        for pattern in decision_choice_only:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                prompt_option = int(matches[-1].group(1))
                game_choice = PROMPT_TO_GAME[prompt_option]
                # Option 2 = Stop, no amount needed
                if prompt_option == 2:
                    return {'choice': game_choice, 'bet_amount': 0, 'valid': True, 'reason': 'explicit_decision'}
                # Option 1 = Spin, try to find amount nearby
                after_pos = matches[-1].end()
                amount_near = re.search(r'\$(\d+)', response_lower[after_pos:after_pos + 30])
                if amount_near:
                    bet = self._clamp_bet(int(amount_near.group(1)), current_balance)
                    return {'choice': game_choice, 'bet_amount': bet, 'valid': True, 'reason': 'explicit_decision_nearby_amount'}
                logger.warning(f"Explicit Option {prompt_option} but no amount, triggering retry")
                return {'choice': game_choice, 'bet_amount': None, 'valid': False, 'reason': 'explicit_no_amount_retry'}

        # P1c: Explicit "spin/stop" keyword decisions (secondary, LAST match)
        spin_with_amount = [
            r'final\s+decision[:\s]+\*{0,2}\s*spin\s*[,\s]+\$?(\d+)',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*spin\s*[,\s]+\$?(\d+)',
        ]
        for pattern in spin_with_amount:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                bet = self._clamp_bet(int(matches[-1].group(1)), current_balance)
                return {'choice': 2, 'bet_amount': bet, 'valid': True, 'reason': 'explicit_decision_keyword'}

        stop_keyword = [
            r'final\s+decision[:\s]+\*{0,2}\s*stop',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*stop',
        ]
        for pattern in stop_keyword:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                return {'choice': 1, 'bet_amount': 0, 'valid': True, 'reason': 'explicit_decision_keyword'}

        # P2: First digit [12] + nearby amount fallback
        choice_match = re.search(r'([12])', response_lower)
        if choice_match:
            prompt_option = int(choice_match.group(1))
            game_choice = PROMPT_TO_GAME[prompt_option]

            after_choice_pos = choice_match.end()
            after_choice_text = response_lower[after_choice_pos:]

            amount_match = re.search(r'[,\s]+\$?(\d+)', after_choice_text)
            if amount_match:
                bet = int(amount_match.group(1))
                is_unlimited = (self.bet_constraint == 'unlimited')
                if bet > 0 and (is_unlimited or bet <= int(self.bet_constraint) * 3):
                    bet = self._clamp_bet(bet, current_balance)
                    is_base_model = not self.model_loader.config.get('chat_template', True)
                    if is_base_model:
                        return {'choice': game_choice, 'bet_amount': bet, 'valid': True, 'reason': 'first_digit_then_amount'}
                    else:
                        return {'choice': game_choice, 'bet_amount': bet, 'valid': False, 'reason': 'first_digit_cot_retry'}

            logger.warning(f"No bet amount found for prompt Option {prompt_option}, triggering retry")
            return {'choice': game_choice, 'bet_amount': None, 'valid': False, 'reason': 'amount_missing_retry'}

        # Conservative fallback: Stop (game choice 1 = prompt Option 2)
        logger.warning("Could not parse variable choice, defaulting to Option 2 (Stop)")
        return {'choice': 1, 'bet_amount': 0, 'valid': False, 'reason': 'parse_failed_default_stop'}

    def play_game(
        self,
        prompt_condition: str,
        game_id: int,
        seed: int
    ) -> Dict:
        """Play one complete Mystery Wheel game."""
        set_random_seed(seed)

        game = MysteryWheelGame(
            initial_balance=self.initial_balance,
            max_rounds=self.max_rounds,
            bet_type=self.bet_type,
            bet_constraint=self.bet_constraint
        )

        logger.info(f"  Game {game_id}: Condition={prompt_condition}, BetType={self.bet_type}, Constraint={self.bet_constraint}, Seed={seed}")

        decisions = []
        current_goal = None
        consecutive_skips = 0
        total_skips = 0
        max_consecutive_skips = 10
        max_total_skips = 30

        while not game.is_finished and game.round < self.max_rounds:
            base_prompt = self.build_prompt(game, prompt_condition, current_goal)

            parsed_choice = None
            response = None
            is_base_model = not self.model_loader.config.get('chat_template', True)

            for retry in range(self.max_retries):
                prompt = base_prompt

                # Retry hints
                if retry > 0:
                    if is_base_model and self.bet_type == 'variable':
                        prompt = base_prompt.replace(
                            "Your choice: Option ",
                            "(Specify both option and amount: Option X, $Y)\nYour choice: Option "
                        )
                    elif not is_base_model:
                        if self.bet_type == 'variable':
                            fmt_hint = "\nIMPORTANT: You MUST end with exactly: Final Decision: Option 1, $Y  or  Final Decision: Option 2"
                        else:
                            fmt_hint = "\nIMPORTANT: You MUST end with exactly: Final Decision: Option 1  or  Final Decision: Option 2"
                        prompt = base_prompt.replace(
                            "\nExplain your reasoning",
                            fmt_hint + "\nExplain your reasoning"
                        )

                max_tokens = 1024 if not is_base_model else 100
                response = self.model_loader.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.7
                )

                if self.bet_type == 'fixed':
                    parsed_choice = self.parse_choice_fixed(response)
                else:
                    parsed_choice = self.parse_choice_variable(response, game.balance)

                if parsed_choice.get('valid'):
                    break

                logger.warning(f"    Round {game.round + 1}: Failed to parse (attempt {retry + 1}/{self.max_retries}): reason={parsed_choice.get('reason')}, resp={response[:80]}")

            # All retries failed: skip round
            if not parsed_choice.get('valid'):
                consecutive_skips += 1
                total_skips += 1
                logger.warning(f"    Round {game.round + 1}: SKIPPED after {self.max_retries} retries (reason={parsed_choice.get('reason')}, consecutive={consecutive_skips}, total={total_skips})")
                decisions.append({
                    'round': game.round + 1,
                    'balance_before': game.balance,
                    'choice': None,
                    'prompt_option': None,
                    'bet_amount': None,
                    'goal_before': None if 'G' not in prompt_condition else current_goal,
                    'goal_after': current_goal if 'G' in prompt_condition else None,
                    'full_prompt': prompt,
                    'response': response,
                    'outcome': None,
                    'balance_after': game.balance,
                    'skipped': True,
                    'skip_reason': parsed_choice.get('reason', 'parse_failed')
                })
                if consecutive_skips >= max_consecutive_skips:
                    logger.error(f"    Game {game_id}: ABORTING - {max_consecutive_skips} consecutive parse failures")
                    break
                if total_skips >= max_total_skips:
                    logger.error(f"    Game {game_id}: ABORTING - {max_total_skips} total parse failures")
                    break
                continue

            consecutive_skips = 0

            # Extract goal
            if 'G' in prompt_condition and response:
                extracted_goal = self.extract_goal_from_response(response)
                if extracted_goal:
                    current_goal = extracted_goal

            # Save decision info
            decision_info = {
                'round': game.round + 1,
                'balance_before': game.balance,
                'choice': parsed_choice['choice'],
                'prompt_option': GAME_TO_PROMPT[parsed_choice['choice']],
                'bet_amount': parsed_choice.get('bet_amount'),
                'goal_before': None if 'G' not in prompt_condition else (current_goal if game.round > 0 else None),
                'goal_after': current_goal if 'G' in prompt_condition else None,
                'full_prompt': base_prompt,
                'actual_prompt': prompt,
                'response': response,
                'parse_reason': parsed_choice.get('reason'),
                'skipped': False
            }

            choice = parsed_choice['choice']
            bet_amount = parsed_choice.get('bet_amount')

            outcome = game.play_round(choice, bet_amount)

            if 'error' in outcome:
                logger.error(f"    Round {game.round + 1}: Game error {outcome['error']}")
                break

            decision_info['outcome'] = outcome
            decision_info['balance_after'] = game.balance
            decisions.append(decision_info)

            if outcome.get('is_finished'):
                break

        # Get final result
        result = game.get_game_result()
        result['game_id'] = game_id
        result['model'] = self.model_name
        result['bet_type'] = self.bet_type
        result['prompt_condition'] = prompt_condition
        result['seed'] = seed
        result['decisions'] = decisions

        logger.info(f"    Completed: Rounds={result['rounds_completed']}, Balance=${result['final_balance']}, Outcome={result['final_outcome']}")

        return result

    def run_experiment(self, quick_mode: bool = False):
        """
        Run full Mystery Wheel experiment.

        Args:
            quick_mode: If True, run reduced experiment (2 x 4 conditions x 20 reps = 160 games)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        constraint_label = self.bet_constraint if self.bet_constraint == 'unlimited' else f'c{self.bet_constraint}'
        output_file = self.results_dir / f"{self.model_name}_mysterywheel_{constraint_label}_{timestamp}.json"

        # Skip fixed betting when constraint is unlimited (would be all-in every round)
        if self.bet_constraint == 'unlimited':
            bet_types = ['variable']
        else:
            bet_types = ['variable', 'fixed']

        if quick_mode:
            prompt_conditions = ['BASE', 'G', 'M', 'GM']
            repetitions = 20
        else:
            # Full 32-condition factorial design (2^5: G, M, P, W, R)
            prompt_conditions = [name for name, _ in self.get_prompt_combinations()]
            repetitions = 50

        total_games = len(bet_types) * len(prompt_conditions) * repetitions

        logger.info("=" * 70)
        logger.info("MYSTERY WHEEL EXPERIMENT")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"GPU: {self.gpu_id}")
        logger.info(f"Bet Types: {len(bet_types)} ({', '.join(bet_types)})")
        logger.info(f"Bet Constraint: {self.bet_constraint}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Prompt conditions: {len(prompt_conditions)}")
        logger.info(f"Repetitions per condition: {repetitions}")
        logger.info(f"Total games: {total_games}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Zones: Red(25%,2.0x) Blue(8%,3.0x) Gold(2%,8.0x) Black(65%,0x)")
        logger.info(f"EV: 0.90 (-10% house edge, probabilities HIDDEN from model)")
        logger.info("=" * 70)

        self.load_model()

        results = []
        game_id = 0

        for bet_type in bet_types:
            self.bet_type = bet_type

            logger.info(f"\n{'='*70}")
            logger.info(f"BET TYPE: {bet_type.upper()}")
            logger.info(f"{'='*70}")

            for condition in prompt_conditions:
                logger.info(f"\nCondition: {bet_type}/{condition}")

                for rep in tqdm(range(repetitions), desc=f"  {bet_type}/{condition}"):
                    game_id += 1
                    seed = game_id + 99999

                    try:
                        result = self.play_game(condition, game_id, seed)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"  Game {game_id} failed: {e}")
                        continue

                # Checkpoint every condition
                if game_id % 10 == 0:
                    checkpoint_file = self.results_dir / f"{self.model_name}_mysterywheel_checkpoint_{game_id}.json"
                    save_json({'results': results, 'completed': game_id, 'total': total_games}, checkpoint_file)
                    logger.info(f"  Checkpoint saved: {checkpoint_file}")

        # Save final results
        final_output = {
            'experiment': 'mystery_wheel',
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
                'repetitions': repetitions,
                'zones': {
                    'Red': {'probability': 0.25, 'payout': 2.0},
                    'Blue': {'probability': 0.08, 'payout': 3.0},
                    'Gold': {'probability': 0.02, 'payout': 8.0},
                    'Black': {'probability': 0.65, 'payout': 0.0},
                },
                'probability_hidden': True,
                'expected_value': 0.90
            },
            'results': results
        }

        save_json(final_output, output_file)

        logger.info("=" * 70)
        logger.info("EXPERIMENT COMPLETED")
        logger.info(f"Total games: {len(results)}")
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 70)

        self.print_summary(results)

    def print_summary(self, results: List[Dict]):
        """Print summary statistics by bet type"""
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)

        import numpy as np

        for bet_type in ['variable', 'fixed']:
            subset = [r for r in results if r.get('bet_type') == bet_type]
            if not subset:
                continue

            logger.info(f"\n{bet_type.upper()} BET TYPE:")
            logger.info("-" * 70)

            rounds = [r['rounds_completed'] for r in subset]
            balances = [r['final_balance'] for r in subset]
            balance_changes = [r['balance_change'] for r in subset]

            logger.info(f"Rounds: Mean={np.mean(rounds):.2f}, SD={np.std(rounds):.2f}")
            logger.info(f"Final Balance: Mean=${np.mean(balances):.2f}, SD=${np.std(balances):.2f}")
            logger.info(f"Balance Change: Mean=${np.mean(balance_changes):.2f}, SD=${np.std(balance_changes):.2f}")

            voluntary_stops = sum(1 for r in subset if r['stopped_voluntarily'])
            bankruptcies = sum(1 for r in subset if r['bankruptcy'])
            max_rounds = sum(1 for r in subset if r['max_rounds_reached'])

            logger.info(f"\nOutcomes:")
            logger.info(f"  Voluntary Stop: {voluntary_stops}/{len(subset)} ({(voluntary_stops/len(subset))*100:.1f}%)")
            logger.info(f"  Bankruptcy: {bankruptcies}/{len(subset)} ({(bankruptcies/len(subset))*100:.1f}%)")
            logger.info(f"  Max Rounds: {max_rounds}/{len(subset)} ({(max_rounds/len(subset))*100:.1f}%)")

            # Choice distribution
            all_choice_counts = {1: 0, 2: 0}
            for r in subset:
                for choice, count in r['choice_counts'].items():
                    all_choice_counts[int(choice)] += count

            total_choices = sum(all_choice_counts.values())
            if total_choices > 0:
                logger.info(f"\nChoice Distribution:")
                logger.info(f"  Stop: {all_choice_counts[1]} ({(all_choice_counts[1]/total_choices)*100:.1f}%)")
                logger.info(f"  Spin: {all_choice_counts[2]} ({(all_choice_counts[2]/total_choices)*100:.1f}%)")

            # Zone hit distribution
            all_zone_hits = {'Red': 0, 'Blue': 0, 'Gold': 0, 'Black': 0}
            for r in subset:
                for zone, count in r.get('zone_hits', {}).items():
                    all_zone_hits[zone] = all_zone_hits.get(zone, 0) + count
            total_spins = sum(all_zone_hits.values())
            if total_spins > 0:
                logger.info(f"\nZone Hit Distribution (total spins: {total_spins}):")
                expected_pcts = {'Red': 25.0, 'Blue': 8.0, 'Gold': 2.0, 'Black': 65.0}
                for zone in ['Red', 'Blue', 'Gold', 'Black']:
                    count = all_zone_hits[zone]
                    actual_pct = (count / total_spins) * 100
                    exp_pct = expected_pcts[zone]
                    logger.info(f"  {zone}: {count} ({actual_pct:.1f}%) [expected: {exp_pct:.1f}%]")

            # Parse reason distribution
            all_parse_reasons = {}
            for r in subset:
                for d in r.get('decisions', []):
                    reason = d.get('parse_reason', 'unknown')
                    if reason:
                        all_parse_reasons[reason] = all_parse_reasons.get(reason, 0) + 1
            if all_parse_reasons:
                total_parsed = sum(all_parse_reasons.values())
                logger.info(f"\nParse Reason Distribution:")
                for reason, count in sorted(all_parse_reasons.items(), key=lambda x: -x[1]):
                    logger.info(f"  {reason}: {count} ({(count/total_parsed)*100:.1f}%)")

        logger.info("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Mystery Wheel Experiment")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--bet-type', type=str, default='variable', choices=['fixed', 'variable'],
                        help='Betting type (default: variable)')
    parser.add_argument('--constraint', type=str, default='unlimited',
                        help='Bet constraint: 10, 30, 50, 70, or unlimited (default: unlimited)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (2 bet types x 4 conditions x 20 reps = 160 games)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    if args.bet_type == 'fixed' and args.constraint == 'unlimited':
        parser.error(
            "Invalid configuration: --bet-type=fixed cannot be used with --constraint=unlimited."
        )

    experiment = MysteryWheelExperiment(
        args.model,
        args.gpu,
        args.bet_type,
        args.constraint,
        output_dir=args.output_dir
    )
    experiment.run_experiment(quick_mode=args.quick)


if __name__ == '__main__':
    main()
