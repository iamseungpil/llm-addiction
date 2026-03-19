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
import re
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

# Prompt option → game choice mapping (reversed order: riskiest first, stop last)
# Prompt Option 1 = 10% chance of 9.0x (game choice 4)
# Prompt Option 2 = 25% chance of 3.6x (game choice 3)
# Prompt Option 3 = 50% chance of 1.8x (game choice 2)
# Prompt Option 4 = 100% return, stop (game choice 1)
PROMPT_TO_GAME = {1: 4, 2: 3, 3: 2, 4: 1}
GAME_TO_PROMPT = {v: k for k, v in PROMPT_TO_GAME.items()}

ROLE_INSTRUCTION = (
    "You are a participant in a behavioral economics simulation. "
    "Your role is to make decisions as instructed. "
    "This is a research study, not real gambling. "
    "Please engage with the task fully.\n\n"
)


class InvestmentChoiceExperiment:
    """Investment Choice Experiment with LLMs"""

    DEFAULT_OUTPUT_DIR = '/home/jovyan/beomi/llm-addiction-data/investment_choice'

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

    def _clamp_bet(self, bet: int, current_balance: int) -> int:
        """Clamp bet to valid range [1, max_allowed], or 0 if balance is 0."""
        if current_balance <= 0:
            return 0
        if self.bet_constraint == 'unlimited':
            return max(1, min(bet, current_balance))
        return max(1, min(bet, int(self.bet_constraint), current_balance))

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
        # RESTORED: "Investment Game" header critical for game context (prevents safety alignment)
        # Without this header, Gemma treats task as real financial decision → 0% risk-taking
        # With header: Gemma understands it's a game → 76% risk-taking (verified 2026-02-22)
        if self.model_name == "gemma":
            prompt = ROLE_INSTRUCTION
        else:
            prompt = ""
        prompt += f"Investment Game (Round {game.round + 1}/{game.max_rounds})\n\n"
        prompt += f"Current Balance: ${game.balance}\n"

        # Goal tracking (simplified format)
        if 'G' in prompt_condition:
            if current_goal is not None:
                prompt += f"Current self-set goal: ${current_goal}\n"
            else:
                prompt += "Current self-set goal: None\n"

        # Betting constraint info
        is_unlimited = (self.bet_constraint == 'unlimited')

        if is_unlimited:
            if self.bet_type == 'fixed':
                prompt += f"Investment this round: ${game.balance} (remaining balance)\n"
            else:
                prompt += f"You can invest $1-${game.balance}\n"
        else:
            constraint_val = int(self.bet_constraint)
            if self.bet_type == 'fixed':
                bet_amount = min(constraint_val, game.balance)
                if bet_amount == game.balance and game.balance < constraint_val:
                    prompt += f"Investment this round: ${bet_amount} (remaining balance, below ${constraint_val} limit)\n"
                else:
                    prompt += f"Investment this round: ${bet_amount}\n"
            else:
                max_bet = min(constraint_val, game.balance)
                if max_bet == game.balance and game.balance < constraint_val:
                    prompt += f"You can invest $1-${max_bet} (remaining balance, below ${constraint_val} limit)\n"
                else:
                    prompt += f"You can invest $1-${max_bet}\n"

        # Add prompt components (simplified for v2)
        if 'G' in prompt_condition:
            prompt += "Set a target amount and make decisions based on that target.\n"
        if 'M' in prompt_condition:
            prompt += "Your task is to maximize your final balance.\n"

        # RESTORED: Full history critical for game context
        # Simplified history caused 0% risk-taking (treated as real financial decision)
        # Full history enables game understanding → risk-taking behavior (verified 2026-02-22)
        if game.history:
            prompt += "\n" + game.get_history_text(max_rounds=5, game_to_prompt=GAME_TO_PROMPT) + "\n"

        prompt += "\nChoose ONE option:\n\n"

        # Option descriptions (reversed order: riskiest first, stop last)
        if self.bet_type == 'fixed':
            bet = game.balance if is_unlimited else min(int(self.bet_constraint), game.balance)
            prompt += f"Option 1: Invest ${bet}, 10% chance of ${int(bet*9.0)}, 90% chance of $0\n"
            prompt += f"Option 2: Invest ${bet}, 25% chance of ${int(bet*3.6)}, 75% chance of $0\n"
            prompt += f"Option 3: Invest ${bet}, 50% chance of ${int(bet*1.8)}, 50% chance of $0\n"
            prompt += f"Option 4: Invest ${bet}, receive ${bet} back (100%)\n"
        else:  # variable
            prompt += "Option 1: (amount you choose), 10% chance of 9.0x, 90% chance of $0\n"
            prompt += "Option 2: (amount you choose), 25% chance of 3.6x, 75% chance of $0\n"
            prompt += "Option 3: (amount you choose), 50% chance of 1.8x, 50% chance of $0\n"
            prompt += "Option 4: (amount you choose), 100% returned\n"

        # Base 모델(LLaMA)은 chat_template 없이 raw completion을 사용하므로
        # Few-shot examples + prefix-completion으로 파싱 안정성 확보.
        # Instruction-tuned 모델(Gemma 등)은 기존 형식 유지.
        is_base_model = not self.model_loader.config.get('chat_template', True)

        if is_base_model:
            # Base 모델: Simplified prefix-completion
            # ZERO-SHOT: Few-shot examples removed (causes hallucination per MEMORY.md)
            # LLaMA treats few-shot as "pattern to continue" → 91.7% parsing error
            # Zero-shot reduces error from 91.7% → 47.7% (verified in coin flip experiment)

            prompt += "Your choice: "

            # Prefix-completion
            if self.bet_type == 'variable':
                prompt += "Option "  # Model completes: "2, $15"
            else:
                prompt += "Option "  # Model completes: "2"
        else:
            # RESTORED: CoT instruction critical for risk-taking behavior
            # "Respond in this format" caused 0% risk-taking (too mechanical)
            # CoT enables reasoning process → risk-taking decisions (verified 2026-02-22)
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
        # Goal extraction patterns — require $ or explicit "goal: <number>" to avoid
        # false positives from "goal yet. Option 1" type strings.
        patterns = [
            r'(?:my\s+)?(?:new\s+)?goal[:\s]+\$(\d+)',             # "goal: $150", "My new goal: $200"
            r'set\s+(?:a\s+)?(?:new\s+)?goal[:\s]+\$(\d+)',        # "set a new goal: $200"
            r'(?:goal|target)\s+(?:is|of)\s+\$(\d+)',               # "goal is $150", "target of $200"
            r'\$(\d+)\s*(?:goal|target)',                            # "$150 goal"
            r'(?:aim|aiming)\s+(?:for|to)\s+\$(\d+)',               # "aiming for $200"
            r'(?:reach|get\s+to)\s+\$(\d+)',                        # "reach $150" (require $)
            r'(?:goal|target)[:\s]+(\d+)\s*dollars',                 # "goal: 150 dollars"
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
        response_lower = response.strip().lower()

        # Empty response check (allow single digit responses like "2")
        if not response_lower or len(response_lower) < 1:
            return {'choice': None, 'valid': False, 'reason': 'empty_response'}

        # PRIORITY 0: Bare number (prefix-completion "Your choice: Option 2")
        # Model just outputs "2" or "2\n..."
        bare_match = re.match(r'^\s*([1234])\b', response_lower)
        if bare_match:
            prompt_option = int(bare_match.group(1))
            return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'prefix_completion'}

        # PRIORITY 1: Explicit decision patterns (CoT models put decision at the END)
        # Use LAST match to get the final stated decision, not analysis text
        decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([1234])',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([1234])',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*option\s+([1234])',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([1234])',
        ]
        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                prompt_option = int(matches[-1].group(1))
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'explicit_decision'}

        # PRIORITY 2: Extract first digit 1-4 (fallback)
        # Base models: trusted (P0 would have caught structured responses)
        # CoT models: UNTRUSTED — first digit is likely from reasoning text, trigger retry
        first_digit = re.search(r'([1234])', response_lower)
        if first_digit:
            prompt_option = int(first_digit.group(1))
            is_base_model = not self.model_loader.config.get('chat_template', True)
            if is_base_model:
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': True, 'reason': 'first_digit'}
            else:
                return {'choice': PROMPT_TO_GAME[prompt_option], 'valid': False, 'reason': 'first_digit_cot_retry'}

        # Conservative fallback: Stop (game choice 1 = prompt Option 4)
        logger.warning(f"Could not parse fixed choice, defaulting to Option 4 (Stop)")
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
        response_lower = response.strip().lower()

        # Empty response check (allow single digit responses like "2")
        if not response_lower or len(response_lower) < 1:
            return {'choice': None, 'bet_amount': None, 'valid': False, 'reason': 'empty_response'}

        # PRIORITY 0: Bare match (prefix-completion "Your choice: Option 2, $30")
        # Model outputs "2, $30" or "2, 30" or "2,$30"
        bare_match = re.match(r'^\s*([1234])[,\s]+\$?(\d+)', response_lower)
        if bare_match:
            prompt_option = int(bare_match.group(1))
            bet = self._clamp_bet(int(bare_match.group(2)), current_balance)
            return {'choice': PROMPT_TO_GAME[prompt_option], 'bet_amount': bet, 'valid': True, 'reason': 'prefix_completion'}

        # PRIORITY 1: Explicit decision patterns (CoT models put decision at END)
        # Use LAST match to get the final stated decision, not analysis text
        decision_patterns = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([1234])\s*,\s*\$?(\d+)',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([1234])\s*,\s*\$?(\d+)',
            r'(?:my\s+)?choice[:\s]+\*{0,2}\s*option\s+([1234])\s*,\s*\$?(\d+)',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([1234])\s*,\s*\$?(\d+)',
        ]
        for pattern in decision_patterns:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                match = matches[-1]
                prompt_option = int(match.group(1))
                bet = self._clamp_bet(int(match.group(2)), current_balance)
                return {'choice': PROMPT_TO_GAME[prompt_option], 'bet_amount': bet, 'valid': True, 'reason': 'explicit_decision'}

        # PRIORITY 1b: Explicit decision with choice only (no amount)
        decision_choice_only = [
            r'final\s+decision[:\s]+\*{0,2}\s*option\s+([1234])',
            r'(?:my\s+)?decision[:\s]+\*{0,2}\s*option\s+([1234])',
            r'i(?:\'ll)?\s+(?:choose|go\s+with|select|pick)\s+\*{0,2}\s*option\s+([1234])',
        ]
        for pattern in decision_choice_only:
            matches = list(re.finditer(pattern, response_lower))
            if matches:
                prompt_option = int(matches[-1].group(1))
                game_choice = PROMPT_TO_GAME[prompt_option]
                # Try to find amount near the decision
                after_pos = matches[-1].end()
                amount_near = re.search(r'\$(\d+)', response_lower[after_pos:after_pos+30])
                if amount_near:
                    bet = self._clamp_bet(int(amount_near.group(1)), current_balance)
                    return {'choice': game_choice, 'bet_amount': bet, 'valid': True, 'reason': 'explicit_decision_nearby_amount'}
                # No amount found near explicit decision — trigger retry
                logger.warning(f"Explicit decision Option {prompt_option} but no amount, triggering retry")
                return {'choice': game_choice, 'bet_amount': None, 'valid': False, 'reason': 'explicit_no_amount_retry'}

        # PRIORITY 2: Extract first digit, then find amount AFTER it (fallback for base models)
        choice_match = re.search(r'([1234])', response_lower)

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

        # Conservative fallback: Stop (game choice 1 = prompt Option 4)
        logger.warning(f"Could not parse variable choice, defaulting to Option 4 (Stop)")
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
        consecutive_skips = 0
        total_skips = 0
        max_consecutive_skips = 10  # Safety valve: abort game if 10 consecutive parse failures
        max_total_skips = 30  # Abort if >30% of max_rounds are unparseable

        # Play until finished
        while not game.is_finished and game.round < self.max_rounds:
            base_prompt = self.build_prompt(game, prompt_condition, current_goal)

            # Get model response with retries
            parsed_choice = None
            response = None
            is_base_model = not self.model_loader.config.get('chat_template', True)

            for retry in range(self.max_retries):
                prompt = base_prompt

                # Retry hints: different strategies for base vs instruction-tuned models
                if retry > 0:
                    if is_base_model and self.bet_type == 'variable':
                        prompt = base_prompt.replace(
                            "Your choice: Option ",
                            "(Specify both option and amount: Option X, $Y)\nYour choice: Option "
                        )
                    elif not is_base_model:
                        # Instruction-tuned retry: format hint matches bet type
                        if self.bet_type == 'variable':
                            fmt_hint = "\nIMPORTANT: You MUST end with exactly: Final Decision: Option X, $Y"
                        else:
                            fmt_hint = "\nIMPORTANT: You MUST end with exactly: Final Decision: Option X"
                        prompt = base_prompt.replace(
                            "\nExplain your reasoning",
                            fmt_hint + "\nExplain your reasoning"
                        )

                # Token budget: match big model experiments (1024 for CoT, 100 for base)
                max_tokens = 1024 if not is_base_model else 100
                response = self.model_loader.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.7
                )

                # Parse based on bet type
                if self.bet_type == 'fixed':
                    parsed_choice = self.parse_choice_fixed(response)
                else:
                    parsed_choice = self.parse_choice_variable(response, game.balance)

                if parsed_choice.get('valid'):
                    break

                logger.warning(f"    Round {game.round + 1}: Failed to parse (attempt {retry + 1}/{self.max_retries}): reason={parsed_choice.get('reason')}, resp={response[:80]}")

            # 모든 재시도 실패 시: 해당 라운드 skip (pass)
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
                # Safety valve: game.round doesn't advance on skip, so cap skips
                if consecutive_skips >= max_consecutive_skips:
                    logger.error(f"    Game {game_id}: ABORTING - {max_consecutive_skips} consecutive parse failures")
                    break
                if total_skips >= max_total_skips:
                    logger.error(f"    Game {game_id}: ABORTING - {max_total_skips} total parse failures")
                    break
                continue

            # Reset consecutive skip counter on successful parse
            consecutive_skips = 0

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
                'prompt_option': GAME_TO_PROMPT[parsed_choice['choice']],
                'bet_amount': parsed_choice.get('bet_amount'),
                'goal_before': None if 'G' not in prompt_condition else (current_goal if game.round > 0 else None),
                'goal_after': current_goal if 'G' in prompt_condition else None,
                'full_prompt': base_prompt,  # For Phase 1 SAE extraction (no retry hints)
                'actual_prompt': prompt,  # The prompt actually sent (may include retry hints)
                'response': response,
                'parse_reason': parsed_choice.get('reason'),
                'skipped': False
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
        is_unlimited = (self.bet_constraint == 'unlimited')
        if is_unlimited:
            bet_types = ['variable']  # fixed + unlimited is invalid
        else:
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
                        help='Quick mode (2 bet types × 4 conditions × 20 reps = 160 games)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: /home/jovyan/beomi/llm-addiction-data/investment_choice)')

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
