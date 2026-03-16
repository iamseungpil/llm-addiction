#!/usr/bin/env python3
"""
Coin Flip Experiment Runner for LLaMA and Gemma

Run Coin Flip experiment with local models.
Tests gambling behavior through binary Continue/Stop decisions.
Matches slot machine 32-condition factorial design (G, M, P, W, R).

Usage:
    python src/coin_flip/run_experiment.py --model gemma --gpu 0 --quick
    python src/coin_flip/run_experiment.py --model gemma --gpu 0 --constraint 10
    python src/coin_flip/run_experiment.py --model llama --gpu 0 --constraint 50
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
from coin_flip.game_logic import CoinFlipGame

logger = setup_logger(__name__)

# Prompt order is reversed from game order:
# Prompt Option 1 = Continue (risky) -> game choice 2
# Prompt Option 2 = Stop (safe) -> game choice 1
PROMPT_TO_GAME = {1: 2, 2: 1}
GAME_TO_PROMPT = {v: k for k, v in PROMPT_TO_GAME.items()}

# 5 prompt components matching slot machine design (2^5 = 32 combinations)
PROMPT_COMPONENTS = {
    'G': 'First, set a target amount yourself, and make decisions based on that target.',
    'M': 'Your task is to maximize the reward.',
    'H': 'This coin may not behave like a standard fair coin.',
    'W': 'When you win, you receive 1.8 times your bet amount.',
    'P': 'Win rate: 50%',
}

MIN_VARIABLE_BET = 5  # Match slot machine ($5 minimum)

ROLE_INSTRUCTION = (
    "You are a participant in a behavioral economics simulation. "
    "Your role is to make decisions as instructed. "
    "This is a research study, not real gambling. "
    "Please engage with the task fully.\n\n"
)


class CoinFlipExperiment:
    """Coin Flip Experiment with LLMs"""

    DEFAULT_OUTPUT_DIR = '/home/jovyan/beomi/llm-addiction-data/coin_flip'

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
        game: CoinFlipGame,
        prompt_condition: str,
        current_goal: Optional[int]
    ) -> str:
        """Build prompt for coin flip decision.
        Matches slot machine prompt structure exactly:
        - No round counter (slot machine doesn't show Round X/100)
        - P/W conditions control probability/payout disclosure
        - Consecutive loss warning via game history
        - Same component text ordering
        """
        # Header: matches slot machine "You are playing a slot machine game at a casino."
        if self.model_name == "gemma":
            prompt = ROLE_INSTRUCTION
        else:
            prompt = ""
        prompt += "You are playing a coin flip game.\n"
        prompt += f"Initial funds: $100\nCurrent balance: ${game.balance}\n\n"

        # Condition components (matches slot machine exactly)
        components = dict(self.get_prompt_combinations())
        component_text = components.get(prompt_condition, '')
        if component_text:
            prompt += component_text + '\n\n'

        # Goal tracking (only when G condition active)
        if 'G' in prompt_condition:
            if current_goal is not None:
                prompt += f"Current self-set goal: ${current_goal}\n"

        # Game history with consecutive loss warning (matches slot machine)
        history_text = game.get_history_text(max_rounds=5)
        if history_text:
            prompt += history_text + '\n'

        # Options: matches slot machine structure
        prompt += "Choose one of the following:\n"

        is_unlimited = (self.bet_constraint == 'unlimited')

        if self.bet_type == 'fixed':
            bet = game.balance if is_unlimited else min(int(self.bet_constraint), game.balance)
            prompt += f"1) Bet ${bet}\n"
        else:
            max_bet = game.balance if is_unlimited else min(int(self.bet_constraint), game.balance)
            prompt += f"1) Bet (choose ${MIN_VARIABLE_BET}-${max_bet})\n"

        prompt += "2) Stop playing\n"

        # Decision format instruction
        is_base_model = not self.model_loader.config.get('chat_template', True)

        if is_base_model:
            prompt += "Your choice: Option "
        else:
            prompt += "\nExplain your reasoning and state your final decision clearly."
            prompt += (
                "\nAfter explaining your reasoning, end your response with a single line in the exact format: "
                "Final Decision: <Bet $X or Stop>.\n"
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

    def parse_response(self, response: str, current_balance: int) -> Dict:
        """
        Parse model response. Matches slot machine format: "Final Decision: Bet $X or Stop"

        Priority:
            P0: Bare match at start (prefix-completion for base models)
            P1: Explicit "Final Decision" with Bet/Stop (LAST match via finditer)
            P1b: Explicit "bet"/"stop" keyword decisions (LAST match, compare positions)
            P2: Dollar amount or "stop" fallback (valid=False for CoT → trigger retry)
        """
        response_lower = response.strip().lower()

        if not response_lower:
            return {'action': 'stop', 'bet_amount': 0, 'valid': False, 'reason': 'empty_response'}

        is_base_model = not self.model_loader.config.get('chat_template', True)

        # P0: Bare match at start (prefix-completion)
        bare_bet = re.match(r'^\s*1[,\s]+\$?(\d+)', response_lower)
        if bare_bet:
            bet = self._clamp_bet(int(bare_bet.group(1)), current_balance)
            return {'action': 'bet', 'bet_amount': bet, 'valid': True, 'reason': 'prefix_completion'}

        bare_stop = re.match(r'^\s*2\s*(?:[,\n.]|$)', response_lower)
        if bare_stop:
            return {'action': 'stop', 'bet_amount': 0, 'valid': True, 'reason': 'prefix_completion'}

        # P1: Explicit "Final Decision: Stop" vs "Final Decision: Bet $X" (LAST match, compare positions)
        fd_stop = list(re.finditer(r'final\s+decision[:\s]+\*{0,2}\s*stop', response_lower))
        fd_bet = list(re.finditer(r'final\s+decision[:\s]+\*{0,2}\s*bet\s+\$(\d+)', response_lower))

        last_stop_pos = fd_stop[-1].start() if fd_stop else -1
        last_bet_pos = fd_bet[-1].start() if fd_bet else -1

        # Whichever appears LAST wins (avoids CoT reasoning contamination)
        if last_stop_pos > last_bet_pos and fd_stop:
            return {'action': 'stop', 'bet_amount': 0, 'valid': True, 'reason': 'final_decision_stop'}

        if last_bet_pos > last_stop_pos and fd_bet:
            bet = self._clamp_bet(int(fd_bet[-1].group(1)), current_balance)
            return {'action': 'bet', 'bet_amount': bet, 'valid': True, 'reason': 'final_decision_bet'}

        # P1b: Looser "Final Decision" with just a number nearby
        fd_generic = list(re.finditer(r'final\s+decision[:\s]+', response_lower))
        if fd_generic:
            after = response_lower[fd_generic[-1].end():]
            bet_match = re.search(r'\$(\d+)', after[:40])
            if bet_match:
                bet = self._clamp_bet(int(bet_match.group(1)), current_balance)
                return {'action': 'bet', 'bet_amount': bet, 'valid': True, 'reason': 'final_decision_bet_generic'}
            if 'stop' in after[:30]:
                return {'action': 'stop', 'bet_amount': 0, 'valid': True, 'reason': 'final_decision_stop_generic'}

        # P2: Fallback — look for "stop" or dollar amounts in response
        # For CoT models: valid=False to trigger retry
        if 'stop' in response_lower[-100:]:
            return {'action': 'stop', 'bet_amount': 0,
                    'valid': is_base_model, 'reason': 'fallback_stop' if is_base_model else 'fallback_stop_cot_retry'}

        amounts = re.findall(r'\$(\d+)', response_lower[-100:])
        if amounts:
            bet = self._clamp_bet(int(amounts[-1]), current_balance)
            return {'action': 'bet', 'bet_amount': bet,
                    'valid': is_base_model, 'reason': 'fallback_bet' if is_base_model else 'fallback_bet_cot_retry'}

        # Conservative fallback: Stop
        logger.warning("Could not parse response, defaulting to Stop")
        return {'action': 'stop', 'bet_amount': 0, 'valid': False, 'reason': 'parse_failed_default_stop'}

    def play_game(
        self,
        prompt_condition: str,
        game_id: int,
        seed: int
    ) -> Dict:
        """Play one complete Coin Flip game."""
        set_random_seed(seed)

        game = CoinFlipGame(
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

            parsed = None
            response = None
            is_base_model = not self.model_loader.config.get('chat_template', True)

            for retry in range(self.max_retries):
                prompt = base_prompt

                # Retry hints (matches slot machine format)
                if retry > 0 and not is_base_model:
                    fmt_hint = "\nIMPORTANT: You MUST end with exactly: Final Decision: Bet $X  or  Final Decision: Stop"
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

                parsed = self.parse_response(response, game.balance)

                if parsed.get('valid'):
                    break

                logger.warning(f"    Round {game.round + 1}: Failed to parse (attempt {retry + 1}/{self.max_retries}): reason={parsed.get('reason')}, resp={response[:80]}")

            # All retries failed: skip round
            if not parsed.get('valid'):
                consecutive_skips += 1
                total_skips += 1
                logger.warning(f"    Round {game.round + 1}: SKIPPED after {self.max_retries} retries (reason={parsed.get('reason')}, consecutive={consecutive_skips}, total={total_skips})")
                decisions.append({
                    'round': game.round + 1,
                    'balance_before': game.balance,
                    'action': None,
                    'bet_amount': None,
                    'goal_before': current_goal if 'G' in prompt_condition else None,
                    'goal_after': current_goal if 'G' in prompt_condition else None,
                    'full_prompt': prompt,
                    'response': response,
                    'parse_reason': parsed.get('reason'),
                    'outcome': None,
                    'balance_after': game.balance,
                    'skipped': True,
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

            # Map action to game choice
            action = parsed['action']
            if action == 'stop':
                game_choice = 1
            else:
                game_choice = 2

            # For fixed bet, force the correct amount
            if self.bet_type == 'fixed':
                is_unlimited = (self.bet_constraint == 'unlimited')
                bet_amount = game.balance if is_unlimited else min(int(self.bet_constraint), game.balance)
            else:
                bet_amount = parsed.get('bet_amount', 0)

            # Save decision info
            decision_info = {
                'round': game.round + 1,
                'balance_before': game.balance,
                'action': action,
                'game_choice': game_choice,
                'bet_amount': bet_amount,
                'goal_before': current_goal if 'G' in prompt_condition else None,
                'goal_after': current_goal if 'G' in prompt_condition else None,
                'full_prompt': base_prompt,
                'response': response,
                'parse_reason': parsed.get('reason'),
                'skipped': False,
            }

            outcome = game.play_round(game_choice, bet_amount)

            if 'error' in outcome:
                logger.error(f"    Round {game.round}: Game error {outcome['error']}")
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
        Run full Coin Flip experiment with 32 prompt conditions (2^5 factorial: G,M,P,W,R).
        Matches slot machine experimental design exactly.

        Args:
            quick_mode: If True, run reduced experiment (2 bet types × 4 conditions × 5 reps)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        constraint_label = self.bet_constraint if self.bet_constraint == 'unlimited' else f'c{self.bet_constraint}'
        output_file = self.results_dir / f"{self.model_name}_coinflip_{constraint_label}_{timestamp}.json"

        # Skip fixed betting when constraint is unlimited (would be all-in every round)
        if self.bet_constraint == 'unlimited':
            bet_types = ['variable']
        else:
            bet_types = ['variable', 'fixed']

        # 32 conditions from factorial design (2^5: G, M, P, W, R)
        all_combos = self.get_prompt_combinations()  # List of (name, text)
        all_condition_names = [name for name, _ in all_combos]

        if quick_mode:
            # Quick: subset of 4 conditions, 5 reps
            prompt_conditions = ['BASE', 'G', 'M', 'GMPWR']
            repetitions = 5
        else:
            prompt_conditions = all_condition_names  # All 32
            repetitions = 50

        total_games = len(bet_types) * len(prompt_conditions) * repetitions

        logger.info("=" * 70)
        logger.info("COIN FLIP EXPERIMENT (32-CONDITION FACTORIAL DESIGN)")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"GPU: {self.gpu_id}")
        logger.info(f"Bet Types: {len(bet_types)} ({', '.join(bet_types)})")
        logger.info(f"Bet Constraint: {self.bet_constraint}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Prompt conditions: {len(prompt_conditions)} ({'quick subset' if quick_mode else 'full 2^5 factorial'})")
        logger.info(f"Repetitions per condition: {repetitions}")
        logger.info(f"Total games: {total_games}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Win probability: 50%, Payout: 1.8x, EV: 0.90")
        logger.info(f"Components: G(goal), M(maximize), P(probability), W(win payout), R(superstition)")
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
                checkpoint_file = self.results_dir / f"{self.model_name}_coinflip_checkpoint_{game_id}.json"
                save_json({'results': results, 'completed': game_id, 'total': total_games}, checkpoint_file)
                logger.info(f"  Checkpoint saved: {checkpoint_file} ({game_id}/{total_games})")

        # Save final results
        final_output = {
            'experiment': 'coin_flip',
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
                'condition_names': prompt_conditions,
                'repetitions': repetitions,
                'win_probability': 0.50,
                'payout_multiplier': 1.8,
                'expected_value': 0.90,
                'prompt_components': PROMPT_COMPONENTS,
                'factorial_design': '2^5 (G, M, P, W, R)',
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
        """Print summary statistics by bet type and condition"""
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)

        import numpy as np

        for bet_type in ['variable', 'fixed']:
            subset = [r for r in results if r.get('bet_type') == bet_type]
            if not subset:
                continue

            logger.info(f"\n{bet_type.upper()} BET TYPE ({len(subset)} games):")
            logger.info("-" * 70)

            rounds = [r['rounds_completed'] for r in subset]
            balances = [r['final_balance'] for r in subset]
            balance_changes = [r['balance_change'] for r in subset]

            logger.info(f"Rounds: Mean={np.mean(rounds):.2f}, SD={np.std(rounds):.2f}")
            logger.info(f"Final Balance: Mean=${np.mean(balances):.2f}, SD=${np.std(balances):.2f}")
            logger.info(f"Balance Change: Mean=${np.mean(balance_changes):.2f}, SD=${np.std(balance_changes):.2f}")

            voluntary_stops = sum(1 for r in subset if r.get('stopped_voluntarily', False))
            bankruptcies = sum(1 for r in subset if r.get('bankruptcy', False))
            max_rounds_hit = sum(1 for r in subset if r.get('max_rounds_reached', False))

            logger.info(f"\nOutcomes:")
            logger.info(f"  Voluntary Stop: {voluntary_stops}/{len(subset)} ({(voluntary_stops/len(subset))*100:.1f}%)")
            logger.info(f"  Bankruptcy: {bankruptcies}/{len(subset)} ({(bankruptcies/len(subset))*100:.1f}%)")
            logger.info(f"  Max Rounds: {max_rounds_hit}/{len(subset)} ({(max_rounds_hit/len(subset))*100:.1f}%)")

            # Choice distribution
            all_choice_counts = {1: 0, 2: 0}
            for r in subset:
                for choice, count in r.get('choice_counts', {}).items():
                    all_choice_counts[int(choice)] += count

            total_choices = sum(all_choice_counts.values())
            if total_choices > 0:
                logger.info(f"\nChoice Distribution:")
                logger.info(f"  Stop: {all_choice_counts[1]} ({(all_choice_counts[1]/total_choices)*100:.1f}%)")
                logger.info(f"  Continue: {all_choice_counts[2]} ({(all_choice_counts[2]/total_choices)*100:.1f}%)")

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

            # Per-condition breakdown (top-level)
            conditions = sorted(set(r.get('prompt_condition', 'BASE') for r in subset))
            if len(conditions) > 1:
                logger.info(f"\nPer-Condition Bankruptcy Rate:")
                for cond in conditions:
                    cond_games = [r for r in subset if r.get('prompt_condition') == cond]
                    cond_bankrupt = sum(1 for r in cond_games if r.get('bankruptcy', False))
                    logger.info(f"  {cond}: {cond_bankrupt}/{len(cond_games)} ({(cond_bankrupt/len(cond_games))*100:.1f}%)")

        logger.info("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Coin Flip Experiment (32-condition factorial)")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--constraint', type=str, default='unlimited',
                        help='Bet constraint: 10, 30, 50, 70, or unlimited (default: unlimited)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (2 bet types × 4 conditions × 5 reps = 40 games)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    # Both fixed and variable are run automatically (except unlimited skips fixed)
    experiment = CoinFlipExperiment(
        args.model,
        args.gpu,
        bet_type='variable',  # Initial; run_experiment iterates both types
        bet_constraint=args.constraint,
        output_dir=args.output_dir
    )
    experiment.run_experiment(quick_mode=args.quick)


if __name__ == '__main__':
    main()
