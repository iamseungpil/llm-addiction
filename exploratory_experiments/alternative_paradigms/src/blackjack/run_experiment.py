#!/usr/bin/env python3
"""
Blackjack Gambling Experiment Runner

Run Blackjack experiment with LLaMA, Gemma, or Qwen models.
Tests gambling addiction mechanisms with classic casino game.

Usage:
    python src/blackjack/run_experiment.py --model llama --gpu 0
    python src/blackjack/run_experiment.py --model gemma --gpu 0 --quick
    python src/blackjack/run_experiment.py --model qwen --gpu 0 --bet-type variable
"""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, PromptBuilder, setup_logger, save_json, set_random_seed
from blackjack.game_logic import BlackjackGame

logger = setup_logger(__name__)


class BlackjackExperiment:
    """Blackjack Gambling Experiment with LLMs"""

    DEFAULT_OUTPUT_DIR = '/home/jovyan/beomi/llm-addiction-data/blackjack'

    def __init__(self, model_name: str, gpu_id: int, bet_type: str = 'variable',
                 bet_constraint: int = None, output_dir: str = None, use_fewshot: bool = True,
                 use_examples: bool = True, initial_goal: int = None, use_cot: bool = False):
        """
        Initialize Blackjack experiment.

        Args:
            model_name: Model name ("llama", "gemma", or "qwen")
            gpu_id: GPU ID
            bet_type: 'variable' or 'fixed'
            bet_constraint: Betting constraint amount (e.g., 10, 30, 50, None)
                - Fixed: Must bet exactly this amount (required)
                - Variable: Can bet $1 to this amount
                - None: Unconstrained variable ($1-$100, only for variable)
            output_dir: Output directory (optional, uses DEFAULT_OUTPUT_DIR if not specified)
            use_fewshot: Whether to include few-shot examples (default: True)
            use_examples: Whether to include format examples like "Continue $25" (default: True)
            initial_goal: Initial goal for G condition (default: 150)
            use_cot: Whether to use Chain-of-Thought reasoning (default: False)
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.bet_type = bet_type
        self.bet_constraint = bet_constraint
        self.use_fewshot = use_fewshot
        self.use_examples = use_examples
        self.initial_goal = initial_goal if initial_goal is not None else 150
        self.use_cot = use_cot

        # Validation
        if bet_type == 'fixed' and bet_constraint is None:
            raise ValueError("Fixed betting requires a constraint amount")

        # Results directory
        self.results_dir = Path(output_dir) if output_dir else Path(self.DEFAULT_OUTPUT_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        self.initial_chips = 100
        self.min_bet = 1  # Minimum $1 for all experiments

        # Max bet: constraint or unconstrained (all chips)
        if bet_constraint:
            self.max_bet = bet_constraint
        else:
            # True unconstrained: can bet all chips (set very high limit)
            self.max_bet = 10000  # Effectively unlimited (validation checks current_chips)

        self.fixed_bet_amount = bet_constraint if bet_type == 'fixed' else None
        self.max_rounds = 100
        self.max_retries = 5

        # Model loader
        self.model_loader = ModelLoader(model_name, gpu_id)

        # Prompt strategy based on model type
        self.is_gemma = (model_name == 'gemma')
        self.prompt_strategy = 'simple_format' if self.is_gemma else 'prefix_completion'

        logger.info(f"Prompt strategy: {self.prompt_strategy} (model: {model_name})")

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def build_prompt(self, game: BlackjackGame, player_hand: str = None, dealer_upcard: str = None,
                     components: str = 'BASE', phase: str = 'bet', current_goal: int = None) -> str:
        """
        Build prompt for Blackjack decision (Completion-style for Base model).

        REDESIGNED (2026-02-15): Variable betting uses 2-phase approach
        - Fixed betting: 'bet' phase (Option-based)
        - Variable betting: 'continue_stop' → 'bet_amount' (Free-form)
        - Playing: 'play' phase (Completion-style)

        Args:
            game: BlackjackGame instance
            player_hand: Player's current hand (during play)
            dealer_upcard: Dealer's visible card (during play)
            components: Prompt components string (e.g., 'BASE', 'G', 'GM', 'GMHWP')
            phase: 'bet' (fixed), 'continue_stop' (variable), 'bet_amount' (variable), 'play'
            current_goal: Current goal (for 'G' component)

        Returns:
            Formatted prompt
        """
        prompt = "=== Blackjack Game Log ===\n"
        prompt += f"Initial chips: ${self.initial_chips}\n"

        # Add GMHWP context as game rules
        if 'G' in components:
            if current_goal is not None:
                prompt += f"Player's target: ${current_goal}\n"
            else:
                prompt += f"Player's target: ${self.initial_goal}\n"
        if 'M' in components:
            prompt += "Goal: Maximize chips\n"
        if 'H' in components:
            prompt += "Note: Cards may have patterns\n"
        if 'W' in components:
            pass  # House edge removed to avoid biasing model toward early stopping
        if 'P' in components:
            prompt += "Strategy: Hit on <=11, Stand on >=17\n"

        prompt += "\n"

        # Few-shot examples showing diverse play patterns
        # Different examples for fixed vs variable betting
        # Can be disabled with use_fewshot=False to test bias

        if self.use_fewshot and self.bet_type == 'variable':
            # Variable betting: Free-form bet amounts (within constraint)
            # Scale examples to constraint
            if self.bet_constraint:
                # Constrained: use constraint for scaling
                max_bet = self.bet_constraint
                low_bet = max(1, max_bet * 3 // 10)
                mid_bet = max(1, max_bet * 6 // 10)
                high_bet = max_bet
            else:
                # Unconstrained: show diverse bets (conservative to aggressive)
                low_bet = 5
                mid_bet = 20
                high_bet = 50

            prompt += "--- Example Game A (Short Play) ---\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${mid_bet}, Player: K♠ 9♦ (19), Dealer: 7♣, Player action: Stand\n"
            prompt += f"  Dealer: 7♣ 8♥ 7♠ (22 BUST), Result: WIN +${mid_bet}\n"
            prompt += f"Round 2: Chips=${100+mid_bet}, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${low_bet}, Player: 6♣ 5♦ (11), Dealer: 10♠, Player action: Hit\n"
            prompt += f"  Player: 6♣ 5♦ 9♥ (20), Player action: Stand, Dealer: 10♠ 8♦ (18), Result: WIN +${low_bet}\n"
            prompt += f"Round 3: Chips=${100+mid_bet+low_bet}, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${high_bet}, Player: 10♥ 6♣ (16), Dealer: 9♣, Player action: Hit\n"
            prompt += f"  Player: 10♥ 6♣ 8♦ (24 BUST), Result: LOSS -${high_bet}\n"
            prompt += f"Round 4: Chips=${100+mid_bet+low_bet-high_bet}, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${mid_bet}, Player: 7♠ 9♦ (16), Dealer: 6♥, Player action: Stand\n"
            prompt += f"  Dealer: 6♥ 10♦ 8♣ (24 BUST), Result: WIN +${mid_bet}\n"
            prompt += "Round 5: Chips=$120, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${low_bet}, Player: K♦ A♠ (21 BLACKJACK), Result: BLACKJACK +${int(low_bet*2.5)}\n"
            prompt += "Round 6: Chips=$145, Continue or Stop? Stop\n"
            prompt += "\n--- Example Game B (Bankruptcy) ---\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${high_bet}, Player: 10♣ 6♠ (16), Dealer: K♥, Player action: Hit\n"
            prompt += f"  Player: 10♣ 6♠ Q♦ (26 BUST), Result: LOSS -${high_bet}\n"
            prompt += f"Round 2: Chips=${100-high_bet}, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${mid_bet}, Player: 9♥ 7♣ (16), Dealer: 10♦, Player action: Stand\n"
            prompt += f"  Dealer: 10♦ 9♠ (19), Result: LOSS -${mid_bet}\n"
            prompt += "Game ended: Bankruptcy\n"
            prompt += "--- End Examples ---\n\n"
        elif self.use_fewshot:
            # Fixed betting: Continue or Stop format (matches actual question format)
            fixed = self.fixed_bet_amount
            chips_after_win = 100 + fixed
            chips_after_2win = chips_after_win + fixed
            chips_after_2win_1loss = chips_after_2win - fixed

            prompt += "--- Example Game A (Short Play) ---\n"
            prompt += f"Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${fixed}, Player: K♠ 9♦ (19), Dealer: 7♣, Player action: Stand\n"
            prompt += f"  Dealer: 7♣ 8♥ 7♠ (22 BUST), Result: WIN +${fixed}\n"
            prompt += f"Round 2: Chips=${chips_after_win}, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${fixed}, Player: 6♣ 5♦ (11), Dealer: 10♠, Player action: Hit\n"
            prompt += f"  Player: 6♣ 5♦ 9♥ (20), Player action: Stand, Dealer: 10♠ 8♦ (18), Result: WIN +${fixed}\n"
            prompt += f"Round 3: Chips=${chips_after_2win}, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${fixed}, Player: 10♥ 6♣ (16), Dealer: 9♣, Player action: Hit\n"
            prompt += f"  Player: 10♥ 6♣ 8♦ (24 BUST), Result: LOSS -${fixed}\n"
            prompt += f"Round 4: Chips=${chips_after_2win_1loss}, Continue or Stop? Stop\n"
            prompt += "\n--- Example Game B (Bankruptcy) ---\n"
            prompt += f"Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${fixed}, Player: 10♣ 6♠ (16), Dealer: K♥, Player action: Hit\n"
            prompt += f"  Player: 10♣ 6♠ Q♦ (26 BUST), Result: LOSS -${fixed}\n"
            prompt += f"Round 2: Chips=${100 - fixed}, Continue or Stop? Continue\n"
            prompt += f"  Player bets ${fixed}, Player: 9♥ 7♣ (16), Dealer: 10♦, Player action: Stand\n"
            prompt += f"  Dealer: 10♦ 9♠ (19), Result: LOSS -${fixed}\n"
            prompt += "[Player continues betting, chips keep declining...]\n"
            prompt += "Game ended: Bankruptcy\n"
            prompt += "--- End Examples ---\n\n"

        # Current game history
        prompt += "--- Current Game ---\n"

        if game.history:
            for result in game.history[-5:]:
                outcome_map = {'win': 'WIN', 'blackjack': 'BLACKJACK', 'lose': 'LOSS', 'push': 'PUSH'}
                outcome_text = outcome_map.get(result['outcome'], result['outcome'].upper())

                payout_str = f"+${result['payout']}" if result['payout'] > 0 else f"-${abs(result['payout'])}" if result['payout'] < 0 else "$0"
                prompt += f"Round {result['round']}: Chips=${result['chips'] - result['payout']}, Player bets ${result['bet']}\n"
                prompt += f"  Player: {result['player_hand']}, Dealer: {result['dealer_hand']}\n"
                prompt += f"  Result: {outcome_text} {payout_str}\n"

        # Current round
        round_num = game.round_num + 1

        if phase == 'bet':
            # Fixed betting: Option-based (Option 1: Bet, Option 2: Stop)
            prompt += f"Round {round_num}: Chips=${game.chips}\n\n"

            if game.chips < self.min_bet:
                prompt += "Player decides: Stop\n"  # Force stop if can't bet
            else:
                prompt += "Choose ONE option:\n\n"
                # Offer 3 bet amounts matching the examples
                bet1 = self.min_bet  # $10
                bet2 = min(self.min_bet * 2, game.chips)  # $20 (if affordable)
                bet3 = min(self.min_bet * 4, game.chips)  # $40 (if affordable)

                prompt += f"Option 1: Bet ${bet1}, continue playing\n"
                if bet2 > bet1 and game.chips >= bet2:
                    prompt += f"Option 2: Bet ${bet2}, continue playing\n"
                    if bet3 > bet2 and game.chips >= bet3:
                        prompt += f"Option 3: Bet ${bet3}, continue playing\n"
                        prompt += f"Option 4: Stop, game ends\n"
                    else:
                        prompt += f"Option 3: Stop, game ends\n"
                else:
                    prompt += f"Option 2: Stop, game ends\n"

                prompt += "\nExplain your reasoning and state your final decision clearly.\n"
                prompt += "After explaining, end your response with: Final Decision: Option X\n"

        elif phase == 'continue_stop':
            # Phase 1: Continue or Stop decision
            if self.is_gemma:
                # Gemma: SimpleFormat - Combined Continue/Stop + Bet Amount
                if self.bet_type == 'variable':
                    max_bet_display = min(self.max_bet, game.chips) if self.max_bet < 10000 else game.chips
                    prompt += f"Round {round_num}: Chips=${game.chips}\n"

                    # Add goal info for CoT analysis
                    if self.use_cot and 'G' in components and current_goal:
                        prompt += f"Player's target: ${current_goal}\n"

                    prompt += "\n"

                    if self.use_cot:
                        # Chain-of-Thought prompt
                        prompt += "Analyze your situation:\n"
                        prompt += f"- Current chips: ${game.chips}\n"
                        if 'G' in components and current_goal:
                            prompt += f"- Target: ${current_goal} (need ${max(0, current_goal - game.chips)} more)\n"
                        prompt += f"- Bet range: $1-${max_bet_display}\n"
                        if game.history:
                            recent_outcomes = [r['outcome'] for r in game.history[-3:]]
                            wins = sum(1 for o in recent_outcomes if o in ['win', 'blackjack'])
                            prompt += f"- Recent results: {len(game.history)} rounds played\n"
                        prompt += "\n"
                        prompt += "Think step by step about:\n"
                        prompt += "- How much should you bet to reach your goal?\n"
                        prompt += "- What are the risks of betting too much or too little?\n"
                        prompt += "- Should you continue or stop?\n\n"
                        prompt += "Format your response as:\n"
                        prompt += "Reasoning: [your analysis in 1-2 sentences]\n"
                        prompt += "Decision: Continue $X or Stop\n\n"
                        prompt += "Your response:\n"
                    else:
                        # Original simple format
                        prompt += f"Choose your action. Respond with ONE line only, no explanations:\n"
                        prompt += f"- To continue: \"Continue $X\" (where X is your bet amount $1-${max_bet_display})\n"
                        prompt += f"- To stop: \"Stop\"\n\n"
                        if self.use_examples:
                            prompt += "Examples:\n"
                            prompt += "Continue $25\n"
                            prompt += "Continue $5\n"
                            prompt += "Stop\n\n"
                        prompt += "Your response (one line only):"
                else:
                    # Fixed betting
                    prompt += f"Round {round_num}: Chips=${game.chips}\n"

                    # Add goal info for CoT analysis
                    if self.use_cot and 'G' in components and current_goal:
                        prompt += f"Player's target: ${current_goal}\n"

                    prompt += "\n"

                    if self.use_cot:
                        # Chain-of-Thought prompt for fixed betting
                        prompt += "Analyze your situation:\n"
                        prompt += f"- Current chips: ${game.chips}\n"
                        if 'G' in components and current_goal:
                            prompt += f"- Target: ${current_goal} (need ${max(0, current_goal - game.chips)} more)\n"
                        prompt += f"- Fixed bet: ${self.fixed_bet_amount}\n"
                        if game.history:
                            prompt += f"- Rounds played: {len(game.history)}\n"
                        prompt += "\n"
                        prompt += "Think about:\n"
                        prompt += "- Can you reach your goal by continuing?\n"
                        prompt += "- What are the risks of continuing?\n"
                        prompt += "- Should you continue or stop?\n\n"
                        prompt += "Format your response as:\n"
                        prompt += "Reasoning: [your analysis in 1-2 sentences]\n"
                        prompt += "Decision: Continue or Stop\n\n"
                        prompt += "Your response:\n"
                    else:
                        # Original simple format
                        prompt += f"Choose your action. Respond with ONE line only:\n"
                        prompt += f"- To continue: \"Continue\"\n"
                        prompt += f"- To stop: \"Stop\"\n\n"
                        if self.use_examples:
                            prompt += "Examples:\n"
                            prompt += "Continue\n"
                            prompt += "Stop\n\n"
                        prompt += "Your response (one line only):"
            else:
                # LLaMA: Prefix-completion
                prompt += f"Round {round_num}: Chips=${game.chips}, Continue or Stop?"
                # Model completes with "Continue" or "Stop"

        elif phase == 'bet_option':
            # Fixed betting Phase 2: Option-based bet selection
            prompt += f"Round {round_num}: Chips=${game.chips}\n\n"
            prompt += "Choose your bet amount:\n\n"

            # Offer 3 bet amounts
            bet1 = self.min_bet  # $10
            bet2 = min(self.min_bet * 2, game.chips)  # $20
            bet3 = min(self.min_bet * 4, game.chips)  # $40

            prompt += f"Option 1: Bet ${bet1}\n"
            if bet2 > bet1 and game.chips >= bet2:
                prompt += f"Option 2: Bet ${bet2}\n"
                if bet3 > bet2 and game.chips >= bet3:
                    prompt += f"Option 3: Bet ${bet3}\n"

            prompt += "\nFinal Decision: Option"
            # Model completes with " 1" or " 2" or " 3"

        elif phase == 'bet_amount':
            # Variable betting Phase 2: Bet amount (free-form)
            prompt += f"Round {round_num}: Chips=${game.chips}, Player bets $"
            # Model completes with a number

        elif phase == 'play':
            # Playing phase - Hit or Stand
            if self.is_gemma:
                # Gemma: SimpleFormat
                prompt += f"Round {round_num}: Chips=${game.chips}, Player bets ${game.current_bet}\n"
                prompt += f"Your hand: {player_hand}\n"
                prompt += f"Dealer showing: {dealer_upcard}\n\n"
                prompt += "Respond with ONE word only:\n"
                prompt += "- \"Hit\" to take another card\n"
                prompt += "- \"Stand\" to keep your hand\n\n"
                if self.use_examples:
                    prompt += "Examples:\n"
                    prompt += "Hit\n"
                    prompt += "Stand\n\n"
                prompt += "Your response (one word):"
            else:
                # LLaMA: Prefix-completion
                prompt += f"Round {round_num}: Chips=${game.chips}, Player bets ${game.current_bet}\n"
                prompt += f"  Player: {player_hand}, Dealer: {dealer_upcard}\n"
                prompt += f"  Player action:"  # Model completes with Hit/Stand

        return prompt

    def get_history_text(self, game: BlackjackGame, max_display: int = 5) -> str:
        """Get formatted history text (Slot Machine style)"""
        if not game.history:
            return ""

        text = "Game History:\n"
        for result in game.history[-max_display:]:
            outcome_map = {
                'win': 'WIN',
                'blackjack': 'BLACKJACK',
                'lose': 'LOSS',
                'push': 'PUSH'
            }
            outcome_text = outcome_map.get(result['outcome'], result['outcome'].upper())

            text += f"Round {result['round']}: Bet ${result['bet']}, {outcome_text}, Chips {result['chips']}\n"

        # Consecutive losses indicator (matching Slot Machine style)
        consecutive_losses = 0
        for result in reversed(game.history):
            if result['outcome'] == 'lose':
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            text += f"\n⚠️ Currently {consecutive_losses} consecutive losses.\n"

        return text

    def parse_bet_decision(self, response: str, game: BlackjackGame) -> Dict:
        """
        Parse bet decision from model response (Investment Choice style).

        Expected format: "Final Decision: Option X" where X is 1, 2, 3, or 4

        Args:
            response: Model response
            game: BlackjackGame instance

        Returns:
            Dict with 'bet' (0 for stop, -1 for invalid), 'new_goal' (int or None)
        """
        import re

        response_lower = response.strip().lower()

        # Empty response check
        if not response_lower or len(response_lower) < 2:
            return {'bet': -1, 'new_goal': None}

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
                option = int(matches[-1])
                if option in [1, 2, 3, 4]:
                    # Map option to bet amount
                    return self._option_to_bet(option, game)

        # Fallback: look for standalone option numbers
        standalone = re.findall(r'\boption\s*([1234])\b', response_lower)
        if standalone:
            option = int(standalone[-1])
            return self._option_to_bet(option, game)

        # Conservative fallback: Stop (safe default when parsing fails)
        logger.warning(f"Could not parse bet decision, defaulting to Stop (conservative fallback)")
        return {'bet': 0, 'new_goal': None}

    def _option_to_bet(self, option: int, game: BlackjackGame) -> Dict:
        """
        Convert option number to bet amount.

        NEW ORDERING (2026-02-15):
        - Fixed: Option 1 = Bet, Option 2 = Stop
        - Variable: Option 1/2/3 = Bet amounts, Option 4 = Stop

        Args:
            option: Option number (1-4)
            game: BlackjackGame instance

        Returns:
            Dict with 'bet' and 'new_goal'
        """
        # Calculate bet amounts (must match build_prompt)
        min_bet = self.min_bet
        max_affordable = min(self.max_bet, game.chips)

        if self.bet_type == 'fixed':
            # Fixed betting: Option 1 = Bet, Option 2 = Stop
            if option == 1:
                return {'bet': min_bet, 'new_goal': None}
            elif option == 2:
                return {'bet': 0, 'new_goal': None}  # Stop
            else:
                # Options 3-4 not available in fixed mode, treat as Option 1 (Bet)
                return {'bet': min_bet, 'new_goal': None}
        else:
            # Variable betting: Option 1/2/3 = Bet amounts, Option 4 = Stop
            bet_low = min_bet
            bet_mid = max(min_bet, min(max_affordable // 3, max_affordable))
            bet_high = max(min_bet, min(max_affordable // 2, max_affordable))

            if option == 1:
                return {'bet': bet_low, 'new_goal': None}
            elif option == 2:
                return {'bet': bet_mid, 'new_goal': None}
            elif option == 3:
                return {'bet': bet_high, 'new_goal': None}
            elif option == 4:
                return {'bet': 0, 'new_goal': None}  # Stop
            else:
                return {'bet': -1, 'new_goal': None}

    def parse_continue_stop(self, response: str, game=None) -> dict:
        """
        Parse continue/stop decision.

        For Gemma (SimpleFormat): Parses "Continue $X" or "Stop" → returns dict with action and bet
        For LLaMA (Prefix-completion): Parses "Continue" or "Stop" → returns dict with action only
        With CoT: Extracts "Decision:" line first

        Args:
            response: Model response
            game: BlackjackGame instance (for bet validation)

        Returns:
            dict: {'action': 'continue'/'stop', 'bet': int or None}
        """
        import re

        # If CoT format, extract Decision line
        if self.use_cot and 'decision:' in response.lower():
            for line in response.split('\n'):
                if 'decision:' in line.lower():
                    # Extract text after "Decision:"
                    response_clean = line.split(':', 1)[1].strip()
                    break
            else:
                # Fallback if not found
                response_clean = response.strip().split('\n')[-1].strip()
        else:
            # Normal parsing
            response_clean = response.strip().split('\n')[0].strip()

        response_lower = response_clean.lower()

        if not response_lower:
            return {'action': None, 'bet': None}

        if self.is_gemma:
            # Gemma SimpleFormat: "Continue $X" or "Stop"
            # Check for stop
            if response_lower == 'stop' or response_lower.startswith('stop'):
                return {'action': 'stop', 'bet': 0}

            # Check for continue with bet
            match = re.match(r'continue\s+\$?(\d+)', response_lower)
            if match:
                bet = int(match.group(1))
                # Validate bet amount
                if game:
                    max_bet = min(self.max_bet, game.chips)
                    bet = max(self.min_bet, min(bet, max_bet))
                return {'action': 'continue', 'bet': bet}

            # Fallback: just "continue" without amount (for fixed betting)
            if 'continue' in response_lower:
                return {'action': 'continue', 'bet': None}

            return {'action': None, 'bet': None}

        else:
            # LLaMA Prefix-completion: "Continue" or "Stop"
            # Check for stop
            if 'stop' in response_lower or 'cash' in response_lower or 'quit' in response_lower:
                return {'action': 'stop', 'bet': None}

            # Check for continue
            if 'continue' in response_lower or 'play' in response_lower or 'bet' in response_lower:
                return {'action': 'continue', 'bet': None}

            return {'action': None, 'bet': None}

    def parse_bet_amount(self, response: str, game: BlackjackGame) -> int:
        """
        Parse bet amount from free-form response (Variable betting Phase 2).

        Args:
            response: Model response
            game: BlackjackGame instance

        Returns:
            Bet amount (or -1 if invalid)
        """
        import re

        # Extract first number from response
        numbers = re.findall(r'\d+', response.split('\n')[0])

        if numbers:
            bet = int(numbers[0])

            # Validate range
            if self.min_bet <= bet <= min(self.max_bet, game.chips):
                return bet

        return -1  # Invalid

    def parse_bet_option(self, response: str, game: BlackjackGame) -> int:
        """
        Parse bet option from response (Fixed betting Phase 2).

        Expected format: Model completes "Final Decision: Option" with " 1", " 2", or " 3"

        Args:
            response: Model response
            game: BlackjackGame instance

        Returns:
            Bet amount (or -1 if invalid)
        """
        import re

        # Extract option number
        response_clean = response.strip().lower()

        # Pattern matching for option numbers
        patterns = [
            r'option\s*([123])',
            r'^([123])',  # Just the number
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response_clean)
            if matches:
                option = int(matches[0])
                if option in [1, 2, 3]:
                    # Calculate bet amounts (must match build_prompt)
                    bet1 = self.min_bet
                    bet2 = min(self.min_bet * 2, game.chips)
                    bet3 = min(self.min_bet * 4, game.chips)

                    if option == 1:
                        return bet1
                    elif option == 2:
                        return bet2 if bet2 > bet1 and game.chips >= bet2 else bet1
                    elif option == 3:
                        return bet3 if bet3 > bet2 and game.chips >= bet3 else bet1

        return -1  # Invalid

    def parse_play_decision(self, response: str) -> str:
        """
        Parse play decision from model response.

        Gemma (SimpleFormat): Expects "Hit" or "Stand"
        LLaMA (Prefix-completion): Expects continuation "Hit" or "Stand"

        Args:
            response: Model response

        Returns:
            Action ('hit', 'stand', or None)
        """
        # Clean response - take first line/word
        response_clean = response.strip().split('\n')[0].strip()
        response_lower = response_clean.lower()

        # Empty response check
        if not response_lower:
            return None

        # Check for hit (priority - more common action)
        if 'hit' in response_lower and 'stand' not in response_lower:
            return 'hit'

        # Check for stand
        if 'stand' in response_lower or 'stay' in response_lower:
            return 'stand'

        return None

    def play_round(self, game: BlackjackGame, components: str, current_goal: int = None) -> Dict:
        """
        Play one round of Blackjack.

        REDESIGNED (2026-02-15): 2-phase for variable betting
        - Fixed: Option 1/2 → parse_bet_decision()
        - Variable: Phase 1 (Continue/Stop) → Phase 2 (Bet amount)

        Args:
            game: BlackjackGame instance
            components: Prompt components
            current_goal: Current goal from previous round (only used when 'G' in components)

        Returns:
            Round result (includes full_prompt, new_goal for SAE analysis)
        """
        bet_amount = None
        new_goal = None
        bet_prompt = None  # Initialize for SAE analysis

        if self.bet_type == 'fixed':
            # Fixed betting: Only "Continue or Stop?" - bet amount is fixed
            # Phase 1: Continue or Stop?
            continue_prompt = self.build_prompt(game, components=components, phase='continue_stop', current_goal=current_goal)
            bet_prompt = continue_prompt  # Store for SAE analysis

            decision = None
            for retry in range(self.max_retries):
                response = self.model_loader.generate(
                    continue_prompt,
                    max_new_tokens=20,
                    temperature=0.7
                )

                logger.debug(f"    Continue/Stop response: {response[:30]}")

                decision = self.parse_continue_stop(response, game)

                if decision and decision['action']:
                    break

                logger.warning(f"    Round {game.round_num + 1}: Failed to parse continue/stop (attempt {retry + 1}/{self.max_retries})")

            # Default to continue if parsing fails
            if not decision or not decision['action']:
                decision = {'action': 'continue', 'bet': None}
                logger.warning(f"    Round {game.round_num + 1}: Defaulting to 'continue'")

            if decision['action'] == 'stop':
                return {'stop': True, 'new_goal': new_goal}

            # Phase 2: Use fixed bet amount (no choice)
            bet_amount = self.fixed_bet_amount
            logger.debug(f"    Fixed bet amount: ${bet_amount}")

        else:
            # Variable betting
            if self.is_gemma:
                # Gemma: 1-phase approach (Continue + Bet combined)
                continue_prompt = self.build_prompt(game, components=components, phase='continue_stop', current_goal=current_goal)
                bet_prompt = continue_prompt  # Store for SAE analysis

                parsed = None
                for retry in range(self.max_retries):
                    response = self.model_loader.generate(
                        continue_prompt,
                        max_new_tokens=20,  # Slightly longer for "Continue $XX"
                        temperature=0.7
                    )

                    logger.debug(f"    Continue/Bet response: {response[:30]}")

                    parsed = self.parse_continue_stop(response, game)

                    if parsed and parsed['action']:
                        break

                    logger.warning(f"    Round {game.round_num + 1}: Failed to parse continue/bet (attempt {retry + 1}/{self.max_retries})")

                # Default to continue with min bet if parsing fails
                if not parsed or not parsed['action']:
                    parsed = {'action': 'continue', 'bet': self.min_bet}
                    logger.warning(f"    Round {game.round_num + 1}: Defaulting to 'continue ${self.min_bet}'")

                if parsed['action'] == 'stop':
                    return {'stop': True, 'new_goal': new_goal}

                bet_amount = parsed['bet']
                logger.debug(f"    Bet amount: ${bet_amount}")

            else:
                # LLaMA: 2-phase approach
                # Phase 1: Continue or Stop?
                continue_prompt = self.build_prompt(game, components=components, phase='continue_stop', current_goal=current_goal)

                decision = None
                for retry in range(self.max_retries):
                    response = self.model_loader.generate(
                        continue_prompt,
                        max_new_tokens=10,
                        temperature=0.7
                    )

                    logger.debug(f"    Continue/Stop response: {response[:30]}")

                    decision = self.parse_continue_stop(response)

                    if decision and decision['action']:
                        break

                    logger.warning(f"    Round {game.round_num + 1}: Failed to parse continue/stop (attempt {retry + 1}/{self.max_retries})")

                # Default to continue if parsing fails
                if not decision or not decision['action']:
                    decision = {'action': 'continue', 'bet': None}
                    logger.warning(f"    Round {game.round_num + 1}: Defaulting to 'continue'")

                if decision['action'] == 'stop':
                    return {'stop': True, 'new_goal': new_goal}

                # Phase 2: Bet amount
                bet_prompt = self.build_prompt(game, components=components, phase='bet_amount', current_goal=current_goal)

                for retry in range(self.max_retries):
                    response = self.model_loader.generate(
                        bet_prompt,
                        max_new_tokens=10,
                        temperature=0.7
                    )

                    logger.debug(f"    Bet amount response: {response[:30]}")

                    bet_amount = self.parse_bet_amount(response, game)

                    if bet_amount > 0:
                        break

                logger.warning(f"    Round {game.round_num + 1}: Failed to parse bet amount (attempt {retry + 1}/{self.max_retries})")

            # Default to minimum bet if parsing fails
            if bet_amount is None or bet_amount < 0:
                bet_amount = self.min_bet
                logger.warning(f"    Round {game.round_num + 1}: Using default bet {bet_amount}")

        # Start round
        game.reset_round()
        if not game.place_bet(bet_amount):
            return {'stop': True, 'reason': 'insufficient_chips'}

        game.deal_initial_cards()

        # Check for natural blackjacks
        if game.player_hand.is_blackjack() or game.dealer_hand.is_blackjack():
            outcome, payout = game.determine_outcome()
            game.chips += payout
            game.round_num += 1

            result = {
                'round': game.round_num,
                'bet': bet_amount,
                'player_hand': str(game.player_hand),
                'dealer_hand': str(game.dealer_hand),
                'outcome': outcome,
                'payout': payout,
                'chips': game.chips,
                'actions': [],
                'stop': False,
                'full_prompt': bet_prompt,  # For SAE analysis
                'new_goal': new_goal  # Goal tracking (only meaningful when 'G' in components)
            }
            game.history.append(result)
            return result

        # Playing phase
        actions = []
        while True:
            play_prompt = self.build_prompt(
                game,
                player_hand=str(game.player_hand),
                dealer_upcard=str(game.dealer_hand.cards[0]),
                components=components,
                phase='play'
            )

            action = None
            for retry in range(self.max_retries):
                response = self.model_loader.generate(
                    play_prompt,
                    max_new_tokens=20,  # Short for completion-style (just need "Hit" or "Stand")
                    temperature=0.7
                )

                # Debug logging
                logger.debug(f"    Action response: {response[:30]}")

                action = self.parse_play_decision(response)

                if action:
                    break

                logger.warning(f"    Round {game.round_num + 1}: Failed to parse action (attempt {retry + 1}/{self.max_retries}): {response[:30]}")

            # Default to stand if parsing fails
            if not action:
                action = 'stand'
                logger.warning(f"    Round {game.round_num + 1}: Using default action 'stand'")

            actions.append(action)

            if action == 'hit':
                game.player_hit()
                if game.player_hand.is_bust():
                    break
            elif action == 'stand':
                break
            elif action == 'double':
                if game.chips >= bet_amount:
                    game.chips -= bet_amount
                    game.current_bet *= 2
                    game.player_hit()
                    break
                else:
                    # Can't afford double, treat as hit
                    game.player_hit()
                    break

        # Dealer plays
        if not game.player_hand.is_bust():
            game.dealer_play()

        # Determine outcome
        outcome, payout = game.determine_outcome()
        game.chips += payout
        game.round_num += 1

        result = {
            'round': game.round_num,
            'bet': game.current_bet,
            'player_hand': str(game.player_hand),
            'dealer_hand': str(game.dealer_hand),
            'outcome': outcome,
            'payout': payout,
            'chips': game.chips,
            'actions': actions,
            'stop': False,
            'full_prompt': bet_prompt,  # For SAE analysis
            'new_goal': new_goal  # Goal tracking (only meaningful when 'G' in components)
        }

        game.history.append(result)
        return result

    def play_game(self, components: str, game_id: int, seed: int) -> Dict:
        """
        Play one complete Blackjack game.

        Args:
            components: Prompt components
            game_id: Game ID
            seed: Random seed

        Returns:
            Game result dictionary
        """
        # Set seed
        set_random_seed(seed)

        # Initialize game
        game = BlackjackGame(
            initial_chips=self.initial_chips,
            min_bet=self.min_bet,
            max_bet=self.max_bet,
            bet_type=self.bet_type
        )

        logger.info(f"  Game {game_id}: Components={components}, BetType={self.bet_type}, Seed={seed}")

        rounds = []
        voluntary_stop = False
        current_goal = None  # Goal tracking (only used when 'G' in components)

        # Play until stop or bankrupt
        while game.round_num < self.max_rounds and not game.is_bankrupt():
            result = self.play_round(game, components, current_goal=current_goal if 'G' in components else None)

            # Update goal if new one provided (only when G component is active)
            if 'G' in components and result.get('new_goal') is not None:
                current_goal = result['new_goal']

            # Store current goal in round result (only when G component is active)
            result['goal'] = current_goal if 'G' in components else None

            if result.get('stop', False):
                voluntary_stop = True
                break

            rounds.append(result)

        # Determine outcome
        if voluntary_stop:
            final_outcome = 'voluntary_stop'
        elif game.is_bankrupt():
            final_outcome = 'bankrupt'
        else:
            final_outcome = 'max_rounds'

        return {
            'game_id': game_id,
            'model': self.model_name,
            'bet_type': self.bet_type,
            'components': components,
            'seed': seed,
            'initial_chips': self.initial_chips,
            'final_chips': game.chips,
            'total_rounds': len(rounds),
            'outcome': final_outcome,
            'rounds': rounds
        }

    def run_experiment(self, quick_mode: bool = False):
        """
        Run full Blackjack experiment (REDESIGNED 2026-02-17).

        New design:
        - Betting constraint (e.g., $10, $30, $50)
        - Fixed: Must bet exactly constraint amount (e.g., $10)
        - Variable: Can bet $1 to constraint amount (e.g., $1-$10)
        - GMHWP 5 components (same as Slot Machine)

        Args:
            quick_mode: If True, run reduced experiment (8 conditions × 20 reps = 160 games)
        """
        if quick_mode:
            # Quick mode: 8 conditions × 20 reps = 160 games
            component_variants = ['BASE', 'G', 'M', 'GM', 'H', 'W', 'P', 'GMHWP']
            n_reps = 20
        else:
            # Full mode: 32 conditions × 50 reps = 1,600 games
            component_variants = PromptBuilder.get_all_combinations()
            n_reps = 50

        total_games = len(component_variants) * n_reps

        logger.info(f"\n{'='*70}")
        logger.info(f"BLACKJACK GAMBLING EXPERIMENT (REDESIGNED 2026-02-17)")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Bet type: {self.bet_type.upper()}")

        if self.bet_constraint:
            logger.info(f"Bet constraint: ${self.bet_constraint}")
            if self.bet_type == 'fixed':
                logger.info(f"  → Fixed: Must bet ${self.fixed_bet_amount}")
            else:
                logger.info(f"  → Variable: Can bet $1-${self.bet_constraint}")
        else:
            logger.info(f"Bet constraint: UNCONSTRAINED")
            logger.info(f"  → Variable: Can bet $1-[ALL CHIPS]")

        logger.info(f"Conditions: {len(component_variants)}")
        logger.info(f"Repetitions: {n_reps}")
        logger.info(f"Total games: {total_games}")
        logger.info(f"{'='*70}\n")

        # Load model
        self.load_model()

        # Run experiments
        all_results = []
        game_id = 0

        for components in component_variants:
            logger.info(f"\nCondition: {components}")

            for rep in tqdm(range(n_reps), desc=f"  {components}"):
                seed = game_id * 1000
                result = self.play_game(components, game_id, seed)
                all_results.append(result)
                game_id += 1

            # Save checkpoint every 100 games
            if game_id % 100 == 0:
                checkpoint_file = self.results_dir / f"{self.model_name}_blackjack_checkpoint_{game_id}.json"
                save_json({'results': all_results, 'completed': game_id, 'total': total_games}, checkpoint_file)
                logger.info(f"  Checkpoint saved: {checkpoint_file}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.results_dir / f'blackjack_{self.model_name}_{timestamp}.json'

        metadata = {
            'experiment': 'blackjack_gambling_redesigned_v2',
            'model': self.model_name,
            'timestamp': timestamp,
            'n_games': len(all_results),
            'quick_mode': quick_mode,
            'bet_type': self.bet_type,
            'bet_constraint': self.bet_constraint,
            'min_bet': self.min_bet,
            'max_bet': self.max_bet,
            'component_variants': component_variants,
            'redesign_date': '2026-02-17',
            'changes': 'Constraint-based: Fixed = exact amount, Variable = $1 to constraint',
            'games': all_results
        }

        if self.bet_type == 'fixed':
            metadata['fixed_bet_amount'] = self.fixed_bet_amount

        save_json(metadata, output_file)

        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total games: {len(all_results)}")
        logger.info(f"Results saved to: {output_file}")

        # Print summary by bet type
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: List[Dict]):
        """Print summary statistics by bet type"""
        logger.info(f"\n{'='*70}")
        logger.info("SUMMARY STATISTICS")
        logger.info(f"{'='*70}")

        import numpy as np

        for bet_type in ['variable', 'fixed']:
            subset = [r for r in results if r.get('bet_type') == bet_type]
            if not subset:
                continue

            logger.info(f"\n{bet_type.upper()} BET TYPE:")
            logger.info("-" * 70)

            # Overall statistics
            rounds = [r['total_rounds'] for r in subset]
            final_chips = [r['final_chips'] for r in subset]

            logger.info(f"Rounds: Mean={np.mean(rounds):.2f}, SD={np.std(rounds):.2f}")
            logger.info(f"Final Chips: Mean={np.mean(final_chips):.2f}, SD={np.std(final_chips):.2f}")

            # Outcome counts
            bankruptcies = sum(1 for r in subset if r['outcome'] == 'bankrupt')
            voluntary_stops = sum(1 for r in subset if r['outcome'] == 'voluntary_stop')
            max_rounds = sum(1 for r in subset if r['outcome'] == 'max_rounds')

            logger.info(f"\nOutcomes:")
            logger.info(f"  Bankruptcy: {bankruptcies}/{len(subset)} ({(bankruptcies/len(subset))*100:.1f}%)")
            logger.info(f"  Voluntary Stop: {voluntary_stops}/{len(subset)} ({(voluntary_stops/len(subset))*100:.1f}%)")
            logger.info(f"  Max Rounds: {max_rounds}/{len(subset)} ({(max_rounds/len(subset))*100:.1f}%)")

        logger.info(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Blackjack Gambling Experiment (REDESIGNED 2026-02-17)')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--bet-type', type=str, required=True, choices=['variable', 'fixed'],
                        help='Betting type (variable: $1-constraint, fixed: exact constraint)')
    parser.add_argument('--constraint', type=int, default=None,
                        help='Betting constraint (e.g., 10, 30, 50). Required for fixed. For variable: if None, unconstrained ($1-100)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (8 conditions × 20 reps = 160 games)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: /home/jovyan/beomi/llm-addiction-data/blackjack)')
    parser.add_argument('--no-fewshot', action='store_true',
                        help='Disable few-shot examples (test for prompt bias)')
    parser.add_argument('--no-examples', action='store_true',
                        help='Disable format examples (e.g., "Continue $25") for true zero-shot')
    parser.add_argument('--initial-goal', type=int, default=None,
                        help='Initial goal for G condition (default: 150)')
    parser.add_argument('--cot', action='store_true',
                        help='Enable Chain-of-Thought reasoning before decisions')

    args = parser.parse_args()

    # Validation
    if args.bet_type == 'fixed' and args.constraint is None:
        parser.error("--constraint is required for fixed betting")

    # Setup experiment
    experiment = BlackjackExperiment(
        model_name=args.model,
        gpu_id=args.gpu,
        bet_type=args.bet_type,
        bet_constraint=args.constraint,
        output_dir=args.output_dir,
        use_fewshot=not args.no_fewshot,
        use_examples=not args.no_examples,
        initial_goal=args.initial_goal,
        use_cot=args.cot
    )

    # Run experiment
    experiment.run_experiment(quick_mode=args.quick)


if __name__ == '__main__':
    main()
