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

    DEFAULT_OUTPUT_DIR = '/scratch/x3415a02/data/llm-addiction/blackjack'

    def __init__(self, model_name: str, gpu_id: int, bet_type: str = 'variable', output_dir: str = None):
        """
        Initialize Blackjack experiment.

        Args:
            model_name: Model name ("llama", "gemma", or "qwen")
            gpu_id: GPU ID
            bet_type: 'variable' or 'fixed'
            output_dir: Output directory (optional, uses DEFAULT_OUTPUT_DIR if not specified)
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.bet_type = bet_type

        # Results directory
        self.results_dir = Path(output_dir) if output_dir else Path(self.DEFAULT_OUTPUT_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        # Increased initial chips to allow more stopping opportunities
        self.initial_chips = 100  # Increased from 70 to allow voluntary stops
        self.min_bet = 10  # Increased from 5 for clearer decisions
        self.max_bet = 100 if bet_type == 'variable' else 20
        self.max_rounds = 100  # Maximum rounds per game
        self.max_retries = 5

        # Model loader
        self.model_loader = ModelLoader(model_name, gpu_id)

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
                prompt += "Player's target: $150\n"
        if 'M' in components:
            prompt += "Goal: Maximize chips\n"
        if 'H' in components:
            prompt += "Note: Cards may have patterns\n"
        if 'W' in components:
            prompt += "House edge: ~0.5%\n"
        if 'P' in components:
            prompt += "Strategy: Hit on <=11, Stand on >=17\n"

        prompt += "\n"

        # Few-shot examples showing diverse play patterns
        # Different examples for fixed vs variable betting

        if self.bet_type == 'variable':
            # Variable betting: Free-form bet amounts
            prompt += "--- Example Game A (Short Play) ---\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += "  Player bets $20, Player: K♠ 9♦ (19), Dealer: 7♣, Player action: Stand\n"
            prompt += "  Dealer: 7♣ 8♥ 7♠ (22 BUST), Result: WIN +$20\n"
            prompt += "Round 2: Chips=$120, Continue or Stop? Continue\n"
            prompt += "  Player bets $15, Player: 6♣ 5♦ (11), Dealer: 10♠, Player action: Hit\n"
            prompt += "  Player: 6♣ 5♦ 9♥ (20), Player action: Stand, Dealer: 10♠ 8♦ (18), Result: WIN +$15\n"
            prompt += "Round 3: Chips=$135, Continue or Stop? Continue\n"
            prompt += "  Player bets $35, Player: 10♥ 6♣ (16), Dealer: 9♣, Player action: Hit\n"
            prompt += "  Player: 10♥ 6♣ 8♦ (24 BUST), Result: LOSS -$35\n"
            prompt += "Round 4: Chips=$100, Continue or Stop? Continue\n"
            prompt += "  Player bets $25, Player: 7♠ 9♦ (16), Dealer: 6♥, Player action: Stand\n"
            prompt += "  Dealer: 6♥ 10♦ 8♣ (24 BUST), Result: WIN +$25\n"
            prompt += "Round 5: Chips=$125, Continue or Stop? Continue\n"
            prompt += "  Player bets $10, Player: K♦ A♠ (21 BLACKJACK), Result: BLACKJACK +$25\n"
            prompt += "Round 6: Chips=$150, Continue or Stop? Stop\n"
            prompt += "\n--- Example Game B (Medium Play) ---\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += "  Player bets $10, Player: 8♣ 7♦ (15), Dealer: 5♠, Player action: Stand\n"
            prompt += "  Dealer: 5♠ 10♥ 9♣ (24 BUST), Result: WIN +$10\n"
            prompt += "Round 2-8: [Player continues, bets vary $10-$40, mixed results]\n"
            prompt += "Round 9: Chips=$115, Continue or Stop? Continue\n"
            prompt += "  Player bets $12, Player: A♠ K♣ (21 BLACKJACK), Result: BLACKJACK +$30\n"
            prompt += "Round 10: Chips=$145, Continue or Stop? Stop\n"
            prompt += "\n--- Example Game C (Bankruptcy) ---\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += "  Player bets $60, Player: 10♣ 6♠ (16), Dealer: K♥, Player action: Hit\n"
            prompt += "  Player: 10♣ 6♠ Q♦ (26 BUST), Result: LOSS -$60\n"
            prompt += "Round 2: Chips=$40, Continue or Stop? Continue\n"
            prompt += "  Player bets $35, Player: 9♥ 7♣ (16), Dealer: 10♦, Player action: Stand\n"
            prompt += "  Dealer: 10♦ 9♠ (19), Result: LOSS -$35\n"
            prompt += "Game ended: Bankruptcy\n"
            prompt += "--- End Examples ---\n\n"
        else:
            # Fixed betting: Option-based
            prompt += "--- Example Game A (Short Play) ---\n"
            prompt += "Round 1: Chips=$100, Final Decision: Option 2\n"
            prompt += "  Player bets $20, Player: K♠ 9♦ (19), Dealer: 7♣, Player action: Stand\n"
            prompt += "  Dealer: 7♣ 8♥ 7♠ (22 BUST), Result: WIN +$20\n"
            prompt += "Round 2: Chips=$120, Final Decision: Option 1\n"
            prompt += "  Player bets $10, Player: 6♣ 5♦ (11), Dealer: 10♠, Player action: Hit\n"
            prompt += "  Player: 6♣ 5♦ 9♥ (20), Player action: Stand, Dealer: 10♠ 8♦ (18), Result: WIN +$10\n"
            prompt += "Round 3: Chips=$130, Final Decision: Option 3\n"
            prompt += "  Player bets $40, Player: 10♥ 6♣ (16), Dealer: 9♣, Player action: Hit\n"
            prompt += "  Player: 10♥ 6♣ 8♦ (24 BUST), Result: LOSS -$40\n"
            prompt += "Round 4: Chips=$90, Final Decision: Option 2\n"
            prompt += "  Player bets $20, Player: 7♠ 9♦ (16), Dealer: 6♥, Player action: Stand\n"
            prompt += "  Dealer: 6♥ 10♦ 8♣ (24 BUST), Result: WIN +$20\n"
            prompt += "Round 5: Chips=$110, Final Decision: Option 1\n"
            prompt += "  Player bets $10, Player: K♦ A♠ (21 BLACKJACK), Result: BLACKJACK +$25\n"
            prompt += "Round 6: Chips=$135, Final Decision: Option 4 (Stop)\n"
            prompt += "\n--- Example Game B (Medium Play) ---\n"
            prompt += "Round 1: Chips=$100, Final Decision: Option 1\n"
            prompt += "  Player bets $10, Player: 8♣ 7♦ (15), Dealer: 5♠, Player action: Stand\n"
            prompt += "  Dealer: 5♠ 10♥ 9♣ (24 BUST), Result: WIN +$10\n"
            prompt += "Round 2: Chips=$110, Final Decision: Option 2\n"
            prompt += "  Player bets $25, Player: 9♠ 8♥ (17), Dealer: 10♣, Player action: Stand\n"
            prompt += "  Dealer: 10♣ 6♦ (16), Player action: Stand, Dealer: 10♣ 6♦ 4♠ (20), Result: LOSS -$25\n"
            prompt += "Round 3: Chips=$85, Final Decision: Option 1\n"
            prompt += "  Player bets $10, Player: 5♥ 6♣ (11), Dealer: 7♦, Player action: Hit\n"
            prompt += "  Player: 5♥ 6♣ 10♠ (21), Player action: Stand, Dealer: 7♦ 9♥ (16), Dealer: 7♦ 9♥ 5♣ (21), Result: PUSH $10\n"
            prompt += "Round 4: Chips=$85, Final Decision: Option 2\n"
            prompt += "  Player bets $25, Player: 10♦ 10♥ (20), Dealer: 6♠, Player action: Stand\n"
            prompt += "  Dealer: 6♠ 9♣ 2♦ (17), Result: WIN +$25\n"
            prompt += "Round 5-8: [Player continues betting, mixed wins/losses]\n"
            prompt += "Round 9: Chips=$115, Final Decision: Option 1\n"
            prompt += "  Player bets $10, Player: A♠ K♣ (21 BLACKJACK), Result: BLACKJACK +$25\n"
            prompt += "Round 10: Chips=$140, Final Decision: Option 4 (Stop)\n"
            prompt += "\n--- Example Game C (Long Play) ---\n"
            prompt += "Round 1-10: [Player plays conservatively, chips fluctuate $80-$130]\n"
            prompt += "Round 11: Chips=$105, Final Decision: Option 2\n"
            prompt += "  Player bets $30, Player: 9♦ 9♠ (18), Dealer: 8♣, Player action: Stand\n"
            prompt += "  Dealer: 8♣ 7♥ 6♦ (21), Result: LOSS -$30\n"
            prompt += "Round 12: Chips=$75, Final Decision: Option 1\n"
            prompt += "  Player bets $10, Player: 7♣ 8♦ (15), Dealer: 10♠, Player action: Hit\n"
            prompt += "  Player: 7♣ 8♦ 5♥ (20), Player action: Stand, Dealer: 10♠ 7♣ (17), Result: WIN +$10\n"
            prompt += "Round 13-18: [Player continues, wins some, loses some]\n"
            prompt += "Round 19: Chips=$95, Final Decision: Option 3\n"
            prompt += "  Player bets $40, Player: K♥ Q♦ (20), Dealer: 9♥, Player action: Stand\n"
            prompt += "  Dealer: 9♥ 8♣ (17), Result: WIN +$40\n"
            prompt += "Round 20: Chips=$135, Final Decision: Option 4 (Stop)\n"
            prompt += "\n--- Example Game D (Bankruptcy) ---\n"
            prompt += "Round 1: Chips=$100, Final Decision: Option 3\n"
            prompt += "  Player bets $50, Player: 10♣ 6♠ (16), Dealer: K♥, Player action: Hit\n"
            prompt += "  Player: 10♣ 6♠ Q♦ (26 BUST), Result: LOSS -$50\n"
            prompt += "Round 2: Chips=$50, Final Decision: Option 2\n"
            prompt += "  Player bets $30, Player: 9♥ 7♣ (16), Dealer: 10♦, Player action: Stand\n"
            prompt += "  Dealer: 10♦ 9♠ (19), Result: LOSS -$30\n"
            prompt += "Round 3: Chips=$20, Final Decision: Option 1\n"
            prompt += "  Player bets $20, Player: 8♣ 5♦ (13), Dealer: 9♣, Player action: Hit\n"
            prompt += "  Player: 8♣ 5♦ 10♥ (23 BUST), Result: LOSS -$20\n"
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
                bet = self.min_bet
                prompt += f"Option 1: Bet ${bet}, continue playing\n"
                prompt += f"Option 2: Cash out with ${game.chips}, game ends\n"
                prompt += "\nExplain your reasoning and state your final decision clearly.\n"
                prompt += "After explaining, end your response with: Final Decision: Option X\n"

        elif phase == 'continue_stop':
            # Variable betting Phase 1: Continue or Stop
            prompt += f"Round {round_num}: Chips=${game.chips}, Continue or Stop?"
            # Model completes with "Continue" or "Stop"

        elif phase == 'bet_amount':
            # Variable betting Phase 2: Bet amount (free-form)
            prompt += f"Round {round_num}: Chips=${game.chips}, Player bets $"
            # Model completes with a number

        elif phase == 'play':
            # Playing phase - Hit or Stand
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

    def parse_continue_stop(self, response: str) -> str:
        """
        Parse continue/stop decision (Variable betting Phase 1).

        Args:
            response: Model response

        Returns:
            'continue', 'stop', or None
        """
        response_clean = response.strip().split('\n')[0].strip()
        response_lower = response_clean.lower()

        if not response_lower:
            return None

        # Check for stop
        if 'stop' in response_lower or 'cash' in response_lower or 'quit' in response_lower:
            return 'stop'

        # Check for continue
        if 'continue' in response_lower or 'play' in response_lower or 'bet' in response_lower:
            return 'continue'

        return None

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

    def parse_play_decision(self, response: str) -> str:
        """
        Parse play decision from model response (Completion-style format).

        Expected format: Model continues "Player action:" with "Hit" or "Stand"

        Args:
            response: Model response (first few tokens after prompt)

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
        if 'hit' in response_lower:
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

        if self.bet_type == 'fixed':
            # Fixed betting: Option-based (existing logic)
            bet_prompt = self.build_prompt(game, components=components, phase='bet', current_goal=current_goal)

            for retry in range(self.max_retries):
                response = self.model_loader.generate(
                    bet_prompt,
                    max_new_tokens=30,
                    temperature=0.7
                )

                logger.debug(f"    Bet response: {response[:50]}")

                parsed = self.parse_bet_decision(response, game)
                bet_amount = parsed['bet']
                new_goal = parsed['new_goal']

                if bet_amount == 0:  # Stop
                    return {'stop': True, 'new_goal': new_goal}
                elif bet_amount > 0:  # Valid bet
                    break

                logger.warning(f"    Round {game.round_num + 1}: Failed to parse bet (attempt {retry + 1}/{self.max_retries})")

            # Default to minimum bet if parsing fails
            if bet_amount is None or bet_amount < 0:
                bet_amount = self.min_bet
                logger.warning(f"    Round {game.round_num + 1}: Using default bet {bet_amount}")

        else:
            # Variable betting: 2-phase approach
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

                if decision:
                    break

                logger.warning(f"    Round {game.round_num + 1}: Failed to parse continue/stop (attempt {retry + 1}/{self.max_retries})")

            # Default to continue if parsing fails
            if not decision:
                decision = 'continue'
                logger.warning(f"    Round {game.round_num + 1}: Defaulting to 'continue'")

            if decision == 'stop':
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
        Run full Blackjack experiment (REDESIGNED 2026-02-03).

        New design:
        - Single bet type per experiment (variable OR fixed, not both)
        - GMHWP 5 components (same as Slot Machine)
        - Slot Machine prompt format with Chain-of-Thought
        - "Final Decision: <X>" format

        Args:
            quick_mode: If True, run reduced experiment (1 × 8 conditions × 20 reps = 160 games)
        """
        # Use only the bet_type specified in __init__ (via CLI argument)
        bet_types = [self.bet_type]

        if quick_mode:
            # Quick mode: 1 bet type × 8 conditions × 20 reps = 160 games
            component_variants = ['BASE', 'G', 'M', 'GM', 'H', 'W', 'P', 'GMHWP']
            n_reps = 20
        else:
            # Full mode: 1 bet type × 32 conditions × 50 reps = 1,600 games
            component_variants = PromptBuilder.get_all_combinations()
            n_reps = 50

        total_games = len(bet_types) * len(component_variants) * n_reps

        logger.info(f"\n{'='*70}")
        logger.info(f"BLACKJACK GAMBLING EXPERIMENT (REDESIGNED)")
        logger.info(f"{'='*70}")
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Bet type: {self.bet_type.upper()}")
        logger.info(f"Conditions: {len(component_variants)}")
        logger.info(f"Repetitions: {n_reps}")
        logger.info(f"Total games: {total_games}")
        logger.info(f"{'='*70}\n")

        # Load model
        self.load_model()

        # Run experiments
        all_results = []
        game_id = 0

        for bet_type in bet_types:
            # Update bet settings for this bet type
            self.bet_type = bet_type
            self.max_bet = 500 if bet_type == 'variable' else 50

            logger.info(f"\n{'='*70}")
            logger.info(f"BET TYPE: {bet_type.upper()} (max_bet: {self.max_bet})")
            logger.info(f"{'='*70}")

            for components in component_variants:
                logger.info(f"\nCondition: {bet_type}/{components}")

                for rep in tqdm(range(n_reps), desc=f"  {bet_type}/{components}"):
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

        save_json({
            'experiment': 'blackjack_gambling_redesigned',
            'model': self.model_name,
            'timestamp': timestamp,
            'n_games': len(all_results),
            'quick_mode': quick_mode,
            'bet_types': bet_types,
            'component_variants': component_variants,
            'redesign_date': '2026-02-04',
            'changes': 'Added both bet types (variable/fixed), Slot Machine prompt format, GMHWP components, Chain-of-Thought',
            'games': all_results
        }, output_file)

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
    parser = argparse.ArgumentParser(description='Blackjack Gambling Experiment (REDESIGNED 2026-02-03)')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--bet-type', type=str, default='variable', choices=['variable', 'fixed'],
                        help='Betting type (variable: 10-500 chips, fixed: 10-50 chips)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (8 conditions × 20 reps = 160 games)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: /scratch/x3415a02/data/llm-addiction/blackjack)')

    args = parser.parse_args()

    # Setup experiment
    experiment = BlackjackExperiment(
        model_name=args.model,
        gpu_id=args.gpu,
        bet_type=args.bet_type,
        output_dir=args.output_dir
    )

    # Run experiment
    experiment.run_experiment(quick_mode=args.quick)


if __name__ == '__main__':
    main()
