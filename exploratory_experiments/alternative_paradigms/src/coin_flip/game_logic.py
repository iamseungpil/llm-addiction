"""
Coin Flip Game Logic

Simple gambling game for testing Fixed vs Variable betting effects.
"""

import random
from typing import Dict, List, Optional


class CoinFlipGame:
    """Coin flip gambling game"""

    def __init__(
        self,
        initial_chips: int = 100,
        min_bet: int = 1,
        max_bet: int = 50,
        bet_type: str = 'variable',
        fixed_bet_amount: int = None,
        win_probability: float = 0.45  # Negative EV (45% win, 55% lose)
    ):
        """
        Initialize Coin Flip game.

        Args:
            initial_chips: Starting chip amount
            min_bet: Minimum bet
            max_bet: Maximum bet (for variable betting)
            bet_type: 'variable' or 'fixed'
            fixed_bet_amount: Fixed bet amount (required for 'fixed' type)
            win_probability: Probability of winning (default 0.45 for negative EV)
        """
        self.initial_chips = initial_chips
        self.chips = initial_chips
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bet_type = bet_type
        self.fixed_bet_amount = fixed_bet_amount
        self.win_probability = win_probability

        # Validation
        if bet_type == 'fixed' and fixed_bet_amount is None:
            raise ValueError("Fixed betting requires fixed_bet_amount")

        self.round_num = 0
        self.history: List[Dict] = []

    def place_bet(self, bet_amount: int) -> bool:
        """
        Place bet for current round.

        Args:
            bet_amount: Bet amount

        Returns:
            True if bet is valid, False otherwise
        """
        # Fixed betting: must match fixed amount
        if self.bet_type == 'fixed':
            if bet_amount != self.fixed_bet_amount:
                return False

        # Validate bet range
        if bet_amount < self.min_bet or bet_amount > self.max_bet:
            return False

        # Check if player has enough chips
        if bet_amount > self.chips:
            return False

        return True

    def flip_coin(self) -> str:
        """
        Flip coin and determine outcome.

        Returns:
            'win' or 'lose'
        """
        return 'win' if random.random() < self.win_probability else 'lose'

    def play_round(self, bet_amount: int) -> Dict:
        """
        Play a complete round.

        Args:
            bet_amount: Bet amount

        Returns:
            Round result dictionary
        """
        self.round_num += 1

        # Validate bet
        if not self.place_bet(bet_amount):
            return {
                'round': self.round_num,
                'valid': False,
                'error': 'Invalid bet amount',
                'bet': bet_amount,
                'chips_before': self.chips,
                'chips_after': self.chips
            }

        # Deduct bet
        chips_before = self.chips
        self.chips -= bet_amount

        # Flip coin
        outcome = self.flip_coin()

        # Calculate payout (2x on win, lose bet on loss)
        if outcome == 'win':
            payout = bet_amount * 2
            self.chips += payout
            profit = bet_amount  # Net profit
        else:
            payout = 0
            profit = -bet_amount  # Net loss

        chips_after = self.chips

        result = {
            'round': self.round_num,
            'valid': True,
            'bet': bet_amount,
            'outcome': outcome,
            'payout': payout,
            'profit': profit,
            'chips_before': chips_before,
            'chips_after': chips_after,
            'chips': chips_after
        }

        self.history.append(result)
        return result

    def is_bankrupt(self) -> bool:
        """Check if player is bankrupt"""
        return self.chips < self.min_bet

    def get_status(self) -> Dict:
        """Get current game status"""
        return {
            'chips': self.chips,
            'round_num': self.round_num,
            'bankrupt': self.is_bankrupt(),
            'min_bet': self.min_bet,
            'max_bet': self.max_bet,
            'bet_type': self.bet_type,
            'fixed_bet_amount': self.fixed_bet_amount,
            'win_probability': self.win_probability
        }

    def get_history(self) -> List[Dict]:
        """Get game history"""
        return self.history
