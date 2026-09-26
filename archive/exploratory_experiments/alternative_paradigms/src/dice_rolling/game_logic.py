"""
Dice Rolling Game Logic

Simple dice betting game for testing Fixed vs Variable betting effects.
Players bet on a number (1-6) and win 5x if correct.
"""

import random
from typing import Dict, List, Optional


class DiceRollingGame:
    """Dice rolling gambling game"""

    def __init__(
        self,
        initial_chips: int = 100,
        min_bet: int = 1,
        max_bet: int = 50,
        bet_type: str = 'variable',
        fixed_bet_amount: int = None,
        num_sides: int = 6,
        payout_multiplier: int = 5
    ):
        """
        Initialize Dice Rolling game.

        Args:
            initial_chips: Starting chip amount
            min_bet: Minimum bet
            max_bet: Maximum bet (for variable betting)
            bet_type: 'variable' or 'fixed'
            fixed_bet_amount: Fixed bet amount (required for 'fixed' type)
            num_sides: Number of dice sides (default: 6)
            payout_multiplier: Payout multiplier on win (default: 5)
        """
        self.initial_chips = initial_chips
        self.chips = initial_chips
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bet_type = bet_type
        self.fixed_bet_amount = fixed_bet_amount
        self.num_sides = num_sides
        self.payout_multiplier = payout_multiplier

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

    def validate_target(self, target_number: int) -> bool:
        """
        Validate target number.

        Args:
            target_number: Target number (1-6)

        Returns:
            True if valid, False otherwise
        """
        return 1 <= target_number <= self.num_sides

    def roll_dice(self) -> int:
        """
        Roll dice and return result.

        Returns:
            Dice result (1-6)
        """
        return random.randint(1, self.num_sides)

    def play_round(self, bet_amount: int, target_number: int) -> Dict:
        """
        Play a complete round.

        Args:
            bet_amount: Bet amount
            target_number: Target number (1-6)

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
                'target': target_number,
                'chips_before': self.chips,
                'chips_after': self.chips
            }

        # Validate target
        if not self.validate_target(target_number):
            return {
                'round': self.round_num,
                'valid': False,
                'error': 'Invalid target number',
                'bet': bet_amount,
                'target': target_number,
                'chips_before': self.chips,
                'chips_after': self.chips
            }

        # Deduct bet
        chips_before = self.chips
        self.chips -= bet_amount

        # Roll dice
        dice_result = self.roll_dice()
        outcome = 'win' if dice_result == target_number else 'lose'

        # Calculate payout (5x on win, lose bet on loss)
        if outcome == 'win':
            payout = bet_amount * self.payout_multiplier
            self.chips += payout
            profit = bet_amount * (self.payout_multiplier - 1)  # Net profit
        else:
            payout = 0
            profit = -bet_amount  # Net loss

        chips_after = self.chips

        result = {
            'round': self.round_num,
            'valid': True,
            'bet': bet_amount,
            'target': target_number,
            'dice_result': dice_result,
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
            'num_sides': self.num_sides,
            'payout_multiplier': self.payout_multiplier,
            'win_probability': 1.0 / self.num_sides,
            'expected_value': self.payout_multiplier / self.num_sides
        }

    def get_history(self) -> List[Dict]:
        """Get game history"""
        return self.history

    def get_history_text(self, max_rounds: int = 5) -> str:
        """
        Get formatted history text for prompt.

        Args:
            max_rounds: Maximum number of recent rounds to show

        Returns:
            Formatted history string
        """
        if not self.history:
            return "No rounds played yet."

        recent = self.history[-max_rounds:]
        lines = []
        for r in recent:
            if r['valid']:
                outcome_str = f"rolled {r['dice_result']}"
                if r['outcome'] == 'win':
                    lines.append(f"R{r['round']}: Bet ${r['bet']} on {r['target']}, {outcome_str} → WON ${r['payout']}")
                else:
                    lines.append(f"R{r['round']}: Bet ${r['bet']} on {r['target']}, {outcome_str} → LOST ${r['bet']}")
            else:
                lines.append(f"R{r['round']}: Invalid ({r.get('error', 'unknown error')})")

        return "\n".join(lines)
