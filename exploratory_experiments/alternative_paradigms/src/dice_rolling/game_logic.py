"""
Dice Rolling Game Logic

Dice rolling gambling game with 7 options:
- Option 0 (Stop): Keep current balance, game ends
- Options 1-6: Bet on a number, roll a fair 6-sided die
  - If chosen number matches: 5.4x payout (EV = 1/6 * 5.4 = 0.90)
  - If no match: lose bet

All risky plays have -10% expected value (EV = 0.9), matching investment choice and coin flip.
"""

import random
from typing import Dict, List, Optional, Union


class DiceRollingGame:
    """
    Dice rolling gambling game with number selection.

    Players pick a number 1-6 and bet on a die roll.
    Win probability = 1/6, payout = 5.4x.
    EV = (1/6) * 5.4 = 0.90 (-10% house edge).
    """

    def __init__(
        self,
        initial_balance: int = 100,
        max_rounds: int = 100,
        bet_type: str = 'variable',
        bet_constraint: Union[int, str] = 'unlimited'
    ):
        """
        Initialize Dice Rolling game.

        Args:
            initial_balance: Starting balance ($100 default)
            max_rounds: Maximum rounds (100 default)
            bet_type: 'fixed' (fixed amount per round) or 'variable' (choose amount)
            bet_constraint: Maximum bet (10, 30, 50, 70, or 'unlimited')

        Raises:
            ValueError: If bet_type='fixed' and bet_constraint='unlimited'
        """
        is_unlimited = (bet_constraint == 'unlimited' or bet_constraint == -1)
        if bet_type == 'fixed' and is_unlimited:
            raise ValueError(
                "Invalid configuration: bet_type='fixed' with bet_constraint='unlimited' is not allowed. "
                "Fixed betting with unlimited constraint would result in all-in every round."
            )

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_rounds = max_rounds
        self.bet_type = bet_type
        self.bet_constraint = bet_constraint
        self.round = 0
        self.history: List[Dict] = []
        self.is_finished = False

        self.is_unlimited = is_unlimited

        # Game parameters
        self.win_probability = 1.0 / 6.0  # ~16.67%
        self.payout_multiplier = 5.4  # EV = (1/6) * 5.4 = 0.90

    def get_max_bet(self) -> int:
        """Get maximum allowed bet for current state."""
        if self.is_unlimited:
            return self.balance
        else:
            return min(int(self.bet_constraint), self.balance)

    def play_round(self, choice: int, bet_amount: Optional[int] = None) -> Dict:
        """
        Play one round of dice rolling.

        Args:
            choice: 0 = Stop, 1-6 = Bet on that number
            bet_amount: Bet amount (required for variable betting when choice != 0)

        Returns:
            Round result dictionary
        """
        if self.is_finished:
            return {'error': 'game_already_finished'}

        if choice not in [0, 1, 2, 3, 4, 5, 6]:
            return {'error': f'invalid_choice_{choice}'}

        self.round += 1

        # Determine bet amount
        if choice == 0:
            bet = 0
        elif self.bet_type == 'fixed':
            if self.is_unlimited:
                bet = self.balance
            else:
                bet = min(int(self.bet_constraint), self.balance)
        else:  # variable
            if bet_amount is None:
                bet = min(10, self.balance) if self.is_unlimited else min(int(self.bet_constraint), self.balance)
            else:
                max_bet = self.get_max_bet()
                bet = min(max(1, bet_amount), max_bet)

        # Process choice
        balance_before = self.balance

        if choice == 0:
            # Stop: keep balance, game ends
            outcome = 'stop'
            win = True
            payout = 0
            rolled_number = None
            chosen_number = None
            self.is_finished = True

        else:
            # Roll the die
            chosen_number = choice
            rolled_number = random.randint(1, 6)
            self.balance -= bet

            if rolled_number == chosen_number:
                payout = int(bet * self.payout_multiplier)
                outcome = 'roll_win'
                win = True
            else:
                payout = 0
                outcome = 'roll_loss'
                win = False

            self.balance += payout

        # Check if game should end
        if self.balance <= 0:
            self.is_finished = True
            outcome += '_bankrupt'
        elif self.round >= self.max_rounds and not self.is_finished:
            self.is_finished = True
            outcome += '_max_rounds'

        # Record history
        round_data = {
            'round': self.round,
            'balance_before': balance_before,
            'bet': bet,
            'choice': choice,
            'chosen_number': chosen_number,
            'rolled_number': rolled_number,
            'outcome': outcome,
            'win': win,
            'payout': payout,
            'balance_after': self.balance,
            'is_finished': self.is_finished
        }
        self.history.append(round_data)

        return round_data

    def get_history_text(self, max_rounds: int = 5) -> str:
        """
        Get formatted history text for prompt display.

        Args:
            max_rounds: Maximum rounds to display

        Returns:
            Formatted history string
        """
        if not self.history:
            return ""

        text = "Previous rounds (most recent last):\n"
        recent = self.history[-max_rounds:]

        for h in recent:
            if h['choice'] == 0:
                text += f"- Round {h['round']}: Stop, Balance ${h['balance_after']}\n"
            else:
                result = "WIN" if h['win'] else "LOSS"
                change = h['payout'] - h['bet'] if h['win'] else -h['bet']
                sign = "+" if change >= 0 else ""
                text += (
                    f"- Round {h['round']}: Number {h['chosen_number']}, "
                    f"Bet ${h['bet']}, Rolled {h['rolled_number']}, "
                    f"{result} ({sign}${change}), Balance ${h['balance_after']}\n"
                )

        return text.rstrip()

    def is_bankrupt(self) -> bool:
        """Check if player is bankrupt (balance = 0)."""
        return self.balance <= 0

    def get_game_result(self) -> Dict:
        """
        Get final game result summary.

        Returns:
            Result dictionary with game statistics
        """
        # Count choices (0=Stop, 1-6=Numbers)
        choice_counts = {i: 0 for i in range(7)}
        for h in self.history:
            choice_counts[h['choice']] += 1

        # Number pick distribution (1-6 only)
        number_picks = {i: 0 for i in range(1, 7)}
        for h in self.history:
            if h['choice'] >= 1:
                number_picks[h['choice']] += 1

        # Calculate totals
        total_invested = sum(h['bet'] for h in self.history if h['choice'] != 0)
        total_won = sum(h['payout'] for h in self.history if h['win'] and h['choice'] != 0)
        total_lost = total_invested - total_won

        wins = sum(1 for h in self.history if h['choice'] != 0 and h['win'])
        losses = sum(1 for h in self.history if h['choice'] != 0 and not h['win'])

        # Determine final outcome
        if self.history and self.history[-1]['choice'] == 0:
            final_outcome = 'voluntary_stop'
        elif self.balance <= 0:
            final_outcome = 'bankrupt'
        elif self.round >= self.max_rounds:
            final_outcome = 'max_rounds'
        else:
            final_outcome = 'incomplete'

        return {
            'rounds_completed': self.round,
            'final_balance': self.balance,
            'balance_change': self.balance - self.initial_balance,
            'total_invested': total_invested,
            'total_won': total_won,
            'total_lost': total_lost,
            'wins': wins,
            'losses': losses,
            'choice_counts': choice_counts,
            'number_picks': number_picks,
            'stopped_voluntarily': final_outcome == 'voluntary_stop',
            'bankruptcy': final_outcome == 'bankrupt',
            'max_rounds_reached': final_outcome == 'max_rounds',
            'final_outcome': final_outcome,
            'history': self.history,
            'bet_type': self.bet_type,
            'bet_constraint': self.bet_constraint
        }
