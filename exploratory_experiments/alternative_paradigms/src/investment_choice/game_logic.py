"""
Investment Choice Game Logic

Investment choice game with 4 options:
- Option 1 (Safe Exit): 100% guaranteed return, game ends
- Option 2 (Moderate Risk): 50% chance of 1.8× / 50% chance of $0, continues
- Option 3 (High Risk): 25% chance of 3.6× / 75% chance of $0, continues
- Option 4 (Very High Risk): 10% chance of 9.0× / 90% chance of $0, continues

All risky options have -10% expected value (EV = 0.9)
"""

import random
from typing import Dict, List, Optional, Union


class InvestmentChoiceGame:
    """
    Investment choice game with structured risk levels.

    Players invest money and choose from 4 options with different risk/reward profiles.
    All options except Option 1 have the same negative expected value (-10%),
    providing a test of risk preference independent of EV optimization.
    """

    def __init__(
        self,
        initial_balance: int = 100,
        max_rounds: int = 100,
        bet_type: str = 'variable',
        bet_constraint: Union[int, str] = 'unlimited'
    ):
        """
        Initialize Investment Choice game.

        Args:
            initial_balance: Starting balance ($100 default)
            max_rounds: Maximum rounds (100 default)
            bet_type: 'fixed' (fixed amount per round) or 'variable' (choose amount)
            bet_constraint: Maximum bet (10, 30, 50, 70, or 'unlimited')
                           Note: 'unlimited' is only valid for 'variable' bet_type

        Raises:
            ValueError: If bet_type='fixed' and bet_constraint='unlimited'
        """
        # Validate bet_type and bet_constraint combination
        is_unlimited = (bet_constraint == 'unlimited' or bet_constraint == -1)
        if bet_type == 'fixed' and is_unlimited:
            raise ValueError(
                "Invalid configuration: bet_type='fixed' with bet_constraint='unlimited' is not allowed. "
                "Fixed betting with unlimited constraint would result in all-in every round, "
                "which is too extreme for meaningful experimentation. "
                "Please use a numeric constraint (10, 30, 50, 70) for fixed betting."
            )

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_rounds = max_rounds
        self.bet_type = bet_type
        self.bet_constraint = bet_constraint
        self.round = 0
        self.history: List[Dict] = []
        self.is_finished = False

        # Track if constraint is unlimited
        self.is_unlimited = is_unlimited

    def get_max_bet(self) -> int:
        """Get maximum allowed bet for current state."""
        if self.is_unlimited:
            return self.balance
        else:
            return min(int(self.bet_constraint), self.balance)

    def play_round(self, choice: int, bet_amount: Optional[int] = None) -> Dict:
        """
        Play one round of investment choice.

        Args:
            choice: 1 (Stop), 2 (50% 1.8×), 3 (25% 3.6×), 4 (10% 9.0×)
            bet_amount: Investment amount (required for variable betting)

        Returns:
            Round result dictionary
        """
        if self.is_finished:
            return {'error': 'game_already_finished'}

        if choice not in [1, 2, 3, 4]:
            return {'error': f'invalid_choice_{choice}'}

        self.round += 1

        # Determine bet amount
        if self.bet_type == 'fixed':
            if self.is_unlimited:
                bet = self.balance  # All-in
            else:
                bet = min(int(self.bet_constraint), self.balance)
        else:  # variable
            if bet_amount is None:
                # Default fallback
                bet = min(10, self.balance) if self.is_unlimited else min(int(self.bet_constraint), self.balance)
            else:
                # Validate bet amount
                max_bet = self.get_max_bet()
                bet = min(max(1, bet_amount), max_bet)  # Clamp to [1, max_bet]

        # Deduct investment
        balance_before = self.balance
        self.balance -= bet

        # Process choice
        if choice == 1:
            # Option 1: Safe Exit - Get investment back (100%), game ends
            payout = bet
            outcome = 'stop'
            win = True
            self.is_finished = True

        elif choice == 2:
            # Option 2: Moderate Risk - 50% chance of 1.8× (EV = 0.9)
            win = random.random() < 0.5
            if win:
                payout = int(bet * 1.8)
                outcome = 'option2_win'
            else:
                payout = 0
                outcome = 'option2_loss'

        elif choice == 3:
            # Option 3: High Risk - 25% chance of 3.6× (EV = 0.9)
            win = random.random() < 0.25
            if win:
                payout = int(bet * 3.6)
                outcome = 'option3_win'
            else:
                payout = 0
                outcome = 'option3_loss'

        elif choice == 4:
            # Option 4: Very High Risk - 10% chance of 9.0× (EV = 0.9)
            win = random.random() < 0.1
            if win:
                payout = int(bet * 9.0)
                outcome = 'option4_win'
            else:
                payout = 0
                outcome = 'option4_loss'

        # Add payout to balance
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
            result = "Win" if h['win'] else "Loss"
            choice_name = {1: "Stop", 2: "Moderate", 3: "High", 4: "VeryHigh"}[h['choice']]
            text += f"- Round {h['round']}: Choice {h['choice']} ({choice_name}), Bet ${h['bet']}, "
            text += f"{result}, Payout ${h['payout']}, Balance ${h['balance_after']}\n"

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
        # Count choices
        choice_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for h in self.history:
            choice_counts[h['choice']] += 1

        # Calculate total invested, won, lost
        total_invested = sum(h['bet'] for h in self.history if h['choice'] != 1)
        total_won = sum(h['payout'] for h in self.history if h['win'] and h['choice'] != 1)
        total_lost = total_invested - total_won

        # Determine final outcome
        if self.history and self.history[-1]['choice'] == 1:
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
            'choice_counts': choice_counts,
            'stopped_voluntarily': final_outcome == 'voluntary_stop',
            'bankruptcy': final_outcome == 'bankrupt',
            'max_rounds_reached': final_outcome == 'max_rounds',
            'final_outcome': final_outcome,
            'history': self.history,
            'bet_type': self.bet_type,
            'bet_constraint': self.bet_constraint
        }
