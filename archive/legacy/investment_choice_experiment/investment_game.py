#!/usr/bin/env python3
"""
Investment Choice Game Logic
- 4 options: Stop, 50% $18, 25% $32, 10% $90
- Fixed betting: $10 per round
- Variable betting: $1-balance (all-in if bet > balance), with multipliers
- Max 10 rounds, starting balance $100
"""

import random
from typing import Dict, List, Optional


class InvestmentGame:
    """Investment choice game with 4 options"""

    def __init__(self, bet_type: str = 'fixed'):
        """
        Initialize game

        Args:
            bet_type: 'fixed' ($10 per round) or 'variable' (choose amount)
        """
        self.initial_balance = 100
        self.balance = 100
        self.bet_type = bet_type
        self.round = 0
        self.max_rounds = 10
        self.history: List[Dict] = []
        self.is_finished = False

    def play_round(self, choice: int, bet_amount: Optional[int] = None) -> Dict:
        """
        Play one round

        Args:
            choice: 1 (Stop), 2 (50% $18), 3 (25% $32), 4 (10% $90)
            bet_amount: Investment amount (only used for variable betting)

        Returns:
            Dict with round results
        """
        if self.is_finished:
            raise ValueError("Game is already finished")

        self.round += 1

        # Determine bet amount
        if self.bet_type == 'fixed':
            bet = min(10, self.balance)  # All-in if balance < $10
        else:  # variable
            if bet_amount is None:
                bet = min(10, self.balance)  # default fallback
            else:
                # All-in if bet > balance, no minimum requirement
                bet = min(bet_amount, self.balance)

        # Deduct investment
        balance_before = self.balance
        self.balance -= bet

        # Process choice
        if choice == 1:
            # Option 1: Stop - Get investment back, game ends
            payout = bet
            outcome = 'stop'
            win = True
            self.is_finished = True

        elif choice == 2:
            # Option 2: 50% chance of 1.8x
            win = random.random() < 0.5
            if win:
                payout = int(bet * 1.8)
                outcome = 'option2_win'
            else:
                payout = 0
                outcome = 'option2_loss'

        elif choice == 3:
            # Option 3: 25% chance of 3.2x
            win = random.random() < 0.25
            if win:
                payout = int(bet * 3.2)
                outcome = 'option3_win'
            else:
                payout = 0
                outcome = 'option3_loss'

        elif choice == 4:
            # Option 4: 10% chance of 9.0x
            win = random.random() < 0.1
            if win:
                payout = int(bet * 9.0)
                outcome = 'option4_win'
            else:
                payout = 0
                outcome = 'option4_loss'
        else:
            raise ValueError(f"Invalid choice: {choice}. Must be 1, 2, 3, or 4.")

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

    def get_last_round_text(self) -> str:
        """Get text description of last round result"""
        if not self.history:
            return ""

        last = self.history[-1]

        if last['win']:
            return f"Last round: Won ${last['payout']} (Option {last['choice']})"
        else:
            return f"Last round: Lost (Option {last['choice']}, no payout)"

    def get_state_dict(self) -> Dict:
        """Get current game state as dict"""
        return {
            'bet_type': self.bet_type,
            'balance': self.balance,
            'round': self.round,
            'max_rounds': self.max_rounds,
            'is_finished': self.is_finished,
            'history': self.history
        }
