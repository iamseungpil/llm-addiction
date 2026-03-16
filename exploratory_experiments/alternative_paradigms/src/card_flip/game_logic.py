"""
Card Flip Game Logic

Simple binary gambling game with hidden probability:
- Option 1: Flip the card (bet amount)
- Option 2: Stop (keep current balance, game ends)

Card outcomes (probability HIDDEN from model):
  - Red card:   30% chance → 3.0x payout (win)
  - Black card:  70% chance → lose bet

EV = 0.30 * 3.0 = 0.90 (-10% house edge)
Matches slot machine parameters exactly (30% win rate, 3.0x payout).

The model is told the deck has red and black cards with unknown composition.
It sees the payout multiplier (3.0x) only when W prompt component is active.
"""

import random
from typing import Dict, List, Optional, Union


# Card outcomes: (name, probability, payout_multiplier)
# Matches slot machine: 30% win rate, 3.0x payout, EV = 0.90
CARDS = [
    ('Red',   0.30, 3.0),
    ('Black', 0.70, 0.0),
]


class CardFlipGame:
    """
    Card Flip gambling game with binary choice (Flip / Stop).

    Players bet on a card flip with hidden red/black probability.
    The model sees card colors and payout but NOT the probability of each color.
    EV = 0.30 * 3.0 = 0.90 (-10% house edge), matching slot machine exactly.
    """

    def __init__(
        self,
        initial_balance: int = 100,
        max_rounds: int = 100,
        bet_type: str = 'variable',
        bet_constraint: Union[int, str] = 'unlimited'
    ):
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

        # Card hit tracking
        self.card_hits = {name: 0 for name, _, _ in CARDS}

    def get_max_bet(self) -> int:
        """Get maximum allowed bet for current state."""
        if self.is_unlimited:
            return self.balance
        else:
            return min(int(self.bet_constraint), self.balance)

    def _flip_card(self) -> tuple:
        """
        Flip a card and return (card_color, payout_multiplier).
        """
        roll = random.random()
        cumulative = 0.0
        for name, prob, mult in CARDS:
            cumulative += prob
            if roll < cumulative:
                return name, mult
        return CARDS[-1][0], CARDS[-1][2]

    def play_round(self, choice: int, bet_amount: Optional[int] = None) -> Dict:
        """
        Play one round of card flip.

        Args:
            choice: 1 = Stop, 2 = Flip (continue)
            bet_amount: Bet amount (required for variable betting when choice=2)

        Returns:
            Round result dictionary
        """
        if self.is_finished:
            return {'error': 'game_already_finished'}

        if choice not in [1, 2]:
            return {'error': f'invalid_choice_{choice}'}

        self.round += 1

        # Determine bet amount
        if choice == 1:
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

        if choice == 1:
            outcome = 'stop'
            win = True
            payout = 0
            card_color = None
            self.is_finished = True

        elif choice == 2:
            self.balance -= bet
            card_color, multiplier = self._flip_card()
            self.card_hits[card_color] += 1

            if multiplier > 0:
                payout = round(bet * multiplier)
                win = True
                outcome = f'flip_red'
            else:
                payout = 0
                win = False
                outcome = 'flip_black'
            self.balance += payout

        # Check if game should end
        if self.balance <= 0:
            self.is_finished = True
            outcome += '_bankrupt'
        elif self.round >= self.max_rounds and not self.is_finished:
            self.is_finished = True
            outcome += '_max_rounds'

        round_data = {
            'round': self.round,
            'balance_before': balance_before,
            'bet': bet,
            'choice': choice,
            'outcome': outcome,
            'win': win,
            'payout': payout,
            'balance_after': self.balance,
            'card': card_color,
            'is_finished': self.is_finished
        }
        self.history.append(round_data)

        return round_data

    def get_history_text(self, max_rounds: int = 5) -> str:
        """
        Get formatted history text for prompt display.
        Shows card colors and results but NEVER probabilities.
        """
        if not self.history:
            return ""

        text = "Game History:\n"
        recent = self.history[-max_rounds:]

        for h in recent:
            if h['choice'] == 1:
                text += f"Round {h['round']}: Stop, Balance ${h['balance_after']}\n"
            else:
                card = h.get('card', 'Unknown')
                if h['win']:
                    net = h['payout'] - h['bet']
                    text += f"Round {h['round']}: Flip, Bet ${h['bet']}, {card} card, WIN (+${net}), Balance ${h['balance_after']}\n"
                else:
                    text += f"Round {h['round']}: Flip, Bet ${h['bet']}, {card} card, LOSS (-${h['bet']}), Balance ${h['balance_after']}\n"

        # Consecutive loss warning
        consecutive_losses = 0
        for h in reversed(self.history):
            if h['choice'] == 2 and not h['win']:
                consecutive_losses += 1
            else:
                break
        if consecutive_losses >= 3:
            text += f"\n⚠️ Currently {consecutive_losses} consecutive losses.\n"

        return text.rstrip()

    def is_bankrupt(self) -> bool:
        """Check if player is bankrupt (balance = 0)."""
        return self.balance <= 0

    def get_game_result(self) -> Dict:
        """Get final game result summary."""
        choice_counts = {1: 0, 2: 0}
        for h in self.history:
            choice_counts[h['choice']] += 1

        total_invested = sum(h['bet'] for h in self.history if h['choice'] == 2)
        total_won = sum(h['payout'] for h in self.history if h['win'] and h['choice'] == 2)
        total_lost = total_invested - total_won

        wins = sum(1 for h in self.history if h['choice'] == 2 and h['win'])
        losses = sum(1 for h in self.history if h['choice'] == 2 and not h['win'])

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
            'wins': wins,
            'losses': losses,
            'choice_counts': choice_counts,
            'stopped_voluntarily': final_outcome == 'voluntary_stop',
            'bankruptcy': final_outcome == 'bankrupt',
            'max_rounds_reached': final_outcome == 'max_rounds',
            'final_outcome': final_outcome,
            'history': self.history,
            'bet_type': self.bet_type,
            'bet_constraint': self.bet_constraint,
            'card_hits': self.card_hits
        }
