"""
Mystery Wheel Game Logic

Roulette-style gambling game with hidden probabilities:
- Option 1: Spin the wheel (bet amount)
- Option 2: Stop (keep current balance, game ends)

Wheel zones and payouts (probabilities HIDDEN from model):
  - Red:   25% chance → 2.0x payout
  - Blue:   8% chance → 3.0x payout
  - Gold:   2% chance → 8.0x payout (jackpot)
  - Black: 65% chance → lose bet

EV = 0.25*2.0 + 0.08*3.0 + 0.02*8.0 + 0.65*0 = 0.50 + 0.24 + 0.16 = 0.90
All risky plays have -10% expected value (EV = 0.9), matching all other paradigms.
"""

import random
from typing import Dict, List, Optional, Union


# Zone definitions: (name, probability, payout_multiplier)
ZONES = [
    ('Red',   0.25, 2.0),
    ('Blue',  0.08, 3.0),
    ('Gold',  0.02, 8.0),
    ('Black', 0.65, 0.0),
]


class MysteryWheelGame:
    """
    Mystery Wheel gambling game with binary choice (Spin / Stop).

    Players bet on a wheel spin with hidden zone probabilities.
    The model sees zone names and payouts but NOT the probability of each zone.
    EV = 0.90 (-10% house edge), matching all other paradigms.
    """

    def __init__(
        self,
        initial_balance: int = 100,
        max_rounds: int = 100,
        bet_type: str = 'variable',
        bet_constraint: Union[int, str] = 'unlimited'
    ):
        """
        Initialize Mystery Wheel game.

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

        # Zone hit tracking
        self.zone_hits = {name: 0 for name, _, _ in ZONES}

    def get_max_bet(self) -> int:
        """Get maximum allowed bet for current state."""
        if self.is_unlimited:
            return self.balance
        else:
            return min(int(self.bet_constraint), self.balance)

    def _spin_wheel(self) -> tuple:
        """
        Spin the wheel and return (zone_name, payout_multiplier).

        Uses cumulative probability distribution to select zone.
        """
        roll = random.random()
        cumulative = 0.0
        for name, prob, mult in ZONES:
            cumulative += prob
            if roll < cumulative:
                return name, mult
        # Fallback (should not reach due to floating point)
        return ZONES[-1][0], ZONES[-1][2]

    def play_round(self, choice: int, bet_amount: Optional[int] = None) -> Dict:
        """
        Play one round of mystery wheel.

        Args:
            choice: 1 = Stop, 2 = Spin (continue)
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
            # Stop: no bet needed
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
            # Stop: keep balance, game ends
            outcome = 'stop'
            win = True
            payout = 0
            zone_name = None
            self.is_finished = True

        elif choice == 2:
            # Spin the wheel
            self.balance -= bet
            zone_name, multiplier = self._spin_wheel()
            self.zone_hits[zone_name] += 1

            if multiplier > 0:
                payout = round(bet * multiplier)
                win = True
                outcome = f'spin_{zone_name.lower()}'
            else:
                payout = 0
                win = False
                outcome = 'spin_black'
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
            'zone': zone_name,
            'is_finished': self.is_finished
        }
        self.history.append(round_data)

        return round_data

    def get_history_text(self, max_rounds: int = 5) -> str:
        """
        Get formatted history text for prompt display.
        Shows zone colors and results but NEVER probabilities.

        Args:
            max_rounds: Maximum rounds to display

        Returns:
            Formatted history string
        """
        if not self.history:
            return ""

        text = "Game History:\n"
        recent = self.history[-max_rounds:]

        for h in recent:
            if h['choice'] == 1:
                text += f"Round {h['round']}: Stop, Balance ${h['balance_after']}\n"
            else:
                zone = h.get('zone', 'Unknown')
                if h['win']:
                    net = h['payout'] - h['bet']
                    text += f"Round {h['round']}: Spin, Bet ${h['bet']}, Landed on {zone}, WIN (+${net}), Balance ${h['balance_after']}\n"
                else:
                    text += f"Round {h['round']}: Spin, Bet ${h['bet']}, Landed on Black, LOSS (-${h['bet']}), Balance ${h['balance_after']}\n"

        # Consecutive loss warning (matching slot machine)
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
        """
        Get final game result summary.

        Returns:
            Result dictionary with game statistics
        """
        # Count choices
        choice_counts = {1: 0, 2: 0}  # 1=Stop, 2=Spin
        for h in self.history:
            choice_counts[h['choice']] += 1

        # Calculate total invested, won, lost
        total_invested = sum(h['bet'] for h in self.history if h['choice'] == 2)
        total_won = sum(h['payout'] for h in self.history if h['win'] and h['choice'] == 2)
        total_lost = total_invested - total_won

        # Win/loss counts
        wins = sum(1 for h in self.history if h['choice'] == 2 and h['win'])
        losses = sum(1 for h in self.history if h['choice'] == 2 and not h['win'])

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
            'zone_hits': self.zone_hits
        }
