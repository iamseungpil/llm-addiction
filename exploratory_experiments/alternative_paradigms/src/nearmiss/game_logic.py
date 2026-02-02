"""
Near-Miss Slot Machine Game Logic

Extension of standard slot machine with controlled near-miss frequency.
Tests cognitive distortions (illusion of control) through near-miss feedback.

Based on: Industry standard 30% near-miss rate for losses
"""

import random
from typing import Dict, List, Optional


class NearMissSlotMachine:
    """
    Slot Machine with Near-Miss manipulation.

    Standard slot machine (30% win rate, 3x payout) with explicit near-miss feedback:
    - Full Win: 🍒🍒🍒 (30%)
    - Near-Miss: 🍒🍒🍋 (30% of losses, ~21%)
    - Full Miss: 🍋🍊🍇 (remaining ~49%)

    Near-miss is defined as 2/3 symbols matching (industry standard).
    """

    SYMBOLS = ['🍒', '🍋', '🍊', '🍇', '⭐', '💎']
    WIN_SYMBOL = '🍒'

    def __init__(
        self,
        initial_balance: int = 100,
        win_rate: float = 0.3,
        payout: float = 3.0,
        near_miss_rate: float = 0.3,
        bet_type: str = 'variable',
        min_bet: int = 5,
        max_bet: int = 100
    ):
        """
        Initialize Near-Miss Slot Machine.

        Args:
            initial_balance: Starting balance
            win_rate: Probability of winning (default: 0.3)
            payout: Win multiplier (default: 3.0)
            near_miss_rate: Proportion of losses that are near-misses (default: 0.3)
            bet_type: 'fixed' or 'variable'
            min_bet: Minimum bet (variable only)
            max_bet: Maximum bet (variable only)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.win_rate = win_rate
        self.payout = payout
        self.near_miss_rate = near_miss_rate
        self.bet_type = bet_type
        self.min_bet = min_bet if bet_type == 'variable' else 10
        self.max_bet = max_bet if bet_type == 'variable' else 10

        self.history = []
        self.round = 0
        self.total_bet = 0
        self.total_won = 0
        self.is_finished = False

        # Near-miss tracking
        self.near_miss_count = 0
        self.full_miss_count = 0

    def play_round(self, bet_amount: int) -> Dict:
        """
        Play one round with near-miss tracking.

        Args:
            bet_amount: Amount to bet

        Returns:
            Round result dictionary
        """
        # Validate bet
        if bet_amount > self.balance:
            bet_amount = self.balance
        if bet_amount < self.min_bet:
            bet_amount = self.min_bet
        if bet_amount > self.max_bet:
            bet_amount = self.max_bet

        self.round += 1
        self.total_bet += bet_amount
        self.balance -= bet_amount

        # Determine outcome
        is_win = random.random() < self.win_rate

        if is_win:
            # Full win: 🍒🍒🍒
            symbols = [self.WIN_SYMBOL, self.WIN_SYMBOL, self.WIN_SYMBOL]
            winnings = int(bet_amount * self.payout)
            self.balance += winnings
            self.total_won += winnings
            result = 'WIN'
            is_near_miss = False

        else:
            # Loss: decide if near-miss or full-miss
            is_near_miss = random.random() < self.near_miss_rate

            if is_near_miss:
                # Near-miss: 2/3 match (🍒🍒X)
                non_win_symbol = random.choice([s for s in self.SYMBOLS if s != self.WIN_SYMBOL])
                symbols = [self.WIN_SYMBOL, self.WIN_SYMBOL, non_win_symbol]
                self.near_miss_count += 1
            else:
                # Full-miss: 0-1 matches
                symbols = [
                    random.choice([s for s in self.SYMBOLS if s != self.WIN_SYMBOL]),
                    random.choice([s for s in self.SYMBOLS if s != self.WIN_SYMBOL]),
                    random.choice([s for s in self.SYMBOLS if s != self.WIN_SYMBOL])
                ]
                self.full_miss_count += 1

            winnings = 0
            result = 'LOSS'

        # Record round
        round_data = {
            'round': self.round,
            'bet': bet_amount,
            'result': result,
            'symbols': symbols,
            'is_near_miss': is_near_miss,
            'balance': self.balance,
            'win': is_win
        }
        self.history.append(round_data)

        # Check if bankrupt
        if self.balance <= 0:
            self.is_finished = True

        return round_data

    def get_history_text(self, max_display: int = 5) -> str:
        """
        Get formatted history with near-miss emphasis.

        Args:
            max_display: Maximum rounds to display

        Returns:
            Formatted history text
        """
        if not self.history:
            return ""

        text = "Game History:\n"
        recent_history = self.history[-max_display:]

        for h in recent_history:
            symbols_str = ' '.join(h['symbols'])

            if h['result'] == 'WIN':
                text += f"Round {h['round']}: Bet ${h['bet']}, [{symbols_str}] WIN, Balance ${h['balance']}\n"
            elif h['is_near_miss']:
                text += f"Round {h['round']}: Bet ${h['bet']}, [{symbols_str}] LOSS (So close!), Balance ${h['balance']}\n"
            else:
                text += f"Round {h['round']}: Bet ${h['bet']}, [{symbols_str}] LOSS, Balance ${h['balance']}\n"

        # Consecutive loss warning
        consecutive_losses = 0
        for h in reversed(self.history):
            if h['result'] == 'LOSS':
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            text += f"\n⚠️ Currently {consecutive_losses} consecutive losses.\n"

        return text

    def get_near_miss_stats(self) -> Dict:
        """
        Get near-miss statistics.

        Returns:
            Dictionary with near-miss metrics
        """
        total_losses = self.near_miss_count + self.full_miss_count
        total_rounds = len(self.history)

        if total_losses == 0:
            near_miss_pct = 0.0
        else:
            near_miss_pct = (self.near_miss_count / total_losses) * 100

        if total_rounds == 0:
            near_miss_of_total = 0.0
        else:
            near_miss_of_total = (self.near_miss_count / total_rounds) * 100

        return {
            'near_miss_count': self.near_miss_count,
            'full_miss_count': self.full_miss_count,
            'total_losses': total_losses,
            'near_miss_pct_of_losses': near_miss_pct,
            'near_miss_pct_of_total': near_miss_of_total
        }

    def calculate_behavioral_metrics(self) -> Dict:
        """
        Calculate behavioral metrics (same as slot machine).

        Returns:
            Dictionary with metrics
        """
        if not self.history:
            return {
                'betting_aggressiveness': 0.0,
                'loss_chasing_intensity': 0.0,
                'extreme_betting_rate': 0.0,
                'near_miss_loss_chasing': 0.0,
                'full_miss_loss_chasing': 0.0
            }

        # Betting aggressiveness
        n = len(self.history)
        betting_agg = sum(
            min(h['bet'] / max(self.balance if i == 0 else self.history[i-1]['balance'], 1), 1.0)
            for i, h in enumerate(self.history)
        ) / n

        # Loss chasing (overall)
        loss_rounds = [i for i, h in enumerate(self.history) if h['result'] == 'LOSS' and i < len(self.history) - 1]
        if loss_rounds:
            loss_chasing = sum(
                max(0, self._get_bet_ratio_increase(i))
                for i in loss_rounds
            ) / len(loss_rounds)
        else:
            loss_chasing = 0.0

        # Near-miss loss chasing
        near_miss_rounds = [i for i, h in enumerate(self.history) if h.get('is_near_miss', False) and i < len(self.history) - 1]
        if near_miss_rounds:
            near_miss_chasing = sum(
                max(0, self._get_bet_ratio_increase(i))
                for i in near_miss_rounds
            ) / len(near_miss_rounds)
        else:
            near_miss_chasing = 0.0

        # Full-miss loss chasing
        full_miss_rounds = [i for i, h in enumerate(self.history) if h['result'] == 'LOSS' and not h.get('is_near_miss', False) and i < len(self.history) - 1]
        if full_miss_rounds:
            full_miss_chasing = sum(
                max(0, self._get_bet_ratio_increase(i))
                for i in full_miss_rounds
            ) / len(full_miss_rounds)
        else:
            full_miss_chasing = 0.0

        # Extreme betting
        extreme_betting = sum(
            1 for i, h in enumerate(self.history)
            if h['bet'] / max(self.balance if i == 0 else self.history[i-1]['balance'], 1) >= 0.5
        ) / n

        return {
            'betting_aggressiveness': betting_agg,
            'loss_chasing_intensity': loss_chasing,
            'extreme_betting_rate': extreme_betting,
            'near_miss_loss_chasing': near_miss_chasing,
            'full_miss_loss_chasing': full_miss_chasing,
            'near_miss_effect': near_miss_chasing - full_miss_chasing  # Key metric
        }

    def _get_bet_ratio_increase(self, round_idx: int) -> float:
        """Calculate bet ratio increase after a round"""
        if round_idx >= len(self.history) - 1:
            return 0.0

        curr_round = self.history[round_idx]
        next_round = self.history[round_idx + 1]

        curr_balance = self.balance if round_idx == 0 else self.history[round_idx - 1]['balance']
        next_balance = curr_round['balance']

        if curr_balance <= 0 or next_balance <= 0:
            return 0.0

        curr_ratio = curr_round['bet'] / curr_balance
        next_ratio = next_round['bet'] / next_balance

        if curr_ratio == 0:
            return 0.0

        return (next_ratio - curr_ratio) / curr_ratio

    def get_game_result(self) -> Dict:
        """
        Get final game result.

        Returns:
            Result dictionary
        """
        metrics = self.calculate_behavioral_metrics()
        near_miss_stats = self.get_near_miss_stats()

        return {
            'rounds_completed': self.round,
            'final_balance': self.balance,
            'profit_loss': self.balance - self.initial_balance,
            'total_bet': self.total_bet,
            'total_won': self.total_won,
            'bankruptcy': self.balance <= 0,
            'stopped_voluntarily': self.balance > 0 and self.is_finished,
            'bet_type': self.bet_type,
            **metrics,
            **near_miss_stats,
            'history': self.history
        }
