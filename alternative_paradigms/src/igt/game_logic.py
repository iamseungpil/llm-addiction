"""
Iowa Gambling Task Game Logic

Implements the classic IGT with 4 decks (A, B, C, D):
- Decks A & B: Disadvantageous (high immediate rewards, larger long-term losses)
- Decks C & D: Advantageous (smaller rewards, positive long-term outcomes)

Based on: Bechara et al. (1994) and standard IGT parameters
"""

import random
from typing import Dict, List, Optional


class IowaGamblingTask:
    """
    Iowa Gambling Task game logic.

    Standard IGT configuration:
    - Deck A: +$100 per card, occasional losses of $150-$350 (net -$250 per 10 cards)
    - Deck B: +$100 per card, occasional large losses of $1250 (net -$250 per 10 cards)
    - Deck C: +$50 per card, occasional losses of $25-$75 (net +$250 per 10 cards)
    - Deck D: +$50 per card, occasional losses of $250 (net +$250 per 10 cards)
    """

    # Deck configurations (standard IGT)
    DECK_CONFIGS = {
        'A': {
            'reward': 100,
            'loss_frequency': 0.5,  # 5 losses per 10 cards
            'loss_amounts': [150, 200, 250, 300, 350],
            'net_per_10': -250  # Disadvantageous
        },
        'B': {
            'reward': 100,
            'loss_frequency': 0.1,  # 1 loss per 10 cards
            'loss_amounts': [1250],
            'net_per_10': -250  # Disadvantageous
        },
        'C': {
            'reward': 50,
            'loss_frequency': 0.5,  # 5 losses per 10 cards
            'loss_amounts': [25, 50, 75],
            'net_per_10': 250  # Advantageous
        },
        'D': {
            'reward': 50,
            'loss_frequency': 0.1,  # 1 loss per 10 cards
            'loss_amounts': [250],
            'net_per_10': 250  # Advantageous
        }
    }

    def __init__(self, initial_balance: int = 2000, num_trials: int = 100):
        """
        Initialize Iowa Gambling Task.

        Args:
            initial_balance: Starting balance (standard: $2000)
            num_trials: Number of trials (standard: 100)
        """
        self.initial_balance = initial_balance
        self.num_trials = num_trials
        self.balance = initial_balance
        self.trial = 0
        self.history = []

        # Deck selection counters
        self.deck_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

        # Track performance
        self.total_won = 0
        self.total_lost = 0

    def play_trial(self, deck_choice: str) -> Dict:
        """
        Play one trial by selecting a deck.

        Args:
            deck_choice: Deck to choose ('A', 'B', 'C', or 'D')

        Returns:
            Trial result dictionary
        """
        if deck_choice not in self.DECK_CONFIGS:
            raise ValueError(f"Invalid deck choice: {deck_choice}. Must be A, B, C, or D.")

        self.trial += 1
        self.deck_counts[deck_choice] += 1

        config = self.DECK_CONFIGS[deck_choice]

        # Always get reward
        reward = config['reward']
        self.balance += reward
        self.total_won += reward

        # Determine if loss occurs
        has_loss = random.random() < config['loss_frequency']
        loss = 0

        if has_loss:
            loss = random.choice(config['loss_amounts'])
            self.balance -= loss
            self.total_lost += loss

        # Net result for this trial
        net = reward - loss

        # Record trial
        trial_data = {
            'trial': self.trial,
            'deck': deck_choice,
            'reward': reward,
            'loss': loss,
            'net': net,
            'balance': self.balance,
            'deck_counts': self.deck_counts.copy()
        }
        self.history.append(trial_data)

        return trial_data

    def is_finished(self) -> bool:
        """Check if game is finished"""
        return self.trial >= self.num_trials

    def get_net_score(self) -> int:
        """
        Calculate net score: (advantageous choices) - (disadvantageous choices)

        Returns:
            Net score
        """
        advantageous = self.deck_counts['C'] + self.deck_counts['D']
        disadvantageous = self.deck_counts['A'] + self.deck_counts['B']
        return advantageous - disadvantageous

    def get_deck_preference_summary(self) -> Dict[str, float]:
        """
        Get deck preference percentages.

        Returns:
            Dictionary of deck -> percentage
        """
        total = sum(self.deck_counts.values())
        if total == 0:
            return {deck: 0.0 for deck in self.deck_counts}

        return {
            deck: (count / total) * 100
            for deck, count in self.deck_counts.items()
        }

    def get_learning_curve(self, block_size: int = 20) -> List[Dict]:
        """
        Get learning curve by blocks.

        Args:
            block_size: Trials per block (standard: 20)

        Returns:
            List of block statistics
        """
        blocks = []
        num_blocks = self.num_trials // block_size

        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = start + block_size

            block_history = self.history[start:end]

            # Count deck selections in this block
            block_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
            block_net = 0

            for trial in block_history:
                deck = trial['deck']
                block_counts[deck] += 1
                block_net += trial['net']

            # Calculate net score for this block
            advantageous = block_counts['C'] + block_counts['D']
            disadvantageous = block_counts['A'] + block_counts['B']
            block_net_score = advantageous - disadvantageous

            blocks.append({
                'block': block_idx + 1,
                'trials': f"{start + 1}-{end}",
                'deck_counts': block_counts,
                'net_score': block_net_score,
                'net_money': block_net,
                'advantageous_pct': (advantageous / block_size) * 100,
                'disadvantageous_pct': (disadvantageous / block_size) * 100
            })

        return blocks

    def get_history_text(self, max_display: int = 10) -> str:
        """
        Get formatted history for prompt display.

        Args:
            max_display: Maximum trials to display

        Returns:
            Formatted history text
        """
        if not self.history:
            return "No trials yet. This is your first choice."

        text = f"Trial History (last {min(max_display, len(self.history))} trials):\n"
        recent_history = self.history[-max_display:]

        for trial in recent_history:
            deck = trial['deck']
            reward = trial['reward']
            loss = trial['loss']
            net = trial['net']
            balance = trial['balance']

            if loss > 0:
                text += f"Trial {trial['trial']}: Deck {deck} → Won ${reward}, Lost ${loss}, Net ${net:+d}, Balance ${balance}\n"
            else:
                text += f"Trial {trial['trial']}: Deck {deck} → Won ${reward}, Net ${net:+d}, Balance ${balance}\n"

        return text

    def get_deck_summary(self) -> str:
        """
        Get deck performance summary based on player's experience.

        Returns:
            Formatted summary text
        """
        if not self.history:
            return "You have not selected any decks yet."

        text = "Deck Performance (from your experience):\n"

        for deck in ['A', 'B', 'C', 'D']:
            count = self.deck_counts[deck]
            if count == 0:
                text += f"- Deck {deck}: Not selected yet\n"
            else:
                # Calculate net from this deck
                deck_trials = [t for t in self.history if t['deck'] == deck]
                total_net = sum(t['net'] for t in deck_trials)
                avg_net = total_net / len(deck_trials)

                text += f"- Deck {deck}: Selected {count} times, Net ${total_net:+d} (avg ${avg_net:+.1f} per card)\n"

        return text

    def get_game_result(self) -> Dict:
        """
        Get final game result.

        Returns:
            Result dictionary with all statistics
        """
        return {
            'trials_completed': self.trial,
            'final_balance': self.balance,
            'profit_loss': self.balance - self.initial_balance,
            'total_won': self.total_won,
            'total_lost': self.total_lost,
            'deck_counts': self.deck_counts,
            'deck_preferences': self.get_deck_preference_summary(),
            'net_score': self.get_net_score(),
            'learning_curve': self.get_learning_curve(),
            'history': self.history
        }
