"""
Blackjack Game Logic

Implements standard Blackjack rules for LLM gambling experiments.
"""

import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Card:
    """Playing card"""
    rank: str  # '2'-'10', 'J', 'Q', 'K', 'A'
    suit: str  # '♠', '♥', '♦', '♣'

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def value(self, current_total: int = 0) -> int:
        """
        Get card value for Blackjack.

        Args:
            current_total: Current hand total (for Ace calculation)

        Returns:
            Card value (1-11)
        """
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            # Ace is 11 if it doesn't bust, else 1
            return 11 if current_total + 11 <= 21 else 1
        else:
            return int(self.rank)


class Deck:
    """Standard 52-card deck"""

    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    SUITS = ['♠', '♥', '♦', '♣']

    def __init__(self, num_decks: int = 1):
        """
        Initialize deck.

        Args:
            num_decks: Number of standard decks to use
        """
        self.num_decks = num_decks
        self.cards: List[Card] = []
        self.reset()

    def reset(self):
        """Reset and shuffle deck"""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in self.SUITS:
                for rank in self.RANKS:
                    self.cards.append(Card(rank, suit))
        random.shuffle(self.cards)

    def draw(self) -> Card:
        """Draw a card from deck"""
        if len(self.cards) < 10:  # Reshuffle if running low
            self.reset()
        return self.cards.pop()


class BlackjackHand:
    """Blackjack hand"""

    def __init__(self):
        self.cards: List[Card] = []

    def add_card(self, card: Card):
        """Add card to hand"""
        self.cards.append(card)

    def value(self) -> int:
        """
        Calculate hand value.

        Handles Ace as 1 or 11 to avoid busting.
        """
        total = 0
        aces = 0

        for card in self.cards:
            if card.rank == 'A':
                aces += 1
                total += 11
            elif card.rank in ['J', 'Q', 'K']:
                total += 10
            else:
                total += int(card.rank)

        # Convert Aces from 11 to 1 if busting
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        return total

    def is_blackjack(self) -> bool:
        """Check if hand is natural Blackjack (21 with 2 cards)"""
        return len(self.cards) == 2 and self.value() == 21

    def is_bust(self) -> bool:
        """Check if hand is bust (over 21)"""
        return self.value() > 21

    def __str__(self) -> str:
        """String representation"""
        cards_str = ' '.join(str(card) for card in self.cards)
        return f"{cards_str} (total: {self.value()})"


class BlackjackGame:
    """Blackjack game session"""

    def __init__(
        self,
        initial_chips: int = 1000,
        min_bet: int = 10,
        max_bet: int = 500,
        bet_type: str = 'variable'
    ):
        """
        Initialize Blackjack game.

        Args:
            initial_chips: Starting chip amount
            min_bet: Minimum bet
            max_bet: Maximum bet
            bet_type: 'variable' or 'fixed'
        """
        self.initial_chips = initial_chips
        self.chips = initial_chips
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.bet_type = bet_type

        self.deck = Deck(num_decks=6)  # Use 6 decks like casinos
        self.player_hand = BlackjackHand()
        self.dealer_hand = BlackjackHand()
        self.current_bet = 0

        self.round_num = 0
        self.history: List[Dict] = []

    def reset_round(self):
        """Reset for new round"""
        self.player_hand = BlackjackHand()
        self.dealer_hand = BlackjackHand()
        self.current_bet = 0

    def place_bet(self, bet_amount: int) -> bool:
        """
        Place bet for current round.

        Args:
            bet_amount: Bet amount

        Returns:
            True if bet is valid, False otherwise
        """
        # Validate bet
        if bet_amount < self.min_bet or bet_amount > self.max_bet:
            return False
        if bet_amount > self.chips:
            return False

        self.current_bet = bet_amount
        self.chips -= bet_amount
        return True

    def deal_initial_cards(self):
        """Deal initial two cards to player and dealer"""
        self.player_hand.add_card(self.deck.draw())
        self.dealer_hand.add_card(self.deck.draw())
        self.player_hand.add_card(self.deck.draw())
        self.dealer_hand.add_card(self.deck.draw())

    def player_hit(self):
        """Player hits (takes a card)"""
        self.player_hand.add_card(self.deck.draw())

    def dealer_play(self):
        """Dealer plays according to standard rules (stand on 17)"""
        while self.dealer_hand.value() < 17:
            self.dealer_hand.add_card(self.deck.draw())

    def determine_outcome(self) -> Tuple[str, int]:
        """
        Determine round outcome.

        Returns:
            (outcome, payout) tuple
            outcome: 'win', 'lose', 'push', 'blackjack'
            payout: Chips won (0 for loss, bet amount for push, 2x bet for win, 2.5x bet for blackjack)
        """
        player_value = self.player_hand.value()
        dealer_value = self.dealer_hand.value()

        # Player bust
        if self.player_hand.is_bust():
            return 'lose', 0

        # Player blackjack
        if self.player_hand.is_blackjack():
            if self.dealer_hand.is_blackjack():
                return 'push', self.current_bet  # Return bet
            else:
                return 'blackjack', int(self.current_bet * 2.5)  # 3:2 payout

        # Dealer blackjack (player doesn't have blackjack)
        if self.dealer_hand.is_blackjack():
            return 'lose', 0

        # Dealer bust
        if self.dealer_hand.is_bust():
            return 'win', self.current_bet * 2

        # Compare values
        if player_value > dealer_value:
            return 'win', self.current_bet * 2
        elif player_value < dealer_value:
            return 'lose', 0
        else:
            return 'push', self.current_bet  # Return bet

    def play_round(self, bet_amount: int, actions: List[str]) -> Dict:
        """
        Play a complete round.

        Args:
            bet_amount: Bet amount
            actions: List of player actions ('hit', 'stand', 'double')

        Returns:
            Round result dictionary
        """
        self.round_num += 1
        self.reset_round()

        # Place bet
        if not self.place_bet(bet_amount):
            return {
                'round': self.round_num,
                'valid': False,
                'error': 'Invalid bet amount'
            }

        # Deal initial cards
        self.deal_initial_cards()

        # Check for natural blackjacks
        if self.player_hand.is_blackjack() or self.dealer_hand.is_blackjack():
            outcome, payout = self.determine_outcome()
            self.chips += payout

            return {
                'round': self.round_num,
                'valid': True,
                'bet': bet_amount,
                'player_hand': str(self.player_hand),
                'dealer_hand': str(self.dealer_hand),
                'outcome': outcome,
                'payout': payout,
                'chips': self.chips,
                'actions': []
            }

        # Player actions
        action_history = []
        for action in actions:
            action_history.append(action)

            if action == 'hit':
                self.player_hit()
                if self.player_hand.is_bust():
                    break
            elif action == 'stand':
                break
            elif action == 'double':
                # Double down: double bet, take one card, then stand
                if self.chips >= bet_amount:
                    self.chips -= bet_amount
                    self.current_bet *= 2
                    self.player_hit()
                    break
                else:
                    # Can't afford to double, treat as hit
                    self.player_hit()
                    break

        # Dealer plays if player didn't bust
        if not self.player_hand.is_bust():
            self.dealer_play()

        # Determine outcome
        outcome, payout = self.determine_outcome()
        self.chips += payout

        result = {
            'round': self.round_num,
            'valid': True,
            'bet': self.current_bet,
            'player_hand': str(self.player_hand),
            'dealer_hand': str(self.dealer_hand),
            'outcome': outcome,
            'payout': payout,
            'chips': self.chips,
            'actions': action_history
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
            'bet_type': self.bet_type
        }
