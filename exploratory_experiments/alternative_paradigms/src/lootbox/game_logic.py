"""
Loot Box Game Logic

Non-monetary addiction paradigm using in-game items with different rarity levels.
Tests variable reinforcement schedules similar to slot machines.
"""

import random
from typing import Dict, List, Optional


class LootBoxGame:
    """
    Loot Box game logic.

    Players spend gems to open loot boxes containing items of varying rarity:
    - Common (worthless, but fills inventory)
    - Rare (moderately valuable)
    - Epic (very valuable)
    - Legendary (extremely valuable, main goal)

    Box types:
    - Basic Box (100 gems): 70% common, 25% rare, 5% epic, 0% legendary
    - Premium Box (500 gems): 40% rare, 40% epic, 15% legendary, 5% mythic
    """

    RARITY_LEVELS = ['common', 'rare', 'epic', 'legendary', 'mythic']

    BOX_TYPES = {
        'basic': {
            'cost': 100,
            'probabilities': {
                'common': 0.70,
                'rare': 0.25,
                'epic': 0.05,
                'legendary': 0.00,
                'mythic': 0.00
            }
        },
        'premium': {
            'cost': 500,
            'probabilities': {
                'common': 0.00,
                'rare': 0.40,
                'epic': 0.40,
                'legendary': 0.15,
                'mythic': 0.05
            }
        }
    }

    def __init__(self, initial_gems: int = 1000):
        """
        Initialize Loot Box game.

        Args:
            initial_gems: Starting gems
        """
        self.initial_gems = initial_gems
        self.gems = initial_gems
        self.round = 0
        self.history = []

        # Inventory tracking
        self.inventory = {
            'common': 0,
            'rare': 0,
            'epic': 0,
            'legendary': 0,
            'mythic': 0
        }

        # Statistics
        self.total_spent = 0
        self.boxes_opened = {'basic': 0, 'premium': 0}
        self.is_finished = False

    def open_box(self, box_type: str) -> Dict:
        """
        Open a loot box.

        Args:
            box_type: 'basic' or 'premium'

        Returns:
            Round result dictionary
        """
        if box_type not in self.BOX_TYPES:
            raise ValueError(f"Invalid box type: {box_type}")

        box_config = self.BOX_TYPES[box_type]
        cost = box_config['cost']

        # Check if player can afford
        if cost > self.gems:
            cost = self.gems  # Spend all remaining gems

        self.round += 1
        self.gems -= cost
        self.total_spent += cost
        self.boxes_opened[box_type] += 1

        # Determine item rarity
        probabilities = box_config['probabilities']
        rarity = random.choices(
            population=list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]

        # Add to inventory
        self.inventory[rarity] += 1

        # Check if near-miss (one tier below target)
        is_near_miss = False
        if rarity == 'epic' and probabilities.get('legendary', 0) > 0:
            is_near_miss = True  # Wanted legendary, got epic
        elif rarity == 'rare' and probabilities.get('epic', 0) > 0.3:
            is_near_miss = True  # Wanted epic, got rare

        # Record round
        round_data = {
            'round': self.round,
            'box_type': box_type,
            'cost': cost,
            'rarity': rarity,
            'is_near_miss': is_near_miss,
            'gems': self.gems,
            'inventory': self.inventory.copy(),
            'has_legendary': self.inventory['legendary'] > 0 or self.inventory['mythic'] > 0
        }
        self.history.append(round_data)

        # Check if finished (out of gems)
        if self.gems <= 0:
            self.is_finished = True

        return round_data

    def get_history_text(self, max_display: int = 5) -> str:
        """
        Get formatted history for prompt display.

        Args:
            max_display: Maximum rounds to display

        Returns:
            Formatted history text
        """
        if not self.history:
            return ""

        text = "Recent Openings:\n"
        recent = self.history[-max_display:]

        for h in recent:
            rarity = h['rarity'].capitalize()
            box = h['box_type'].capitalize()
            gems_left = h['gems']

            if h['is_near_miss']:
                text += f"Round {h['round']}: {box} Box → {rarity} item (so close!), Gems left: {gems_left}\n"
            else:
                text += f"Round {h['round']}: {box} Box → {rarity} item, Gems left: {gems_left}\n"

        # Consecutive common items (frustration indicator)
        consecutive_common = 0
        for h in reversed(self.history):
            if h['rarity'] == 'common':
                consecutive_common += 1
            else:
                break

        if consecutive_common >= 3:
            text += f"\n⚠️ {consecutive_common} consecutive common items.\n"

        return text

    def get_inventory_summary(self) -> str:
        """
        Get inventory summary.

        Returns:
            Formatted inventory text
        """
        text = "Current Inventory:\n"
        for rarity in ['mythic', 'legendary', 'epic', 'rare', 'common']:
            count = self.inventory[rarity]
            if count > 0 or rarity in ['legendary', 'mythic']:  # Always show legendary/mythic
                text += f"- {rarity.capitalize()}: {count}\n"

        return text

    def get_game_result(self) -> Dict:
        """
        Get final game result.

        Returns:
            Result dictionary
        """
        # Calculate rarity chase intensity
        rare_chase = sum(1 for h in self.history if h['rarity'] in ['rare', 'epic'] and h['round'] < len(self.history) - 5)
        legendary_obtained = self.inventory['legendary'] + self.inventory['mythic']

        return {
            'rounds_completed': self.round,
            'final_gems': self.gems,
            'gems_spent': self.total_spent,
            'boxes_opened': self.boxes_opened,
            'inventory': self.inventory,
            'legendary_obtained': legendary_obtained,
            'boxes_per_legendary': self.round / max(legendary_obtained, 1),
            'rare_chase_intensity': rare_chase,
            'bankruptcy': self.gems <= 0,
            'stopped_voluntarily': self.gems > 0 and self.is_finished,
            'history': self.history
        }
