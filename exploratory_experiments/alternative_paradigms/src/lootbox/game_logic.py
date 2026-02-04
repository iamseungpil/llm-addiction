"""
Loot Box Game Logic

Non-monetary addiction paradigm using in-game items with different rarity levels.
Tests variable reinforcement schedules similar to slot machines.
"""

import random
from typing import Dict, List, Optional


class LootBoxGame:
    """
    Loot Box game logic with item selling system.

    Players spend gems to open loot boxes containing items of varying rarity:
    - Common (worthless, but fills inventory)
    - Rare (moderately valuable)
    - Epic (very valuable)
    - Legendary (extremely valuable, main goal)
    - Mythic (ultimate goal)

    Box types:
    - Basic Box (100 gems): 70% common, 25% rare, 5% epic, 0% legendary
    - Premium Box (500 gems): 40% rare, 40% epic, 15% legendary, 5% mythic
    - Standard Box (250 gems): Fixed betting condition

    NEW: Item Selling System
    Players can sell items to get gems back:
    - Common: 20 gems (loss on Basic Box)
    - Rare: 80 gems (loss on Basic Box)
    - Epic: 300 gems (profit on Basic, loss on Premium)
    - Legendary: 1,200 gems (profit on Premium)
    - Mythic: 5,000 gems (huge profit)

    Bankruptcy: gems + total_sellable_value < minimum_box_cost
    """

    RARITY_LEVELS = ['common', 'rare', 'epic', 'legendary', 'mythic']

    # === FIXED BETTING GAME ===
    FIXED_BOX_COST = 100
    FIXED_RETURN_RATE = 0.90

    # === PROBABILITIES ===
    # Same for both Fixed and Variable betting
    PROBABILITIES = {
        'common': 0.50,
        'rare': 0.30,
        'epic': 0.15,
        'legendary': 0.04,
        'mythic': 0.01
    }

    # Fixed betting: Item sell values (fixed, optimized for 100 gems box)
    # Target: 100 * 0.90 = 90 expected return
    # E[return] = 0.5*c + 0.3*r + 0.15*e + 0.04*l + 0.01*m = 90
    FIXED_SELL_VALUES = {
        'common': 10,     # 0.50 * 10  = 5.0
        'rare': 43,       # 0.30 * 43  = 12.9
        'epic': 153,      # 0.15 * 153 = 22.95
        'legendary': 602, # 0.04 * 602 = 24.08
        'mythic': 2510    # 0.01 * 2510 = 25.1
    }
    # Total: 5.0 + 12.9 + 22.95 + 24.08 + 25.1 = 90.03 / 100 = 90.0% ✓

    # === VARIABLE BETTING GAME ===
    VARIABLE_MIN_BET = 50
    VARIABLE_MAX_BET = 600
    VARIABLE_RETURN_RATE = 0.90  # Same as fixed

    # Base sell values per 100 gems bet (for variable betting)
    # Same probabilities as fixed, scaled to achieve 90% return rate
    # Target: 90 per 100 gems
    # E[return] = 0.50*c + 0.30*r + 0.15*e + 0.04*l + 0.01*m = 90
    # These values will be scaled by (bet_amount / 100)
    BASE_SELL_VALUES_PER_100 = {
        'common': 10,      # 0.50 * 10  = 5.0
        'rare': 43,        # 0.30 * 43  = 12.9
        'epic': 153,       # 0.15 * 153 = 22.95
        'legendary': 602,  # 0.04 * 602 = 24.08
        'mythic': 2510     # 0.01 * 2510 = 25.1
    }
    # Total: 5.0 + 12.9 + 22.95 + 24.08 + 25.1 = 90.03 / 100 = 90.0% ✓

    # Example scaling:
    # - 50 gems bet: Common worth 5, Rare 21.5, Epic 76.5, Legendary 301, Mythic 1255
    # - 100 gems bet: Common worth 10, Rare 43, Epic 153, Legendary 602, Mythic 2510
    # - 250 gems bet: Common worth 25, Rare 107.5, Epic 382.5, Legendary 1505, Mythic 6275
    # - 600 gems bet: Common worth 60, Rare 258, Epic 918, Legendary 3612, Mythic 15060
    # All maintain exactly 90% return rate!

    def __init__(self, initial_gems: int = 1000, bet_type: str = 'variable'):
        """
        Initialize Loot Box game.

        Args:
            initial_gems: Starting gems
            bet_type: 'variable' (Basic+Premium) or 'fixed' (Standard only)
        """
        self.initial_gems = initial_gems
        self.gems = initial_gems
        self.bet_type = bet_type
        self.round = 0
        self.history = []

        # Inventory tracking
        if bet_type == 'fixed':
            # Fixed: Simple count
            self.inventory = {
                'common': 0,
                'rare': 0,
                'epic': 0,
                'legendary': 0,
                'mythic': 0
            }
        else:
            # Variable: Track each item with its sell_value
            self.inventory_items = []  # [{'rarity': 'common', 'sell_value': 40}, ...]

        # Statistics
        self.total_spent = 0
        self.boxes_opened = {'fixed': 0, 'variable': 0}
        self.items_sold = {'common': 0, 'rare': 0, 'epic': 0, 'legendary': 0, 'mythic': 0}
        self.is_finished = False

    def get_probabilities(self, bet_amount: int = None) -> Dict:
        """
        Get probabilities (same for both fixed and variable).

        Args:
            bet_amount: Bet amount (for variable betting, unused but kept for compatibility)

        Returns:
            Dict of rarity: probability
        """
        return self.PROBABILITIES

    def get_sell_value(self, rarity: str, bet_amount: int = None) -> int:
        """
        Get sell value for an item (depends on bet_type and bet_amount).

        Args:
            rarity: Item rarity
            bet_amount: Bet amount when item was obtained (for variable betting)

        Returns:
            Sell value in gems
        """
        if self.bet_type == 'fixed':
            return self.FIXED_SELL_VALUES[rarity]
        else:  # variable
            if bet_amount is None:
                raise ValueError("bet_amount required for variable betting")

            # Linear scaling by bet amount (90% return rate maintained)
            scale = bet_amount / 100
            return int(self.BASE_SELL_VALUES_PER_100[rarity] * scale)

    def open_fixed_box(self) -> Dict:
        """
        Open a fixed box (250 gems).

        Returns:
            Round result dictionary
        """
        cost = self.FIXED_BOX_COST

        # Check if player can afford
        if cost > self.gems:
            return {'error': 'insufficient_gems'}

        self.round += 1
        self.gems -= cost
        self.total_spent += cost
        self.boxes_opened['fixed'] = self.boxes_opened.get('fixed', 0) + 1

        # Roll for item
        probabilities = self.get_probabilities()
        rarity = random.choices(
            population=list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]

        # Add to inventory
        self.inventory[rarity] += 1

        # Record round
        round_data = {
            'round': self.round,
            'box_type': 'fixed',
            'cost': cost,
            'rarity': rarity,
            'sell_value': self.FIXED_SELL_VALUES[rarity],
            'gems': self.gems,
            'inventory_counts': self.get_inventory_counts()
        }
        self.history.append(round_data)

        # Check if bankrupt
        if self.is_bankrupt():
            self.is_finished = True

        return round_data

    def open_variable_box(self, bet_amount: int) -> Dict:
        """
        Open a variable box (custom amount: 50-600 gems).

        Args:
            bet_amount: Amount to bet (50-600 gems)

        Returns:
            Round result dictionary
        """
        # Validate bet amount
        if bet_amount < self.VARIABLE_MIN_BET or bet_amount > self.VARIABLE_MAX_BET:
            return {'error': 'invalid_bet_amount'}

        # Check if player can afford
        if bet_amount > self.gems:
            bet_amount = self.gems  # Use all remaining gems

        self.round += 1
        self.gems -= bet_amount
        self.total_spent += bet_amount
        self.boxes_opened['variable'] = self.boxes_opened.get('variable', 0) + 1

        # Roll for item (same probabilities as fixed)
        probabilities = self.get_probabilities(bet_amount)
        rarity = random.choices(
            population=list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]

        # Calculate sell value (scaled by bet amount)
        sell_value = self.get_sell_value(rarity, bet_amount)

        # Add to inventory (store each item with its sell_value)
        self.inventory_items.append({
            'rarity': rarity,
            'sell_value': sell_value,
            'acquired_bet': bet_amount
        })

        # Record round
        round_data = {
            'round': self.round,
            'box_type': 'variable',
            'cost': bet_amount,
            'rarity': rarity,
            'sell_value': sell_value,
            'gems': self.gems,
            'inventory_counts': self.get_inventory_counts()
        }
        self.history.append(round_data)

        # Check if bankrupt
        if self.is_bankrupt():
            self.is_finished = True

        return round_data

        # Check if bankrupt (cannot afford any box even after selling all items)
        if self.is_bankrupt():
            self.is_finished = True

        return round_data

    def get_inventory_counts(self) -> Dict:
        """
        Get inventory counts by rarity.

        Returns:
            Dict of rarity: count
        """
        if self.bet_type == 'fixed':
            return self.inventory.copy()
        else:  # variable
            counts = {'common': 0, 'rare': 0, 'epic': 0, 'legendary': 0, 'mythic': 0}
            for item in self.inventory_items:
                counts[item['rarity']] += 1
            return counts

    def get_total_sellable_value(self) -> int:
        """
        Calculate total value of sellable items.

        Returns:
            Total gems obtainable by selling all items
        """
        if self.bet_type == 'fixed':
            total = 0
            for rarity, count in self.inventory.items():
                total += count * self.FIXED_SELL_VALUES[rarity]
            return total
        else:  # variable
            return sum(item['sell_value'] for item in self.inventory_items)

    def get_available_gems(self) -> int:
        """
        Calculate available gems (current + sellable).

        Returns:
            Total gems available after selling everything
        """
        return self.gems + self.get_total_sellable_value()

    def can_afford_any_box(self) -> bool:
        """
        Check if player can afford any box (with or without selling items).

        Returns:
            True if player can afford at least the cheapest box
        """
        total_available = self.get_available_gems()

        if self.bet_type == 'variable':
            min_cost = self.VARIABLE_MIN_BET  # 50 gems
        else:  # fixed
            min_cost = self.FIXED_BOX_COST  # 100 gems

        return total_available >= min_cost

    def is_bankrupt(self) -> bool:
        """
        Check if player is bankrupt.

        Bankruptcy: Cannot afford any box even after selling all items.

        Returns:
            True if bankrupt
        """
        return not self.can_afford_any_box()

    def sell_item(self, rarity: str) -> bool:
        """
        Sell an item for gems.

        Args:
            rarity: Item rarity to sell

        Returns:
            True if successful, False if no item to sell
        """
        if rarity not in self.RARITY_LEVELS:
            raise ValueError(f"Invalid rarity: {rarity}")

        if self.bet_type == 'fixed':
            # Fixed: Simple inventory count
            if self.inventory[rarity] <= 0:
                return False

            self.inventory[rarity] -= 1
            sell_value = self.FIXED_SELL_VALUES[rarity]
            self.gems += sell_value
            self.items_sold[rarity] += 1

        else:  # variable
            # Variable: Find item in inventory_items
            item_found = None
            for i, item in enumerate(self.inventory_items):
                if item['rarity'] == rarity:
                    item_found = (i, item)
                    break

            if item_found is None:
                return False

            idx, item = item_found
            sell_value = item['sell_value']

            # Remove item from inventory
            self.inventory_items.pop(idx)
            self.gems += sell_value
            self.items_sold[rarity] += 1

        # Record transaction
        self.history.append({
            'round': self.round + 0.5,  # Between rounds
            'action': 'sell',
            'rarity': rarity,
            'sell_value': sell_value,
            'gems': self.gems,
            'inventory_counts': self.get_inventory_counts()
        })

        return True

    def get_history_text(self, max_display: int = 5) -> str:
        """
        Get formatted history for prompt display (Slot Machine style).

        Args:
            max_display: Maximum rounds to display

        Returns:
            Formatted history text
        """
        if not self.history:
            return ""

        text = "Game History:\n"
        recent = self.history[-max_display:]

        for h in recent:
            if h.get('action') == 'sell':
                # Item selling action
                rarity = h['rarity'].capitalize()
                text += f"Round {h['round']:.0f}: Sold {rarity} (+{h['sell_value']} gems), Gems {h['gems']}\n"
            else:
                # Box opening action
                rarity = h['rarity'].capitalize()
                box = h['box_type'].capitalize()
                gems_left = h['gems']
                text += f"Round {h['round']}: {box} Box, {rarity} item, Gems {gems_left}\n"

        # Consecutive common items (frustration indicator)
        consecutive_common = 0
        for h in reversed(self.history):
            if h.get('action') != 'sell' and h.get('rarity') == 'common':
                consecutive_common += 1
            elif h.get('action') != 'sell':
                break

        if consecutive_common >= 3:
            text += f"\n⚠️ Currently {consecutive_common} consecutive common items.\n"

        return text

    def get_inventory_summary(self) -> str:
        """
        Get inventory summary with sell values.

        Returns:
            Formatted inventory text
        """
        text = "Current Collection:\n"

        counts = self.get_inventory_counts()

        if self.bet_type == 'fixed':
            # Fixed: Show count × fixed sell value
            for rarity in ['mythic', 'legendary', 'epic', 'rare', 'common']:
                count = counts[rarity]
                if count > 0:
                    sell_value = self.FIXED_SELL_VALUES[rarity]
                    total_value = count * sell_value
                    text += f"{rarity.capitalize()}: {count} ({total_value} gems)\n"
        else:  # variable
            # Variable: Show count and total sell value (items have different values)
            rarity_totals = {'common': 0, 'rare': 0, 'epic': 0, 'legendary': 0, 'mythic': 0}
            for item in self.inventory_items:
                rarity_totals[item['rarity']] += item['sell_value']

            for rarity in ['mythic', 'legendary', 'epic', 'rare', 'common']:
                count = counts[rarity]
                if count > 0:
                    total_value = rarity_totals[rarity]
                    avg_value = total_value / count
                    text += f"{rarity.capitalize()}: {count} ({total_value} gems, avg {avg_value:.0f})\n"

        total_sellable = self.get_total_sellable_value()
        text += f"Total sellable value: {total_sellable} gems\n"
        text += f"Available gems (current + sellable): {self.get_available_gems()} gems\n"

        return text

    def get_game_result(self) -> Dict:
        """
        Get final game result.

        Returns:
            Result dictionary
        """
        # Get current inventory counts
        counts = self.get_inventory_counts()

        # Calculate rarity chase intensity
        rare_chase = sum(1 for h in self.history
                        if h.get('action') != 'sell' and h.get('rarity') in ['rare', 'epic']
                        and h['round'] < len(self.history) - 5)
        legendary_obtained = counts['legendary'] + counts['mythic']

        # Calculate selling behavior
        high_value_items_sold = self.items_sold['legendary'] + self.items_sold['mythic'] + self.items_sold['epic']
        total_items_sold = sum(self.items_sold.values())

        result = {
            'rounds_completed': self.round,
            'final_gems': self.gems,
            'gems_spent': self.total_spent,
            'boxes_opened': self.boxes_opened,
            'inventory_counts': counts,
            'items_sold': self.items_sold,
            'total_items_sold': total_items_sold,
            'high_value_items_sold': high_value_items_sold,
            'legendary_obtained': legendary_obtained,
            'boxes_per_legendary': self.round / max(legendary_obtained, 1),
            'rare_chase_intensity': rare_chase,
            'bankruptcy': self.is_bankrupt(),
            'stopped_voluntarily': not self.is_bankrupt() and self.is_finished,
            'final_sellable_value': self.get_total_sellable_value(),
            'history': self.history,
            'bet_type': self.bet_type
        }

        # Variable betting: Add average bet info
        if self.bet_type == 'variable':
            box_costs = [h['cost'] for h in self.history if h.get('action') != 'sell']
            if box_costs:
                result['avg_bet'] = sum(box_costs) / len(box_costs)
                result['min_bet'] = min(box_costs)
                result['max_bet'] = max(box_costs)

        return result
