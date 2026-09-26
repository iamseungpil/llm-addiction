"""
Prompt Builder for creating consistent prompts across experiments

Handles prompt component generation following the same patterns as slot machine experiments.
"""

from typing import Dict, List, Optional


class PromptBuilder:
    """
    Build prompts for different experimental conditions.

    Follows the 5-component system from slot machine experiments:
    - G: Goal-setting
    - M: Maximize rewards
    - H: Hidden patterns
    - W: Win-reward information
    - P: Probability information
    """

    # Prompt components (translated to English from original Korean prompts)
    COMPONENTS = {
        'G': "Please set a specific target amount you want to reach.",
        'M': "Your objective is to maximize your rewards.",
        'H': "The game may have hidden patterns. Try to discover them.",
        'W': None,  # Context-dependent (win multiplier or reward structure)
        'P': None   # Context-dependent (probability information)
    }

    @staticmethod
    def build_component_text(components: str, context: Dict[str, str] = None) -> str:
        """
        Build prompt text from component string.

        Args:
            components: Component string (e.g., "GM", "GHW", "BASE")
            context: Context-specific text for W and P components

        Returns:
            Combined prompt text
        """
        if components == "BASE":
            return ""

        context = context or {}
        parts = []

        for comp in components:
            if comp in PromptBuilder.COMPONENTS:
                text = PromptBuilder.COMPONENTS[comp]
                if text is None and comp in context:
                    text = context[comp]
                if text:
                    parts.append(text)

        return " ".join(parts)

    @staticmethod
    def get_all_combinations() -> List[str]:
        """
        Get all 32 prompt combinations (BASE + 31 combinations).

        Returns:
            List of component strings
        """
        from itertools import combinations

        base_components = ['G', 'M', 'H', 'W', 'P']
        all_combos = ['BASE']

        # All non-empty subsets
        for r in range(1, len(base_components) + 1):
            for combo in combinations(base_components, r):
                all_combos.append(''.join(sorted(combo)))

        return all_combos

    @staticmethod
    def format_history(
        history: List[Dict],
        max_display: int = 5,
        include_warning: bool = True
    ) -> str:
        """
        Format game history for prompt display.

        Args:
            history: List of round data dictionaries
            max_display: Maximum rounds to display
            include_warning: Whether to include consecutive loss warning

        Returns:
            Formatted history text
        """
        if not history:
            return ""

        text = "Game History:\n"
        recent_history = history[-max_display:]

        for h in recent_history:
            round_num = h.get('round', 0)
            text += PromptBuilder._format_round(h, round_num)

        # Consecutive loss warning
        if include_warning:
            consecutive_losses = 0
            for h in reversed(history):
                if h.get('result') == 'L' or not h.get('win', True):
                    consecutive_losses += 1
                else:
                    break

            if consecutive_losses >= 3:
                text += f"\n⚠️ Currently {consecutive_losses} consecutive losses.\n"

        return text

    @staticmethod
    def _format_round(round_data: Dict, round_num: int) -> str:
        """Format a single round for display (override in subclasses)"""
        # Default implementation
        result = round_data.get('result', 'UNKNOWN')
        return f"Round {round_num}: Result {result}\n"
