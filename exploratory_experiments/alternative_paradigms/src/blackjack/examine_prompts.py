#!/usr/bin/env python3
"""
Examine actual prompts shown to models to understand why they never stop
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import setup_logger

logger = setup_logger(__name__)

def load_data(file_path):
    """Load experiment JSON data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def examine_prompts(data, label, n_examples=2):
    """Examine actual prompts and responses"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Examining Prompts: {label}")
    logger.info(f"{'='*80}")

    games = data['games']

    # Get a few random games
    import random
    random.seed(42)
    sample_games = random.sample(games, min(n_examples, len(games)))

    for i, game in enumerate(sample_games):
        logger.info(f"\n{'-'*80}")
        logger.info(f"Game Example {i+1}")
        logger.info(f"Game ID: {game['game_id']}, Outcome: {game['outcome']}, Rounds: {game['total_rounds']}")
        logger.info(f"{'-'*80}")

        # Show first 3 rounds
        first_rounds = game['rounds'][:3] if len(game['rounds']) >= 3 else game['rounds']

        for round_data in first_rounds:
            logger.info(f"\n📋 Round {round_data['round']}:")
            logger.info(f"   Chips: ${round_data['chips']}, Bet: ${round_data['bet']}")

            # Show prompt
            if 'full_prompt' in round_data and round_data['full_prompt']:
                prompt = round_data['full_prompt']
                logger.info(f"\n   PROMPT:")
                # Show last 500 chars of prompt (most relevant part)
                prompt_preview = prompt[-800:] if len(prompt) > 800 else prompt
                for line in prompt_preview.split('\n'):
                    logger.info(f"   {line}")

            # Show model response if available
            if 'response' in round_data and round_data['response']:
                logger.info(f"\n   MODEL RESPONSE: {round_data['response']}")

            logger.info(f"   Outcome: {round_data['outcome']}, Payout: ${round_data['payout']}")
            logger.info(f"   Stop decision: {round_data['stop']}")
            logger.info("")

def main():
    # File paths
    gemma_fixed_file = "/home/jovyan/beomi/llm-addiction-data/blackjack/blackjack_gemma_20260220_023838.json"
    llama_variable_file = "/home/jovyan/beomi/llm-addiction-data/blackjack/blackjack_llama_20260219_005625.json"

    logger.info("Loading data files...")

    # Load data
    gemma_fixed = load_data(gemma_fixed_file)
    llama_variable = load_data(llama_variable_file)

    # Examine prompts
    examine_prompts(gemma_fixed, "Gemma - Fixed Betting", n_examples=2)
    examine_prompts(llama_variable, "LLaMA - Variable Betting", n_examples=2)

if __name__ == '__main__':
    main()
