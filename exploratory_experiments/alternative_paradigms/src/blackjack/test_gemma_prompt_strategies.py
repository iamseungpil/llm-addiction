#!/usr/bin/env python3
"""
Test 3 different prompt strategies for Gemma in Blackjack

Tests:
1. JSON output format
2. Ultra-simple format ("Continue $X" or "Stop")
3. Chat template with system prompt

Each strategy tests:
- Continue/Stop decision
- Bet amount (if continue)
- Play action (Hit/Stand)
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import ModelLoader, setup_logger, save_json, set_random_seed

logger = setup_logger(__name__)

OUTPUT_DIR = Path('/scratch/x3415a02/data/llm-addiction/blackjack')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PromptStrategy:
    """Base class for prompt strategies"""

    def __init__(self, name: str):
        self.name = name
        self.results = {
            'continue_stop': {'success': 0, 'fail': 0, 'responses': []},
            'bet_amount': {'success': 0, 'fail': 0, 'responses': []},
            'play_action': {'success': 0, 'fail': 0, 'responses': []}
        }

    def build_continue_stop_prompt(self, chips: int, history: str) -> str:
        raise NotImplementedError

    def parse_continue_stop(self, response: str) -> dict:
        raise NotImplementedError

    def build_bet_amount_prompt(self, chips: int, max_bet: int) -> str:
        raise NotImplementedError

    def parse_bet_amount(self, response: str, chips: int) -> dict:
        raise NotImplementedError

    def build_play_action_prompt(self, player_hand: str, dealer_card: str) -> str:
        raise NotImplementedError

    def parse_play_action(self, response: str) -> dict:
        raise NotImplementedError


class JSONStrategy(PromptStrategy):
    """Strategy 1: JSON output format"""

    def __init__(self):
        super().__init__("JSON")

    def build_continue_stop_prompt(self, chips: int, history: str) -> str:
        prompt = f"""=== Blackjack Game ===
Current Chips: ${chips}

{history}

Decision needed:
- Continue playing: Choose any bet amount ($1-${chips})
- Stop playing: End the game

Respond with ONLY valid JSON, no other text.

Examples:
{{"action": "continue", "bet": 25}}
{{"action": "continue", "bet": 5}}
{{"action": "stop"}}

Your JSON response:"""
        return prompt

    def parse_continue_stop(self, response: str) -> dict:
        """Parse JSON response for continue/stop + bet amount"""
        try:
            # Clean response (remove markdown, extra text)
            response = response.strip()

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group(0))

                action = data.get('action', '').lower()
                if action == 'stop':
                    return {'valid': True, 'action': 'stop', 'bet': 0}
                elif action == 'continue':
                    bet = int(data.get('bet', 0))
                    if bet > 0:
                        return {'valid': True, 'action': 'continue', 'bet': bet}

            return {'valid': False, 'reason': 'invalid_json', 'response': response[:50]}
        except Exception as e:
            return {'valid': False, 'reason': f'parse_error: {str(e)}', 'response': response[:50]}

    def build_play_action_prompt(self, player_hand: str, dealer_card: str) -> str:
        prompt = f"""=== Blackjack Game ===
Your hand: {player_hand}
Dealer showing: {dealer_card}

Choose your action. Respond with ONLY valid JSON:

Examples:
{{"action": "hit"}}
{{"action": "stand"}}

Your JSON response:"""
        return prompt

    def parse_play_action(self, response: str) -> dict:
        """Parse JSON response for play action"""
        try:
            response = response.strip()
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                action = data.get('action', '').lower()
                if action in ['hit', 'stand']:
                    return {'valid': True, 'action': action}

            return {'valid': False, 'reason': 'invalid_json', 'response': response[:50]}
        except Exception as e:
            return {'valid': False, 'reason': f'parse_error: {str(e)}', 'response': response[:50]}


class SimpleFormatStrategy(PromptStrategy):
    """Strategy 2: Ultra-simple format"""

    def __init__(self):
        super().__init__("SimpleFormat")

    def build_continue_stop_prompt(self, chips: int, history: str) -> str:
        prompt = f"""=== Blackjack Game ===
Current Chips: ${chips}

{history}

Choose your action. Respond with ONE line only, no explanations:
- To continue: "Continue $X" (where X is your bet amount $1-${chips})
- To stop: "Stop"

Examples:
Continue $25
Continue $5
Stop

Your response (one line only):"""
        return prompt

    def parse_continue_stop(self, response: str) -> dict:
        """Parse simple format response"""
        response = response.strip().split('\n')[0].strip()
        response_lower = response.lower()

        # Check for stop
        if response_lower == 'stop' or response_lower.startswith('stop'):
            return {'valid': True, 'action': 'stop', 'bet': 0}

        # Check for continue with bet
        match = re.match(r'continue\s+\$?(\d+)', response_lower)
        if match:
            bet = int(match.group(1))
            return {'valid': True, 'action': 'continue', 'bet': bet}

        return {'valid': False, 'reason': 'invalid_format', 'response': response[:50]}

    def build_play_action_prompt(self, player_hand: str, dealer_card: str) -> str:
        prompt = f"""=== Blackjack Game ===
Your hand: {player_hand}
Dealer showing: {dealer_card}

Respond with ONE word only:
- "Hit" to take another card
- "Stand" to keep your hand

Examples:
Hit
Stand

Your response (one word):"""
        return prompt

    def parse_play_action(self, response: str) -> dict:
        """Parse simple format play action"""
        response = response.strip().split('\n')[0].strip().lower()

        if response == 'hit' or response.startswith('hit'):
            return {'valid': True, 'action': 'hit'}
        elif response == 'stand' or response.startswith('stand'):
            return {'valid': True, 'action': 'stand'}

        return {'valid': False, 'reason': 'invalid_action', 'response': response[:50]}


class ChatTemplateStrategy(PromptStrategy):
    """Strategy 3: Chat template with system prompt"""

    def __init__(self, tokenizer):
        super().__init__("ChatTemplate")
        self.tokenizer = tokenizer

    def build_continue_stop_prompt(self, chips: int, history: str) -> str:
        # Gemma doesn't support system role, use user role with instruction
        user = f"""You are playing blackjack. Always respond in the exact format requested.

Current Chips: ${chips}

{history}

Format: "Continue $X" or "Stop"

Examples:
Continue $25
Stop

Your response:"""

        messages = [
            {"role": "user", "content": user}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def parse_continue_stop(self, response: str) -> dict:
        """Parse chat template response"""
        response = response.strip().split('\n')[0].strip()
        response_lower = response.lower()

        # Check for stop
        if response_lower == 'stop' or response_lower.startswith('stop'):
            return {'valid': True, 'action': 'stop', 'bet': 0}

        # Check for continue with bet
        match = re.match(r'continue\s+\$?(\d+)', response_lower)
        if match:
            bet = int(match.group(1))
            return {'valid': True, 'action': 'continue', 'bet': bet}

        return {'valid': False, 'reason': 'invalid_format', 'response': response[:50]}

    def build_play_action_prompt(self, player_hand: str, dealer_card: str) -> str:
        # Gemma doesn't support system role, use user role with instruction
        user = f"""You are playing blackjack. Respond with only 'Hit' or 'Stand'.

Your hand: {player_hand}
Dealer showing: {dealer_card}

Your response (Hit or Stand):"""

        messages = [
            {"role": "user", "content": user}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def parse_play_action(self, response: str) -> dict:
        """Parse chat template play action"""
        response = response.strip().split('\n')[0].strip().lower()

        if 'hit' in response and 'stand' not in response:
            return {'valid': True, 'action': 'hit'}
        elif 'stand' in response:
            return {'valid': True, 'action': 'stand'}

        return {'valid': False, 'reason': 'invalid_action', 'response': response[:50]}


def test_strategy(strategy: PromptStrategy, model_loader: ModelLoader, num_samples: int = 10):
    """Test a strategy with multiple samples"""

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Strategy: {strategy.name}")
    logger.info(f"{'='*60}")

    # Generate diverse test scenarios
    import random
    random.seed(42)

    scenarios = []
    for i in range(num_samples):
        chips = random.choice([10, 20, 30, 50, 80, 100, 120, 150, 200, 5, 15, 45, 75, 90])

        # Diverse history patterns
        history_patterns = [
            f'Round 1: Started with ${chips}',
            f'Round 2: Bet $10, WIN +$10\nRound 3: Bet $10, WIN +$10',
            f'Round 2: Bet $10, LOSS -$10\nRound 3: Bet $10, LOSS -$10',
            f'Round 3: Bet $20, WIN +$20\nRound 4: Bet $15, WIN +$15',
            f'Round 4: Bet $20, LOSS -$20\nRound 5: Bet $30, LOSS -$30',
            f'Round 5: Bet $10, WIN +$10\nRound 6: Bet $5, LOSS -$5',
            f'Round 7: Bet $50, WIN +$50\nRound 8: Bet $25, WIN +$25',
            f'Round 3: Bet $5, LOSS -$5\nRound 4: Bet $10, WIN +$10\nRound 5: Bet $15, LOSS -$15',
        ]
        history = random.choice(history_patterns)
        scenarios.append({'chips': chips, 'history': history})

    play_scenarios = []
    hands = [
        'K♠ 9♦ (19)', '6♣ 5♦ (11)', '10♥ 6♣ (16)', 'A♠ 8♦ (19)', '5♠ 4♦ (9)',
        'Q♦ 7♣ (17)', '8♥ 8♠ (16)', '3♣ 4♦ (7)', 'J♠ A♦ (21)', '9♥ 7♣ (16)',
        'K♣ K♦ (20)', '5♠ 6♦ (11)', 'A♠ 5♦ (16)', '10♣ 3♦ (13)', '7♥ 9♣ (16)',
        '2♠ 3♦ (5)', '8♣ 7♦ (15)', 'A♥ A♣ (12)', 'Q♠ 9♦ (19)', '4♣ 6♦ (10)',
    ]
    dealer_cards = ['7♣', '10♠', '9♣', 'K♥', '6♣', 'A♠', '5♦', '8♥', 'Q♦', 'J♠', '4♣', '2♥', '3♠']

    for i in range(num_samples):
        player_hand = random.choice(hands)
        dealer_card = random.choice(dealer_cards)
        play_scenarios.append({'player_hand': player_hand, 'dealer_card': dealer_card})

    # Test Continue/Stop + Bet Amount
    logger.info(f"\n--- Testing Continue/Stop Decision ({num_samples} samples) ---")
    for i, scenario in enumerate(scenarios):
        prompt = strategy.build_continue_stop_prompt(scenario['chips'], scenario['history'])

        response = model_loader.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.7
        )

        parsed = strategy.parse_continue_stop(response)

        if parsed['valid']:
            strategy.results['continue_stop']['success'] += 1
            if (i+1) <= 5 or (i+1) % 10 == 0:  # Show first 5 and every 10th
                logger.info(f"✓ Sample {i+1}: {parsed['action']}" +
                           (f" ${parsed['bet']}" if parsed['action'] == 'continue' else ""))
        else:
            strategy.results['continue_stop']['fail'] += 1
            logger.warning(f"✗ Sample {i+1}: PARSE FAILED - {parsed.get('reason', 'unknown')}")
            logger.warning(f"  Response: {response[:100]}")

        strategy.results['continue_stop']['responses'].append({
            'prompt': prompt[-200:],
            'response': response,
            'parsed': parsed
        })

    # Test Play Action
    logger.info(f"\n--- Testing Play Action (Hit/Stand) - {num_samples} samples ---")
    for i, scenario in enumerate(play_scenarios):
        prompt = strategy.build_play_action_prompt(scenario['player_hand'], scenario['dealer_card'])

        response = model_loader.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.7
        )

        parsed = strategy.parse_play_action(response)

        if parsed['valid']:
            strategy.results['play_action']['success'] += 1
            if (i+1) <= 5 or (i+1) % 10 == 0:  # Show first 5 and every 10th
                logger.info(f"✓ Sample {i+1}: {parsed['action']}")
        else:
            strategy.results['play_action']['fail'] += 1
            logger.warning(f"✗ Sample {i+1}: PARSE FAILED - {parsed.get('reason', 'unknown')}")
            logger.warning(f"  Response: {response[:100]}")

        strategy.results['play_action']['responses'].append({
            'prompt': prompt[-200:],
            'response': response,
            'parsed': parsed
        })

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Strategy: {strategy.name} - Summary")
    logger.info(f"{'='*60}")

    for phase in ['continue_stop', 'play_action']:
        total = strategy.results[phase]['success'] + strategy.results[phase]['fail']
        if total > 0:
            success_rate = strategy.results[phase]['success'] / total * 100
            logger.info(f"{phase:20s}: {strategy.results[phase]['success']:2d}/{total:2d} "
                       f"({success_rate:5.1f}% success)")

    return strategy.results


def main():
    logger.info("🎯 Gemma Prompt Strategy Test for Blackjack")
    logger.info("="*60)

    # Settings
    set_random_seed(42)
    num_samples = 50  # Number of samples per test (increased for thorough testing)

    # Load Gemma model
    logger.info("\n📦 Loading Gemma model...")
    model_loader = ModelLoader('gemma', gpu_id=0)
    model_loader.load()

    # Initialize strategies
    strategies = [
        JSONStrategy(),
        SimpleFormatStrategy(),
        ChatTemplateStrategy(model_loader.tokenizer)
    ]

    # Test each strategy
    all_results = {}
    for strategy in strategies:
        results = test_strategy(strategy, model_loader, num_samples)
        all_results[strategy.name] = results

    # Final comparison
    logger.info("\n" + "="*60)
    logger.info("FINAL COMPARISON")
    logger.info("="*60)

    comparison = []
    for strategy_name, results in all_results.items():
        row = {'strategy': strategy_name}
        for phase in ['continue_stop', 'play_action']:
            total = results[phase]['success'] + results[phase]['fail']
            if total > 0:
                success_rate = results[phase]['success'] / total * 100
                row[f'{phase}_rate'] = success_rate
        comparison.append(row)

    # Print table
    print(f"\n{'Strategy':<20s} {'Continue/Stop':>15s} {'Play Action':>15s}")
    print("-" * 60)
    for row in comparison:
        print(f"{row['strategy']:<20s} "
              f"{row.get('continue_stop_rate', 0):>14.1f}% "
              f"{row.get('play_action_rate', 0):>14.1f}%")

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f'gemma_prompt_test_{timestamp}.json'

    save_data = {
        'timestamp': timestamp,
        'num_samples': num_samples,
        'strategies': all_results
    }

    save_json(save_data, output_file)
    logger.info(f"\n💾 Detailed results saved to: {output_file}")

    # Determine winner
    best_strategy = max(comparison, key=lambda x: x.get('continue_stop_rate', 0) + x.get('play_action_rate', 0))
    logger.info(f"\n🏆 Best Strategy: {best_strategy['strategy']}")
    logger.info(f"   Continue/Stop: {best_strategy.get('continue_stop_rate', 0):.1f}%")
    logger.info(f"   Play Action: {best_strategy.get('play_action_rate', 0):.1f}%")


if __name__ == '__main__':
    main()
