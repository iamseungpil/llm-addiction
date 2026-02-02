"""
Prompt loading and processing utilities for investment choice experiment.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_investment_choice_data(data_dir: str, experiment_variant: str = "bet_constraint") -> List[Dict[str, Any]]:
    """
    Load investment choice experiment data from JSON files.

    Args:
        data_dir: Base directory containing investment choice data
        experiment_variant: Which experiment variant to load
                          ('bet_constraint', 'bet_constraint_cot', 'extended_cot', 'initial')

    Returns:
        List of all decisions with metadata
    """
    variant_dir = Path(data_dir) / experiment_variant

    if not variant_dir.exists():
        raise ValueError(f"Experiment variant directory not found: {variant_dir}")

    logger.info(f"Loading data from: {variant_dir}")

    # Find all JSON files in the directory
    json_files = list(variant_dir.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {variant_dir}")

    logger.info(f"Found {len(json_files)} JSON files")

    all_decisions = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract decisions from each game
            results = data.get('results', [])

            for game in results:
                game_id = game.get('game_id')
                model = game.get('model')
                bet_type = game.get('bet_type')
                prompt_condition = game.get('prompt_condition')

                decisions = game.get('decisions', [])

                for decision in decisions:
                    # Create decision record with metadata
                    decision_record = {
                        'game_id': game_id,
                        'model': model,
                        'bet_type': bet_type,
                        'prompt_condition': prompt_condition,
                        'round': decision.get('round'),
                        'balance_before': decision.get('balance_before'),
                        'balance_after': decision.get('balance_after'),
                        'bet': decision.get('bet'),
                        'choice': decision.get('choice'),
                        'outcome': decision.get('outcome'),
                        'win': decision.get('win'),
                        'payout': decision.get('payout', 0),
                        'prompt': decision.get('prompt', ''),
                        'response': decision.get('response', ''),
                        'source_file': json_file.name
                    }

                    all_decisions.append(decision_record)

        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
            continue

    logger.info(f"Loaded {len(all_decisions)} total decisions")

    return all_decisions


def filter_decisions(
    decisions: List[Dict[str, Any]],
    include_models: Optional[List[str]] = None,
    include_bet_types: Optional[List[str]] = None,
    include_conditions: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filter decisions by model, bet type, and prompt condition.

    Args:
        decisions: List of decision records
        include_models: Models to include (None = all)
        include_bet_types: Bet types to include (None = all)
        include_conditions: Prompt conditions to include (None = all)

    Returns:
        Filtered list of decisions
    """
    filtered = decisions

    if include_models and 'all' not in include_models:
        filtered = [d for d in filtered if d['model'] in include_models]
        logger.info(f"Filtered by models {include_models}: {len(filtered)} decisions")

    if include_bet_types and 'both' not in include_bet_types:
        filtered = [d for d in filtered if d['bet_type'] in include_bet_types]
        logger.info(f"Filtered by bet types {include_bet_types}: {len(filtered)} decisions")

    if include_conditions and 'all' not in include_conditions:
        filtered = [d for d in filtered if d['prompt_condition'] in include_conditions]
        logger.info(f"Filtered by conditions {include_conditions}: {len(filtered)} decisions")

    logger.info(f"Final filtered decisions: {len(filtered)}")

    return filtered


def get_prompt_text(decision: Dict[str, Any]) -> str:
    """
    Extract prompt text from decision record.

    Args:
        decision: Decision record

    Returns:
        Prompt text
    """
    return decision.get('prompt', '')


def get_choice_label(choice: int) -> str:
    """
    Convert choice number to human-readable label.

    Args:
        choice: Choice number (1-4)

    Returns:
        Choice label
    """
    labels = {
        1: "Safe (Option 1)",
        2: "Low Risk (Option 2)",
        3: "Medium Risk (Option 3)",
        4: "High Risk (Option 4)"
    }
    return labels.get(choice, f"Unknown ({choice})")


def create_dataset_summary(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics of the dataset.

    Args:
        decisions: List of decision records

    Returns:
        Summary dictionary
    """
    from collections import Counter

    total = len(decisions)

    models = Counter(d['model'] for d in decisions)
    bet_types = Counter(d['bet_type'] for d in decisions)
    conditions = Counter(d['prompt_condition'] for d in decisions)
    choices = Counter(d['choice'] for d in decisions)
    rounds = Counter(d['round'] for d in decisions)

    summary = {
        'total_decisions': total,
        'by_model': dict(models),
        'by_bet_type': dict(bet_types),
        'by_condition': dict(conditions),
        'by_choice': dict(choices),
        'by_round': dict(rounds),
        'unique_games': len(set(d['game_id'] for d in decisions))
    }

    return summary


def print_dataset_summary(decisions: List[Dict[str, Any]]):
    """
    Print dataset summary to console.

    Args:
        decisions: List of decision records
    """
    summary = create_dataset_summary(decisions)

    print("\n" + "="*60)
    print("INVESTMENT CHOICE DATASET SUMMARY")
    print("="*60)
    print(f"Total decisions: {summary['total_decisions']:,}")
    print(f"Unique games: {summary['unique_games']:,}")
    print()

    print("By Model:")
    for model, count in sorted(summary['by_model'].items()):
        print(f"  {model:20s}: {count:6,} ({count/summary['total_decisions']*100:5.1f}%)")
    print()

    print("By Bet Type:")
    for bet_type, count in sorted(summary['by_bet_type'].items()):
        print(f"  {bet_type:20s}: {count:6,} ({count/summary['total_decisions']*100:5.1f}%)")
    print()

    print("By Prompt Condition:")
    for condition, count in sorted(summary['by_condition'].items()):
        print(f"  {condition:20s}: {count:6,} ({count/summary['total_decisions']*100:5.1f}%)")
    print()

    print("By Choice:")
    for choice in sorted(summary['by_choice'].keys()):
        count = summary['by_choice'][choice]
        label = get_choice_label(choice)
        print(f"  {label:25s}: {count:6,} ({count/summary['total_decisions']*100:5.1f}%)")
    print()

    print("By Round:")
    for round_num in sorted(summary['by_round'].keys()):
        count = summary['by_round'][round_num]
        print(f"  Round {round_num:2d}: {count:6,} ({count/summary['total_decisions']*100:5.1f}%)")
    print("="*60 + "\n")
