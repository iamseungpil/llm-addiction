#!/usr/bin/env python3
"""
Real Prompt Extractor for 8-Condition Analysis

Extracts representative prompts from actual experimental data:
- 4 conditions per model: safe/risky × fixed/variable
- 8 total conditions when considering both LLaMA and Gemma

Key insight:
- Safe prompts: Use final balance (after last round, before stop decision)
- Risky prompts: Use pre-bankruptcy balance (before the fatal bet)

References:
- Original experiment: experiment_0_llama_gemma_restart
- Data: 3,200 experiments per model (64 conditions × 50 repetitions)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sys

# Add src directory
sys.path.insert(0, str(Path(__file__).parent))
from utils import PromptBuilder


@dataclass
class ConditionPrompt:
    """Representative prompt for a specific condition."""
    condition_name: str  # e.g., "safe_fixed", "risky_variable"
    model: str  # "llama" or "gemma"
    bet_type: str  # "fixed" or "variable"
    outcome: str  # "voluntary_stop" or "bankruptcy"
    prompt: str  # The actual prompt text
    balance: int  # Balance at decision point
    rounds_played: int
    prompt_combo: str  # Original prompt combination (e.g., "GMPRW")
    history_summary: str  # Brief summary of game history


def extract_multiple_prompts(
    results: List[Dict],
    outcome: str,
    bet_type: str,
    model: str,
    n_prompts: int = 25,
    random_seed: int = 42
) -> List[ConditionPrompt]:
    """
    Extract multiple prompts for a specific condition.

    For risky (bankruptcy) cases:
      - Use balance BEFORE the final bet (history[-2]['balance'])
      - Use history EXCLUDING the final round (history[:-1])

    For safe (voluntary_stop) cases:
      - Use final balance (history[-1]['balance'])
      - Use complete history

    Args:
        results: List of experiment results
        outcome: "voluntary_stop" or "bankruptcy"
        bet_type: "fixed" or "variable"
        model: "llama" or "gemma"
        n_prompts: Number of prompts to extract per condition
        random_seed: Random seed for reproducibility

    Returns:
        List of ConditionPrompt objects
    """
    import random
    random.seed(random_seed)

    # Filter matching cases with at least 2 rounds of history
    matches = [
        r for r in results
        if r.get('outcome') == outcome
        and r.get('bet_type') == bet_type
        and r.get('history')  # Must have history
        and len(r.get('history', [])) >= 2  # At least 2 rounds for context
    ]

    if not matches:
        # Fallback: try with just 1 round of history
        matches = [
            r for r in results
            if r.get('outcome') == outcome
            and r.get('bet_type') == bet_type
            and r.get('history')
        ]

    if not matches:
        return []

    # Shuffle and limit to n_prompts
    random.shuffle(matches)
    selected_matches = matches[:min(n_prompts, len(matches))]

    prompts = []
    for selected in selected_matches:
        history = selected.get('history', [])
        prompt_combo = selected.get('prompt_combo', 'BASE')

        # Determine balance and history based on outcome
        if outcome == 'bankruptcy' and len(history) >= 2:
            # For bankruptcy: use state BEFORE the fatal bet
            decision_balance = history[-2]['balance']
            decision_history = history[:-1]
            condition_name = f"risky_{bet_type}"
        elif outcome == 'bankruptcy' and len(history) == 1:
            # Edge case: bankruptcy on first bet
            decision_balance = 100  # Initial balance
            decision_history = []
            condition_name = f"risky_{bet_type}"
        else:
            # For voluntary stop: use final state
            decision_balance = history[-1]['balance'] if history else 100
            decision_history = history
            condition_name = f"safe_{bet_type}"

        # Build prompt
        prompt = PromptBuilder.create_prompt(
            bet_type=bet_type,
            prompt_combo=prompt_combo,
            balance=decision_balance,
            history=decision_history
        )

        # Create history summary
        if decision_history:
            wins = sum(1 for h in decision_history if h.get('win') or h.get('result') == 'W')
            losses = len(decision_history) - wins
            history_summary = f"{len(decision_history)} rounds, {wins}W/{losses}L"
        else:
            history_summary = "0 rounds"

        prompts.append(ConditionPrompt(
            condition_name=condition_name,
            model=model,
            bet_type=bet_type,
            outcome=outcome,
            prompt=prompt,
            balance=decision_balance,
            rounds_played=len(decision_history),
            prompt_combo=prompt_combo,
            history_summary=history_summary
        ))

    return prompts


def extract_representative_prompt(
    results: List[Dict],
    outcome: str,
    bet_type: str,
    model: str,
    prefer_longer_history: bool = True
) -> Optional[ConditionPrompt]:
    """
    Extract a single representative prompt (legacy function).
    Now wraps extract_multiple_prompts for backwards compatibility.
    """
    prompts = extract_multiple_prompts(results, outcome, bet_type, model, n_prompts=1)
    return prompts[0] if prompts else None


def extract_all_conditions(
    llama_data_path: str,
    gemma_data_path: str,
    n_prompts_per_condition: int = 25
) -> Dict[str, List[ConditionPrompt]]:
    """
    Extract multiple prompts for all 8 conditions.

    Args:
        llama_data_path: Path to LLaMA experimental data
        gemma_data_path: Path to Gemma experimental data
        n_prompts_per_condition: Number of prompts per condition (default: 25)

    Returns:
        Dict with 'llama' and 'gemma' keys, each containing list of ConditionPrompts
    """
    result = {'llama': [], 'gemma': []}

    conditions = [
        ('voluntary_stop', 'fixed'),   # safe_fixed
        ('voluntary_stop', 'variable'), # safe_variable
        ('bankruptcy', 'fixed'),        # risky_fixed
        ('bankruptcy', 'variable'),     # risky_variable
    ]

    # Load and process LLaMA data
    with open(llama_data_path, 'r') as f:
        llama_data = json.load(f)

    for outcome, bet_type in conditions:
        prompts = extract_multiple_prompts(
            results=llama_data['results'],
            outcome=outcome,
            bet_type=bet_type,
            model='llama',
            n_prompts=n_prompts_per_condition
        )
        result['llama'].extend(prompts)

    # Load and process Gemma data
    with open(gemma_data_path, 'r') as f:
        gemma_data = json.load(f)

    for outcome, bet_type in conditions:
        prompts = extract_multiple_prompts(
            results=gemma_data['results'],
            outcome=outcome,
            bet_type=bet_type,
            model='gemma',
            n_prompts=n_prompts_per_condition
        )
        result['gemma'].extend(prompts)

    return result


def save_condition_prompts(
    prompts: Dict[str, List[ConditionPrompt]],
    output_path: str
):
    """Save extracted prompts to JSON file."""
    output = {
        'description': '8-condition representative prompts from real experimental data',
        'source': {
            'llama': '/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json',
            'gemma': '/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json'
        },
        'conditions': {
            'llama': [asdict(p) for p in prompts['llama']],
            'gemma': [asdict(p) for p in prompts['gemma']]
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output_path


def print_summary(prompts: Dict[str, List[ConditionPrompt]]):
    """Print summary of extracted prompts."""
    print("=" * 70)
    print("MULTI-CONDITION PROMPTS EXTRACTION")
    print("=" * 70)

    for model in ['llama', 'gemma']:
        print(f"\n{'='*35}")
        print(f" {model.upper()} - {len(prompts[model])} prompts")
        print(f"{'='*35}")

        # Group by condition
        from collections import Counter
        conditions = Counter(p.condition_name for p in prompts[model])
        for cond, count in conditions.items():
            print(f"  {cond}: {count} prompts")

        # Show first prompt of each condition as example
        seen_conditions = set()
        for p in prompts[model]:
            if p.condition_name not in seen_conditions:
                seen_conditions.add(p.condition_name)
                print(f"\n[{p.condition_name} - example]")
                print(f"  Balance: ${p.balance}")
                print(f"  History: {p.history_summary}")
                print(f"  Prompt combo: {p.prompt_combo}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract multi-condition prompts')
    parser.add_argument('--llama-data', type=str,
                        default='/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json')
    parser.add_argument('--gemma-data', type=str,
                        default='/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json')
    parser.add_argument('--output', type=str,
                        default='/data/llm_addiction/steering_vector_experiment_full/condition_prompts.json')
    parser.add_argument('--n-prompts', type=int, default=25,
                        help='Number of prompts per condition (default: 25)')

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Extract prompts
    prompts = extract_all_conditions(
        args.llama_data,
        args.gemma_data,
        n_prompts_per_condition=args.n_prompts
    )

    # Print summary
    print_summary(prompts)

    # Save to file
    output_path = save_condition_prompts(prompts, args.output)
    print(f"\nSaved to: {output_path}")
