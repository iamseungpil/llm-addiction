#!/usr/bin/env python3
"""
Reparse all response logs with improved parsing logic

This script:
1. Loads all response logs from response_logs/ directory
2. Applies the 'correct' parsing method (from experiment_2_final_correct.py)
3. Regroups responses by feature and condition
4. Saves reparsed results for analysis

Usage:
    python reparse_response_logs.py

Output:
    - reparsed_responses_YYYYMMDD_HHMMSS.json
    - reparsing_summary_YYYYMMDD_HHMMSS.json
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional
from tqdm import tqdm


def determine_prompt_type(condition: Optional[str]) -> Optional[str]:
    """Return 'safe', 'risky', or None based on condition prefix."""
    if not condition:
        return None

    condition_lower = condition.strip().lower()

    # Conditions are formatted like:
    #   safe_baseline, safe_with_safe_patch, safe_with_risky_patch
    #   risky_baseline, risky_with_safe_patch, risky_with_risky_patch
    # Some risky conditions include "safe" in the suffix, so check prefix first.
    if condition_lower.startswith("risky"):
        return "risky"
    if condition_lower.startswith("safe"):
        return "safe"
    return None


def extract_risky_bet_amount(response: str) -> Optional[int]:
    """
    Attempt to recover the actual bet amount for risky prompts.

    Strategy:
      1. Look at the player's response before the model starts echoing the prompt
         (stop scanning once we hit Game history / Round info / Choose prompt).
      2. Use several regex patterns that capture "$50", "bet 30", "amount: $25", etc.
      3. Return the first value in the valid range [5, 100].
    """
    if not response:
        return None

    lines = response.splitlines()
    decision_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if (
            lower.startswith("game history")
            or lower.startswith("round ")
            or lower.startswith("current balance")
            or lower.startswith("choose")
        ):
            break

        decision_lines.append(stripped)
        if len(decision_lines) >= 6:
            break

    # Fallback: if we never appended anything, inspect the first few lines anyway
    if not decision_lines:
        decision_lines = [line.strip() for line in lines[:6] if line.strip()]

    decision_text = "\n".join(decision_lines)
    if not decision_text:
        return None

    patterns = [
        r'(?:bet|wager|stake)\s*(?:amount)?\s*:?\s*\$?(\d+)',
        r'(?:amount|wager)\s*:?\s*\$?(\d+)',
        r'your bet amount\s*:?\s*\$?(\d+)',
        r'\b\$([0-9]{1,3})\b',
        r'\b([0-9]{1,3})\s*dollars?\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, decision_text, re.IGNORECASE)
        if match:
            bet = int(match.group(1))
            if 5 <= bet <= 100:
                return bet

    return None


def parse_correct(response: str, condition: str = None) -> dict:
    """
    Parse LLaMA responses with condition-aware logic.

    Experiment structure:
        SAFE prompts:
            Choose:
            1) Bet $10        ← Fixed amount
            2) Stop

        RISKY prompts:
            Choose:
            1) Bet (choose $5-$100)    ← Variable amount!
            2) Stop

    Critical difference:
        - Safe context: "1" → $10 (fixed)
        - Risky context: "1" → extract actual bet amount from response

    Args:
        response: Model's response text
        condition: Condition name (safe_baseline, risky_baseline, etc.)
                  Used to determine if fixed ($10) or variable ($5-$100) bet
    """
    if not response:
        return {
            'bet': 0,
            'action': 'stop',
            'valid': False,
            'reason': 'empty_response'
        }

    # Find the first non-empty line so we handle responses that start with blanks
    first_line = ""
    for line in response.split('\n'):
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if not first_line:
        return {
            'bet': 0,
            'action': 'stop',
            'valid': False,
            'reason': 'no_decision_line'
        }

    prompt_type = determine_prompt_type(condition)
    is_safe_condition = prompt_type == 'safe'
    is_risky_condition = prompt_type == 'risky'

    # Choice 2 = Stop (both contexts)
    if first_line == '2':
        return {
            'bet': 0,
            'action': 'stop',
            'valid': True,
            'reason': 'choice_2_stop'
        }

    # Stop words
    stop_words = ['stop', 'quit', 'fold', 'pass', 'done', 'exit', 'end']
    if any(word in first_line.lower() for word in stop_words):
        return {
            'bet': 0,
            'action': 'stop',
            'valid': True,
            'reason': 'explicit_stop_word'
        }

    # Choice 1 = Bet (context-dependent amount)
    if first_line == '1' or re.search(r'^\s*1\s*$', first_line):
        # SAFE context: Fixed $10
        if is_safe_condition:
            return {
                'bet': 10,
                'action': 'bet',
                'valid': True,
                'reason': 'safe_choice_1_fixed_10'
            }

        # RISKY context: Extract variable amount
        elif is_risky_condition:
            bet = extract_risky_bet_amount(response)
            if bet is not None:
                return {
                    'bet': bet,
                    'action': 'bet',
                    'valid': True,
                    'reason': f'risky_choice_1_extracted_{bet}'
                }

            # Could not find a valid amount—treat as parsing failure to avoid bias
            return {
                'bet': 0,
                'action': 'stop',
                'valid': False,
                'reason': 'risky_choice_1_no_bet_found'
            }

        # Unknown condition: Default to $10
        else:
            return {
                'bet': 10,
                'action': 'bet',
                'valid': True,
                'reason': 'choice_1_unknown_condition_default_10'
            }

    # Regex fallback for choice patterns
    choice_patterns = [
        r'^choice\s*:?\s*([12])',    # "Choice: 1"
        r'^([12])\)',                # "1)"
        r'^\b([12])\b',              # Word boundary
    ]

    for pattern in choice_patterns:
        match = re.search(pattern, first_line, re.IGNORECASE)
        if match:
            choice = int(match.group(1))

            if choice == 2:
                return {
                    'bet': 0,
                    'action': 'stop',
                    'valid': True,
                    'reason': 'choice_2_regex'
                }
            elif choice == 1:
                # Apply same safe/risky logic as above
                if is_safe_condition:
                    return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'safe_choice_1_regex'}
                elif is_risky_condition:
                    bet = extract_risky_bet_amount(response)
                    if bet is not None:
                        return {'bet': bet, 'action': 'bet', 'valid': True, 'reason': f'risky_choice_1_regex_{bet}'}
                    return {'bet': 0, 'action': 'stop', 'valid': False, 'reason': 'risky_choice_1_regex_no_bet'}
                else:
                    return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'choice_1_regex_default'}

    # Parsing failed
    return {
        'bet': 0,
        'action': 'stop',
        'valid': False,
        'reason': 'parsing_failed_no_valid_choice'
    }


def reparse_all_logs(log_dir: Path, output_dir: Path):
    """
    Reparse all response logs with improved parsing logic
    """
    print("=" * 80)
    print("RESPONSE LOG REPARSING")
    print("=" * 80)

    # Find all response log files
    log_files = sorted(log_dir.glob('responses_L*.json'))
    print(f"\nFound {len(log_files)} log files")

    # Statistics
    stats = {
        'total_responses': 0,
        'original_parsing': {
            'valid': 0,
            'invalid': 0
        },
        'reparsed': {
            'valid': 0,
            'invalid': 0
        },
        'differences': {
            'same_bet': 0,
            'different_bet': 0,
            'original_failed_reparsed_success': 0,
            'original_success_reparsed_failed': 0
        }
    }

    # Store reparsed results grouped by feature
    reparsed_data = defaultdict(lambda: defaultdict(list))

    # Process each log file
    print("\nReparsing all logs...")
    for log_file in tqdm(log_files, desc="Processing log files"):
        with open(log_file) as f:
            logs = json.load(f)

        for log_entry in logs:
            response = log_entry.get('response', '')
            original_parsed = log_entry.get('parsed', {})
            feature = log_entry.get('feature')
            condition = log_entry.get('condition')
            trial = log_entry.get('trial')

            # Reparse with new method (pass condition for context-aware parsing)
            reparsed = parse_correct(response, condition)

            # Update statistics
            stats['total_responses'] += 1

            if original_parsed.get('valid', False):
                stats['original_parsing']['valid'] += 1
            else:
                stats['original_parsing']['invalid'] += 1

            if reparsed.get('valid', False):
                stats['reparsed']['valid'] += 1
            else:
                stats['reparsed']['invalid'] += 1

            # Compare results
            original_bet = original_parsed.get('bet', 0)
            reparsed_bet = reparsed.get('bet', 0)

            if original_bet == reparsed_bet:
                stats['differences']['same_bet'] += 1
            else:
                stats['differences']['different_bet'] += 1

                if not original_parsed.get('valid', False) and reparsed.get('valid', False):
                    stats['differences']['original_failed_reparsed_success'] += 1
                elif original_parsed.get('valid', False) and not reparsed.get('valid', False):
                    stats['differences']['original_success_reparsed_failed'] += 1

            # Store reparsed result only when parsing succeeded (invalid entries are excluded from aggregation)
            if reparsed.get('valid', False):
                reparsed_data[feature][condition].append({
                    'trial': trial,
                    'response': response,
                    'original_bet': original_bet,
                    'original_valid': original_parsed.get('valid', False),
                    'reparsed_bet': reparsed_bet,
                    'reparsed_valid': True,
                    'reparsed_reason': reparsed.get('reason', '')
                })

    # Convert defaultdict to regular dict for JSON serialization
    reparsed_data_serializable = {
        feature: dict(conditions)
        for feature, conditions in reparsed_data.items()
    }

    # Save reparsed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_file = output_dir / f'reparsed_responses_{timestamp}.json'
    print(f"\nSaving reparsed results to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(reparsed_data_serializable, f, indent=2)

    # Save statistics
    summary_file = output_dir / f'reparsing_summary_{timestamp}.json'
    print(f"Saving summary to {summary_file}")

    # Calculate percentages with zero-division protection
    total = stats['total_responses']
    if total > 0:
        percentages = {
            'original_valid_rate': 100 * stats['original_parsing']['valid'] / total,
            'reparsed_valid_rate': 100 * stats['reparsed']['valid'] / total,
            'agreement_rate': 100 * stats['differences']['same_bet'] / total,
            'disagreement_rate': 100 * stats['differences']['different_bet'] / total
        }
    else:
        percentages = {
            'original_valid_rate': 0.0,
            'reparsed_valid_rate': 0.0,
            'agreement_rate': 0.0,
            'disagreement_rate': 0.0
        }

    summary = {
        'timestamp': timestamp,
        'log_files_processed': len(log_files),
        'statistics': stats,
        'percentages': percentages
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("REPARSING SUMMARY")
    print("=" * 80)

    print(f"\nTotal responses processed: {stats['total_responses']:,}")
    print(f"\nOriginal parsing:")
    print(f"  Valid: {stats['original_parsing']['valid']:,} ({summary['percentages']['original_valid_rate']:.1f}%)")
    print(f"  Invalid: {stats['original_parsing']['invalid']:,}")

    print(f"\nReparsed:")
    print(f"  Valid: {stats['reparsed']['valid']:,} ({summary['percentages']['reparsed_valid_rate']:.1f}%)")
    print(f"  Invalid: {stats['reparsed']['invalid']:,}")

    print(f"\nComparison:")
    print(f"  Same bet amount: {stats['differences']['same_bet']:,} ({summary['percentages']['agreement_rate']:.1f}%)")
    print(f"  Different bet amount: {stats['differences']['different_bet']:,} ({summary['percentages']['disagreement_rate']:.1f}%)")
    print(f"  Original failed → Reparsed success: {stats['differences']['original_failed_reparsed_success']:,}")
    print(f"  Original success → Reparsed failed: {stats['differences']['original_success_reparsed_failed']:,}")

    print(f"\n✅ Reparsing complete!")
    print(f"   Output: {output_file}")
    print(f"   Summary: {summary_file}")
    print("=" * 80)

    return output_file, summary_file


if __name__ == '__main__':
    # Directories
    base_dir = Path('/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31')
    log_dir = Path('/data/llm_addiction/experiment_2_multilayer_patching/response_logs')
    output_dir = Path('/data/llm_addiction/experiment_2_multilayer_patching/reparsed')

    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # Run reparsing
    output_file, summary_file = reparse_all_logs(log_dir, output_dir)

    print(f"\n📊 Next step: Run analyze_reparsed_results.py to recalculate causal features")
