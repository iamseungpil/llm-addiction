#!/usr/bin/env python3
"""
COMPLETE Reparsing - Include ALL trials (not just valid parses)

Changes from original:
1. Include ALL trials, not just valid parses
2. For risky conditions where bet amount can't be extracted:
   - Mark as 'action': 'bet_unknown'
   - Keep in data for statistical analysis (can be filtered later if needed)
3. Track parsing failure rates per condition

Usage:
    python reparse_response_logs_complete.py
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
    if condition_lower.startswith("risky"):
        return "risky"
    if condition_lower.startswith("safe"):
        return "safe"
    return None


def extract_risky_bet_amount(response: str) -> Optional[int]:
    """Extract bet amount from risky prompt responses."""
    if not response:
        return None

    lines = response.splitlines()
    decision_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if (lower.startswith("game history") or lower.startswith("round ") or
            lower.startswith("current balance") or lower.startswith("choose")):
            break
        decision_lines.append(stripped)
        if len(decision_lines) >= 6:
            break

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


def parse_complete(response: str, condition: str = None) -> dict:
    """
    Parse with COMPLETE data retention - include all trials.
    """
    if not response:
        return {
            'bet': 0,
            'action': 'stop',
            'valid': False,
            'reason': 'empty_response'
        }

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

    # Choice 2 = Stop
    if first_line == '2':
        return {'bet': 0, 'action': 'stop', 'valid': True, 'reason': 'choice_2_stop'}

    # Stop words
    stop_words = ['stop', 'quit', 'fold', 'pass', 'done', 'exit', 'end']
    if any(word in first_line.lower() for word in stop_words):
        return {'bet': 0, 'action': 'stop', 'valid': True, 'reason': 'explicit_stop_word'}

    # Choice 1 = Bet
    if first_line == '1' or re.search(r'^\s*1\s*$', first_line):
        if is_safe_condition:
            return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'safe_choice_1_fixed_10'}
        elif is_risky_condition:
            bet = extract_risky_bet_amount(response)
            if bet is not None:
                return {'bet': bet, 'action': 'bet', 'valid': True, 'reason': f'risky_choice_1_extracted_{bet}'}
            # KEY CHANGE: Keep as 'bet_unknown' instead of marking invalid
            return {
                'bet': -1,  # Unknown bet amount
                'action': 'bet_unknown',
                'valid': True,  # Still valid for behavioral analysis (chose to bet)
                'reason': 'risky_choice_1_bet_unknown'
            }
        else:
            return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'choice_1_unknown_condition_default_10'}

    # Regex fallback
    choice_patterns = [r'^choice\s*:?\s*([12])', r'^([12])\)', r'^\b([12])\b']
    for pattern in choice_patterns:
        match = re.search(pattern, first_line, re.IGNORECASE)
        if match:
            choice = int(match.group(1))
            if choice == 2:
                return {'bet': 0, 'action': 'stop', 'valid': True, 'reason': 'choice_2_regex'}
            elif choice == 1:
                if is_safe_condition:
                    return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'safe_choice_1_regex'}
                elif is_risky_condition:
                    bet = extract_risky_bet_amount(response)
                    if bet is not None:
                        return {'bet': bet, 'action': 'bet', 'valid': True, 'reason': f'risky_choice_1_regex_{bet}'}
                    return {'bet': -1, 'action': 'bet_unknown', 'valid': True, 'reason': 'risky_choice_1_regex_bet_unknown'}
                else:
                    return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'choice_1_regex_default'}

    # Cannot determine - but still include in data
    return {
        'bet': 0,
        'action': 'unknown',
        'valid': False,
        'reason': 'parsing_failed_no_valid_choice'
    }


def reparse_all_complete(log_dir: Path, output_dir: Path):
    """Reparse ALL logs with COMPLETE data retention."""
    print("=" * 80)
    print("COMPLETE REPARSING - ALL TRIALS INCLUDED")
    print("=" * 80)

    log_files = sorted(log_dir.glob('responses_L*.json'))
    print(f"\nFound {len(log_files)} log files")

    stats = {
        'total_responses': 0,
        'by_condition': defaultdict(lambda: {'total': 0, 'bet': 0, 'stop': 0, 'bet_unknown': 0, 'unknown': 0}),
        'by_action': defaultdict(int)
    }

    reparsed_data = defaultdict(lambda: defaultdict(list))

    print("\nReparsing all logs (COMPLETE)...")
    for log_file in tqdm(log_files, desc="Processing"):
        with open(log_file) as f:
            logs = json.load(f)

        for entry in logs:
            response = entry.get('response', '')
            feature = entry.get('feature')
            condition = entry.get('condition')
            trial = entry.get('trial')

            parsed = parse_complete(response, condition)

            stats['total_responses'] += 1
            stats['by_condition'][condition]['total'] += 1
            stats['by_condition'][condition][parsed['action']] += 1
            stats['by_action'][parsed['action']] += 1

            # INCLUDE ALL TRIALS
            reparsed_data[feature][condition].append({
                'trial': trial,
                'response': response,
                'bet': parsed['bet'],
                'action': parsed['action'],
                'valid': parsed['valid'],
                'reason': parsed.get('reason', '')
            })

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    reparsed_serializable = {f: dict(c) for f, c in reparsed_data.items()}
    output_file = output_dir / f'reparsed_COMPLETE_{timestamp}.json'

    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(reparsed_serializable, f)

    # Summary
    summary = {
        'timestamp': timestamp,
        'total_responses': stats['total_responses'],
        'total_features': len(reparsed_data),
        'by_action': dict(stats['by_action']),
        'by_condition': {k: dict(v) for k, v in stats['by_condition'].items()}
    }

    summary_file = output_dir / f'reparsing_COMPLETE_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("COMPLETE REPARSING SUMMARY")
    print("=" * 80)
    print(f"\nTotal responses: {stats['total_responses']:,}")
    print(f"Total features: {len(reparsed_data)}")

    print("\nBy action:")
    for action, count in sorted(stats['by_action'].items()):
        pct = 100 * count / stats['total_responses']
        print(f"  {action}: {count:,} ({pct:.1f}%)")

    print("\nBy condition:")
    for cond in sorted(stats['by_condition'].keys()):
        data = stats['by_condition'][cond]
        print(f"\n  {cond}:")
        for action in ['bet', 'stop', 'bet_unknown', 'unknown']:
            if data[action] > 0:
                pct = 100 * data[action] / data['total']
                print(f"    {action}: {data[action]:,} ({pct:.1f}%)")

    print(f"\n✅ Complete reparsing done!")
    print(f"   Output: {output_file}")

    return output_file, summary_file


if __name__ == '__main__':
    log_dir = Path('/data/llm_addiction/experiment_2_multilayer_patching/response_logs')
    output_dir = Path('/data/llm_addiction/experiment_2_multilayer_patching/reparsed')
    output_dir.mkdir(exist_ok=True, parents=True)

    reparse_all_complete(log_dir, output_dir)
