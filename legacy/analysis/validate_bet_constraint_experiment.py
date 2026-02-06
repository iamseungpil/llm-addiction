#!/usr/bin/env python3
"""
Comprehensive validation of Investment Choice Bet Constraint Experiment
Check for errors, missing data, and data integrity issues
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

RESULTS_DIR = Path('/data/llm_addiction/investment_choice_bet_constraint/results')

def validate_experiment():
    print("="*80)
    print("INVESTMENT CHOICE BET CONSTRAINT - COMPREHENSIVE VALIDATION")
    print("="*80)

    # Expected configuration
    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']
    constraints = [10, 30, 50, 70]
    bet_types = ['fixed', 'variable']
    prompts = ['BASE', 'G', 'M', 'GM']

    expected_files = len(models) * len(constraints) * len(bet_types)  # 32 files
    expected_games_per_file = 200  # 50 per prompt × 4 prompts
    expected_total_games = expected_files * expected_games_per_file  # 6,400

    print(f"\n📋 Expected Configuration:")
    print(f"  Files: {expected_files}")
    print(f"  Games per file: {expected_games_per_file}")
    print(f"  Total games: {expected_total_games}")

    # Track all files
    files_found = list(RESULTS_DIR.glob('*.json'))
    print(f"\n📁 Files found: {len(files_found)}")

    # Validation results
    issues = []
    file_stats = []
    total_games = 0

    # Check each file
    print("\n" + "="*80)
    print("FILE-BY-FILE VALIDATION")
    print("="*80)

    for result_file in sorted(files_found):
        try:
            with open(result_file) as f:
                data = json.load(f)

            # Extract metadata
            config = data.get('experiment_config', {})
            model = config.get('model', 'UNKNOWN')
            bet_constraint = config.get('bet_constraint', 'UNKNOWN')
            bet_type = config.get('bet_type', 'UNKNOWN')

            results = data.get('results', [])
            n_games = len(results)
            total_games += n_games

            # Count games by prompt
            prompt_counts = Counter()
            decision_counts = []
            error_games = []

            for i, game in enumerate(results):
                prompt = game.get('prompt_condition', 'UNKNOWN')
                prompt_counts[prompt] += 1

                # Count decisions per game
                decisions = game.get('decisions', [])
                decision_counts.append(len(decisions))

                # Check for errors
                if 'error' in game or 'exception' in str(game).lower():
                    error_games.append(i)

            # Validate this file
            file_issues = []

            # Check game count
            if n_games != expected_games_per_file:
                file_issues.append(f"Game count: {n_games} (expected {expected_games_per_file})")

            # Check prompt distribution
            for prompt in prompts:
                count = prompt_counts[prompt]
                if count != 50:
                    file_issues.append(f"Prompt {prompt}: {count} games (expected 50)")

            # Check for unknown prompts
            unknown_prompts = set(prompt_counts.keys()) - set(prompts)
            if unknown_prompts:
                file_issues.append(f"Unknown prompts: {unknown_prompts}")

            # Check for games with no decisions
            games_with_no_decisions = sum(1 for d in decision_counts if d == 0)
            if games_with_no_decisions > 0:
                file_issues.append(f"{games_with_no_decisions} games with no decisions")

            # Check for errors
            if error_games:
                file_issues.append(f"{len(error_games)} games with errors")

            # Store stats
            file_stats.append({
                'file': result_file.name,
                'model': model,
                'constraint': bet_constraint,
                'bet_type': bet_type,
                'n_games': n_games,
                'prompt_counts': dict(prompt_counts),
                'avg_decisions': sum(decision_counts) / len(decision_counts) if decision_counts else 0,
                'issues': file_issues
            })

            # Print file status
            status = "✅" if not file_issues else "⚠️"
            print(f"\n{status} {result_file.name}")
            print(f"   Model: {model}, Constraint: ${bet_constraint}, Type: {bet_type}")
            print(f"   Games: {n_games}, Prompts: {dict(prompt_counts)}")
            if file_issues:
                for issue in file_issues:
                    print(f"   ⚠️  {issue}")
                issues.extend([(result_file.name, issue) for issue in file_issues])

        except Exception as e:
            print(f"\n❌ {result_file.name}")
            print(f"   ERROR: {str(e)}")
            issues.append((result_file.name, f"File read error: {str(e)}"))

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total files: {len(files_found)} (expected {expected_files})")
    print(f"  Total games: {total_games} (expected {expected_total_games})")
    print(f"  Average games per file: {total_games / len(files_found) if files_found else 0:.1f}")

    # Check for missing files
    print(f"\n📋 Configuration Coverage:")
    found_configs = set()
    for stat in file_stats:
        config_key = (stat['model'], stat['constraint'], stat['bet_type'])
        found_configs.add(config_key)

    missing_configs = []
    for model in models:
        for constraint in constraints:
            for bet_type in bet_types:
                config_key = (model, constraint, bet_type)
                if config_key not in found_configs:
                    missing_configs.append(config_key)

    if missing_configs:
        print(f"  ⚠️  Missing configurations: {len(missing_configs)}")
        for config in missing_configs:
            print(f"     - {config}")
    else:
        print(f"  ✅ All {expected_files} configurations present")

    # Issue summary
    print(f"\n🔍 Issues Found: {len(issues)}")
    if issues:
        issue_types = Counter([issue[1].split(':')[0] for issue in issues])
        print(f"\n  Issue breakdown:")
        for issue_type, count in issue_types.most_common():
            print(f"    - {issue_type}: {count}")

        print(f"\n  Detailed issues:")
        for file, issue in issues[:10]:  # Show first 10
            print(f"    {file}: {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print(f"  ✅ No issues found!")

    # Final verdict
    print("\n" + "="*80)
    if len(issues) == 0 and len(missing_configs) == 0:
        print("✅ VALIDATION PASSED - All checks successful!")
    elif len(issues) > 0 or len(missing_configs) > 0:
        print("⚠️ VALIDATION WARNINGS - Some issues found (see above)")
    print("="*80)

    return file_stats, issues

if __name__ == '__main__':
    validate_experiment()
