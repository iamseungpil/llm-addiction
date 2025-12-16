"""
Investment Choice Experiment - Case Study Example
사용 예시: python3 case_study_example.py
"""

import json

# 데이터 파일 경로
DATA_DIR = "/data/llm_addiction/investment_choice_experiment/results/"

FILES = {
    'gpt4o_mini_fixed': f"{DATA_DIR}gpt4o_mini_fixed_20251119_042406.json",
    'gpt4o_mini_variable': f"{DATA_DIR}gpt4o_mini_variable_20251119_035805.json",
    'gpt41_mini_fixed': f"{DATA_DIR}gpt41_mini_fixed_20251119_032133.json",
    'gpt41_mini_variable': f"{DATA_DIR}gpt41_mini_variable_20251119_022306.json",
    'claude_haiku_fixed': f"{DATA_DIR}claude_haiku_fixed_20251119_044100.json",
    'claude_haiku_variable': f"{DATA_DIR}claude_haiku_variable_20251119_035809.json",
    'gemini_flash_fixed': f"{DATA_DIR}gemini_flash_fixed_20251119_110752.json",
    'gemini_flash_variable': f"{DATA_DIR}gemini_flash_variable_20251119_043257.json"
}

def load_data(model_bet):
    """Load experiment data"""
    with open(FILES[model_bet]) as f:
        return json.load(f)

def find_games_by_condition(data, condition):
    """Find all games for a specific condition"""
    return [g for g in data['results'] if g['prompt_condition'] == condition]

def analyze_single_game(game):
    """Analyze a single game in detail"""
    print("="*80)
    print(f"Game {game['game_id']} Analysis")
    print("="*80)
    print(f"Model: {game['model']}")
    print(f"Bet Type: {game['bet_type']}")
    print(f"Condition: {game['prompt_condition']}")
    print(f"Rounds: {game['rounds_played']}")
    print(f"Final Balance: ${game['final_balance']}")
    print(f"Exit: {game['exit_reason']}")

    print("\nDecision History:")
    for decision in game['decisions']:
        print(f"\n  Round {decision['round']}:")
        print(f"    Balance: ${decision['balance_before']} → ${decision['balance_after']}")
        print(f"    Choice: Option {decision['choice']}")
        print(f"    Outcome: {decision['outcome']}")
        print(f"    Response (first 150 chars):")
        print(f"      {decision['response'][:150]}...")

def compare_conditions(data, conditions=['BASE', 'G', 'M', 'GM']):
    """Compare behavior across conditions"""
    print("="*80)
    print("Condition Comparison")
    print("="*80)

    for condition in conditions:
        games = find_games_by_condition(data, condition)

        # Calculate statistics
        avg_rounds = sum(g['rounds_played'] for g in games) / len(games)
        avg_balance = sum(g['final_balance'] for g in games) / len(games)

        # Choice distribution
        all_choices = []
        for game in games:
            for dec in game['decisions']:
                all_choices.append(dec['choice'])

        choice_dist = {i: all_choices.count(i) / len(all_choices) * 100 for i in range(1, 5)}

        print(f"\n{condition}:")
        print(f"  Games: {len(games)}")
        print(f"  Avg Rounds: {avg_rounds:.2f}")
        print(f"  Avg Final Balance: ${avg_balance:.2f}")
        print(f"  Choice Distribution:")
        for i in range(1, 5):
            print(f"    Option {i}: {choice_dist[i]:.1f}%")

def find_extreme_cases(data):
    """Find interesting extreme cases"""
    print("="*80)
    print("Extreme Cases")
    print("="*80)

    all_games = data['results']

    # Most rounds survived
    max_rounds = max(all_games, key=lambda g: g['rounds_played'])
    print(f"\n🏆 Most Rounds: Game {max_rounds['game_id']}")
    print(f"   Condition: {max_rounds['prompt_condition']}")
    print(f"   Rounds: {max_rounds['rounds_played']}")
    print(f"   Final: ${max_rounds['final_balance']}")

    # Immediate bankruptcy
    immediate_bankrupt = [g for g in all_games if g['rounds_played'] == 1 and g['final_balance'] == 0]
    if immediate_bankrupt:
        print(f"\n💥 Immediate Bankruptcies: {len(immediate_bankrupt)}")
        sample = immediate_bankrupt[0]
        print(f"   Example: Game {sample['game_id']} ({sample['prompt_condition']})")
        print(f"   Choice: Option {sample['decisions'][0]['choice']}")

    # Highest final balance
    richest = max(all_games, key=lambda g: g['final_balance'])
    print(f"\n💰 Highest Balance: Game {richest['game_id']}")
    print(f"   Condition: {richest['prompt_condition']}")
    print(f"   Balance: ${richest['final_balance']}")
    print(f"   Rounds: {richest['rounds_played']}")

# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Investment Choice Experiment - Case Study")
    print("="*80)

    # Example 1: Analyze Gemini Variable
    print("\n\n### Example 1: Gemini Variable M Condition ###")
    gemini_var = load_data('gemini_flash_variable')
    m_games = find_games_by_condition(gemini_var, 'M')
    print(f"Found {len(m_games)} games in M condition")

    # Analyze first game
    analyze_single_game(m_games[0])

    # Example 2: Compare conditions for GPT-4o-mini Fixed
    print("\n\n### Example 2: GPT-4o-mini Fixed Condition Comparison ###")
    gpt4o_fixed = load_data('gpt4o_mini_fixed')
    compare_conditions(gpt4o_fixed)

    # Example 3: Find extreme cases in Claude
    print("\n\n### Example 3: Claude Variable Extreme Cases ###")
    claude_var = load_data('claude_haiku_variable')
    find_extreme_cases(claude_var)
