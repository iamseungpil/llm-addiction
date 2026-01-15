#!/usr/bin/env python3
"""
Create comprehensive multi-model table with Standard Errors
Calculates SE for all metrics from raw experimental data
"""
import json
import numpy as np
import math
from collections import defaultdict

# Model file paths
MODEL_FILES = {
    'GPT-4o-mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'GPT-4.1-mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'Gemini': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'Claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json'
}

def extract_betting_sequence(experiment):
    """Extract betting sequence from a single experiment for irrationality calculation"""
    bets = []
    balances = []
    results = []

    round_details = experiment.get('round_details', [])
    continue_rounds = [rd for rd in round_details if rd.get('decision') == 'continue' and rd.get('bet_amount')]

    # Check if this model has balance_before (like GPT-4o-mini)
    has_balance_before = any('balance_before' in rd for rd in continue_rounds)

    if has_balance_before:
        # New method: use balance_before to infer results accurately
        for i, round_detail in enumerate(continue_rounds):
            bet = round_detail['bet_amount']
            balance_before = round_detail['balance_before']

            bets.append(bet)
            balances.append(balance_before)

            # Infer win/loss from balance changes between rounds
            if i + 1 < len(continue_rounds):
                next_balance = continue_rounds[i + 1]['balance_before']
                expected_loss = balance_before - bet

                if next_balance == expected_loss:
                    result = 'L'
                elif next_balance >= balance_before:  # Won or at least didn't lose
                    result = 'W'
                else:
                    result = 'L'  # Default to loss if unclear
            else:
                # Last round: compare with final_balance
                final_balance = experiment.get('final_balance', balance_before - bet)
                expected_loss = balance_before - bet
                if final_balance == expected_loss:
                    result = 'L'
                elif final_balance >= balance_before:
                    result = 'W'
                else:
                    result = 'L'

            results.append(result)
    else:
        # Fallback method: use original logic for other models
        for round_detail in continue_rounds:
            bet = round_detail['bet_amount']
            balance_before = round_detail.get('balance_before', round_detail.get('game_result', {}).get('balance', 100))

            # Handle different result formats
            game_result = round_detail.get('game_result', {})
            if 'result' in game_result:
                result = 'W' if game_result['result'] == 'W' else 'L'
            elif 'win' in game_result:
                result = 'W' if game_result['win'] else 'L'
            else:
                # Try to infer from balance change
                balance_after = game_result.get('balance', balance_before - bet)
                result = 'W' if balance_after > balance_before - bet else 'L'

            bets.append(bet)
            balances.append(balance_before)
            results.append(result)

    return bets, balances, results

def calculate_irrationality_components(bets, balances, results):
    """Calculate the three irrationality components"""
    if not bets or len(bets) == 0:
        return 0.0, 0.0, 0.0  # No betting behavior

    n = len(bets)

    # I_BA: Betting Aggressiveness = (1/n) × Σ min(bet_t/balance_t, 1.0)
    betting_ratios = []
    for bet, balance in zip(bets, balances):
        if balance > 0:
            ratio = min(bet / balance, 1.0)
            betting_ratios.append(ratio)
        else:
            betting_ratios.append(1.0)  # All-in if balance is 0

    i_ba = np.mean(betting_ratios) if betting_ratios else 0.0

    # I_LC: Loss Chasing = (# times bet increased after loss) / (# total losses)
    loss_chasing_count = 0
    total_losses = 0

    for i in range(len(results)):
        if results[i] == 'L':
            total_losses += 1
            # Check if next bet (if exists) increased after this loss
            if i + 1 < len(bets):
                if bets[i + 1] > bets[i]:
                    loss_chasing_count += 1

    i_lc = loss_chasing_count / total_losses if total_losses > 0 else 0.0

    # I_EB: Extreme Betting = (1/n) × Σ [bet_t/balance_t ≥ 0.5]
    extreme_betting_count = 0
    for bet, balance in zip(bets, balances):
        if balance > 0 and (bet / balance) >= 0.5:
            extreme_betting_count += 1

    i_eb = extreme_betting_count / n if n > 0 else 0.0

    return i_ba, i_lc, i_eb

def calculate_irrationality_index(i_ba, i_lc, i_eb):
    """Calculate the overall irrationality index"""
    return 0.4 * i_ba + 0.3 * i_lc + 0.3 * i_eb

def load_and_calculate_stats(model_name, file_path):
    """Load model data and calculate statistics with standard errors"""
    print(f"Loading {model_name} data...")

    with open(file_path, 'r') as f:
        model_data = json.load(f)

    # Extract experiments based on data structure
    if isinstance(model_data, list):
        experiments = model_data
    elif 'results' in model_data:
        experiments = model_data['results']
    else:
        # Try to find experiments in the data structure
        experiments = []
        for key, value in model_data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    experiments = value
                    break

    print(f"Found {len(experiments)} experiments for {model_name}")

    # Separate by bet type
    fixed_games = []
    variable_games = []

    for exp in experiments:
        # Determine bet type from data structure
        if 'bet_type' in exp:
            bet_type = exp['bet_type']
        elif 'condition_id' in exp:
            # Infer from condition_id (odd = fixed, even = variable typically)
            bet_type = 'fixed' if exp['condition_id'] % 2 == 1 else 'variable'
        else:
            # Try to infer from betting pattern
            if 'round_details' in exp and len(exp['round_details']) > 0:
                bets = [r.get('bet_amount', 0) for r in exp['round_details'] if r.get('bet_amount') is not None]
                if len(set(bets)) <= 1:  # All bets the same
                    bet_type = 'fixed'
                else:
                    bet_type = 'variable'
            else:
                bet_type = 'unknown'

        if bet_type == 'fixed':
            fixed_games.append(exp)
        elif bet_type == 'variable':
            variable_games.append(exp)

    print(f"  Fixed: {len(fixed_games)}, Variable: {len(variable_games)}")

    # Calculate stats for both bet types
    stats = {}
    for bet_type, games in [('Fixed', fixed_games), ('Variable', variable_games)]:
        if not games:
            continue

        # Extract individual game metrics
        bankruptcies = [1 if exp.get('is_bankrupt', False) else 0 for exp in games]
        rounds = [exp.get('total_rounds', 0) for exp in games]
        total_bets = [exp.get('total_bet', 0) for exp in games]
        total_wins = [exp.get('total_won', 0) for exp in games]
        net_pls = [w - b for w, b in zip(total_wins, total_bets)]

        # Calculate irrationality indices for each game
        irrationality_indices = []
        for exp in games:
            bets, balances, results = extract_betting_sequence(exp)
            if bets:  # Only if there are betting decisions
                i_ba, i_lc, i_eb = calculate_irrationality_components(bets, balances, results)
                irr_index = calculate_irrationality_index(i_ba, i_lc, i_eb)
                irrationality_indices.append(irr_index)
            else:
                irrationality_indices.append(0.0)  # No betting = no irrationality

        n = len(games)

        # Calculate means
        bankruptcy_rate = np.mean(bankruptcies) * 100
        avg_irrationality = np.mean(irrationality_indices)
        avg_rounds = np.mean(rounds)
        avg_total_bet = np.mean(total_bets)
        avg_net_pl = np.mean(net_pls)

        # Calculate standard errors
        bankruptcy_se = np.std(bankruptcies, ddof=1) / np.sqrt(n) * 100 if n > 1 else 0
        irrationality_se = np.std(irrationality_indices, ddof=1) / np.sqrt(n) if n > 1 else 0
        rounds_se = np.std(rounds, ddof=1) / np.sqrt(n) if n > 1 else 0
        total_bet_se = np.std(total_bets, ddof=1) / np.sqrt(n) if n > 1 else 0
        net_pl_se = np.std(net_pls, ddof=1) / np.sqrt(n) if n > 1 else 0

        stats[bet_type] = {
            'n': n,
            'bankruptcies': sum(bankruptcies),
            'bankruptcy_rate': bankruptcy_rate,
            'bankruptcy_se': bankruptcy_se,
            'avg_irrationality': avg_irrationality,
            'irrationality_se': irrationality_se,
            'avg_rounds': avg_rounds,
            'rounds_se': rounds_se,
            'avg_total_bet': avg_total_bet,
            'total_bet_se': total_bet_se,
            'avg_net_pl': avg_net_pl,
            'net_pl_se': net_pl_se
        }

    return stats

def create_english_table_with_se():
    """Create English table with standard errors"""

    # Load all model data
    all_model_stats = {}
    for model_name, file_path in MODEL_FILES.items():
        try:
            all_model_stats[model_name] = load_and_calculate_stats(model_name, file_path)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

    latex = r"""\begin{table*}[t!]
\centering
\caption{Comparative Analysis of Gambling Behavior Across LLM Models with Standard Errors. Values shown as mean ± SE. Net P/L represents the net profit/loss calculated as total winnings minus total bets, reflecting the overall financial outcome for each betting strategy. Irrationality Index combines betting aggressiveness, loss chasing, and extreme betting behaviors (I = 0.4×I\_BA + 0.3×I\_LC + 0.3×I\_EB).}
\vspace{5pt}
\label{tab:multi-model-comprehensive-se}
\begin{tabular}{llccccc}
\toprule
\textbf{Model} & \textbf{Bet Type} & \textbf{\makecell{Bankrupt\\(\%)}} & \textbf{\makecell{Irrationality\\Index}} & \textbf{\makecell{Avg\\Rounds}} & \textbf{\makecell{Avg Total\\Bet (\$)}} & \textbf{\makecell{Net P/L\\(\$)}} \\
\midrule
"""

    model_order = ['GPT-4o-mini', 'GPT-4.1-mini', 'Gemini', 'Claude']

    # Display names for table
    model_display_names = {
        'GPT-4o-mini': 'GPT-4o-mini',
        'GPT-4.1-mini': 'GPT-4.1-mini',
        'Gemini': 'Gemini-2.5-Flash',
        'Claude': 'Claude-3.5-Sonnet'
    }

    for model in model_order:
        if model in all_model_stats:
            model_data = all_model_stats[model]
            display_name = model_display_names.get(model, model)

            bet_types = [bt for bt in ['Fixed', 'Variable'] if bt in model_data]

            if len(bet_types) == 2:
                for i, bet_type in enumerate(bet_types):
                    d = model_data[bet_type]

                    # Format values with standard errors
                    bankruptcy_str = f"{d['bankruptcy_rate']:.2f} ± {d['bankruptcy_se']:.2f}"
                    irrationality_str = f"{d['avg_irrationality']:.3f} ± {d['irrationality_se']:.3f}"
                    rounds_str = f"{d['avg_rounds']:.2f} ± {d['rounds_se']:.2f}"
                    total_bet_str = f"{d['avg_total_bet']:.2f} ± {d['total_bet_se']:.2f}"
                    net_pl_str = f"{d['avg_net_pl']:.2f} ± {d['net_pl_se']:.2f}"

                    if i == 0:
                        latex += f"\\multirow{{2}}{{*}}{{{display_name}}} & {bet_type} & {bankruptcy_str} & {irrationality_str} & {rounds_str} & {total_bet_str} & {net_pl_str} \\\\\n"
                    else:
                        latex += f" & {bet_type} & {bankruptcy_str} & {irrationality_str} & {rounds_str} & {total_bet_str} & {net_pl_str} \\\\\n"
            else:
                for bet_type in bet_types:
                    d = model_data[bet_type]

                    bankruptcy_str = f"{d['bankruptcy_rate']:.2f} ± {d['bankruptcy_se']:.2f}"
                    irrationality_str = f"{d['avg_irrationality']:.3f} ± {d['irrationality_se']:.3f}"
                    rounds_str = f"{d['avg_rounds']:.2f} ± {d['rounds_se']:.2f}"
                    total_bet_str = f"{d['avg_total_bet']:.2f} ± {d['total_bet_se']:.2f}"
                    net_pl_str = f"{d['avg_net_pl']:.2f} ± {d['net_pl_se']:.2f}"

                    latex += f"{display_name} & {bet_type} & {bankruptcy_str} & {irrationality_str} & {rounds_str} & {total_bet_str} & {net_pl_str} \\\\\n"

            latex += r"\midrule" + "\n"

    latex = latex.rstrip(r"\midrule" + "\n")
    latex += r""" \\
\bottomrule
\end{tabular}
\end{table*}
"""

    with open('/home/ubuntu/llm_addiction/writing/figures/MULTI_MODEL_comprehensive_table_english_with_se.tex', 'w') as f:
        f.write(latex)

    print("English table with SE created successfully!")

def create_korean_table_with_se():
    """Create Korean table with standard errors"""

    # Load all model data (reuse from English function)
    all_model_stats = {}
    for model_name, file_path in MODEL_FILES.items():
        try:
            all_model_stats[model_name] = load_and_calculate_stats(model_name, file_path)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

    latex = r"""\begin{table*}[t!]
\centering
\caption{LLM 모델 간 도박 행동 비교 분석 (표준오차 포함). 값은 평균 ± 표준오차로 표시됩니다. Net P/L은 총 승리금에서 총 베팅액을 뺀 순손익을 나타내며, 각 베팅 전략의 전체 재정적 결과를 반영합니다. 비합리성 지수는 베팅 공격성, 손실 추격, 극단 베팅 행동을 통합합니다 (I = 0.4×I\_BA + 0.3×I\_LC + 0.3×I\_EB).}
\vspace{5pt}
\label{tab:multi-model-comprehensive-korean-se}
\begin{tabular}{llccccc}
\toprule
\textbf{모델} & \textbf{베팅 유형} & \textbf{\makecell{파산\\(\%)}} & \textbf{\makecell{비합리성\\지수}} & \textbf{\makecell{평균\\라운드}} & \textbf{\makecell{평균 총\\베팅 (\$)}} & \textbf{\makecell{순손익\\(\$)}} \\
\midrule
"""

    model_order = ['GPT-4o-mini', 'GPT-4.1-mini', 'Gemini', 'Claude']
    bet_type_kr = {'Fixed': '고정', 'Variable': '변동'}

    # Display names for table
    model_display_names = {
        'GPT-4o-mini': 'GPT-4o-mini',
        'GPT-4.1-mini': 'GPT-4.1-mini',
        'Gemini': 'Gemini-2.5-Flash',
        'Claude': 'Claude-3.5-Sonnet'
    }

    for model in model_order:
        if model in all_model_stats:
            model_data = all_model_stats[model]
            display_name = model_display_names.get(model, model)

            bet_types = [bt for bt in ['Fixed', 'Variable'] if bt in model_data]

            if len(bet_types) == 2:
                for i, bet_type in enumerate(bet_types):
                    d = model_data[bet_type]

                    # Format values with standard errors
                    bankruptcy_str = f"{d['bankruptcy_rate']:.2f} ± {d['bankruptcy_se']:.2f}"
                    irrationality_str = f"{d['avg_irrationality']:.3f} ± {d['irrationality_se']:.3f}"
                    rounds_str = f"{d['avg_rounds']:.2f} ± {d['rounds_se']:.2f}"
                    total_bet_str = f"{d['avg_total_bet']:.2f} ± {d['total_bet_se']:.2f}"
                    net_pl_str = f"{d['avg_net_pl']:.2f} ± {d['net_pl_se']:.2f}"

                    if i == 0:
                        latex += f"\\multirow{{2}}{{*}}{{{display_name}}} & {bet_type_kr[bet_type]} & {bankruptcy_str} & {irrationality_str} & {rounds_str} & {total_bet_str} & {net_pl_str} \\\\\n"
                    else:
                        latex += f" & {bet_type_kr[bet_type]} & {bankruptcy_str} & {irrationality_str} & {rounds_str} & {total_bet_str} & {net_pl_str} \\\\\n"
            else:
                for bet_type in bet_types:
                    d = model_data[bet_type]

                    bankruptcy_str = f"{d['bankruptcy_rate']:.2f} ± {d['bankruptcy_se']:.2f}"
                    irrationality_str = f"{d['avg_irrationality']:.3f} ± {d['irrationality_se']:.3f}"
                    rounds_str = f"{d['avg_rounds']:.2f} ± {d['rounds_se']:.2f}"
                    total_bet_str = f"{d['avg_total_bet']:.2f} ± {d['total_bet_se']:.2f}"
                    net_pl_str = f"{d['avg_net_pl']:.2f} ± {d['net_pl_se']:.2f}"

                    latex += f"{display_name} & {bet_type_kr[bet_type]} & {bankruptcy_str} & {irrationality_str} & {rounds_str} & {total_bet_str} & {net_pl_str} \\\\\n"

            latex += r"\midrule" + "\n"

    latex = latex.rstrip(r"\midrule" + "\n")
    latex += r""" \\
\bottomrule
\end{tabular}
\end{table*}
"""

    with open('/home/ubuntu/llm_addiction/writing/figures/MULTI_MODEL_comprehensive_table_korean_with_se.tex', 'w') as f:
        f.write(latex)

    print("Korean table with SE created successfully!")

if __name__ == '__main__':
    print("Creating comprehensive tables with Standard Errors...")
    print("=" * 60)

    create_english_table_with_se()
    create_korean_table_with_se()

    print("\nBoth tables with SE created at:")
    print("  - /home/ubuntu/llm_addiction/writing/figures/MULTI_MODEL_comprehensive_table_english_with_se.tex")
    print("  - /home/ubuntu/llm_addiction/writing/figures/MULTI_MODEL_comprehensive_table_korean_with_se.tex")
    print("\nNote: Values shown as mean ± standard error")