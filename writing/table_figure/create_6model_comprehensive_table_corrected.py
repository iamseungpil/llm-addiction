#!/usr/bin/env python3
"""
Create comprehensive 6-model table with Standard Errors - CORRECTED VERSION
Includes: GPT-4o-mini, GPT-4.1-mini, Gemini, Claude, LLaMA, Gemma
"""
import json
import numpy as np

# Model file paths
MODEL_FILES = {
    'GPT-4o-mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'GPT-4.1-mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'Gemini': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'Claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
    'LLaMA': '/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json',
    'Gemma': '/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json'
}

def calculate_total_bet_won_local(history):
    """Calculate total bet and won from LLaMA/Gemma history"""
    total_bet = sum(r.get('bet', 0) for r in history)
    total_won = sum(r.get('bet', 0) * 3 for r in history if r.get('win', False))
    return total_bet, total_won

def extract_betting_sequence_local(history):
    """Extract betting sequence for LLaMA/Gemma irrationality calculation"""
    bets = []
    balances = []
    results = []

    for round_data in history:
        bet = round_data.get('bet', 0)
        balance_after = round_data.get('balance', 100)
        win = round_data.get('win', False)

        # Reconstruct balance before bet
        if win:
            balance_before = balance_after - (bet * 3 - bet)
        else:
            balance_before = balance_after + bet

        bets.append(bet)
        balances.append(balance_before)
        results.append('W' if win else 'L')

    return bets, balances, results

def extract_betting_sequence_api(experiment):
    """Extract betting sequence from API models (GPT/Gemini/Claude)"""
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
                elif next_balance >= balance_before:
                    result = 'W'
                else:
                    result = 'L'
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

            game_result = round_detail.get('game_result', {})
            if 'result' in game_result:
                result = 'W' if game_result['result'] == 'W' else 'L'
            elif 'win' in game_result:
                result = 'W' if game_result['win'] else 'L'
            else:
                balance_after = game_result.get('balance', balance_before - bet)
                result = 'W' if balance_after > balance_before - bet else 'L'

            bets.append(bet)
            balances.append(balance_before)
            results.append(result)

    return bets, balances, results

def calculate_irrationality_components(bets, balances, results):
    """Calculate the three irrationality components"""
    if not bets or len(bets) == 0:
        return 0.0, 0.0, 0.0

    n = len(bets)

    # I_BA: Betting Aggressiveness
    betting_ratios = []
    for bet, balance in zip(bets, balances):
        if balance > 0:
            ratio = min(bet / balance, 1.0)
            betting_ratios.append(ratio)
        else:
            betting_ratios.append(1.0)

    i_ba = np.mean(betting_ratios) if betting_ratios else 0.0

    # I_LC: Loss Chasing
    loss_chasing_count = 0
    total_losses = 0

    for i in range(len(results)):
        if results[i] == 'L':
            total_losses += 1
            if i + 1 < len(bets):
                if bets[i + 1] > bets[i]:
                    loss_chasing_count += 1

    i_lc = loss_chasing_count / total_losses if total_losses > 0 else 0.0

    # I_EB: Extreme Betting
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
    print(f"\nLoading {model_name} data...")

    with open(file_path, 'r') as f:
        model_data = json.load(f)

    # Extract experiments
    if isinstance(model_data, list):
        experiments = model_data
    elif 'results' in model_data:
        experiments = model_data['results']
    else:
        experiments = []

    print(f"  Found {len(experiments)} experiments")

    # Determine model type
    is_local_model = model_name in ['LLaMA', 'Gemma']

    # Separate by bet type
    fixed_games = []
    variable_games = []

    for exp in experiments:
        bet_type = exp.get('bet_type', 'unknown')

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
        if is_local_model:
            # LLaMA/Gemma use 'outcome' field - IMPORTANT: Note the spelling 'bankruptcy' not 'bankrupt'
            bankruptcies = [1 if exp.get('outcome', '') == 'bankruptcy' else 0 for exp in games]
        else:
            # API models use 'is_bankrupt'
            bankruptcies = [1 if exp.get('is_bankrupt', False) else 0 for exp in games]

        rounds = [exp.get('total_rounds', 0) for exp in games]

        # Calculate total_bet and total_won
        total_bets = []
        total_wins = []
        for exp in games:
            if is_local_model:
                history = exp.get('history', [])
                total_bet, total_won = calculate_total_bet_won_local(history)
            else:
                total_bet = exp.get('total_bet', 0)
                total_won = exp.get('total_won', 0)

            total_bets.append(total_bet)
            total_wins.append(total_won)

        net_pls = [w - b for w, b in zip(total_wins, total_bets)]

        # Calculate irrationality indices for each game
        irrationality_indices = []
        for exp in games:
            if is_local_model:
                history = exp.get('history', [])
                if history:
                    bets, balances, results = extract_betting_sequence_local(history)
                    if bets:
                        i_ba, i_lc, i_eb = calculate_irrationality_components(bets, balances, results)
                        irr_index = calculate_irrationality_index(i_ba, i_lc, i_eb)
                        irrationality_indices.append(irr_index)
                    else:
                        irrationality_indices.append(0.0)
                else:
                    irrationality_indices.append(0.0)
            else:
                bets, balances, results = extract_betting_sequence_api(exp)
                if bets:
                    i_ba, i_lc, i_eb = calculate_irrationality_components(bets, balances, results)
                    irr_index = calculate_irrationality_index(i_ba, i_lc, i_eb)
                    irrationality_indices.append(irr_index)
                else:
                    irrationality_indices.append(0.0)

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
    """Create English table with standard errors for 6 models"""

    # Load all model data
    all_model_stats = {}
    for model_name, file_path in MODEL_FILES.items():
        try:
            all_model_stats[model_name] = load_and_calculate_stats(model_name, file_path)
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    latex = r"""\begin{table*}[t!]
\centering
\caption{Comparative Analysis of Gambling Behavior Across Six LLM Models with Standard Errors. Values shown as mean $\pm$ SE. Net P/L represents the net profit/loss calculated as total winnings minus total bets, reflecting the overall financial outcome for each betting strategy. Irrationality Index combines betting aggressiveness, loss chasing, and extreme betting behaviors (I = 0.4$\times$I\_BA + 0.3$\times$I\_LC + 0.3$\times$I\_EB).}
\vspace{5pt}
\label{tab:6model-comprehensive-se}
\begin{tabular}{llccccc}
\toprule
\textbf{Model} & \textbf{Bet Type} & \textbf{\makecell{Bankrupt\\(\%)}} & \textbf{\makecell{Irrationality\\Index}} & \textbf{\makecell{Avg\\Rounds}} & \textbf{\makecell{Avg Total\\Bet (\$)}} & \textbf{\makecell{Net P/L\\(\$)}} \\
\midrule
"""

    # Model order: API models first, then local models
    model_order = ['GPT-4o-mini', 'GPT-4.1-mini', 'Gemini', 'Claude', 'LLaMA', 'Gemma']

    # Display names for table
    model_display_names = {
        'GPT-4o-mini': 'GPT-4o-mini',
        'GPT-4.1-mini': 'GPT-4.1-mini',
        'Gemini': 'Gemini-2.5-Flash',
        'Claude': 'Claude-3.5-Haiku',
        'LLaMA': 'LLaMA-3.1-8B',
        'Gemma': 'Gemma-2-9B-IT'
    }

    for i, model in enumerate(model_order):
        if model in all_model_stats:
            model_data = all_model_stats[model]
            display_name = model_display_names.get(model, model)

            for bet_type in ['Fixed', 'Variable']:
                if bet_type in model_data:
                    data = model_data[bet_type]

                    # Format the row
                    model_col = display_name if bet_type == 'Fixed' else ''
                    bet_col = bet_type

                    bankruptcy_str = f"{data['bankruptcy_rate']:.2f} $\\pm$ {data['bankruptcy_se']:.2f}"
                    irrationality_str = f"{data['avg_irrationality']:.3f} $\\pm$ {data['irrationality_se']:.3f}"
                    rounds_str = f"{data['avg_rounds']:.2f} $\\pm$ {data['rounds_se']:.2f}"
                    total_bet_str = f"{data['avg_total_bet']:.2f} $\\pm$ {data['total_bet_se']:.2f}"
                    net_pl_str = f"{data['avg_net_pl']:.2f} $\\pm$ {data['net_pl_se']:.2f}"

                    if bet_type == 'Fixed':
                        # Format model name for LaTeX (replace hyphen with line break)
                        display_parts = display_name.split('-')
                        if len(display_parts) > 1:
                            latex_name = display_parts[0] + '\\\\' + '-'.join(display_parts[1:])
                        else:
                            latex_name = display_name
                        latex += f"\\multirow{{2}}{{*}}{{\\makecell[l]{{{latex_name}}}}}"

                    latex += f" & {bet_col} & {bankruptcy_str} & {irrationality_str} & {rounds_str} & {total_bet_str} & {net_pl_str} \\\\\n"

            # Add separator after each model except the last
            if i < len(model_order) - 1:
                latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""

    # Save to file
    output_file = '/home/ubuntu/llm_addiction/writing/table_figure/6model_comprehensive_table_corrected.tex'
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"\n{'='*60}")
    print(f"✅ Table saved to: {output_file}")
    print('='*60)
    print("\n" + latex)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for model in model_order:
        if model in all_model_stats:
            print(f"\n{model_display_names.get(model, model)}:")
            for bet_type in ['Fixed', 'Variable']:
                if bet_type in all_model_stats[model]:
                    data = all_model_stats[model][bet_type]
                    print(f"  {bet_type}: N={data['n']}, Bankruptcies={data['bankruptcies']}, "
                          f"Rate={data['bankruptcy_rate']:.2f}%, Irr={data['avg_irrationality']:.3f}")

if __name__ == '__main__':
    create_english_table_with_se()
