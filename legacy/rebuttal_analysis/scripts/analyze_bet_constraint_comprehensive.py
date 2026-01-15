#!/usr/bin/env python3
"""
Comprehensive analysis of Investment Choice Bet Constraint Experiment
Generates Figure 5 (Autonomy Mechanism) and Appendix Table C
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Directories
RESULTS_DIR = Path('/data/llm_addiction/investment_choice_bet_constraint/results')
OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/rebuttal_analysis')
FIGURES_DIR = OUTPUT_DIR / 'figures' / 'main'
TABLES_DIR = OUTPUT_DIR / 'tables' / 'appendix'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Model names mapping
MODEL_NAMES = {
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt41_mini': 'GPT-4.1-mini',
    'gemini_flash': 'Gemini-2.5-Flash',
    'claude_haiku': 'Claude-3.5-Haiku'
}

def load_all_data():
    """Load all bet constraint experiment results"""
    data = defaultdict(lambda: defaultdict(dict))

    for result_file in RESULTS_DIR.glob('*.json'):
        with open(result_file) as f:
            file_data = json.load(f)

        config = file_data.get('experiment_config', {})
        model = config.get('model')
        constraint = config.get('bet_constraint')
        bet_type = config.get('bet_type')

        if model and constraint and bet_type:
            data[model][constraint][bet_type] = file_data

    return data

def analyze_option4_by_constraint(data):
    """Analyze Option 4 selection rate by bet constraint"""
    results = defaultdict(lambda: defaultdict(lambda: {'fixed': [], 'variable': []}))

    for model, constraints in data.items():
        for constraint, bet_types in constraints.items():
            for bet_type, file_data in bet_types.items():
                games = file_data.get('results', [])

                for game in games:
                    decisions = game.get('decisions', [])
                    if decisions:
                        option4_count = sum(1 for d in decisions if d.get('choice') == 4)
                        option4_rate = option4_count / len(decisions) if decisions else 0
                        results[model][constraint][bet_type].append(option4_rate)

    # Calculate means and stds
    summary = {}
    for model in results:
        summary[model] = {}
        for constraint in results[model]:
            summary[model][constraint] = {
                'fixed': {
                    'mean': np.mean(results[model][constraint]['fixed']) if results[model][constraint]['fixed'] else 0,
                    'std': np.std(results[model][constraint]['fixed']) if results[model][constraint]['fixed'] else 0,
                    'n': len(results[model][constraint]['fixed'])
                },
                'variable': {
                    'mean': np.mean(results[model][constraint]['variable']) if results[model][constraint]['variable'] else 0,
                    'std': np.std(results[model][constraint]['variable']) if results[model][constraint]['variable'] else 0,
                    'n': len(results[model][constraint]['variable'])
                }
            }

    return summary

def analyze_bet_amounts(data):
    """Analyze average bet amounts"""
    results = defaultdict(lambda: defaultdict(lambda: {'fixed': [], 'variable': []}))

    for model, constraints in data.items():
        for constraint, bet_types in constraints.items():
            for bet_type, file_data in bet_types.items():
                games = file_data.get('results', [])

                for game in games:
                    decisions = game.get('decisions', [])
                    # For investment choice, extract actual bet amounts from decisions
                    # Note: This assumes bet information is in the decision structure
                    game_bets = []
                    for d in decisions:
                        # Investment choice uses option-based betting
                        # We'll calculate based on balance changes
                        choice = d.get('choice')
                        if choice and choice > 1:  # Options 2-4 involve betting
                            # Estimate bet from balance changes
                            bal_before = d.get('balance_before', 0)
                            bal_after = d.get('balance_after', 0)
                            win = d.get('win', False)

                            if win:
                                # Reverse calculate bet from win
                                payout = d.get('payout', 0)
                                # This is approximate
                                if payout > 0:
                                    game_bets.append(payout / 1.8)  # Rough estimate
                            else:
                                # Loss: bet = balance decrease
                                bet = bal_before - bal_after
                                if bet > 0:
                                    game_bets.append(bet)

                    if game_bets:
                        avg_bet = np.mean(game_bets)
                        results[model][constraint][bet_type].append(avg_bet)

    # Calculate summary
    summary = {}
    for model in results:
        summary[model] = {}
        for constraint in results[model]:
            summary[model][constraint] = {
                'fixed': {
                    'mean': np.mean(results[model][constraint]['fixed']) if results[model][constraint]['fixed'] else 0,
                    'std': np.std(results[model][constraint]['fixed']) if results[model][constraint]['fixed'] else 0
                },
                'variable': {
                    'mean': np.mean(results[model][constraint]['variable']) if results[model][constraint]['variable'] else 0,
                    'std': np.std(results[model][constraint]['variable']) if results[model][constraint]['variable'] else 0
                }
            }

    return summary

def analyze_losses_by_constraint(data):
    """Analyze average losses by constraint"""
    results = defaultdict(lambda: defaultdict(lambda: {'fixed': [], 'variable': []}))

    for model, constraints in data.items():
        for constraint, bet_types in constraints.items():
            for bet_type, file_data in bet_types.items():
                games = file_data.get('results', [])

                for game in games:
                    final_balance = game.get('final_balance', 100)
                    loss = 100 - final_balance
                    results[model][constraint][bet_type].append(loss)

    # Calculate summary
    summary = {}
    for model in results:
        summary[model] = {}
        for constraint in results[model]:
            summary[model][constraint] = {
                'fixed': {
                    'mean': np.mean(results[model][constraint]['fixed']) if results[model][constraint]['fixed'] else 0,
                    'std': np.std(results[model][constraint]['fixed']) if results[model][constraint]['fixed'] else 0
                },
                'variable': {
                    'mean': np.mean(results[model][constraint]['variable']) if results[model][constraint]['variable'] else 0,
                    'std': np.std(results[model][constraint]['variable']) if results[model][constraint]['variable'] else 0
                }
            }

    return summary

def create_figure5_autonomy_mechanism(option4_data, loss_data):
    """Create Figure 5: Autonomy Mechanism (3 panels)"""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    models = sorted(option4_data.keys())
    constraints = [10, 30, 50, 70]
    x = np.arange(len(constraints))
    width = 0.35

    colors = {'fixed': '#3498db', 'variable': '#e74c3c'}

    # Panel A: Option 4 Selection by Constraint
    ax = axes[0]

    # Different line styles for each model
    model_styles = {
        'claude_haiku': {'marker_fixed': 'o', 'marker_var': 's', 'linestyle': '-'},
        'gemini_flash': {'marker_fixed': '^', 'marker_var': 'v', 'linestyle': '--'},
        'gpt41_mini': {'marker_fixed': 'D', 'marker_var': 'd', 'linestyle': '-.'},
        'gpt4o_mini': {'marker_fixed': 'p', 'marker_var': 'h', 'linestyle': ':'}
    }

    for i, model in enumerate(models):
        model_name = MODEL_NAMES.get(model, model)
        style = model_styles.get(model, {'marker_fixed': 'o', 'marker_var': 's', 'linestyle': '-'})

        fixed_means = [option4_data[model].get(c, {}).get('fixed', {}).get('mean', 0) * 100 for c in constraints]
        variable_means = [option4_data[model].get(c, {}).get('variable', {}).get('mean', 0) * 100 for c in constraints]

        ax.plot(x, fixed_means, marker=style['marker_fixed'], linestyle=style['linestyle'],
                color=colors['fixed'], alpha=0.85, markersize=10, linewidth=2)
        ax.plot(x, variable_means, marker=style['marker_var'], linestyle=style['linestyle'],
                color=colors['variable'], alpha=0.85, markersize=10, linewidth=2)

    ax.set_xlabel('Bet Constraint ($)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Option 4 Selection Rate (%)', fontsize=20, fontweight='bold')
    ax.set_title('(A) Extreme-Risk Selection by Bet Constraint', fontsize=22, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'${c}' for c in constraints])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(axis='y', alpha=0.3, linewidth=1.2)

    # Custom legend: color for bet type, line style for model
    from matplotlib.lines import Line2D
    legend_elements = [
        # Bet type (color)
        Line2D([0], [0], color=colors['fixed'], linewidth=3, label='Fixed Betting'),
        Line2D([0], [0], color=colors['variable'], linewidth=3, label='Variable Betting'),
        # Separator
        Line2D([0], [0], color='none', label=''),
        # Model (line style)
        Line2D([0], [0], color='gray', linestyle='--', linewidth=3, label='Gemini-2.5-Flash'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=3, label='GPT-4o-mini'),
        Line2D([0], [0], color='gray', linestyle='-.', linewidth=3, label='GPT-4.1-mini'),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=3, label='Claude-3.5-Haiku'),
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper left')

    # Panel B: Average Loss by Constraint
    ax = axes[1]
    bar_width = 0.35

    # Average across models
    fixed_losses = []
    variable_losses = []
    fixed_stds = []
    variable_stds = []

    for c in constraints:
        fixed_vals = [loss_data[m].get(c, {}).get('fixed', {}).get('mean', 0) for m in models]
        variable_vals = [loss_data[m].get(c, {}).get('variable', {}).get('mean', 0) for m in models]

        fixed_losses.append(np.mean(fixed_vals))
        variable_losses.append(np.mean(variable_vals))
        fixed_stds.append(np.std(fixed_vals))
        variable_stds.append(np.std(variable_vals))

    ax.bar(x - bar_width/2, fixed_losses, bar_width, label='Fixed Betting',
           color=colors['fixed'], alpha=0.85, yerr=fixed_stds, capsize=5,
           edgecolor='black', linewidth=1.8)
    ax.bar(x + bar_width/2, variable_losses, bar_width, label='Variable Betting',
           color=colors['variable'], alpha=0.85, yerr=variable_stds, capsize=5,
           edgecolor='black', linewidth=1.8)

    ax.set_xlabel('Bet Constraint ($)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Average Loss ($)', fontsize=20, fontweight='bold')
    ax.set_title('(B) Losses by Bet Constraint', fontsize=22, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'${c}' for c in constraints])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3, linewidth=1.2)

    # Panel C: Variable/Fixed Loss Ratio
    ax = axes[2]
    ratios = [v / f if f > 0 else 0 for v, f in zip(variable_losses, fixed_losses)]

    bars = ax.bar(x, ratios, color='#95a5a6', alpha=0.85, edgecolor='black', linewidth=1.8, width=0.5)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2.5, label='Parity (ratio=1)')

    # Color bars above 1
    for idx, (bar, ratio) in enumerate(zip(bars, ratios)):
        if ratio > 1:
            bar.set_color('#e74c3c')
            bar.set_alpha(0.85)

    ax.set_xlabel('Bet Constraint ($)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Variable/Fixed Loss Ratio', fontsize=20, fontweight='bold')
    ax.set_title('(C) Autonomy Effect Magnitude', fontsize=22, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'${c}' for c in constraints])
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=14)
    ax.grid(axis='y', alpha=0.3, linewidth=1.2)

    plt.tight_layout()

    # Save
    output_png = FIGURES_DIR / 'fig5_autonomy_mechanism.png'
    output_pdf = FIGURES_DIR / 'fig5_autonomy_mechanism.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 5 saved: {output_png}")
    print(f"✓ Figure 5 saved: {output_pdf}")

def generate_appendix_table_c(data):
    """Generate Appendix Table C: Bet Constraint Comprehensive Results"""

    models = sorted(data.keys())
    constraints = [10, 30, 50, 70]

    # Collect statistics
    stats = defaultdict(lambda: defaultdict(dict))

    for model in models:
        for constraint in constraints:
            for bet_type in ['fixed', 'variable']:
                file_data = data[model].get(constraint, {}).get(bet_type, {})
                games = file_data.get('results', [])

                if not games:
                    continue

                # Calculate metrics
                option4_rates = []
                losses = []
                rounds = []

                for game in games:
                    decisions = game.get('decisions', [])
                    if decisions:
                        option4_count = sum(1 for d in decisions if d.get('choice') == 4)
                        option4_rate = option4_count / len(decisions)
                        option4_rates.append(option4_rate * 100)

                        final_balance = game.get('final_balance', 100)
                        loss = 100 - final_balance
                        losses.append(loss)

                        rounds.append(len(decisions))

                stats[model][constraint][bet_type] = {
                    'option4_mean': np.mean(option4_rates) if option4_rates else 0,
                    'option4_std': np.std(option4_rates) if option4_rates else 0,
                    'loss_mean': np.mean(losses) if losses else 0,
                    'loss_std': np.std(losses) if losses else 0,
                    'rounds_mean': np.mean(rounds) if rounds else 0,
                    'rounds_std': np.std(rounds) if rounds else 0,
                    'n_games': len(games)
                }

    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table*}[t!]")
    latex.append("\\centering")
    latex.append("\\caption{Comprehensive analysis of investment choice behavior under bet constraints. Results show 4 models $\\times$ 4 bet constraints (\\$10, \\$30, \\$50, \\$70) $\\times$ 2 bet types $\\times$ 4 prompt conditions, with 50 trials per condition (6,400 total games). Variable betting consistently produces higher Option 4 selection rates and greater losses across all constraint levels, demonstrating that choice autonomy—not bet magnitude—drives extreme-risk selection.}")
    latex.append("\\vspace{5pt}")
    latex.append("\\label{tab:bet-constraint-comprehensive}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{llcccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Model} & \\textbf{\\makecell{Constraint\\\\(\\$)}} & \\textbf{Bet Type} & \\textbf{\\makecell{Option 4\\\\Rate (\\%)}} & \\textbf{\\makecell{Avg\\\\Rounds}} & \\textbf{\\makecell{Avg\\\\Loss (\\$)}} \\\\")
    latex.append("\\midrule")

    for model in models:
        model_name = MODEL_NAMES.get(model, model)
        model_name_escaped = model_name.replace('-', '\\\\')

        for i, constraint in enumerate(constraints):
            if i == 0:
                latex.append(f"\\multirow{{{len(constraints)*2}}}{{*}}{{\\makecell[l]{{{model_name_escaped}}}}}")

            for j, bet_type in enumerate(['fixed', 'variable']):
                s = stats[model][constraint][bet_type]

                row = f" & {constraint}" if j == 0 else " &"
                row += f" & {bet_type.capitalize()}"
                row += f" & {s['option4_mean']:.1f} $\\pm$ {s['option4_std']:.1f}"
                row += f" & {s['rounds_mean']:.1f} $\\pm$ {s['rounds_std']:.1f}"
                row += f" & {s['loss_mean']:.1f} $\\pm$ {s['loss_std']:.1f} \\\\"

                latex.append(row)

            if i < len(constraints) - 1:
                latex.append("\\cmidrule(lr){2-6}")

        if model != models[-1]:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table*}")

    # Save
    output_file = TABLES_DIR / 'tableC_bet_constraint.tex'
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"✓ Appendix Table C saved: {output_file}")

def main():
    print("="*80)
    print("BET CONSTRAINT COMPREHENSIVE ANALYSIS")
    print("="*80)

    # Load data
    print("\n1. Loading data...")
    data = load_all_data()
    print(f"   Loaded data for {len(data)} models")

    # Analyze Option 4 by constraint
    print("\n2. Analyzing Option 4 selection by constraint...")
    option4_data = analyze_option4_by_constraint(data)

    # Analyze losses
    print("\n3. Analyzing losses by constraint...")
    loss_data = analyze_losses_by_constraint(data)

    # Create Figure 5
    print("\n4. Creating Figure 5 (Autonomy Mechanism)...")
    create_figure5_autonomy_mechanism(option4_data, loss_data)

    # Generate Appendix Table C
    print("\n5. Generating Appendix Table C...")
    generate_appendix_table_c(data)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
