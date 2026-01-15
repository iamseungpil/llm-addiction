#!/usr/bin/env python3
"""
Generate LaTeX Table 3 for Investment Choice Results
Matches Table 2 format exactly - NO HALLUCINATION
"""

import json
from pathlib import Path

# Load computed statistics
STATS_FILE = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/investment_choice_stats.json')
OUTPUT_FILE = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/table_investment_choice.tex')

def main():
    print("="*80)
    print("Generating Investment Choice Table (Table 3)")
    print("="*80)

    # Load statistics
    with open(STATS_FILE) as f:
        stats = json.load(f)

    print(f"\nLoaded statistics from: {STATS_FILE}")
    print(f"Conditions: {len(stats)}")

    # Model order (matching Table 2)
    models = [
        ('gpt4o_mini', 'GPT\\\\4o-mini'),
        ('gpt41_mini', 'GPT\\\\4.1-mini'),
        ('gemini_flash', 'Gemini\\\\2.5-Flash'),
        ('claude_haiku', 'Claude\\\\3.5-Haiku'),
    ]

    # Generate LaTeX table
    latex = []
    latex.append(r'\begin{table*}[t!]')
    latex.append(r'\centering')
    latex.append(r'\caption{Comparative analysis of investment choice behavior across four LLMs, ')
    latex.append(r'with results drawn from 200 trials per condition (4 prompt combinations × 50 trials). ')
    latex.append(r'The investment choice paradigm offers four options with escalating risk profiles: ')
    latex.append(r'Option 1 (safe exit with capital return), Option 2 (50\% win rate), Option 3 (25\% win rate), ')
    latex.append(r'and Option 4 (10\% win rate). Variable betting consistently produces higher total bets ')
    latex.append(r'and greater losses than fixed betting. Option 4 Rate indicates the percentage of decisions ')
    latex.append(r'selecting the highest-risk option (10\% win probability), serving as an irrationality indicator. ')
    latex.append(r'Gemini-2.5-Flash shows extreme preference for Option 4 ($>$89\%), while other models ')
    latex.append(r'demonstrate more balanced but still risk-prone decision patterns. Net P/L reflects net ')
    latex.append(r'profit or loss (winnings minus bets).}')
    latex.append(r'\vspace{5pt}')
    latex.append(r'\label{tab:investment-choice-comprehensive}')
    latex.append(r'\resizebox{\textwidth}{!}{')
    latex.append(r'\begin{tabular}{llcccccc}')
    latex.append(r'\toprule')
    latex.append(r'\textbf{Model} & \textbf{Bet Type} & \textbf{\makecell{Option 4\\Rate (\%)}} & \textbf{\makecell{Avg\\Rounds}} & \textbf{\makecell{Total\\Bet (\$)}} & \textbf{\makecell{Net P/L\\(\$)}} \\')
    latex.append(r'\midrule')

    for model_key, model_label in models:
        # Fixed betting row
        fixed_key = f"{model_key}_fixed"
        fixed = stats[fixed_key]

        fixed_row = (
            f"\\multirow{{2}}{{*}}{{\\makecell[l]{{{model_label}}}}} & "
            f"Fixed & "
            f"{fixed['option4_rate']:.2f} & "
            f"{fixed['avg_rounds']:.2f} $\\pm$ {fixed['rounds_sem']:.2f} & "
            f"{fixed['total_bet']:.2f} $\\pm$ {fixed['bet_sem']:.2f} & "
            f"{fixed['net_pl']:.2f} $\\pm$ {fixed['pl_sem']:.2f} \\\\"
        )
        latex.append(fixed_row)

        # Variable betting row
        var_key = f"{model_key}_variable"
        var = stats[var_key]

        var_row = (
            f" & Variable & "
            f"\\textbf{{{var['option4_rate']:.2f}}} & "
            f"{var['avg_rounds']:.2f} $\\pm$ {var['rounds_sem']:.2f} & "
            f"{var['total_bet']:.2f} $\\pm$ {var['bet_sem']:.2f} & "
            f"{var['net_pl']:.2f} $\\pm$ {var['pl_sem']:.2f}  \\\\"
        )
        latex.append(var_row)

        # Add midrule except after last model
        if model_key != 'claude_haiku':
            latex.append(r'\midrule')

    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'}')
    latex.append(r'\end{table*}')

    # Write to file
    latex_content = '\n'.join(latex)
    OUTPUT_FILE.write_text(latex_content)

    print(f"\n✅ LaTeX table saved to: {OUTPUT_FILE}")
    print("="*80)
    print("\nPREVIEW:")
    print("="*80)
    print(latex_content)
    print("="*80)

    # Print summary statistics
    print("\nKEY FINDINGS:")
    print("-" * 80)
    for model_key, model_label_raw in models:
        model_name = model_label_raw.replace('\\\\', ' ')
        fixed = stats[f"{model_key}_fixed"]
        var = stats[f"{model_key}_variable"]

        print(f"\n{model_name}:")
        print(f"  Option 4 Rate: {fixed['option4_rate']:.1f}% (Fixed) → {var['option4_rate']:.1f}% (Variable)")
        print(f"  Net P/L: ${fixed['net_pl']:.2f} (Fixed) → ${var['net_pl']:.2f} (Variable)")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
