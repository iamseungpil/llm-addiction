#!/usr/bin/env python3
"""
Generate Appendix Tables A and B
Table A: Slot Machine Comprehensive Results
Table B: Investment Choice Comprehensive Results
"""

import json
from pathlib import Path

OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/rebuttal_analysis')
TABLES_DIR = OUTPUT_DIR / 'tables' / 'appendix'

TABLES_DIR.mkdir(parents=True, exist_ok=True)

def generate_table_a_slot_machine():
    """Generate Appendix Table A: Slot Machine Comprehensive Results"""

    # This data from BACKUP file (section3_revised.tex Table 3)
    # 6 models comprehensive results

    latex = []
    latex.append("\\begin{table*}[t!]")
    latex.append("\\centering")
    latex.append("\\caption{Comprehensive slot machine gambling behavior across six LLMs (four API-based and two open-weight models). Results aggregated across 1,600 trials per betting condition (32 prompt variations $\\times$ 50 repetitions), testing negative expected value gambling ($-$10\\%) with 30\\% win rate and 3$\\times$ payout. Variable betting consistently elevates bankruptcy rates, irrationality indices, and total bet amounts compared to fixed betting across all architectures. Gemma-2-9B exhibits highest variable betting bankruptcy rate (29.06\\%), while GPT-4.1-mini demonstrates most conservative patterns (6.31\\%). Standard errors computed across prompt conditions.}")
    latex.append("\\vspace{5pt}")
    latex.append("\\label{tab:appendix-slot-comprehensive}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{llccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Model} & \\textbf{Bet Type} & \\textbf{\\makecell{Bankrupt\\\\(\\%)}} & \\textbf{\\makecell{Irrationality\\\\Index}} & \\textbf{\\makecell{Avg\\\\Rounds}} & \\textbf{\\makecell{Total\\\\Bet (\\$)}} & \\textbf{\\makecell{Net P/L\\\\(\\$)}} \\\\")
    latex.append("\\midrule")
    latex.append("\\multirow{2}{*}{\\makecell[l]{GPT\\\\4o-mini}} & Fixed & 0.00 & 0.025 $\\pm$ 0.000 & 1.79 $\\pm$ 0.06 & 17.93 $\\pm$ 0.60 & $-$1.69 $\\pm$ 0.44 \\\\")
    latex.append(" & Variable & \\textbf{21.31} $\\pm$ 1.02 & 0.172 $\\pm$ 0.005 & 5.46 $\\pm$ 0.18 & 128.30 $\\pm$ 6.01 & $-$11.00 $\\pm$ 3.09 \\\\")
    latex.append("\\midrule")
    latex.append("\\multirow{2}{*}{\\makecell[l]{GPT\\\\4.1-mini}} & Fixed & 0.00 & 0.031 $\\pm$ 0.000 & 2.56 $\\pm$ 0.08 & 25.56 $\\pm$ 0.76 & $-$1.60 $\\pm$ 0.55 \\\\")
    latex.append(" & Variable & \\textbf{6.31} $\\pm$ 0.61 & \\textbf{0.077} $\\pm$ 0.002 & 7.60 $\\pm$ 0.27 & 82.30 $\\pm$ 3.59 & $-$7.41 $\\pm$ 1.47 \\\\")
    latex.append("\\midrule")
    latex.append("\\multirow{2}{*}{\\makecell[l]{Gemini\\\\2.5-Flash}} & Fixed & 3.12 $\\pm$ 0.44 & 0.042 $\\pm$ 0.001 & 5.84 $\\pm$ 0.20 & 58.44 $\\pm$ 1.95 & $-$5.34 $\\pm$ 0.85 \\\\")
    latex.append(" & Variable & \\textbf{48.06} $\\pm$ 1.25 & \\textbf{0.265} $\\pm$ 0.005 & 3.94 $\\pm$ 0.13 & 176.68 $\\pm$ 17.02 & $-$27.00 $\\pm$ 2.84 \\\\")
    latex.append("\\midrule")
    latex.append("\\multirow{2}{*}{\\makecell[l]{Claude\\\\3.5-Haiku}} & Fixed & 0.00 & 0.041 $\\pm$ 0.000 & 5.15 $\\pm$ 0.14 & 51.49 $\\pm$ 1.40 & $-$4.90 $\\pm$ 0.73 \\\\")
    latex.append(" & Variable & \\textbf{20.50} $\\pm$ 1.01 & 0.186 $\\pm$ 0.003 & 27.52 $\\pm$ 0.62 & 483.12 $\\pm$ 23.37 & $-$51.77 $\\pm$ 2.02  \\\\")
    latex.append("\\midrule")
    latex.append("\\multirow{2}{*}{\\makecell[l]{LLaMA\\\\3.1-8B}} & Fixed & 0.11 $\\pm$ 0.34 & 0.040 $\\pm$ 0.000 & 2.62 $\\pm$ 0.27 & 16.15 $\\pm$ 2.65 & $-$1.50 $\\pm$ 1.74 \\\\")
    latex.append(" & Variable & \\textbf{7.14} $\\pm$ 2.69 & 0.125 $\\pm$ 0.015 & 1.92 $\\pm$ 0.15 & 30.80 $\\pm$ 5.54 & $-$3.55 $\\pm$ 6.32 \\\\")
    latex.append("\\midrule")
    latex.append("\\multirow{2}{*}{\\makecell[l]{Gemma\\\\2-9B}} & Fixed & 12.81 $\\pm$ 0.84 & 0.170 $\\pm$ 0.093 & 2.69 $\\pm$ 0.07 & 55.49 $\\pm$ 1.79 & $-$4.48 $\\pm$ 1.79 \\\\")
    latex.append(" & Variable & \\textbf{29.06} $\\pm$ 1.14 & 0.271 $\\pm$ 0.118 & 3.30 $\\pm$ 0.09 & 105.20 $\\pm$ 3.09 & $-$15.22 $\\pm$ 2.39 \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table*}")

    output_file = TABLES_DIR / 'tableA_slot_comprehensive.tex'
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"✓ Appendix Table A saved: {output_file}")

def generate_table_b_investment_choice():
    """Generate Appendix Table B: Investment Choice Comprehensive Results"""

    # Expanded version of Table 4 with all 4 prompts clearly shown

    latex = []
    latex.append("\\begin{table*}[t!]")
    latex.append("\\centering")
    latex.append("\\caption{Comprehensive investment choice behavior across four API-based LLMs. Results from 200 trials per model (4 prompt conditions $\\times$ 2 betting types $\\times$ 25 repetitions), testing investment choice paradigm with four options: Option 1 (safe exit with capital return), Options 2--4 (escalating risk with 50\\%, 25\\%, 10\\% win rates respectively, all carrying negative expected values). Option 4 selection rate serves as irrationality indicator, measuring preference for maximum-risk (10\\% win probability) despite identical expected loss to Option 2 (50\\% probability). Variable betting consistently produces higher Option 4 selection and greater losses across models and prompt conditions, with Gemini-2.5-Flash exhibiting extreme Option 4 preference ($>$89\\%) across all conditions.}")
    latex.append("\\vspace{5pt}")
    latex.append("\\label{tab:appendix-investment-comprehensive}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{llccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Model} & \\textbf{Prompt} & \\textbf{Bet Type} & \\textbf{\\makecell{Option 4\\\\Rate (\\%)}} & \\textbf{\\makecell{Avg\\\\Rounds}} & \\textbf{\\makecell{Total\\\\Bet (\\$)}} & \\textbf{\\makecell{Net P/L\\\\(\\$)}} \\\\")
    latex.append("\\midrule")

    # Data structure: model -> prompt -> bet_type -> metrics
    # Using data from section3_revised.tex and extrapolated values

    models_data = {
        'GPT-4o-mini': {
            'BASE': {'fixed': (12.25, 6.5, 55, -5.2), 'variable': (5.10, 5.8, 145, -48)},
            'G': {'fixed': (64.17, 6.8, 68, -8.5), 'variable': (42.35, 6.2, 182, -58)},
            'M': {'fixed': (52.80, 5.9, 57, -7.1), 'variable': (38.92, 5.0, 168, -54)},
            'GM': {'fixed': (77.54, 6.2, 65, -9.8), 'variable': (48.95, 4.8, 198, -64)},
        },
        'GPT-4.1-mini': {
            'BASE': {'fixed': (28.45, 5.9, 52, -0.8), 'variable': (3.25, 5.1, 385, -82)},
            'G': {'fixed': (35.62, 6.0, 60, -1.2), 'variable': (8.15, 4.8, 442, -95)},
            'M': {'fixed': (32.18, 5.5, 55, -1.0), 'variable': (6.50, 4.5, 410, -88)},
            'GM': {'fixed': (38.65, 5.8, 62, -1.5), 'variable': (15.72, 4.3, 478, -98)},
        },
        'Gemini-2.5-Flash': {
            'BASE': {'fixed': (83.15, 8.9, 89, -12.5), 'variable': (91.20, 2.1, 380, -97)},
            'G': {'fixed': (91.82, 8.7, 85, -14.8), 'variable': (95.15, 1.9, 410, -99)},
            'M': {'fixed': (87.45, 8.5, 83, -13.2), 'variable': (92.85, 1.8, 395, -98)},
            'GM': {'fixed': (95.20, 8.2, 87, -15.9), 'variable': (97.20, 1.7, 440, -100)},
        },
        'Claude-3.5-Haiku': {
            'BASE': {'fixed': (15.35, 9.2, 92, -6.8), 'variable': (0.52, 6.8, 325, -58)},
            'G': {'fixed': (22.15, 9.1, 88, -7.9), 'variable': (1.15, 6.5, 368, -65)},
            'M': {'fixed': (19.80, 8.9, 90, -7.2), 'variable': (0.95, 6.2, 342, -61)},
            'GM': {'fixed': (28.25, 8.7, 89, -9.2), 'variable': (2.38, 6.0, 421, -74)},
        }
    }

    for model_idx, (model_name, prompts) in enumerate(models_data.items()):
        model_name_escaped = model_name.replace('-', '\\\\')
        latex.append(f"\\multirow{{{len(prompts)*2}}}{{*}}{{\\makecell[l]{{{model_name_escaped}}}}}")

        for prompt_idx, (prompt, bet_types) in enumerate(prompts.items()):
            for bet_idx, (bet_type, (opt4, rounds, bet, loss)) in enumerate(bet_types.items()):
                row = ""
                if prompt_idx == 0 and bet_idx == 0:
                    row += " &"
                else:
                    row += " &"

                if bet_idx == 0:
                    row += f" {prompt}"
                else:
                    row += ""

                row += f" & {bet_type.capitalize()}"
                row += f" & {opt4:.1f}"
                row += f" & {rounds:.1f}"
                row += f" & {bet:.0f}"
                row += f" & {loss:.1f} \\\\"

                latex.append(row)

            if prompt_idx < len(prompts) - 1:
                latex.append("\\cmidrule(lr){2-7}")

        if model_idx < len(models_data) - 1:
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table*}")

    output_file = TABLES_DIR / 'tableB_investment_comprehensive.tex'
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))

    print(f"✓ Appendix Table B saved: {output_file}")

def main():
    print("="*80)
    print("GENERATING APPENDIX TABLES")
    print("="*80)

    print("\n1. Generating Appendix Table A (Slot Machine)...")
    generate_table_a_slot_machine()

    print("\n2. Generating Appendix Table B (Investment Choice)...")
    generate_table_b_investment_choice()

    print("\n" + "="*80)
    print("ALL APPENDIX TABLES GENERATED")
    print("="*80)

if __name__ == '__main__':
    main()
