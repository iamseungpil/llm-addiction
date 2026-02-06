#!/usr/bin/env python3
import json

def create_english_table():
    with open('/home/ubuntu/llm_addiction/all_models_comprehensive_stats.json', 'r') as f:
        data = json.load(f)

    latex = r"""\begin{table*}[t!]
\centering
\caption{Comparative Analysis of Gambling Behavior Across LLM Models. Net P/L represents the net profit/loss calculated as total winnings minus total bets, reflecting the overall financial outcome for each betting strategy. Bankruptcy rates and total betting amounts vary significantly across models, with GPT-4o-mini showing the highest risk-taking behavior in variable betting conditions.}
\vspace{5pt}
\label{tab:multi-model-comprehensive}
\begin{tabular}{llcccccc}
\toprule
\textbf{Model} & \textbf{Bet Type} & \textbf{N} & \textbf{\makecell{Bankrupt\\(\%)}} & \textbf{\makecell{Voluntary\\Stop (\%)}} & \textbf{\makecell{Avg\\Rounds}} & \textbf{\makecell{Avg Total\\Bet (\$)}} & \textbf{\makecell{Net P/L\\(\$)}} \\
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
        if model in data:
            model_data = data[model]
            display_name = model_display_names.get(model, model)

            bet_types = [bt for bt in ['Fixed', 'Variable'] if bt in model_data]

            if len(bet_types) == 2:
                for i, bet_type in enumerate(bet_types):
                    d = model_data[bet_type]
                    vol_stop_rate = (d['voluntary_stops'] / d['n'] * 100) if d['n'] > 0 else 0
                    net_pl = d.get('avg_net_pl', d.get('avg_total_won', 0) - d['avg_total_bet'])

                    if i == 0:
                        latex += f"\\multirow{{2}}{{*}}{{{display_name}}} & {bet_type} & {d['n']:,} & {d['bankruptcy_rate']:.2f} & {vol_stop_rate:.2f} & {d['avg_rounds']:.2f} & {d['avg_total_bet']:.2f} & {net_pl:.2f} \\\\\n"
                    else:
                        latex += f" & {bet_type} & {d['n']:,} & {d['bankruptcy_rate']:.2f} & {vol_stop_rate:.2f} & {d['avg_rounds']:.2f} & {d['avg_total_bet']:.2f} & {net_pl:.2f} \\\\\n"
            else:
                for bet_type in bet_types:
                    d = model_data[bet_type]
                    vol_stop_rate = (d['voluntary_stops'] / d['n'] * 100) if d['n'] > 0 else 0
                    net_pl = d.get('avg_net_pl', d.get('avg_total_won', 0) - d['avg_total_bet'])
                    latex += f"{display_name} & {bet_type} & {d['n']:,} & {d['bankruptcy_rate']:.2f} & {vol_stop_rate:.2f} & {d['avg_rounds']:.2f} & {d['avg_total_bet']:.2f} & {net_pl:.2f} \\\\\n"

            latex += r"\midrule" + "\n"

    latex = latex.rstrip(r"\midrule" + "\n")
    latex += r""" \\
\bottomrule
\end{tabular}
\end{table*}
"""

    with open('/home/ubuntu/llm_addiction/writing/figures/MULTI_MODEL_comprehensive_table_english.tex', 'w') as f:
        f.write(latex)

    print("English table created successfully!")

def create_korean_table():
    with open('/home/ubuntu/llm_addiction/all_models_comprehensive_stats.json', 'r') as f:
        data = json.load(f)

    latex = r"""\begin{table*}[t!]
\centering
\caption{LLM 모델 간 도박 행동 비교 분석. Net P/L은 총 승리금에서 총 베팅액을 뺀 순손익을 나타내며, 각 베팅 전략의 전체 재정적 결과를 반영합니다. 파산율과 총 베팅액은 모델 간 큰 차이를 보이며, GPT-4o-mini가 변동 베팅 조건에서 가장 높은 위험 감수 행동을 보였습니다.}
\vspace{5pt}
\label{tab:multi-model-comprehensive-korean}
\begin{tabular}{llcccccc}
\toprule
\textbf{모델} & \textbf{베팅 유형} & \textbf{N} & \textbf{\makecell{파산\\(\%)}} & \textbf{\makecell{자발적\\중단 (\%)}} & \textbf{\makecell{평균\\라운드}} & \textbf{\makecell{평균 총\\베팅 (\$)}} & \textbf{\makecell{순손익\\(\$)}} \\
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
        if model in data:
            model_data = data[model]
            display_name = model_display_names.get(model, model)

            bet_types = [bt for bt in ['Fixed', 'Variable'] if bt in model_data]

            if len(bet_types) == 2:
                for i, bet_type in enumerate(bet_types):
                    d = model_data[bet_type]
                    vol_stop_rate = (d['voluntary_stops'] / d['n'] * 100) if d['n'] > 0 else 0
                    net_pl = d.get('avg_net_pl', d.get('avg_total_won', 0) - d['avg_total_bet'])

                    if i == 0:
                        latex += f"\\multirow{{2}}{{*}}{{{display_name}}} & {bet_type_kr[bet_type]} & {d['n']:,} & {d['bankruptcy_rate']:.2f} & {vol_stop_rate:.2f} & {d['avg_rounds']:.2f} & {d['avg_total_bet']:.2f} & {net_pl:.2f} \\\\\n"
                    else:
                        latex += f" & {bet_type_kr[bet_type]} & {d['n']:,} & {d['bankruptcy_rate']:.2f} & {vol_stop_rate:.2f} & {d['avg_rounds']:.2f} & {d['avg_total_bet']:.2f} & {net_pl:.2f} \\\\\n"
            else:
                for bet_type in bet_types:
                    d = model_data[bet_type]
                    vol_stop_rate = (d['voluntary_stops'] / d['n'] * 100) if d['n'] > 0 else 0
                    net_pl = d.get('avg_net_pl', d.get('avg_total_won', 0) - d['avg_total_bet'])
                    latex += f"{display_name} & {bet_type_kr[bet_type]} & {d['n']:,} & {d['bankruptcy_rate']:.2f} & {vol_stop_rate:.2f} & {d['avg_rounds']:.2f} & {d['avg_total_bet']:.2f} & {net_pl:.2f} \\\\\n"

            latex += r"\midrule" + "\n"

    latex = latex.rstrip(r"\midrule" + "\n")
    latex += r""" \\
\bottomrule
\end{tabular}
\end{table*}
"""

    with open('/home/ubuntu/llm_addiction/writing/figures/MULTI_MODEL_comprehensive_table_korean.tex', 'w') as f:
        f.write(latex)

    print("Korean table created successfully!")

if __name__ == '__main__':
    create_english_table()
    create_korean_table()
    print("\nBoth tables created at:")
    print("  - /home/ubuntu/llm_addiction/writing/figures/MULTI_MODEL_comprehensive_table_english.tex")
    print("  - /home/ubuntu/llm_addiction/writing/figures/MULTI_MODEL_comprehensive_table_korean.tex")