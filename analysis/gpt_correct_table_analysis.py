#!/usr/bin/env python3
"""
GPT Table 2 정확한 계산 - 베팅 타입별로 승리/패배 조건을 구분하여 계산
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_bet_type_by_first_result():
    """베팅 타입별로 첫 게임 결과를 구분하여 정확한 통계 계산"""
    
    # Load data
    data_path = "/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each experiment
    processed_results = []
    
    for exp in data['results']:
        # Basic info
        bet_type = exp['bet_type']
        first_result = exp['first_result']
        
        # Calculate baseline (first game end balance)
        first_game_end_balance = 130 if first_result == 'W' else 90
        
        # Final balance and net profit from baseline
        final_balance = exp['final_balance']
        net_profit_from_baseline = final_balance - first_game_end_balance
        
        # Additional rounds (excluding first game)
        additional_rounds = exp['total_rounds']
        total_rounds_including_first = additional_rounds + 1
        
        # Immediate stop
        immediate_stop = (additional_rounds == 0)
        
        # Betting amounts
        betting_amounts = []
        if exp.get('round_details'):
            for round_detail in exp['round_details']:
                bet_amount = round_detail.get('bet_amount')
                if bet_amount:
                    betting_amounts.append(bet_amount)
        
        avg_bet_amount = np.mean(betting_amounts) if betting_amounts else 0
        
        processed_results.append({
            'bet_type': bet_type,
            'first_result': first_result,
            'net_profit_from_baseline': net_profit_from_baseline,
            'additional_rounds': additional_rounds,
            'total_rounds_including_first': total_rounds_including_first,
            'immediate_stop': immediate_stop,
            'is_bankrupt': exp['is_bankrupt'],
            'avg_bet_amount': avg_bet_amount,
            'first_game_end_balance': first_game_end_balance,
            'final_balance': final_balance
        })
    
    df = pd.DataFrame(processed_results)
    
    print("📊 정확한 베팅 타입별 분석 (첫 게임 결과 구분)")
    print("=" * 60)
    
    # Calculate stats for each combination
    combinations = [
        ('fixed', 'W', '고정 베팅 (승리 후)', 130),
        ('fixed', 'L', '고정 베팅 (패배 후)', 90),
        ('variable', 'W', '가변 베팅 (승리 후)', 130),
        ('variable', 'L', '가변 베팅 (패배 후)', 90)
    ]
    
    detailed_stats = {}
    
    for bet_type, first_result, korean_name, baseline in combinations:
        mask = (df['bet_type'] == bet_type) & (df['first_result'] == first_result)
        subset = df[mask]
        
        stats = {
            'korean_name': korean_name,
            'baseline': baseline,
            'n': len(subset),
            'bankruptcy_rate': subset['is_bankrupt'].mean() * 100,
            'immediate_stop_rate': subset['immediate_stop'].mean() * 100,
            'avg_net_profit': subset['net_profit_from_baseline'].mean(),
            'avg_total_rounds': subset['total_rounds_including_first'].mean(),
            'avg_bet_amount': subset['avg_bet_amount'].mean(),
        }
        
        detailed_stats[f"{bet_type}_{first_result}"] = stats
        
        print(f"\n{korean_name} (기준: ${baseline})")
        print(f"  N: {stats['n']}")
        print(f"  파산율: {stats['bankruptcy_rate']:.1f}%")
        print(f"  즉시중단율: {stats['immediate_stop_rate']:.1f}%")
        print(f"  평균 순손익: ${stats['avg_net_profit']:.2f}")
        print(f"  평균 라운드: {stats['avg_total_rounds']:.1f}")
        print(f"  평균 베팅: ${stats['avg_bet_amount']:.2f}")
    
    # Calculate overall bet type stats (combining W and L)
    print("\n" + "=" * 60)
    print("📊 전체 베팅 타입별 통계 (승리+패배 통합)")
    print("=" * 60)
    
    overall_stats = {}
    for bet_type in ['fixed', 'variable']:
        mask = df['bet_type'] == bet_type
        subset = df[mask]
        
        korean_name = '고정 베팅' if bet_type == 'fixed' else '가변 베팅'
        
        stats = {
            'korean_name': korean_name,
            'n': len(subset),
            'bankruptcy_rate': subset['is_bankrupt'].mean() * 100,
            'immediate_stop_rate': subset['immediate_stop'].mean() * 100,
            'avg_net_profit': subset['net_profit_from_baseline'].mean(),
            'avg_total_rounds': subset['total_rounds_including_first'].mean(),
            'avg_bet_amount': subset['avg_bet_amount'].mean(),
        }
        
        overall_stats[bet_type] = stats
        
        print(f"\n{korean_name} (전체)")
        print(f"  N: {stats['n']}")
        print(f"  파산율: {stats['bankruptcy_rate']:.1f}%")
        print(f"  즉시중단율: {stats['immediate_stop_rate']:.1f}%")
        print(f"  평균 순손익: ${stats['avg_net_profit']:.2f}")
        print(f"  평균 라운드: {stats['avg_total_rounds']:.1f}")
        print(f"  평균 베팅: ${stats['avg_bet_amount']:.2f}")
    
    # Check if current table values are correct
    print("\n" + "=" * 60)
    print("🔍 현재 논문 Table 2 값 검증")
    print("=" * 60)
    
    current_table_values = {
        'fixed': {'net_profit': 0.02, 'immediate_stop': 94.5, 'avg_bet': 0.55},
        'variable': {'net_profit': -0.08, 'immediate_stop': 39.4, 'avg_bet': 22.26}
    }
    
    for bet_type in ['fixed', 'variable']:
        korean_name = '고정 베팅' if bet_type == 'fixed' else '가변 베팅'
        calculated = overall_stats[bet_type]
        current = current_table_values[bet_type]
        
        print(f"\n{korean_name}:")
        print(f"  순손익: 논문=${current['net_profit']:.2f} vs 계산=${calculated['avg_net_profit']:.2f}")
        print(f"  즉시중단: 논문={current['immediate_stop']:.1f}% vs 계산={calculated['immediate_stop_rate']:.1f}%")
        print(f"  평균베팅: 논문=${current['avg_bet']:.2f} vs 계산=${calculated['avg_bet_amount']:.2f}")
        
        # Check if values match
        profit_match = abs(current['net_profit'] - calculated['avg_net_profit']) < 0.01
        stop_match = abs(current['immediate_stop'] - calculated['immediate_stop_rate']) < 0.1
        bet_match = abs(current['avg_bet'] - calculated['avg_bet_amount']) < 0.1
        
        print(f"  ✅ 일치도: 순손익={profit_match}, 즉시중단={stop_match}, 베팅={bet_match}")
    
    # Generate corrected LaTeX table
    print("\n" + "=" * 60)
    print("📄 수정된 LaTeX Table 생성")
    print("=" * 60)
    
    latex_lines = []
    
    # Fixed betting
    fixed_w = detailed_stats['fixed_W']
    fixed_l = detailed_stats['fixed_L']
    fixed_overall = overall_stats['fixed']
    
    latex_lines.append(f"고정 베팅 (전체) & {fixed_overall['n']} & {fixed_overall['bankruptcy_rate']:.1f} & {fixed_overall['immediate_stop_rate']:.1f} & {fixed_overall['avg_net_profit']:.2f} & {fixed_overall['avg_total_rounds']:.1f} & {fixed_overall['avg_bet_amount']:.2f} \\\\")
    latex_lines.append(f"\\quad 승리 후 (\\$130 기준) & {fixed_w['n']} & {fixed_w['bankruptcy_rate']:.1f} & {fixed_w['immediate_stop_rate']:.1f} & {fixed_w['avg_net_profit']:.2f} & {fixed_w['avg_total_rounds']:.1f} & {fixed_w['avg_bet_amount']:.2f} \\\\")
    latex_lines.append(f"\\quad 패배 후 (\\$90 기준) & {fixed_l['n']} & {fixed_l['bankruptcy_rate']:.1f} & {fixed_l['immediate_stop_rate']:.1f} & {fixed_l['avg_net_profit']:.2f} & {fixed_l['avg_total_rounds']:.1f} & {fixed_l['avg_bet_amount']:.2f} \\\\")
    
    # Variable betting
    var_w = detailed_stats['variable_W']
    var_l = detailed_stats['variable_L']
    var_overall = overall_stats['variable']
    
    latex_lines.append(f"가변 베팅 (전체) & {var_overall['n']} & {var_overall['bankruptcy_rate']:.1f} & {var_overall['immediate_stop_rate']:.1f} & {var_overall['avg_net_profit']:.2f} & {var_overall['avg_total_rounds']:.1f} & {var_overall['avg_bet_amount']:.2f} \\\\")
    latex_lines.append(f"\\quad 승리 후 (\\$130 기준) & {var_w['n']} & {var_w['bankruptcy_rate']:.1f} & {var_w['immediate_stop_rate']:.1f} & {var_w['avg_net_profit']:.2f} & {var_w['avg_total_rounds']:.1f} & {var_w['avg_bet_amount']:.2f} \\\\")
    latex_lines.append(f"\\quad 패배 후 (\\$90 기준) & {var_l['n']} & {var_l['bankruptcy_rate']:.1f} & {var_l['immediate_stop_rate']:.1f} & {var_l['avg_net_profit']:.2f} & {var_l['avg_total_rounds']:.1f} & {var_l['avg_bet_amount']:.2f} \\\\")
    
    print("\n=== CORRECTED TABLE ROWS ===")
    for line in latex_lines:
        print(line)
    
    return detailed_stats, overall_stats, latex_lines

if __name__ == "__main__":
    detailed_stats, overall_stats, latex_lines = analyze_bet_type_by_first_result()