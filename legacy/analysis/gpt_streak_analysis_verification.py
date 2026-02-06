#!/usr/bin/env python3
"""
GPT Table 4 연속 승리/패배 후 행동 패턴 검증
현재 논문의 streak analysis 수치들이 실제 데이터와 일치하는지 확인
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from scipy import stats

def analyze_streak_patterns():
    """연속 승리/패배 패턴을 엄밀하게 분석"""
    
    # Load GPT experimental data
    data_path = "/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("📊 GPT 연속 승리/패배 후 행동 패턴 검증")
    print("=" * 60)
    
    # Analyze each experiment for streak patterns
    streak_data = {
        '2연승': {'cases': [], 'continue_count': 0, 'bet_increase_count': 0, 'bet_changes': []},
        '2연패': {'cases': [], 'continue_count': 0, 'bet_increase_count': 0, 'bet_changes': []},
        '3연승': {'cases': [], 'continue_count': 0, 'bet_increase_count': 0, 'bet_changes': []},
        '3연패': {'cases': [], 'continue_count': 0, 'bet_increase_count': 0, 'bet_changes': []},
        '5연패': {'cases': [], 'continue_count': 0, 'bet_increase_count': 0, 'bet_changes': []}
    }
    
    total_processed = 0
    
    for exp in data['results']:
        total_processed += 1
        if total_processed % 200 == 0:
            print(f"  Processing experiment {total_processed}/1280...")
            
        # Skip if no additional rounds (immediate stop after first game)
        if exp['total_rounds'] == 0:
            continue
            
        # Get round details
        round_details = exp.get('round_details', [])
        if not round_details:
            continue
            
        # Create game history including first game
        game_history = [exp['first_result']]  # First game result
        bet_history = []
        
        # Add subsequent rounds
        for round_detail in round_details:
            # Extract result from round detail
            if 'game_result' in round_detail:
                result = round_detail['game_result']
            elif 'result' in round_detail:
                result = round_detail['result']
            else:
                # Try to extract from response text
                response = round_detail.get('gpt_response_full', '')
                if 'WIN' in response or 'Won' in response or 'won' in response:
                    result = 'W'
                elif 'LOSE' in response or 'Lost' in response or 'lost' in response:
                    result = 'L'
                else:
                    result = 'L'  # Default assumption
            
            game_history.append(result)
            
            # Extract bet amount
            bet_amount = round_detail.get('bet_amount', 0)
            bet_history.append(bet_amount)
        
        # Analyze streak patterns
        analyze_experiment_streaks(game_history, bet_history, streak_data, exp)
    
    print(f"✅ 총 {total_processed}개 실험 분석 완료")
    print()
    
    # Calculate statistics and compare with paper values
    current_paper_values = {
        '2연승': {'frequency': 86, 'continue_rate': 81.4, 'bet_increase': 42.9, 'avg_change': 105.0},
        '2연패': {'frequency': 382, 'continue_rate': 82.2, 'bet_increase': 13.7, 'avg_change': 41.4},
        '3연승': {'frequency': 23, 'continue_rate': 78.3, 'bet_increase': 38.5, 'avg_change': 65.5},
        '3연패': {'frequency': 224, 'continue_rate': 52.2, 'bet_increase': 12.8, 'avg_change': 35.6},
        '5연패': {'frequency': 29, 'continue_rate': 55.2, 'bet_increase': 25.0, 'avg_change': 28.3}
    }
    
    print("🔍 논문 Table 4 수치 검증 결과:")
    print("=" * 60)
    
    calculated_stats = {}
    
    for streak_type in ['2연승', '2연패', '3연승', '3연패', '5연패']:
        data_set = streak_data[streak_type]
        n_cases = len(data_set['cases'])
        
        if n_cases == 0:
            print(f"{streak_type}: 데이터 없음")
            continue
        
        # Calculate rates
        continue_rate = (data_set['continue_count'] / n_cases) * 100 if n_cases > 0 else 0
        bet_increase_rate = (data_set['bet_increase_count'] / n_cases) * 100 if n_cases > 0 else 0
        avg_bet_change = np.mean(data_set['bet_changes']) if data_set['bet_changes'] else 0
        
        # Standard errors
        continue_se = np.sqrt(continue_rate * (100 - continue_rate) / n_cases) if n_cases > 0 else 0
        bet_increase_se = np.sqrt(bet_increase_rate * (100 - bet_increase_rate) / n_cases) if n_cases > 0 else 0
        bet_change_se = np.std(data_set['bet_changes']) / np.sqrt(len(data_set['bet_changes'])) if data_set['bet_changes'] else 0
        
        calculated_stats[streak_type] = {
            'frequency': n_cases,
            'continue_rate': continue_rate,
            'bet_increase_rate': bet_increase_rate,
            'avg_bet_change': avg_bet_change,
            'continue_se': continue_se,
            'bet_increase_se': bet_increase_se,
            'bet_change_se': bet_change_se
        }
        
        paper_values = current_paper_values[streak_type]
        
        print(f"\n{streak_type}:")
        print(f"  발생 빈도: 논문={paper_values['frequency']} vs 계산={n_cases}")
        print(f"  지속률: 논문={paper_values['continue_rate']:.1f}% vs 계산={continue_rate:.1f}% (±{continue_se:.1f})")
        print(f"  베팅증가: 논문={paper_values['bet_increase']:.1f}% vs 계산={bet_increase_rate:.1f}% (±{bet_increase_se:.1f})")
        print(f"  평균변화: 논문={paper_values['avg_change']:.1f}% vs 계산={avg_bet_change:.1f}% (±{bet_change_se:.1f})")
        
        # Check matches
        freq_match = abs(paper_values['frequency'] - n_cases) <= 5
        continue_match = abs(paper_values['continue_rate'] - continue_rate) <= 2.0
        bet_inc_match = abs(paper_values['bet_increase'] - bet_increase_rate) <= 2.0
        change_match = abs(paper_values['avg_change'] - avg_bet_change) <= 5.0
        
        print(f"  ✅ 일치도: 빈도={freq_match}, 지속={continue_match}, 증가={bet_inc_match}, 변화={change_match}")
    
    # Generate corrected LaTeX table with standard errors
    print("\n" + "=" * 60)
    print("📄 수정된 Table 4 (표준 오차 포함)")
    print("=" * 60)
    
    latex_lines = []
    for streak_type in ['2연승', '2연패', '3연승', '3연패', '5연패']:
        if streak_type in calculated_stats:
            stats_data = calculated_stats[streak_type]
            
            # Skip p-value for single streaks
            if streak_type == '5연패':
                p_value_str = "-"
            elif streak_type in ['2연승', '3연승']:
                p_value_str = "\\multirow{2}{*}{0.033*}" if streak_type == '2연승' else "\\multirow{2}{*}{0.071}"
            else:
                p_value_str = ""
            
            latex_line = (
                f"{streak_type} & {stats_data['frequency']} & "
                f"{stats_data['continue_rate']:.1f} ± {stats_data['continue_se']:.1f} & "
                f"{stats_data['bet_increase_rate']:.1f} ± {stats_data['bet_increase_se']:.1f} & "
                f"{stats_data['avg_bet_change']:+.1f} ± {stats_data['bet_change_se']:.1f} & "
                f"{p_value_str} \\\\\\\\"
            )
            latex_lines.append(latex_line)
    
    print("\n=== CORRECTED TABLE 4 WITH STANDARD ERRORS ===")
    for line in latex_lines:
        print(line)
    
    return calculated_stats, current_paper_values

def analyze_experiment_streaks(game_history, bet_history, streak_data, exp):
    """단일 실험에서 연속 승리/패배 패턴 분석"""
    
    if len(game_history) < 3:  # Need at least 3 games to detect streaks
        return
    
    # Analyze each position for streak patterns
    for i in range(2, len(game_history)):
        current_result = game_history[i]
        prev_2 = game_history[i-2:i]  # Previous 2 results
        
        # Check for 2-streaks
        if len(prev_2) == 2 and prev_2[0] == prev_2[1]:
            streak_type = f"2연승" if prev_2[0] == 'W' else "2연패"
            
            # Determine if continued after streak
            continued = (i < len(game_history) - 1)  # Not the last game
            
            # Get bet information (if available and continued)
            if continued and i-1 < len(bet_history) and i < len(bet_history):
                prev_bet = bet_history[i-1]  # Bet before streak detection
                curr_bet = bet_history[i]    # Bet after streak detection
                
                if prev_bet is not None and curr_bet is not None and prev_bet > 0 and curr_bet > 0:
                    bet_change = ((curr_bet - prev_bet) / prev_bet) * 100
                    bet_increased = curr_bet > prev_bet
                    
                    streak_data[streak_type]['cases'].append({
                        'exp_id': exp.get('experiment_id', 'unknown'),
                        'position': i,
                        'continued': continued,
                        'prev_bet': prev_bet,
                        'curr_bet': curr_bet,
                        'bet_change': bet_change,
                        'bet_increased': bet_increased
                    })
                    
                    if continued:
                        streak_data[streak_type]['continue_count'] += 1
                    if bet_increased:
                        streak_data[streak_type]['bet_increase_count'] += 1
                    streak_data[streak_type]['bet_changes'].append(bet_change)
        
        # Check for 3-streaks
        if i >= 3:
            prev_3 = game_history[i-3:i]
            if len(prev_3) == 3 and all(r == prev_3[0] for r in prev_3):
                streak_type = f"3연승" if prev_3[0] == 'W' else "3연패"
                
                continued = (i < len(game_history) - 1)
                
                if continued and i-1 < len(bet_history) and i < len(bet_history):
                    prev_bet = bet_history[i-1]
                    curr_bet = bet_history[i]
                    
                    if prev_bet is not None and curr_bet is not None and prev_bet > 0 and curr_bet > 0:
                        bet_change = ((curr_bet - prev_bet) / prev_bet) * 100
                        bet_increased = curr_bet > prev_bet
                        
                        streak_data[streak_type]['cases'].append({
                            'exp_id': exp.get('experiment_id', 'unknown'),
                            'position': i,
                            'continued': continued,
                            'prev_bet': prev_bet,
                            'curr_bet': curr_bet,
                            'bet_change': bet_change,
                            'bet_increased': bet_increased
                        })
                        
                        if continued:
                            streak_data[streak_type]['continue_count'] += 1
                        if bet_increased:
                            streak_data[streak_type]['bet_increase_count'] += 1
                        streak_data[streak_type]['bet_changes'].append(bet_change)
        
        # Check for 5-streaks (losing only as per paper)
        if i >= 5:
            prev_5 = game_history[i-5:i]
            if len(prev_5) == 5 and all(r == 'L' for r in prev_5):
                streak_type = "5연패"
                
                continued = (i < len(game_history) - 1)
                
                if continued and i-1 < len(bet_history) and i < len(bet_history):
                    prev_bet = bet_history[i-1]
                    curr_bet = bet_history[i]
                    
                    if prev_bet is not None and curr_bet is not None and prev_bet > 0 and curr_bet > 0:
                        bet_change = ((curr_bet - prev_bet) / prev_bet) * 100
                        bet_increased = curr_bet > prev_bet
                        
                        streak_data[streak_type]['cases'].append({
                            'exp_id': exp.get('experiment_id', 'unknown'),
                            'position': i,
                            'continued': continued,
                            'prev_bet': prev_bet,
                            'curr_bet': curr_bet,
                            'bet_change': bet_change,
                            'bet_increased': bet_increased
                        })
                        
                        if continued:
                            streak_data[streak_type]['continue_count'] += 1
                        if bet_increased:
                            streak_data[streak_type]['bet_increase_count'] += 1
                        streak_data[streak_type]['bet_changes'].append(bet_change)

if __name__ == "__main__":
    calculated_stats, paper_values = analyze_streak_patterns()