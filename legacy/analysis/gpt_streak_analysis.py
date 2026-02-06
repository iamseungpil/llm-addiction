#!/usr/bin/env python3
"""
Analyze winning and losing streak patterns in GPT gambling experiments
Focus on behavioral changes after streaks
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

def load_gpt_data():
    """Load corrected GPT experiment results"""
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        data = json.load(f)
    return data

def analyze_streak_patterns(data):
    """Analyze behavior patterns during winning and losing streaks"""
    
    print("="*80)
    print("STREAK PATTERN ANALYSIS - WINNING VS LOSING")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = data['results']
    
    # Initialize tracking variables
    streak_data = {
        'after_first_win': {'continue': 0, 'quit': 0, 'bet_changes': [], 'bankruptcies': 0},
        'after_first_loss': {'continue': 0, 'quit': 0, 'bet_changes': [], 'bankruptcies': 0},
        '2_wins': {'continue': 0, 'quit': 0, 'bet_changes': [], 'next_results': []},
        '3_wins': {'continue': 0, 'quit': 0, 'bet_changes': [], 'next_results': []},
        '2_losses': {'continue': 0, 'quit': 0, 'bet_changes': [], 'next_results': []},
        '3_losses': {'continue': 0, 'quit': 0, 'bet_changes': [], 'next_results': []},
        '5_losses': {'continue': 0, 'quit': 0, 'bet_changes': [], 'next_results': []}
    }
    
    # Analyze each experiment
    for exp in results:
        if 'game_history' not in exp or len(exp['game_history']) == 0:
            continue
        
        history = exp['game_history']
        bet_type = exp['bet_type']
        is_bankrupt = exp.get('is_bankrupt', False)
        
        # 1. After first game
        if len(history) >= 1:
            first_result = history[0]['result']
            if len(history) > 1:
                # Continued playing
                if first_result == 'W':
                    streak_data['after_first_win']['continue'] += 1
                    if bet_type == 'variable':
                        bet_change = (history[1]['bet'] - history[0]['bet']) / history[0]['bet'] * 100
                        streak_data['after_first_win']['bet_changes'].append(bet_change)
                    if is_bankrupt:
                        streak_data['after_first_win']['bankruptcies'] += 1
                else:  # Loss
                    streak_data['after_first_loss']['continue'] += 1
                    if bet_type == 'variable':
                        bet_change = (history[1]['bet'] - history[0]['bet']) / history[0]['bet'] * 100
                        streak_data['after_first_loss']['bet_changes'].append(bet_change)
                    if is_bankrupt:
                        streak_data['after_first_loss']['bankruptcies'] += 1
            else:
                # Quit after first game
                if first_result == 'W':
                    streak_data['after_first_win']['quit'] += 1
                else:
                    streak_data['after_first_loss']['quit'] += 1
        
        # 2. Check for consecutive wins
        for i in range(len(history)):
            # 2 consecutive wins
            if i >= 1 and all(h['result'] == 'W' for h in history[i-1:i+1]):
                if i+1 < len(history):
                    streak_data['2_wins']['continue'] += 1
                    streak_data['2_wins']['next_results'].append(history[i+1]['result'])
                    if bet_type == 'variable' and i > 0:
                        bet_change = (history[i+1]['bet'] - history[i]['bet']) / history[i]['bet'] * 100
                        streak_data['2_wins']['bet_changes'].append(bet_change)
                else:
                    streak_data['2_wins']['quit'] += 1
            
            # 3 consecutive wins
            if i >= 2 and all(h['result'] == 'W' for h in history[i-2:i+1]):
                if i+1 < len(history):
                    streak_data['3_wins']['continue'] += 1
                    streak_data['3_wins']['next_results'].append(history[i+1]['result'])
                    if bet_type == 'variable':
                        bet_change = (history[i+1]['bet'] - history[i]['bet']) / history[i]['bet'] * 100
                        streak_data['3_wins']['bet_changes'].append(bet_change)
                else:
                    streak_data['3_wins']['quit'] += 1
            
            # 2 consecutive losses
            if i >= 1 and all(h['result'] == 'L' for h in history[i-1:i+1]):
                if i+1 < len(history):
                    streak_data['2_losses']['continue'] += 1
                    streak_data['2_losses']['next_results'].append(history[i+1]['result'])
                    if bet_type == 'variable' and i > 0:
                        bet_change = (history[i+1]['bet'] - history[i]['bet']) / history[i]['bet'] * 100
                        streak_data['2_losses']['bet_changes'].append(bet_change)
                else:
                    streak_data['2_losses']['quit'] += 1
            
            # 3 consecutive losses
            if i >= 2 and all(h['result'] == 'L' for h in history[i-2:i+1]):
                if i+1 < len(history):
                    streak_data['3_losses']['continue'] += 1
                    streak_data['3_losses']['next_results'].append(history[i+1]['result'])
                    if bet_type == 'variable':
                        bet_change = (history[i+1]['bet'] - history[i]['bet']) / history[i]['bet'] * 100
                        streak_data['3_losses']['bet_changes'].append(bet_change)
                else:
                    streak_data['3_losses']['quit'] += 1
            
            # 5 consecutive losses
            if i >= 4 and all(h['result'] == 'L' for h in history[i-4:i+1]):
                if i+1 < len(history):
                    streak_data['5_losses']['continue'] += 1
                    streak_data['5_losses']['next_results'].append(history[i+1]['result'])
                    if bet_type == 'variable':
                        bet_change = (history[i+1]['bet'] - history[i]['bet']) / history[i]['bet'] * 100
                        streak_data['5_losses']['bet_changes'].append(bet_change)
                else:
                    streak_data['5_losses']['quit'] += 1
    
    return streak_data

def print_streak_analysis(streak_data):
    """Print detailed analysis of streak patterns"""
    
    print("\n1. FIRST GAME IMPACT")
    print("-"*60)
    
    # First win analysis
    total_first_win = streak_data['after_first_win']['continue'] + streak_data['after_first_win']['quit']
    if total_first_win > 0:
        continue_rate_win = streak_data['after_first_win']['continue'] / total_first_win * 100
        bankruptcy_rate_win = streak_data['after_first_win']['bankruptcies'] / total_first_win * 100
        avg_bet_change_win = np.mean(streak_data['after_first_win']['bet_changes']) if streak_data['after_first_win']['bet_changes'] else 0
        
        print(f"After First WIN (n={total_first_win}):")
        print(f"  Continue rate: {continue_rate_win:.1f}%")
        print(f"  Quit rate: {100-continue_rate_win:.1f}%")
        print(f"  Bankruptcy rate: {bankruptcy_rate_win:.1f}%")
        print(f"  Avg bet change: {avg_bet_change_win:+.1f}%")
    
    # First loss analysis
    total_first_loss = streak_data['after_first_loss']['continue'] + streak_data['after_first_loss']['quit']
    if total_first_loss > 0:
        continue_rate_loss = streak_data['after_first_loss']['continue'] / total_first_loss * 100
        bankruptcy_rate_loss = streak_data['after_first_loss']['bankruptcies'] / total_first_loss * 100
        avg_bet_change_loss = np.mean(streak_data['after_first_loss']['bet_changes']) if streak_data['after_first_loss']['bet_changes'] else 0
        
        print(f"\nAfter First LOSS (n={total_first_loss}):")
        print(f"  Continue rate: {continue_rate_loss:.1f}%")
        print(f"  Quit rate: {100-continue_rate_loss:.1f}%")
        print(f"  Bankruptcy rate: {bankruptcy_rate_loss:.1f}%")
        print(f"  Avg bet change: {avg_bet_change_loss:+.1f}%")
    
    print("\n2. WINNING STREAK BEHAVIOR")
    print("-"*60)
    
    # 2-win streak
    total_2wins = streak_data['2_wins']['continue'] + streak_data['2_wins']['quit']
    if total_2wins > 0:
        continue_rate = streak_data['2_wins']['continue'] / total_2wins * 100
        avg_bet_change = np.mean(streak_data['2_wins']['bet_changes']) if streak_data['2_wins']['bet_changes'] else 0
        next_win_rate = sum(1 for r in streak_data['2_wins']['next_results'] if r == 'W') / len(streak_data['2_wins']['next_results']) * 100 if streak_data['2_wins']['next_results'] else 0
        
        print(f"After 2 WINS (n={total_2wins}):")
        print(f"  Continue rate: {continue_rate:.1f}%")
        print(f"  Avg bet change: {avg_bet_change:+.1f}%")
        print(f"  Next game win rate: {next_win_rate:.1f}% (expected: 30%)")
    
    # 3-win streak
    total_3wins = streak_data['3_wins']['continue'] + streak_data['3_wins']['quit']
    if total_3wins > 0:
        continue_rate = streak_data['3_wins']['continue'] / total_3wins * 100
        avg_bet_change = np.mean(streak_data['3_wins']['bet_changes']) if streak_data['3_wins']['bet_changes'] else 0
        next_win_rate = sum(1 for r in streak_data['3_wins']['next_results'] if r == 'W') / len(streak_data['3_wins']['next_results']) * 100 if streak_data['3_wins']['next_results'] else 0
        
        print(f"\nAfter 3 WINS (n={total_3wins}):")
        print(f"  Continue rate: {continue_rate:.1f}%")
        print(f"  Avg bet change: {avg_bet_change:+.1f}%")
        print(f"  Next game win rate: {next_win_rate:.1f}% (expected: 30%)")
    
    print("\n3. LOSING STREAK BEHAVIOR")
    print("-"*60)
    
    # 2-loss streak
    total_2losses = streak_data['2_losses']['continue'] + streak_data['2_losses']['quit']
    if total_2losses > 0:
        continue_rate = streak_data['2_losses']['continue'] / total_2losses * 100
        avg_bet_change = np.mean(streak_data['2_losses']['bet_changes']) if streak_data['2_losses']['bet_changes'] else 0
        next_win_rate = sum(1 for r in streak_data['2_losses']['next_results'] if r == 'W') / len(streak_data['2_losses']['next_results']) * 100 if streak_data['2_losses']['next_results'] else 0
        
        print(f"After 2 LOSSES (n={total_2losses}):")
        print(f"  Continue rate: {continue_rate:.1f}%")
        print(f"  Avg bet change: {avg_bet_change:+.1f}%")
        print(f"  Next game win rate: {next_win_rate:.1f}% (expected: 30%)")
    
    # 3-loss streak
    total_3losses = streak_data['3_losses']['continue'] + streak_data['3_losses']['quit']
    if total_3losses > 0:
        continue_rate = streak_data['3_losses']['continue'] / total_3losses * 100
        avg_bet_change = np.mean(streak_data['3_losses']['bet_changes']) if streak_data['3_losses']['bet_changes'] else 0
        next_win_rate = sum(1 for r in streak_data['3_losses']['next_results'] if r == 'W') / len(streak_data['3_losses']['next_results']) * 100 if streak_data['3_losses']['next_results'] else 0
        
        print(f"\nAfter 3 LOSSES (n={total_3losses}):")
        print(f"  Continue rate: {continue_rate:.1f}%")
        print(f"  Avg bet change: {avg_bet_change:+.1f}%")
        print(f"  Next game win rate: {next_win_rate:.1f}% (expected: 30%)")
    
    # 5-loss streak
    total_5losses = streak_data['5_losses']['continue'] + streak_data['5_losses']['quit']
    if total_5losses > 0:
        continue_rate = streak_data['5_losses']['continue'] / total_5losses * 100
        avg_bet_change = np.mean(streak_data['5_losses']['bet_changes']) if streak_data['5_losses']['bet_changes'] else 0
        
        print(f"\nAfter 5 LOSSES (n={total_5losses}):")
        print(f"  Continue rate: {continue_rate:.1f}%")
        print(f"  Avg bet change: {avg_bet_change:+.1f}%")

def create_comparison_table(streak_data):
    """Create comparison table for winning vs losing streaks"""
    
    print("\n4. WINNING VS LOSING STREAK COMPARISON")
    print("-"*60)
    
    # Prepare comparison data
    comparison = []
    
    # 2 streaks
    total_2wins = streak_data['2_wins']['continue'] + streak_data['2_wins']['quit']
    total_2losses = streak_data['2_losses']['continue'] + streak_data['2_losses']['quit']
    
    if total_2wins > 0 and total_2losses > 0:
        comparison.append({
            'Streak': '2 consecutive',
            'Win_Continue': f"{streak_data['2_wins']['continue'] / total_2wins * 100:.1f}%",
            'Win_BetChange': f"{np.mean(streak_data['2_wins']['bet_changes']) if streak_data['2_wins']['bet_changes'] else 0:+.1f}%",
            'Loss_Continue': f"{streak_data['2_losses']['continue'] / total_2losses * 100:.1f}%",
            'Loss_BetChange': f"{np.mean(streak_data['2_losses']['bet_changes']) if streak_data['2_losses']['bet_changes'] else 0:+.1f}%"
        })
    
    # 3 streaks
    total_3wins = streak_data['3_wins']['continue'] + streak_data['3_wins']['quit']
    total_3losses = streak_data['3_losses']['continue'] + streak_data['3_losses']['quit']
    
    if total_3wins > 0 and total_3losses > 0:
        comparison.append({
            'Streak': '3 consecutive',
            'Win_Continue': f"{streak_data['3_wins']['continue'] / total_3wins * 100:.1f}%",
            'Win_BetChange': f"{np.mean(streak_data['3_wins']['bet_changes']) if streak_data['3_wins']['bet_changes'] else 0:+.1f}%",
            'Loss_Continue': f"{streak_data['3_losses']['continue'] / total_3losses * 100:.1f}%",
            'Loss_BetChange': f"{np.mean(streak_data['3_losses']['bet_changes']) if streak_data['3_losses']['bet_changes'] else 0:+.1f}%"
        })
    
    if comparison:
        df = pd.DataFrame(comparison)
        print("\nTable: Winning vs Losing Streak Behavior")
        print(df.to_string(index=False))
    
    # Hot hand vs gambler's fallacy analysis
    print("\n5. HOT HAND VS GAMBLER'S FALLACY")
    print("-"*60)
    
    # Calculate bet increases after wins vs losses
    win_bet_increases = [bc for bc in streak_data['2_wins']['bet_changes'] if bc > 0]
    loss_bet_increases = [bc for bc in streak_data['2_losses']['bet_changes'] if bc > 0]
    
    if win_bet_increases and loss_bet_increases:
        print(f"Bet increases after 2 wins: {len(win_bet_increases)}/{len(streak_data['2_wins']['bet_changes'])} ({len(win_bet_increases)/len(streak_data['2_wins']['bet_changes'])*100:.1f}%)")
        print(f"Bet increases after 2 losses: {len(loss_bet_increases)}/{len(streak_data['2_losses']['bet_changes'])} ({len(loss_bet_increases)/len(streak_data['2_losses']['bet_changes'])*100:.1f}%)")
        print(f"Average increase after wins: {np.mean(win_bet_increases):.1f}%")
        print(f"Average increase after losses: {np.mean(loss_bet_increases):.1f}%")
    
    # Statistical significance
    if streak_data['2_wins']['bet_changes'] and streak_data['2_losses']['bet_changes']:
        t_stat, p_value = stats.ttest_ind(streak_data['2_wins']['bet_changes'], 
                                          streak_data['2_losses']['bet_changes'])
        print(f"\nBet change difference (wins vs losses): p = {p_value:.4f}")
        if p_value < 0.05:
            print("  → Statistically significant difference in betting behavior")

def main():
    print("="*80)
    print("GPT WINNING AND LOSING STREAK ANALYSIS")
    print("="*80)
    
    # Load data
    data = load_gpt_data()
    print(f"\nTotal experiments: {len(data['results'])}")
    
    # Analyze streaks
    streak_data = analyze_streak_patterns(data)
    
    # Print analysis
    print_streak_analysis(streak_data)
    
    # Create comparison table
    create_comparison_table(streak_data)
    
    # Summary for paper
    print("\n" + "="*80)
    print("KEY FINDINGS FOR PAPER")
    print("="*80)
    
    # Calculate key metrics
    first_win_total = streak_data['after_first_win']['continue'] + streak_data['after_first_win']['quit']
    first_loss_total = streak_data['after_first_loss']['continue'] + streak_data['after_first_loss']['quit']
    
    if first_win_total > 0 and first_loss_total > 0:
        win_continue = streak_data['after_first_win']['continue'] / first_win_total * 100
        loss_continue = streak_data['after_first_loss']['continue'] / first_loss_total * 100
        
        print(f"1. First game impact:")
        print(f"   - After WIN: {win_continue:.1f}% continue playing")
        print(f"   - After LOSS: {loss_continue:.1f}% continue playing")
        print(f"   - Difference: {win_continue - loss_continue:+.1f}%p")
    
    print(f"\n2. Hot hand fallacy (after wins):")
    if streak_data['2_wins']['bet_changes']:
        print(f"   - Average bet change after 2 wins: {np.mean(streak_data['2_wins']['bet_changes']):+.1f}%")
    if streak_data['3_wins']['bet_changes']:
        print(f"   - Average bet change after 3 wins: {np.mean(streak_data['3_wins']['bet_changes']):+.1f}%")
    
    print(f"\n3. Loss chasing (after losses):")
    if streak_data['2_losses']['bet_changes']:
        print(f"   - Average bet change after 2 losses: {np.mean(streak_data['2_losses']['bet_changes']):+.1f}%")
    if streak_data['3_losses']['bet_changes']:
        print(f"   - Average bet change after 3 losses: {np.mean(streak_data['3_losses']['bet_changes']):+.1f}%")
    
    # Save results
    output_file = '/home/ubuntu/llm_addiction/analysis/gpt_streak_analysis.json'
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for key in streak_data:
            if 'bet_changes' in streak_data[key]:
                streak_data[key]['bet_changes'] = list(streak_data[key]['bet_changes'])
            if 'next_results' in streak_data[key]:
                streak_data[key]['next_results'] = list(streak_data[key]['next_results'])
        
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'streak_data': streak_data
        }, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()