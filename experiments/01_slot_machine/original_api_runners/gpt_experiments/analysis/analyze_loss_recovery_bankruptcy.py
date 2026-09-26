#!/usr/bin/env python3
"""
Analysis of Loss Recovery Attempts Leading to Bankruptcy in GPT Experiments
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_loss_recovery_bankruptcy():
    """Analyze cases where loss recovery attempts lead to bankruptcy"""
    
    print("="*80)
    print("LOSS RECOVERY → BANKRUPTCY ANALYSIS")
    print("="*80)
    
    # Load data
    results_dir = Path('/data/llm_addiction/gpt_results')
    latest_file = sorted(results_dir.glob('*complete*.json'), 
                        key=lambda x: x.stat().st_mtime)[-1]
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['results'])
    df['profit'] = df['final_balance'] - 100
    
    print(f"\nDataset: {len(df)} experiments")
    print(f"Goal amount: $200 (2x initial capital)")
    
    # 1. Analyze loss recovery patterns
    loss_recovery_cases = []
    
    for _, row in df.iterrows():
        if not row['round_details'] or len(row['round_details']) < 3:
            continue
            
        rounds = row['round_details']
        
        # Track consecutive losses followed by increased betting
        for i in range(1, len(rounds)-1):
            prev_round = rounds[i-1]
            curr_round = rounds[i]
            next_round = rounds[i+1]
            
            # Check for loss followed by continued play with higher bet
            if (curr_round.get('balance_before', 0) < prev_round.get('balance_before', 0) and  # Lost money
                next_round.get('decision') == 'continue'):  # Continued playing
                
                # Check if bet increased after loss (for variable betting)
                if (row['bet_type'] == 'variable' and 
                    next_round.get('bet_amount', 0) > curr_round.get('bet_amount', 0)):
                    
                    loss_recovery_cases.append({
                        'condition_id': row['condition_id'],
                        'prompt': row['prompt_combo'],
                        'bet_type': row['bet_type'],
                        'round': i+1,
                        'balance_before_loss': prev_round.get('balance_before', 0),
                        'balance_after_loss': curr_round.get('balance_before', 0),
                        'loss_amount': prev_round.get('balance_before', 0) - curr_round.get('balance_before', 0),
                        'recovery_bet': next_round.get('bet_amount', 0),
                        'original_bet': curr_round.get('bet_amount', 0),
                        'bet_increase': next_round.get('bet_amount', 0) - curr_round.get('bet_amount', 0),
                        'final_bankrupt': row['is_bankrupt'],
                        'final_balance': row['final_balance'],
                        'total_rounds': row['total_rounds']
                    })
    
    print(f"\n1. LOSS RECOVERY ATTEMPTS")
    print("-"*40)
    print(f"Found {len(loss_recovery_cases)} loss recovery attempts")
    
    if not loss_recovery_cases:
        print("No loss recovery patterns found")
        return
    
    recovery_df = pd.DataFrame(loss_recovery_cases)
    
    # 2. Success vs Bankruptcy in recovery attempts
    print(f"\n2. OUTCOMES OF RECOVERY ATTEMPTS")
    print("-"*40)
    
    bankruptcy_rate = recovery_df['final_bankrupt'].mean()
    print(f"Bankruptcy rate in recovery attempts: {bankruptcy_rate*100:.1f}%")
    print(f"Average bet increase: ${recovery_df['bet_increase'].mean():.2f}")
    print(f"Average recovery bet: ${recovery_df['recovery_bet'].mean():.2f}")
    
    # Compare to overall bankruptcy rate
    overall_bankruptcy = df['is_bankrupt'].mean()
    print(f"Overall bankruptcy rate: {overall_bankruptcy*100:.1f}%")
    print(f"Recovery attempt bankruptcy risk: {bankruptcy_rate/overall_bankruptcy:.1f}x higher")
    
    # 3. Analyze specific bankruptcy cases
    bankruptcy_recoveries = recovery_df[recovery_df['final_bankrupt'] == True]
    print(f"\n3. BANKRUPTCY CASES FROM RECOVERY ATTEMPTS")
    print("-"*40)
    print(f"Recovery attempts that ended in bankruptcy: {len(bankruptcy_recoveries)}")
    
    if len(bankruptcy_recoveries) > 0:
        print(f"Average loss they were trying to recover: ${bankruptcy_recoveries['loss_amount'].mean():.2f}")
        print(f"Average bet size in recovery attempt: ${bankruptcy_recoveries['recovery_bet'].mean():.2f}")
        print(f"Average bet increase: ${bankruptcy_recoveries['bet_increase'].mean():.2f}")
        
        # Show examples
        print(f"\nTop 5 biggest recovery bets that led to bankruptcy:")
        top_recoveries = bankruptcy_recoveries.nlargest(5, 'recovery_bet')
        for _, case in top_recoveries.iterrows():
            print(f"  Loss: ${case['loss_amount']:.0f} → Recovery bet: ${case['recovery_bet']:.0f} "
                  f"(+${case['bet_increase']:.0f}) → Bankrupt at round {case['total_rounds']}")
    
    # 4. Pattern by prompt type
    print(f"\n4. RECOVERY PATTERNS BY PROMPT")
    print("-"*40)
    
    prompt_recovery = recovery_df.groupby('prompt').agg({
        'final_bankrupt': ['count', 'mean'],
        'bet_increase': 'mean',
        'recovery_bet': 'mean'
    }).round(2)
    
    print("Most risky prompts for recovery attempts:")
    if len(prompt_recovery) > 0:
        risky_prompts = prompt_recovery.sort_values(('final_bankrupt', 'mean'), ascending=False).head(5)
        print(risky_prompts)
    
    # 5. Martingale-like behavior
    print(f"\n5. MARTINGALE-LIKE BEHAVIOR")
    print("-"*40)
    
    # Look for cases where bet doubled or more after loss
    martingale_cases = recovery_df[recovery_df['recovery_bet'] >= recovery_df['original_bet'] * 2]
    print(f"Doubling-or-more bets after loss: {len(martingale_cases)}")
    
    if len(martingale_cases) > 0:
        martingale_bankruptcy = martingale_cases['final_bankrupt'].mean()
        print(f"Bankruptcy rate for doubling bets: {martingale_bankruptcy*100:.1f}%")
        print(f"Average doubling bet size: ${martingale_cases['recovery_bet'].mean():.2f}")
        
        # Extreme cases
        extreme_cases = martingale_cases[martingale_cases['recovery_bet'] >= 50]
        print(f"Extreme recovery bets (≥$50): {len(extreme_cases)}")
        if len(extreme_cases) > 0:
            print(f"Extreme recovery bankruptcy rate: {extreme_cases['final_bankrupt'].mean()*100:.1f}%")
    
    # 6. Time to bankruptcy analysis
    print(f"\n6. TIME TO BANKRUPTCY AFTER RECOVERY ATTEMPTS")
    print("-"*40)
    
    quick_bankruptcies = bankruptcy_recoveries[bankruptcy_recoveries['total_rounds'] <= 5]
    print(f"Quick bankruptcies (≤5 rounds): {len(quick_bankruptcies)}")
    
    if len(bankruptcy_recoveries) > 0:
        print(f"Average rounds to bankruptcy after recovery attempt: {bankruptcy_recoveries['total_rounds'].mean():.1f}")
        print(f"Median rounds to bankruptcy: {bankruptcy_recoveries['total_rounds'].median():.1f}")
    
    # 7. Summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY: LOSS RECOVERY → BANKRUPTCY PATTERN")
    print("="*80)
    
    print(f"\n📊 Key Findings:")
    print(f"  • Recovery attempts found: {len(recovery_df)}")
    print(f"  • Recovery bankruptcy rate: {bankruptcy_rate*100:.1f}% vs overall {overall_bankruptcy*100:.1f}%")
    print(f"  • Risk multiplier: {bankruptcy_rate/overall_bankruptcy:.1f}x")
    print(f"  • Average recovery bet: ${recovery_df['recovery_bet'].mean():.2f}")
    
    if len(martingale_cases) > 0:
        print(f"\n🎲 Martingale-like behavior:")
        print(f"  • Doubling bets: {len(martingale_cases)} cases")
        print(f"  • Doubling bankruptcy rate: {martingale_bankruptcy*100:.1f}%")
    
    return {
        'total_recovery_attempts': len(recovery_df),
        'recovery_bankruptcy_rate': bankruptcy_rate,
        'overall_bankruptcy_rate': overall_bankruptcy,
        'risk_multiplier': bankruptcy_rate/overall_bankruptcy if overall_bankruptcy > 0 else 0,
        'avg_recovery_bet': recovery_df['recovery_bet'].mean(),
        'martingale_cases': len(martingale_cases) if len(martingale_cases) > 0 else 0
    }

if __name__ == "__main__":
    analyze_loss_recovery_bankruptcy()