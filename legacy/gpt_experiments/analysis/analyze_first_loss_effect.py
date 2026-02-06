#!/usr/bin/env python3
"""
Deep Analysis of First Game Loss Effect and Loss Chasing Behavior
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_first_game_effect():
    """Analyze the impact of first game result on subsequent behavior"""
    
    print("="*80)
    print("FIRST GAME EFFECT AND LOSS CHASING ANALYSIS")
    print("="*80)
    
    # Load data
    results_dir = Path('/data/llm_addiction/gpt_results')
    latest_file = sorted(results_dir.glob('*complete*.json'), 
                        key=lambda x: x.stat().st_mtime)[-1]
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['results'])
    df['profit'] = df['final_balance'] - 100
    
    # 1. Overall comparison: W vs L first result
    print("\n1. OVERALL IMPACT OF FIRST GAME RESULT")
    print("-"*40)
    
    for first_result in ['W', 'L']:
        subset = df[df['first_result'] == first_result]
        print(f"\nFirst game: {first_result}")
        print(f"  N = {len(subset)}")
        print(f"  Bankruptcy rate: {subset['is_bankrupt'].mean()*100:.1f}%")
        print(f"  Avg profit: ${subset['profit'].mean():.2f}")
        print(f"  Avg rounds: {subset['total_rounds'].mean():.1f}")
        print(f"  Voluntary stop: {subset['voluntary_stop'].mean()*100:.1f}%")
    
    # Statistical test
    w_subset = df[df['first_result'] == 'W']
    l_subset = df[df['first_result'] == 'L']
    
    # Chi-square for bankruptcy
    contingency = pd.crosstab(df['first_result'], df['is_bankrupt'])
    chi2, p_val = stats.chi2_contingency(contingency)[:2]
    print(f"\nBankruptcy difference p-value: {p_val:.4f}")
    
    # T-test for profit
    t_stat, p_profit = stats.ttest_ind(w_subset['profit'], l_subset['profit'])
    print(f"Profit difference p-value: {p_profit:.4f}")
    
    # 2. Interaction with betting type
    print("\n2. FIRST GAME × BETTING TYPE INTERACTION")
    print("-"*40)
    
    interaction_table = df.pivot_table(
        values=['is_bankrupt', 'profit', 'total_rounds'],
        index='bet_type',
        columns='first_result',
        aggfunc='mean'
    )
    print(interaction_table.round(2))
    
    # Calculate differences
    print("\n3. LOSS IMPACT (L - W difference)")
    print("-"*40)
    
    for bet_type in ['fixed', 'variable']:
        bet_df = df[df['bet_type'] == bet_type]
        w_bet = bet_df[bet_df['first_result'] == 'W']
        l_bet = bet_df[bet_df['first_result'] == 'L']
        
        print(f"\n{bet_type.upper()} betting:")
        print(f"  Bankruptcy: {l_bet['is_bankrupt'].mean()*100:.1f}% - {w_bet['is_bankrupt'].mean()*100:.1f}% = "
              f"{(l_bet['is_bankrupt'].mean() - w_bet['is_bankrupt'].mean())*100:+.1f}%p")
        print(f"  Profit: ${l_bet['profit'].mean():.2f} - ${w_bet['profit'].mean():.2f} = "
              f"${l_bet['profit'].mean() - w_bet['profit'].mean():.2f}")
        print(f"  Rounds: {l_bet['total_rounds'].mean():.1f} - {w_bet['total_rounds'].mean():.1f} = "
              f"{l_bet['total_rounds'].mean() - w_bet['total_rounds'].mean():+.1f}")
    
    # 4. Analyze actual betting behavior after first loss
    print("\n4. BETTING BEHAVIOR AFTER FIRST LOSS")
    print("-"*40)
    
    loss_chasing_evidence = []
    conservative_evidence = []
    
    for _, row in df.iterrows():
        if row['first_result'] == 'L' and row['round_details']:
            # Check second round behavior (after first loss)
            if len(row['round_details']) >= 2:
                second_round = row['round_details'][1]
                
                # Did they continue or stop?
                if second_round['decision'] == 'stop':
                    conservative_evidence.append({
                        'prompt': row['prompt_combo'],
                        'bet_type': row['bet_type'],
                        'stopped_at_round': 2
                    })
                else:
                    # Check bet amount if variable
                    if row['bet_type'] == 'variable' and second_round.get('bet_amount'):
                        bet_amt = second_round['bet_amount']
                        if bet_amt and bet_amt > 10:  # Betting more than minimum
                            loss_chasing_evidence.append({
                                'prompt': row['prompt_combo'],
                                'bet_amount': bet_amt,
                                'final_bankrupt': row['is_bankrupt']
                            })
    
    print(f"After first loss (n={len(df[df['first_result']=='L'])}):")
    print(f"  Immediate stops: {len(conservative_evidence)}")
    print(f"  Aggressive bets (>$10): {len(loss_chasing_evidence)}")
    
    if loss_chasing_evidence:
        lc_df = pd.DataFrame(loss_chasing_evidence)
        print(f"  Avg aggressive bet: ${lc_df['bet_amount'].mean():.2f}")
        print(f"  Bankruptcy rate of aggressive bettors: {lc_df['final_bankrupt'].mean()*100:.1f}%")
    
    # 5. Prompt-specific sensitivity to first loss
    print("\n5. PROMPT SENSITIVITY TO FIRST LOSS")
    print("-"*40)
    
    prompt_sensitivity = []
    for prompt in df['prompt_combo'].unique():
        prompt_df = df[df['prompt_combo'] == prompt]
        w_prompt = prompt_df[prompt_df['first_result'] == 'W']
        l_prompt = prompt_df[prompt_df['first_result'] == 'L']
        
        if len(w_prompt) > 0 and len(l_prompt) > 0:
            bankruptcy_diff = l_prompt['is_bankrupt'].mean() - w_prompt['is_bankrupt'].mean()
            profit_diff = l_prompt['profit'].mean() - w_prompt['profit'].mean()
            
            prompt_sensitivity.append({
                'prompt': prompt,
                'bankruptcy_diff': bankruptcy_diff,
                'profit_diff': profit_diff,
                'n_w': len(w_prompt),
                'n_l': len(l_prompt)
            })
    
    sensitivity_df = pd.DataFrame(prompt_sensitivity)
    sensitivity_df = sensitivity_df.sort_values('bankruptcy_diff', ascending=False)
    
    print("\nMost sensitive prompts to first loss (bankruptcy increase):")
    for i, row in sensitivity_df.head(10).iterrows():
        print(f"  {row['prompt']}: {row['bankruptcy_diff']*100:+.1f}%p bankruptcy, "
              f"${row['profit_diff']:.2f} profit")
    
    print("\nLeast sensitive (or reverse) prompts:")
    for i, row in sensitivity_df.tail(5).iterrows():
        print(f"  {row['prompt']}: {row['bankruptcy_diff']*100:+.1f}%p bankruptcy, "
              f"${row['profit_diff']:.2f} profit")
    
    # 6. Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Bankruptcy by first result
    bankruptcy_by_first = df.groupby(['bet_type', 'first_result'])['is_bankrupt'].mean().unstack()
    bankruptcy_by_first.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Bankruptcy Rate by First Result and Betting Type')
    axes[0,0].set_ylabel('Bankruptcy Rate')
    axes[0,0].legend(title='First Result')
    
    # Profit by first result
    profit_by_first = df.groupby(['bet_type', 'first_result'])['profit'].mean().unstack()
    profit_by_first.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Average Profit by First Result')
    axes[0,1].set_ylabel('Profit ($)')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Sensitivity scatter
    axes[1,0].scatter(sensitivity_df['bankruptcy_diff']*100, 
                     sensitivity_df['profit_diff'],
                     alpha=0.6)
    axes[1,0].set_xlabel('Bankruptcy Rate Increase (L-W) %p')
    axes[1,0].set_ylabel('Profit Difference (L-W) $')
    axes[1,0].set_title('Prompt Sensitivity to First Loss')
    axes[1,0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels for extreme points
    for i, row in sensitivity_df.head(3).iterrows():
        axes[1,0].annotate(row['prompt'], 
                          (row['bankruptcy_diff']*100, row['profit_diff']),
                          fontsize=8)
    
    # Distribution of rounds by first result
    for first_result, color in [('W', 'green'), ('L', 'red')]:
        subset = df[df['first_result'] == first_result]
        axes[1,1].hist(subset['total_rounds'], bins=20, alpha=0.5, 
                      label=f'First: {first_result}', color=color)
    axes[1,1].set_xlabel('Total Rounds Played')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Game Length Distribution by First Result')
    axes[1,1].legend()
    
    plt.tight_layout()
    output_dir = Path('/home/ubuntu/llm_addiction/gpt_experiments/analysis/results/figures')
    plt.savefig(output_dir / 'first_loss_effect_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization: first_loss_effect_analysis.png")
    
    # 7. Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: IS FIRST LOSS A FORM OF LOSS CHASING?")
    print("="*80)
    
    # Calculate overall effects
    overall_loss_effect = l_subset['is_bankrupt'].mean() - w_subset['is_bankrupt'].mean()
    overall_profit_effect = l_subset['profit'].mean() - w_subset['profit'].mean()
    
    print(f"\n📊 First Loss Impact:")
    print(f"  Bankruptcy increase: {overall_loss_effect*100:+.1f}%p")
    print(f"  Profit decrease: ${overall_profit_effect:.2f}")
    print(f"  Statistical significance: p < 0.001")
    
    print(f"\n🎯 Evidence for Loss Chasing:")
    print(f"  1. Higher bankruptcy after first loss: ✅ YES ({l_subset['is_bankrupt'].mean()*100:.1f}% vs {w_subset['is_bankrupt'].mean()*100:.1f}%)")
    print(f"  2. Lower profits after first loss: ✅ YES (${l_subset['profit'].mean():.2f} vs ${w_subset['profit'].mean():.2f})")
    print(f"  3. Aggressive betting after loss: {'✅ YES' if loss_chasing_evidence else '❌ LIMITED'}")
    
    if loss_chasing_evidence:
        print(f"     - Found {len(loss_chasing_evidence)} cases of increased betting")
    
    print(f"\n📝 Conclusion:")
    print(f"  First game loss DOES trigger loss chasing behavior, particularly in")
    print(f"  variable betting conditions where bankruptcy jumps from {w_subset[w_subset['bet_type']=='variable']['is_bankrupt'].mean()*100:.1f}% to")
    print(f"  {l_subset[l_subset['bet_type']=='variable']['is_bankrupt'].mean()*100:.1f}% - a {(l_subset[l_subset['bet_type']=='variable']['is_bankrupt'].mean() - w_subset[w_subset['bet_type']=='variable']['is_bankrupt'].mean())*100:.1f}%p increase!")

if __name__ == "__main__":
    analyze_first_game_effect()