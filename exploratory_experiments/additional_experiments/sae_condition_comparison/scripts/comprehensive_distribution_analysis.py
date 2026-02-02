#!/usr/bin/env python3
"""
Comprehensive Distribution Analysis for Two-Way ANOVA Results

Analyzes the FULL distribution of effect sizes across all features
to validate whether conclusions based on top features are representative.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def load_two_way_anova_results(model_type='llama'):
    """Load Two-Way ANOVA results"""
    results_dir = Path('results')
    anova_files = sorted(results_dir.glob(f'two_way_anova_{model_type}_*.json'))

    if not anova_files:
        raise FileNotFoundError(f"No Two-Way ANOVA results found for {model_type}")

    with open(anova_files[-1]) as f:
        return json.load(f)


def comprehensive_analysis(data, model_type='llama'):
    """
    Perform comprehensive statistical analysis on full distribution
    """
    all_results = data['all_results']

    # Extract effect sizes
    bet_type_eta = np.array([r['bet_type_effect']['eta_squared'] for r in all_results])
    outcome_eta = np.array([r['outcome_effect']['eta_squared'] for r in all_results])
    interaction_eta = np.array([r['interaction_effect']['eta_squared'] for r in all_results])

    # Extract p-values
    bet_type_p = np.array([r['bet_type_effect']['p_value'] for r in all_results])
    outcome_p = np.array([r['outcome_effect']['p_value'] for r in all_results])
    interaction_p = np.array([r['interaction_effect']['p_value'] for r in all_results])

    # Extract layers
    layers = np.array([r['layer'] for r in all_results])

    print("\n" + "="*70)
    print("COMPREHENSIVE DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"Model: {model_type.upper()}")
    print(f"Total features: {len(all_results)}")
    print(f"Layers: {np.unique(layers)}")

    # 1. EFFECT SIZE DISTRIBUTION
    print("\n" + "="*70)
    print("1. EFFECT SIZE DISTRIBUTION (η²)")
    print("="*70)

    for name, eta in [('Bet Type', bet_type_eta),
                      ('Outcome', outcome_eta),
                      ('Interaction', interaction_eta)]:
        print(f"\n{name}:")
        print(f"  Mean:   {np.mean(eta):.6f}")
        print(f"  Median: {np.median(eta):.6f}")
        print(f"  Std:    {np.std(eta):.6f}")
        print(f"  Min:    {np.min(eta):.6f}")
        print(f"  Max:    {np.max(eta):.6f}")
        print(f"  Q1:     {np.percentile(eta, 25):.6f}")
        print(f"  Q3:     {np.percentile(eta, 75):.6f}")

    # 2. EFFECT SIZE THRESHOLDS (Cohen's conventions)
    print("\n" + "="*70)
    print("2. EFFECT SIZE THRESHOLDS (Cohen 1988)")
    print("="*70)
    print("Small: η² ≥ 0.01, Medium: η² ≥ 0.06, Large: η² ≥ 0.14")

    thresholds = {
        'Small (≥0.01)': 0.01,
        'Medium (≥0.06)': 0.06,
        'Large (≥0.14)': 0.14
    }

    for name, eta in [('Bet Type', bet_type_eta),
                      ('Outcome', outcome_eta),
                      ('Interaction', interaction_eta)]:
        print(f"\n{name}:")
        for threshold_name, threshold_val in thresholds.items():
            count = np.sum(eta >= threshold_val)
            pct = count / len(eta) * 100
            print(f"  {threshold_name}: {count:5d} ({pct:5.1f}%)")

    # 3. DOMINANCE ANALYSIS
    print("\n" + "="*70)
    print("3. DOMINANCE ANALYSIS")
    print("="*70)

    bet_dominant = np.sum(bet_type_eta > outcome_eta)
    outcome_dominant = np.sum(outcome_eta > bet_type_eta)
    equal = np.sum(bet_type_eta == outcome_eta)

    print(f"Bet Type dominant:  {bet_dominant:5d} ({bet_dominant/len(all_results)*100:.1f}%)")
    print(f"Outcome dominant:   {outcome_dominant:5d} ({outcome_dominant/len(all_results)*100:.1f}%)")
    print(f"Equal:              {equal:5d} ({equal/len(all_results)*100:.1f}%)")

    # Statistical test: Paired comparison
    stat, pval = stats.wilcoxon(bet_type_eta, outcome_eta, alternative='greater')
    print(f"\nWilcoxon signed-rank test (Bet Type > Outcome):")
    print(f"  Test statistic: {stat}")
    print(f"  P-value: {pval:.2e}")
    print(f"  Result: {'SIGNIFICANT' if pval < 0.001 else 'NOT SIGNIFICANT'}")

    # Effect size of the difference
    diff = bet_type_eta - outcome_eta
    print(f"\nEffect size difference (Bet Type - Outcome):")
    print(f"  Mean difference: {np.mean(diff):.6f}")
    print(f"  Median difference: {np.median(diff):.6f}")
    print(f"  Cohen's d: {np.mean(diff) / np.std(diff):.3f}")

    # 4. SIGNIFICANCE RATES
    print("\n" + "="*70)
    print("4. SIGNIFICANCE RATES (p < 0.05)")
    print("="*70)

    for name, p_vals in [('Bet Type', bet_type_p),
                         ('Outcome', outcome_p),
                         ('Interaction', interaction_p)]:
        sig_count = np.sum(p_vals < 0.05)
        sig_pct = sig_count / len(p_vals) * 100
        print(f"{name}: {sig_count:5d} / {len(p_vals)} ({sig_pct:.1f}%)")

    # 5. LAYER-WISE BREAKDOWN
    print("\n" + "="*70)
    print("5. LAYER-WISE BREAKDOWN")
    print("="*70)

    for layer in np.unique(layers):
        layer_mask = layers == layer
        layer_bet = bet_type_eta[layer_mask]
        layer_out = outcome_eta[layer_mask]
        layer_int = interaction_eta[layer_mask]

        bet_dom = np.sum(layer_bet > layer_out)
        out_dom = np.sum(layer_out > layer_bet)

        print(f"\nLayer {layer} (n={np.sum(layer_mask)}):")
        print(f"  Bet Type dominant:  {bet_dom:4d} ({bet_dom/np.sum(layer_mask)*100:.1f}%)")
        print(f"  Outcome dominant:   {out_dom:4d} ({out_dom/np.sum(layer_mask)*100:.1f}%)")
        print(f"  Mean η² - Bet Type: {np.mean(layer_bet):.6f}")
        print(f"  Mean η² - Outcome:  {np.mean(layer_out):.6f}")
        print(f"  Mean η² - Interaction: {np.mean(layer_int):.6f}")

    # 6. TOP FEATURES REPRESENTATIVENESS
    print("\n" + "="*70)
    print("6. TOP FEATURES REPRESENTATIVENESS")
    print("="*70)

    # Sort by total effect
    total_eta = bet_type_eta + outcome_eta + interaction_eta
    sorted_indices = np.argsort(-total_eta)

    for n_top in [10, 20, 50, 100]:
        top_indices = sorted_indices[:n_top]
        top_bet = bet_type_eta[top_indices]
        top_out = outcome_eta[top_indices]

        top_bet_dom = np.sum(top_bet > top_out) / n_top * 100

        print(f"\nTop {n_top} features:")
        print(f"  Bet Type dominant: {top_bet_dom:.1f}%")
        print(f"  Mean η² (Bet):     {np.mean(top_bet):.6f}")
        print(f"  Mean η² (Outcome): {np.mean(top_out):.6f}")
        print(f"  Ratio:             {np.mean(top_bet)/np.mean(top_out):.2f}x")

    print("\n" + "="*70)

    return {
        'bet_type_eta': bet_type_eta,
        'outcome_eta': outcome_eta,
        'interaction_eta': interaction_eta,
        'bet_type_p': bet_type_p,
        'outcome_p': outcome_p,
        'interaction_p': interaction_p,
        'layers': layers
    }


def create_distribution_plots(stats_dict, model_type='llama'):
    """
    Create comprehensive distribution visualization plots
    """
    bet_eta = stats_dict['bet_type_eta']
    out_eta = stats_dict['outcome_eta']
    int_eta = stats_dict['interaction_eta']
    layers = stats_dict['layers']

    fig = plt.figure(figsize=(16, 12))

    # 1. Histogram comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(bet_eta, bins=50, alpha=0.6, label='Bet Type', color='blue', density=True)
    ax1.hist(out_eta, bins=50, alpha=0.6, label='Outcome', color='red', density=True)
    ax1.axvline(np.mean(bet_eta), color='blue', linestyle='--', linewidth=2, label=f'Mean Bet={np.mean(bet_eta):.4f}')
    ax1.axvline(np.mean(out_eta), color='red', linestyle='--', linewidth=2, label=f'Mean Out={np.mean(out_eta):.4f}')
    ax1.set_xlabel('Effect Size (η²)', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Effect Size Distribution (All Features)', fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, 0.3)

    # 2. Log-scale histogram (to see small effects)
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(np.log10(bet_eta + 1e-6), bins=50, alpha=0.6, label='Bet Type', color='blue', density=True)
    ax2.hist(np.log10(out_eta + 1e-6), bins=50, alpha=0.6, label='Outcome', color='red', density=True)
    ax2.set_xlabel('log10(Effect Size)', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Log-Scale Distribution', fontweight='bold')
    ax2.legend()

    # 3. Scatter plot: Bet Type vs Outcome
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(out_eta, bet_eta, alpha=0.3, s=10, c='gray')
    max_val = max(np.max(bet_eta), np.max(out_eta))
    ax3.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Equal Effect')
    ax3.set_xlabel('Outcome Effect (η²)', fontweight='bold')
    ax3.set_ylabel('Bet Type Effect (η²)', fontweight='bold')
    ax3.set_title(f'Effect Comparison (n={len(bet_eta)})', fontweight='bold')
    ax3.legend()
    ax3.set_xlim(0, max_val)
    ax3.set_ylim(0, max_val)
    ax3.text(0.95, 0.05, f'Above diagonal: {np.sum(bet_eta > out_eta)/len(bet_eta)*100:.1f}%',
             transform=ax3.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. Violin plot by layer
    ax4 = plt.subplot(2, 3, 4)
    unique_layers = np.unique(layers)
    data_for_violin = []
    labels_for_violin = []
    for layer in unique_layers:
        mask = layers == layer
        data_for_violin.append(bet_eta[mask])
        data_for_violin.append(out_eta[mask])
        labels_for_violin.extend([f'L{layer}\nBet', f'L{layer}\nOut'])

    parts = ax4.violinplot(data_for_violin, positions=range(len(data_for_violin)),
                           showmeans=True, showmedians=True)
    ax4.set_xticks(range(len(labels_for_violin)))
    ax4.set_xticklabels(labels_for_violin, fontsize=9)
    ax4.set_ylabel('Effect Size (η²)', fontweight='bold')
    ax4.set_title('Layer-wise Distribution', fontweight='bold')
    ax4.set_ylim(0, 0.3)

    # 5. Cumulative distribution
    ax5 = plt.subplot(2, 3, 5)
    sorted_bet = np.sort(bet_eta)
    sorted_out = np.sort(out_eta)
    ax5.plot(sorted_bet, np.arange(len(sorted_bet)) / len(sorted_bet),
             label='Bet Type', color='blue', linewidth=2)
    ax5.plot(sorted_out, np.arange(len(sorted_out)) / len(sorted_out),
             label='Outcome', color='red', linewidth=2)
    ax5.axvline(0.01, color='gray', linestyle=':', label='Small (0.01)')
    ax5.axvline(0.06, color='gray', linestyle='--', label='Medium (0.06)')
    ax5.axvline(0.14, color='gray', linestyle='-', label='Large (0.14)')
    ax5.set_xlabel('Effect Size (η²)', fontweight='bold')
    ax5.set_ylabel('Cumulative Probability', fontweight='bold')
    ax5.set_title('Cumulative Distribution Function', fontweight='bold')
    ax5.legend()
    ax5.set_xlim(0, 0.3)
    ax5.grid(True, alpha=0.3)

    # 6. Effect size difference distribution
    ax6 = plt.subplot(2, 3, 6)
    diff = bet_eta - out_eta
    ax6.hist(diff, bins=50, alpha=0.7, color='purple', density=True)
    ax6.axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax6.axvline(np.mean(diff), color='red', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(diff):.4f}')
    ax6.axvline(np.median(diff), color='orange', linestyle='-', linewidth=2,
                label=f'Median={np.median(diff):.4f}')
    ax6.set_xlabel('Difference (Bet Type - Outcome)', fontweight='bold')
    ax6.set_ylabel('Density', fontweight='bold')
    ax6.set_title('Effect Size Difference Distribution', fontweight='bold')
    ax6.legend()

    plt.suptitle(f'Comprehensive Distribution Analysis ({model_type.upper()})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_file = f'results/comprehensive_distribution_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {save_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Comprehensive distribution analysis")
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gemma'])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE ANALYSIS FOR {args.model.upper()}")
    print(f"{'='*70}")

    # Load data
    data = load_two_way_anova_results(args.model)

    # Statistical analysis
    stats_dict = comprehensive_analysis(data, args.model)

    # Visualization
    create_distribution_plots(stats_dict, args.model)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
