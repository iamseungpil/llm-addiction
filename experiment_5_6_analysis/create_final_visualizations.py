#!/usr/bin/env python3
"""
Create final comprehensive visualizations for Experiment 5 & 6 analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_layer_effects_plot(results_dir: Path):
    """Create comprehensive layer-wise effects visualization"""

    # Load data
    df = pd.read_csv(results_dir / "exp5_all_features.csv")

    # Calculate layer-wise statistics
    layer_stats = df.groupby('layer').agg({
        'delta_bankruptcy_safe': ['mean', 'std', 'count'],
        'delta_bankruptcy_risky': ['mean', 'std'],
        'delta_balance_safe': ['mean', 'std'],
        'delta_balance_risky': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    layer_stats.columns = ['layer',
                          'safe_br_mean', 'safe_br_std', 'count',
                          'risky_br_mean', 'risky_br_std',
                          'safe_bal_mean', 'safe_bal_std',
                          'risky_bal_mean', 'risky_bal_std']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean bankruptcy rate change by layer
    ax1 = axes[0, 0]
    x = layer_stats['layer']
    ax1.plot(x, layer_stats['safe_br_mean'] * 100, 'o-', label='Safe Mean',
             color='green', linewidth=2, markersize=8)
    ax1.plot(x, layer_stats['risky_br_mean'] * 100, 's-', label='Risky Mean',
             color='red', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Layer', fontsize=12, weight='bold')
    ax1.set_ylabel('Mean Δ Bankruptcy Rate (%)', fontsize=12, weight='bold')
    ax1.set_title('Layer-Wise Mean Bankruptcy Rate Change', fontsize=14, weight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x)

    # Highlight Layer 27 (protective)
    protective_idx = layer_stats[layer_stats['layer'] == 27].index[0]
    ax1.scatter([27], [layer_stats.loc[protective_idx, 'safe_br_mean'] * 100],
               s=200, facecolors='none', edgecolors='blue', linewidth=3,
               label='Protective Layer', zorder=5)

    # Plot 2: Feature count and error bars
    ax2 = axes[0, 1]
    ax2.bar(x, layer_stats['count'], alpha=0.6, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Layer', fontsize=12, weight='bold')
    ax2.set_ylabel('Number of Features', fontsize=12, weight='bold')
    ax2.set_title('Features Tested per Layer', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(x)

    # Add counts on bars
    for i, (layer, count) in enumerate(zip(layer_stats['layer'], layer_stats['count'])):
        ax2.text(layer, count + 2, str(int(count)), ha='center', fontsize=10, weight='bold')

    # Plot 3: Standard deviation (variability)
    ax3 = axes[1, 0]
    ax3.plot(x, layer_stats['safe_br_std'] * 100, 'o-', label='Safe Mean STD',
             color='green', linewidth=2, markersize=8)
    ax3.plot(x, layer_stats['risky_br_std'] * 100, 's-', label='Risky Mean STD',
             color='red', linewidth=2, markersize=8)
    ax3.set_xlabel('Layer', fontsize=12, weight='bold')
    ax3.set_ylabel('Std Dev of Δ Bankruptcy Rate (%)', fontsize=12, weight='bold')
    ax3.set_title('Layer-Wise Feature Effect Variability', fontsize=14, weight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x)

    # Plot 4: Balance change by layer
    ax4 = axes[1, 1]
    ax4.plot(x, layer_stats['safe_bal_mean'], 'o-', label='Safe Mean',
             color='green', linewidth=2, markersize=8)
    ax4.plot(x, layer_stats['risky_bal_mean'], 's-', label='Risky Mean',
             color='red', linewidth=2, markersize=8)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Layer', fontsize=12, weight='bold')
    ax4.set_ylabel('Mean Δ Final Balance ($)', fontsize=12, weight='bold')
    ax4.set_title('Layer-Wise Mean Balance Change', fontsize=14, weight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(x)

    plt.suptitle('Experiment 5: Layer-Wise Mean Value Patching Effects',
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()

    output_path = results_dir / "layer_wise_comprehensive_effects.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved layer-wise effects plot to {output_path}")

def create_harmful_distribution_plot(results_dir: Path):
    """Create distribution plot of harmful features across layers"""

    # Load harmful features
    harmful_both = pd.read_csv(results_dir / "exp5_harmful_both.csv")
    protective_safe = pd.read_csv(results_dir / "exp5_protective_safe.csv")

    # Count by layer
    harmful_counts = harmful_both['layer'].value_counts().sort_index()
    protective_counts = protective_safe['layer'].value_counts().sort_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    layers = sorted(harmful_both['layer'].unique())
    x = np.arange(len(layers))
    width = 0.35

    harmful_values = [harmful_counts.get(layer, 0) for layer in layers]
    protective_values = [protective_counts.get(layer, 0) for layer in layers]

    bars1 = ax.bar(x - width/2, harmful_values, width, label='Harmful (Both)',
                   color='salmon', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, protective_values, width, label='Protective (Safe)',
                   color='lightgreen', edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Layer', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Features', fontsize=12, weight='bold')
    ax.set_title('Distribution of Harmful vs Protective Features by Layer',
                 fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    output_path = results_dir / "harmful_protective_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {output_path}")

def create_top_features_comparison(results_dir: Path):
    """Create comparison of top harmful vs protective features"""

    # Load data
    harmful = pd.read_csv(results_dir / "exp5_harmful_both.csv")
    protective = pd.read_csv(results_dir / "exp5_protective_safe.csv")

    # Get top 10 of each
    harmful_top = harmful.nlargest(10, 'delta_bankruptcy_safe')
    protective_top = protective.nsmallest(10, 'delta_bankruptcy_safe')

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot harmful features
    harmful_labels = [f"L{int(row['layer'])}-{int(row['feature_id'])}"
                     for _, row in harmful_top.iterrows()]
    harmful_values = harmful_top['delta_bankruptcy_safe'].values * 100

    ax1.barh(range(len(harmful_labels)), harmful_values, color='salmon',
            edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(harmful_labels)))
    ax1.set_yticklabels(harmful_labels, fontsize=9)
    ax1.set_xlabel('Δ Bankruptcy Rate (%)', fontsize=12, weight='bold')
    ax1.set_title('Top 10 Most Harmful Features', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Add value labels
    for i, v in enumerate(harmful_values):
        ax1.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9, weight='bold')

    # Plot protective features
    protective_labels = [f"L{int(row['layer'])}-{int(row['feature_id'])}"
                        for _, row in protective_top.iterrows()]
    protective_values = protective_top['delta_bankruptcy_safe'].values * 100

    ax2.barh(range(len(protective_labels)), protective_values, color='lightgreen',
            edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(protective_labels)))
    ax2.set_yticklabels(protective_labels, fontsize=9)
    ax2.set_xlabel('Δ Bankruptcy Rate (%)', fontsize=12, weight='bold')
    ax2.set_title('Top 10 Most Protective Features', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # Add value labels
    for i, v in enumerate(protective_values):
        ax2.text(v - 0.5, i, f'{v:.1f}%', va='center', ha='right',
                fontsize=9, weight='bold')

    plt.suptitle('Feature Impact Comparison: Harmful vs Protective',
                 fontsize=16, weight='bold')
    plt.tight_layout()

    output_path = results_dir / "top_features_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved top features comparison to {output_path}")

def main():
    results_dir = Path("/home/ubuntu/llm_addiction/experiment_5_6_analysis/results")

    print("="*80)
    print("CREATING FINAL VISUALIZATIONS")
    print("="*80)

    print("\n1. Layer-wise comprehensive effects plot...")
    create_layer_effects_plot(results_dir)

    print("\n2. Harmful vs protective distribution plot...")
    create_harmful_distribution_plot(results_dir)

    print("\n3. Top features comparison plot...")
    create_top_features_comparison(results_dir)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nSaved to: {results_dir}")

if __name__ == "__main__":
    main()
