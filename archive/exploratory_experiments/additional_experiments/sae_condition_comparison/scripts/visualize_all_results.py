#!/usr/bin/env python3
"""
Comprehensive Visualization for ALL Prompt Analyses

Generates visualizations for:
1. Component Analysis (G/M/R/W/P)
2. Complexity Analysis (0-5)
3. Individual Combo Analysis (32 combos)

Usage:
    python visualize_all_results.py --model llama
    python visualize_all_results.py --model both
"""

import os
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import pandas as pd

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_latest_json(directory: Path, pattern: str) -> Optional[dict]:
    """Load the most recent JSON file matching pattern"""
    files = sorted(directory.glob(pattern))
    if not files:
        return None

    latest_file = files[-1]
    print(f"Loading: {latest_file.name}")

    with open(latest_file, 'r') as f:
        return json.load(f)


def visualize_complexity_trend(
    results: dict,
    model: str,
    output_dir: Path
):
    """
    Create line plot showing complexity trend.

    X-axis: Complexity level (0-5)
    Y-axis: Mean activation or effect size
    """
    if not results or 'top_features' not in results:
        print(f"No complexity results for {model}")
        return

    top_features = results['top_features'][:20]  # Top 20 features

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Complexity means for top features
    ax = axes[0, 0]
    for i, feat in enumerate(top_features[:10]):
        comp_means = feat.get('complexity_means', {})
        levels = range(6)
        means = [comp_means.get(f'level_{l}', 0) for l in levels]

        ax.plot(levels, means, marker='o', label=f"L{feat['layer']}-{feat['feature_id']}", alpha=0.7)

    ax.set_xlabel('Complexity Level (# components)')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Top 10 Features: Complexity Trend', fontweight='bold')
    ax.set_xticks(range(6))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Interaction effect size distribution
    ax = axes[0, 1]
    interaction_etas = [f['interaction_eta'] for f in top_features]
    ax.hist(interaction_etas, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Interaction η² (Complexity × Outcome)')
    ax.set_ylabel('Count')
    ax.set_title('Interaction Effect Size Distribution', fontweight='bold')
    ax.axvline(0.14, color='red', linestyle='--', label='Large effect', alpha=0.7)
    ax.axvline(0.06, color='orange', linestyle='--', label='Medium effect', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Layer distribution
    ax = axes[1, 0]
    layers = [f['layer'] for f in top_features]
    layer_counts = pd.Series(layers).value_counts().sort_index()
    ax.bar(layer_counts.index, layer_counts.values, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count (top features)')
    ax.set_title('Layer Distribution of Significant Features', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary = results.get('summary', {})
    summary_text = f"""
    Complexity Analysis Summary ({model.upper()})

    Total features analyzed: {summary.get('total_features_analyzed', 0):,}
    FDR significant: {summary.get('fdr_significant_count', 0):,}
    With min η² (≥0.01): {summary.get('significant_with_min_eta', 0):,}
    Max interaction η²: {summary.get('max_interaction_eta', 0):.4f}

    Sample sizes by complexity:
      Level 0 (BASE): 100 games
      Level 1 (1 comp): 500 games
      Level 2 (2 comps): 1,000 games
      Level 3 (3 comps): 1,000 games
      Level 4 (4 comps): 500 games
      Level 5 (5 comps): 100 games
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace')

    plt.suptitle(
        f'Prompt Complexity Analysis Results ({model.upper()})',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / f'complexity_analysis_{model}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def visualize_combo_comparison(
    results: dict,
    model: str,
    output_dir: Path
):
    """
    Create visualizations comparing all 32 prompt combos.
    """
    if not results or 'combo_results' not in results:
        print(f"No combo results for {model}")
        return

    combo_results = results['combo_results']
    combo_stats = results.get('combo_stats', {})

    # Extract data
    combo_names = []
    bankruptcy_rates = []
    n_significant_features = []

    for combo, data in combo_results.items():
        combo_names.append(combo)
        stats = data.get('stats', combo_stats.get(combo, {}))
        bankruptcy_rates.append(stats.get('bankruptcy_rate', 0))
        n_significant_features.append(data.get('significant_with_min_d', 0))

    # Sort by bankruptcy rate
    sorted_indices = np.argsort(bankruptcy_rates)
    combo_names = [combo_names[i] for i in sorted_indices]
    bankruptcy_rates = [bankruptcy_rates[i] for i in sorted_indices]
    n_significant_features = [n_significant_features[i] for i in sorted_indices]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Bankruptcy rate by combo
    ax = axes[0]
    bars = ax.barh(range(len(combo_names)), bankruptcy_rates, edgecolor='black')

    # Color code by rate
    for i, (bar, rate) in enumerate(zip(bars, bankruptcy_rates)):
        if rate > 0.15:
            bar.set_color('darkred')
        elif rate > 0.08:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')

    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=8)
    ax.set_xlabel('Bankruptcy Rate', fontsize=12)
    ax.set_title('Bankruptcy Rate by Prompt Combination', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    # Add vertical lines for reference
    ax.axvline(0.05, color='green', linestyle='--', alpha=0.5, linewidth=0.8, label='Low risk')
    ax.axvline(0.15, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label='High risk')
    ax.legend()

    # Plot 2: Number of significant features
    ax = axes[1]
    ax.barh(range(len(combo_names)), n_significant_features, color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=8)
    ax.set_xlabel('Number of Significant Features (|d| ≥ 0.3)', fontsize=12)
    ax.set_title('Discriminative Features by Prompt Combination', fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle(
        f'Individual Prompt Combo Analysis ({model.upper()})\n'
        f'{len(combo_names)} combinations analyzed',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / f'combo_comparison_{model}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_comprehensive_summary(
    component_results: Dict[str, dict],
    complexity_results: Optional[dict],
    combo_results: Optional[dict],
    model: str,
    output_dir: Path
):
    """Create comprehensive summary table for all analyses"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Table 1: Component Analysis Summary
    ax = axes[0]
    ax.axis('tight')
    ax.axis('off')

    if component_results and any(component_results.values()):
        comp_data = []
        for comp in ['G', 'M', 'R', 'W', 'P']:
            if comp in component_results and component_results[comp]:
                summary = component_results[comp].get('summary', {})
                comp_data.append([
                    f"{comp}",
                    summary.get('total_features_analyzed', 0),
                    summary.get('fdr_significant_count', 0),
                    summary.get('significant_with_min_eta', 0),
                    f"{summary.get('max_interaction_eta', 0):.4f}"
                ])

        df_comp = pd.DataFrame(
            comp_data,
            columns=['Component', 'Total Features', 'FDR Sig', 'With Min η²', 'Max η²']
        )

        table1 = ax.table(
            cellText=df_comp.values,
            colLabels=df_comp.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.2, 0.15, 0.15, 0.15]
        )
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1, 2)

        for i in range(len(df_comp.columns)):
            table1[(0, i)].set_facecolor('#4CAF50')
            table1[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Component Analysis Summary', fontsize=13, fontweight='bold', pad=20)

    # Table 2: Complexity Analysis Summary
    ax = axes[1]
    ax.axis('tight')
    ax.axis('off')

    if complexity_results and 'summary' in complexity_results:
        summary = complexity_results['summary']
        comp_data = [[
            'Complexity (0-5)',
            summary.get('total_features_analyzed', 0),
            summary.get('fdr_significant_count', 0),
            summary.get('significant_with_min_eta', 0),
            f"{summary.get('max_interaction_eta', 0):.4f}"
        ]]

        df_complex = pd.DataFrame(
            comp_data,
            columns=['Analysis', 'Total Features', 'FDR Sig', 'With Min η²', 'Max η²']
        )

        table2 = ax.table(
            cellText=df_complex.values,
            colLabels=df_complex.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.2, 0.15, 0.15, 0.15]
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 2)

        for i in range(len(df_complex.columns)):
            table2[(0, i)].set_facecolor('#2196F3')
            table2[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Complexity Analysis Summary', fontsize=13, fontweight='bold', pad=20)

    # Table 3: Combo Analysis Summary
    ax = axes[2]
    ax.axis('tight')
    ax.axis('off')

    if combo_results and 'summary' in combo_results:
        summary = combo_results['summary']
        combo_data = [[
            'Individual Combos',
            summary.get('n_combos_analyzed', 0),
            summary.get('total_combos', 0),
            summary.get('combos_with_significant_features', 0),
            'Exploratory'
        ]]

        df_combo = pd.DataFrame(
            combo_data,
            columns=['Analysis', 'Analyzed', 'Total', 'With Sig Features', 'Note']
        )

        table3 = ax.table(
            cellText=df_combo.values,
            colLabels=df_combo.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.2, 0.15, 0.15, 0.2, 0.15]
        )
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1, 2)

        for i in range(len(df_combo.columns)):
            table3[(0, i)].set_facecolor('#FF9800')
            table3[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title('Individual Combo Analysis Summary', fontsize=13, fontweight='bold', pad=20)

    plt.suptitle(
        f'Comprehensive Prompt Analysis Summary ({model.upper()})',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / f'comprehensive_summary_{model}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Visualization for All Prompt Analyses')
    parser.add_argument(
        '--model', type=str, default='both', choices=['llama', 'gemma', 'both'],
        help='Which model results to visualize'
    )
    parser.add_argument(
        '--config', type=str,
        default=str(Path(__file__).parent.parent / 'configs' / 'prompt_analysis_config.yaml'),
        help='Path to config file'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup paths
    results_dir = Path(config['data']['output_dir'])
    figure_dir = Path(config['visualization']['figure_dir'])
    figure_dir.mkdir(parents=True, exist_ok=True)

    components = config['prompt_component_analysis']['components']

    print("=" * 70)
    print("Generating Comprehensive Visualizations for ALL Prompt Analyses")
    print("=" * 70)

    models = ['llama', 'gemma'] if args.model == 'both' else [args.model]

    for model in models:
        print(f"\n\nProcessing {model.upper()} results...")
        print("-" * 70)

        # Load Component Analysis results
        component_results = {}
        comp_dir = results_dir / 'prompt_component'
        for comp in components:
            result = load_latest_json(comp_dir, f'{comp}_{model}_*.json')
            if result:
                component_results[comp] = result

        # Load Complexity Analysis results
        complex_dir = results_dir / 'prompt_complexity'
        complexity_results = load_latest_json(complex_dir, f'complexity_{model}_*.json')

        # Load Combo Analysis results
        combo_dir = results_dir / 'prompt_combo'
        combo_results = load_latest_json(combo_dir, f'combo_explorer_{model}_*.json')

        # Generate visualizations
        if complexity_results:
            print("\nGenerating Complexity visualizations...")
            visualize_complexity_trend(complexity_results, model, figure_dir)

        if combo_results:
            print("\nGenerating Combo comparison...")
            visualize_combo_comparison(combo_results, model, figure_dir)

        if component_results or complexity_results or combo_results:
            print("\nGenerating comprehensive summary...")
            create_comprehensive_summary(
                component_results,
                complexity_results,
                combo_results,
                model,
                figure_dir
            )

    print("\n" + "=" * 70)
    print("Comprehensive Visualization Complete")
    print("=" * 70)
    print(f"Figures saved to: {figure_dir}")


if __name__ == '__main__':
    main()
