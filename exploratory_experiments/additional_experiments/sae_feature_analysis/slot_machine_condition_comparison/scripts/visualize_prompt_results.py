#!/usr/bin/env python3
"""
Visualization Script for Prompt Component Analysis Results

Generates comprehensive visualizations including:
1. Component × Layer heatmaps (interaction effect size)
2. Bar plots showing top features per component
3. Comparison plots between LLaMA and Gemma
4. Summary tables

Usage:
    python visualize_prompt_results.py --model llama
    python visualize_prompt_results.py --model both
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


def load_latest_results(results_dir: Path, component: str, model: str) -> Optional[dict]:
    """Load the most recent result file for a component"""
    pattern = f"{component}_{model}_*.json"
    files = sorted(results_dir.glob(pattern))

    if not files:
        print(f"Warning: No results found for {component}_{model}")
        return None

    latest_file = files[-1]
    print(f"Loading: {latest_file.name}")

    with open(latest_file, 'r') as f:
        return json.load(f)


def create_component_layer_heatmap(
    results_dict: Dict[str, dict],
    model: str,
    output_dir: Path,
    metric: str = 'interaction_eta',
    top_n: int = 20
):
    """
    Create heatmap showing Component × Layer effect sizes.

    Args:
        results_dict: Dict mapping component -> results
        model: 'llama' or 'gemma'
        output_dir: Where to save figure
        metric: Which metric to visualize
        top_n: Top N features per component to include
    """
    components = sorted(results_dict.keys())
    component_names = {
        'G': 'Goal-setting',
        'M': 'Maximize',
        'R': 'Hidden patterns',
        'W': 'Win multiplier',
        'P': 'Win rate'
    }

    # Collect top features per component
    all_data = []

    for comp in components:
        if comp not in results_dict or results_dict[comp] is None:
            continue

        top_features = results_dict[comp].get('top_features', [])[:top_n]

        for feat in top_features:
            all_data.append({
                'component': component_names.get(comp, comp),
                'layer': feat['layer'],
                'feature_id': feat['feature_id'],
                'interaction_eta': feat['interaction_eta'],
                'component_eta': feat['component_eta'],
                'outcome_eta': feat['outcome_eta'],
                'interaction_p_fdr': feat['interaction_p_fdr']
            })

    if not all_data:
        print(f"No data to visualize for {model}")
        return

    df = pd.DataFrame(all_data)

    # Create pivot table: Component × Layer
    # Aggregate by taking mean effect size
    pivot = df.pivot_table(
        values='interaction_eta',
        index='component',
        columns='layer',
        aggfunc='mean',
        fill_value=0
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot heatmap
    sns.heatmap(
        pivot,
        annot=False,
        fmt='.3f',
        cmap='RdYlBu_r',
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Mean Interaction Effect Size (η²)'},
        ax=ax
    )

    ax.set_title(
        f'Prompt Component × Layer Interaction Effects ({model.upper()})\n'
        f'Top {top_n} features per component',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Prompt Component', fontsize=12)

    plt.tight_layout()

    # Save
    output_file = output_dir / f'component_layer_heatmap_{model}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_component_barplot(
    results_dict: Dict[str, dict],
    model: str,
    output_dir: Path,
    n_features: int = 10
):
    """
    Create bar plot showing top features per component.

    Args:
        results_dict: Dict mapping component -> results
        model: 'llama' or 'gemma'
        output_dir: Where to save figure
        n_features: Number of top features to show per component
    """
    components = ['G', 'M', 'R', 'W', 'P']
    component_names = {
        'G': 'Goal-setting',
        'M': 'Maximize',
        'R': 'Hidden patterns',
        'W': 'Win multiplier',
        'P': 'Win rate'
    }

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

    for idx, comp in enumerate(components):
        ax = axes[idx]

        if comp not in results_dict or results_dict[comp] is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(component_names[comp])
            continue

        top_features = results_dict[comp].get('top_features', [])[:n_features]

        if not top_features:
            ax.text(0.5, 0.5, 'No significant\nfeatures', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(component_names[comp])
            continue

        # Extract data
        labels = [f"L{f['layer']}-{f['feature_id']}" for f in top_features]
        values = [f['interaction_eta'] for f in top_features]

        # Plot
        bars = ax.barh(range(len(labels)), values, color='skyblue', edgecolor='black')

        # Color code by effect size
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.5:
                bar.set_color('darkred')
            elif val > 0.2:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Interaction η²', fontsize=10)
        ax.set_title(component_names[comp], fontweight='bold', fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.axvline(0.14, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)  # Medium effect
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle(
        f'Top {n_features} Features per Component ({model.upper()})\n'
        f'Interaction Effect Size (Component × Outcome)',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / f'component_barplot_{model}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_table(
    results_dict: Dict[str, dict],
    model: str,
    output_dir: Path
):
    """Create summary statistics table"""
    components = ['G', 'M', 'R', 'W', 'P']
    component_names = {
        'G': 'Goal-setting',
        'M': 'Maximize',
        'R': 'Hidden patterns',
        'W': 'Win multiplier',
        'P': 'Win rate'
    }

    summary_data = []

    for comp in components:
        if comp not in results_dict or results_dict[comp] is None:
            continue

        summary = results_dict[comp].get('summary', {})

        summary_data.append({
            'Component': f"{comp} ({component_names[comp]})",
            'Total Features': summary.get('total_features_analyzed', 0),
            'FDR Significant': summary.get('fdr_significant_count', 0),
            'With Min η²': summary.get('significant_with_min_eta', 0),
            'Max η²': f"{summary.get('max_interaction_eta', 0):.4f}"
        })

    if not summary_data:
        print(f"No summary data for {model}")
        return

    df = pd.DataFrame(summary_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title(
        f'Summary Statistics: Prompt Component Analysis ({model.upper()})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()

    # Save
    output_file = output_dir / f'component_summary_table_{model}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # Also save as CSV
    csv_file = output_dir / f'component_summary_{model}.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")


def create_model_comparison(
    llama_results: Dict[str, dict],
    gemma_results: Dict[str, dict],
    output_dir: Path
):
    """Create comparison plot between LLaMA and Gemma"""
    components = ['G', 'M', 'R', 'W', 'P']
    component_names = {
        'G': 'Goal',
        'M': 'Max',
        'R': 'Risk',
        'W': 'Win×',
        'P': 'Prob'
    }

    # Collect data
    llama_counts = []
    gemma_counts = []
    labels = []

    for comp in components:
        labels.append(component_names[comp])

        llama_count = 0
        if comp in llama_results and llama_results[comp]:
            llama_count = llama_results[comp]['summary'].get('significant_with_min_eta', 0)
        llama_counts.append(llama_count)

        gemma_count = 0
        if comp in gemma_results and gemma_results[comp]:
            gemma_count = gemma_results[comp]['summary'].get('significant_with_min_eta', 0)
        gemma_counts.append(gemma_count)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, llama_counts, width, label='LLaMA-3.1-8B', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, gemma_counts, width, label='Gemma-2-9B', color='coral', edgecolor='black')

    ax.set_xlabel('Prompt Component', fontsize=12)
    ax.set_ylabel('Number of Significant Features (η² ≥ 0.01)', fontsize=12)
    ax.set_title('Model Comparison: Significant Component × Outcome Interactions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save
    output_file = output_dir / 'model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Prompt Component Analysis Results')
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
    results_dir = Path(config['data']['output_dir']) / 'prompt_component'
    figure_dir = Path(config['visualization']['figure_dir'])
    figure_dir.mkdir(parents=True, exist_ok=True)

    components = config['prompt_component_analysis']['components']

    print("="*70)
    print("Generating Visualizations for Prompt Component Analysis")
    print("="*70)

    # Process LLaMA
    if args.model in ['llama', 'both']:
        print("\nProcessing LLaMA results...")
        llama_results = {}
        for comp in components:
            llama_results[comp] = load_latest_results(results_dir, comp, 'llama')

        if any(llama_results.values()):
            create_component_layer_heatmap(llama_results, 'llama', figure_dir, top_n=20)
            create_component_barplot(llama_results, 'llama', figure_dir, n_features=10)
            create_summary_table(llama_results, 'llama', figure_dir)
        else:
            print("No LLaMA results found")

    # Process Gemma
    if args.model in ['gemma', 'both']:
        print("\nProcessing Gemma results...")
        gemma_results = {}
        for comp in components:
            gemma_results[comp] = load_latest_results(results_dir, comp, 'gemma')

        if any(gemma_results.values()):
            create_component_layer_heatmap(gemma_results, 'gemma', figure_dir, top_n=20)
            create_component_barplot(gemma_results, 'gemma', figure_dir, n_features=10)
            create_summary_table(gemma_results, 'gemma', figure_dir)
        else:
            print("No Gemma results found")

    # Create comparison if both models processed
    if args.model == 'both':
        if any(llama_results.values()) and any(gemma_results.values()):
            print("\nCreating model comparison...")
            create_model_comparison(llama_results, gemma_results, figure_dir)

    print("\n" + "="*70)
    print("Visualization Complete")
    print("="*70)
    print(f"Figures saved to: {figure_dir}")


if __name__ == '__main__':
    main()
