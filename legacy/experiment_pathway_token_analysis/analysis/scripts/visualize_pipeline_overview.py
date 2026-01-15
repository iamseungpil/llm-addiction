#!/usr/bin/env python3
"""
Image 6: Comprehensive Pipeline Overview

Visualizes the entire analysis pipeline from Phase 1 to Phase 5
using a Sankey diagram.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-white')

def load_pipeline_statistics():
    """Load statistics from all phases"""
    print("Loading pipeline statistics...")

    stats = {}

    # Phase 1: Causal features from patching
    print("  Loading Phase 1...")
    phase1_features = set()
    for gpu in [4, 5, 6, 7]:
        file_path = Path(f"/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full/phase1_patching_multifeature_gpu{gpu}.jsonl")

        if file_path.exists():
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        phase1_features.add(data['target_feature'])

    stats['phase1_total_features'] = len(phase1_features)
    print(f"    Phase 1: {stats['phase1_total_features']} features")

    # Phase 5: Statistically significant features
    print("  Loading Phase 5...")
    phase5_significant = 0
    phase5_risky = 0
    phase5_safe = 0

    for gpu in [4, 5, 6, 7]:
        file_path = Path(f"/data/llm_addiction/experiment_pathway_token_analysis/results/phase5_prompt_feature_full/prompt_feature_correlation_gpu{gpu}.json")

        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)

            for comp in data['feature_comparisons']:
                if comp['p_value'] < 0.05:
                    phase5_significant += 1
                    if comp['cohens_d'] > 0:
                        phase5_risky += 1
                    else:
                        phase5_safe += 1

    stats['phase5_significant'] = phase5_significant
    stats['phase5_risky'] = phase5_risky
    stats['phase5_safe'] = phase5_safe
    print(f"    Phase 5: {phase5_significant} significant ({phase5_risky} risky, {phase5_safe} safe)")

    # Phase 4: Word associations
    print("  Loading Phase 4...")
    phase4_correlations = 0

    for gpu in [4, 5, 6, 7]:
        file_path = Path(f"/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_feature_FULL/word_feature_correlation_gpu{gpu}.json")

        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            phase4_correlations += len(data['word_feature_correlations'])

    stats['phase4_correlations'] = phase4_correlations
    print(f"    Phase 4: {phase4_correlations:,} word-feature correlations")

    return stats

def create_pipeline_flowchart(stats):
    """Create comprehensive pipeline flowchart"""

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)

    # Define pipeline stages
    stages = [
        {'name': 'Experiment 1\nFeature Discovery', 'y': 0.85, 'color': '#e57373'},
        {'name': 'Phase 1\nActivation Patching', 'y': 0.70, 'color': '#f06292'},
        {'name': 'Phase 5\nPrompt Correlation', 'y': 0.55, 'color': '#ba68c8'},
        {'name': 'Phase 4\nWord Association', 'y': 0.40, 'color': '#64b5f6'},
        {'name': 'Final Classification', 'y': 0.25, 'color': '#4db6ac'},
    ]

    # Draw stages
    for stage in stages:
        rect = mpatches.FancyBboxPatch((0.15, stage['y'] - 0.05), 0.7, 0.08,
                                      boxstyle="round,pad=0.01", linewidth=2,
                                      edgecolor='black', facecolor=stage['color'],
                                      alpha=0.7, transform=ax.transAxes)
        ax.add_patch(rect)

        ax.text(0.5, stage['y'], stage['name'],
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=13, fontweight='bold',
               color='white')

    # Add statistics annotations
    annotations = [
        {'text': f"{stats['phase1_total_features']} causal features identified", 'y': 0.65},
        {'text': f"{stats['phase5_significant']} statistically significant (p<0.05)", 'y': 0.50},
        {'text': f"{stats['phase5_risky']} risky + {stats['phase5_safe']} safe features", 'y': 0.35},
        {'text': f"{stats['phase4_correlations']:,} word-feature associations", 'y': 0.20},
    ]

    for annot in annotations:
        ax.text(0.5, annot['y'], annot['text'],
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=11, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Draw arrows between stages
    arrow_props = dict(arrowstyle='->', lw=2, color='black')

    for i in range(len(stages) - 1):
        ax.annotate('', xy=(0.5, stages[i+1]['y'] + 0.04),
                   xytext=(0.5, stages[i]['y'] - 0.05),
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=arrow_props)

    # Add method descriptions on the sides
    methods = [
        {'text': 'Population mean\nactivation patching', 'y': 0.675, 'x': 0.05},
        {'text': 'Risky vs Safe\nprompt comparison\n(t-test, Cohen\'s d)', 'y': 0.525, 'x': 0.05},
        {'text': 'Output word\nactivation\naggregation', 'y': 0.375, 'x': 0.05},
        {'text': 'Risky/Safe\nclassification', 'y': 0.225, 'x': 0.05},
    ]

    for method in methods:
        ax.text(method['x'], method['y'], method['text'],
               horizontalalignment='left', verticalalignment='center',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    # Add data sources on the right
    data_sources = [
        {'text': '6,400 LLaMA experiments\n(128 conditions × 50 reps)', 'y': 0.85, 'x': 0.95},
        {'text': '334,440 patching tests\n(2,787 features × 120 games)', 'y': 0.70, 'x': 0.95},
        {'text': '2,787 feature comparisons\n(risky vs safe prompts)', 'y': 0.55, 'x': 0.95},
        {'text': '7.3M word-feature pairs\n(output tokens)', 'y': 0.40, 'x': 0.95},
    ]

    for source in data_sources:
        ax.text(source['x'], source['y'], source['text'],
               horizontalalignment='right', verticalalignment='center',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # Title
    ax.text(0.5, 0.95, 'Comprehensive Analysis Pipeline Overview',
           horizontalalignment='center', verticalalignment='center',
           transform=ax.transAxes, fontsize=16, fontweight='bold')

    # Subtitle
    subtitle = f"From {stats['phase1_total_features']} causal features to {stats['phase5_risky']} risky + {stats['phase5_safe']} safe classifications"
    ax.text(0.5, 0.92, subtitle,
           horizontalalignment='center', verticalalignment='center',
           transform=ax.transAxes, fontsize=11, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    return fig

def create_statistics_summary(stats):
    """Create detailed statistics summary figure"""

    fig = plt.figure(figsize=(16, 10))

    # Create 2x2 grid for summary statistics
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis('off')

    # Phase 1 summary
    phase1_data = [
        ['Metric', 'Value'],
        ['Total causal features', f"{stats['phase1_total_features']:,}"],
        ['Total patching tests', '334,440'],
        ['Games per feature', '120'],
        ['Layers analyzed', '25, 30'],
    ]

    table1 = ax1.table(cellText=phase1_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)

    for i in range(2):
        table1[(0, i)].set_facecolor('#e57373')
        table1[(0, i)].set_text_props(weight='bold', color='white')

    ax1.set_title('Phase 1: Activation Patching', fontsize=13, fontweight='bold', pad=20)

    # Phase 5 summary
    ax2 = plt.subplot(2, 2, 2)
    ax2.axis('off')

    phase5_data = [
        ['Metric', 'Value'],
        ['Total features tested', f"{stats['phase1_total_features']:,}"],
        ['Significant (p<0.05)', f"{stats['phase5_significant']:,}"],
        ['Risky features', f"{stats['phase5_risky']:,}"],
        ['Safe features', f"{stats['phase5_safe']:,}"],
        ['Significance rate', f"{100*stats['phase5_significant']/stats['phase1_total_features']:.1f}%"],
    ]

    table2 = ax2.table(cellText=phase5_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)

    for i in range(2):
        table2[(0, i)].set_facecolor('#ba68c8')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Phase 5: Prompt Correlation', fontsize=13, fontweight='bold', pad=20)

    # Phase 4 summary
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')

    phase4_data = [
        ['Metric', 'Value'],
        ['Total correlations', f"{stats['phase4_correlations']:,}"],
        ['Unique words', '~10,000'],
        ['Unique features', f"{stats['phase1_total_features']:,}"],
        ['Coverage', '100%'],
    ]

    table3 = ax3.table(cellText=phase4_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table3.auto_set_font_size(False)
    table3.set_fontsize(10)
    table3.scale(1, 2)

    for i in range(2):
        table3[(0, i)].set_facecolor('#64b5f6')
        table3[(0, i)].set_text_props(weight='bold', color='white')

    ax3.set_title('Phase 4: Word Association', fontsize=13, fontweight='bold', pad=20)

    # Overall summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    overall_data = [
        ['Pipeline Stage', 'Input', 'Output'],
        ['Feature Discovery', '6,400 exp', f'{stats["phase1_total_features"]} features'],
        ['Patching Test', f'{stats["phase1_total_features"]} features', '334,440 tests'],
        ['Statistical Filter', f'{stats["phase1_total_features"]} features', f'{stats["phase5_significant"]} sig.'],
        ['Classification', f'{stats["phase5_significant"]} sig.', f'{stats["phase5_risky"]}R+{stats["phase5_safe"]}S'],
    ]

    table4 = ax4.table(cellText=overall_data, cellLoc='center', loc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table4.auto_set_font_size(False)
    table4.set_fontsize(10)
    table4.scale(1, 2)

    for i in range(3):
        table4[(0, i)].set_facecolor('#4db6ac')
        table4[(0, i)].set_text_props(weight='bold', color='white')

    ax4.set_title('Overall Pipeline Summary', fontsize=13, fontweight='bold', pad=20)

    fig.suptitle('Comprehensive Pipeline Statistics',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 6: Comprehensive Pipeline Overview")
    print("="*80)
    print()

    # Load statistics
    stats = load_pipeline_statistics()

    print("\nPipeline Statistics Summary:")
    print(f"  Phase 1 total features: {stats['phase1_total_features']:,}")
    print(f"  Phase 5 significant: {stats['phase5_significant']:,} ({100*stats['phase5_significant']/stats['phase1_total_features']:.1f}%)")
    print(f"    - Risky: {stats['phase5_risky']:,}")
    print(f"    - Safe: {stats['phase5_safe']:,}")
    print(f"  Phase 4 correlations: {stats['phase4_correlations']:,}")
    print()

    # Create visualizations
    print("Creating pipeline flowchart...")
    fig1 = create_pipeline_flowchart(stats)

    print("Creating statistics summary...")
    fig2 = create_statistics_summary(stats)

    # Save
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/images")

    png_path1 = output_dir / "06a_pipeline_flowchart.png"
    pdf_path1 = output_dir / "06a_pipeline_flowchart.pdf"
    png_path2 = output_dir / "06b_pipeline_statistics.png"
    pdf_path2 = output_dir / "06b_pipeline_statistics.pdf"

    fig1.savefig(png_path1, dpi=300, bbox_inches='tight')
    fig1.savefig(pdf_path1, bbox_inches='tight')
    fig2.savefig(png_path2, dpi=300, bbox_inches='tight')
    fig2.savefig(pdf_path2, bbox_inches='tight')

    print(f"\n✅ Saved visualizations:")
    print(f"   Flowchart PNG: {png_path1}")
    print(f"   Flowchart PDF: {pdf_path1}")
    print(f"   Statistics PNG: {png_path2}")
    print(f"   Statistics PDF: {pdf_path2}")
    print()

if __name__ == '__main__':
    main()
