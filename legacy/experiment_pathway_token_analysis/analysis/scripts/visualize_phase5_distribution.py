#!/usr/bin/env python3
"""
Image 1: Phase 5 - Risky vs Safe Feature Distribution

Visualizes the distribution of statistically significant features across layers,
showing which features are activated by risky vs safe prompts.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_phase5_data():
    """Load Phase 5 prompt-feature correlation data"""
    print("Loading Phase 5 data...")

    all_features = []

    for gpu in [4, 5, 6, 7]:
        file_path = Path(f"/data/llm_addiction/experiment_pathway_token_analysis/results/phase5_prompt_feature_full/prompt_feature_correlation_gpu{gpu}.json")

        with open(file_path, 'r') as f:
            data = json.load(f)

        for comp in data['feature_comparisons']:
            # Only include statistically significant features
            if comp['p_value'] < 0.05:
                feature = comp['feature']
                layer = int(feature.split('-')[0][1:])

                all_features.append({
                    'feature': feature,
                    'layer': layer,
                    'cohens_d': comp['cohens_d'],
                    'p_value': comp['p_value'],
                    'risky_mean': comp['risky_mean'],
                    'safe_mean': comp['safe_mean']
                })

    print(f"Loaded {len(all_features)} significant features")
    return all_features

def create_visualization(features):
    """Create comprehensive visualization"""

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 12))

    # Subplot 1: Scatter plot (layer vs Cohen's d)
    ax1 = plt.subplot(2, 2, 1)

    # Separate risky and safe features
    risky_features = [f for f in features if f['cohens_d'] > 0]
    safe_features = [f for f in features if f['cohens_d'] < 0]

    # Color by p-value significance
    def get_color(p_value):
        if p_value < 0.001:
            return '#d32f2f'  # Dark red - highly significant
        elif p_value < 0.01:
            return '#f57c00'  # Orange - significant
        else:
            return '#fbc02d'  # Yellow - marginally significant

    # Size by activation strength
    def get_size(risky_mean, safe_mean):
        return abs(risky_mean - safe_mean) * 100 + 20

    # Plot risky features
    for f in risky_features:
        ax1.scatter(f['layer'], f['cohens_d'],
                   c=get_color(f['p_value']),
                   s=get_size(f['risky_mean'], f['safe_mean']),
                   alpha=0.6, edgecolors='black', linewidths=0.5)

    # Plot safe features
    for f in safe_features:
        ax1.scatter(f['layer'], f['cohens_d'],
                   c=get_color(f['p_value']),
                   s=get_size(f['risky_mean'], f['safe_mean']),
                   alpha=0.6, edgecolors='black', linewidths=0.5,
                   marker='v')

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel("Cohen's d (Risky > 0, Safe < 0)", fontsize=12, fontweight='bold')
    ax1.set_title('Feature Distribution Across Layers', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Legend for p-values
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d32f2f', label='p < 0.001 (***)', alpha=0.6),
        Patch(facecolor='#f57c00', label='p < 0.01 (**)', alpha=0.6),
        Patch(facecolor='#fbc02d', label='p < 0.05 (*)', alpha=0.6)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Subplot 2: Layer-wise count distribution
    ax2 = plt.subplot(2, 2, 2)

    layer_counts = defaultdict(lambda: {'risky': 0, 'safe': 0})
    for f in features:
        if f['cohens_d'] > 0:
            layer_counts[f['layer']]['risky'] += 1
        else:
            layer_counts[f['layer']]['safe'] += 1

    layers = sorted(layer_counts.keys())
    risky_counts = [layer_counts[l]['risky'] for l in layers]
    safe_counts = [-layer_counts[l]['safe'] for l in layers]  # Negative for plotting

    ax2.barh(layers, risky_counts, color='#e57373', label='Risky Features', alpha=0.7)
    ax2.barh(layers, safe_counts, color='#64b5f6', label='Safe Features', alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Feature Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_title('Risky vs Safe Feature Count by Layer', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')

    # Subplot 3: Effect size distribution
    ax3 = plt.subplot(2, 2, 3)

    risky_cohens = [f['cohens_d'] for f in risky_features]
    safe_cohens = [abs(f['cohens_d']) for f in safe_features]

    ax3.hist(risky_cohens, bins=30, color='#e57373', alpha=0.7, label='Risky Features')
    ax3.hist(safe_cohens, bins=30, color='#64b5f6', alpha=0.7, label='Safe Features')
    ax3.set_xlabel("Absolute Cohen's d", fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Effect Size Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Top features table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # Get top 10 risky and safe features by effect size
    top_risky = sorted(risky_features, key=lambda x: x['cohens_d'], reverse=True)[:10]
    top_safe = sorted(safe_features, key=lambda x: x['cohens_d'])[:10]

    table_data = []
    table_data.append(['Type', 'Feature', 'Layer', "Cohen's d", 'p-value'])

    for f in top_risky[:5]:
        table_data.append([
            'Risky',
            f['feature'][:15],
            str(f['layer']),
            f"{f['cohens_d']:.3f}",
            f"{f['p_value']:.4f}"
        ])

    for f in top_safe[:5]:
        table_data.append([
            'Safe',
            f['feature'][:15],
            str(f['layer']),
            f"{f['cohens_d']:.3f}",
            f"{f['p_value']:.4f}"
        ])

    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.15, 0.25, 0.1, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#e0e0e0')
        table[(0, i)].set_text_props(weight='bold')

    ax4.set_title('Top 5 Risky and Safe Features', fontsize=14, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Phase 5: Risky vs Safe Feature Distribution Analysis\n' +
                f'Total Significant Features: {len(features)} (p < 0.05)',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 1: Phase 5 - Risky vs Safe Feature Distribution")
    print("="*80)
    print()

    # Load data
    features = load_phase5_data()

    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(features)

    # Save
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "01_phase5_risky_safe_distribution.png"
    pdf_path = output_dir / "01_phase5_risky_safe_distribution.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    print(f"\n✅ Saved visualization:")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    print()

    # Summary statistics
    risky_count = sum(1 for f in features if f['cohens_d'] > 0)
    safe_count = sum(1 for f in features if f['cohens_d'] < 0)

    print("="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"Total significant features: {len(features)}")
    print(f"Risky features (Cohen's d > 0): {risky_count} ({100*risky_count/len(features):.1f}%)")
    print(f"Safe features (Cohen's d < 0): {safe_count} ({100*safe_count/len(features):.1f}%)")
    print()

    # Layer distribution
    layer_dist = defaultdict(lambda: {'risky': 0, 'safe': 0})
    for f in features:
        if f['cohens_d'] > 0:
            layer_dist[f['layer']]['risky'] += 1
        else:
            layer_dist[f['layer']]['safe'] += 1

    print("Layer Distribution:")
    for layer in sorted(layer_dist.keys()):
        print(f"  Layer {layer:2d}: {layer_dist[layer]['risky']:3d} risky, {layer_dist[layer]['safe']:3d} safe")
    print()

if __name__ == '__main__':
    main()
