#!/usr/bin/env python3
"""
Phase 2: Feature-Feature Correlation Analysis & Visualization
Ultra-deep analysis of 11,919 correlation files
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class Phase2Analyzer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.all_correlations = []
        self.feature_pairs = set()
        self.target_features = set()
        self.conditions = set()

    def load_data_sample(self, max_files=1000):
        """Load a representative sample of correlation files"""
        print("=== Phase 2 Data Loading ===")

        files = list(self.data_dir.glob("correlations_gpu*_*.jsonl"))
        print(f"Total files found: {len(files)}")

        # Sample evenly across GPUs and conditions
        files_by_gpu = defaultdict(list)
        for f in files:
            gpu_id = f.name.split('_')[1].replace('gpu', '')
            files_by_gpu[gpu_id].append(f)

        sampled_files = []
        per_gpu = max_files // len(files_by_gpu)
        for gpu_id, gpu_files in files_by_gpu.items():
            sampled = np.random.choice(gpu_files, min(per_gpu, len(gpu_files)), replace=False)
            sampled_files.extend(sampled)

        print(f"Sampling {len(sampled_files)} files for analysis...")

        # Load sampled files
        for i, file in enumerate(sampled_files, 1):
            if i % 100 == 0:
                print(f"  Loaded {i}/{len(sampled_files)} files...")

            with open(file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.all_correlations.append(data)
                        self.target_features.add(data['target_feature'])
                        self.conditions.add(data['condition'])
                        self.feature_pairs.add((data['feature_A'], data['feature_B']))

        print(f"\n✅ Loaded {len(self.all_correlations):,} correlations")
        print(f"   Unique target features: {len(self.target_features)}")
        print(f"   Unique conditions: {len(self.conditions)}")
        print(f"   Unique feature pairs: {len(self.feature_pairs)}")

    def ultra_deep_analysis(self):
        """Perform comprehensive statistical analysis"""
        print("\n=== Ultra-Deep Analysis ===")

        df = pd.DataFrame(self.all_correlations)

        # 1. Overall correlation distribution
        print("\n1️⃣ Correlation Distribution:")
        print(f"   Pearson r: mean={df['pearson_r'].mean():.4f}, std={df['pearson_r'].std():.4f}")
        print(f"   Spearman ρ: mean={df['spearman_rho'].mean():.4f}, std={df['spearman_rho'].std():.4f}")

        # 2. Strong correlations
        strong_pos = df[df['pearson_r'] > 0.7]
        strong_neg = df[df['pearson_r'] < -0.7]
        print(f"\n2️⃣ Strong Correlations:")
        print(f"   Positive (r > 0.7): {len(strong_pos):,} ({100*len(strong_pos)/len(df):.2f}%)")
        print(f"   Negative (r < -0.7): {len(strong_neg):,} ({100*len(strong_neg)/len(df):.2f}%)")

        # 3. Condition-wise analysis
        print(f"\n3️⃣ Condition-wise Analysis:")
        for cond in sorted(self.conditions):
            cond_df = df[df['condition'] == cond]
            mean_r = cond_df['pearson_r'].mean()
            strong = len(cond_df[abs(cond_df['pearson_r']) > 0.7])
            print(f"   {cond:20s}: mean_r={mean_r:+.4f}, strong={strong:5d}")

        # 4. Layer analysis
        print(f"\n4️⃣ Cross-Layer Analysis:")
        df['layer_A'] = df['feature_A'].str.extract(r'L(\d+)')[0].astype(int)
        df['layer_B'] = df['feature_B'].str.extract(r'L(\d+)')[0].astype(int)
        df['same_layer'] = df['layer_A'] == df['layer_B']

        same_layer = df[df['same_layer']]
        cross_layer = df[~df['same_layer']]
        print(f"   Same layer:  mean_r={same_layer['pearson_r'].mean():+.4f} (n={len(same_layer):,})")
        print(f"   Cross layer: mean_r={cross_layer['pearson_r'].mean():+.4f} (n={len(cross_layer):,})")

        # 5. Top correlated pairs
        print(f"\n5️⃣ Top 10 Most Correlated Pairs:")
        top10 = df.nlargest(10, 'pearson_r')[['feature_A', 'feature_B', 'pearson_r', 'condition']]
        for idx, row in top10.iterrows():
            print(f"   {row['feature_A']:15s} ↔ {row['feature_B']:15s}: r={row['pearson_r']:.4f} ({row['condition']})")

        return df

    def visualize_correlation_distribution(self, df):
        """Visualize overall correlation distribution"""
        print("\n📊 Generating correlation distribution plot...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Pearson r distribution
        ax = axes[0, 0]
        ax.hist(df['pearson_r'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Pearson r', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Pearson Correlation Distribution\n(n={len(df):,} pairs)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Spearman rho distribution
        ax = axes[0, 1]
        ax.hist(df['spearman_rho'], bins=100, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Spearman ρ', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Spearman Correlation Distribution\n(n={len(df):,} pairs)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3. Scatter: Pearson vs Spearman
        ax = axes[1, 0]
        ax.scatter(df['pearson_r'], df['spearman_rho'], alpha=0.1, s=1, c='purple')
        ax.plot([-1, 1], [-1, 1], 'r--', linewidth=2)
        ax.set_xlabel('Pearson r', fontsize=12)
        ax.set_ylabel('Spearman ρ', fontsize=12)
        ax.set_title('Pearson vs Spearman Correlation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # 4. Condition-wise boxplot
        ax = axes[1, 1]
        condition_data = [df[df['condition'] == c]['pearson_r'].values for c in sorted(self.conditions)]
        ax.boxplot(condition_data, labels=sorted(self.conditions))
        ax.set_xlabel('Condition', fontsize=12)
        ax.set_ylabel('Pearson r', fontsize=12)
        ax.set_title('Correlation by Condition', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        output_file = self.output_dir / 'phase2_correlation_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    def visualize_layer_interactions(self, df):
        """Visualize cross-layer correlation patterns"""
        print("\n📊 Generating layer interaction heatmap...")

        # Create layer-layer correlation matrix
        layers = range(1, 31)
        layer_matrix = np.zeros((30, 30))
        layer_counts = np.zeros((30, 30))

        for _, row in df.iterrows():
            l_a = row['layer_A'] - 1
            l_b = row['layer_B'] - 1
            if 0 <= l_a < 30 and 0 <= l_b < 30:
                layer_matrix[l_a, l_b] += row['pearson_r']
                layer_counts[l_a, l_b] += 1

        # Average
        layer_matrix = np.divide(layer_matrix, layer_counts, where=layer_counts > 0)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(layer_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)

        ax.set_xticks(np.arange(30))
        ax.set_yticks(np.arange(30))
        ax.set_xticklabels([f'L{i+1}' for i in range(30)], fontsize=8)
        ax.set_yticklabels([f'L{i+1}' for i in range(30)], fontsize=8)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        ax.set_xlabel('Feature Layer B', fontsize=14, fontweight='bold')
        ax.set_ylabel('Feature Layer A', fontsize=14, fontweight='bold')
        ax.set_title('Cross-Layer Feature Correlation Heatmap\n(Average Pearson r)',
                     fontsize=16, fontweight='bold', pad=20)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average Pearson r', rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout()
        output_file = self.output_dir / 'phase2_layer_interaction_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    def visualize_strong_correlations(self, df):
        """Visualize network of strong correlations"""
        print("\n📊 Generating strong correlation summary...")

        # Filter strong correlations
        strong = df[abs(df['pearson_r']) > 0.7].copy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Distribution of strong correlations by layer
        ax = axes[0, 0]
        layer_counts = Counter(strong['layer_A'].tolist() + strong['layer_B'].tolist())
        layers_sorted = sorted(layer_counts.keys())
        counts = [layer_counts[l] for l in layers_sorted]

        bars = ax.bar(layers_sorted, counts, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Strong Correlations by Layer\n(|r| > 0.7, n={len(strong):,})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Positive vs Negative strong correlations
        ax = axes[0, 1]
        pos_strong = strong[strong['pearson_r'] > 0.7]
        neg_strong = strong[strong['pearson_r'] < -0.7]

        categories = ['Positive\n(r > 0.7)', 'Negative\n(r < -0.7)']
        counts = [len(pos_strong), len(neg_strong)]
        colors = ['#2ecc71', '#e74c3c']

        bars = ax.bar(categories, counts, color=colors, edgecolor='black', alpha=0.7)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({100*count/len(strong):.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Strong Correlations: Positive vs Negative', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Same-layer vs Cross-layer strong correlations
        ax = axes[1, 0]
        same_layer_strong = strong[strong['same_layer']]
        cross_layer_strong = strong[~strong['same_layer']]

        categories = ['Same Layer', 'Cross Layer']
        counts = [len(same_layer_strong), len(cross_layer_strong)]
        colors = ['#3498db', '#9b59b6']

        bars = ax.bar(categories, counts, color=colors, edgecolor='black', alpha=0.7)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({100*count/len(strong):.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Strong Correlations: Same vs Cross Layer', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Condition-wise strong correlation counts
        ax = axes[1, 1]
        condition_counts = strong['condition'].value_counts()

        bars = ax.bar(range(len(condition_counts)), condition_counts.values,
                     color='coral', edgecolor='black', alpha=0.7)
        ax.set_xticks(range(len(condition_counts)))
        ax.set_xticklabels(condition_counts.index, rotation=45, ha='right')
        ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Strong Correlations by Condition', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.output_dir / 'phase2_strong_correlations_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_file}")
        plt.close()

    def generate_summary_report(self, df):
        """Generate comprehensive summary report"""
        print("\n📝 Generating summary report...")

        report = []
        report.append("=" * 80)
        report.append("PHASE 2: FEATURE-FEATURE CORRELATION ANALYSIS")
        report.append("Ultra-Deep Analysis Report")
        report.append("=" * 80)
        report.append("")

        report.append("## 1. Dataset Overview")
        report.append(f"   Total correlations analyzed: {len(df):,}")
        report.append(f"   Unique target features: {len(self.target_features)}")
        report.append(f"   Unique conditions: {len(self.conditions)}")
        report.append(f"   Unique feature pairs: {len(self.feature_pairs)}")
        report.append("")

        report.append("## 2. Correlation Statistics")
        report.append(f"   Pearson r:")
        report.append(f"      Mean: {df['pearson_r'].mean():+.4f}")
        report.append(f"      Std:  {df['pearson_r'].std():.4f}")
        report.append(f"      Min:  {df['pearson_r'].min():+.4f}")
        report.append(f"      Max:  {df['pearson_r'].max():+.4f}")
        report.append("")
        report.append(f"   Spearman ρ:")
        report.append(f"      Mean: {df['spearman_rho'].mean():+.4f}")
        report.append(f"      Std:  {df['spearman_rho'].std():.4f}")
        report.append(f"      Min:  {df['spearman_rho'].min():+.4f}")
        report.append(f"      Max:  {df['spearman_rho'].max():+.4f}")
        report.append("")

        report.append("## 3. Strong Correlations (|r| > 0.7)")
        strong = df[abs(df['pearson_r']) > 0.7]
        report.append(f"   Total: {len(strong):,} ({100*len(strong)/len(df):.2f}%)")
        report.append(f"   Positive (r > 0.7):  {len(df[df['pearson_r'] > 0.7]):,}")
        report.append(f"   Negative (r < -0.7): {len(df[df['pearson_r'] < -0.7]):,}")
        report.append("")

        report.append("## 4. Layer Analysis")
        same_layer = df[df['same_layer']]
        cross_layer = df[~df['same_layer']]
        report.append(f"   Same-layer correlations:  {len(same_layer):,} (mean_r={same_layer['pearson_r'].mean():+.4f})")
        report.append(f"   Cross-layer correlations: {len(cross_layer):,} (mean_r={cross_layer['pearson_r'].mean():+.4f})")
        report.append("")

        report.append("## 5. Condition-wise Analysis")
        for cond in sorted(self.conditions):
            cond_df = df[df['condition'] == cond]
            mean_r = cond_df['pearson_r'].mean()
            strong_count = len(cond_df[abs(cond_df['pearson_r']) > 0.7])
            report.append(f"   {cond:25s}: mean_r={mean_r:+.4f}, strong={strong_count:,}")
        report.append("")

        report.append("## 6. Top 20 Strongest Correlations")
        top20 = df.nlargest(20, 'pearson_r')[['feature_A', 'feature_B', 'pearson_r', 'condition']]
        for i, (idx, row) in enumerate(top20.iterrows(), 1):
            report.append(f"   {i:2d}. {row['feature_A']:15s} ↔ {row['feature_B']:15s}: r={row['pearson_r']:.4f} ({row['condition']})")
        report.append("")

        report.append("=" * 80)
        report.append("Analysis completed successfully!")
        report.append("=" * 80)

        # Save report
        report_file = self.output_dir / 'phase2_analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"   ✅ Saved: {report_file}")

        # Also print to console
        print("\n" + '\n'.join(report))

    def run_all(self):
        """Run complete analysis pipeline"""
        print("=" * 80)
        print("PHASE 2: FEATURE-FEATURE CORRELATION ANALYSIS")
        print("=" * 80)

        # Load data
        self.load_data_sample(max_files=1000)

        # Analyze
        df = self.ultra_deep_analysis()

        # Visualize
        self.visualize_correlation_distribution(df)
        self.visualize_layer_interactions(df)
        self.visualize_strong_correlations(df)

        # Report
        self.generate_summary_report(df)

        print("\n" + "=" * 80)
        print("✅ Phase 2 Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("=" * 80)


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # Paths
    data_dir = Path('/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations')
    output_dir = Path('/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/visualizations/phase2')

    # Run analysis
    analyzer = Phase2Analyzer(data_dir, output_dir)
    analyzer.run_all()
