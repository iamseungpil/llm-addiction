#!/usr/bin/env python3
"""
Create feature activation distribution figure for the paper
Shows safe vs bankruptcy activation differences for most significant features
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_feature_data():
    """Load feature arrays and experiment results"""
    print("Loading feature data...")
    
    # Load feature arrays (6,400 samples)
    feature_data = np.load('/data/llm_addiction/results/llama_feature_arrays_20250829_150110_v2.npz')
    
    # Load experiment results to identify bankrupt vs safe samples
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        exp1_main = json.load(f)
    
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        exp1_add = json.load(f)
    
    all_results = exp1_main['results'] + exp1_add['results']
    
    # Extract bankruptcy labels
    is_bankrupt = np.array([r.get('is_bankrupt', False) for r in all_results])
    
    print(f"Total samples: {len(is_bankrupt)}")
    print(f"Bankrupt samples: {np.sum(is_bankrupt)}")
    print(f"Safe samples: {np.sum(~is_bankrupt)}")
    
    return feature_data, is_bankrupt

def calculate_cohen_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    if pooled_std > 0:
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
    else:
        d = 0
    
    return d

def find_top_features(feature_data, is_bankrupt, n_top=8):
    """Find features with highest Cohen's d between safe and bankrupt groups"""
    
    print("\nFinding top discriminative features...")
    
    # Get feature arrays
    features_l25 = feature_data['features_l25']
    features_l30 = feature_data['features_l30']
    
    # Combine features
    all_features = np.concatenate([features_l25, features_l30], axis=1)
    n_samples, n_features = all_features.shape
    
    print(f"Total features: {n_features} (L25: {features_l25.shape[1]}, L30: {features_l30.shape[1]})")
    
    # Calculate Cohen's d for each feature
    cohen_ds = []
    p_values = []
    
    for i in range(n_features):
        if i % 5000 == 0:
            print(f"  Processing feature {i}/{n_features}...")
        
        bankrupt_values = all_features[is_bankrupt, i]
        safe_values = all_features[~is_bankrupt, i]
        
        # Skip if no variance
        if np.var(bankrupt_values) == 0 and np.var(safe_values) == 0:
            cohen_ds.append(0)
            p_values.append(1.0)
            continue
        
        # Calculate Cohen's d
        d = calculate_cohen_d(bankrupt_values, safe_values)
        cohen_ds.append(d)
        
        # Calculate p-value
        _, p = stats.ttest_ind(bankrupt_values, safe_values)
        p_values.append(p)
    
    cohen_ds = np.array(cohen_ds)
    p_values = np.array(p_values)
    
    # Find top features (both positive and negative Cohen's d)
    abs_cohen_ds = np.abs(cohen_ds)
    
    # Filter by significance
    significant_mask = p_values < 0.01
    significant_indices = np.where(significant_mask)[0]
    
    print(f"\nSignificant features (p < 0.01): {len(significant_indices)}")
    
    # Get top features by absolute Cohen's d
    top_indices = significant_indices[np.argsort(abs_cohen_ds[significant_indices])[-n_top:]]
    top_indices = top_indices[np.argsort(cohen_ds[top_indices])[::-1]]  # Sort by actual Cohen's d
    
    # Prepare feature info
    top_features = []
    for idx in top_indices:
        if idx < features_l25.shape[1]:
            layer = 25
            feature_id = idx
        else:
            layer = 30
            feature_id = idx - features_l25.shape[1]
        
        feature_info = {
            'index': idx,
            'layer': layer,
            'feature_id': feature_id,
            'cohen_d': cohen_ds[idx],
            'p_value': p_values[idx],
            'bankrupt_values': all_features[is_bankrupt, idx],
            'safe_values': all_features[~is_bankrupt, idx]
        }
        top_features.append(feature_info)
        
        print(f"  L{layer}-{feature_id}: Cohen's d = {cohen_ds[idx]:.3f}, p = {p_values[idx]:.2e}")
    
    return top_features

def create_distribution_figure(top_features):
    """Create the feature activation distribution figure"""
    
    print("\nCreating distribution figure...")
    
    # Set up the figure
    n_features = len(top_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Color scheme
    safe_color = '#2E86AB'  # Blue
    bankrupt_color = '#A23B72'  # Red/Purple
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        # Get values
        safe_vals = feature['safe_values']
        bankrupt_vals = feature['bankrupt_values']
        
        # Create violin plots
        parts = ax.violinplot([safe_vals, bankrupt_vals], 
                              positions=[0, 1],
                              widths=0.7,
                              showmeans=True,
                              showmedians=True)
        
        # Color the violin plots
        for j, pc in enumerate(parts['bodies']):
            if j == 0:
                pc.set_facecolor(safe_color)
                pc.set_alpha(0.6)
            else:
                pc.set_facecolor(bankrupt_color)
                pc.set_alpha(0.6)
        
        # Customize other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                vp = parts[partname]
                vp.set_edgecolor('black')
                vp.set_linewidth(1)
        
        # Add scatter points for better visibility
        np.random.seed(42)
        jitter = 0.1
        
        # Sample points for visualization (max 100 per group)
        n_safe_show = min(100, len(safe_vals))
        n_bankrupt_show = min(100, len(bankrupt_vals))
        
        safe_sample = np.random.choice(safe_vals, n_safe_show, replace=False)
        bankrupt_sample = np.random.choice(bankrupt_vals, n_bankrupt_show, replace=False)
        
        ax.scatter(np.random.normal(0, jitter, n_safe_show), 
                  safe_sample, 
                  alpha=0.3, s=10, color=safe_color)
        ax.scatter(np.random.normal(1, jitter, n_bankrupt_show), 
                  bankrupt_sample, 
                  alpha=0.3, s=10, color=bankrupt_color)
        
        # Labels and title
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Safe\n(n=6189)', 'Bankrupt\n(n=211)'])
        ax.set_ylabel('Feature Activation', fontsize=10)
        
        # Title with statistics
        title = f"L{feature['layer']}-{feature['feature_id']}"
        subtitle = f"Cohen's d = {feature['cohen_d']:.3f}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight='bold')
        
        # Add significance stars
        if feature['p_value'] < 0.001:
            sig_text = '***'
        elif feature['p_value'] < 0.01:
            sig_text = '**'
        elif feature['p_value'] < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Add significance indicator
        y_max = max(np.max(safe_vals), np.max(bankrupt_vals))
        ax.text(0.5, y_max * 1.05, sig_text, ha='center', fontsize=12)
        
        # Add horizontal line at 0 if range includes it
        if np.min(safe_vals) < 0 or np.min(bankrupt_vals) < 0:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        # Grid
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_axisbelow(True)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    # Main title
    fig.suptitle('Feature Activation Distributions: Safe vs Bankruptcy States', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=safe_color, alpha=0.6, label='Safe (Voluntary Stop)'),
        Patch(facecolor=bankrupt_color, alpha=0.6, label='Bankruptcy')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/ubuntu/llm_addiction/writing/figures/feature_activation_distribution'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved: {output_path}.png/pdf")
    
    return fig

def main():
    """Main function"""
    print("="*60)
    print("CREATING FEATURE ACTIVATION DISTRIBUTION FIGURE")
    print("="*60)
    
    # Load data
    feature_data, is_bankrupt = load_feature_data()
    
    # Find top features
    top_features = find_top_features(feature_data, is_bankrupt, n_top=8)
    
    # Create figure
    fig = create_distribution_figure(top_features)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: 6,400")
    print(f"Bankrupt: {np.sum(is_bankrupt)} ({np.sum(is_bankrupt)/len(is_bankrupt)*100:.1f}%)")
    print(f"Safe: {np.sum(~is_bankrupt)} ({np.sum(~is_bankrupt)/len(is_bankrupt)*100:.1f}%)")
    print(f"Features shown: {len(top_features)}")
    print(f"Average |Cohen's d|: {np.mean([abs(f['cohen_d']) for f in top_features]):.3f}")
    
    print("\n✅ Figure creation complete!")

if __name__ == '__main__':
    main()