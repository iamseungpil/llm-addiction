#!/usr/bin/env python3
"""
Analyze L1-31 Feature Pathway from 6,400 Exp1 Games
Compare bankruptcy vs voluntary stop groups across all layers
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class PathwayAnalysis6400:
    def __init__(self):
        """Initialize analysis of 6,400 games"""
        self.results_dir = Path('/data/llm_addiction/experiment_1_pathway_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_exp1_data(self):
        """Load 6,400 Exp1 responses with L1-31 features"""
        print("📂 Loading 6,400 Exp1 games...")

        file1 = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
        file2 = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'

        with open(file1, 'r') as f:
            data1 = json.load(f)

        with open(file2, 'r') as f:
            data2 = json.load(f)

        # Extract ALL rounds for pathway analysis
        all_games = []

        for entry in data1.get('results', []):
            if 'round_features' not in entry or not entry['round_features']:
                continue

            game_data = {
                'outcome': entry.get('outcome', ''),
                'final_balance': entry.get('final_balance', 0),
                'rounds': []
            }

            for round_data in entry['round_features']:
                # Extract layer-specific features
                layer_features = {}
                for layer in range(1, 32):  # L1-L31
                    # Check if layer features exist
                    layer_key = f'layer_{layer}_features'
                    if layer_key in round_data:
                        layer_features[f'L{layer}'] = round_data[layer_key]

                game_data['rounds'].append({
                    'round': round_data.get('round', 0),
                    'balance': round_data.get('balance', 0),
                    'decision': round_data.get('decision', ''),
                    'bet_amount': round_data.get('bet_amount', 0),
                    'prompt': round_data.get('prompt', ''),
                    'response': round_data.get('response', ''),
                    'layer_features': layer_features
                })

            all_games.append(game_data)

        for entry in data2.get('results', []):
            if 'round_features' not in entry or not entry['round_features']:
                continue

            game_data = {
                'outcome': entry.get('outcome', ''),
                'final_balance': entry.get('final_balance', 0),
                'rounds': []
            }

            for round_data in entry['round_features']:
                layer_features = {}
                for layer in range(1, 32):
                    layer_key = f'layer_{layer}_features'
                    if layer_key in round_data:
                        layer_features[f'L{layer}'] = round_data[layer_key]

                game_data['rounds'].append({
                    'round': round_data.get('round', 0),
                    'balance': round_data.get('balance', 0),
                    'decision': round_data.get('decision', ''),
                    'bet_amount': round_data.get('bet_amount', 0),
                    'prompt': round_data.get('prompt', ''),
                    'response': round_data.get('response', ''),
                    'layer_features': layer_features
                })

            all_games.append(game_data)

        print(f"✅ Loaded {len(all_games)} games")

        # Separate by outcome using actual field names
        bankruptcy_games = [g for g in all_games if g.get('is_bankrupt', False)]
        voluntary_stop_games = [g for g in all_games if g.get('voluntary_stop', False)]

        print(f"   Bankruptcy: {len(bankruptcy_games)} games")
        print(f"   Voluntary stop: {len(voluntary_stop_games)} games")

        return bankruptcy_games, voluntary_stop_games

    def load_l1_31_features(self):
        """Load pre-extracted L1-31 features from experiment_1_L1_31_extraction"""
        print("\n🔍 Loading L1-31 features from extraction results...")

        feature_file = '/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json'

        with open(feature_file, 'r') as f:
            features_data = json.load(f)

        print(f"✅ Loaded features for {features_data['total_experiments_processed']} games")
        print(f"   Total significant features: {features_data['total_significant_features']}")

        return features_data

    def analyze_last_round_pathway(self, bankruptcy_games, voluntary_stop_games):
        """Analyze feature evolution L1→L31 using pre-extracted features"""
        print("\n🔍 Analyzing last round feature pathway L1→L31...")

        # Load L1-31 features
        features_data = self.load_l1_31_features()

        # Use pre-extracted features instead of re-extracting
        # Note: The extraction already has all features, we just need to group by outcome
        print(f"\n📊 Using {features_data['total_significant_features']} pre-extracted features")
        print(f"   Layers covered: {len(features_data['layer_results'])}")

        # Return features data for plotting
        return features_data
                    for feat_id, activation in enumerate(features):
                        bankruptcy_pathways[f'{layer_key}-{feat_id}'].append(activation)

        # Process voluntary stop games
        for game in voluntary_stop_games:
            if not game['rounds']:
                continue
            last_round = game['rounds'][-1]

            for layer in range(1, 32):
                layer_key = f'L{layer}'
                if layer_key in last_round['layer_features']:
                    features = last_round['layer_features'][layer_key]
                    for feat_id, activation in enumerate(features):
                        voluntary_pathways[f'{layer_key}-{feat_id}'].append(activation)

        # Calculate mean activation per layer
        bankruptcy_layer_means = {}
        voluntary_layer_means = {}

        for layer in range(1, 32):
            # Bankruptcy mean
            layer_activations = []
            for feat_key in bankruptcy_pathways:
                if feat_key.startswith(f'L{layer}-'):
                    layer_activations.extend(bankruptcy_pathways[feat_key])
            bankruptcy_layer_means[layer] = np.mean(layer_activations) if layer_activations else 0

            # Voluntary mean
            layer_activations = []
            for feat_key in voluntary_pathways:
                if feat_key.startswith(f'L{layer}-'):
                    layer_activations.extend(voluntary_pathways[feat_key])
            voluntary_layer_means[layer] = np.mean(layer_activations) if layer_activations else 0

        print(f"✅ Analyzed {len(bankruptcy_pathways)} features in bankruptcy group")
        print(f"✅ Analyzed {len(voluntary_pathways)} features in voluntary stop group")

        return bankruptcy_layer_means, voluntary_layer_means

    def plot_pathway_comparison(self, bankruptcy_means, voluntary_means):
        """Plot L1→L31 feature evolution comparison"""
        print("\n📊 Plotting pathway comparison...")

        layers = list(range(1, 32))
        bankruptcy_vals = [bankruptcy_means[l] for l in layers]
        voluntary_vals = [voluntary_means[l] for l in layers]

        plt.figure(figsize=(14, 6))
        plt.plot(layers, bankruptcy_vals, 'r-o', label='Bankruptcy', linewidth=2, markersize=6)
        plt.plot(layers, voluntary_vals, 'b-o', label='Voluntary Stop', linewidth=2, markersize=6)

        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Mean Feature Activation', fontsize=12)
        plt.title('Feature Activation Pathway (L1→L31): Last Round Decision', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        output_path = self.results_dir / 'pathway_L1_31_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot: {output_path}")

        plt.close()

    def run(self):
        """Main analysis"""
        print("=" * 80)
        print("🚀 LAYER PATHWAY ANALYSIS: 6,400 Exp1 Games")
        print("   Comparing bankruptcy vs voluntary stop across L1-L31")
        print("=" * 80)

        bankruptcy_games, voluntary_games = self.load_exp1_data()
        bankruptcy_means, voluntary_means = self.analyze_last_round_pathway(
            bankruptcy_games, voluntary_games
        )

        self.plot_pathway_comparison(bankruptcy_means, voluntary_means)

        # Save numerical results
        results = {
            'bankruptcy_layer_means': {f'L{k}': v for k, v in bankruptcy_means.items()},
            'voluntary_layer_means': {f'L{k}': v for k, v in voluntary_means.items()},
            'n_bankruptcy_games': len(bankruptcy_games),
            'n_voluntary_games': len(voluntary_games)
        }

        output_file = self.results_dir / 'pathway_analysis_6400.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved: {output_file}")
        print("=" * 80)

if __name__ == '__main__':
    analyzer = PathwayAnalysis6400()
    analyzer.run()
