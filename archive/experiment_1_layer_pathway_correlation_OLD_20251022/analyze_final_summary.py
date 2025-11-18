#!/usr/bin/env python3
"""
Final Summary: L1-31 Features + Pathway + Patching Results
Uses pre-extracted L1-31 features to create comprehensive analysis
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class FinalSummary:
    def __init__(self):
        self.results_dir = Path('/data/llm_addiction/comprehensive_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_data(self):
        """Load all experiment results"""
        print("=" * 80)
        print("📊 LOADING ALL EXPERIMENT RESULTS")
        print("=" * 80)
        
        # 1. L1-31 Feature Extraction
        print("\n1. Loading L1-31 feature extraction...")
        with open('/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json', 'r') as f:
            l1_31_features = json.load(f)
        
        print(f"   ✅ {l1_31_features['total_experiments_processed']} games analyzed")
        print(f"   ✅ {l1_31_features['total_significant_features']} significant features")
        
        # 2. 6400 Exp1 Games
        print("\n2. Loading 6400 games data...")
        with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
            exp1_main = json.load(f)
            
        with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
            exp1_add = json.load(f)
        
        total_games = len(exp1_main.get('results', [])) + len(exp1_add.get('results', []))
        print(f"   ✅ {total_games} total games")
        
        # Count outcomes
        bankruptcy_count = sum(1 for g in exp1_main.get('results', []) if g.get('is_bankrupt', False))
        bankruptcy_count += sum(1 for g in exp1_add.get('results', []) if g.get('is_bankrupt', False))
        
        voluntary_count = sum(1 for g in exp1_main.get('results', []) if g.get('voluntary_stop', False))
        voluntary_count += sum(1 for g in exp1_add.get('results', []) if g.get('voluntary_stop', False))
        
        print(f"   📌 Bankruptcy: {bankruptcy_count} ({100*bankruptcy_count/total_games:.1f}%)")
        print(f"   📌 Voluntary stop: {voluntary_count} ({100*voluntary_count/total_games:.1f}%)")
        
        return {
            'l1_31_features': l1_31_features,
            'exp1_games': total_games,
            'bankruptcy_count': bankruptcy_count,
            'voluntary_count': voluntary_count
        }
    
    def create_layer_distribution_plot(self, l1_31_data):
        """Plot feature count distribution across layers"""
        print("\n📊 Creating layer distribution plot...")
        
        layers = []
        counts = []
        
        for layer_num in sorted(l1_31_data['significant_features_by_layer'].keys(), key=int):
            layers.append(int(layer_num))
            counts.append(l1_31_data['significant_features_by_layer'][layer_num])
        
        plt.figure(figsize=(14, 6))
        plt.bar(layers, counts, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Significant Features Count', fontsize=12)
        plt.title('Significant Features Distribution Across L1-L31', fontsize=14)
        plt.xticks(layers)
        plt.grid(True, alpha=0.3, axis='y')
        
        output_path = self.results_dir / 'layer_distribution_L1_31.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {output_path}")
        plt.close()
        
    def create_final_report(self, data):
        """Create comprehensive final report"""
        print("\n📝 Creating final comprehensive report...")
        
        l1_31 = data['l1_31_features']
        
        report = {
            'timestamp': '20251001_final',
            'experiment_1_games': {
                'total_games': data['exp1_games'],
                'bankruptcy': data['bankruptcy_count'],
                'voluntary_stop': data['voluntary_count'],
                'bankruptcy_rate': f"{100*data['bankruptcy_count']/data['exp1_games']:.2f}%"
            },
            'l1_31_feature_extraction': {
                'total_features': l1_31['total_significant_features'],
                'layers_analyzed': len(l1_31['layer_results']),
                'layer_distribution': l1_31['significant_features_by_layer']
            },
            'top_5_layers_by_features': []
        }
        
        # Find top 5 layers
        layer_counts = [(int(k), v) for k, v in l1_31['significant_features_by_layer'].items()]
        layer_counts.sort(key=lambda x: x[1], reverse=True)
        
        for layer, count in layer_counts[:5]:
            report['top_5_layers_by_features'].append({
                'layer': layer,
                'features': count
            })
        
        # Save report
        output_file = self.results_dir / 'final_comprehensive_report.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ Saved: {output_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 FINAL COMPREHENSIVE SUMMARY")
        print("=" * 80)
        print(f"\n1. Experiment 1 (6400 Games):")
        print(f"   Total games: {report['experiment_1_games']['total_games']}")
        print(f"   Bankruptcy: {report['experiment_1_games']['bankruptcy']} ({report['experiment_1_games']['bankruptcy_rate']})")
        print(f"   Voluntary stop: {report['experiment_1_games']['voluntary_stop']}")
        
        print(f"\n2. L1-31 Feature Extraction:")
        print(f"   Total significant features: {report['l1_31_feature_extraction']['total_features']:,}")
        print(f"   Layers analyzed: {report['l1_31_feature_extraction']['layers_analyzed']}")
        
        print(f"\n3. Top 5 Layers by Feature Count:")
        for item in report['top_5_layers_by_features']:
            print(f"   Layer {item['layer']}: {item['features']:,} features")
        
        print("\n" + "=" * 80)
        
        return report

if __name__ == "__main__":
    analyzer = FinalSummary()
    
    # Load all data
    data = analyzer.load_all_data()
    
    # Create visualizations
    analyzer.create_layer_distribution_plot(data['l1_31_features'])
    
    # Create final report
    report = analyzer.create_final_report(data)
    
    print("\n✅ Analysis complete!")
