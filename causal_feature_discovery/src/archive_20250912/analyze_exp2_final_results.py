#!/usr/bin/env python3
"""
Analyze final Experiment 2 results from GPU4 and GPU5
"""

import json

def analyze_exp2_results():
    print("📊 EXPERIMENT 2 FINAL RESULTS ANALYSIS")
    print("=" * 60)
    
    # Load both GPU results
    gpu5_file = "/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json"
    gpu4_file = "/data/llm_addiction/results/patching_population_mean_final_20250905_150612.json"
    
    with open(gpu5_file, 'r') as f:
        gpu5_data = json.load(f)
    
    with open(gpu4_file, 'r') as f:
        gpu4_data = json.load(f)
    
    print(f"📈 GPU 5 Results:")
    print(f"   Features tested: {gpu5_data['experiment_config']['n_features_tested']}")
    print(f"   Causal betting features: {gpu5_data['summary']['n_causal_bet']} ({gpu5_data['summary']['causal_rate_bet']:.1%})")
    print(f"   Causal stopping features: {gpu5_data['summary']['n_causal_stop']} ({gpu5_data['summary']['causal_rate_stop']:.1%})")
    print(f"   Total causal features: {gpu5_data['summary']['n_causal_any']}")
    
    print(f"\n📈 GPU 4 Results:")
    print(f"   Features tested: {gpu4_data['experiment_config']['n_features_tested']}")  
    print(f"   Causal betting features: {gpu4_data['summary']['n_causal_bet']} ({gpu4_data['summary']['causal_rate_bet']:.1%})")
    print(f"   Causal stopping features: {gpu4_data['summary']['n_causal_stop']} ({gpu4_data['summary']['causal_rate_stop']:.1%})")
    print(f"   Total causal features: {gpu4_data['summary']['n_causal_any']}")
    
    # Extract feature IDs for overlap analysis
    gpu5_bet_ids = set((f['layer'], f['feature_id']) for f in gpu5_data['causal_features_bet'])
    gpu5_stop_ids = set((f['layer'], f['feature_id']) for f in gpu5_data['causal_features_stop'])
    gpu4_bet_ids = set((f['layer'], f['feature_id']) for f in gpu4_data['causal_features_bet'])
    gpu4_stop_ids = set((f['layer'], f['feature_id']) for f in gpu4_data['causal_features_stop'])
    
    # Calculate overlaps
    bet_overlap = len(gpu5_bet_ids & gpu4_bet_ids)
    stop_overlap = len(gpu5_stop_ids & gpu4_stop_ids)
    
    gpu5_total = gpu5_bet_ids | gpu5_stop_ids
    gpu4_total = gpu4_bet_ids | gpu4_stop_ids
    total_overlap = len(gpu5_total & gpu4_total)
    
    print(f"\n🔄 OVERLAP ANALYSIS:")
    print(f"   Betting features overlap: {bet_overlap} / {min(len(gpu5_bet_ids), len(gpu4_bet_ids))}")
    print(f"   Stopping features overlap: {stop_overlap} / {min(len(gpu5_stop_ids), len(gpu4_stop_ids))}")
    print(f"   Total features overlap: {total_overlap}")
    print(f"   GPU5 unique: {len(gpu5_total - gpu4_total)}")
    print(f"   GPU4 unique: {len(gpu4_total - gpu5_total)}")
    
    # Combined union statistics
    union_total = len(gpu5_total | gpu4_total)
    print(f"\n🎯 COMBINED RESULTS:")
    print(f"   Total unique causal features: {union_total}")
    print(f"   Coverage rate: {union_total}/356 = {union_total/356:.1%}")
    
    # Top performing features analysis
    print(f"\n⭐ TOP PERFORMING FEATURES:")
    
    print(f"   GPU5 Top Betting Features:")
    for i, f in enumerate(gpu5_data['causal_features_bet'][:3]):
        print(f"     {i+1}. L{f['layer']}-{f['feature_id']}: Effect={f['bet_effect']:.1f}, Corr={f['bet_correlation']:.2f}")
        
    print(f"   GPU4 Top Betting Features:")
    for i, f in enumerate(gpu4_data['causal_features_bet'][:3]):
        print(f"     {i+1}. L{f['layer']}-{f['feature_id']}: Effect={f['bet_effect']:.1f}, Corr={f['bet_correlation']:.2f}")
    
    return {
        'gpu5_causal': gpu5_data['summary']['n_causal_any'],
        'gpu4_causal': gpu4_data['summary']['n_causal_any'],
        'total_union': union_total,
        'overlap': total_overlap,
        'coverage_rate': union_total/356
    }

if __name__ == "__main__":
    results = analyze_exp2_results()
    print(f"\n✅ Analysis complete!")
    print(f"   Combined causal features: {results['total_union']}")
    print(f"   Success rate: {results['coverage_rate']:.1%}")