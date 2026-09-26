#!/usr/bin/env python3
"""
Direct analysis of LLaMA experiments - optimized for large files
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import gc

def analyze_llama_direct():
    """Direct analysis without streaming"""
    
    print("="*80)
    print("LLAMA EXPERIMENT 1 ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Load main file
    print("\n1. Loading main experiment file...")
    main_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
    
    try:
        print("   Reading file (this will take a few minutes for 14GB)...")
        with open(main_file, 'r') as f:
            main_data = json.load(f)
        
        if isinstance(main_data, list):
            main_count = len(main_data)
            main_bankruptcies = sum(1 for exp in main_data if exp.get('is_bankrupt', False))
        else:
            main_count = len(main_data.get('results', []))
            main_bankruptcies = sum(1 for exp in main_data.get('results', []) if exp.get('is_bankrupt', False))
        
        print(f"   ✓ Main file: {main_count} experiments")
        print(f"   - Bankruptcies: {main_bankruptcies} ({main_bankruptcies/main_count*100:.1f}%)")
        
        del main_data
        gc.collect()
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        main_count = 0
        main_bankruptcies = 0
    
    # 2. Load missing file  
    print("\n2. Loading missing experiments file...")
    missing_file = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'
    
    try:
        with open(missing_file, 'r') as f:
            missing_data = json.load(f)
        
        missing_count = 0
        missing_bankruptcies = 0
        
        if isinstance(missing_data, dict):
            for key, value in missing_data.items():
                if isinstance(value, list):
                    missing_count += len(value)
                    missing_bankruptcies += sum(1 for exp in value if exp.get('is_bankrupt', False))
        else:
            missing_count = len(missing_data)
            missing_bankruptcies = sum(1 for exp in missing_data if exp.get('is_bankrupt', False))
        
        print(f"   ✓ Missing file: {missing_count} experiments")
        print(f"   - Bankruptcies: {missing_bankruptcies} ({missing_bankruptcies/missing_count*100:.1f}%)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        missing_count = 0
        missing_bankruptcies = 0
    
    # 3. Total
    total = main_count + missing_count
    total_bankrupt = main_bankruptcies + missing_bankruptcies
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total experiments: {total}/6400 ({total/6400*100:.1f}%)")
    print(f"Total bankruptcies: {total_bankrupt}/{total} ({total_bankrupt/total*100:.1f}%)")
    
    if total == 6400:
        print("\n✅ ALL 6400 experiments completed!")
    else:
        print(f"\n⚠️ Missing {6400-total} experiments")
    
    # 4. Check feature extraction
    print("\n" + "="*80)
    print("FEATURE EXTRACTION STATUS")
    print("="*80)
    
    feature_file = '/data/llm_addiction/results/llama_exp1_6400_features_20250825_144520.json'
    if Path(feature_file).exists():
        size_mb = Path(feature_file).stat().st_size / (1024**2)
        print(f"✓ Feature file exists ({size_mb:.1f} MB)")
        
        # Quick check of content
        with open(feature_file, 'r') as f:
            # Read first 1000 chars to check structure
            content = f.read(1000)
            if 'bankrupt_features' in content:
                print("  - Contains bankrupt_features")
            if 'safe_features' in content:
                print("  - Contains safe_features")
    else:
        print("✗ Feature file not found")
    
    stats_file = '/data/llm_addiction/results/llama_exp1_6400_stats_20250825_144520.json'
    if Path(stats_file).exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        
        print(f"\n✓ Statistics file loaded:")
        print(f"  - Significant features (p<0.01): {stats.get('n_significant_p001', 'N/A')}")
        print(f"  - Features with |d|>0.3: {stats.get('n_significant_cohen', 'N/A')}")
        print(f"  - Total analyzed: {stats.get('total_analyzed', 'N/A')}")
    
    # 5. Check patching results
    print("\n" + "="*80)
    print("PATCHING EXPERIMENT STATUS")
    print("="*80)
    
    import glob
    causal_files = glob.glob('/data/llm_addiction/results/comprehensive_causal_*.json')
    
    if causal_files:
        latest = sorted(causal_files)[-1]
        print(f"✓ Found causal results: {Path(latest).name}")
        
        with open(latest, 'r') as f:
            causal = json.load(f)
        
        if 'results' in causal:
            n_tested = len(causal['results'])
            print(f"  - Features tested: {n_tested}")
            
            # Find most causal features
            causal_features = []
            for fid, data in causal['results'].items():
                if 'safe_to_risky' in data and 'risky_to_safe' in data:
                    s2r_delta = data['safe_to_risky']['bankruptcy_rate']['after'] - data['safe_to_risky']['bankruptcy_rate']['before']
                    r2s_delta = data['risky_to_safe']['bankruptcy_rate']['after'] - data['risky_to_safe']['bankruptcy_rate']['before']
                    
                    if s2r_delta > 10 and r2s_delta < -10:
                        causal_features.append((fid, s2r_delta, r2s_delta))
            
            print(f"  - Causally significant: {len(causal_features)}")
            
            if causal_features:
                # Sort by effect size
                causal_features.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)
                print("\n  Top 5 causal features:")
                for i, (fid, s2r, r2s) in enumerate(causal_features[:5], 1):
                    print(f"    {i}. Feature {fid}: safe→risky +{s2r:.1f}%, risky→safe {r2s:.1f}%")
    else:
        print("✗ No patching results found")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    analyze_llama_direct()