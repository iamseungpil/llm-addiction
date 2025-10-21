#!/usr/bin/env python3
"""
Complete analysis of LLaMA Experiment 1 data
Checks both main file (14GB) and missing file (433MB)
Analyzes feature extraction and patching results
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import gc

def analyze_llama_experiments():
    """Analyze all LLaMA experiment data"""
    
    print("="*80)
    print("LLAMA EXPERIMENT 1 COMPLETE ANALYSIS")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Check main file
    print("\n1. Loading main experiment file (14GB)...")
    main_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
    try:
        with open(main_file, 'r') as f:
            main_data = json.load(f)
        main_count = len(main_data) if isinstance(main_data, list) else len(main_data.get('results', []))
        print(f"   ✓ Main file loaded: {main_count} experiments")
        
        # Analyze main data
        if isinstance(main_data, list):
            experiments = main_data
        else:
            experiments = main_data.get('results', [])
        
        bankruptcies = sum(1 for exp in experiments if exp.get('is_bankrupt', False))
        print(f"   - Bankruptcies: {bankruptcies}/{main_count} ({bankruptcies/main_count*100:.1f}%)")
        
        del main_data  # Free memory
        gc.collect()
    except Exception as e:
        print(f"   ✗ Error loading main file: {e}")
        main_count = 0
        experiments = []
    
    # 2. Check missing file
    print("\n2. Loading missing experiments file (433MB)...")
    missing_file = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'
    try:
        with open(missing_file, 'r') as f:
            missing_data = json.load(f)
        
        # Count experiments in dict structure
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
        
        print(f"   ✓ Missing file loaded: {missing_count} experiments")
        print(f"   - Bankruptcies: {missing_bankruptcies}/{missing_count} ({missing_bankruptcies/missing_count*100:.1f}% if missing_count > 0 else 0)")
        
        del missing_data
        gc.collect()
    except Exception as e:
        print(f"   ✗ Error loading missing file: {e}")
        missing_count = 0
        missing_bankruptcies = 0
    
    # 3. Total statistics
    total_count = main_count + missing_count
    total_bankruptcies = bankruptcies + missing_bankruptcies
    
    print("\n" + "="*80)
    print("LLAMA EXPERIMENT 1 SUMMARY")
    print("="*80)
    print(f"Total experiments completed: {total_count}/6400 ({total_count/6400*100:.1f}%)")
    print(f"Total bankruptcies: {total_bankruptcies}/{total_count} ({total_bankruptcies/total_count*100:.1f}% if total_count > 0 else 0)")
    
    if total_count < 6400:
        print(f"\n⚠️ WARNING: Experiment incomplete!")
        print(f"   Missing: {6400 - total_count} experiments ({(6400-total_count)/6400*100:.1f}%)")
        print(f"   Status: Need to resume experiment to complete dataset")
    else:
        print(f"\n✅ All 6400 experiments completed!")
    
    # 4. Check for SAE feature extraction results
    print("\n" + "="*80)
    print("SAE FEATURE EXTRACTION STATUS")
    print("="*80)
    
    feature_files = [
        '/data/llm_addiction/results/llama_exp1_6400_features_20250825_144520.json',
        '/data/llm_addiction/results/llama_exp1_6400_stats_20250825_144520.json'
    ]
    
    for file_path in feature_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
            print(f"   ✓ Found: {Path(file_path).name} ({file_size:.1f} MB)")
        else:
            print(f"   ✗ Not found: {Path(file_path).name}")
    
    # 5. Check patching experiment results
    print("\n" + "="*80)
    print("FEATURE PATCHING EXPERIMENT STATUS")
    print("="*80)
    
    patching_log = '/home/ubuntu/llm_addiction/causal_feature_discovery/src/comprehensive_patching_20250825_163012.log'
    if Path(patching_log).exists():
        with open(patching_log, 'r') as f:
            lines = f.readlines()
            
        # Find progress indicators
        for line in reversed(lines[-100:]):  # Check last 100 lines
            if 'safe_to_safe:' in line and '%' in line:
                print(f"   Latest progress: {line.strip()}")
                break
            elif 'Analysis complete!' in line:
                print(f"   ✓ Patching experiment completed!")
                break
    else:
        print(f"   ✗ Patching log not found")
    
    # Check for comprehensive causal results
    import glob
    causal_files = glob.glob('/data/llm_addiction/results/comprehensive_causal_*.json')
    if causal_files:
        latest_causal = sorted(causal_files)[-1]
        file_size = Path(latest_causal).stat().st_size / (1024*1024)
        print(f"   ✓ Latest causal result: {Path(latest_causal).name} ({file_size:.1f} MB)")
        
        # Load and summarize if not too large
        if file_size < 10:  # If less than 10MB
            try:
                with open(latest_causal, 'r') as f:
                    causal_data = json.load(f)
                print(f"     - Features tested: {len(causal_data.get('results', []))}")
                print(f"     - Conditions: {list(causal_data.keys())}")
            except:
                pass

if __name__ == "__main__":
    analyze_llama_experiments()