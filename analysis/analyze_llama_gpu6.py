#!/usr/bin/env python3
"""
Analyze LLaMA experiments using GPU6 for processing large files
Handles 14GB main file + 433MB missing file efficiently
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import gc
import sys
import ijson  # For streaming JSON parsing

def count_experiments_streaming(file_path):
    """Count experiments using streaming to handle large files"""
    count = 0
    bankruptcies = 0
    
    try:
        with open(file_path, 'rb') as f:
            # Try to parse as array first
            parser = ijson.items(f, 'item')
            for item in parser:
                count += 1
                if item.get('is_bankrupt', False):
                    bankruptcies += 1
                    
                # Print progress every 100 items
                if count % 100 == 0:
                    print(f"    Processed {count} experiments...", end='\r')
                    
    except Exception as e:
        # If array parsing fails, try as dict
        print(f"    Trying alternative parsing method...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        count += len(value)
                        bankruptcies += sum(1 for exp in value if exp.get('is_bankrupt', False))
            else:
                count = len(data)
                bankruptcies = sum(1 for exp in data if exp.get('is_bankrupt', False))
    
    return count, bankruptcies

def analyze_llama_complete():
    """Complete analysis of LLaMA experiments"""
    
    print("="*80)
    print("LLAMA EXPERIMENT 1 COMPLETE ANALYSIS (GPU6)")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Analyze main file
    print("\n1. Analyzing main experiment file (14GB)...")
    main_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
    
    if Path(main_file).exists():
        file_size_gb = Path(main_file).stat().st_size / (1024**3)
        print(f"   File size: {file_size_gb:.2f} GB")
        
        print("   Counting experiments (this may take a few minutes)...")
        main_count, main_bankruptcies = count_experiments_streaming(main_file)
        print(f"   ✓ Main file: {main_count} experiments")
        print(f"   - Bankruptcies: {main_bankruptcies}/{main_count} ({main_bankruptcies/main_count*100:.1f}%)")
    else:
        print(f"   ✗ Main file not found")
        main_count, main_bankruptcies = 0, 0
    
    # 2. Analyze missing file
    print("\n2. Analyzing missing experiments file (433MB)...")
    missing_file = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'
    
    if Path(missing_file).exists():
        file_size_mb = Path(missing_file).stat().st_size / (1024**2)
        print(f"   File size: {file_size_mb:.1f} MB")
        
        print("   Loading missing experiments...")
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
        print(f"   - Bankruptcies: {missing_bankruptcies}/{missing_count} ({missing_bankruptcies/missing_count*100:.1f}%)")
    else:
        print(f"   ✗ Missing file not found")
        missing_count, missing_bankruptcies = 0, 0
    
    # 3. Total statistics
    total_count = main_count + missing_count
    total_bankruptcies = main_bankruptcies + missing_bankruptcies
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 SUMMARY")
    print("="*80)
    print(f"Total experiments completed: {total_count}/6400 ({total_count/6400*100:.1f}%)")
    print(f"Total bankruptcies: {total_bankruptcies}/{total_count} ({total_bankruptcies/total_count*100:.1f}%)")
    
    if total_count == 6400:
        print("\n✅ ALL 6400 experiments completed!")
    else:
        print(f"\n⚠️ WARNING: {6400 - total_count} experiments missing ({(6400-total_count)/6400*100:.1f}%)")
    
    # 4. Check SAE feature extraction
    print("\n" + "="*80)
    print("SAE FEATURE EXTRACTION STATUS")
    print("="*80)
    
    feature_files = [
        '/data/llm_addiction/results/llama_exp1_6400_features_20250825_144520.json',
        '/data/llm_addiction/results/llama_exp1_6400_stats_20250825_144520.json'
    ]
    
    for file_path in feature_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / (1024**2)
            print(f"   ✓ Found: {Path(file_path).name} ({file_size:.1f} MB)")
            
            # Check feature extraction completeness
            if 'features' in file_path:
                try:
                    with open(file_path, 'r') as f:
                        feature_data = json.load(f)
                    
                    n_samples = len(feature_data.get('bankrupt_features', {}).get('layer_25', []))
                    print(f"     - Bankrupt samples with features: {n_samples}")
                    
                    n_safe = len(feature_data.get('safe_features', {}).get('layer_25', []))
                    print(f"     - Safe samples with features: {n_safe}")
                    
                    total_features = n_samples + n_safe
                    print(f"     - Total samples analyzed: {total_features}")
                    
                    if total_features < total_count:
                        print(f"     ⚠️ Feature extraction incomplete: {total_features}/{total_count}")
                    else:
                        print(f"     ✅ Feature extraction complete for all samples")
                        
                except Exception as e:
                    print(f"     ⚠️ Could not analyze feature file: {e}")
        else:
            print(f"   ✗ Not found: {Path(file_path).name}")
    
    # 5. Check patching experiment results
    print("\n" + "="*80)
    print("EXPERIMENT 2: ACTIVATION PATCHING STATUS")
    print("="*80)
    
    patching_log = '/home/ubuntu/llm_addiction/causal_feature_discovery/src/comprehensive_patching_20250825_163012.log'
    if Path(patching_log).exists():
        print(f"   ✓ Patching log found")
        with open(patching_log, 'r') as f:
            lines = f.readlines()
        
        # Find completion status
        completed = False
        last_progress = ""
        for line in reversed(lines[-200:]):
            if 'Analysis complete!' in line:
                completed = True
                print(f"   ✅ Patching experiment COMPLETED")
                break
            elif 'safe_to_safe:' in line and '%' in line:
                last_progress = line.strip()
        
        if not completed and last_progress:
            print(f"   ⏳ In progress: {last_progress}")
    else:
        print(f"   ✗ Patching log not found")
    
    # Check for causal validation results
    import glob
    causal_files = glob.glob('/data/llm_addiction/results/comprehensive_causal_*.json')
    if causal_files:
        latest_causal = sorted(causal_files)[-1]
        file_size = Path(latest_causal).stat().st_size / (1024**2)
        print(f"\n   ✓ Causal validation results: {Path(latest_causal).name} ({file_size:.1f} MB)")
        
        # Analyze causal results
        try:
            with open(latest_causal, 'r') as f:
                causal_data = json.load(f)
            
            if 'results' in causal_data:
                n_features = len(causal_data['results'])
                print(f"     - Features tested: {n_features}")
                
                # Count significant features
                significant = []
                for feat_id, feat_data in causal_data['results'].items():
                    if 'safe_to_risky' in feat_data and 'risky_to_safe' in feat_data:
                        s2r = feat_data['safe_to_risky']
                        r2s = feat_data['risky_to_safe']
                        
                        # Check if behavior changed significantly
                        if (s2r['bankruptcy_rate']['after'] > s2r['bankruptcy_rate']['before'] + 10 and
                            r2s['bankruptcy_rate']['after'] < r2s['bankruptcy_rate']['before'] - 10):
                            significant.append(feat_id)
                
                print(f"     - Causally significant features: {len(significant)}")
                if significant[:5]:
                    print(f"     - Top causal features: {', '.join(significant[:5])}")
                    
        except Exception as e:
            print(f"     ⚠️ Could not analyze causal results: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Install ijson if needed
    try:
        import ijson
    except ImportError:
        print("Installing ijson for streaming JSON parsing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "ijson"], check=True)
        import ijson
    
    analyze_llama_complete()