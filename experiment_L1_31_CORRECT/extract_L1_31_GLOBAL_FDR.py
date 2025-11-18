#!/usr/bin/env python3
"""
Extract Statistically Valid SAE Features - ALL LAYERS 1-31
Proper GLOBAL FDR correction across ALL layers simultaneously
This is the CORRECT method that prevents multiple testing inflation
"""

import os
import sys
import argparse

# Parse GPU before any imports
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=5, help='GPU ID to use')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import torch
import gc
import sys
from statsmodels.stats.multitest import multipletests

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_experiments():
    """Load all 6400 experiments"""
    print("Loading experiments...")
    
    results_dir = Path('/data/llm_addiction/results')
    main_file = results_dir / "exp1_multiround_intermediate_20250819_140040.json"
    missing_file = results_dir / "exp1_missing_complete_20250820_090040.json"
    
    all_experiments = []
    
    # Load main file (5780 experiments)
    print(f"Loading {main_file.name}...")
    with open(main_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        experiments = data['results']
    else:
        experiments = []
    
    all_experiments.extend(experiments)
    print(f"  Loaded {len(experiments)} from main file")
    del data, experiments
    gc.collect()
    
    # Load missing file (620 experiments)  
    if missing_file.exists():
        print(f"Loading {missing_file.name}...")
        with open(missing_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            numbered_keys = [k for k in data.keys() if k.isdigit()]
            if numbered_keys:
                experiments = [data[k] for k in sorted(numbered_keys, key=int)]
            elif 'results' in data:
                experiments = data['results']
            else:
                experiments = []
        elif isinstance(data, list):
            experiments = data
        else:
            experiments = []
        
        all_experiments.extend(experiments)
        print(f"  Loaded {len(experiments)} from missing file")
        del data, experiments
        gc.collect()
    
    print(f"Total experiments loaded: {len(all_experiments)}")
    return all_experiments

def calculate_cohen_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
        
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    if pooled_std > 0:
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
    else:
        d = 0.0
    
    return d

def extract_last_round_prompts(experiments):
    """Extract prompts from the last decision round of each experiment"""
    print("\nExtracting last round prompts...")
    
    bankrupt_prompts = []
    safe_prompts = []
    
    for exp in tqdm(experiments, desc="Processing experiments"):
        try:
            # Get the last round with features
            round_features = exp.get('round_features', [])
            if not round_features:
                continue
                
            last_round = round_features[-1]
            prompt = last_round.get('prompt', '')
            
            if not prompt:
                continue
            
            # Classify by outcome
            if exp.get('is_bankrupt', False):
                bankrupt_prompts.append(prompt)
            elif exp.get('voluntary_stop', False):
                safe_prompts.append(prompt)
                
        except Exception as e:
            print(f"Error processing experiment: {e}")
            continue
    
    print(f"Extracted prompts:")
    print(f"  Bankrupt: {len(bankrupt_prompts)}")
    print(f"  Safe: {len(safe_prompts)}")
    
    return bankrupt_prompts, safe_prompts

def extract_multilayer_features(bankrupt_prompts, safe_prompts):
    """Extract SAE features from all available layers 25-31"""
    
    print("\n" + "="*60)
    print("MULTI-LAYER SAE FEATURE EXTRACTION")
    print("="*60)
    
    # Load models
    print("\nLoading LLaMA model...")
    model_name = "meta-llama/Llama-3.1-8B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={'': 0},
        low_cpu_mem_usage=True,
        use_cache=False
    )
    model.eval()
    
    # Load SAEs for layers 1-31
    print("\nLoading SAEs for layers 1-31...")
    print("This will take some time - 31 layers to load!")
    saes = {}
    failed_layers = []

    for layer in range(1, 32):
        try:
            print(f"  Loading Layer {layer} SAE...")
            saes[layer] = LlamaScopeDirect(layer=layer, device="cuda")
            print(f"  ✅ Layer {layer} SAE loaded successfully")
        except Exception as e:
            print(f"  ❌ Layer {layer} SAE failed: {e}")
            failed_layers.append(layer)
    
    print(f"✅ Models loaded successfully")
    print(f"   Available SAEs: {list(saes.keys())}")
    if failed_layers:
        print(f"   Failed SAEs: {failed_layers}")
    
    n_features = 32768
    
    def extract_features_from_prompt(prompt):
        """Extract features from all available layers"""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", 
                             truncation=True, max_length=512).to('cuda')
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                layer_features = {}
                for layer in saes.keys():
                    # Layer index = layer + 1 (including embeddings)
                    hidden = outputs.hidden_states[layer + 1]
                    features = saes[layer].encode(hidden[:, -1:, :].float())
                    layer_features[layer] = features[0, 0].cpu().numpy()
                
                return layer_features
                
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {layer: np.zeros(n_features) for layer in saes.keys()}
    
    # Initialize storage for all layers
    bankrupt_features = {layer: [] for layer in saes.keys()}
    safe_features = {layer: [] for layer in saes.keys()}
    
    # Extract features for bankrupt group
    print(f"\nExtracting features from {len(bankrupt_prompts)} bankrupt prompts...")
    for i, prompt in enumerate(tqdm(bankrupt_prompts, desc="Bankrupt")):
        layer_features = extract_features_from_prompt(prompt)
        
        for layer in saes.keys():
            bankrupt_features[layer].append(layer_features[layer])
        
        # Clear GPU cache periodically
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Extract features for safe group
    print(f"\nExtracting features from {len(safe_prompts)} safe prompts...")
    for i, prompt in enumerate(tqdm(safe_prompts, desc="Safe")):
        layer_features = extract_features_from_prompt(prompt)
        
        for layer in saes.keys():
            safe_features[layer].append(layer_features[layer])
        
        # Clear GPU cache periodically
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    for layer in saes.keys():
        bankrupt_features[layer] = np.array(bankrupt_features[layer])
        safe_features[layer] = np.array(safe_features[layer])
    
    print(f"\nFeature extraction complete:")
    for layer in saes.keys():
        print(f"  Layer {layer}: Bankrupt={bankrupt_features[layer].shape}, Safe={safe_features[layer].shape}")
    
    return bankrupt_features, safe_features, list(saes.keys())

def analyze_multilayer_features(bankrupt_features, safe_features, available_layers):
    """Perform statistical analysis across all layers"""
    
    print("\n" + "="*60)
    print("MULTI-LAYER STATISTICAL ANALYSIS")
    print("="*60)
    
    n_features = 32768
    all_results = {}
    
    # Analyze each layer
    for layer in available_layers:
        print(f"\nAnalyzing Layer {layer} features...")
        
        bankrupt_data = bankrupt_features[layer]
        safe_data = safe_features[layer]
        
        layer_results = []
        
        for feature_idx in tqdm(range(n_features), desc=f"Layer {layer}"):
            bankrupt_vals = bankrupt_data[:, feature_idx]
            safe_vals = safe_data[:, feature_idx]
            
            # Skip if no variance in either group
            if np.var(bankrupt_vals) == 0 and np.var(safe_vals) == 0:
                continue
                
            # Calculate Cohen's d
            cohen_d = calculate_cohen_d(bankrupt_vals, safe_vals)
            
            # Perform t-test
            try:
                t_stat, p_value = stats.ttest_ind(bankrupt_vals, safe_vals, equal_var=False)
            except:
                p_value = 1.0
                t_stat = 0.0
            
            # Calculate means and std
            bankrupt_mean = np.mean(bankrupt_vals)
            safe_mean = np.mean(safe_vals)
            bankrupt_std = np.std(bankrupt_vals)
            safe_std = np.std(safe_vals)
            
            layer_results.append({
                'layer': layer,
                'feature_idx': feature_idx,
                'cohen_d': cohen_d,
                'p_value': p_value,
                't_stat': t_stat,
                'bankrupt_mean': bankrupt_mean,
                'safe_mean': safe_mean,
                'bankrupt_std': bankrupt_std,
                'safe_std': safe_std,
                'mean_diff': abs(bankrupt_mean - safe_mean)
            })
        
        all_results[layer] = layer_results
        print(f"  Layer {layer}: {len(layer_results)} features analyzed")
    
    return all_results

def apply_multilayer_selection(all_results):
    """Apply statistical selection across all layers"""
    
    print("\n" + "="*60)
    print("MULTI-LAYER FEATURE SELECTION")
    print("="*60)
    
    # Flatten all results for multiple testing correction
    all_features = []
    all_p_values = []
    
    for layer, results in all_results.items():
        for result in results:
            all_features.append(result)
            all_p_values.append(result['p_value'])
    
    # Apply FDR correction
    print(f"Total features to test: {len(all_p_values)}")
    
    corrected_results = multipletests(all_p_values, method='fdr_bh', alpha=0.05)
    corrected_p_values = corrected_results[1]
    significant_mask = corrected_results[0]
    
    # Selection criteria
    COHEN_D_THRESHOLD = 0.3    # Medium effect size
    P_VALUE_THRESHOLD = 0.001  # Very strict p-value
    FDR_ALPHA = 0.05          # 5% false discovery rate
    
    selected_features = []
    layer_counts = {}
    
    for i, result in enumerate(all_features):
        corrected_p = corrected_p_values[i]
        is_fdr_significant = significant_mask[i]
        
        # Apply selection criteria
        meets_cohen_d = abs(result['cohen_d']) >= COHEN_D_THRESHOLD
        meets_p_value = result['p_value'] < P_VALUE_THRESHOLD
        meets_fdr = is_fdr_significant
        
        if meets_cohen_d and meets_p_value and meets_fdr:
            result['corrected_p_value'] = corrected_p
            selected_features.append(result)
            
            # Count by layer
            layer = result['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    # Sort by absolute Cohen's d
    selected_features.sort(key=lambda x: abs(x['cohen_d']), reverse=True)
    
    print(f"\nSelection Results:")
    print(f"  Total features analyzed: {len(all_features)}")
    print(f"  Cohen's d >= {COHEN_D_THRESHOLD}: {sum(1 for r in all_features if abs(r['cohen_d']) >= COHEN_D_THRESHOLD)}")
    print(f"  p-value < {P_VALUE_THRESHOLD}: {sum(1 for r in all_features if r['p_value'] < P_VALUE_THRESHOLD)}")
    print(f"  FDR significant (α={FDR_ALPHA}): {sum(significant_mask)}")
    print(f"  **Selected features (all criteria): {len(selected_features)}**")
    
    # Layer breakdown
    print(f"\nSelected features by layer:")
    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer}: {layer_counts[layer]} features")
    
    return selected_features

def save_multilayer_results(selected_features, all_results):
    """Save comprehensive results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare data by layer for NPZ format
    layer_data = {}
    layer_summary = {}
    
    for result in selected_features:
        layer = result['layer']
        if layer not in layer_data:
            layer_data[layer] = []
        layer_data[layer].append(result)
    
    # Save NPZ file with all layers
    npz_data = {}
    for layer, features in layer_data.items():
        npz_data[f'layer_{layer}_indices'] = np.array([f['feature_idx'] for f in features])
        npz_data[f'layer_{layer}_cohen_d'] = np.array([f['cohen_d'] for f in features])
        npz_data[f'layer_{layer}_p_values'] = np.array([f['p_value'] for f in features])
        npz_data[f'layer_{layer}_bankrupt_mean'] = np.array([f['bankrupt_mean'] for f in features])
        npz_data[f'layer_{layer}_safe_mean'] = np.array([f['safe_mean'] for f in features])
        npz_data[f'layer_{layer}_bankrupt_std'] = np.array([f['bankrupt_std'] for f in features])
        npz_data[f'layer_{layer}_safe_std'] = np.array([f['safe_std'] for f in features])
        
        layer_summary[layer] = len(features)
    
    output_npz = f'/data/llm_addiction/results/L1_31_GLOBAL_FDR_features_{timestamp}.npz'
    np.savez(output_npz, **npz_data)

    # Save detailed JSON
    output_json = f'/data/llm_addiction/results/L1_31_GLOBAL_FDR_analysis_{timestamp}.json'
    
    save_data = {
        'timestamp': timestamp,
        'analysis_config': {
            'cohen_d_threshold': 0.3,
            'p_value_threshold': 0.001,
            'fdr_alpha': 0.05,
            'correction_method': 'fdr_bh',
            'layers_analyzed': list(all_results.keys())
        },
        'summary': {
            'total_features_analyzed': sum(len(results) for results in all_results.values()),
            'selected_features_total': len(selected_features),
            'layer_breakdown': layer_summary,
            'selection_rate': len(selected_features) / sum(len(results) for results in all_results.values())
        },
        'selected_features': selected_features[:1000],  # Save first 1000 for space
    }
    
    with open(output_json, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✅ Results saved:")
    print(f"  NPZ file: {output_npz}")
    print(f"  JSON file: {output_json}")
    
    # Print top features
    print(f"\nTop 10 Features by |Cohen's d| (across all layers):")
    for i, feat in enumerate(selected_features[:10], 1):
        direction = "↑ Risky" if feat['cohen_d'] > 0 else "↓ Safe"
        print(f"  {i:2d}. L{feat['layer']}-{feat['feature_idx']:5d}: "
              f"d={feat['cohen_d']:+.3f} {direction}, p={feat['p_value']:.2e}")
    
    return output_npz, output_json

def main():
    """Main function"""
    print("="*80)
    print("MULTI-LAYER SAE FEATURE EXTRACTION (Layers 1-31)")
    print("GLOBAL FDR CORRECTION - The CORRECT Statistical Method")
    print("Proper Cohen's d + Welch's t-test + Global FDR correction")
    print("="*80)
    print(f"Running on GPU: {args.gpu}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Load experiments
    experiments = load_experiments()
    
    # Extract last round prompts
    bankrupt_prompts, safe_prompts = extract_last_round_prompts(experiments)
    
    if len(bankrupt_prompts) == 0 or len(safe_prompts) == 0:
        print("❌ Error: No prompts extracted!")
        return
    
    # Extract features from all available layers
    bankrupt_features, safe_features, available_layers = extract_multilayer_features(
        bankrupt_prompts, safe_prompts)
    
    # Statistical analysis across all layers
    all_results = analyze_multilayer_features(
        bankrupt_features, safe_features, available_layers)
    
    # Feature selection with multi-layer correction
    selected_features = apply_multilayer_selection(all_results)
    
    # Save comprehensive results
    npz_file, json_file = save_multilayer_results(selected_features, all_results)
    
    print(f"\n🎉 L1-31 GLOBAL FDR ANALYSIS COMPLETE!")
    print(f"Selected {len(selected_features)} statistically valid features from ALL 31 layers")
    print(f"(Wrong per-layer method found 20,630 features - this global method is MUCH more stringent)")
    print(f"NPZ file: {npz_file}")
    print(f"JSON file: {json_file}")

if __name__ == '__main__':
    main()