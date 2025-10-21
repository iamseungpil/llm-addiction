#!/usr/bin/env python3
"""
Extract Statistically Valid SAE Features
Proper statistical analysis with Cohen's d, t-test, and multiple testing correction
Replaces the flawed extract_extreme_features_6400.py approach
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

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

def extract_statistical_features(bankrupt_prompts, safe_prompts):
    """Extract SAE features with proper statistical analysis"""
    
    print("\n" + "="*60)
    print("STATISTICAL SAE FEATURE EXTRACTION")
    print("="*60)
    
    # Load models
    print("\nLoading models...")
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
    
    # Load SAEs for layers 25-31
    print("Loading SAEs for layers 25-31...")
    saes = {}
    for layer in range(25, 32):
        try:
            print(f"  Loading Layer {layer} SAE...")
            saes[layer] = LlamaScopeDirect(layer=layer, device="cuda")
            print(f"  ✅ Layer {layer} SAE loaded successfully")
        except Exception as e:
            print(f"  ❌ Layer {layer} SAE failed: {e}")
            # Continue with other layers
    
    print(f"✅ Models loaded successfully - {len(saes)} SAEs available")
    
    n_features = 32768
    
    def extract_features_from_prompt(prompt):
        """Extract features from a single prompt for all available layers"""
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
            # Return zero features for all available layers
            return {layer: np.zeros(n_features) for layer in saes.keys()}
    
    # Extract features for bankrupt group
    print("\nExtracting features from bankrupt prompts...")
    bankrupt_features_25 = []
    bankrupt_features_30 = []
    
    for i, prompt in enumerate(tqdm(bankrupt_prompts, desc="Bankrupt")):
        f25, f30 = extract_features_from_prompt(prompt)
        bankrupt_features_25.append(f25)
        bankrupt_features_30.append(f30)
        
        # Clear GPU cache periodically
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Extract features for safe group
    print("\nExtracting features from safe prompts...")
    safe_features_25 = []
    safe_features_30 = []
    
    for i, prompt in enumerate(tqdm(safe_prompts, desc="Safe")):
        f25, f30 = extract_features_from_prompt(prompt)
        safe_features_25.append(f25)
        safe_features_30.append(f30)
        
        # Clear GPU cache periodically
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Convert to numpy arrays
    bankrupt_features_25 = np.array(bankrupt_features_25)  # Shape: (n_bankrupt, 32768)
    bankrupt_features_30 = np.array(bankrupt_features_30)
    safe_features_25 = np.array(safe_features_25)          # Shape: (n_safe, 32768)
    safe_features_30 = np.array(safe_features_30)
    
    print(f"\nFeature extraction complete:")
    print(f"  Bankrupt features shape: L25={bankrupt_features_25.shape}, L30={bankrupt_features_30.shape}")
    print(f"  Safe features shape: L25={safe_features_25.shape}, L30={safe_features_30.shape}")
    
    return bankrupt_features_25, bankrupt_features_30, safe_features_25, safe_features_30

def analyze_features_statistically(bankrupt_25, bankrupt_30, safe_25, safe_30):
    """Perform proper statistical analysis"""
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    n_features = 32768
    results_25 = []
    results_30 = []
    
    # Layer 25 analysis
    print("\nAnalyzing Layer 25 features...")
    for feature_idx in tqdm(range(n_features), desc="Layer 25"):
        bankrupt_vals = bankrupt_25[:, feature_idx]
        safe_vals = safe_25[:, feature_idx]
        
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
        
        results_25.append({
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
    
    # Layer 30 analysis
    print("\nAnalyzing Layer 30 features...")
    for feature_idx in tqdm(range(n_features), desc="Layer 30"):
        bankrupt_vals = bankrupt_30[:, feature_idx]
        safe_vals = safe_30[:, feature_idx]
        
        # Skip if no variance
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
        
        results_30.append({
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
    
    print(f"\nAnalysis complete:")
    print(f"  Layer 25: {len(results_25)} features analyzed")
    print(f"  Layer 30: {len(results_30)} features analyzed")
    
    return results_25, results_30

def apply_statistical_selection(results_25, results_30):
    """Apply proper statistical selection criteria"""
    
    print("\n" + "="*60)
    print("FEATURE SELECTION WITH STATISTICAL CRITERIA")
    print("="*60)
    
    # Combine all p-values for multiple testing correction
    all_p_values = []
    all_features = []
    
    for result in results_25:
        all_p_values.append(result['p_value'])
        all_features.append(('L25', result))
    
    for result in results_30:
        all_p_values.append(result['p_value'])
        all_features.append(('L30', result))
    
    # Apply FDR correction (Benjamini-Hochberg)
    print(f"Total features to test: {len(all_p_values)}")
    
    corrected_results = multipletests(all_p_values, method='fdr_bh', alpha=0.05)
    corrected_p_values = corrected_results[1]
    significant_mask = corrected_results[0]
    
    # Selection criteria
    COHEN_D_THRESHOLD = 0.3    # Medium effect size
    P_VALUE_THRESHOLD = 0.001  # Very strict p-value
    FDR_ALPHA = 0.05          # 5% false discovery rate
    
    selected_features = []
    
    for i, (layer, result) in enumerate(all_features):
        corrected_p = corrected_p_values[i]
        is_fdr_significant = significant_mask[i]
        
        # Apply selection criteria
        meets_cohen_d = abs(result['cohen_d']) >= COHEN_D_THRESHOLD
        meets_p_value = result['p_value'] < P_VALUE_THRESHOLD
        meets_fdr = is_fdr_significant
        
        if meets_cohen_d and meets_p_value and meets_fdr:
            selected_features.append({
                'layer': 25 if layer == 'L25' else 30,
                'feature_id': result['feature_idx'],
                'cohen_d': result['cohen_d'],
                'p_value': result['p_value'],
                'corrected_p_value': corrected_p,
                'bankrupt_mean': result['bankrupt_mean'],
                'safe_mean': result['safe_mean'],
                'bankrupt_std': result['bankrupt_std'],
                'safe_std': result['safe_std'],
                'mean_diff': result['mean_diff']
            })
    
    # Sort by absolute Cohen's d
    selected_features.sort(key=lambda x: abs(x['cohen_d']), reverse=True)
    
    print(f"\nSelection Results:")
    print(f"  Total features analyzed: {len(all_features)}")
    print(f"  Cohen's d >= {COHEN_D_THRESHOLD}: {sum(1 for _, r in all_features if abs(r['cohen_d']) >= COHEN_D_THRESHOLD)}")
    print(f"  p-value < {P_VALUE_THRESHOLD}: {sum(1 for _, r in all_features if r['p_value'] < P_VALUE_THRESHOLD)}")
    print(f"  FDR significant (α={FDR_ALPHA}): {sum(significant_mask)}")
    print(f"  **Selected features (all criteria): {len(selected_features)}**")
    
    # Layer breakdown
    l25_selected = [f for f in selected_features if f['layer'] == 25]
    l30_selected = [f for f in selected_features if f['layer'] == 30]
    print(f"    - Layer 25: {len(l25_selected)}")
    print(f"    - Layer 30: {len(l30_selected)}")
    
    return selected_features

def save_results(selected_features, all_results_25, all_results_30):
    """Save statistical analysis results"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save selected features (NPZ format for compatibility)
    output_npz = f'/data/llm_addiction/results/statistically_valid_features_{timestamp}.npz'
    
    # Prepare data for NPZ
    l25_features = [f for f in selected_features if f['layer'] == 25]
    l30_features = [f for f in selected_features if f['layer'] == 30]
    
    np.savez(output_npz,
             # Layer 25
             layer_25_indices=np.array([f['feature_id'] for f in l25_features]),
             layer_25_cohen_d=np.array([f['cohen_d'] for f in l25_features]),
             layer_25_p_values=np.array([f['p_value'] for f in l25_features]),
             layer_25_bankrupt_mean=np.array([f['bankrupt_mean'] for f in l25_features]),
             layer_25_safe_mean=np.array([f['safe_mean'] for f in l25_features]),
             layer_25_bankrupt_std=np.array([f['bankrupt_std'] for f in l25_features]),
             layer_25_safe_std=np.array([f['safe_std'] for f in l25_features]),
             # Layer 30
             layer_30_indices=np.array([f['feature_id'] for f in l30_features]),
             layer_30_cohen_d=np.array([f['cohen_d'] for f in l30_features]),
             layer_30_p_values=np.array([f['p_value'] for f in l30_features]),
             layer_30_bankrupt_mean=np.array([f['bankrupt_mean'] for f in l30_features]),
             layer_30_safe_mean=np.array([f['safe_mean'] for f in l30_features]),
             layer_30_bankrupt_std=np.array([f['bankrupt_std'] for f in l30_features]),
             layer_30_safe_std=np.array([f['safe_std'] for f in l30_features]))
    
    # Save detailed JSON results
    output_json = f'/data/llm_addiction/results/statistical_analysis_detailed_{timestamp}.json'
    
    save_data = {
        'timestamp': timestamp,
        'analysis_config': {
            'cohen_d_threshold': 0.3,
            'p_value_threshold': 0.001,
            'fdr_alpha': 0.05,
            'correction_method': 'fdr_bh'
        },
        'summary': {
            'total_features_analyzed': len(all_results_25) + len(all_results_30),
            'selected_features': len(selected_features),
            'layer_25_selected': len(l25_features),
            'layer_30_selected': len(l30_features),
            'selection_rate': len(selected_features) / (len(all_results_25) + len(all_results_30))
        },
        'selected_features': selected_features,
        'all_results_layer_25': all_results_25[:1000],  # Save first 1000 for space
        'all_results_layer_30': all_results_30[:1000]
    }
    
    with open(output_json, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✅ Results saved:")
    print(f"  NPZ file: {output_npz}")
    print(f"  JSON file: {output_json}")
    
    # Print top 10 features by Cohen's d
    print(f"\nTop 10 Features by |Cohen's d|:")
    for i, feat in enumerate(selected_features[:10], 1):
        direction = "↑ Risky" if feat['cohen_d'] > 0 else "↓ Safe"
        print(f"  {i:2d}. L{feat['layer']}-{feat['feature_id']:5d}: "
              f"d={feat['cohen_d']:+.3f} {direction}, p={feat['p_value']:.2e}")
    
    return output_npz, output_json

def main():
    """Main function"""
    print("="*80)
    print("STATISTICALLY VALID SAE FEATURE EXTRACTION")
    print("Proper Cohen's d + t-test + FDR correction")
    print("="*80)
    
    # Load experiments
    experiments = load_experiments()
    
    # Extract last round prompts
    bankrupt_prompts, safe_prompts = extract_last_round_prompts(experiments)
    
    if len(bankrupt_prompts) == 0 or len(safe_prompts) == 0:
        print("❌ Error: No prompts extracted!")
        return
    
    # Extract features
    bankrupt_25, bankrupt_30, safe_25, safe_30 = extract_statistical_features(
        bankrupt_prompts, safe_prompts)
    
    # Statistical analysis
    results_25, results_30 = analyze_features_statistically(
        bankrupt_25, bankrupt_30, safe_25, safe_30)
    
    # Feature selection
    selected_features = apply_statistical_selection(results_25, results_30)
    
    # Save results
    npz_file, json_file = save_results(selected_features, results_25, results_30)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"Selected {len(selected_features)} statistically valid features")
    print(f"(Previous flawed method selected 356 features)")

if __name__ == '__main__':
    main()