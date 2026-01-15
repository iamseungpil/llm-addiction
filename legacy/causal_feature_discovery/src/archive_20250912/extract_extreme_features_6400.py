#!/usr/bin/env python3
"""
Extract extreme (min/max) SAE features from all 6400 experiments
Instead of averaging, track minimum and maximum values for each feature
in both bankrupt and voluntary stop groups.

Output NPZ format:
- layer_25_indices: feature IDs with significant differences
- layer_25_bankrupt_min/max: min/max values in bankruptcy group
- layer_25_safe_min/max: min/max values in voluntary stop group
- layer_25_bankrupt_mean: mean for comparison
- layer_25_safe_mean: mean for comparison
(same for layer_30)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import torch
import gc
import sys

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_experiments():
    """Load the 6400 experiments"""
    print("Loading experiments...")
    
    results_dir = Path('/data/llm_addiction/results')
    main_file = results_dir / "exp1_multiround_intermediate_20250819_140040.json"
    
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
    missing_file = results_dir / "exp1_missing_complete_20250820_090040.json"
    if missing_file.exists():
        print(f"Loading {missing_file.name}...")
        with open(missing_file, 'r') as f:
            data = json.load(f)
        
        # Handle different formats
        if isinstance(data, dict):
            # Check for numbered keys (0, 1, 2, ...)
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

def extract_extreme_features(experiments):
    """Extract min/max SAE features from last round of each experiment"""
    
    print("\n" + "="*60)
    print("EXTREME SAE FEATURE EXTRACTION")
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
    
    print("Loading SAEs...")
    sae_25 = LlamaScopeDirect(layer=25, device="cuda")
    print("✅ Layer 25 SAE loaded")
    
    sae_30 = LlamaScopeDirect(layer=30, device="cuda")
    print("✅ Layer 30 SAE loaded")
    
    # Collect prompts
    print("\nCollecting last round prompts...")
    bankrupt_prompts = []
    safe_prompts = []
    
    for exp in tqdm(experiments, desc="Collecting prompts"):
        round_features = exp.get('round_features', [])
        if not round_features:
            continue
        
        last_round = round_features[-1]
        prompt = last_round.get('prompt', '')
        
        if not prompt:
            continue
        
        if exp.get('is_bankrupt', False):
            bankrupt_prompts.append(prompt)
        else:
            safe_prompts.append(prompt)
    
    print(f"Collected {len(bankrupt_prompts)} bankrupt, {len(safe_prompts)} safe prompts")
    
    # Initialize tracking arrays for extremes
    n_features = 32768
    
    # Layer 25
    l25_bankrupt_min = np.full(n_features, np.inf)
    l25_bankrupt_max = np.full(n_features, -np.inf)
    l25_bankrupt_sum = np.zeros(n_features)
    l25_bankrupt_count = 0
    
    l25_safe_min = np.full(n_features, np.inf)
    l25_safe_max = np.full(n_features, -np.inf)
    l25_safe_sum = np.zeros(n_features)
    l25_safe_count = 0
    
    # Layer 30
    l30_bankrupt_min = np.full(n_features, np.inf)
    l30_bankrupt_max = np.full(n_features, -np.inf)
    l30_bankrupt_sum = np.zeros(n_features)
    l30_bankrupt_count = 0
    
    l30_safe_min = np.full(n_features, np.inf)
    l30_safe_max = np.full(n_features, -np.inf)
    l30_safe_sum = np.zeros(n_features)
    l30_safe_count = 0
    
    def process_prompt(prompt):
        """Extract features from a single prompt"""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", 
                             truncation=True, max_length=512).to('cuda')
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Layer 25
                hidden_25 = outputs.hidden_states[26]  # +1 for embeddings
                features_25 = sae_25.encode(hidden_25[:, -1:, :].float())
                features_25_np = features_25[0, 0].cpu().numpy()
                
                # Layer 30
                hidden_30 = outputs.hidden_states[31]  # +1 for embeddings
                features_30 = sae_30.encode(hidden_30[:, -1:, :].float())
                features_30_np = features_30[0, 0].cpu().numpy()
                
                return features_25_np, features_30_np
                
        except Exception as e:
            print(f"Error processing prompt: {e}")
            return np.zeros(n_features), np.zeros(n_features)
    
    # Process bankrupt prompts
    print("\nProcessing bankrupt prompts...")
    for prompt in tqdm(bankrupt_prompts, desc="Bankrupt"):
        f25, f30 = process_prompt(prompt)
        
        # Update Layer 25 extremes
        l25_bankrupt_min = np.minimum(l25_bankrupt_min, f25)
        l25_bankrupt_max = np.maximum(l25_bankrupt_max, f25)
        l25_bankrupt_sum += f25
        l25_bankrupt_count += 1
        
        # Update Layer 30 extremes
        l30_bankrupt_min = np.minimum(l30_bankrupt_min, f30)
        l30_bankrupt_max = np.maximum(l30_bankrupt_max, f30)
        l30_bankrupt_sum += f30
        l30_bankrupt_count += 1
        
        # Clear GPU cache periodically
        if l25_bankrupt_count % 50 == 0:
            torch.cuda.empty_cache()
    
    # Process safe prompts
    print("\nProcessing voluntary stop prompts...")
    for prompt in tqdm(safe_prompts, desc="Safe"):
        f25, f30 = process_prompt(prompt)
        
        # Update Layer 25 extremes
        l25_safe_min = np.minimum(l25_safe_min, f25)
        l25_safe_max = np.maximum(l25_safe_max, f25)
        l25_safe_sum += f25
        l25_safe_count += 1
        
        # Update Layer 30 extremes
        l30_safe_min = np.minimum(l30_safe_min, f30)
        l30_safe_max = np.maximum(l30_safe_max, f30)
        l30_safe_sum += f30
        l30_safe_count += 1
        
        # Clear GPU cache periodically
        if l25_safe_count % 50 == 0:
            torch.cuda.empty_cache()
    
    # Calculate means
    l25_bankrupt_mean = l25_bankrupt_sum / max(l25_bankrupt_count, 1)
    l25_safe_mean = l25_safe_sum / max(l25_safe_count, 1)
    l30_bankrupt_mean = l30_bankrupt_sum / max(l30_bankrupt_count, 1)
    l30_safe_mean = l30_safe_sum / max(l30_safe_count, 1)
    
    # Fix infinities (features that were never activated)
    l25_bankrupt_min[np.isinf(l25_bankrupt_min)] = 0
    l25_bankrupt_max[np.isinf(l25_bankrupt_max)] = 0
    l25_safe_min[np.isinf(l25_safe_min)] = 0
    l25_safe_max[np.isinf(l25_safe_max)] = 0
    l30_bankrupt_min[np.isinf(l30_bankrupt_min)] = 0
    l30_bankrupt_max[np.isinf(l30_bankrupt_max)] = 0
    l30_safe_min[np.isinf(l30_safe_min)] = 0
    l30_safe_max[np.isinf(l30_safe_max)] = 0
    
    print(f"\nExtraction complete:")
    print(f"  Bankrupt samples: L25={l25_bankrupt_count}, L30={l30_bankrupt_count}")
    print(f"  Safe samples: L25={l25_safe_count}, L30={l30_safe_count}")
    
    # Find significant features based on extreme differences
    print("\nFinding significant features based on extremes...")
    significant_l25 = []
    significant_l30 = []
    
    # Layer 25: Use max difference between groups
    for idx in range(n_features):
        # Compare extremes
        max_diff = max(
            abs(l25_bankrupt_max[idx] - l25_safe_max[idx]),
            abs(l25_bankrupt_min[idx] - l25_safe_min[idx]),
            abs(l25_bankrupt_max[idx] - l25_safe_min[idx]),
            abs(l25_bankrupt_min[idx] - l25_safe_max[idx])
        )
        
        # Also check mean difference
        mean_diff = abs(l25_bankrupt_mean[idx] - l25_safe_mean[idx])
        
        # Include if extreme difference is large OR mean difference is significant
        if max_diff > 0.5 or mean_diff > 0.1:
            significant_l25.append(idx)
    
    # Layer 30: Same logic
    for idx in range(n_features):
        max_diff = max(
            abs(l30_bankrupt_max[idx] - l30_safe_max[idx]),
            abs(l30_bankrupt_min[idx] - l30_safe_min[idx]),
            abs(l30_bankrupt_max[idx] - l30_safe_min[idx]),
            abs(l30_bankrupt_min[idx] - l30_safe_max[idx])
        )
        
        mean_diff = abs(l30_bankrupt_mean[idx] - l30_safe_mean[idx])
        
        if max_diff > 0.5 or mean_diff > 0.1:
            significant_l30.append(idx)
    
    print(f"Found {len(significant_l25)} significant L25 features")
    print(f"Found {len(significant_l30)} significant L30 features")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'/data/llm_addiction/results/llama_extreme_features_6400_{timestamp}.npz'
    
    np.savez(output_file,
             # Layer 25
             layer_25_indices=np.array(significant_l25),
             layer_25_bankrupt_min=l25_bankrupt_min[significant_l25],
             layer_25_bankrupt_max=l25_bankrupt_max[significant_l25],
             layer_25_bankrupt_mean=l25_bankrupt_mean[significant_l25],
             layer_25_safe_min=l25_safe_min[significant_l25],
             layer_25_safe_max=l25_safe_max[significant_l25],
             layer_25_safe_mean=l25_safe_mean[significant_l25],
             # Layer 30
             layer_30_indices=np.array(significant_l30),
             layer_30_bankrupt_min=l30_bankrupt_min[significant_l30],
             layer_30_bankrupt_max=l30_bankrupt_max[significant_l30],
             layer_30_bankrupt_mean=l30_bankrupt_mean[significant_l30],
             layer_30_safe_min=l30_safe_min[significant_l30],
             layer_30_safe_max=l30_safe_max[significant_l30],
             layer_30_safe_mean=l30_safe_mean[significant_l30])
    
    print(f"\nSaved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EXTREME VALUE STATISTICS")
    print("="*60)
    
    print("\nLayer 25 Top 10 Features by Max Difference:")
    l25_max_diffs = [abs(l25_bankrupt_max[i] - l25_safe_max[i]) for i in significant_l25]
    top_l25 = sorted(zip(significant_l25, l25_max_diffs), key=lambda x: x[1], reverse=True)[:10]
    for feat_id, diff in top_l25:
        print(f"  Feature {feat_id}: max_diff={diff:.3f}")
        print(f"    Bankrupt: min={l25_bankrupt_min[feat_id]:.3f}, max={l25_bankrupt_max[feat_id]:.3f}")
        print(f"    Safe: min={l25_safe_min[feat_id]:.3f}, max={l25_safe_max[feat_id]:.3f}")
    
    print("\nLayer 30 Top 10 Features by Max Difference:")
    l30_max_diffs = [abs(l30_bankrupt_max[i] - l30_safe_max[i]) for i in significant_l30]
    top_l30 = sorted(zip(significant_l30, l30_max_diffs), key=lambda x: x[1], reverse=True)[:10]
    for feat_id, diff in top_l30:
        print(f"  Feature {feat_id}: max_diff={diff:.3f}")
        print(f"    Bankrupt: min={l30_bankrupt_min[feat_id]:.3f}, max={l30_bankrupt_max[feat_id]:.3f}")
        print(f"    Safe: min={l30_safe_min[feat_id]:.3f}, max={l30_safe_max[feat_id]:.3f}")
    
    return output_file

if __name__ == '__main__':
    experiments = load_experiments()
    output_file = extract_extreme_features(experiments)
    print(f"\n✅ Extreme feature extraction complete!")
    print(f"Output: {output_file}")