#!/usr/bin/env python3
"""
Optimized extreme feature extraction focusing on the 356 significant features only.
This version:
1. Only extracts the 356 features we care about (not all 32768)
2. Uses the existing NPZ indices
3. Runs faster by focusing on significant features
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import gc
import sys

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_target_features():
    """Load the 356 features we want to track"""
    npz_file = '/data/llm_addiction/results/llama_feature_arrays_20250829_150110_v2.npz'
    data = np.load(npz_file)
    
    l25_indices = data['layer_25_indices']
    l30_indices = data['layer_30_indices']
    
    print(f"Target features: L25={len(l25_indices)}, L30={len(l30_indices)}")
    return l25_indices, l30_indices

def load_experiments():
    """Load the 6400 experiments"""
    print("Loading experiments...")
    
    results_dir = Path('/data/llm_addiction/results')
    main_file = results_dir / "exp1_multiround_intermediate_20250819_140040.json"
    
    all_experiments = []
    
    # Load main file
    print(f"Loading main file...")
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
    
    # Load missing file
    missing_file = results_dir / "exp1_missing_complete_20250820_090040.json"
    if missing_file.exists():
        print(f"Loading missing file...")
        with open(missing_file, 'r') as f:
            data = json.load(f)
        
        # Handle different formats
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

def extract_extreme_features_optimized(experiments, sample_size=None):
    """
    Extract extreme features for the 356 target features only.
    sample_size: If provided, only process this many prompts per group (for testing)
    """
    
    # Load target features
    l25_indices, l30_indices = load_target_features()
    
    print("\n" + "="*60)
    print("OPTIMIZED EXTREME FEATURE EXTRACTION")
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
    print("\nCollecting decision moment prompts...")
    bankrupt_prompts = []
    safe_prompts = []
    
    for exp in tqdm(experiments, desc="Collecting prompts"):
        round_features = exp.get('round_features', [])
        if not round_features:
            continue
        
        # Get the last round (decision moment)
        last_round = round_features[-1]
        prompt = last_round.get('prompt', '')
        
        if not prompt:
            continue
        
        if exp.get('is_bankrupt', False):
            bankrupt_prompts.append(prompt)
        else:
            safe_prompts.append(prompt)
    
    print(f"Collected {len(bankrupt_prompts)} bankrupt, {len(safe_prompts)} safe prompts")
    
    # Sample if requested (for testing)
    if sample_size:
        bankrupt_prompts = bankrupt_prompts[:sample_size]
        safe_prompts = safe_prompts[:sample_size]
        print(f"Sampling {sample_size} prompts per group for testing")
    
    # Initialize tracking arrays for extremes (only for target features)
    n_l25 = len(l25_indices)
    n_l30 = len(l30_indices)
    
    # Layer 25
    l25_bankrupt_min = np.full(n_l25, np.inf)
    l25_bankrupt_max = np.full(n_l25, -np.inf)
    l25_bankrupt_sum = np.zeros(n_l25)
    
    l25_safe_min = np.full(n_l25, np.inf)
    l25_safe_max = np.full(n_l25, -np.inf)
    l25_safe_sum = np.zeros(n_l25)
    
    # Layer 30
    l30_bankrupt_min = np.full(n_l30, np.inf)
    l30_bankrupt_max = np.full(n_l30, -np.inf)
    l30_bankrupt_sum = np.zeros(n_l30)
    
    l30_safe_min = np.full(n_l30, np.inf)
    l30_safe_max = np.full(n_l30, -np.inf)
    l30_safe_sum = np.zeros(n_l30)
    
    def process_prompt(prompt):
        """Extract only target features from a single prompt"""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", 
                             truncation=True, max_length=512).to('cuda')
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Layer 25 - extract only target features
                hidden_25 = outputs.hidden_states[26]  # +1 for embeddings
                # Convert to float32 before encoding to avoid BFloat16 issues
                hidden_25_float = hidden_25[:, -1:, :].float()
                features_25 = sae_25.encode(hidden_25_float)
                # Ensure we're working with float32
                features_25_selected = features_25[0, 0, l25_indices].float().cpu().numpy()
                
                # Layer 30 - extract only target features  
                hidden_30 = outputs.hidden_states[31]  # +1 for embeddings
                # Convert to float32 before encoding to avoid BFloat16 issues
                hidden_30_float = hidden_30[:, -1:, :].float()
                features_30 = sae_30.encode(hidden_30_float)
                # Ensure we're working with float32
                features_30_selected = features_30[0, 0, l30_indices].float().cpu().numpy()
                
                return features_25_selected, features_30_selected
                
        except Exception as e:
            print(f"Error processing prompt: {e}")
            return np.zeros(n_l25), np.zeros(n_l30)
    
    # Process bankrupt prompts
    print("\nProcessing bankrupt prompts...")
    n_bankrupt = 0
    for prompt in tqdm(bankrupt_prompts, desc="Bankrupt"):
        f25, f30 = process_prompt(prompt)
        
        # Update Layer 25 extremes
        l25_bankrupt_min = np.minimum(l25_bankrupt_min, f25)
        l25_bankrupt_max = np.maximum(l25_bankrupt_max, f25)
        l25_bankrupt_sum += f25
        
        # Update Layer 30 extremes
        l30_bankrupt_min = np.minimum(l30_bankrupt_min, f30)
        l30_bankrupt_max = np.maximum(l30_bankrupt_max, f30)
        l30_bankrupt_sum += f30
        
        n_bankrupt += 1
        
        # Clear GPU cache periodically
        if n_bankrupt % 50 == 0:
            torch.cuda.empty_cache()
    
    # Process safe prompts
    print("\nProcessing voluntary stop prompts...")
    n_safe = 0
    for prompt in tqdm(safe_prompts, desc="Safe"):
        f25, f30 = process_prompt(prompt)
        
        # Update Layer 25 extremes
        l25_safe_min = np.minimum(l25_safe_min, f25)
        l25_safe_max = np.maximum(l25_safe_max, f25)
        l25_safe_sum += f25
        
        # Update Layer 30 extremes
        l30_safe_min = np.minimum(l30_safe_min, f30)
        l30_safe_max = np.maximum(l30_safe_max, f30)
        l30_safe_sum += f30
        
        n_safe += 1
        
        # Clear GPU cache periodically
        if n_safe % 50 == 0:
            torch.cuda.empty_cache()
    
    # Calculate means
    l25_bankrupt_mean = l25_bankrupt_sum / max(n_bankrupt, 1)
    l25_safe_mean = l25_safe_sum / max(n_safe, 1)
    l30_bankrupt_mean = l30_bankrupt_sum / max(n_bankrupt, 1)
    l30_safe_mean = l30_safe_sum / max(n_safe, 1)
    
    # Fix infinities
    l25_bankrupt_min[np.isinf(l25_bankrupt_min)] = 0
    l25_bankrupt_max[np.isinf(l25_bankrupt_max)] = 0
    l25_safe_min[np.isinf(l25_safe_min)] = 0
    l25_safe_max[np.isinf(l25_safe_max)] = 0
    l30_bankrupt_min[np.isinf(l30_bankrupt_min)] = 0
    l30_bankrupt_max[np.isinf(l30_bankrupt_max)] = 0
    l30_safe_min[np.isinf(l30_safe_min)] = 0
    l30_safe_max[np.isinf(l30_safe_max)] = 0
    
    print(f"\nExtraction complete:")
    print(f"  Bankrupt samples: {n_bankrupt}")
    print(f"  Safe samples: {n_safe}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if sample_size:
        output_file = f'/data/llm_addiction/results/llama_extreme_features_sample_{timestamp}.npz'
    else:
        output_file = f'/data/llm_addiction/results/llama_extreme_features_6400_{timestamp}.npz'
    
    np.savez(output_file,
             # Layer 25
             layer_25_indices=l25_indices,
             layer_25_bankrupt_min=l25_bankrupt_min,
             layer_25_bankrupt_max=l25_bankrupt_max,
             layer_25_bankrupt_mean=l25_bankrupt_mean,
             layer_25_safe_min=l25_safe_min,
             layer_25_safe_max=l25_safe_max,
             layer_25_safe_mean=l25_safe_mean,
             # Layer 30
             layer_30_indices=l30_indices,
             layer_30_bankrupt_min=l30_bankrupt_min,
             layer_30_bankrupt_max=l30_bankrupt_max,
             layer_30_bankrupt_mean=l30_bankrupt_mean,
             layer_30_safe_min=l30_safe_min,
             layer_30_safe_max=l30_safe_max,
             layer_30_safe_mean=l30_safe_mean)
    
    print(f"\nSaved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EXTREME VALUE STATISTICS")
    print("="*60)
    
    print("\nLayer 25 Analysis:")
    max_diffs = np.abs(l25_bankrupt_max - l25_safe_max)
    top_indices = np.argsort(max_diffs)[-10:][::-1]
    print("Top 10 features by max difference:")
    for i in top_indices:
        feat_id = l25_indices[i]
        print(f"  Feature {feat_id}: max_diff={max_diffs[i]:.3f}")
        print(f"    Bankrupt: min={l25_bankrupt_min[i]:.3f}, max={l25_bankrupt_max[i]:.3f}, mean={l25_bankrupt_mean[i]:.3f}")
        print(f"    Safe: min={l25_safe_min[i]:.3f}, max={l25_safe_max[i]:.3f}, mean={l25_safe_mean[i]:.3f}")
    
    print("\nLayer 30 Analysis:")
    max_diffs = np.abs(l30_bankrupt_max - l30_safe_max)
    top_indices = np.argsort(max_diffs)[-10:][::-1]
    print("Top 10 features by max difference:")
    for i in top_indices:
        feat_id = l30_indices[i]
        print(f"  Feature {feat_id}: max_diff={max_diffs[i]:.3f}")
        print(f"    Bankrupt: min={l30_bankrupt_min[i]:.3f}, max={l30_bankrupt_max[i]:.3f}, mean={l30_bankrupt_mean[i]:.3f}")
        print(f"    Safe: min={l30_safe_min[i]:.3f}, max={l30_safe_max[i]:.3f}, mean={l30_safe_mean[i]:.3f}")
    
    return output_file

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None, 
                       help='Number of samples per group (for testing)')
    args = parser.parse_args()
    
    experiments = load_experiments()
    output_file = extract_extreme_features_optimized(experiments, sample_size=args.sample)
    print(f"\n✅ Extreme feature extraction complete!")
    print(f"Output: {output_file}")