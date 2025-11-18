#!/usr/bin/env python3
"""
Experiment 1: Extract L1-31 Hidden State Features from 6,400 LLaMA Experiments
Extracts raw hidden states from layers 1-31 and performs statistical analysis
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import torch
import gc
from statsmodels.stats.multitest import multipletests
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
        experiments = data if isinstance(data, list) else []

    all_experiments.extend(experiments)
    print(f"  Loaded {len(experiments)} from main file")
    del data, experiments
    gc.collect()

    # Load missing file (620 experiments)
    print(f"Loading {missing_file.name}...")
    with open(missing_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'results' in data:
        experiments = data['results']
    else:
        experiments = data if isinstance(data, list) else []

    all_experiments.extend(experiments)
    print(f"  Loaded {len(experiments)} from missing file")
    del data, experiments
    gc.collect()

    print(f"Total experiments loaded: {len(all_experiments)}")
    return all_experiments

def load_llama_model():
    """Load LLaMA model"""
    print("Loading LLaMA model...")
    model_name = 'meta-llama/Llama-3.1-8B'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model.eval()
    print("✅ LLaMA model loaded")
    return model, tokenizer

def extract_hidden_states(model, tokenizer, prompt, target_layers):
    """Extract hidden states from specific layers"""
    try:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            features = {}
            for layer in target_layers:
                if layer < len(hidden_states):
                    # Use last token hidden state [4096]
                    layer_hidden = hidden_states[layer][0, -1, :].cpu().numpy()
                    features[f'layer_{layer}'] = layer_hidden

            return features
    except Exception as e:
        print(f"Error extracting hidden states: {e}")
        return {}

def process_experiments_batch(experiments, model, tokenizer, target_layers):
    """Process experiments and extract features for target layers"""
    print(f"Processing {len(experiments)} experiments for layers {target_layers[0]}-{target_layers[-1]}...")

    bankrupt_features = {f'layer_{layer}': [] for layer in target_layers}
    safe_features = {f'layer_{layer}': [] for layer in target_layers}

    processed_count = 0
    error_count = 0

    for exp in tqdm(experiments, desc=f"Extracting L{target_layers[0]}-L{target_layers[-1]}"):
        try:
            # Determine group
            is_bankrupt = exp.get('is_bankrupt', False)

            # Get the final round prompt
            round_features = exp.get('round_features', [])
            if not round_features:
                continue

            final_round = round_features[-1]
            prompt = final_round.get('prompt', '')

            if not prompt:
                continue

            # Extract hidden states
            features = extract_hidden_states(model, tokenizer, prompt, target_layers)

            if not features:
                error_count += 1
                continue

            # Store in appropriate group
            for layer in target_layers:
                layer_key = f'layer_{layer}'
                if layer_key in features:
                    if is_bankrupt:
                        bankrupt_features[layer_key].append(features[layer_key])
                    else:
                        safe_features[layer_key].append(features[layer_key])

            processed_count += 1

            # Memory cleanup every 100 experiments
            if processed_count % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"Error processing experiment: {e}")
            continue

    print(f"Successfully processed {processed_count} experiments ({error_count} errors)")

    # Convert to numpy arrays
    for layer in target_layers:
        layer_key = f'layer_{layer}'
        if bankrupt_features[layer_key]:
            bankrupt_features[layer_key] = np.array(bankrupt_features[layer_key])
        if safe_features[layer_key]:
            safe_features[layer_key] = np.array(safe_features[layer_key])

    return bankrupt_features, safe_features

def analyze_layer_features(bankrupt_features, safe_features, layer, p_threshold=0.01, cohen_d_threshold=0.3):
    """Analyze features for a single layer with statistical tests"""
    layer_key = f'layer_{layer}'

    if layer_key not in bankrupt_features or layer_key not in safe_features:
        return None

    bankrupt_data = bankrupt_features[layer_key]
    safe_data = safe_features[layer_key]

    if len(bankrupt_data) == 0 or len(safe_data) == 0:
        return None

    print(f"\nAnalyzing Layer {layer}:")
    print(f"  Bankrupt samples: {len(bankrupt_data)}")
    print(f"  Safe samples: {len(safe_data)}")
    print(f"  Feature dimensions: {bankrupt_data.shape[1]}")

    n_features = bankrupt_data.shape[1]
    significant_features = []
    p_values = []
    cohen_d_values = []

    for feature_idx in tqdm(range(n_features), desc=f"Analyzing Layer {layer}"):
        bankrupt_values = bankrupt_data[:, feature_idx]
        safe_values = safe_data[:, feature_idx]

        # Skip if no variance
        if np.std(bankrupt_values) == 0 and np.std(safe_values) == 0:
            p_values.append(1.0)
            cohen_d_values.append(0.0)
            continue

        # t-test
        try:
            t_stat, p_value = stats.ttest_ind(bankrupt_values, safe_values)

            # Cohen's d
            pooled_std = np.sqrt(
                ((len(bankrupt_values) - 1) * np.var(bankrupt_values, ddof=1) +
                 (len(safe_values) - 1) * np.var(safe_values, ddof=1)) /
                (len(bankrupt_values) + len(safe_values) - 2)
            )

            if pooled_std > 0:
                cohen_d = (np.mean(bankrupt_values) - np.mean(safe_values)) / pooled_std
            else:
                cohen_d = 0.0

            p_values.append(p_value)
            cohen_d_values.append(cohen_d)

            # Check significance
            if p_value < p_threshold and abs(cohen_d) > cohen_d_threshold:
                significant_features.append({
                    'feature_idx': feature_idx,
                    'p_value': float(p_value),
                    'cohen_d': float(cohen_d),
                    'bankrupt_mean': float(np.mean(bankrupt_values)),
                    'safe_mean': float(np.mean(safe_values)),
                    'bankrupt_std': float(np.std(bankrupt_values)),
                    'safe_std': float(np.std(safe_values))
                })

        except Exception as e:
            p_values.append(1.0)
            cohen_d_values.append(0.0)

    # Multiple testing correction (FDR)
    if p_values:
        rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

        # Filter significant features with corrected p-values
        corrected_significant = []
        for feature in significant_features:
            feature_idx = feature['feature_idx']
            if p_corrected[feature_idx] < p_threshold:
                feature['p_corrected'] = float(p_corrected[feature_idx])
                corrected_significant.append(feature)

        significant_features = corrected_significant

    print(f"  Found {len(significant_features)} significant features (after FDR correction)")

    return {
        'layer': layer,
        'n_features': n_features,
        'n_bankrupt': int(len(bankrupt_data)),
        'n_safe': int(len(safe_data)),
        'significant_features': significant_features,
        'n_significant': len(significant_features)
    }

def save_intermediate_results(all_results, experiments_count, output_dir):
    """Save intermediate results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'L1_31_features_intermediate_{timestamp}.json'

    total_significant = sum(result['n_significant'] for result in all_results.values())

    summary = {
        'timestamp': timestamp,
        'total_experiments_processed': experiments_count,
        'layers_analyzed': sorted(list(all_results.keys())),
        'total_layers': len(all_results),
        'total_significant_features': total_significant,
        'significant_features_by_layer': {str(layer): result['n_significant'] for layer, result in all_results.items()},
        'layer_results': {str(k): v for k, v in all_results.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Intermediate results saved: {output_file}")
    return output_file

def main():
    print("🚀 Experiment 1: L1-31 Hidden State Feature Extraction")
    print("="*80)

    # Setup
    output_dir = Path('/data/llm_addiction/experiment_1_L1_31_extraction')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiments
    experiments = load_experiments()

    # Load model
    model, tokenizer = load_llama_model()

    # Process layers in 3 batches to manage memory
    all_results = {}

    # Batch 1: Layers 1-10
    print("\n" + "="*80)
    print("BATCH 1: Layers 1-10")
    print("="*80)
    target_layers = list(range(1, 11))
    bankrupt_features, safe_features = process_experiments_batch(
        experiments, model, tokenizer, target_layers
    )

    for layer in target_layers:
        result = analyze_layer_features(bankrupt_features, safe_features, layer)
        if result:
            all_results[layer] = result

    del bankrupt_features, safe_features
    torch.cuda.empty_cache()
    gc.collect()

    save_intermediate_results(all_results, len(experiments), output_dir)

    # Batch 2: Layers 11-20
    print("\n" + "="*80)
    print("BATCH 2: Layers 11-20")
    print("="*80)
    target_layers = list(range(11, 21))
    bankrupt_features, safe_features = process_experiments_batch(
        experiments, model, tokenizer, target_layers
    )

    for layer in target_layers:
        result = analyze_layer_features(bankrupt_features, safe_features, layer)
        if result:
            all_results[layer] = result

    del bankrupt_features, safe_features
    torch.cuda.empty_cache()
    gc.collect()

    save_intermediate_results(all_results, len(experiments), output_dir)

    # Batch 3: Layers 21-31
    print("\n" + "="*80)
    print("BATCH 3: Layers 21-31")
    print("="*80)
    target_layers = list(range(21, 32))
    bankrupt_features, safe_features = process_experiments_batch(
        experiments, model, tokenizer, target_layers
    )

    for layer in target_layers:
        result = analyze_layer_features(bankrupt_features, safe_features, layer)
        if result:
            all_results[layer] = result

    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'L1_31_features_FINAL_{timestamp}.json'

    total_significant = sum(result['n_significant'] for result in all_results.values())

    summary = {
        'timestamp': timestamp,
        'total_experiments_processed': len(experiments),
        'layers_analyzed': sorted(list(all_results.keys())),
        'total_layers': len(all_results),
        'total_significant_features': total_significant,
        'significant_features_by_layer': {str(layer): result['n_significant'] for layer, result in all_results.items()},
        'layer_results': {str(k): v for k, v in all_results.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("✅ Experiment 1 COMPLETE!")
    print("="*80)
    print(f"Layers analyzed: {len(all_results)}")
    print(f"Total significant features: {total_significant}")
    print(f"Results saved: {output_file}")

    # Print summary by layer
    print(f"\n📊 Significant features by layer:")
    for layer in sorted(all_results.keys()):
        result = all_results[layer]
        print(f"  Layer {layer:2d}: {result['n_significant']:4d} features")

    return all_results

if __name__ == '__main__':
    main()