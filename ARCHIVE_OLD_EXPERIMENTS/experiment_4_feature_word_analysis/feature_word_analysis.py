#!/usr/bin/env python3
"""
Experiment 4: Feature Word Analysis
Analyzes 441 causal features using 3 methods:
1. SAE decoder weight analysis (which tokens activate each feature)
2. Response text pattern analysis (word frequency differences between safe/risky)
3. Automatic interpretation generation
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict
import re
from tqdm import tqdm
import sys

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_causal_features():
    """Load list of 441 causal features from CSV (361 safe + 80 risky)"""
    print("Loading causal features...")

    import pandas as pd
    csv_file = Path('/home/ubuntu/llm_addiction/analysis/exp2_feature_group_summary.csv')
    df = pd.read_csv(csv_file)

    # Filter for causal features only (exclude neutral)
    causal_df = df[df['classified_as'].isin(['safe', 'risky'])]
    print(f"Loaded {len(causal_df)} causal features from CSV")

    all_causal_features = []
    for _, row in causal_df.iterrows():
        feature_string = row['feature']
        layer = int(feature_string.split('-')[0][1:])  # Extract layer from "L25-1234"
        feature_id = int(feature_string.split('-')[1])  # Extract feature_id

        all_causal_features.append({
            'layer': layer,
            'feature_id': feature_id,
            'feature_string': feature_string,
            'classification': row['classified_as'],
            'safe_stop_delta': row.get('safe_stop_delta', 0),
            'risky_stop_delta': row.get('risky_stop_delta', 0),
            'safe_bankruptcy_delta': row.get('safe_bankruptcy_delta', 0),
            'risky_bankruptcy_delta': row.get('risky_bankruptcy_delta', 0)
        })

    print(f"Loaded {len(all_causal_features)} unique causal features")
    print(f"  Safe features: {sum(1 for f in all_causal_features if f['classification'] == 'safe')}")
    print(f"  Risky features: {sum(1 for f in all_causal_features if f['classification'] == 'risky')}")
    return all_causal_features

def load_response_logs():
    """Load exp2_response_log_*.json files (202 files)"""
    print("Loading response logs...")

    results_dir = Path('/data/llm_addiction/results')
    response_files = list(results_dir.glob('exp2_response_log_*.json'))

    print(f"Found {len(response_files)} response log files")

    all_responses = []
    for file in tqdm(response_files, desc="Loading response logs"):
        with open(file, 'r') as f:
            data = json.load(f)
        all_responses.extend(data if isinstance(data, list) else [data])

    print(f"Loaded {len(all_responses)} total responses")
    return all_responses

def method1_decoder_analysis(feature, sae, model, tokenizer):
    """Method 1: Analyze SAE decoder weights to find top tokens"""
    layer = feature['layer']
    feature_id = feature['feature_id']

    try:
        # Get decoder weight for this feature (W_D is decoder weight matrix)
        decoder_weight = sae[layer].sae.W_D[feature_id].detach().cpu().numpy()  # [4096]

        # Get token embeddings from the model (not tokenizer)
        embedding_weight = model.get_input_embeddings().weight.detach().cpu().numpy()  # [vocab_size, 4096]

        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embedding_weight, decoder_weight.reshape(1, -1)).flatten()

        # Get top 50 tokens
        top_indices = np.argsort(similarities)[-50:][::-1]
        top_tokens = [(tokenizer.decode([idx]), float(similarities[idx])) for idx in top_indices]

        return {
            'method': 'decoder_weight_analysis',
            'top_tokens': top_tokens,
            'top_words': [token for token, _ in top_tokens[:20]]
        }

    except Exception as e:
        print(f"Error in decoder analysis for {feature['feature_string']}: {e}")
        return {'method': 'decoder_weight_analysis', 'error': str(e), 'top_words': []}

def method2_response_pattern_analysis(feature, responses):
    """Method 2: Analyze response text patterns for safe vs risky conditions"""
    feature_string = feature['feature_string']

    # Filter responses for this feature
    feature_responses = [r for r in responses if r.get('feature') == feature_string]

    if len(feature_responses) == 0:
        return {
            'method': 'response_pattern_analysis',
            'error': 'No responses found',
            'safe_words': [],
            'risky_words': [],
            'differentiating_words': []
        }

    # Split by safe/risky patching
    safe_responses = [r['response'] for r in feature_responses if 'safe_patch' in r.get('condition', '')]
    risky_responses = [r['response'] for r in feature_responses if 'risky_patch' in r.get('condition', '')]

    # Tokenize and count words
    def extract_words(texts):
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            all_words.extend(words)
        return Counter(all_words)

    safe_word_counts = extract_words(safe_responses)
    risky_word_counts = extract_words(risky_responses)

    # Find differentiating words (>1.5x frequency difference)
    differentiating_words = []
    all_words = set(safe_word_counts.keys()) | set(risky_word_counts.keys())

    for word in all_words:
        safe_count = safe_word_counts.get(word, 0)
        risky_count = risky_word_counts.get(word, 0)

        if safe_count > 0 and risky_count > 0:
            ratio = safe_count / risky_count if safe_count > risky_count else risky_count / safe_count
            if ratio > 1.5:
                differentiating_words.append({
                    'word': word,
                    'safe_count': safe_count,
                    'risky_count': risky_count,
                    'ratio': float(ratio),
                    'direction': 'safe' if safe_count > risky_count else 'risky'
                })

    # Sort by ratio
    differentiating_words.sort(key=lambda x: x['ratio'], reverse=True)

    return {
        'method': 'response_pattern_analysis',
        'n_safe_responses': len(safe_responses),
        'n_risky_responses': len(risky_responses),
        'top_safe_words': [w for w, _ in safe_word_counts.most_common(20)],
        'top_risky_words': [w for w, _ in risky_word_counts.most_common(20)],
        'differentiating_words': differentiating_words[:20]
    }

def method3_auto_interpretation(decoder_analysis, pattern_analysis, feature):
    """Method 3: Generate automatic interpretation based on methods 1 and 2"""
    decoder_words = set(decoder_analysis.get('top_words', []))
    pattern_results = pattern_analysis.get('differentiating_words', [])

    safe_words = set(w['word'] for w in pattern_results if w.get('direction') == 'safe')
    risky_words = set(w['word'] for w in pattern_results if w.get('direction') == 'risky')

    # Rule-based interpretation
    interpretation = []
    confidence = 0.0

    # Safety/risk keywords
    safety_keywords = {'stop', 'quit', 'careful', 'safe', 'conservative', 'cautious', 'avoid', 'risk'}
    risk_keywords = {'bet', 'gamble', 'continue', 'try', 'win', 'maximize', 'aggressive', 'opportunity'}

    if safe_words & safety_keywords:
        interpretation.append("Loss Aversion / Stop Signal")
        confidence += 0.3

    if risky_words & risk_keywords:
        interpretation.append("Risk-Taking / Gambling Tendency")
        confidence += 0.3

    if decoder_words & {'stop', 'quit', 'end'}:
        interpretation.append("Stop Decision Feature")
        confidence += 0.2

    if decoder_words & {'bet', 'risk', 'gamble'}:
        interpretation.append("Betting Decision Feature")
        confidence += 0.2

    # Feature effect direction (use classification)
    classification = feature.get('classification', 'neutral')
    if classification == 'safe':
        interpretation.append("Promotes Safe Behavior")
        confidence += 0.1
    elif classification == 'risky':
        interpretation.append("Promotes Risky Behavior")
        confidence += 0.1

    if not interpretation:
        interpretation = ["Unknown / Ambiguous"]
        confidence = 0.1

    return {
        'method': 'automatic_interpretation',
        'interpretation': ' | '.join(interpretation),
        'confidence': min(confidence, 1.0),
        'supporting_evidence': {
            'decoder_words_sample': list(decoder_words)[:10],
            'safe_words_sample': list(safe_words)[:5],
            'risky_words_sample': list(risky_words)[:5]
        }
    }

def main():
    print("🚀 Experiment 4: Feature Word Analysis")
    print("="*80)

    output_dir = Path('/data/llm_addiction/experiment_4_feature_word_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    causal_features = load_causal_features()
    responses = load_response_logs()

    # Load models
    print("Loading LLaMA and SAEs...")
    model_name = 'meta-llama/Llama-3.1-8B'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    # Load SAEs for L25-31
    sae = {}
    for layer in range(25, 32):
        print(f"Loading SAE for layer {layer}...")
        sae[layer] = LlamaScopeWorking(layer=layer, device='cuda:0')

    print("✅ Models loaded")

    # Analyze each feature
    all_results = []

    for feature in tqdm(causal_features, desc="Analyzing features"):
        feature_result = {
            'feature': feature['feature_string'],
            'layer': feature['layer'],
            'feature_id': feature['feature_id'],
            'classification': feature['classification'],
            'safe_stop_delta': feature['safe_stop_delta'],
            'risky_stop_delta': feature['risky_stop_delta'],
            'safe_bankruptcy_delta': feature['safe_bankruptcy_delta'],
            'risky_bankruptcy_delta': feature['risky_bankruptcy_delta']
        }

        # Method 1: Decoder analysis
        if feature['layer'] >= 25:  # Only for SAE layers
            decoder_result = method1_decoder_analysis(feature, sae, model, tokenizer)
            feature_result['decoder_analysis'] = decoder_result
        else:
            feature_result['decoder_analysis'] = {'method': 'decoder_weight_analysis', 'error': 'No SAE for this layer'}

        # Method 2: Response pattern analysis
        pattern_result = method2_response_pattern_analysis(feature, responses)
        feature_result['pattern_analysis'] = pattern_result

        # Method 3: Auto interpretation
        auto_result = method3_auto_interpretation(
            feature_result.get('decoder_analysis', {}),
            pattern_result,
            feature
        )
        feature_result['auto_interpretation'] = auto_result

        all_results.append(feature_result)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'feature_word_analysis_{timestamp}.json'

    summary = {
        'timestamp': timestamp,
        'total_features_analyzed': len(all_results),
        'analysis_results': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Experiment 4 complete!")
    print(f"Results saved: {output_file}")
    print(f"Total features analyzed: {len(all_results)}")

if __name__ == '__main__':
    main()