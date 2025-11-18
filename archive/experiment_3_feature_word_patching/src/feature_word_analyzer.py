#!/usr/bin/env python3
"""
Feature-Word Association Analysis for 2,787 Causal Features
Using Experiment 2 Patching Data (1M+ trials)

Method: High vs Low activation word frequency comparison
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import re

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class FeatureWordAnalyzer:
    """
    Analyze word associations for 2,787 causal features
    Using Experiment 2 Patching response data
    """

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}'

        # Paths
        self.safe_features_csv = Path("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv")
        self.risky_features_csv = Path("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv")

        self.exp2_response_dir = Path("/data/llm_addiction/experiment_2_multilayer_patching/response_logs")

        self.results_dir = Path("/home/ubuntu/llm_addiction/experiment_3_feature_word_patching/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load causal features
        print("Loading 2,787 causal features...")
        self.safe_features = pd.read_csv(self.safe_features_csv)
        self.risky_features = pd.read_csv(self.risky_features_csv)

        self.all_features = self._prepare_features()
        print(f"  Total: {len(self.all_features)}")

        # Load Exp2 response data
        print("\nLoading Experiment 2 response logs...")
        self.exp2_data = self._load_exp2_responses()
        print(f"  Total trials loaded: {len(self.exp2_data):,}")

    def _prepare_features(self):
        """Prepare 2,787 features list"""
        all_features = []

        for _, row in self.safe_features.iterrows():
            layer, feat_id = self._parse_feature(row['feature'])
            all_features.append({
                'feature': row['feature'],
                'type': 'safe',
                'layer': layer,
                'feature_id': feat_id
            })

        for _, row in self.risky_features.iterrows():
            layer, feat_id = self._parse_feature(row['feature'])
            all_features.append({
                'feature': row['feature'],
                'type': 'risky',
                'layer': layer,
                'feature_id': feat_id
            })

        all_features.sort(key=lambda x: (x['layer'], x['feature_id']))
        return all_features

    def _parse_feature(self, feature_str):
        """Parse 'L25-1234' -> (25, 1234)"""
        parts = feature_str.split('-')
        layer = int(parts[0][1:])
        feat_id = int(parts[1])
        return layer, feat_id

    def _load_exp2_responses(self):
        """
        Load Experiment 2 response logs
        Structure: {feature: [responses...]}
        """
        feature_responses = {}

        log_files = sorted(self.exp2_response_dir.glob("*.json"))

        for log_file in tqdm(log_files, desc="Loading response logs"):
            with open(log_file, 'r') as f:
                data = json.load(f)

            for record in data:
                feature = record['feature']
                response = record['response']

                if feature not in feature_responses:
                    feature_responses[feature] = []

                feature_responses[feature].append({
                    'condition': record['condition'],
                    'trial': record['trial'],
                    'response': response
                })

        return feature_responses

    def extract_words(self, text):
        """Extract words from response text"""
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Remove very common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'}

        words = [w for w in words if w not in stopwords and len(w) > 2]

        return words

    def analyze_feature_words(self, feature_info):
        """
        Analyze word associations for a specific feature

        Method:
        1. Get all responses for this feature
        2. Need to extract SAE activations for each response
           (NOT stored in response logs!)
        3. Split by high/low activation
        4. Compare word frequencies

        Problem: We don't have SAE activations stored!
        """

        feature = feature_info['feature']

        if feature not in self.exp2_data:
            print(f"  Warning: {feature} not in response logs")
            return None

        responses = self.exp2_data[feature]

        # PROBLEM: We need SAE activations to split high/low groups
        # But response logs don't have them!

        # ALTERNATIVE: Use condition-based grouping
        # safe_baseline vs risky_baseline for activation proxy

        print(f"\n⚠️  Cannot compute high/low activation without SAE features!")
        print(f"  Alternative: Compare safe_baseline vs risky_baseline conditions")

        return None

    def run_analysis(self):
        """
        Main analysis

        ISSUE DISCOVERED: We need SAE activations for each response!
        But Exp2 response logs only have text, not activations.

        Solutions:
        1. Re-run inference to get activations (slow)
        2. Use condition as proxy (less precise)
        3. Use Exp1 data instead (different experiment)
        """

        print(f"\n{'='*80}")
        print(f"CRITICAL ISSUE DISCOVERED")
        print(f"{'='*80}")
        print(f"\nExp2 response logs contain:")
        print(f"  ✅ Response text")
        print(f"  ❌ SAE feature activations (NOT STORED!)")
        print(f"\nWord analysis requires:")
        print(f"  ✅ Response text")
        print(f"  ✅ SAE activation values (to split high/low groups)")
        print(f"\nPossible solutions:")
        print(f"  1. Re-extract SAE activations from responses (SLOW: ~5-10 hours)")
        print(f"  2. Use condition as activation proxy (IMPRECISE)")
        print(f"  3. Use Exp1 data (DIFFERENT: no feature manipulation)")
        print(f"\nRecommendation: Option 1 (re-extract activations)")
        print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=4, help='GPU ID')
    args = parser.parse_args()

    analyzer = FeatureWordAnalyzer(gpu_id=args.gpu)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
