#!/usr/bin/env python3
"""
Experiment 6 Token Activation Analysis
Analyzes SAE feature activations at critical token positions across different gambling scenarios
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def load_exp6_data(filepath: str) -> Dict:
    """Load Experiment 6 token-level tracking data"""
    print(f"Loading Experiment 6 data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {data['n_scenarios']} scenarios")
    print(f"Critical layers: {data['critical_layers']}")
    return data

def extract_token_activations(data: Dict, layer: str = 'L31') -> pd.DataFrame:
    """
    Extract feature activations at key token positions for each scenario

    Returns DataFrame with columns:
    - scenario, balance, seq_len
    - balance_pos, goal_pos, prob_pos
    - For each feature: activation at each key position
    """

    activation_records = []

    for result in data['results']:
        scenario = result['scenario']
        balance = result['balance']
        seq_len = result['seq_len']
        key_positions = result['key_positions']

        # Get positions
        balance_pos = key_positions['balance'][0] if key_positions['balance'] else None
        goal_pos = key_positions['goal'][0] if key_positions['goal'] else None
        prob_pos = key_positions['probability'][0] if key_positions['probability'] else None
        choice_positions = key_positions['choices']

        # Get features for this layer
        if layer not in result['layers']:
            continue

        features = result['layers'][layer]['features']  # Shape: [seq_len, 32768]

        # Extract activations at each key position
        record = {
            'scenario': scenario,
            'balance': balance,
            'seq_len': seq_len,
            'balance_pos': balance_pos,
            'goal_pos': goal_pos,
            'prob_pos': prob_pos,
            'choice_positions': choice_positions
        }

        # Store feature activations at each key position
        if balance_pos is not None:
            record['balance_activations'] = features[balance_pos]
        if goal_pos is not None:
            record['goal_activations'] = features[goal_pos]
        if prob_pos is not None:
            record['prob_activations'] = features[prob_pos]

        # Average across choice positions
        if choice_positions:
            choice_acts = [features[pos] for pos in choice_positions]
            record['choice_activations'] = np.mean(choice_acts, axis=0).tolist()

        activation_records.append(record)

    return activation_records

def analyze_feature_at_positions(activation_records: List[Dict], feature_id: int, layer: str) -> pd.DataFrame:
    """
    Analyze a specific feature's activation across scenarios and token positions
    """

    results = []

    for record in activation_records:
        scenario_result = {
            'scenario': record['scenario'],
            'balance': record['balance'],
            'layer': layer,
            'feature_id': feature_id
        }

        # Extract this feature's activation at each position
        if 'balance_activations' in record:
            scenario_result['balance_activation'] = record['balance_activations'][feature_id]
        else:
            scenario_result['balance_activation'] = 0.0

        if 'goal_activations' in record:
            scenario_result['goal_activation'] = record['goal_activations'][feature_id]
        else:
            scenario_result['goal_activation'] = 0.0

        if 'prob_activations' in record:
            scenario_result['prob_activation'] = record['prob_activations'][feature_id]
        else:
            scenario_result['prob_activation'] = 0.0

        if 'choice_activations' in record:
            scenario_result['choice_activation'] = record['choice_activations'][feature_id]
        else:
            scenario_result['choice_activation'] = 0.0

        # Calculate max and mean activation
        acts = [scenario_result['balance_activation'],
                scenario_result['goal_activation'],
                scenario_result['prob_activation'],
                scenario_result['choice_activation']]
        scenario_result['max_activation'] = max(acts)
        scenario_result['mean_activation'] = np.mean(acts)

        results.append(scenario_result)

    return pd.DataFrame(results)

def identify_position_specific_features(activation_records: List[Dict], threshold: float = 5.0) -> Dict:
    """
    Identify features that are specifically active at certain token positions

    Returns dict with keys: balance_specific, goal_specific, prob_specific, choice_specific
    Each contains list of (feature_id, mean_activation) tuples
    """

    n_features = 32768

    # Aggregate activations by position across all scenarios
    balance_acts = defaultdict(list)
    goal_acts = defaultdict(list)
    prob_acts = defaultdict(list)
    choice_acts = defaultdict(list)

    for record in activation_records:
        if 'balance_activations' in record:
            for feat_id, act in enumerate(record['balance_activations']):
                if act > threshold:
                    balance_acts[feat_id].append(act)

        if 'goal_activations' in record:
            for feat_id, act in enumerate(record['goal_activations']):
                if act > threshold:
                    goal_acts[feat_id].append(act)

        if 'prob_activations' in record:
            for feat_id, act in enumerate(record['prob_activations']):
                if act > threshold:
                    prob_acts[feat_id].append(act)

        if 'choice_activations' in record:
            for feat_id, act in enumerate(record['choice_activations']):
                if act > threshold:
                    choice_acts[feat_id].append(act)

    # Calculate means and sort
    balance_specific = [(fid, np.mean(acts)) for fid, acts in balance_acts.items()]
    balance_specific.sort(key=lambda x: x[1], reverse=True)

    goal_specific = [(fid, np.mean(acts)) for fid, acts in goal_acts.items()]
    goal_specific.sort(key=lambda x: x[1], reverse=True)

    prob_specific = [(fid, np.mean(acts)) for fid, acts in prob_acts.items()]
    prob_specific.sort(key=lambda x: x[1], reverse=True)

    choice_specific = [(fid, np.mean(acts)) for fid, acts in choice_acts.items()]
    choice_specific.sort(key=lambda x: x[1], reverse=True)

    return {
        'balance_specific': balance_specific[:100],  # Top 100
        'goal_specific': goal_specific[:100],
        'prob_specific': prob_specific[:100],
        'choice_specific': choice_specific[:100]
    }

def categorize_scenarios(activation_records: List[Dict]) -> Dict[str, List[str]]:
    """
    Categorize scenarios by risk level based on balance
    """

    categories = {
        'desperate': [],      # balance < 30
        'risky': [],          # 30 <= balance < 60
        'medium': [],         # 60 <= balance < 120
        'safe': [],           # 120 <= balance < 180
        'very_safe': [],      # balance >= 180
        'bankrupt_edge': []   # specific bankruptcy scenarios
    }

    for record in activation_records:
        balance = record['balance']
        scenario = record['scenario']

        if 'Bankruptcy' in scenario:
            categories['bankrupt_edge'].append(scenario)
        elif balance < 30:
            categories['desperate'].append(scenario)
        elif balance < 60:
            categories['risky'].append(scenario)
        elif balance < 120:
            categories['medium'].append(scenario)
        elif balance < 180:
            categories['safe'].append(scenario)
        else:
            categories['very_safe'].append(scenario)

    return categories

def main():
    # Paths
    exp6_path = "/data/llm_addiction/experiment_6_token_level/token_level_tracking_20251013_145433.json"
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_5_6_analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_exp6_data(exp6_path)

    # Analyze each layer
    for layer in ['L8', 'L15', 'L31']:
        print(f"\n{'='*80}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*80}")

        # Extract activations
        print(f"\nExtracting token activations for {layer}...")
        activation_records = extract_token_activations(data, layer=layer)

        # Categorize scenarios
        categories = categorize_scenarios(activation_records)
        print(f"\nScenario categories:")
        for cat, scenarios in categories.items():
            if scenarios:
                print(f"  {cat}: {scenarios}")

        # Identify position-specific features
        print(f"\nIdentifying position-specific features...")
        position_features = identify_position_specific_features(activation_records, threshold=5.0)

        print(f"\nTop features by token position:")
        for position, features in position_features.items():
            print(f"\n{position}:")
            for i, (fid, mean_act) in enumerate(features[:10]):
                print(f"  {i+1}. Feature {fid}: {mean_act:.2f}")

            # Save to CSV
            df = pd.DataFrame(features, columns=['feature_id', 'mean_activation'])
            df.to_csv(output_dir / f"exp6_{layer}_{position}_features.csv", index=False)

        print(f"\nSaved position-specific features for {layer}")

    print(f"\n\nAll Experiment 6 analysis saved to: {output_dir}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
