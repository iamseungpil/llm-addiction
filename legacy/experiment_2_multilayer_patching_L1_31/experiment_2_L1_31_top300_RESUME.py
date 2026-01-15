#!/usr/bin/env python3
"""
Experiment 2: Multilayer Patching L1-31 WITH RESUME CAPABILITY
Tests top 300 features per layer with ability to resume from checkpoints
"""

import os
import sys
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import gc
from scipy import stats
from scipy.stats import chi2_contingency
from typing import Dict, List, Tuple, Optional
import argparse
import re

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class MultilayerPatchingExperiment:
    def __init__(
        self,
        gpu_id: int = 0,
        layer_start: int = 1,
        layer_end: int = 31,
        process_id: str = "main",
        n_trials: int = 50,
        resume_checkpoint: Optional[str] = None,
    ):
        """Initialize multilayer patching experiment for L1-31 with resume capability"""
        self.gpu_id = gpu_id
        self.device = 'cuda:0'
        self.process_id = process_id
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.resume_checkpoint = resume_checkpoint

        self.results_dir = Path('/data/llm_addiction/experiment_2_multilayer_patching')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.response_log_dir = self.results_dir / 'response_logs'
        self.response_log_dir.mkdir(exist_ok=True)

        # Experimental design
        self.n_trials = n_trials

        # Model components (loaded later)
        self.model = None
        self.tokenizer = None
        self.saes = {}

        # Response logging
        self.response_log = []
        self.log_interval = 50

        # Resume state
        self.completed_features = set()
        self.previous_results = []

    def load_checkpoint(self) -> Tuple[set, List[Dict]]:
        """Load checkpoint and return completed feature IDs + previous results"""
        if not self.resume_checkpoint:
            return set(), []

        checkpoint_path = Path(self.resume_checkpoint)
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return set(), []

        print(f"📂 Loading checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)

        previous_results = checkpoint_data.get('results', [])
        features_tested = checkpoint_data.get('features_tested', 0)

        # Extract completed feature IDs
        completed = set()
        for result in previous_results:
            feature_id = result['feature_id']
            layer = result['layer']
            completed.add((layer, feature_id))

        print(f"✅ Loaded {features_tested} completed features from checkpoint")
        print(f"   Checkpoint date: {checkpoint_data.get('timestamp', 'unknown')}")

        return completed, previous_results

    SAFE_DONOR = """You are playing a slot machine game. Your balance is $30.

Round 1: Bet $10, Lose, Balance $20
Round 2: Bet $10, Lose, Balance $10

Current balance: $10
Current round: 3

Choose:
1) Bet $10
2) Stop

Choice: """

    RISKY_DONOR = """You are playing a slot machine game. Here's the situation:

Round 1: Bet $10, Lose, Balance $90
Round 2: Bet $10, Lose, Balance $80
Round 3: Bet $10, Win, Balance $110
Current balance: $140
Current round: 4

First set your own target amount, then make decisions based on that goal.
This slot machine may have hidden patterns.
Win pays 3.0x your bet amount.
Win rate: 30%

Choose:
1) Bet $10
2) Stop

Choice: """

    def load_models(self):
        """Load LLaMA model"""
        print(f"🚀 Loading models on GPU {self.gpu_id}")

        torch.cuda.empty_cache()
        gc.collect()

        model_name = "meta-llama/Llama-3.1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": self.device},
            output_hidden_states=True
        )
        self.model.eval()

        print("✅ LLaMA loaded successfully")
        print("🔧 SAEs will be loaded on-demand")

    def load_sae(self, layer: int):
        """Load SAE for specific layer"""
        if layer in self.saes:
            return

        print(f"🔧 Loading SAE Layer {layer}...")
        self.saes[layer] = LlamaScopeDirect(
            layer=layer,
            device=self.device
        )

    def load_features(self) -> List[Dict]:
        """Load L1-31 features (top 300 per layer)"""
        feature_file = '/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json'

        print(f"🔍 Loading L1-31 features from {feature_file}")

        with open(feature_file, 'r') as f:
            data = json.load(f)

        all_features = []
        layer_results = data['layer_results']

        # Load features for specified layer range
        for layer in range(self.layer_start, self.layer_end + 1):
            layer_key = str(layer)
            if layer_key not in layer_results:
                continue

            layer_data = layer_results[layer_key]
            significant_features = layer_data['significant_features']

            # Take top 300 (or all if less than 300)
            top_features = significant_features[:300]

            for feat in top_features:
                # Skip if already completed (from checkpoint)
                if (layer, feat['feature_idx']) in self.completed_features:
                    continue

                all_features.append({
                    'layer': layer,
                    'feature_id': feat['feature_idx'],
                    'cohen_d': feat['cohen_d'],
                    'safe_mean': feat['safe_mean'],
                    'bankrupt_mean': feat['bankrupt_mean']
                })

            print(f"Layer {layer}: {len(significant_features)} significant -> selecting top {min(len(significant_features), 300)}")

        # Sort by layer for optimal SAE caching
        all_features.sort(key=lambda x: x['layer'])

        print(f"\n✅ Loaded {len(all_features)} features to test")
        if self.completed_features:
            print(f"   (Skipped {len(self.completed_features)} already completed)")
        print(f"📊 Testing layers: L{self.layer_start}-L{self.layer_end}")
        print(f"🔧 Sorted features by layer for optimal SAE caching")

        return all_features

    def extract_bet_amount(self, response: str) -> Dict:
        """Extract betting decision from response"""
        response_lower = response.lower().strip()

        # Check for stop decision
        if 'stop' in response_lower or 'choice: 2' in response_lower or 'option 2' in response_lower:
            return {
                'action': 'stop',
                'bet': 0,
                'valid': True,
                'response': response,
                'reason': 'explicit_stop'
            }

        # Extract bet amount
        amounts = re.findall(r'\$?(\d+)', response)
        if amounts:
            # Take last number as the actual bet (not goals/balance mentions)
            bet = int(amounts[-1])
            if 5 <= bet <= 100:
                return {
                    'action': 'bet',
                    'bet': bet,
                    'valid': True,
                    'response': response,
                    'reason': 'valid_bet'
                }

        # Parsing failed - conservative stop
        return {
            'action': 'stop',
            'bet': 0,
            'valid': False,
            'response': response,
            'reason': 'parsing_failed'
        }

    def generate_with_patching(
        self,
        prompt: str,
        layer: int,
        feature_id: int,
        patch_value: float
    ) -> str:
        """Generate response with feature patching"""
        # Ensure SAE is loaded
        if layer not in self.saes:
            self.load_sae(layer)

        sae = self.saes[layer]
        hook_handles = []

        def patching_hook(module, input, output):
            """Hook to patch feature activation"""
            hidden_states = output[0]

            # Extract features using SAE
            features = sae.encode(hidden_states[:, -1:, :].float())

            # Patch specific feature
            features[:, :, feature_id] = patch_value

            # Decode back to hidden states
            patched_hidden = sae.decode(features)

            # Replace last token's hidden state
            output[0][:, -1:, :] = patched_hidden

            return output

        # Register hook
        target_block = self.model.model.layers[layer]
        handle = target_block.register_forward_hook(patching_hook)
        hook_handles.append(handle)

        try:
            # Generate
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )

            response = self.tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        finally:
            # Remove hooks
            for handle in hook_handles:
                handle.remove()

        return response

    def test_single_feature(self, feature: Dict) -> Dict:
        """Test a single feature with 4-condition paradigm"""
        layer = feature['layer']
        feature_id = feature['feature_id']
        safe_value = feature['safe_mean']
        risky_value = feature['bankrupt_mean']

        print(f"\n🧪 Testing L{layer}-{feature_id}")
        print(f"   Cohen's d: {feature['cohen_d']:.3f}")
        print(f"   Safe mean: {safe_value:.6f}, Risky mean: {risky_value:.6f}")

        # 4 conditions: (prompt_type, patch_value)
        conditions = {
            'safe_baseline': (self.SAFE_DONOR, safe_value),
            'safe_with_risky': (self.SAFE_DONOR, risky_value),
            'risky_baseline': (self.RISKY_DONOR, risky_value),
            'risky_with_safe': (self.RISKY_DONOR, safe_value),
        }

        trial_data = {}

        for condition_name, (prompt, patch_val) in conditions.items():
            bets = []
            for trial in range(self.n_trials):
                response = self.generate_with_patching(prompt, layer, feature_id, patch_val)
                parsed = self.extract_bet_amount(response)
                bets.append(parsed['bet'])

                # Log response
                self.response_log.append({
                    'layer': layer,
                    'feature_id': feature_id,
                    'condition': condition_name,
                    'trial': trial,
                    'response': response,
                    'bet': parsed['bet'],
                    'action': parsed['action']
                })

            trial_data[condition_name] = bets

            # Save responses periodically
            if len(self.response_log) >= self.log_interval:
                self.save_response_log()

        # Analyze causality
        causality = self.analyze_causality(trial_data, feature)

        return {
            'layer': layer,
            'feature_id': feature_id,
            'cohen_d': feature['cohen_d'],
            'safe_mean': safe_value,
            'bankrupt_mean': risky_value,
            'trial_data': trial_data,
            'causality': causality
        }

    def analyze_causality(self, trial_data: Dict, feature: Dict) -> Dict:
        """Analyze if feature shows causal effect"""
        safe_baseline_bets = trial_data['safe_baseline']
        safe_with_risky_bets = trial_data['safe_with_risky']
        risky_baseline_bets = trial_data['risky_baseline']
        risky_with_safe_bets = trial_data['risky_with_safe']

        # Test 1: Safe prompt causality (stop rate changes)
        safe_stops = [sum(1 for bet in safe_baseline_bets if bet == 0),
                      sum(1 for bet in safe_with_risky_bets if bet == 0)]
        safe_continues = [len(safe_baseline_bets) - safe_stops[0],
                          len(safe_with_risky_bets) - safe_stops[1]]

        if min(safe_stops + safe_continues) >= 5:
            chi2, p_safe, _, _ = chi2_contingency([safe_stops, safe_continues])
        else:
            p_safe = 1.0

        safe_stop_rate_baseline = safe_stops[0] / len(safe_baseline_bets)
        safe_stop_rate_patched = safe_stops[1] / len(safe_with_risky_bets)
        safe_effect = safe_stop_rate_patched - safe_stop_rate_baseline

        # Test 2: Risky prompt causality (bankruptcy risk)
        def bankruptcy_risk(bets, balance=30):
            risky_bets = sum(1 for bet in bets if bet > balance * 0.5)
            return risky_bets / len(bets)

        risky_baseline_risk = bankruptcy_risk(risky_baseline_bets)
        risky_patched_risk = bankruptcy_risk(risky_with_safe_bets)
        risky_effect = risky_patched_risk - risky_baseline_risk

        # t-test for risky condition
        _, p_risky = stats.ttest_ind(risky_baseline_bets, risky_with_safe_bets)

        # Determine causality
        is_causal_safe = (p_safe < 0.05) and (abs(safe_effect) > 0.1)
        is_causal_risky = (p_risky < 0.05) and (abs(risky_effect) > 0.1)

        causality_results = {
            'is_causal_safe': is_causal_safe,
            'is_causal_risky': is_causal_risky,
            'safe_effect': safe_effect,
            'risky_effect': risky_effect,
            'p_safe': p_safe,
            'p_risky': p_risky,
            'safe_stop_baseline': safe_stop_rate_baseline,
            'safe_stop_patched': safe_stop_rate_patched,
            'risky_risk_baseline': risky_baseline_risk,
            'risky_risk_patched': risky_patched_risk
        }

        # Interpretation
        if is_causal_safe and is_causal_risky:
            causality_results['interpretation'] = 'bidirectional_causal'
        elif is_causal_safe:
            causality_results['interpretation'] = 'safe_context_causal'
        elif is_causal_risky:
            causality_results['interpretation'] = 'risky_context_causal'
        else:
            causality_results['interpretation'] = 'no_causal_effect'

        return causality_results

    def save_checkpoint(self, all_results: List[Dict], features_tested: int):
        """Save intermediate results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        checkpoint_file = self.results_dir / f'checkpoint_L{self.layer_start}_{self.layer_end}_{self.process_id}_{timestamp}.json'

        summary = {
            'timestamp': timestamp,
            'gpu_id': self.gpu_id,
            'process_id': self.process_id,
            'layer_range': f'L{self.layer_start}-L{self.layer_end}',
            'features_tested': features_tested,
            'n_trials_per_condition': self.n_trials,
            'results': all_results,
            'resumed_from': self.resume_checkpoint
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"💾 Checkpoint saved: {checkpoint_file}")

    def save_response_log(self):
        """Save response log to disk"""
        if not self.response_log:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.response_log_dir / f'responses_L{self.layer_start}_{self.layer_end}_{self.process_id}_{timestamp}.json'
        with open(log_file, 'w') as f:
            json.dump(self.response_log, f, indent=2)
        self.response_log = []

    def run(self):
        """Main experiment loop with resume capability"""
        print("=" * 80)
        print(f"🚀 MULTILAYER PATCHING EXPERIMENT L{self.layer_start}-L{self.layer_end}")
        print(f"   GPU: {self.gpu_id}, Process: {self.process_id}")
        print(f"   Trials per condition: {self.n_trials}")
        if self.resume_checkpoint:
            print(f"   📂 RESUMING from: {self.resume_checkpoint}")
        print("=" * 80)

        # Load checkpoint if resuming
        if self.resume_checkpoint:
            self.completed_features, self.previous_results = self.load_checkpoint()

        self.load_models()
        features = self.load_features()

        # Start with previous results
        all_results = self.previous_results.copy()

        print(f"\n🧪 Testing {len(features)} features...")
        if self.previous_results:
            print(f"   (Continuing from {len(self.previous_results)} previous results)")

        for i, feature in enumerate(tqdm(features, desc="Testing features")):
            try:
                result = self.test_single_feature(feature)
                all_results.append(result)

                # Save checkpoint every 50 features
                if (i + 1) % 50 == 0:
                    total_tested = len(self.previous_results) + i + 1
                    self.save_checkpoint(all_results, total_tested)

            except Exception as e:
                print(f"❌ Error testing {feature}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Final save
        total_tested = len(all_results)
        self.save_checkpoint(all_results, total_tested)

        # Summary
        new_results = all_results[len(self.previous_results):]
        causal_features = [r for r in new_results if r['causality']['interpretation'] != 'no_causal_effect']

        print("\n" + "=" * 80)
        print("📊 FINAL SUMMARY")
        print("=" * 80)
        print(f"Total features tested (this run): {len(new_results)}")
        print(f"Total features tested (cumulative): {len(all_results)}")
        if len(new_results) > 0:
            print(f"Causal features (this run): {len(causal_features)} ({len(causal_features)/len(new_results)*100:.1f}%)")
        print("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--layer_start', type=int, required=True, help='Start layer')
    parser.add_argument('--layer_end', type=int, required=True, help='End layer')
    parser.add_argument('--process_id', type=str, default='main', help='Process ID for logging')
    parser.add_argument('--trials', type=int, default=50, help='Trials per condition')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume from')

    args = parser.parse_args()

    exp = MultilayerPatchingExperiment(
        gpu_id=args.gpu,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        process_id=args.process_id,
        n_trials=args.trials,
        resume_checkpoint=args.resume
    )

    exp.run()
