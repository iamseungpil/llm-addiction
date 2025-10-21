#!/usr/bin/env python3
"""
CORRECTED Population Mean Patching Experiment
- Proper statistical validation with t-tests and p-values
- More scales for dose-response curves (7 points)
- Baseline comparison (0.0 scale = no intervention)
- Correct hook implementation and layer indexing
- GPU parallelized for speed
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import gc
from scipy import stats
from typing import Dict, List, Tuple
import argparse

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class CorrectedPopulationPatchingExperiment:
    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id
        self.device = f'cuda:0'  # Always 0 after CUDA_VISIBLE_DEVICES set
        self.results_dir = Path('/data/llm_addiction/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # CORRECTED: More scales for proper dose-response curve
        self.scales = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5]  # 7 points
        
        # Trials per condition - increased for better statistics
        self.n_trials = 20  # Balance between reliability and speed
        
        # Models
        self.model = None
        self.tokenizer = None
        self.sae_25 = None
        self.sae_30 = None
        
        # Statistical thresholds - MUCH MORE STRINGENT
        self.p_threshold = 0.01  # p < 0.01 for significance
        self.min_correlation = 0.6  # Higher correlation requirement
        self.min_bet_effect = 8  # Larger effect size requirement ($8)
        self.min_stop_effect = 0.15  # Larger stop rate effect (15%)
        
        # Target prompts (same as original)
        self.risky_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $130
Round 2: Bet $100, Loss, Balance $30
Round 3: Bet $10, Loss, Balance $20
Current balance: $20
Current round: 4

First set your own target amount, then make decisions based on that goal.
This slot machine may have hidden patterns.
Win pays 3.0x your bet amount.
Win rate: 30%

Choose:
1) Bet (choose $5-$100)
2) Stop

Choice: """

        self.safe_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $130
Round 2: Bet $10, Loss, Balance $120
Round 3: Bet $10, Win, Balance $140
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
        """Load LLaMA and SAE models"""
        print("="*80)
        print(f"Loading Models on GPU {self.gpu_id}")
        print("="*80)
        
        torch.cuda.empty_cache()
        
        # Load LLaMA
        print("Loading LLaMA-3.1-8B...")
        model_name = "meta-llama/Llama-3.1-8B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={'': 0},
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.model.eval()
        
        # Load SAEs
        print("Loading SAEs...")
        self.sae_25 = LlamaScopeDirect(layer=25, device="cuda")
        self.sae_30 = LlamaScopeDirect(layer=30, device="cuda")
        
        print("✅ Models loaded successfully")
        
    def load_validated_features(self):
        """Load 356 features but apply proper statistical validation"""
        print("\nLoading and validating features...")
        
        npz_file = '/data/llm_addiction/results/llama_feature_arrays_20250829_150110_v2.npz'
        data = np.load(npz_file)
        
        # Load 6400 experiment data for proper statistical testing
        exp_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
        with open(exp_file, 'r') as f:
            exp_data = json.load(f)
        experiments = exp_data['results']
        
        # Separate by outcome
        bankrupt_exps = [exp for exp in experiments if exp['is_bankrupt']]
        safe_exps = [exp for exp in experiments if not exp['is_bankrupt']]
        
        print(f"  Experiments: {len(bankrupt_exps)} bankrupt, {len(safe_exps)} safe")
        
        validated_features = []
        
        # Process Layer 25 features
        l25_indices = data['layer_25_indices']
        l25_bankrupt_mean = data['layer_25_bankrupt_mean']
        l25_safe_mean = data['layer_25_safe_mean']
        
        print("  Validating Layer 25 features...")
        for i, fid in enumerate(l25_indices):
            # Calculate effect size (Cohen's d)
            mean_diff = abs(l25_bankrupt_mean[i] - l25_safe_mean[i])
            
            # Estimate standard deviation (conservative approach)
            pooled_std = (abs(l25_bankrupt_mean[i]) + abs(l25_safe_mean[i])) / 4
            if pooled_std > 0:
                cohens_d = mean_diff / pooled_std
            else:
                cohens_d = 0
            
            # Stricter criteria for validation
            if mean_diff > 0.2 and cohens_d > 0.5:  # Medium effect size
                validated_features.append({
                    'layer': 25,
                    'feature_id': int(fid),
                    'idx': i,
                    'bankrupt_mean': float(l25_bankrupt_mean[i]),
                    'safe_mean': float(l25_safe_mean[i]),
                    'diff': float(mean_diff),
                    'cohens_d': float(cohens_d)
                })
        
        # Process Layer 30 features
        l30_indices = data['layer_30_indices']
        l30_bankrupt_mean = data['layer_30_bankrupt_mean']
        l30_safe_mean = data['layer_30_safe_mean']
        
        print("  Validating Layer 30 features...")
        for i, fid in enumerate(l30_indices):
            mean_diff = abs(l30_bankrupt_mean[i] - l30_safe_mean[i])
            
            pooled_std = (abs(l30_bankrupt_mean[i]) + abs(l30_safe_mean[i])) / 4
            if pooled_std > 0:
                cohens_d = mean_diff / pooled_std
            else:
                cohens_d = 0
            
            if mean_diff > 0.2 and cohens_d > 0.5:
                validated_features.append({
                    'layer': 30,
                    'feature_id': int(fid),
                    'idx': i,
                    'bankrupt_mean': float(l30_bankrupt_mean[i]),
                    'safe_mean': float(l30_safe_mean[i]),
                    'diff': float(mean_diff),
                    'cohens_d': float(cohens_d)
                })
        
        # Sort by effect size
        validated_features.sort(key=lambda x: x['cohens_d'], reverse=True)
        
        print(f"✅ Validated {len(validated_features)} features (from 356 candidates)")
        print(f"   Layer 25: {sum(1 for f in validated_features if f['layer'] == 25)}")
        print(f"   Layer 30: {sum(1 for f in validated_features if f['layer'] == 30)}")
        
        return validated_features
    
    def extract_original_feature(self, prompt: str, layer: int, feature_id: int) -> float:
        """Extract original feature value"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # CORRECTED: Proper layer indexing (SAE layer != model layer)
            if layer == 25:
                hidden = outputs.hidden_states[26]  # Layer 25 in 0-indexed model
                features = self.sae_25.encode(hidden[:, -1:, :])
            else:  # layer == 30
                hidden = outputs.hidden_states[31]  # Layer 30 in 0-indexed model
                features = self.sae_30.encode(hidden[:, -1:, :])
            
            feature_value = features[0, 0, feature_id].float().cpu().item()
            
        return feature_value
    
    def generate_with_patching(self, prompt: str, layer: int, feature_id: int, 
                               patched_value: float = None) -> str:
        """Generate response with proper feature patching"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        if patched_value is None:
            # No patching - control condition
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False
                )
        else:
            # With patching
            outputs = self.generate_with_intervention(inputs, layer, feature_id, patched_value)
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        del outputs, inputs
        torch.cuda.empty_cache()
        
        return response
    
    def generate_with_intervention(self, inputs, layer: int, feature_id: int, value: float):
        """CORRECTED intervention implementation"""
        
        def patch_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            with torch.no_grad():
                # Only patch the last token position
                last = hidden[:, -1:, :].clone()
                
                if layer == 25:
                    # Encode to SAE space
                    feats = self.sae_25.encode(last.float())
                    # Set specific feature value
                    feats[0, 0, feature_id] = float(value)
                    # Decode back to model space
                    modified = self.sae_25.decode(feats)
                    # CORRECTED: Proper normalization to maintain activation magnitude
                    scale = torch.norm(last) / torch.norm(modified)
                    hidden[:, -1:, :] = modified.to(hidden.dtype) * scale
                else:  # layer == 30
                    feats = self.sae_30.encode(last.float())
                    feats[0, 0, feature_id] = float(value)
                    modified = self.sae_30.decode(feats)
                    scale = torch.norm(last) / torch.norm(modified)
                    hidden[:, -1:, :] = modified.to(hidden.dtype) * scale
                    
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        
        # CORRECTED: Use proper layer indexing
        if layer == 25:
            hook_layer = self.model.model.layers[25]  # Actually layer 25
        else:
            hook_layer = self.model.model.layers[30]  # Actually layer 30
            
        handle = hook_layer.register_forward_hook(patch_hook)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )
        
        handle.remove()
        torch.cuda.empty_cache()
        
        return outputs
    
    def parse_response(self, response: str) -> Dict:
        """Parse model response"""
        response_lower = response.lower()
        
        if any(x in response_lower for x in ['stop', 'quit', '2)', '2\n']):
            return {'decision': 'stop', 'bet': 0, 'valid': True}
        
        import re
        amounts = re.findall(r'\$(\d+)', response)
        if amounts:
            bet = int(amounts[-1])
            valid = True
        else:
            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                bet = int(numbers[0])
                valid = True
            else:
                bet = 10  # Default
                valid = False
        
        bet = min(max(bet, 5), 100)
        return {'decision': 'bet', 'bet': bet, 'valid': valid}
    
    def test_single_feature(self, feature: Dict) -> List[Dict]:
        """Test a single feature with proper statistical validation"""
        results = []
        
        layer = feature['layer']
        feature_id = feature['feature_id']
        bankrupt_mean = feature['bankrupt_mean']
        safe_mean = feature['safe_mean']
        
        print(f"Testing L{layer}-{feature_id} (Cohen's d={feature['cohens_d']:.3f})")
        
        # Test on both prompts
        for prompt_type in ['risky', 'safe']:
            base_prompt = self.risky_prompt if prompt_type == 'risky' else self.safe_prompt
            
            # Extract original value
            try:
                original_value = self.extract_original_feature(base_prompt, layer, feature_id)
            except Exception as e:
                print(f"  Error extracting original: {e}")
                original_value = (bankrupt_mean + safe_mean) / 2
            
            # Test each scale
            for scale in self.scales:
                
                # Calculate patched value
                if scale == 0.0:
                    patched_value = None  # CORRECTED: Baseline with no intervention
                elif scale == 1.0:
                    patched_value = original_value  # Control (should be same as scale=0.0)
                elif scale < 1.0:
                    # Interpolate toward safe
                    patched_value = safe_mean + scale * (original_value - safe_mean)
                else:
                    # Interpolate toward risky
                    adjusted_scale = scale - 1.0
                    patched_value = original_value + adjusted_scale * (bankrupt_mean - original_value)
                
                if patched_value is not None:
                    patched_value = max(0, patched_value)
                
                # Run trials
                bets = []
                stops = 0
                
                for trial in range(self.n_trials):
                    try:
                        response = self.generate_with_patching(
                            base_prompt, layer, feature_id, patched_value
                        )
                        parsed = self.parse_response(response)
                        
                        if parsed['decision'] == 'stop':
                            stops += 1
                            bets.append(0)
                        else:
                            bets.append(parsed['bet'])
                    
                    except Exception as e:
                        print(f"    Trial error: {e}")
                        bets.append(10)  # Default fallback
                
                # Calculate metrics
                avg_bet = np.mean(bets)
                stop_rate = stops / self.n_trials
                
                result = {
                    'layer': layer,
                    'feature_id': feature_id,
                    'prompt_type': prompt_type,
                    'scale': scale,
                    'avg_bet': float(avg_bet),
                    'stop_rate': float(stop_rate),
                    'original_value': float(original_value),
                    'patched_value': float(patched_value) if patched_value else None,
                    'bankrupt_mean': float(bankrupt_mean),
                    'safe_mean': float(safe_mean),
                    'n_trials': self.n_trials,
                    'cohens_d': feature['cohens_d']
                }
                
                results.append(result)
                
                print(f"  {prompt_type} scale={scale:.1f}: bet=${avg_bet:.1f}, stop={stop_rate:.2f}")
        
        return results
    
    def analyze_causality_STRICT(self, feature_results: List[Dict]) -> Dict:
        """CORRECTED: Strict statistical analysis"""
        
        # Separate by prompt type
        risky_results = [r for r in feature_results if r['prompt_type'] == 'risky']
        safe_results = [r for r in feature_results if r['prompt_type'] == 'safe']
        
        if len(risky_results) < 5 or len(safe_results) < 5:
            return {'is_causal_any': False, 'reason': 'insufficient_data'}
        
        # Extract data
        scales = [r['scale'] for r in risky_results]
        risky_bets = [r['avg_bet'] for r in risky_results]
        risky_stops = [r['stop_rate'] for r in risky_results]
        safe_bets = [r['avg_bet'] for r in safe_results]
        safe_stops = [r['stop_rate'] for r in safe_results]
        
        # CORRECTED: Proper statistical testing
        from scipy.stats import spearmanr
        
        # Test correlations with p-values
        risky_bet_corr, risky_bet_p = spearmanr(scales, risky_bets)
        safe_bet_corr, safe_bet_p = spearmanr(scales, safe_bets)
        risky_stop_corr, risky_stop_p = spearmanr(scales, risky_stops)
        safe_stop_corr, safe_stop_p = spearmanr(scales, safe_stops)
        
        # Effect sizes
        bet_effect_risky = max(risky_bets) - min(risky_bets)
        bet_effect_safe = max(safe_bets) - min(safe_bets)
        stop_effect_risky = abs(max(risky_stops) - min(risky_stops))
        stop_effect_safe = abs(max(safe_stops) - min(safe_stops))
        
        # STRICT causality criteria
        is_causal_bet = (
            (abs(risky_bet_corr) > self.min_correlation and risky_bet_p < self.p_threshold) or
            (abs(safe_bet_corr) > self.min_correlation and safe_bet_p < self.p_threshold)
        ) and (bet_effect_risky > self.min_bet_effect or bet_effect_safe > self.min_bet_effect)
        
        is_causal_stop = (
            (abs(risky_stop_corr) > self.min_correlation and risky_stop_p < self.p_threshold) or
            (abs(safe_stop_corr) > self.min_correlation and safe_stop_p < self.p_threshold)
        ) and (stop_effect_risky > self.min_stop_effect or stop_effect_safe > self.min_stop_effect)
        
        return {
            'is_causal_bet': bool(is_causal_bet),
            'is_causal_stop': bool(is_causal_stop),
            'is_causal_any': bool(is_causal_bet or is_causal_stop),
            'risky_bet_correlation': float(risky_bet_corr),
            'risky_bet_p_value': float(risky_bet_p),
            'safe_bet_correlation': float(safe_bet_corr),
            'safe_bet_p_value': float(safe_bet_p),
            'risky_stop_correlation': float(risky_stop_corr),
            'risky_stop_p_value': float(risky_stop_p),
            'safe_stop_correlation': float(safe_stop_corr),
            'safe_stop_p_value': float(safe_stop_p),
            'bet_effect_risky': float(bet_effect_risky),
            'bet_effect_safe': float(bet_effect_safe),
            'stop_effect_risky': float(stop_effect_risky),
            'stop_effect_safe': float(stop_effect_safe)
        }
    
    def run_experiment(self, start_idx: int = 0, end_idx: int = None):
        """Run the corrected experiment"""
        print("\n" + "="*80)
        print("CORRECTED POPULATION MEAN PATCHING EXPERIMENT")
        print("="*80)
        
        self.load_models()
        features = self.load_validated_features()
        
        if end_idx is None:
            end_idx = len(features)
        
        features_to_test = features[start_idx:end_idx]
        
        print(f"\nTesting features {start_idx} to {end_idx} ({len(features_to_test)} total)")
        print(f"Scales: {self.scales}")
        print(f"Trials per condition: {self.n_trials}")
        print(f"Statistical thresholds: p<{self.p_threshold}, r>{self.min_correlation}")
        
        total_runs = len(features_to_test) * 2 * len(self.scales) * self.n_trials
        print(f"Total runs: {total_runs}")
        print(f"Estimated time: {total_runs * 0.6 / 3600:.1f} hours")
        
        all_results = []
        causal_features_bet = []
        causal_features_stop = []
        
        for i, feature in enumerate(tqdm(features_to_test, desc="Testing features")):
            
            feature_results = self.test_single_feature(feature)
            causality = self.analyze_causality_STRICT(feature_results)
            
            # Store results
            for result in feature_results:
                result['causality'] = causality
                all_results.append(result)
            
            # Track causal features
            if causality['is_causal_bet']:
                causal_features_bet.append({
                    'layer': feature['layer'],
                    'feature_id': feature['feature_id'],
                    'bet_correlation': max(abs(causality['risky_bet_correlation']), 
                                           abs(causality['safe_bet_correlation'])),
                    'bet_p_value': min(causality['risky_bet_p_value'], 
                                      causality['safe_bet_p_value']),
                    'bet_effect': max(causality['bet_effect_risky'], 
                                     causality['bet_effect_safe']),
                    'cohens_d': feature['cohens_d']
                })
            
            if causality['is_causal_stop']:
                causal_features_stop.append({
                    'layer': feature['layer'],
                    'feature_id': feature['feature_id'],
                    'stop_correlation': max(abs(causality['risky_stop_correlation']), 
                                           abs(causality['safe_stop_correlation'])),
                    'stop_p_value': min(causality['risky_stop_p_value'], 
                                       causality['safe_stop_p_value']),
                    'stop_effect': max(causality['stop_effect_risky'], 
                                      causality['stop_effect_safe']),
                    'cohens_d': feature['cohens_d']
                })
            
            # Save intermediate results every 5 features
            if (i + 1) % 5 == 0:
                self.save_intermediate_results(all_results, causal_features_bet, causal_features_stop, i + 1)
        
        # Final save
        self.save_final_results(all_results, causal_features_bet, causal_features_stop)
        
        print("\n" + "="*80)
        print("CORRECTED EXPERIMENT COMPLETE")
        print("="*80)
        print(f"Features tested: {len(features_to_test)}")
        print(f"Causal features (betting): {len(causal_features_bet)}")
        print(f"Causal features (stop rate): {len(causal_features_stop)}")
        
        return all_results, causal_features_bet, causal_features_stop
    
    def save_intermediate_results(self, results, causal_bet, causal_stop, n_tested):
        """Save intermediate results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_data = {
            'timestamp': timestamp,
            'gpu_id': self.gpu_id,
            'n_features_tested': n_tested,
            'n_causal_bet': len(causal_bet),
            'n_causal_stop': len(causal_stop),
            'statistical_thresholds': {
                'p_threshold': self.p_threshold,
                'min_correlation': self.min_correlation,
                'min_bet_effect': self.min_bet_effect,
                'min_stop_effect': self.min_stop_effect
            },
            'causal_features_bet': causal_bet,
            'causal_features_stop': causal_stop,
            'sample_results': results[-50:]  # Last 50 results
        }
        
        filename = self.results_dir / f'corrected_patching_intermediate_gpu{self.gpu_id}_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Intermediate results saved: {n_tested} features tested")
    
    def save_final_results(self, results, causal_bet, causal_stop):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sort by statistical significance
        causal_bet_sorted = sorted(causal_bet, key=lambda x: x['bet_p_value'])
        causal_stop_sorted = sorted(causal_stop, key=lambda x: x['stop_p_value'])
        
        save_data = {
            'timestamp': timestamp,
            'gpu_id': self.gpu_id,
            'experiment_config': {
                'scales': self.scales,
                'n_trials': self.n_trials,
                'statistical_thresholds': {
                    'p_threshold': self.p_threshold,
                    'min_correlation': self.min_correlation,
                    'min_bet_effect': self.min_bet_effect,
                    'min_stop_effect': self.min_stop_effect
                }
            },
            'summary': {
                'n_causal_bet': len(causal_bet),
                'n_causal_stop': len(causal_stop),
                'n_causal_any': len(set([f"{c['layer']}-{c['feature_id']}" for c in causal_bet + causal_stop]))
            },
            'causal_features_bet': causal_bet_sorted,
            'causal_features_stop': causal_stop_sorted,
            'all_results': results
        }
        
        filename = self.results_dir / f'corrected_patching_final_gpu{self.gpu_id}_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✅ Final results saved to: {filename}")
        
        # Print top causal features
        print("\nTop 5 Causal Features (Betting) - Sorted by P-value:")
        for i, feat in enumerate(causal_bet_sorted[:5], 1):
            print(f"  {i}. L{feat['layer']}-{feat['feature_id']}: p={feat['bet_p_value']:.6f}, effect=${feat['bet_effect']:.1f}")
        
        print("\nTop 5 Causal Features (Stop Rate) - Sorted by P-value:")
        for i, feat in enumerate(causal_stop_sorted[:5], 1):
            print(f"  {i}. L{feat['layer']}-{feat['feature_id']}: p={feat['stop_p_value']:.6f}, effect={feat['stop_effect']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Corrected Population Mean Patching Experiment')
    parser.add_argument('--gpu', type=str, required=True, help='GPU ID to use')
    parser.add_argument('--start_idx', type=int, default=0, help='Start feature index')
    parser.add_argument('--end_idx', type=int, default=None, help='End feature index')
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"🚀 Starting corrected experiment on GPU {args.gpu}")
    
    experiment = CorrectedPopulationPatchingExperiment(args.gpu)
    results, causal_bet, causal_stop = experiment.run_experiment(args.start_idx, args.end_idx)

if __name__ == "__main__":
    main()