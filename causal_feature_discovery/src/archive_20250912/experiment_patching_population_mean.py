#!/usr/bin/env python3
"""
Population Mean Patching Experiment
Tests 356 features with multiple scales using actual population means
GPU-parallelized for efficiency
"""

import os
import sys
# GPU will be set by command line argument

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import gc
import sys
from scipy import stats
from typing import Dict, List, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class PopulationMeanPatchingExperiment:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_dir = Path('/data/llm_addiction/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Scales to test - 3 key points without 0.0 baseline
        self.scales = [0.5, 1.0, 1.5]  # mid-safe, original, mid-risky
        
        # Trials per condition
        self.n_trials = 30  # Optimized for balance between reliability and speed
        
        # Batch size for GPU parallelization
        self.batch_size = 4  # Process multiple trials simultaneously
        
        # Load models
        self.model = None
        self.tokenizer = None
        self.sae_25 = None
        self.sae_30 = None
        
        # Analysis/logging options
        # If True, exclude invalid trials (parse failures/default fallbacks and exceptions)
        self.exclude_invalid = False
        
        # Target prompts from Experiment 1
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
        import os
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        print(f"Loading Models on GPU {gpu_id} (appears as cuda:0 to PyTorch)")
        print("="*80)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Load LLaMA
        print("Loading LLaMA-3.1-8B...")
        model_name = "meta-llama/Llama-3.1-8B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 to avoid dtype issues
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
        
    def load_features(self):
        """Load 356 features and their population means"""
        print("\nLoading 356 features...")
        
        # Load from the correct v2 file
        npz_file = '/data/llm_addiction/results/llama_feature_arrays_20250829_150110_v2.npz'
        data = np.load(npz_file)
        
        features = []
        
        # Layer 25 features (53)
        l25_indices = data['layer_25_indices']
        l25_bankrupt_mean = data['layer_25_bankrupt_mean']
        l25_safe_mean = data['layer_25_safe_mean']
        
        for i, fid in enumerate(l25_indices):
            features.append({
                'layer': 25,
                'feature_id': int(fid),
                'idx': i,
                'bankrupt_mean': float(l25_bankrupt_mean[i]),
                'safe_mean': float(l25_safe_mean[i]),
                'diff': float(abs(l25_bankrupt_mean[i] - l25_safe_mean[i]))
            })
        
        # Layer 30 features (303)
        l30_indices = data['layer_30_indices']
        l30_bankrupt_mean = data['layer_30_bankrupt_mean']
        l30_safe_mean = data['layer_30_safe_mean']
        
        for i, fid in enumerate(l30_indices):
            features.append({
                'layer': 30,
                'feature_id': int(fid),
                'idx': i,
                'bankrupt_mean': float(l30_bankrupt_mean[i]),
                'safe_mean': float(l30_safe_mean[i]),
                'diff': float(abs(l30_bankrupt_mean[i] - l30_safe_mean[i]))
            })
        
        print(f"✅ Loaded {len(features)} features (L25: 53, L30: 303)")
        return features
    
    def extract_original_feature(self, prompt: str, layer: int, feature_id: int) -> float:
        """Extract original feature value from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            if layer == 25:
                hidden = outputs.hidden_states[25]
                features = self.sae_25.encode(hidden[:, -1:, :])
            else:
                hidden = outputs.hidden_states[30]
                features = self.sae_30.encode(hidden[:, -1:, :])
            
            # Get specific feature value
            feature_value = features[0, 0, feature_id].float().cpu().item()
            
        return feature_value
    
    def generate_with_patching(self, prompt: str, layer: int, feature_id: int, 
                               patched_value: float = None) -> str:
        """Generate response with feature patching"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            if patched_value is None:
                # No patching (control)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False
                )
            else:
                # With patching - need custom forward pass
                outputs = self.generate_with_feature_intervention(
                    inputs, layer, feature_id, patched_value
                )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Clear memory
        del outputs, inputs
        torch.cuda.empty_cache()
        
        return response
    
    def generate_with_feature_intervention(self, inputs, layer: int, feature_id: int, value: float):
        """Custom generation with feature intervention"""
        # This is simplified - in reality need to hook into model forward pass
        # For now, using approximation
        
        def patch_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                last = hidden[:, -1:, :]
                if layer == 25:
                    feats = self.sae_25.encode(last.float())
                    feats[0, 0, feature_id] = float(value)
                    mod = self.sae_25.decode(feats)
                    hidden[:, -1:, :] = mod.to(hidden.dtype)
                else:
                    feats = self.sae_30.encode(last.float())
                    feats[0, 0, feature_id] = float(value)
                    mod = self.sae_30.decode(feats)
                    hidden[:, -1:, :] = mod.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        
        # Register hook - FIXED: use layer directly, not layer-1
        hook_layer = self.model.model.layers[layer]
        handle = hook_layer.register_forward_hook(patch_hook)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=False
        )
        
        # Remove hook
        handle.remove()
        
        # Clear GPU cache to prevent OOM
        torch.cuda.empty_cache()
        
        return outputs
    
    def generate_batch_with_patching(self, prompts: List[str], layer: int, feature_id: int, value: float):
        """Generate batch responses with feature intervention"""
        # Tokenize all prompts
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        def patch_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                # Patch all sequences in batch
                for i in range(hidden.shape[0]):
                    last = hidden[i:i+1, -1:, :]
                    if layer == 25:
                        feats = self.sae_25.encode(last.float())
                        feats[0, 0, feature_id] = float(value) if value is not None else feats[0, 0, feature_id]
                        mod = self.sae_25.decode(feats)
                        hidden[i:i+1, -1:, :] = mod.to(hidden.dtype)
                    else:
                        feats = self.sae_30.encode(last.float())
                        feats[0, 0, feature_id] = float(value) if value is not None else feats[0, 0, feature_id]
                        mod = self.sae_30.decode(feats)
                        hidden[i:i+1, -1:, :] = mod.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        
        # Register hook if patching
        handle = None
        if value is not None:
            hook_layer = self.model.model.layers[layer]
            handle = hook_layer.register_forward_hook(patch_hook)
        
        # Generate for batch
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=False
        )
        
        # Remove hook
        if handle:
            handle.remove()
        
        # Decode all outputs
        responses = []
        for i in range(outputs.shape[0]):
            response = self.tokenizer.decode(outputs[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def parse_response(self, response: str) -> Dict:
        """Parse model response for bet amount and decision"""
        response_lower = response.lower()
        
        # Check for stop decision
        if any(x in response_lower for x in ['stop', 'quit', '2)', '2\n']):
            return {'decision': 'stop', 'bet': 0, 'source': 'stop', 'valid': True}
        
        # Extract bet amount
        import re
        
        # Try to find dollar amounts
        amounts = re.findall(r'\$(\d+)', response)
        if amounts:
            bet = int(amounts[-1])  # Take last amount mentioned
            source = 'dollar'
            valid = True
        else:
            # Try to find plain numbers
            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                bet = int(numbers[0])
                source = 'number'
                valid = True
            else:
                # Fallback default (treated as invalid if exclude_invalid=True)
                bet = 10
                source = 'default'
                valid = False
        
        # Cap bet amount
        bet = min(max(bet, 5), 100)
        
        return {'decision': 'bet', 'bet': bet, 'source': source, 'valid': valid}
    
    def test_single_feature(self, feature: Dict) -> List[Dict]:
        """Test a single feature across all conditions"""
        results = []
        
        layer = feature['layer']
        feature_id = feature['feature_id']
        bankrupt_mean = feature['bankrupt_mean']
        safe_mean = feature['safe_mean']
        
        print(f"\nTesting L{layer}-{feature_id} (diff={feature['diff']:.4f})")
        
        # Test on both prompts
        for prompt_type in ['risky', 'safe']:
            
            if prompt_type == 'risky':
                base_prompt = self.risky_prompt
            else:
                base_prompt = self.safe_prompt
            
            # Extract original feature value
            try:
                original_value = self.extract_original_feature(base_prompt, layer, feature_id)
            except Exception as e:
                print(f"  Error extracting original: {e}")
                original_value = (bankrupt_mean + safe_mean) / 2
            
            # Test each scale
            for scale in self.scales:
                
                # Calculate patched value
                if scale == 0:
                    patched_value = None  # No patching
                else:
                    # Scale interpolates between safe (0.0) and risky (2.0)
                    # scale=0.0 -> safe_mean, scale=1.0 -> original, scale=2.0 -> bankrupt_mean
                    if scale < 1.0:
                        # Interpolate toward safe
                        patched_value = safe_mean + scale * (original_value - safe_mean)
                    else:
                        # Interpolate toward risky
                        adjusted_scale = scale - 1.0  # 0.0 to 1.0 range
                        patched_value = original_value + adjusted_scale * (bankrupt_mean - original_value)
                    patched_value = max(0, patched_value)  # Ensure non-negative
                
                # Run trials sequentially - more reliable
                bets = []
                stops = 0
                invalids = 0
                
                for trial in range(self.n_trials):
                    try:
                        response = self.generate_with_patching(
                            base_prompt, layer, feature_id, patched_value
                        )
                        parsed = self.parse_response(response)
                        
                        # Exclude invalid parses if configured
                        if self.exclude_invalid and not parsed.get('valid', True):
                            invalids += 1
                            continue
                        
                        if parsed['decision'] == 'stop':
                            stops += 1
                            bets.append(0)
                        else:
                            bets.append(parsed['bet'])
                    
                    except Exception as e:
                        print(f"    Trial error: {e}")
                        if self.exclude_invalid:
                            invalids += 1
                            continue
                        else:
                            bets.append(10)  # Backward-compatible default
                
                # Calculate metrics
                valid_trials = len(bets)
                total_trials = self.n_trials
                invalid_trials = total_trials - valid_trials
                
                if valid_trials > 0:
                    avg_bet = np.mean(bets)
                    stop_rate = stops / valid_trials
                else:
                    avg_bet = 0.0
                    stop_rate = 0.0
                
                # Store result
                result = {
                    'layer': layer,
                    'feature_id': feature_id,
                    'prompt_type': prompt_type,
                    'scale': scale,
                    'avg_bet': float(avg_bet),
                    'stop_rate': float(stop_rate),
                    'bankruptcy_risk': float(1 - stop_rate),  # Inverse of stop rate
                    'original_value': float(original_value),
                    'patched_value': float(patched_value) if patched_value else None,
                    'bankrupt_mean': float(bankrupt_mean),
                    'safe_mean': float(safe_mean),
                    'n_trials': self.n_trials,
                    'valid_trials': int(valid_trials),
                    'invalid_trials': int(invalid_trials),
                    'invalid_rate': float(invalid_trials / total_trials if total_trials else 0.0),
                    'exclude_invalid': bool(self.exclude_invalid)
                }
                
                results.append(result)
                
                print(f"  {prompt_type} scale={scale:.1f}: bet=${avg_bet:.1f}, stop={stop_rate:.2f}")
        
        return results
    
    def analyze_causality(self, feature_results: List[Dict]) -> Dict:
        """Analyze if feature shows causal effect"""
        # Separate by prompt type
        risky_results = [r for r in feature_results if r['prompt_type'] == 'risky']
        safe_results = [r for r in feature_results if r['prompt_type'] == 'safe']
        
        # Extract scales and metrics
        scales = [r['scale'] for r in risky_results]
        
        risky_bets = [r['avg_bet'] for r in risky_results]
        risky_stops = [r['stop_rate'] for r in risky_results]
        
        safe_bets = [r['avg_bet'] for r in safe_results]
        safe_stops = [r['stop_rate'] for r in safe_results]
        
        # Calculate correlations (Spearman for monotonic relationship)
        from scipy.stats import spearmanr
        
        # Betting correlations
        risky_bet_corr, risky_bet_p = spearmanr(scales, risky_bets)
        safe_bet_corr, safe_bet_p = spearmanr(scales, safe_bets)
        
        # Stop rate correlations (should be negative if feature increases risk)
        risky_stop_corr, risky_stop_p = spearmanr(scales, risky_stops)
        safe_stop_corr, safe_stop_p = spearmanr(scales, safe_stops)
        
        # Effect sizes
        bet_effect_risky = max(risky_bets) - min(risky_bets)
        bet_effect_safe = max(safe_bets) - min(safe_bets)
        stop_effect_risky = abs(max(risky_stops) - min(risky_stops))
        stop_effect_safe = abs(max(safe_stops) - min(safe_stops))
        
        # Determine causality - CORRECTED: Include p-value check
        is_causal_bet = ((abs(risky_bet_corr) > 0.5 and risky_bet_p < 0.05) or 
                         (abs(safe_bet_corr) > 0.5 and safe_bet_p < 0.05)) and \
                        (bet_effect_risky > 5 or bet_effect_safe > 5)
        
        is_causal_stop = ((abs(risky_stop_corr) > 0.5 and risky_stop_p < 0.05) or 
                          (abs(safe_stop_corr) > 0.5 and safe_stop_p < 0.05)) and \
                         (stop_effect_risky > 0.1 or stop_effect_safe > 0.1)
        
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
        """Run the full experiment"""
        print("\n" + "="*80)
        print("POPULATION MEAN PATCHING EXPERIMENT")
        print("="*80)
        
        # Load models
        self.load_models()
        
        # Load features
        features = self.load_features()
        
        if end_idx is None:
            end_idx = len(features)
        
        features_to_test = features[start_idx:end_idx]
        
        print(f"\nTesting features {start_idx} to {end_idx} ({len(features_to_test)} total)")
        print(f"Scales: {self.scales}")
        print(f"Trials per condition: {self.n_trials}")
        total_runs = len(features_to_test) * 2 * len(self.scales) * self.n_trials
        print(f"Total runs: {total_runs}")
        print(f"Estimated time: {total_runs * 0.5 / 60:.1f} hours (assuming 0.5s per trial)")
        
        from tqdm import tqdm
        all_results = []
        causal_features_bet = []
        causal_features_stop = []
        
        # Test each feature
        for i, feature in enumerate(tqdm(features_to_test, desc="Testing features")):
            
            # Test feature
            feature_results = self.test_single_feature(feature)
            
            # Analyze causality
            causality = self.analyze_causality(feature_results)
            
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
                    'bet_effect': max(causality['bet_effect_risky'], 
                                     causality['bet_effect_safe'])
                })
            
            if causality['is_causal_stop']:
                causal_features_stop.append({
                    'layer': feature['layer'],
                    'feature_id': feature['feature_id'],
                    'stop_correlation': max(abs(causality['risky_stop_correlation']), 
                                           abs(causality['safe_stop_correlation'])),
                    'stop_effect': max(causality['stop_effect_risky'], 
                                      causality['stop_effect_safe'])
                })
            
            # Save intermediate results every 10 features
            if (i + 1) % 10 == 0:
                self.save_intermediate_results(all_results, causal_features_bet, causal_features_stop, i + 1)
        
        # Final analysis and save
        self.save_final_results(all_results, causal_features_bet, causal_features_stop)
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        print(f"Total features tested: {len(features_to_test)}")
        print(f"Causal features (betting): {len(causal_features_bet)}")
        print(f"Causal features (stop rate): {len(causal_features_stop)}")
        
        return all_results, causal_features_bet, causal_features_stop
    
    def save_intermediate_results(self, results, causal_bet, causal_stop, n_tested):
        """Save intermediate results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        save_data = {
            'timestamp': timestamp,
            'n_features_tested': n_tested,
            'n_causal_bet': len(causal_bet),
            'n_causal_stop': len(causal_stop),
            'causal_features_bet': causal_bet,
            'causal_features_stop': causal_stop,
            'sample_results': results[-100:]  # Last 100 results
        }
        
        filename = self.results_dir / f'patching_intermediate_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Intermediate results saved: {n_tested} features tested")
    
    def save_final_results(self, results, causal_bet, causal_stop):
        """Save final results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sort causal features by effect size
        causal_bet_sorted = sorted(causal_bet, key=lambda x: x['bet_effect'], reverse=True)
        causal_stop_sorted = sorted(causal_stop, key=lambda x: x['stop_effect'], reverse=True)
        
        save_data = {
            'timestamp': timestamp,
            'experiment_config': {
                'scales': self.scales,
                'n_trials': self.n_trials,
                'n_features_tested': 356,
                'risky_prompt_balance': 20,
                'safe_prompt_balance': 140
            },
            'summary': {
                'n_causal_bet': len(causal_bet),
                'n_causal_stop': len(causal_stop),
                'n_causal_any': len(set([f"{c['layer']}-{c['feature_id']}" for c in causal_bet + causal_stop])),
                'causal_rate_bet': len(causal_bet) / 356,
                'causal_rate_stop': len(causal_stop) / 356
            },
            'causal_features_bet': causal_bet_sorted,
            'causal_features_stop': causal_stop_sorted,
            'all_results': results
        }
        
        filename = self.results_dir / f'patching_population_mean_final_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✅ Final results saved to: {filename}")
        
        # Print top causal features
        print("\nTop 5 Causal Features (Betting):")
        for i, feat in enumerate(causal_bet_sorted[:5], 1):
            print(f"  {i}. L{feat['layer']}-{feat['feature_id']}: effect=${feat['bet_effect']:.1f}")
        
        print("\nTop 5 Causal Features (Stop Rate):")
        for i, feat in enumerate(causal_stop_sorted[:5], 1):
            print(f"  {i}. L{feat['layer']}-{feat['feature_id']}: effect={feat['stop_effect']:.3f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Population Mean Patching Experiment')
    parser.add_argument('--gpu', type=str, required=True, help='GPU ID to use (4 or 7)')
    parser.add_argument('--start_idx', type=int, default=0, help='Start feature index')
    parser.add_argument('--end_idx', type=int, default=None, help='End feature index')
    parser.add_argument('--process_id', type=str, default='main', help='Process identifier for output')
    parser.add_argument('--exclude_invalid', action='store_true', help='Exclude invalid trials (parse failures/default/exception) from aggregates')
    args = parser.parse_args()
    
    # Set GPU before importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"🚀 Starting process {args.process_id} on GPU {args.gpu}")
    print(f"📊 Processing features {args.start_idx} to {args.end_idx or 'end'}")
    
    # Import torch after setting GPU
    global torch
    import torch
    
    experiment = PopulationMeanPatchingExperiment()
    experiment.exclude_invalid = bool(args.exclude_invalid)
    experiment.load_models()
    
    # Run experiment with specified range
    results, causal_bet, causal_stop = experiment.run_experiment(args.start_idx, args.end_idx)

if __name__ == "__main__":
    main()
