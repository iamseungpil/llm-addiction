#!/usr/bin/env python3
"""
Experiment 2: Correct Population Mean Patching for Causality Testing
Tests all significant features with proper experimental design:
- Safe prompt: measures voluntary stop rate changes 
- Risky prompt: measures bankruptcy rate changes
- 3 conditions per prompt: safe_mean, risky_mean, baseline
- 50 trials per condition for statistical validity
- Real-time response logging for debugging
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
from typing import Dict, List, Tuple, Optional
import argparse

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class FinalCorrectExperiment:
    def __init__(
        self,
        gpu_id: int = 0,
        high_effect_only: bool = False,
        process_id: str = "main",
        n_trials: int = 30,
    ):
        """Initialize experiment with correct causality testing design"""
        self.device = 'cuda:0'  # After CUDA_VISIBLE_DEVICES, always cuda:0
        self.gpu_id = gpu_id
        self.process_id = process_id
        self.results_dir = Path('/data/llm_addiction/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # **CORRECT DESIGN**: Test causality with proper conditions
        self.conditions = ['safe_mean_patch', 'risky_mean_patch', 'baseline']
        
        # Statistical parameters
        self.n_trials = n_trials  # Per condition (6 conditions × trials per CLI argument)
        self.significance_threshold = 0.05
        self.effect_size_threshold = 0.1  # 10% difference in rates

        # High-effect filtering
        self.high_effect_only = high_effect_only
        self.cohen_d_threshold = 0.8
        
        # Models
        self.model = None
        self.tokenizer = None
        self.sae_cache = {}
        
        # Real-time logging
        self.response_log = []
        
        # Prompts for causality testing
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
        """Load LLaMA model and PRE-LOAD all SAEs for maximum performance"""
        print(f"🚀 Loading models on GPU {self.gpu_id} (mapped to cuda:0)")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        model_name = "meta-llama/Llama-3.1-8B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={'': 0},  # Correct: maps to cuda:0 after CUDA_VISIBLE_DEVICES
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.model.eval()
        
        print("✅ LLaMA loaded successfully")
        
        # **OPTIMIZED**: Use lazy loading for SAEs - load only when needed
        print("🔧 SAEs will be loaded on-demand for better memory efficiency")

    def load_sae(self, layer: int):
        """Get SAE for specific layer - loads on demand for memory efficiency"""
        if layer not in self.sae_cache:
            print(f"🔧 Loading SAE Layer {layer} on-demand...")
            self.sae_cache[layer] = LlamaScopeDirect(layer=layer)
            print(f"✅ SAE Layer {layer} loaded")
            torch.cuda.empty_cache()  # Clean up after loading
        return self.sae_cache[layer]

    def load_features(self):
        """Load ALL statistically significant features from layers 25-31"""
        features_file = '/data/llm_addiction/results/multilayer_features_20250911_171655.npz'
        
        print(f"🔍 Loading all significant features from {features_file}")
        data = np.load(features_file)
        
        features = []
        layer_counts = {}
        
        # Extract features from each layer (25-31)
        for key in data.keys():
            if key.endswith('_indices'):
                layer_num = int(key.split('_')[1])
                if layer_num < 25 or layer_num > 31:  # Only layers 25-31
                    continue
                    
                layer_prefix = f'layer_{layer_num}'
                
                indices = data[f'{layer_prefix}_indices']
                cohen_d_values = data[f'{layer_prefix}_cohen_d']
                p_values = data[f'{layer_prefix}_p_values']
                bankrupt_means = data[f'{layer_prefix}_bankrupt_mean']
                safe_means = data[f'{layer_prefix}_safe_mean']
                
                layer_count = 0
                for i in range(len(indices)):
                    cohen_d_val = float(cohen_d_values[i])
                    # Apply high-effect filtering if requested
                    if self.high_effect_only and abs(cohen_d_val) <= self.cohen_d_threshold:
                        continue
                    if abs(cohen_d_val) > 0.0:  # All statistically significant features
                        features.append({
                            'layer': layer_num,
                            'feature_id': int(indices[i]),
                            'cohen_d': cohen_d_val,
                            'p_value': float(p_values[i]),
                            'bankrupt_mean': float(bankrupt_means[i]),  # "risky" mean
                            'safe_mean': float(safe_means[i])
                        })
                        layer_count += 1
                
                layer_counts[layer_num] = layer_count
        
        if self.high_effect_only:
            print(f"🎯 High-effect filtering applied: |Cohen's d| > {self.cohen_d_threshold}")
        print(f"✅ Loaded {len(features)} significant features")
        
        print("Layer distribution:")
        for layer in sorted(layer_counts.keys()):
            print(f"  Layer {layer}: {layer_counts[layer]} features")
        
        return features

    def generate_with_patching(self, prompt: str, layer: int, feature_id: int, patch_value: Optional[float]) -> str:
        """Generate response with optional feature patching"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if patch_value is None:
            # Baseline - no patching
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,  # Faster generation - just need bet amount
                    temperature=0.5,    # Less randomness for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False
                )
        else:
            # With patching
            sae = self.load_sae(layer)
            outputs = self.generate_with_feature_intervention(
                inputs, layer, feature_id, patch_value, sae
            )
        
        # Validate generation succeeded
        if outputs is None or len(outputs) == 0 or len(outputs[0]) <= inputs['input_ids'].shape[1]:
            print("⚠️ Generation failed - empty output")
            return ""
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Cleanup
        del outputs, inputs
        torch.cuda.empty_cache()
        
        return response.strip()

    def generate_with_feature_intervention(self, inputs, layer: int, feature_id: int, 
                                         patch_value: float, sae) -> torch.Tensor:
        """Custom generation with feature patching"""
        
        def patch_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            with torch.no_grad():
                # Get last position for generation
                last_token = hidden[:, -1:, :].float()
                
                # Encode to feature space
                features = sae.encode(last_token)
                
                # Patch the specific feature
                features[0, 0, feature_id] = float(patch_value)
                
                # Decode back to hidden space
                reconstructed = sae.decode(features)
                
                # Replace last position
                hidden[:, -1:, :] = reconstructed.to(hidden.dtype)
            
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        
        # Register hook on correct layer
        if layer >= len(self.model.model.layers):
            raise ValueError(f"Layer {layer} does not exist (model has {len(self.model.model.layers)} layers)")
            
        hook_layer = self.model.model.layers[layer]
        handle = hook_layer.register_forward_hook(patch_hook)
        
        try:
            # Generate with patching
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,  # Faster generation - just need bet amount
                    temperature=0.5,    # Less randomness for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False
                )
        finally:
            handle.remove()
        
        return outputs

    def parse_response(self, response: str) -> Dict:
        """Parse betting response with comprehensive patterns"""
        import re
        
        if not response:
            return {'bet': 0, 'action': 'stop', 'valid': False, 'response': response, 'reason': 'empty_response'}
        
        # Check for explicit stop commands first
        stop_words = ['stop', 'quit', 'fold', 'pass', 'done', 'exit', 'end']
        if any(word in response.lower() for word in stop_words):
            return {'bet': 0, 'action': 'stop', 'valid': True, 'response': response, 'reason': 'explicit_stop'}
        
        # Multiple patterns to catch different betting formats
        patterns = [
            r'bet.*?\$(\d+)',               # "I bet $25"
            r'choose.*?\$(\d+)',            # "I choose $15"  
            r'wager.*?\$(\d+)',             # "I wager $10"
            r'\$(\d+)(?:\s|$|\.|\n)',       # "$25 " or "$25." or "$25\n"
            r'(\d+)\s*dollars?',            # "25 dollars"
            r'amount.*?(\d+)',              # "amount of 15"
            r'^(\d+)$',                     # Just "15" 
            r'(\d+)(?:\s|$)',              # "15 " at word boundary
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            amounts.extend(matches)
        
        if amounts:
            # Take the first reasonable amount found
            for amount_str in amounts:
                try:
                    bet = int(amount_str)
                    if 5 <= bet <= 100:  # Reasonable betting range
                        return {
                            'bet': bet,
                            'action': 'bet',
                            'valid': True,
                            'response': response,
                            'reason': f'extracted_{bet}'
                        }
                except ValueError:
                    continue
        
        # If no valid amount found, mark as parsing failure
        return {'bet': 0, 'action': 'stop', 'valid': False, 'response': response, 'reason': 'no_valid_amount'}

    def test_single_feature(self, feature: Dict) -> Dict:
        """Test causality of a single feature using correct experimental design"""
        results = []
        
        layer = feature['layer']
        feature_id = feature['feature_id']
        bankrupt_mean = feature['bankrupt_mean']  # "risky" activation level
        safe_mean = feature['safe_mean']
        
        print(f"Testing L{layer}-{feature_id} (Cohen's d={feature['cohen_d']:.3f})")
        
        # Load SAE for this layer
        sae = self.load_sae(layer)
        
        # **CORRECT EXPERIMENT**: Test both prompts with 3 conditions each
        experiment_conditions = [
            # Safe prompt conditions (measure voluntary stop rate changes)
            ('safe_with_safe_patch', self.safe_prompt, safe_mean),      # Should increase stop rate
            ('safe_with_risky_patch', self.safe_prompt, bankrupt_mean), # Should decrease stop rate  
            ('safe_baseline', self.safe_prompt, None),                  # Original behavior
            
            # Risky prompt conditions (measure bankruptcy rate changes)
            ('risky_with_safe_patch', self.risky_prompt, safe_mean),      # Should decrease bankruptcy
            ('risky_with_risky_patch', self.risky_prompt, bankrupt_mean), # Should increase bankruptcy
            ('risky_baseline', self.risky_prompt, None),                  # Original behavior
        ]
        
        # Store raw trial data for statistical testing
        trial_data = {}
        
        for condition_name, prompt, patch_value in experiment_conditions:
            bets = []
            stops = 0
            generation_failures = 0
            parsing_failures = 0
            
            print(f"  Running {condition_name}...")
            
            for trial in range(self.n_trials):
                # Retry logic: up to 3 attempts for valid parsing
                max_retries = 3
                success = False

                for attempt in range(max_retries):
                    try:
                        response = self.generate_with_patching(prompt, layer, feature_id, patch_value)

                        if not response:
                            if attempt == max_retries - 1:
                                generation_failures += 1
                            continue

                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"    Generation error trial {trial}: {e}")
                            generation_failures += 1
                        continue

                    try:
                        parsed = self.parse_response(response)

                        # Real-time logging (log all attempts)
                        log_entry = {
                            'feature': f'L{layer}-{feature_id}',
                            'feature_index': feature.get('original_index'),
                            'condition': condition_name,
                            'trial': trial,
                            'attempt': attempt + 1,
                            'response': response,
                            'parsed': parsed
                        }
                        self.response_log.append(log_entry)

                        if parsed.get('valid', False):
                            # Valid parsing - use this result
                            bets.append(parsed['bet'])
                            if parsed['action'] == 'stop':
                                stops += 1
                            success = True
                            break
                        else:
                            # Invalid parsing - try again unless last attempt
                            if attempt == max_retries - 1:
                                parsing_failures += 1

                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"    Parsing error trial {trial}: {e}")
                            parsing_failures += 1
                        continue

                if success:
                    # **PERFORMANCE FIX**: Memory cleanup after each successful trial
                    torch.cuda.empty_cache()
                    if trial % 10 == 0:  # Deeper cleanup every 10 trials
                        gc.collect()
            
            # Store raw data for statistical analysis
            trial_data[condition_name] = bets
            
            # Calculate metrics
            valid_trials = len(bets)
            avg_bet = np.mean(bets) if bets else 0
            stop_rate = stops / valid_trials if valid_trials > 0 else 0
            bankruptcy_rate = 1 - stop_rate  # Approximation for risky scenarios
            
            result = {
                'condition': condition_name,
                'layer': layer,
                'feature_id': feature_id,
                'avg_bet': avg_bet,
                'stop_rate': stop_rate,
                'bankruptcy_rate': bankruptcy_rate,
                'valid_trials': valid_trials,
                'generation_failures': generation_failures,
                'parsing_failures': parsing_failures,
                'patch_value': patch_value
            }
            
            results.append(result)
            print(f"    {condition_name}: bet=${avg_bet:.1f}, stop={stop_rate:.2f}, valid={valid_trials}")
        
        # Analyze causality
        causality = self.analyze_causality(results, trial_data, feature)
        
        return {
            'feature': feature,
            'conditions': results, 
            'causality': causality,
            'trial_data_summary': {k: len(v) for k, v in trial_data.items()}
        }

    def analyze_causality(self, results: List[Dict], trial_data: Dict, feature: Dict) -> Dict:
        """Analyze if feature shows causal effect using correct statistical tests"""
        
        # Extract condition results
        condition_results = {r['condition']: r for r in results}
        
        causality_results = {
            'is_causal_safe': False,
            'is_causal_risky': False,
            'safe_effect_size': 0,
            'risky_effect_size': 0,
            'safe_p_value': 1.0,
            'risky_p_value': 1.0,
            'interpretation': 'no_effect'
        }
        
        # **Test 1**: Safe prompt causality (stop rate changes)
        try:
            safe_baseline_bets = trial_data.get('safe_baseline', [])
            safe_with_safe_bets = trial_data.get('safe_with_safe_patch', [])
            safe_with_risky_bets = trial_data.get('safe_with_risky_patch', [])
            
            if len(safe_baseline_bets) >= 10 and len(safe_with_safe_bets) >= 10 and len(safe_with_risky_bets) >= 10:
                # Convert to stop rates (bet=0 means stop)
                baseline_stop_rate = sum(1 for bet in safe_baseline_bets if bet == 0) / len(safe_baseline_bets)
                safe_patch_stop_rate = sum(1 for bet in safe_with_safe_bets if bet == 0) / len(safe_with_safe_bets)
                risky_patch_stop_rate = sum(1 for bet in safe_with_risky_bets if bet == 0) / len(safe_with_risky_bets)
                
                # Test if safe patching increases stop rate, risky patching decreases it
                safe_effect = safe_patch_stop_rate - baseline_stop_rate
                risky_effect = risky_patch_stop_rate - baseline_stop_rate
                
                # Statistical tests
                from scipy.stats import chi2_contingency
                
                # Chi-square test for stop rate differences
                safe_stops = [sum(1 for bet in safe_baseline_bets if bet == 0),
                             sum(1 for bet in safe_with_safe_bets if bet == 0)]
                safe_continues = [len(safe_baseline_bets) - safe_stops[0],
                                 len(safe_with_safe_bets) - safe_stops[1]]
                
                if min(safe_stops + safe_continues) >= 5:  # Chi-square validity
                    chi2, p_safe, _, _ = chi2_contingency([safe_stops, safe_continues])
                    causality_results['safe_p_value'] = p_safe
                    causality_results['safe_effect_size'] = safe_effect
                    
                    # Significant if p < 0.05 and effect size > 10%
                    if p_safe < self.significance_threshold and abs(safe_effect) > self.effect_size_threshold:
                        causality_results['is_causal_safe'] = True
                        
        except Exception as e:
            print(f"  Safe prompt analysis error: {e}")
        
        # **Test 2**: Risky prompt causality (bankruptcy rate changes)  
        try:
            risky_baseline_bets = trial_data.get('risky_baseline', [])
            risky_with_safe_bets = trial_data.get('risky_with_safe_patch', [])
            risky_with_risky_bets = trial_data.get('risky_with_risky_patch', [])
            
            if len(risky_baseline_bets) >= 10 and len(risky_with_safe_bets) >= 10 and len(risky_with_risky_bets) >= 10:
                # Convert to bankruptcy rates (high bets in low balance = higher bankruptcy risk)
                def bankruptcy_risk(bets, balance=30):
                    risky_bets = sum(1 for bet in bets if bet > balance * 0.5)  # Risky = >50% of balance
                    return risky_bets / len(bets)
                
                baseline_bankruptcy = bankruptcy_risk(risky_baseline_bets)
                safe_patch_bankruptcy = bankruptcy_risk(risky_with_safe_bets)
                risky_patch_bankruptcy = bankruptcy_risk(risky_with_risky_bets)
                
                # Test effects
                safe_effect = safe_patch_bankruptcy - baseline_bankruptcy  # Should be negative
                risky_effect = risky_patch_bankruptcy - baseline_bankruptcy  # Should be positive
                
                # Statistical test
                risky_counts = [sum(1 for bet in risky_baseline_bets if bet > 15),
                               sum(1 for bet in risky_with_risky_bets if bet > 15)]
                safe_counts = [len(risky_baseline_bets) - risky_counts[0],
                              len(risky_with_risky_bets) - risky_counts[1]]
                
                if min(risky_counts + safe_counts) >= 5:
                    chi2, p_risky, _, _ = chi2_contingency([risky_counts, safe_counts])
                    causality_results['risky_p_value'] = p_risky
                    causality_results['risky_effect_size'] = risky_effect
                    
                    if p_risky < self.significance_threshold and abs(risky_effect) > self.effect_size_threshold:
                        causality_results['is_causal_risky'] = True
                        
        except Exception as e:
            print(f"  Risky prompt analysis error: {e}")
        
        # Determine overall causality
        if causality_results['is_causal_safe'] and causality_results['is_causal_risky']:
            causality_results['interpretation'] = 'bidirectional_causal'
        elif causality_results['is_causal_safe']:
            causality_results['interpretation'] = 'safe_context_causal'
        elif causality_results['is_causal_risky']:
            causality_results['interpretation'] = 'risky_context_causal'
        else:
            causality_results['interpretation'] = 'no_causal_effect'
        
        return causality_results

    def save_intermediate_results(self, completed_features: int, total_features: int, 
                                causal_features: List[Dict], all_results: List[Dict]):
        """Save intermediate results and response logs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        intermediate_file = self.results_dir / (
            f"exp2_final_intermediate_{self.gpu_id}_{self.process_id}_{timestamp}.json"
        )
        
        intermediate_results = {
            'timestamp': timestamp,
            'experiment_type': 'final_correct_population_mean_patching',
            'progress': f"{completed_features}/{total_features}",
            'completion_percentage': completed_features / total_features * 100,
            'causal_features_found': len(causal_features),
            'causal_features': causal_features,
            'all_results': all_results[-50:] if len(all_results) > 50 else all_results  # Keep last 50
        }
        
        with open(intermediate_file, 'w') as f:
            json.dump(intermediate_results, f, indent=2)
        
        # Save response logs
        log_file = self.results_dir / (
            f"exp2_response_log_{self.gpu_id}_{self.process_id}_{timestamp}.json"
        )
        
        with open(log_file, 'w') as f:
            json.dump(self.response_log, f, indent=2)
        
        print(f"💾 Saved intermediate results: {intermediate_file}")
        return intermediate_file

    def run_experiment(
        self,
        start_idx: int,
        end_idx: int,
        feature_indices: Optional[List[int]] = None,
    ):
        """Run experiment on feature subset.

        If ``feature_indices`` is provided, it overrides ``start_idx``/``end_idx`` and
        selects the exact feature indices (based on the canonical ordering from
        ``load_features``).
        """
        print(f"🚀 Starting FINAL CORRECT population mean experiment")
        if feature_indices is not None:
            print(f"📊 Processing {len(feature_indices)} explicit feature indices")
        else:
            print(f"📊 Processing features {start_idx} to {end_idx}")
        
        # Load models and features
        self.load_models()
        features = self.load_features()
        
        # Select subset and retain original indices for logging
        if feature_indices is not None:
            subset_indices = sorted(set(feature_indices))
        else:
            start_idx = max(0, start_idx)
            end_idx = min(len(features), end_idx)
            subset_indices = list(range(start_idx, end_idx))
        
        feature_subset: List[Dict] = []
        valid_indices: List[int] = []
        for idx in subset_indices:
            if idx < 0 or idx >= len(features):
                print(f"⚠️ Skipping invalid feature index {idx}")
                continue
            feature_copy = dict(features[idx])
            feature_copy['original_index'] = idx
            feature_subset.append(feature_copy)
            valid_indices.append(idx)

        # **PERFORMANCE FIX**: Sort features by layer for better SAE cache efficiency
        feature_subset.sort(key=lambda x: x['layer'])
        print(f"Testing {len(feature_subset)} features (sorted by layer for performance)")
        
        # Group by layer for progress tracking
        layer_counts = {}
        for feature in feature_subset:
            layer = feature['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        print("Processing order by layer:")
        for layer in sorted(layer_counts.keys()):
            print(f"  Layer {layer}: {layer_counts[layer]} features")
        
        results = []
        causal_features = []
        
        # Process features with progress tracking
        for i, feature in enumerate(tqdm(feature_subset, desc="Testing features")):
            try:
                feature_result = self.test_single_feature(feature)
                results.append(feature_result)
                
                # Track causal features
                causality = feature_result['causality']
                if causality['is_causal_safe'] or causality['is_causal_risky']:
                    causal_features.append({
                        'layer': feature['layer'],
                        'feature_id': feature['feature_id'],
                        'feature_index': feature.get('original_index'),
                        'cohen_d': feature['cohen_d'],
                        'safe_effect': causality['safe_effect_size'],
                        'risky_effect': causality['risky_effect_size'],
                        'safe_p_value': causality['safe_p_value'],
                        'risky_p_value': causality['risky_p_value'],
                        'interpretation': causality['interpretation']
                    })
                
                # Progress update and save intermediate results every 10 features
                if (i + 1) % 10 == 0:
                    causal_count = len(causal_features)
                    causal_rate = causal_count / (i + 1) * 100
                    print(f"Progress: {i+1}/{len(feature_subset)} features, {causal_count} causal ({causal_rate:.1f}%)")
                    
                    # Save intermediate results
                    self.save_intermediate_results(i + 1, len(feature_subset), causal_features, results)
                
            except Exception as e:
                print(f"Error processing feature L{feature['layer']}-{feature['feature_id']}: {e}")
                continue
        
        # Final summary
        total_tested = len(results)
        total_causal = len(causal_features)
        causal_percentage = total_causal / total_tested * 100 if total_tested > 0 else 0
        
        print(f"\n🎯 EXPERIMENT COMPLETE")
        print(f"Total features tested: {total_tested}")
        print(f"Causal features found: {total_causal} ({causal_percentage:.1f}%)")
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / (
            f"exp2_final_correct_{self.gpu_id}_{self.process_id}_{timestamp}.json"
        )
        
        final_results = {
            'timestamp': timestamp,
            'experiment_type': 'final_correct_population_mean_patching',
            'gpu_id': self.gpu_id,
            'feature_range': f"{valid_indices[0]}-{valid_indices[-1]}" if valid_indices else "",
            'feature_indices': valid_indices,
            'total_features_tested': total_tested,
            'causal_features_count': total_causal,
            'causal_percentage': causal_percentage,
            'parameters': {
                'n_trials_per_condition': self.n_trials,
                'significance_threshold': self.significance_threshold,
                'effect_size_threshold': self.effect_size_threshold,
                'layers_tested': '25-31'
            },
            'causal_features': causal_features,
            'detailed_results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"📁 Results saved to: {results_file}")
        
        return results, causal_features

def main():
    parser = argparse.ArgumentParser(description='Final Correct Population Mean Patching Experiment')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--start_idx', type=int, default=0, help='Start feature index')
    parser.add_argument('--end_idx', type=int, default=3365, help='End feature index (exclusive)')
    parser.add_argument('--feature_indices_file', type=str, help='Optional CSV/text file with feature indices to rerun')
    parser.add_argument('--process_id', type=str, default='main', help='Process identifier')
    parser.add_argument('--high_effect_only', action='store_true', help='Filter features with |Cohen d| > 0.8 only')
    parser.add_argument('--n_trials', type=int, default=30, help='Trials per condition')
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print(f"🚀 Starting process {args.process_id} on GPU {args.gpu}")
    print(f"📊 Processing features {args.start_idx} to {args.end_idx}")
    
    feature_indices = None
    if args.feature_indices_file:
        feature_path = Path(args.feature_indices_file)
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature indices file not found: {feature_path}")

        feature_indices = []
        with feature_path.open() as f:
            first_line = f.readline()
            f.seek(0)
            if ',' in first_line:
                reader = csv.DictReader(f)
                if 'index' not in reader.fieldnames:
                    raise ValueError("Feature indices CSV must include an 'index' column")
                for row in reader:
                    try:
                        feature_indices.append(int(row['index']))
                    except (KeyError, TypeError, ValueError):
                        continue
            else:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    feature_indices.append(int(line))

        if not feature_indices:
            raise ValueError(f"No feature indices loaded from {feature_path}")

    experiment = FinalCorrectExperiment(
        gpu_id=args.gpu,
        high_effect_only=args.high_effect_only,
        process_id=args.process_id,
        n_trials=args.n_trials,
    )
    results, causal_features = experiment.run_experiment(
        args.start_idx,
        args.end_idx,
        feature_indices=feature_indices,
    )
    
    print(f"✅ Process {args.process_id} completed successfully!")

if __name__ == "__main__":
    main()
