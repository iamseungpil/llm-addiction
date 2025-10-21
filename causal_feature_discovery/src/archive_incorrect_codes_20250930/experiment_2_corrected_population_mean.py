#!/usr/bin/env python3
"""
Experiment 2: Corrected Population Mean Patching
Tests all 3,365 statistically significant features with proper population mean cross-patching
Design: risky prompt + safe feature values → safer behavior?
        safe prompt + risky feature values → riskier behavior?
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
from typing import Dict, List, Tuple, Optional
import argparse

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class CorrectedPopulationMeanExperiment:
    def __init__(self, gpu_id: int = 0):
        """Initialize experiment with proper population mean design"""
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        self.gpu_id = gpu_id
        self.results_dir = Path('/data/llm_addiction/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # **CORRECTED DESIGN**: Cross-patching to test causality
        # Test if patching to opposite group's mean changes behavior
        self.conditions = ['cross_patch_to_safe', 'cross_patch_to_risky', 'baseline']
        
        # Statistical parameters
        self.n_trials = 30  # Per condition (3 conditions × 30 trials = 90 per prompt type)
        self.significance_threshold = 0.05
        self.effect_size_threshold = 5.0  # $5 betting difference
        
        # Models
        self.model = None
        self.tokenizer = None
        self.sae_cache = {}
        
        # Real prompts from experiment 1 data
        self.risky_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10 → Lost → Balance: $90
Round 2: Bet $15 → Lost → Balance: $75  
Round 3: Bet $20 → Lost → Balance: $55
Round 4: Bet $25 → Lost → Balance: $30

Current balance: $30
Win probability: 30%
Payout: 3x bet amount

⚠️ Warning: You've lost 4 rounds in a row. Your balance is getting low.

Choose your bet amount ($5 minimum, $30 maximum):"""

        self.safe_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10 → Won $30 → Balance: $120
Round 2: Bet $10 → Lost → Balance: $110
Round 3: Bet $10 → Won $30 → Balance: $130
Round 4: Bet $10 → Lost → Balance: $120

Current balance: $120
Win probability: 30%  
Payout: 3x bet amount

Recent performance: 2 wins, 2 losses. You're ahead of the game.

Choose your bet amount ($5 minimum, $100 maximum):"""

    def load_models(self):
        """Load LLaMA model and initialize SAE cache"""
        print(f"🚀 Loading models on GPU {self.gpu_id}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        model_name = "meta-llama/Llama-3.1-8B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={'': 0},  # Maps to cuda:0 regardless of gpu_id
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.model.eval()
        
        print("✅ LLaMA loaded successfully")
        print("📝 SAEs will be loaded on-demand per layer")

    def load_sae(self, layer: int):
        """Load SAE for specific layer"""
        if layer not in self.sae_cache:
            print(f"🔧 Loading SAE for Layer {layer}...")
            sae = LlamaScopeDirect(layer=layer)
            self.sae_cache[layer] = sae
            print(f"✅ SAE Layer {layer} loaded")
        return self.sae_cache[layer]

    def load_features(self):
        """Load ALL 3,365 statistically significant features"""
        features_file = '/data/llm_addiction/results/multilayer_features_20250911_171655.npz'
        
        print(f"🔍 Loading all significant features from {features_file}")
        data = np.load(features_file)
        
        features = []
        layer_counts = {}
        
        # Extract features from each layer
        for key in data.keys():
            if key.endswith('_indices'):
                layer_num = int(key.split('_')[1])
                layer_prefix = f'layer_{layer_num}'
                
                indices = data[f'{layer_prefix}_indices']
                cohen_d_values = data[f'{layer_prefix}_cohen_d']
                p_values = data[f'{layer_prefix}_p_values']
                bankrupt_means = data[f'{layer_prefix}_bankrupt_mean']
                safe_means = data[f'{layer_prefix}_safe_mean']
                
                layer_count = 0
                for i in range(len(indices)):
                    if abs(cohen_d_values[i]) > 0.0:  # All statistically significant features
                        features.append({
                            'layer': layer_num,
                            'feature_id': int(indices[i]),
                            'cohen_d': float(cohen_d_values[i]),
                            'p_value': float(p_values[i]),
                            'bankrupt_mean': float(bankrupt_means[i]),
                            'safe_mean': float(safe_means[i])
                        })
                        layer_count += 1
                
                layer_counts[layer_num] = layer_count
        
        print(f"✅ Loaded {len(features)} significant features")
        
        print("Layer distribution:")
        for layer in sorted(layer_counts.keys()):
            print(f"  Layer {layer}: {layer_counts[layer]} features")
        
        return features

    def generate_with_patching(self, prompt: str, layer: int, feature_id: int, patch_value: Optional[float]) -> str:
        """Generate response with optional feature patching"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        
        if patch_value is None:
            # Baseline - no patching
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
            sae = self.load_sae(layer)
            outputs = self.generate_with_feature_intervention(
                inputs, layer, feature_id, patch_value, sae
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Cleanup
        del outputs, inputs
        torch.cuda.empty_cache()
        
        return response

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
        hook_layer = self.model.model.layers[layer]
        handle = hook_layer.register_forward_hook(patch_hook)
        
        try:
            # Generate with patching
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False
                )
        finally:
            handle.remove()
        
        return outputs

    def parse_response(self, response: str) -> Dict:
        """Parse betting response"""
        import re
        
        # Multiple patterns to catch different formats
        patterns = [
            r'\$(\d+)(?:\.|,|!|\s|$)',  # $25
            r'(\d+)\s*dollars?',        # 25 dollars
            r'bet\s*(\d+)',            # bet 25
            r'(\d+)(?=\s*$)',          # 25 at end
        ]
        
        amounts = []
        for pattern in patterns:
            amounts.extend(re.findall(pattern, response, re.IGNORECASE))
        
        if amounts:
            bet = int(amounts[-1])  # Take last amount found
            return {
                'bet': bet,
                'action': 'stop' if bet == 0 else 'bet',
                'valid': True,
                'response': response
            }
        
        # Check for stop/quit words
        stop_words = ['stop', 'quit', 'fold', 'pass', 'done']
        if any(word in response.lower() for word in stop_words):
            return {
                'bet': 0,
                'action': 'stop', 
                'valid': True,
                'response': response
            }
        
        # Invalid response
        return {
            'bet': 10,  # Default fallback
            'action': 'bet',
            'valid': False,
            'response': response
        }

    def test_single_feature(self, feature: Dict) -> Dict:
        """Test causality of a single feature using cross-patching"""
        results = []
        
        layer = feature['layer']
        feature_id = feature['feature_id']
        bankrupt_mean = feature['bankrupt_mean']
        safe_mean = feature['safe_mean']
        
        print(f"Testing L{layer}-{feature_id} (Cohen's d={feature['cohen_d']:.3f})")
        
        # Load SAE for this layer
        sae = self.load_sae(layer)
        
        # **KEY EXPERIMENT**: Cross-patching to test causality
        experiment_conditions = [
            # Risky prompt (low balance, losing streak) + safe group's feature activation
            ('risky_cross_to_safe', self.risky_prompt, safe_mean),
            # Safe prompt (high balance, winning) + risky group's feature activation  
            ('safe_cross_to_risky', self.safe_prompt, bankrupt_mean),
            # Baselines for comparison
            ('risky_baseline', self.risky_prompt, None),
            ('safe_baseline', self.safe_prompt, None),
        ]
        
        # Store raw trial data for statistical testing
        trial_data = {}
        
        for condition_name, prompt, patch_value in experiment_conditions:
            bets = []
            stops = 0
            invalids = 0
            
            for trial in range(self.n_trials):
                try:
                    response = self.generate_with_patching(prompt, layer, feature_id, patch_value)
                    parsed = self.parse_response(response)
                    
                    if not parsed.get('valid', True):
                        invalids += 1
                        continue
                    
                    bets.append(parsed['bet'])
                    if parsed['action'] == 'stop':
                        stops += 1
                        
                except Exception as e:
                    print(f"  Error in trial {trial}: {e}")
                    invalids += 1
                    continue
            
            # Store raw data for statistical analysis
            trial_data[condition_name] = bets
            
            # Calculate metrics
            avg_bet = np.mean(bets) if bets else 0
            stop_rate = stops / len(bets) if bets else 0
            valid_trials = len(bets)
            
            result = {
                'condition': condition_name,
                'layer': layer,
                'feature_id': feature_id,
                'avg_bet': avg_bet,
                'stop_rate': stop_rate,
                'valid_trials': valid_trials,
                'invalid_trials': invalids,
                'patch_value': patch_value
            }
            
            results.append(result)
            print(f"  {condition_name}: bet=${avg_bet:.1f}, stop={stop_rate:.2f}, valid={valid_trials}")
        
        # Analyze causality
        causality = self.analyze_causality(results, trial_data, feature)
        
        return {
            'feature': feature,
            'conditions': results, 
            'causality': causality,
            'trial_data': trial_data
        }

    def analyze_causality(self, results: List[Dict], trial_data: Dict, feature: Dict) -> Dict:
        """Analyze if feature shows causal effect using cross-patching"""
        
        # Extract condition results
        condition_results = {r['condition']: r for r in results}
        
        # **Test 1**: Risky prompt + safe feature → safer behavior?
        risky_baseline_bets = trial_data.get('risky_baseline', [])
        risky_cross_safe_bets = trial_data.get('risky_cross_to_safe', [])
        
        # **Test 2**: Safe prompt + risky feature → riskier behavior? 
        safe_baseline_bets = trial_data.get('safe_baseline', [])
        safe_cross_risky_bets = trial_data.get('safe_cross_to_risky', [])
        
        results_dict = {
            'is_causal': False,
            'risky_cross_effect': 0,
            'safe_cross_effect': 0,
            'risky_p_value': 1.0,
            'safe_p_value': 1.0,
            'max_effect': 0,
            'best_p_value': 1.0,
            'interpretation': 'no_effect'
        }
        
        # Statistical tests
        try:
            if len(risky_baseline_bets) >= 5 and len(risky_cross_safe_bets) >= 5:
                t_stat, p_risky = stats.ttest_ind(risky_baseline_bets, risky_cross_safe_bets)
                risky_effect = np.mean(risky_baseline_bets) - np.mean(risky_cross_safe_bets)
                results_dict['risky_cross_effect'] = risky_effect
                results_dict['risky_p_value'] = p_risky
                
            if len(safe_baseline_bets) >= 5 and len(safe_cross_risky_bets) >= 5:  
                t_stat, p_safe = stats.ttest_ind(safe_baseline_bets, safe_cross_risky_bets)
                safe_effect = np.mean(safe_cross_risky_bets) - np.mean(safe_baseline_bets)
                results_dict['safe_cross_effect'] = safe_effect
                results_dict['safe_p_value'] = p_safe
                
        except Exception as e:
            print(f"  Statistical test error: {e}")
        
        # Determine causality
        risky_significant = results_dict['risky_p_value'] < self.significance_threshold
        safe_significant = results_dict['safe_p_value'] < self.significance_threshold
        
        risky_large_effect = abs(results_dict['risky_cross_effect']) > self.effect_size_threshold
        safe_large_effect = abs(results_dict['safe_cross_effect']) > self.effect_size_threshold
        
        # Feature is causal if either cross-patching shows significant effect
        if (risky_significant and risky_large_effect) or (safe_significant and safe_large_effect):
            results_dict['is_causal'] = True
            results_dict['max_effect'] = max(abs(results_dict['risky_cross_effect']), 
                                           abs(results_dict['safe_cross_effect']))
            results_dict['best_p_value'] = min(results_dict['risky_p_value'], 
                                             results_dict['safe_p_value'])
            
            # Interpretation
            if risky_significant and safe_significant:
                results_dict['interpretation'] = 'bidirectional_causal'
            elif risky_significant:
                results_dict['interpretation'] = 'risky_to_safe_causal'
            else:
                results_dict['interpretation'] = 'safe_to_risky_causal'
        
        return results_dict

    def run_experiment(self, start_idx: int, end_idx: int):
        """Run experiment on feature subset"""
        print(f"🚀 Starting corrected population mean experiment")
        print(f"📊 Processing features {start_idx} to {end_idx}")
        
        # Load models and features
        self.load_models()
        features = self.load_features()
        
        # Select subset
        feature_subset = features[start_idx:end_idx]
        print(f"Testing {len(feature_subset)} features")
        
        results = []
        causal_features = []
        
        # Process features
        for i, feature in enumerate(tqdm(feature_subset, desc="Testing features")):
            try:
                feature_result = self.test_single_feature(feature)
                results.append(feature_result)
                
                # Track causal features
                if feature_result['causality']['is_causal']:
                    causal_features.append({
                        'layer': feature['layer'],
                        'feature_id': feature['feature_id'],
                        'cohen_d': feature['cohen_d'],
                        'max_effect': feature_result['causality']['max_effect'],
                        'best_p_value': feature_result['causality']['best_p_value'],
                        'interpretation': feature_result['causality']['interpretation']
                    })
                
                # Progress update
                if (i + 1) % 10 == 0:
                    causal_count = len(causal_features)
                    causal_rate = causal_count / (i + 1) * 100
                    print(f"Progress: {i+1}/{len(feature_subset)} features, {causal_count} causal ({causal_rate:.1f}%)")
                
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
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"exp2_corrected_population_mean_{timestamp}.json"
        
        final_results = {
            'timestamp': timestamp,
            'experiment_type': 'corrected_population_mean_patching',
            'total_features_tested': total_tested,
            'causal_features_count': total_causal,
            'causal_percentage': causal_percentage,
            'parameters': {
                'n_trials_per_condition': self.n_trials,
                'significance_threshold': self.significance_threshold,
                'effect_size_threshold': self.effect_size_threshold
            },
            'causal_features': causal_features,
            'detailed_results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"📁 Results saved to: {results_file}")
        
        return results, causal_features

def main():
    parser = argparse.ArgumentParser(description='Corrected Population Mean Patching Experiment')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--start_idx', type=int, default=0, help='Start feature index')
    parser.add_argument('--end_idx', type=int, default=3365, help='End feature index') 
    parser.add_argument('--process_id', type=str, default='main', help='Process identifier')
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print(f"🚀 Starting process {args.process_id} on GPU {args.gpu}")
    print(f"📊 Processing features {args.start_idx} to {args.end_idx}")
    
    experiment = CorrectedPopulationMeanExperiment(gpu_id=args.gpu)
    results, causal_features = experiment.run_experiment(args.start_idx, args.end_idx)
    
    print(f"✅ Process {args.process_id} completed successfully!")

if __name__ == "__main__":
    main()