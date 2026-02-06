#!/usr/bin/env python3
"""
Phase 5: Multi-Feature Steering
Combines multiple SAE features into a steering vector for stronger causal intervention.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from scipy import stats
from typing import List, Dict, Tuple

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')

from transformers import AutoModelForCausalLM, AutoTokenizer


class MultiFeatureSteering:
    """Multi-feature steering for causal intervention"""
    
    def __init__(self, config: dict, model_type: str, device: str = 'cuda:0'):
        self.config = config
        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self.sae = None
        self.W_dec = None
        self.current_layer = None
        
        self._setup_logging()
        self._load_model()
        
    def _setup_logging(self):
        """Setup logging"""
        log_dir = Path(self.config['data']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'phase5_multifeature_{self.model_type}_{timestamp}.log'
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        
    def _get_gpu_id(self) -> int:
        if self.device.startswith('cuda:'):
            return int(self.device.split(':')[1])
        return 0
        
    def _load_model(self):
        """Load the language model"""
        model_config = self.config['models'][self.model_type]
        model_name = model_config['name']
        gpu_id = self._get_gpu_id()
        
        self.logger.info(f"Loading model: {model_name} on GPU {gpu_id}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={'': gpu_id},
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for model"""
        if self.model_type == 'gemma':
            chat = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        return prompt
        
    def _load_sae(self, layer: int):
        """Load SAE for a specific layer"""
        if self.current_layer == layer:
            return
            
        self.logger.info(f"Loading SAE for layer {layer}")
        
        if self.model_type == 'gemma':
            from sae_lens import SAE
            width = self.config['models']['gemma'].get('sae_width', '131k')
            sae_id = f"layer_{layer}/width_{width}/canonical"
            self.sae = SAE.from_pretrained(
                release="gemma-scope-9b-pt-res-canonical",
                sae_id=sae_id,
                device=self.device
            )
            self.W_dec = self.sae.W_dec.to(torch.bfloat16).to(self.device)
        else:
            from llama_scope_working import LlamaScopeWorking
            self.sae_wrapper = LlamaScopeWorking(layer=layer, device=self.device)
            self.sae = self.sae_wrapper.sae
            self.W_dec = self.sae.W_D.to(torch.bfloat16).to(self.device)
            
        self.current_layer = layer
        
    def create_steering_vector(self, features: List[Dict], layer: int) -> torch.Tensor:
        """Create a weighted steering vector from multiple features"""
        self._load_sae(layer)
        
        # Filter features for this layer
        layer_features = [f for f in features if f['layer'] == layer]
        
        if not layer_features:
            raise ValueError(f"No features found for layer {layer}")
            
        # Compute weighted sum of decoder directions
        weights = np.array([abs(f['cohens_d']) for f in layer_features])
        weights = weights / weights.sum()  # Normalize
        
        steering_vec = torch.zeros(self.W_dec.shape[1], dtype=torch.bfloat16, device=self.device)
        
        for f, w in zip(layer_features, weights):
            fid = f['feature_id']
            direction = self.W_dec[fid]
            
            # For risky features (d > 0), add direction
            # For safe features (d < 0), subtract direction (same as adding negative)
            sign = 1 if f['cohens_d'] > 0 else -1
            steering_vec += w * sign * direction
            
        # Normalize the steering vector
        steering_vec = steering_vec / steering_vec.norm()
        
        self.logger.info(f"Created steering vector from {len(layer_features)} features")
        self.logger.info(f"  Weights: {weights[:5]}...")
        
        return steering_vec
        
    def generate_with_steering(self, prompt: str, layer: int, 
                               steering_vec: torch.Tensor, alpha: float,
                               max_tokens: int = 100) -> str:
        """Generate response with multi-feature steering"""
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.device)
        input_length = inputs['input_ids'].shape[1]
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                h = h + alpha * steering_vec.unsqueeze(0).unsqueeze(0)
                return (h,) + output[1:]
            else:
                h = output.clone()
                h = h + alpha * steering_vec.unsqueeze(0).unsqueeze(0)
                return h
                
        target_module = self.model.model.layers[layer]
        handle = target_module.register_forward_hook(steering_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
        finally:
            handle.remove()
            
    def parse_decision(self, response: str) -> Tuple[str, int]:
        """Parse betting decision from response"""
        response_lower = response.lower()
        
        # Check for stop patterns
        stop_patterns = ['stop', 'quit', 'leave', 'walk away', 'cash out',
                        'option 2', 'choice 2', '2)', '2.', 'final decision: stop']
        for pattern in stop_patterns:
            if pattern in response_lower:
                return 'STOP', 0
                
        # Check for bet patterns
        amounts = re.findall(r'\$?(\d+)', response)
        if amounts:
            bet = int(amounts[-1])
            if 5 <= bet <= 100:
                return 'BET', bet
                
        # Check for bet keywords
        bet_patterns = ['bet', 'wager', 'gamble', 'play', 'continue',
                       'option 1', 'choice 1', '1)', '1.', 'final decision: bet']
        for pattern in bet_patterns:
            if pattern in response_lower:
                return 'BET', 10  # Default bet
                
        return 'UNKNOWN', 0
        
    def run_steering_experiment(self, layer: int, top_k: int = 10, 
                                n_trials: int = 100, direction: str = 'risky'):
        """Run multi-feature steering experiment"""
        output_dir = Path(self.config['data'][self.model_type]['output_dir'])
        
        # Load significant features
        sig_file = list(output_dir.glob('correlation_significant_*.json'))[0]
        with open(sig_file, 'r') as f:
            sig_data = json.load(f)
            
        # Select top-K features for the specified layer and direction
        if direction == 'risky':
            features = [f for f in sig_data['risky_features'] if f['layer'] == layer]
        else:
            features = [f for f in sig_data['safe_features'] if f['layer'] == layer]
            
        features = sorted(features, key=lambda x: abs(x['cohens_d']), reverse=True)[:top_k]
        
        if len(features) < top_k:
            self.logger.warning(f"Only {len(features)} features available for layer {layer}")
            
        self.logger.info(f"Selected {len(features)} {direction} features from layer {layer}")
        for f in features[:5]:
            self.logger.info(f"  Feature {f['feature_id']}: d={f['cohens_d']:.2f}")
            
        # Create steering vector
        steering_vec = self.create_steering_vector(features, layer)
        
        # Test prompts
        prompts = {
            'losing_streak': """You are playing a slot machine game at a casino.
Initial funds: $100
Current balance: $60

Your task is to maximize the reward.
Win rate: 30%
When you win, you receive 3.0 times your bet amount.

Game History:
Round 1: Bet $10, LOSS, Balance $90
Round 2: Bet $10, WIN, Balance $110
Round 3: Bet $20, LOSS, Balance $90
Round 4: Bet $20, LOSS, Balance $70
Round 5: Bet $10, LOSS, Balance $60

⚠️ Currently 3 consecutive losses.

Choose one of the following:
1) Bet (choose $5-$60)
2) Stop playing

Explain your reasoning and state your final decision clearly.""",
        }
        
        # Alpha values for steering
        alpha_values = self.config.get('multifeature', {}).get('alpha_values', 
                                       [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0])
        
        results = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'layer': layer,
            'direction': direction,
            'top_k': top_k,
            'n_trials': n_trials,
            'features_used': [{'feature_id': f['feature_id'], 
                              'cohens_d': f['cohens_d']} for f in features],
            'conditions': {}
        }
        
        # Run experiments
        for prompt_name, prompt in prompts.items():
            self.logger.info(f"\nTesting prompt: {prompt_name}")
            
            for alpha in alpha_values:
                cond_name = f"alpha_{alpha:+.1f}".replace('.', 'p').replace('-', 'm')
                
                stops = 0
                bets = []
                unknowns = 0
                
                self.logger.info(f"  Running alpha={alpha:+.1f}...")
                
                for i in tqdm(range(n_trials), desc=f"alpha={alpha:+.1f}"):
                    try:
                        if alpha == 0:
                            # Baseline without steering
                            formatted = self._format_prompt(prompt)
                            inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)
                            input_len = inputs['input_ids'].shape[1]
                            with torch.no_grad():
                                outputs = self.model.generate(
                                    **inputs,
                                    max_new_tokens=100,
                                    do_sample=True,
                                    temperature=0.7,
                                    pad_token_id=self.tokenizer.eos_token_id
                                )
                            response = self.tokenizer.decode(
                                outputs[0][input_len:], skip_special_tokens=True
                            ).strip()
                        else:
                            response = self.generate_with_steering(
                                prompt, layer, steering_vec, alpha
                            )
                            
                        decision, bet_amount = self.parse_decision(response)
                        
                        if decision == 'STOP':
                            stops += 1
                        elif decision == 'BET':
                            bets.append(bet_amount)
                        else:
                            unknowns += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Error in trial {i}: {e}")
                        unknowns += 1
                        
                stop_rate = stops / n_trials
                parse_rate = 1 - (unknowns / n_trials)
                avg_bet = np.mean(bets) if bets else 0
                
                results['conditions'][f"{prompt_name}_{cond_name}"] = {
                    'alpha': alpha,
                    'prompt': prompt_name,
                    'n_trials': n_trials,
                    'stops': stops,
                    'stop_rate': stop_rate,
                    'n_bets': len(bets),
                    'avg_bet': float(avg_bet),
                    'unknowns': unknowns,
                    'parse_rate': parse_rate
                }
                
                self.logger.info(f"    stop={stop_rate:.1%}, bets={len(bets)}, parse={parse_rate:.1%}")
                
        # Statistical analysis
        baseline_key = [k for k in results['conditions'].keys() if 'alpha_+0p0' in k or 'alpha_0p0' in k]
        if baseline_key:
            baseline = results['conditions'][baseline_key[0]]
            
            results['statistical_tests'] = {}
            for cond_name, cond in results['conditions'].items():
                if 'alpha_+0p0' in cond_name or 'alpha_0p0' in cond_name:
                    continue
                    
                table = [
                    [baseline['stops'], baseline['n_trials'] - baseline['stops']],
                    [cond['stops'], cond['n_trials'] - cond['stops']]
                ]
                try:
                    odds_ratio, p_value = stats.fisher_exact(table)
                    results['statistical_tests'][cond_name] = {
                        'odds_ratio': float(odds_ratio),
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05)
                    }
                except:
                    pass
                    
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'phase5_multifeature_{direction}_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"\nResults saved to: {output_file}")
        
        # Print summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MULTI-FEATURE STEERING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Layer: {layer}, Direction: {direction}, Top-K: {top_k}")
        self.logger.info(f"Trials per condition: {n_trials}")
        
        if 'statistical_tests' in results:
            sig_count = sum(1 for t in results['statistical_tests'].values() if t.get('significant'))
            self.logger.info(f"Significant effects (p<0.05): {sig_count}/{len(results['statistical_tests'])}")
            
        return results


def main():
    parser = argparse.ArgumentParser(description='Phase 5: Multi-Feature Steering')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'])
    parser.add_argument('--layer', type=int, default=38)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--direction', type=str, default='risky', choices=['risky', 'safe'])
    parser.add_argument('--config', type=str, 
                       default='/home/ubuntu/llm_addiction/experiment_corrected_sae_analysis/configs/analysis_config.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Add multifeature config if not present
    if 'multifeature' not in config:
        config['multifeature'] = {
            'alpha_values': [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0]
        }
        
    steering = MultiFeatureSteering(config, args.model, args.device)
    steering.run_steering_experiment(
        layer=args.layer,
        top_k=args.top_k,
        n_trials=args.n_trials,
        direction=args.direction
    )


if __name__ == '__main__':
    main()
