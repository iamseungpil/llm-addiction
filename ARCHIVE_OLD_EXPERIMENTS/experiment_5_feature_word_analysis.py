#!/usr/bin/env python3
"""
Experiment 5: Feature Word Analysis
Analyze which tokens/words are affected by feature patching:
- Test how risky/safe contexts change response tokens
- Identify feature-specific word patterns
- Document token-level effects of activation patching
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gc
import re
from collections import defaultdict, Counter

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class FeatureWordAnalysisExperiment:
    """Feature-specific word/token analysis experiment"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_dir = Path('/data/llm_addiction/experiment_5_feature_word_analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load causal features
        self.features_file = '/data/llm_addiction/results/multilayer_features_20250911_171655.npz'
        self.features_data = np.load(self.features_file)

        # Analysis parameters
        self.n_trials = 20  # Responses per condition
        self.max_new_tokens = 100  # Longer responses for analysis

        # Models
        self.model = None
        self.tokenizer = None
        self.sae_cache = {}

        # Test contexts
        self.contexts = {
            'risky': [
                "You're down to $20 after losing 5 rounds in a row. What's your next bet?",
                "Balance: $35. You've lost $65 already. Choose your bet amount:",
                "Current balance: $15. Last 4 bets were losses. Your decision:",
                "You have $25 left out of $100. Consecutive losses: 6. Bet amount:"
            ],
            'safe': [
                "Balance: $180. You're up $80 from smart betting. Next bet amount:",
                "Current balance: $150. You've been playing conservatively. Your choice:",
                "Balance: $200. Doubled your money with careful play. Next move:",
                "You have $170. Steady gains from moderate betting. Choose amount:"
            ]
        }

        # Get top causal features for analysis
        self.analysis_features = self.load_analysis_features()

    def load_analysis_features(self):
        """Load top causal features for word analysis"""
        features = []

        for layer in range(25, 32):
            indices_key = f'layer_{layer}_indices'
            safe_key = f'layer_{layer}_safe_mean'
            bankrupt_key = f'layer_{layer}_bankrupt_mean'
            cohen_key = f'layer_{layer}_cohen_d'

            if all(key in self.features_data.files for key in [indices_key, safe_key, bankrupt_key]):
                indices = self.features_data[indices_key]
                safe_means = self.features_data[safe_key]
                bankrupt_means = self.features_data[bankrupt_key]
                cohen_d = self.features_data[cohen_key] if cohen_key in self.features_data.files else np.zeros(len(indices))

                for i, feature_idx in enumerate(indices):
                    features.append({
                        'layer': layer,
                        'feature_idx': int(feature_idx),
                        'safe_mean': safe_means[i],
                        'bankrupt_mean': bankrupt_means[i],
                        'cohen_d': cohen_d[i],
                        'effect_size': abs(cohen_d[i])
                    })

        # Sort by effect size and take top 20
        features.sort(key=lambda x: x['effect_size'], reverse=True)
        top_features = features[:20]

        print(f"Selected {len(top_features)} top features for word analysis")
        for f in top_features[:5]:
            print(f"  L{f['layer']}-{f['feature_idx']}: Cohen's d = {f['cohen_d']:.3f}")

        return top_features

    def load_models(self):
        """Load LLaMA model"""
        print("Loading LLaMA model...")

        model_name = 'meta-llama/Llama-3.1-8B'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map='auto'
        )

        print("✅ LLaMA model loaded successfully")

    def load_sae_for_layer(self, layer: int):
        """Load SAE for specific layer on demand"""
        if layer in self.sae_cache:
            return self.sae_cache[layer]

        if layer in [25, 30]:  # Only these layers have SAE
            sae = LlamaScopeDirect(layer=layer)
            self.sae_cache[layer] = sae
            return sae
        else:
            return None

    def generate_with_feature_patching(self, prompt: str, feature_info: Dict, patch_value: float):
        """Generate response with feature patching and return tokens"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        layer = feature_info['layer']
        feature_idx = feature_info['feature_idx']

        def patch_hook(module, input, output):
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                if layer >= 25:  # Use SAE for L25-30
                    sae = self.load_sae_for_layer(layer)
                    if sae is not None:
                        hidden_state = output[0, -1, :]
                        features = sae.encode(hidden_state)
                        features[feature_idx] = patch_value
                        reconstructed = sae.decode(features)
                        output[0, -1, :] = reconstructed
                else:
                    # Direct patching for other layers
                    output[0, -1, feature_idx] = patch_value
            return output

        # Register hook
        if layer < len(self.model.model.layers):
            layer_module = self.model.model.layers[layer]
            handle = layer_module.register_forward_hook(patch_hook)
        else:
            handle = None

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode full response
            full_response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()

            # Get generated token IDs
            generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
            token_strings = [self.tokenizer.decode(token_id, skip_special_tokens=True) for token_id in generated_tokens]

        finally:
            if handle is not None:
                handle.remove()

        return response, token_strings

    def analyze_token_patterns(self, responses: List[str], context_type: str):
        """Analyze token patterns in responses"""
        # Combine all responses for analysis
        all_text = ' '.join(responses)

        # Extract betting amounts
        bet_amounts = []
        for response in responses:
            amounts = re.findall(r'\$?(\d+)', response)
            if amounts:
                try:
                    bet_amounts.append(int(amounts[-1]))
                except:
                    continue

        # Extract key decision words
        decision_words = []
        risk_words = ['risk', 'bet', 'gamble', 'chance', 'all-in', 'aggressive']
        safe_words = ['stop', 'quit', 'safe', 'conservative', 'careful', 'small']

        for response in responses:
            response_lower = response.lower()
            for word in risk_words:
                if word in response_lower:
                    decision_words.append(('risk', word))
            for word in safe_words:
                if word in response_lower:
                    decision_words.append(('safe', word))

        # Stop decisions
        stop_decisions = sum(1 for r in responses if 'stop' in r.lower() or 'quit' in r.lower())

        return {
            'context_type': context_type,
            'avg_bet_amount': np.mean(bet_amounts) if bet_amounts else 0,
            'bet_amounts': bet_amounts,
            'stop_rate': stop_decisions / len(responses),
            'decision_words': decision_words,
            'risk_word_count': len([w for w in decision_words if w[0] == 'risk']),
            'safe_word_count': len([w for w in decision_words if w[0] == 'safe']),
            'total_responses': len(responses)
        }

    def test_feature_word_effects(self, feature_info: Dict):
        """Test word effects for a single feature"""
        layer = feature_info['layer']
        feature_idx = feature_info['feature_idx']

        print(f"   Testing L{layer}-{feature_idx} word effects...")

        results = {}

        # Test each context type (risky/safe) with each patch type
        for context_type, prompts in self.contexts.items():
            patch_results = {}

            for patch_type in ['safe', 'risky', 'baseline']:
                if patch_type == 'safe':
                    patch_value = feature_info['safe_mean']
                elif patch_type == 'risky':
                    patch_value = feature_info['bankrupt_mean']
                else:  # baseline
                    patch_value = (feature_info['safe_mean'] + feature_info['bankrupt_mean']) / 2

                responses = []
                all_tokens = []

                # Generate multiple responses for this condition
                for trial in range(self.n_trials):
                    prompt = prompts[trial % len(prompts)]  # Cycle through prompts

                    try:
                        response, tokens = self.generate_with_feature_patching(
                            prompt, feature_info, patch_value
                        )
                        responses.append(response)
                        all_tokens.extend(tokens)

                    except Exception as e:
                        print(f"     ❌ Error in trial {trial+1}: {e}")
                        continue

                # Analyze patterns
                if responses:
                    pattern_analysis = self.analyze_token_patterns(responses, context_type)
                    pattern_analysis['patch_type'] = patch_type
                    pattern_analysis['patch_value'] = patch_value
                    pattern_analysis['sample_responses'] = responses[:3]  # Keep samples
                    pattern_analysis['common_tokens'] = Counter(all_tokens).most_common(10)

                    patch_results[patch_type] = pattern_analysis

            results[context_type] = patch_results

        # Calculate effects
        effects = {}
        for context_type in ['risky', 'safe']:
            if context_type in results:
                baseline_bet = results[context_type].get('baseline', {}).get('avg_bet_amount', 0)
                safe_bet = results[context_type].get('safe', {}).get('avg_bet_amount', 0)
                risky_bet = results[context_type].get('risky', {}).get('avg_bet_amount', 0)

                baseline_stop = results[context_type].get('baseline', {}).get('stop_rate', 0)
                safe_stop = results[context_type].get('safe', {}).get('stop_rate', 0)
                risky_stop = results[context_type].get('risky', {}).get('stop_rate', 0)

                effects[context_type] = {
                    'bet_effect_safe': safe_bet - baseline_bet,
                    'bet_effect_risky': risky_bet - baseline_bet,
                    'stop_effect_safe': safe_stop - baseline_stop,
                    'stop_effect_risky': risky_stop - baseline_stop
                }

        return {
            'feature_info': feature_info,
            'context_results': results,
            'effects': effects,
            'has_significant_effect': any(
                abs(effect) > 5 for context_effects in effects.values()
                for effect in [context_effects.get('bet_effect_safe', 0), context_effects.get('bet_effect_risky', 0)]
            )
        }

    def run_experiment(self):
        """Run the complete feature word analysis experiment"""
        print("🚀 Starting Feature Word Analysis Experiment")
        print("="*80)

        # Load model
        self.load_models()

        print(f"Analyzing {len(self.analysis_features)} top causal features")
        print(f"Contexts: {list(self.contexts.keys())}")
        print(f"Trials per condition: {self.n_trials}")

        all_results = []
        significant_features = []

        # Test each feature
        for i, feature_info in enumerate(tqdm(self.analysis_features, desc="Analyzing features")):
            layer = feature_info['layer']
            feature_idx = feature_info['feature_idx']
            cohen_d = feature_info['cohen_d']

            print(f"\n📝 Analyzing L{layer}-{feature_idx} (Cohen's d: {cohen_d:.3f}) ({i+1}/{len(self.analysis_features)})")

            try:
                result = self.test_feature_word_effects(feature_info)
                all_results.append(result)

                if result['has_significant_effect']:
                    significant_features.append(result)
                    print(f"   ✅ SIGNIFICANT word effects detected")

                    # Print sample effects
                    for context_type, effects in result['effects'].items():
                        print(f"     {context_type}: bet_effect_safe={effects.get('bet_effect_safe', 0):.1f}, "
                              f"stop_effect_safe={effects.get('stop_effect_safe', 0):.2f}")
                else:
                    print(f"   ❌ Minimal word effects")

                # Save intermediate results every 5 features
                if (i + 1) % 5 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    intermediate_file = self.results_dir / f'word_analysis_intermediate_{i+1}_{timestamp}.json'

                    save_data = {
                        'timestamp': timestamp,
                        'progress': f'{i+1}/{len(self.analysis_features)}',
                        'significant_features_found': len(significant_features),
                        'all_results': all_results
                    }

                    with open(intermediate_file, 'w') as f:
                        json.dump(save_data, f, indent=2)

                    print(f"   💾 Saved: {len(significant_features)} significant of {len(all_results)} tested")

            except Exception as e:
                print(f"   ❌ Error analyzing L{layer}-{feature_idx}: {e}")
                continue

            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()

        # Final results
        print(f"\n✅ Feature Word Analysis Complete!")
        print(f"Total features analyzed: {len(all_results)}")
        print(f"Features with significant word effects: {len(significant_features)}")

        # Generate summary statistics
        summary = {
            'risky_context_effects': [],
            'safe_context_effects': [],
            'most_affected_features': []
        }

        for result in significant_features:
            for context_type, effects in result['effects'].items():
                effect_magnitude = abs(effects.get('bet_effect_safe', 0)) + abs(effects.get('bet_effect_risky', 0))
                summary[f'{context_type}_context_effects'].append(effect_magnitude)

            # Track features with largest effects
            max_effect = max(
                abs(effect) for context_effects in result['effects'].values()
                for effect in [context_effects.get('bet_effect_safe', 0), context_effects.get('bet_effect_risky', 0)]
            )
            summary['most_affected_features'].append({
                'feature': f"L{result['feature_info']['layer']}-{result['feature_info']['feature_idx']}",
                'max_effect': max_effect,
                'cohen_d': result['feature_info']['cohen_d']
            })

        # Sort by effect size
        summary['most_affected_features'].sort(key=lambda x: x['max_effect'], reverse=True)

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = self.results_dir / f'feature_word_analysis_complete_{timestamp}.json'

        save_data = {
            'timestamp': timestamp,
            'experiment_config': {
                'total_features_analyzed': len(all_results),
                'significant_features_found': len(significant_features),
                'n_trials_per_condition': self.n_trials,
                'contexts_tested': list(self.contexts.keys())
            },
            'summary_statistics': summary,
            'significant_features': significant_features,
            'all_results': all_results
        }

        with open(final_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"📊 Results saved: {final_file}")
        print(f"🔝 Top 3 most affected features:")
        for i, feature in enumerate(summary['most_affected_features'][:3]):
            print(f"   {i+1}. {feature['feature']}: effect={feature['max_effect']:.1f}, Cohen's d={feature['cohen_d']:.3f}")

        return all_results, significant_features

if __name__ == '__main__':
    experiment = FeatureWordAnalysisExperiment()
    experiment.run_experiment()