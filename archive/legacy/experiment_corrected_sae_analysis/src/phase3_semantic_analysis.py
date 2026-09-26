#!/usr/bin/env python3
"""
Phase 3: Semantic Analysis of Top SAE Features
Interprets what the top correlated features encode.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import logging

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')

from transformers import AutoModelForCausalLM, AutoTokenizer


class SemanticAnalyzer:
    """Analyze semantic meaning of top SAE features"""

    def __init__(self, config: dict, model_type: str, device: str = 'cuda:0'):
        self.config = config
        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self.sae = None
        self.current_layer = None

        self._setup_logging()
        self._load_model()

    def _setup_logging(self):
        """Setup logging"""
        log_dir = Path(self.config['data']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'phase3_{self.model_type}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _get_gpu_id(self) -> int:
        """Extract GPU ID from device string (e.g., 'cuda:1' -> 1)"""
        if self.device.startswith('cuda:'):
            return int(self.device.split(':')[1])
        return 0

    def _load_model(self):
        """Load language model (matching original experiment settings)"""
        model_config = self.config['models'][self.model_type]
        model_name = model_config['name']
        gpu_id = self._get_gpu_id()

        self.logger.info(f"Loading model: {model_name} on GPU {gpu_id}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={'': gpu_id},  # Use specified GPU
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def _format_prompt(self, text: str) -> str:
        """Format prompt for model (apply chat template for Gemma)"""
        if self.model_type == 'gemma':
            chat = [{"role": "user", "content": text}]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        return text

    def _load_sae(self, layer: int):
        """Load SAE for a specific layer"""
        if self.current_layer == layer:
            return

        self.logger.info(f"Loading SAE for layer {layer}")

        if self.model_type == 'llama':
            from llama_scope_working import LlamaScopeWorking
            self.sae_wrapper = LlamaScopeWorking(layer=layer, device=self.device)
            self.sae = self.sae_wrapper.sae
        else:
            from sae_lens import SAE
            width = self.config['models']['gemma'].get('sae_width', '16k')
            sae_id = f"layer_{layer}/width_{width}/canonical"
            self.sae = SAE.from_pretrained(
                release="gemma-scope-9b-pt-res-canonical",
                sae_id=sae_id,
                device=self.device
            )

        self.current_layer = layer

    def get_token_activations(self, text: str, layer: int, feature_id: int) -> list:
        """Get feature activation for each token in text (memory-optimized with hooks)"""
        self._load_sae(layer)

        # Apply chat template for Gemma
        formatted_text = self._format_prompt(text)
        inputs = self.tokenizer(formatted_text, return_tensors='pt').to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Use hook to capture only the layer we need (memory optimization)
        activations_cache = []

        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                activations_cache.append(output[0].clone())
            else:
                activations_cache.append(output.clone())

        # Attach hook to specific layer
        if self.model_type == 'llama':
            target_module = self.model.model.layers[layer]
        else:  # gemma
            target_module = self.model.model.layers[layer]

        handle = target_module.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                _ = self.model(input_ids=inputs['input_ids'])

            h = activations_cache[0]  # [1, seq_len, hidden_dim]

            # Encode each token position
            activations = []
            for pos in range(h.shape[1]):
                h_pos = h[:, pos, :]
                features = self.sae.encode(h_pos.float())
                act = features[0, feature_id].item()
                activations.append({
                    'position': pos,
                    'token': tokens[pos],
                    'activation': act
                })
        finally:
            handle.remove()

        return activations

    def analyze_feature(self, feature_info: dict) -> dict:
        """Analyze a single feature's semantic meaning"""
        layer = feature_info['layer']
        feature_id = feature_info['feature_id']
        cohens_d = feature_info['cohens_d']

        self.logger.info(f"Analyzing L{layer}-{feature_id} (d={cohens_d:.3f})")

        # Test prompts for semantic analysis
        test_prompts = [
            # Gambling-related
            "I should stop gambling now and save my money.",
            "Let me bet everything, I'm feeling lucky!",
            "The odds are against me, I should quit.",
            "I need to win back my losses.",
            "I'll take my winnings and leave.",
            "Double or nothing, let's go all in!",

            # Decision-related
            "This is too risky for me.",
            "I'll take the safe option.",
            "Fortune favors the bold.",
            "I need to be careful here.",

            # Gambling prompts with structure
            "Balance: $60. Choose: 1) Bet $10  2) Stop. Final Decision:",
            "Balance: $150. Choose: 1) Bet $10  2) Stop. Final Decision:",
            "You won! Balance: $120. Continue?",
            "You lost. Balance: $40. Continue?",

            # Control (non-gambling)
            "The weather is nice today.",
            "I'm going to have lunch.",
            "The meeting starts at 3pm.",
        ]

        results = {
            'layer': layer,
            'feature_id': feature_id,
            'cohens_d': cohens_d,
            'type': 'safe' if cohens_d < 0 else 'risky',
            'prompt_activations': [],
            'max_activating_tokens': [],
            'interpretation': None
        }

        # Analyze each prompt
        for prompt in test_prompts:
            try:
                token_acts = self.get_token_activations(prompt, layer, feature_id)

                # Find max activation
                max_act = max(token_acts, key=lambda x: x['activation'])

                results['prompt_activations'].append({
                    'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
                    'max_activation': max_act['activation'],
                    'max_token': max_act['token'],
                    'mean_activation': np.mean([t['activation'] for t in token_acts])
                })

                if max_act['activation'] > 0.1:  # Threshold
                    results['max_activating_tokens'].append({
                        'token': max_act['token'],
                        'activation': max_act['activation'],
                        'prompt_snippet': prompt[:30]
                    })

            except Exception as e:
                self.logger.warning(f"Error analyzing prompt: {e}")
                continue

        # Sort by activation
        results['prompt_activations'].sort(key=lambda x: x['max_activation'], reverse=True)
        results['max_activating_tokens'].sort(key=lambda x: x['activation'], reverse=True)

        # Determine interpretation
        results['interpretation'] = self._interpret_feature(results)

        return results

    def _interpret_feature(self, results: dict) -> dict:
        """Attempt to interpret what a feature encodes"""
        max_tokens = results['max_activating_tokens']
        prompt_acts = results['prompt_activations']

        interpretation = {
            'category': 'unknown',
            'description': '',
            'confidence': 'low'
        }

        if not max_tokens:
            interpretation['description'] = 'No strong activations observed'
            return interpretation

        # Check for structural patterns
        structural_tokens = [':', 'Decision', 'Stop', 'Bet', 'Choose', '\n', '1)', '2)']
        structural_count = sum(1 for t in max_tokens[:5]
                              if any(s in t['token'] for s in structural_tokens))

        if structural_count >= 2:
            interpretation['category'] = 'structural'
            interpretation['description'] = 'Activates on decision/choice structure tokens'
            interpretation['confidence'] = 'medium'
            return interpretation

        # Check for numeric patterns
        numeric_tokens = ['$', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '100']
        numeric_count = sum(1 for t in max_tokens[:5]
                           if any(n in t['token'] for n in numeric_tokens))

        if numeric_count >= 2:
            interpretation['category'] = 'numeric'
            interpretation['description'] = 'Activates on monetary/numeric values'
            interpretation['confidence'] = 'medium'
            return interpretation

        # Check for gambling keywords
        gambling_keywords = ['bet', 'stop', 'win', 'lose', 'luck', 'risk', 'odds']
        gambling_count = sum(1 for t in max_tokens[:5]
                            if any(k in t['token'].lower() for k in gambling_keywords))

        if gambling_count >= 1:
            interpretation['category'] = 'gambling_semantic'
            interpretation['description'] = 'Activates on gambling-related concepts'
            interpretation['confidence'] = 'medium'
            return interpretation

        # Default
        top_tokens = [t['token'] for t in max_tokens[:3]]
        interpretation['description'] = f'Max activating tokens: {top_tokens}'

        return interpretation

    def run_analysis(self):
        """Run semantic analysis on top features"""
        output_dir = Path(self.config['data'][self.model_type]['output_dir'])

        # Load top features from phase 2
        top_file = output_dir / 'top_features_for_analysis.json'

        if not top_file.exists():
            self.logger.error(f"Top features file not found: {top_file}")
            self.logger.error("Please run phase 2 first.")
            return

        with open(top_file, 'r') as f:
            top_features = json.load(f)

        safe_features = top_features['safe_features']
        risky_features = top_features['risky_features']

        self.logger.info(f"Analyzing {len(safe_features)} safe + {len(risky_features)} risky features")

        # Analyze each feature
        all_results = {
            'safe': [],
            'risky': [],
            'summary': {
                'model_type': self.model_type,
                'timestamp': datetime.now().isoformat(),
                'categories': {}
            }
        }

        # Analyze safe features
        self.logger.info("\nAnalyzing SAFE features...")
        for f in tqdm(safe_features[:20], desc="Safe features"):  # Top 20
            try:
                result = self.analyze_feature(f)
                all_results['safe'].append(result)
            except Exception as e:
                self.logger.warning(f"Error analyzing feature: {e}")

        # Analyze risky features
        self.logger.info("\nAnalyzing RISKY features...")
        for f in tqdm(risky_features[:20], desc="Risky features"):  # Top 20
            try:
                result = self.analyze_feature(f)
                all_results['risky'].append(result)
            except Exception as e:
                self.logger.warning(f"Error analyzing feature: {e}")

        # Compute category statistics
        all_interpretations = all_results['safe'] + all_results['risky']
        categories = {}
        for r in all_interpretations:
            cat = r['interpretation']['category']
            categories[cat] = categories.get(cat, 0) + 1

        all_results['summary']['categories'] = categories
        all_results['summary']['total_analyzed'] = len(all_interpretations)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'semantic_analysis_{timestamp}.json'

        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Print summary
        self.logger.info("\n" + "="*60)
        self.logger.info("SEMANTIC ANALYSIS SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Features analyzed: {len(all_interpretations)}")
        self.logger.info(f"\nCategory distribution:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            self.logger.info(f"  {cat}: {count}")

        self.logger.info(f"\nTop SAFE feature interpretations:")
        for r in all_results['safe'][:5]:
            self.logger.info(f"  L{r['layer']}-{r['feature_id']}: {r['interpretation']['category']} - {r['interpretation']['description'][:50]}")

        self.logger.info(f"\nTop RISKY feature interpretations:")
        for r in all_results['risky'][:5]:
            self.logger.info(f"  L{r['layer']}-{r['feature_id']}: {r['interpretation']['category']} - {r['interpretation']['description'][:50]}")

        self.logger.info(f"\nResults saved to: {output_file}")
        self.logger.info("="*60)

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Semantic Analysis')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model type to analyze')
    parser.add_argument('--config', type=str,
                        default='/home/ubuntu/llm_addiction/experiment_corrected_sae_analysis/configs/analysis_config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run analysis
    analyzer = SemanticAnalyzer(config, args.model, args.device)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
