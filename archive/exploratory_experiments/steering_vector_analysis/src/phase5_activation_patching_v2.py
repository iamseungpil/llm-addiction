#!/usr/bin/env python3
"""
Phase 5: Activation Patching (Improved Version)

Key improvements over v1:
1. Baseline caching - compute once, reuse across layers/heads
2. Statistical testing - Fisher's exact test with p-values
3. Response saving - raw responses stored for audit
4. Random seed - reproducible results
5. Better response parsing - robust stop/bet detection
6. Efficiency - no redundant baseline computation

Usage:
    python phase5_activation_patching_v2.py --model llama --gpu 0 --mode layer
    python phase5_activation_patching_v2.py --model llama --gpu 0 --mode head --layer 23
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import yaml
import re
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class PatchingConfig:
    """Configuration for patching experiments."""
    n_trials: int = 20
    random_seed: int = 42
    p_value_threshold: float = 0.05
    min_effect_size: float = 0.15
    temperature: float = 0.7
    max_new_tokens: int = 100


class Phase5ActivationPatchingV2:
    """
    Phase 5: Activation Patching Analysis (Improved)

    Key improvements:
    - Baseline computed once and cached
    - Proper statistical testing with Fisher's exact test
    - Responses saved for audit
    - Reproducible with random seed
    """

    MODEL_CONFIGS = {
        'llama': {
            'model_id': 'meta-llama/Llama-3.1-8B',
            'n_layers': 32,
            'n_heads': 32,
            'head_dim': 128,
            'd_model': 4096,
        },
        'gemma_base': {
            'model_id': 'google/gemma-2-9b',
            'n_layers': 42,
            'n_heads': 16,
            'head_dim': 256,
            'd_model': 3584,
        },
        'gemma': {
            'model_id': 'google/gemma-2-9b',
            'n_layers': 42,
            'n_heads': 16,
            'head_dim': 256,
            'd_model': 3584,
        }
    }

    def __init__(
        self,
        model_name: str = 'llama',
        gpu_id: int = 0,
        output_dir: str = None,
        config: PatchingConfig = None,
        config_path: str = None,
        prompts_path: str = None
    ):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        self.config = config or PatchingConfig()

        # Set random seeds for reproducibility
        self._set_random_seeds(self.config.random_seed)

        # Load experiment config if provided
        self.experiment_config = {}
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "experiment_config_direct_steering.yaml"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.experiment_config = yaml.safe_load(f)

        # Model config
        model_cfg = self.experiment_config.get('models', {}).get(model_name)
        if model_cfg is None:
            model_cfg = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS['llama'])
        self.model_id = model_cfg['model_id']
        self.n_layers = model_cfg['n_layers']
        self.n_heads = model_cfg['n_heads']
        self.d_model = model_cfg['d_model']
        self.head_dim = self.d_model // self.n_heads
        self.use_chat_template = model_cfg.get('use_chat_template', False)

        # Validate head dimensions (WARNING 3 fix)
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

        # Output directory
        if output_dir is None:
            output_dir = '/data/llm_addiction/steering_vector_experiment_direct/phase5_results_v2'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_path = prompts_path or self.experiment_config.get('prompts_path')

        # Model and tokenizer
        self.model = None
        self.tokenizer = None

        # Cached baseline data
        self._baseline_cache = None
        self._baseline_responses = []

    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_model(self):
        """Load model and tokenizer."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_id} on GPU {self.gpu_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map={'': self.gpu_id}
        )
        self.model.eval()
        print(f"Model loaded: {self.model_id}")

    def format_prompt(self, prompt: str) -> str:
        """Apply chat template when configured."""
        if self.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            chat = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return prompt

    def load_prompts(self, prompts_path: str = None) -> Dict[str, List[Dict]]:
        """Load prompts organized by condition."""
        if prompts_path is None:
            prompts_path = self.prompts_path or '/data/llm_addiction/steering_vector_experiment_full/condition_prompts.json'

        with open(prompts_path) as f:
            data = json.load(f)

        conditions = data.get('conditions', {}).get(self.model_name, [])
        if not conditions and self.model_name == 'gemma_base':
            conditions = data.get('conditions', {}).get('gemma', [])

        prompts_by_condition = {
            'safe_fixed': [],
            'safe_variable': [],
            'risky_fixed': [],
            'risky_variable': []
        }

        for item in conditions:
            cond = item.get('condition_name', '')
            if cond in prompts_by_condition:
                prompts_by_condition[cond].append(item)

        print(f"Loaded prompts by condition:")
        for cond, prompts in prompts_by_condition.items():
            print(f"  {cond}: {len(prompts)}")

        return prompts_by_condition

    def parse_response(self, response: str) -> Tuple[str, Optional[int]]:
        """
        Parse response to determine action and bet amount.

        Returns:
            (action, bet_amount) where action is 'stop' or 'bet'
        """
        response_lower = response.lower().strip()

        # Stop indicators (prioritize explicit stop)
        stop_patterns = [
            r'\bstop\b', r'\bquit\b', r'\bcash\s*out\b',
            r'\bleave\b', r'\bwalk\s*away\b', r'\bend\b',
            r"i'?ll stop", r"i'?m stopping", r"i'?m done",
            r"no more", r"that'?s enough"
        ]

        for pattern in stop_patterns:
            if re.search(pattern, response_lower):
                return 'stop', None

        # Bet indicators
        bet_match = re.search(r'\$\s*(\d+)', response)
        if bet_match:
            return 'bet', int(bet_match.group(1))

        bet_patterns = [
            r'\bbet\b', r'\bwager\b', r'\bgamble\b',
            r'\bcontinue\b', r'\bplay\b', r'\bkeep going\b'
        ]

        for pattern in bet_patterns:
            if re.search(pattern, response_lower):
                return 'bet', None

        # Default to bet if unclear (conservative for detecting stops)
        return 'bet', None

    def generate_response(self, prompt: str) -> str:
        """Generate response for a prompt."""
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response

    def get_activations(self, prompt: str, layers: List[int]) -> Dict[int, torch.Tensor]:
        """Get hidden state activations for a prompt at specified layers."""
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)
        activations = {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                activations[layer_idx] = hidden[:, -1, :].detach().clone()
            return hook

        for layer_idx in layers:
            layer_module = self.model.model.layers[layer_idx]
            handle = layer_module.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        return activations

    def generate_with_patch(
        self,
        prompt: str,
        patch_layer: int,
        patch_activation: torch.Tensor,
        head_id: Optional[int] = None
    ) -> str:
        """Generate response with patched activation."""
        formatted = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)

        def create_patch_hook(activation, head_id=None):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    if head_id is not None:
                        start_idx = head_id * self.head_dim
                        end_idx = start_idx + self.head_dim
                        hidden[:, -1, start_idx:end_idx] = activation[start_idx:end_idx]
                    else:
                        hidden[:, -1, :] = activation
                    return (hidden,) + output[1:]
                else:
                    if head_id is not None:
                        start_idx = head_id * self.head_dim
                        end_idx = start_idx + self.head_dim
                        output[:, -1, start_idx:end_idx] = activation[start_idx:end_idx]
                    else:
                        output[:, -1, :] = activation
                    return output
            return hook

        layer_module = self.model.model.layers[patch_layer]
        handle = layer_module.register_forward_hook(
            create_patch_hook(patch_activation, head_id)
        )

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        finally:
            handle.remove()

        return response

    def compute_baseline(
        self,
        risky_prompts: List[Dict],
        n_trials: int
    ) -> Tuple[float, List[Dict]]:
        """
        Compute baseline stop rate ONCE for risky prompts.

        Returns:
            (stop_rate, detailed_results)
        """
        if self._baseline_cache is not None:
            print("Using cached baseline")
            return self._baseline_cache, self._baseline_responses

        print("Computing baseline (will be cached)...")
        results = []

        max_trials = min(n_trials, len(risky_prompts))
        for i, prompt_data in enumerate(tqdm(risky_prompts[:max_trials], desc="Baseline")):
            prompt = prompt_data.get('prompt', '')
            response = self.generate_response(prompt)
            action, bet_amount = self.parse_response(response)

            results.append({
                'prompt_id': i,
                'prompt': prompt[:100] + '...',
                'response': response,
                'action': action,
                'bet_amount': bet_amount,
                'is_stop': action == 'stop'
            })

        stop_rate = np.mean([r['is_stop'] for r in results])

        # Cache the results
        self._baseline_cache = stop_rate
        self._baseline_responses = results

        print(f"Baseline stop rate: {stop_rate:.3f} (n={len(results)})")
        return stop_rate, results

    def compute_statistics(
        self,
        baseline_stops: int,
        baseline_total: int,
        patched_stops: int,
        patched_total: int
    ) -> Dict:
        """
        Compute statistical significance using Fisher's exact test.

        Returns:
            Dict with effect_size, p_value, is_significant
        """
        # Effect size (difference in proportions)
        baseline_rate = baseline_stops / baseline_total if baseline_total > 0 else 0
        patched_rate = patched_stops / patched_total if patched_total > 0 else 0
        effect_size = patched_rate - baseline_rate

        # Handle edge cases (WARNING 2 fix)
        if baseline_stops == 0 and patched_stops == 0:
            return {
                'effect_size': 0.0,
                'p_value': 1.0,
                'odds_ratio': None,
                'is_significant': False,
                'baseline_rate': 0.0,
                'patched_rate': 0.0,
                'warning': 'No stops in either condition'
            }

        if baseline_stops == baseline_total and patched_stops == patched_total:
            return {
                'effect_size': 0.0,
                'p_value': 1.0,
                'odds_ratio': None,
                'is_significant': False,
                'baseline_rate': 1.0,
                'patched_rate': 1.0,
                'warning': 'All stops in both conditions'
            }

        # 2x2 contingency table
        # [[baseline_stop, baseline_continue], [patched_stop, patched_continue]]
        table = [
            [baseline_stops, baseline_total - baseline_stops],
            [patched_stops, patched_total - patched_stops]
        ]

        # Fisher's exact test
        odds_ratio, p_value = fisher_exact(table)

        # Significance check
        is_significant = (
            p_value < self.config.p_value_threshold and
            abs(effect_size) >= self.config.min_effect_size
        )

        return {
            'effect_size': float(effect_size),
            'p_value': float(p_value),
            'odds_ratio': float(odds_ratio) if not np.isinf(odds_ratio) else None,
            'is_significant': bool(is_significant),
            'baseline_rate': float(baseline_rate),
            'patched_rate': float(patched_rate)
        }

    def run_layer_patching(
        self,
        prompts_by_condition: Dict[str, List[Dict]],
        n_trials: int = None
    ) -> Dict:
        """Run layer-level activation patching with cached baseline."""
        if n_trials is None:
            n_trials = self.config.n_trials

        print(f"\n{'='*70}")
        print("LAYER-LEVEL ACTIVATION PATCHING (Improved)")
        print(f"{'='*70}")

        # Prepare prompts (balanced sampling across fixed/variable)
        per_condition = min(
            n_trials,
            len(prompts_by_condition['safe_fixed']),
            len(prompts_by_condition['safe_variable']),
            len(prompts_by_condition['risky_fixed']),
            len(prompts_by_condition['risky_variable'])
        )
        safe_prompts = (
            prompts_by_condition['safe_fixed'][:per_condition] +
            prompts_by_condition['safe_variable'][:per_condition]
        )
        risky_prompts = (
            prompts_by_condition['risky_fixed'][:per_condition] +
            prompts_by_condition['risky_variable'][:per_condition]
        )
        total_trials = min(len(safe_prompts), len(risky_prompts))
        if per_condition < n_trials:
            print(f"Warning: using {per_condition} prompts per condition (requested {n_trials})")

        # Compute baseline ONCE
        baseline_rate, baseline_results = self.compute_baseline(risky_prompts, total_trials)
        baseline_stops = sum(1 for r in baseline_results if r['is_stop'])
        baseline_total = len(baseline_results)

        layer_results = []
        all_responses = {'baseline': baseline_results, 'layers': {}}

        for layer in tqdm(range(self.n_layers), desc="Patching layers"):
            patched_results = []

            for i in range(total_trials):
                risky_data = risky_prompts[i]
                safe_data = safe_prompts[i % len(safe_prompts)]

                # Get safe activation
                safe_act = self.get_activations(safe_data.get('prompt', ''), [layer])
                safe_activation = safe_act[layer].squeeze()

                # Generate with patch
                response = self.generate_with_patch(
                    risky_data.get('prompt', ''),
                    layer,
                    safe_activation
                )
                action, bet_amount = self.parse_response(response)

                patched_results.append({
                    'prompt_id': i,
                    'response': response,
                    'action': action,
                    'is_stop': action == 'stop'
                })

            patched_stops = sum(1 for r in patched_results if r['is_stop'])
            patched_total = len(patched_results)

            # Compute statistics
            stats = self.compute_statistics(
                baseline_stops, baseline_total,
                patched_stops, patched_total
            )

            result = {
                'layer': layer,
                'baseline_stop_rate': stats['baseline_rate'],
                'patched_stop_rate': stats['patched_rate'],
                'effect': stats['effect_size'],
                'p_value': stats['p_value'],
                'is_significant': stats['is_significant']
            }
            layer_results.append(result)
            all_responses['layers'][layer] = patched_results

            if stats['is_significant'] or abs(stats['effect_size']) > 0.1:
                print(f"  Layer {layer}: effect={stats['effect_size']:+.3f}, p={stats['p_value']:.4f} {'*' if stats['is_significant'] else ''}")

            # Cleanup
            del safe_activation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        significant_layers = [r['layer'] for r in layer_results if r['is_significant']]

        return {
            'model': self.model_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': asdict(self.config),
            'n_layers': self.n_layers,
            'n_trials_per_condition': per_condition,
            'n_trials_total': total_trials,
            'patching_direction': 'safe_to_risky',
            'layer_results': layer_results,
            'significant_layers': significant_layers,
            'responses': all_responses
        }

    def run_head_patching(
        self,
        layer: int,
        prompts_by_condition: Dict[str, List[Dict]],
        n_trials: int = None
    ) -> Dict:
        """Run head-level patching for a specific layer with cached baseline."""
        if n_trials is None:
            n_trials = self.config.n_trials

        print(f"\n{'='*70}")
        print(f"HEAD-LEVEL ACTIVATION PATCHING - Layer {layer}")
        print(f"{'='*70}")

        # Prepare prompts (balanced sampling across fixed/variable)
        per_condition = min(
            n_trials,
            len(prompts_by_condition['safe_fixed']),
            len(prompts_by_condition['safe_variable']),
            len(prompts_by_condition['risky_fixed']),
            len(prompts_by_condition['risky_variable'])
        )
        safe_prompts = (
            prompts_by_condition['safe_fixed'][:per_condition] +
            prompts_by_condition['safe_variable'][:per_condition]
        )
        risky_prompts = (
            prompts_by_condition['risky_fixed'][:per_condition] +
            prompts_by_condition['risky_variable'][:per_condition]
        )
        total_trials = min(len(safe_prompts), len(risky_prompts))
        if per_condition < n_trials:
            print(f"Warning: using {per_condition} prompts per condition (requested {n_trials})")

        # Compute baseline ONCE (or use cache)
        baseline_rate, baseline_results = self.compute_baseline(risky_prompts, total_trials)
        baseline_stops = sum(1 for r in baseline_results if r['is_stop'])
        baseline_total = len(baseline_results)

        head_results = []
        all_responses = {'baseline': baseline_results, 'heads': {}}

        for head_id in tqdm(range(self.n_heads), desc=f"Patching heads (L{layer})"):
            patched_results = []

            for i in range(total_trials):
                risky_data = risky_prompts[i]
                safe_data = safe_prompts[i % len(safe_prompts)]

                # Get safe activation
                safe_act = self.get_activations(safe_data.get('prompt', ''), [layer])
                safe_activation = safe_act[layer].squeeze()

                # Generate with head patch
                response = self.generate_with_patch(
                    risky_data.get('prompt', ''),
                    layer,
                    safe_activation,
                    head_id=head_id
                )
                action, bet_amount = self.parse_response(response)

                patched_results.append({
                    'prompt_id': i,
                    'response': response,
                    'action': action,
                    'is_stop': action == 'stop'
                })

            patched_stops = sum(1 for r in patched_results if r['is_stop'])
            patched_total = len(patched_results)

            # Compute statistics
            stats = self.compute_statistics(
                baseline_stops, baseline_total,
                patched_stops, patched_total
            )

            result = {
                'head_id': head_id,
                'baseline_stop_rate': stats['baseline_rate'],
                'patched_stop_rate': stats['patched_rate'],
                'effect': stats['effect_size'],
                'p_value': stats['p_value'],
                'is_significant': stats['is_significant']
            }
            head_results.append(result)
            all_responses['heads'][head_id] = patched_results

            if stats['is_significant'] or abs(stats['effect_size']) > 0.1:
                print(f"  Head {head_id}: effect={stats['effect_size']:+.3f}, p={stats['p_value']:.4f} {'*' if stats['is_significant'] else ''}")

            # Cleanup every 5 heads
            if (head_id + 1) % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Sort by effect magnitude
        head_results.sort(key=lambda x: abs(x['effect']), reverse=True)
        significant_heads = [h['head_id'] for h in head_results if h['is_significant']]

        return {
            'model': self.model_name,
            'layer': layer,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': asdict(self.config),
            'n_heads': self.n_heads,
            'n_trials_per_condition': per_condition,
            'n_trials_total': total_trials,
            'head_results': head_results,
            'significant_heads': significant_heads,
            'n_significant': len(significant_heads),
            'responses': all_responses
        }

    def save_results(self, results: Dict, prefix: str) -> Path:
        """Save results to JSON (without responses for main file, separate for responses)."""
        timestamp = results.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        layer = results.get('layer', 'all')

        # Main results file (without full responses)
        results_slim = {k: v for k, v in results.items() if k != 'responses'}
        filename = f"{prefix}_{self.model_name}_L{layer}_{timestamp}.json"
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results_slim, f, indent=2, cls=NumpyEncoder)
        print(f"Results saved: {output_path}")

        # Responses file (for audit)
        if 'responses' in results:
            responses_filename = f"{prefix}_{self.model_name}_L{layer}_{timestamp}_responses.json"
            responses_path = self.output_dir / responses_filename
            with open(responses_path, 'w') as f:
                json.dump(results['responses'], f, indent=2, cls=NumpyEncoder)
            print(f"Responses saved: {responses_path}")

        return output_path

    def run(
        self,
        mode: str = 'layer',
        layers: List[int] = None,
        n_trials: int = None
    ):
        """
        Run Phase 5 analysis.

        Args:
            mode: 'layer', 'head', or 'all'
            layers: Target layers for head-level patching (defaults to causal layers)
            n_trials: Number of trials per condition
        """
        print("=" * 70)
        print("PHASE 5: ACTIVATION PATCHING (v2 - Improved)")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Mode: {mode}")
        print(f"Random seed: {self.config.random_seed}")
        print(f"P-value threshold: {self.config.p_value_threshold}")
        print(f"Min effect size: {self.config.min_effect_size}")

        self.load_model()
        prompts = self.load_prompts()

        if mode == 'layer' or mode == 'all':
            # Reset baseline cache for layer patching
            self._baseline_cache = None
            self._baseline_responses = []

            layer_results = self.run_layer_patching(prompts, n_trials)
            self.save_results(layer_results, 'layer_patching')

            print(f"\nSignificant layers: {layer_results['significant_layers']}")

            # Use significant layers for head patching if not specified
            if layers is None:
                layers = layer_results['significant_layers']

        if mode == 'head' or mode == 'all':
            if layers is None:
                layers = [23]  # Default to layer 23 (known causal for LLaMA)

            for layer in layers:
                # Reset baseline cache for each layer
                self._baseline_cache = None
                self._baseline_responses = []

                head_results = self.run_head_patching(layer, prompts, n_trials)
                self.save_results(head_results, 'head_patching')

                print(f"\nSignificant heads (L{layer}): {head_results['significant_heads']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Phase 5: Activation Patching (v2)')
    parser.add_argument('--model', type=str, default='llama',
                       choices=['llama', 'gemma', 'gemma_base'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='layer',
                       choices=['layer', 'head', 'all'])
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Target layers for head-level patching')
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--p-threshold', type=float, default=0.05)
    parser.add_argument('--effect-threshold', type=float, default=0.15)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to experiment config (direct steering)')
    parser.add_argument('--prompts', type=str, default=None,
                       help='Path to condition_prompts.json')

    args = parser.parse_args()

    config = PatchingConfig(
        n_trials=args.n_trials,
        random_seed=args.seed,
        p_value_threshold=args.p_threshold,
        min_effect_size=args.effect_threshold
    )

    phase5 = Phase5ActivationPatchingV2(
        model_name=args.model,
        gpu_id=args.gpu,
        config=config,
        config_path=args.config,
        prompts_path=args.prompts
    )

    phase5.run(
        mode=args.mode,
        layers=args.layers,
        n_trials=args.n_trials
    )


if __name__ == '__main__':
    main()
