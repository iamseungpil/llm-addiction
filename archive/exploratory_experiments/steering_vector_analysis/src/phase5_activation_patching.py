#!/usr/bin/env python3
"""
Phase 5: Activation Patching

Causal intervention through activation replacement (not addition).
This is more direct than steering - we replace activations from one condition
with activations from another condition and measure behavioral change.

Key Difference from Steering:
- Steering: hidden = hidden + alpha * steering_vector (additive)
- Patching: hidden = hidden_from_other_condition (substitution)

Analysis Components:
1. Layer-Level Patching
   - Replace entire layer activations between conditions
   - Compare safe→risky and risky→safe patching

2. Head-Level Patching
   - Replace specific head activations
   - Find which heads are causally important

3. Cross-Condition Patching
   - Patch activations between different experimental conditions
   - Measure behavioral flip rate

Usage:
    # Layer-level patching
    python phase5_activation_patching.py --model llama --gpu 0 --mode layer

    # Head-level patching for specific layer
    python phase5_activation_patching.py --model llama --gpu 0 --mode head --layer 23

    # All analyses
    python phase5_activation_patching.py --model llama --gpu 0 --mode all
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
class PatchingResult:
    """Result of a single patching experiment."""
    layer: int
    head_id: Optional[int]  # None for layer-level patching
    source_condition: str  # e.g., "safe"
    target_condition: str  # e.g., "risky"
    baseline_stop_rate: float
    patched_stop_rate: float
    effect_size: float
    n_trials: int
    is_significant: bool


@dataclass
class LayerPatchingResults:
    """Results for layer-level patching across all layers."""
    model: str
    timestamp: str
    n_layers: int
    patching_direction: str  # "safe_to_risky" or "risky_to_safe"
    layer_results: List[Dict]
    significant_layers: List[int]


class Phase5ActivationPatching:
    """
    Phase 5: Activation Patching Analysis

    Replace activations between conditions to test causal effects.
    """

    # Model architectures
    MODEL_CONFIGS = {
        'llama': {
            'model_id': 'meta-llama/Llama-3.1-8B',
            'n_layers': 32,
            'n_heads': 32,
            'head_dim': 128,
            'd_model': 4096,
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
        config_path: str = None,
        prompts_path: str = None
    ):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

        # Load experiment config if provided
        self.config = {}
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "experiment_config_direct_steering.yaml"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        # Model config (prefer experiment config, fallback to defaults)
        model_cfg = self.config.get('models', {}).get(model_name)
        if model_cfg is None:
            model_cfg = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS['llama'])

        self.model_id = model_cfg['model_id']
        self.n_layers = model_cfg.get('n_layers', self.MODEL_CONFIGS['llama']['n_layers'])
        self.n_heads = model_cfg.get('n_heads', self.MODEL_CONFIGS['llama']['n_heads'])
        self.d_model = model_cfg.get('d_model', self.MODEL_CONFIGS['llama']['d_model'])
        self.head_dim = self.d_model // self.n_heads
        self.use_chat_template = model_cfg.get('use_chat_template', False)

        # Output directory
        if output_dir is None:
            output_dir = '/data/llm_addiction/steering_vector_experiment_direct/phase5_results'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_path = prompts_path or self.config.get('prompts_path')

        # Model and tokenizer (loaded on demand)
        self.model = None
        self.tokenizer = None

        # Cached activations
        self.cached_activations = {}

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

    def load_prompts(self, prompts_path: str = None) -> Dict[str, List[Dict]]:
        """
        Load prompts organized by condition.

        Returns:
            Dict with keys like 'safe_fixed', 'risky_variable', etc.
        """
        if prompts_path is None:
            prompts_path = self.prompts_path or '/data/llm_addiction/steering_vector_experiment_full/condition_prompts.json'

        with open(prompts_path) as f:
            data = json.load(f)

        # Organize by condition
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

    def get_activations(
        self,
        prompt: str,
        layers: List[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Get hidden state activations for a prompt at specified layers.

        Returns:
            Dict mapping layer -> activation tensor (last token position)
        """
        if layers is None:
            layers = list(range(self.n_layers))

        # Format prompt
        if self.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            chat = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)

        activations = {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Store last token activation
                activations[layer_idx] = hidden[:, -1, :].detach().clone()
            return hook

        # Register hooks
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

    def generate_with_patched_activation(
        self,
        prompt: str,
        patch_layer: int,
        patch_activation: torch.Tensor,
        head_id: Optional[int] = None
    ) -> str:
        """
        Generate response with patched activation at specified layer.

        Args:
            prompt: Input prompt
            patch_layer: Layer to patch
            patch_activation: Activation to use for patching
            head_id: If specified, only patch this head's dimensions

        Returns:
            Generated response
        """
        # Format prompt
        if self.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            chat = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)

        def create_patch_hook(activation, head_id=None):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    if head_id is not None:
                        # Only patch specific head dimensions
                        start_idx = head_id * self.head_dim
                        end_idx = start_idx + self.head_dim
                        hidden[:, -1, start_idx:end_idx] = activation[start_idx:end_idx]
                    else:
                        # Patch entire layer
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
        hook = create_patch_hook(patch_activation, head_id)
        handle = layer_module.register_forward_hook(hook)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        finally:
            handle.remove()

        return response

    def run_layer_patching(
        self,
        prompts_by_condition: Dict[str, List[Dict]],
        n_trials: int = 20
    ) -> LayerPatchingResults:
        """
        Run layer-level activation patching.

        For each layer:
        1. Get activation from safe prompt
        2. Patch into risky prompt generation
        3. Measure if behavior changes
        """
        print(f"\n{'='*70}")
        print("LAYER-LEVEL ACTIVATION PATCHING")
        print(f"{'='*70}")

        # Balanced sampling across conditions
        safe_fixed = prompts_by_condition['safe_fixed'][:n_trials]
        safe_variable = prompts_by_condition['safe_variable'][:n_trials]
        risky_fixed = prompts_by_condition['risky_fixed'][:n_trials]
        risky_variable = prompts_by_condition['risky_variable'][:n_trials]

        safe_prompts = safe_fixed + safe_variable
        risky_prompts = risky_fixed + risky_variable

        print(f"Testing {len(safe_prompts)} safe prompts, {len(risky_prompts)} risky prompts")

        layer_results = []

        for layer in tqdm(range(self.n_layers), desc="Patching layers"):
            # Baseline: risky prompts without patching
            baseline_stops = []
            for prompt_data in risky_prompts[:10]:
                prompt = prompt_data.get('prompt', '')

                # Format and generate
                if self.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    chat = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                else:
                    formatted = prompt

                inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=100, do_sample=True,
                        temperature=0.7, pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
                )
                baseline_stops.append(1 if 'stop' in response.lower() else 0)

            baseline_rate = np.mean(baseline_stops)

            # Patching: replace risky activation with safe activation
            patched_stops = []
            for i, risky_data in enumerate(risky_prompts[:10]):
                safe_data = safe_prompts[i % len(safe_prompts)]

                # Get safe activation
                safe_act = self.get_activations(safe_data.get('prompt', ''), [layer])
                safe_activation = safe_act[layer].squeeze()

                # Generate with patched activation
                response = self.generate_with_patched_activation(
                    risky_data.get('prompt', ''),
                    layer,
                    safe_activation
                )
                patched_stops.append(1 if 'stop' in response.lower() else 0)

            patched_rate = np.mean(patched_stops)
            effect = patched_rate - baseline_rate

            result = {
                'layer': layer,
                'baseline_stop_rate': float(baseline_rate),
                'patched_stop_rate': float(patched_rate),
                'effect': float(effect),
                'is_significant': abs(effect) > 0.15
            }
            layer_results.append(result)

            if abs(effect) > 0.1:
                print(f"  Layer {layer}: effect={effect:+.3f} (baseline={baseline_rate:.2f}, patched={patched_rate:.2f})")

            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Find significant layers
        significant_layers = [r['layer'] for r in layer_results if r['is_significant']]

        results = LayerPatchingResults(
            model=self.model_name,
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
            n_layers=self.n_layers,
            patching_direction='safe_to_risky',
            layer_results=layer_results,
            significant_layers=significant_layers
        )

        return results

    def run_head_patching(
        self,
        layer: int,
        prompts_by_condition: Dict[str, List[Dict]],
        n_trials: int = 10
    ) -> Dict:
        """
        Run head-level activation patching for a specific layer.
        """
        print(f"\n{'='*70}")
        print(f"HEAD-LEVEL ACTIVATION PATCHING - Layer {layer}")
        print(f"{'='*70}")

        safe_fixed = prompts_by_condition['safe_fixed'][:n_trials]
        safe_variable = prompts_by_condition['safe_variable'][:n_trials]
        risky_fixed = prompts_by_condition['risky_fixed'][:n_trials]
        risky_variable = prompts_by_condition['risky_variable'][:n_trials]

        safe_prompts = safe_fixed + safe_variable
        risky_prompts = risky_fixed + risky_variable

        head_results = []

        for head_id in tqdm(range(self.n_heads), desc=f"Patching heads (L{layer})"):
            # Baseline
            baseline_stops = []
            for prompt_data in risky_prompts:
                prompt = prompt_data.get('prompt', '')

                if self.use_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    chat = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                else:
                    formatted = prompt

                inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=100, do_sample=True,
                        temperature=0.7, pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
                )
                baseline_stops.append(1 if 'stop' in response.lower() else 0)

            baseline_rate = np.mean(baseline_stops)

            # Head patching
            patched_stops = []
            for i, risky_data in enumerate(risky_prompts):
                safe_data = safe_prompts[i % len(safe_prompts)]

                # Get safe activation
                safe_act = self.get_activations(safe_data.get('prompt', ''), [layer])
                safe_activation = safe_act[layer].squeeze()

                # Generate with head-patched activation
                response = self.generate_with_patched_activation(
                    risky_data.get('prompt', ''),
                    layer,
                    safe_activation,
                    head_id=head_id
                )
                patched_stops.append(1 if 'stop' in response.lower() else 0)

            patched_rate = np.mean(patched_stops)
            effect = patched_rate - baseline_rate

            head_results.append({
                'head_id': head_id,
                'baseline_stop_rate': float(baseline_rate),
                'patched_stop_rate': float(patched_rate),
                'effect': float(effect),
                'is_significant': abs(effect) > 0.15
            })

            if abs(effect) > 0.1:
                print(f"  Head {head_id}: effect={effect:+.3f}")

            # GPU cleanup
            if (head_id + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Sort by effect
        head_results.sort(key=lambda x: abs(x['effect']), reverse=True)

        significant_heads = [h['head_id'] for h in head_results if h['is_significant']]

        return {
            'model': self.model_name,
            'layer': layer,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'n_heads': self.n_heads,
            'head_results': head_results,
            'significant_heads': significant_heads,
            'n_significant': len(significant_heads)
        }

    def save_results(self, results: Dict, prefix: str):
        """Save results to JSON."""
        timestamp = results.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))

        if isinstance(results, LayerPatchingResults):
            filename = f"{prefix}_{self.model_name}_{timestamp}.json"
            results = asdict(results)
        else:
            layer = results.get('layer', 'all')
            filename = f"{prefix}_{self.model_name}_L{layer}_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        print(f"Results saved: {output_path}")
        return output_path

    def run(
        self,
        mode: str = 'layer',
        layer: int = None,
        n_trials: int = 20
    ):
        """
        Run Phase 5 analysis.

        Args:
            mode: 'layer', 'head', or 'all'
            layer: Target layer for head-level patching
            n_trials: Number of trials per condition
        """
        print("=" * 70)
        print("PHASE 5: ACTIVATION PATCHING")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Mode: {mode}")

        # Load model
        self.load_model()

        # Load prompts
        prompts = self.load_prompts()

        if mode == 'layer' or mode == 'all':
            # Layer-level patching
            layer_results = self.run_layer_patching(prompts, n_trials)
            self.save_results(asdict(layer_results), 'layer_patching')

            print(f"\nSignificant layers: {layer_results.significant_layers}")

        if mode == 'head' or mode == 'all':
            # Head-level patching
            if layer is None:
                layer = 23  # Default causal layer

            head_results = self.run_head_patching(layer, prompts, n_trials)
            self.save_results(head_results, 'head_patching')

            print(f"\nSignificant heads (L{layer}): {head_results['significant_heads']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Phase 5: Activation Patching'
    )
    parser.add_argument('--model', type=str, default='llama',
                       choices=['llama', 'gemma', 'gemma_base'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='layer',
                       choices=['layer', 'head', 'all'],
                       help='Patching mode: layer, head, or all')
    parser.add_argument('--layer', type=int, default=None,
                       help='Target layer for head-level patching')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of trials per condition')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to experiment config (direct steering)')
    parser.add_argument('--prompts', type=str, default=None,
                       help='Path to condition_prompts.json')

    args = parser.parse_args()

    phase5 = Phase5ActivationPatching(
        model_name=args.model,
        gpu_id=args.gpu,
        config_path=args.config,
        prompts_path=args.prompts
    )

    phase5.run(
        mode=args.mode,
        layer=args.layer,
        n_trials=args.n_trials
    )


if __name__ == '__main__':
    main()
