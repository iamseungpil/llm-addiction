#!/usr/bin/env python3
"""
Phase 2: Direct Steering Validation

Apply steering vector directly to hidden states and measure behavioral change.
No SAE encode/decode cycle - 0% reconstruction error.

Input: Steering vectors from Phase 1
Output: Validated layers with causal effect measurements

Usage:
    python phase2_direct_steering.py --model llama --gpu 0
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
from scipy import stats

# Add parent for imports
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

from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class SteeringResult:
    """Result for a single test prompt."""
    prompt_id: str
    condition: str
    layer: int
    alpha: float
    response: str
    action: str  # 'bet' or 'stop'
    bet_amount: Optional[int]
    stop_logit: float
    bet_logit: float


@dataclass
class LayerValidation:
    """Validation result for a layer."""
    layer: int
    is_causal: bool
    effect_size: float  # Cohen's d
    p_value: float
    direction_correct: bool  # +alpha increases risky?
    stop_rates: Dict[float, float]  # alpha -> stop_rate
    mean_bets: Dict[float, float]  # alpha -> mean bet


class Phase2DirectSteering:
    """Direct steering validation without SAE."""

    def __init__(
        self,
        model_name: str = "llama",
        gpu_id: int = 0,
        config_path: str = None
    ):
        self.model_name = model_name
        self.gpu_id = gpu_id

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "experiment_config_direct_steering.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        self.output_dir = Path(self.config['output_dir']) / "phase2_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.steering_vectors = {}  # layer -> tensor

    def load_model(self):
        """Load base model."""
        model_config = self.config['models'].get(self.model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {self.model_name}")

        model_id = model_config['model_id']
        print(f"Loading {model_id} on GPU {self.gpu_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={'': 0},
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        self.model.eval()
        print(f"Model loaded: {model_id}")

    def load_steering_vectors(self, vectors_path: str = None):
        """Load steering vectors from Phase 1.

        Phase 1 saves as .npz in output_dir root (not phase1_results subdir).
        """
        if vectors_path is None:
            # Find latest steering vectors in output_dir root
            output_dir = Path(self.config['output_dir'])

            # Try .npz first (Phase 1 default format)
            vector_files = sorted(
                output_dir.glob(f"steering_vectors_{self.model_name}*.npz"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Fallback to .pt if no .npz found
            if not vector_files:
                vector_files = sorted(
                    output_dir.glob(f"steering_vectors_{self.model_name}*.pt"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

            if not vector_files:
                raise FileNotFoundError(
                    f"No steering vectors found in {output_dir}. "
                    f"Run Phase 1 (extract_steering_vectors.py) first."
                )
            vectors_path = vector_files[0]

        print(f"Loading steering vectors: {vectors_path}")

        # Handle different file formats
        vectors_path = Path(vectors_path)
        if vectors_path.suffix == '.npz':
            # Load .npz format (Phase 1 default)
            import numpy as np
            data = np.load(vectors_path, allow_pickle=True)

            # Extract vectors from npz
            # Format: layer_{layer}_vector (from extract_steering_vectors.py)
            for key in data.files:
                if key.endswith('_vector') and key.startswith('layer_'):
                    # Parse layer number from 'layer_15_vector'
                    parts = key.split('_')
                    layer = int(parts[1])
                    self.steering_vectors[layer] = torch.from_numpy(data[key]).to(self.device)
                elif key == 'metadata':
                    try:
                        metadata = data[key].item()
                        print(f"  Metadata: {metadata.get('model', 'unknown')}, "
                              f"layers: {metadata.get('target_layers', [])}")
                    except:
                        pass
        else:
            # Load .pt format
            data = torch.load(vectors_path, map_location='cpu')

            if isinstance(data, dict) and 'vectors' in data:
                vectors = data['vectors']
            else:
                vectors = data

            for layer, vec in vectors.items():
                layer_int = int(layer) if isinstance(layer, str) else layer
                self.steering_vectors[layer_int] = vec.to(self.device)

        print(f"Loaded {len(self.steering_vectors)} layer vectors")

    def load_test_prompts(self, prompts_path: str = None) -> List[Dict]:
        """Load test prompts from condition_prompts.json.

        Samples equally from all 4 conditions: safe_fixed, safe_variable,
        risky_fixed, risky_variable to ensure balanced testing.
        """
        if prompts_path is None:
            # Use prompts_path from config if available
            prompts_path = self.config.get('prompts_path')

            if prompts_path is None:
                # Fallback to default location
                prompts_path = Path(self.config['output_dir']).parent / "steering_vector_experiment_full" / "condition_prompts.json"

        print(f"Loading prompts: {prompts_path}")
        with open(prompts_path) as f:
            data = json.load(f)

        conditions = data['conditions'].get(self.model_name, [])
        if not conditions and self.model_name == 'gemma_base':
            # Fallback to gemma prompts for gemma_base
            conditions = data['conditions'].get('gemma', [])

        # Group by condition for balanced sampling
        from collections import defaultdict
        by_condition = defaultdict(list)
        for c in conditions:
            by_condition[c['condition_name']].append(c)

        # Calculate samples per condition
        n_total = self.config['phase2_direct_steering']['n_test_prompts']
        n_conditions = len(by_condition)
        n_per_condition = n_total // n_conditions if n_conditions > 0 else n_total

        # Sample equally from each condition
        prompts = []
        for cond_name in sorted(by_condition.keys()):
            cond_prompts = by_condition[cond_name]
            for i, c in enumerate(cond_prompts[:n_per_condition]):
                prompts.append({
                    'id': f"{c['condition_name']}_{i}",
                    'condition': c['condition_name'],
                    'prompt': c['prompt']
                })

        print(f"Loaded {len(prompts)} test prompts (balanced across {n_conditions} conditions)")
        for cond_name in sorted(by_condition.keys()):
            count = sum(1 for p in prompts if p['condition'] == cond_name)
            print(f"  {cond_name}: {count}")

        return prompts

    def create_steering_hook(self, layer: int, alpha: float):
        """Create hook that applies steering vector."""
        steering_vector = self.steering_vectors[layer]

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                # Modify last token position
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * steering_vector
                return (hidden,) + output[1:]
            else:
                output[:, -1, :] = output[:, -1, :] + alpha * steering_vector
                return output

        return hook

    def get_decision_logits(self, prompt: str) -> Tuple[float, float]:
        """Get logits for 'stop' vs 'bet' decision tokens."""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Get token IDs for decision keywords
        stop_tokens = self.tokenizer.encode(" stop", add_special_tokens=False)
        bet_tokens = self.tokenizer.encode(" bet", add_special_tokens=False)

        stop_logit = logits[stop_tokens[0]].item() if stop_tokens else 0.0
        bet_logit = logits[bet_tokens[0]].item() if bet_tokens else 0.0

        return stop_logit, bet_logit

    def generate_with_steering(
        self,
        prompt: str,
        layer: int,
        alpha: float
    ) -> Tuple[str, str, Optional[int], float, float]:
        """Generate response with steering and parse decision."""
        import re

        # Get layer module
        layer_module = self.model.model.layers[layer]

        # Register hook
        hook = self.create_steering_hook(layer, alpha)
        handle = layer_module.register_forward_hook(hook)

        try:
            # Apply chat template for models that use it (e.g., Gemma-it)
            formatted_prompt = prompt
            if self.config['models'][self.model_name].get('use_chat_template', False):
                chat = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True
                )

            inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                # Get logits for decision
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]

                stop_tokens = self.tokenizer.encode("stop", add_special_tokens=False)
                bet_tokens = self.tokenizer.encode("bet", add_special_tokens=False)
                stop_logit = logits[stop_tokens[0]].item() if stop_tokens else 0.0
                bet_logit = logits[bet_tokens[0]].item() if bet_tokens else 0.0

                # Generate full response
                gen_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the newly generated tokens (after input)
            input_len = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                gen_outputs[0][input_len:],
                skip_special_tokens=True
            ).strip()

        finally:
            handle.remove()

        # Parse decision
        response_lower = response.lower()
        if 'stop' in response_lower:
            action = 'stop'
            bet_amount = None
        else:
            action = 'bet'
            amounts = re.findall(r'\$(\d+)', response)
            bet_amount = int(amounts[-1]) if amounts else 10

        return response, action, bet_amount, stop_logit, bet_logit

    def run_layer_validation(
        self,
        layer: int,
        test_prompts: List[Dict],
        alphas: List[float]
    ) -> Tuple[LayerValidation, List[SteeringResult]]:
        """Validate a single layer."""
        results = []

        for prompt_info in tqdm(test_prompts, desc=f"Layer {layer}", leave=False):
            for alpha in alphas:
                response, action, bet_amount, stop_logit, bet_logit = \
                    self.generate_with_steering(
                        prompt_info['prompt'], layer, alpha
                    )

                results.append(SteeringResult(
                    prompt_id=prompt_info['id'],
                    condition=prompt_info['condition'],
                    layer=layer,
                    alpha=alpha,
                    response=response[:200],  # Truncate for storage
                    action=action,
                    bet_amount=bet_amount,
                    stop_logit=stop_logit,
                    bet_logit=bet_logit
                ))

        # Analyze results
        stop_rates = {}
        mean_bets = {}

        for alpha in alphas:
            alpha_results = [r for r in results if r.alpha == alpha]
            stop_count = sum(1 for r in alpha_results if r.action == 'stop')
            stop_rates[alpha] = stop_count / len(alpha_results) if alpha_results else 0

            bets = [r.bet_amount for r in alpha_results if r.bet_amount is not None]
            mean_bets[alpha] = np.mean(bets) if bets else 0

        # Calculate effect size: compare -2.0 vs +2.0 (full steering range)
        neg_stops = [1 if r.action == 'stop' else 0
                    for r in results if r.alpha == -2.0]
        pos_stops = [1 if r.action == 'stop' else 0
                    for r in results if r.alpha == 2.0]

        if neg_stops and pos_stops:
            # Cohen's d: positive effect means +alpha decreases stop rate (more risky)
            mean_diff = np.mean(pos_stops) - np.mean(neg_stops)
            pooled_std = np.sqrt(
                (np.var(neg_stops) + np.var(pos_stops)) / 2
            )
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

            # t-test
            t_stat, p_value = stats.ttest_ind(neg_stops, pos_stops)
        else:
            effect_size = 0
            p_value = 1.0

        # Check direction: +alpha should decrease stop rate (more risky)
        # Steering vector = mean(bankrupt) - mean(safe)
        # +alpha pushes toward bankrupt (risky) -> lower stop rate
        direction_correct = stop_rates.get(2.0, 0) < stop_rates.get(-2.0, 1)

        # Check consistency across conditions (safe vs risky prompts)
        conditions_consistent = True
        for cond_type in ['safe', 'risky']:
            cond_results = [r for r in results if cond_type in r.condition]
            if cond_results:
                neg_rate = np.mean([1 if r.action == 'stop' else 0
                                   for r in cond_results if r.alpha == -2.0])
                pos_rate = np.mean([1 if r.action == 'stop' else 0
                                   for r in cond_results if r.alpha == 2.0])
                # Check if direction is consistent (pos_rate < neg_rate)
                if pos_rate >= neg_rate:
                    conditions_consistent = False

        # Determine if layer is causal
        min_effect = self.config['phase2_direct_steering']['validation']['min_effect_size']
        p_threshold = self.config['phase2_direct_steering']['validation']['p_value_threshold']
        is_causal = (abs(effect_size) >= min_effect and
                    p_value < p_threshold and
                    direction_correct and
                    conditions_consistent)

        validation = LayerValidation(
            layer=layer,
            is_causal=is_causal,
            effect_size=effect_size,
            p_value=p_value,
            direction_correct=direction_correct,
            stop_rates={k: round(v, 4) for k, v in stop_rates.items()},
            mean_bets={k: round(v, 2) for k, v in mean_bets.items()}
        )

        return validation, results

    def run(self, vectors_path: str = None, prompts_path: str = None, target_layers: list = None):
        """Run Phase 2 validation."""
        print("=" * 70)
        print("PHASE 2: DIRECT STEERING VALIDATION")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"GPU: {self.gpu_id}")

        # Load components
        self.load_model()
        self.load_steering_vectors(vectors_path)

        # Prompts are already balanced-sampled in load_test_prompts
        test_prompts = self.load_test_prompts(prompts_path)

        # Get config values
        alphas = self.config['phase2_direct_steering']['alpha_values']

        # Use provided target_layers or fall back to config/all layers
        if target_layers is None:
            # Use all available layers from steering vectors
            target_layers = sorted(self.steering_vectors.keys())

        # Filter to available layers
        available_layers = [l for l in target_layers if l in self.steering_vectors]
        print(f"\nTarget layers: {available_layers}")
        print(f"Test prompts: {len(test_prompts)}")
        print(f"Alpha values: {alphas}")

        # Run validation for each layer
        all_validations = []
        all_results = []

        for layer in available_layers:
            print(f"\n{'='*50}")
            print(f"Validating Layer {layer}")
            print(f"{'='*50}")

            validation, results = self.run_layer_validation(
                layer, test_prompts, alphas
            )
            all_validations.append(validation)
            all_results.extend(results)

            # Print summary
            print(f"  Effect size: {validation.effect_size:.3f}")
            print(f"  P-value: {validation.p_value:.4f}")
            print(f"  Direction correct: {validation.direction_correct}")
            print(f"  Is causal: {validation.is_causal}")
            print(f"  Stop rates: {validation.stop_rates}")

        # Save results
        self.save_results(all_validations, all_results)

        # Summary
        print("\n" + "=" * 70)
        print("PHASE 2 SUMMARY")
        print("=" * 70)
        causal_layers = [v for v in all_validations if v.is_causal]
        print(f"Causal layers: {len(causal_layers)}/{len(all_validations)}")
        for v in causal_layers:
            print(f"  Layer {v.layer}: d={v.effect_size:.3f}, p={v.p_value:.4f}")

        return all_validations, all_results

    def save_results(
        self,
        validations: List[LayerValidation],
        results: List[SteeringResult]
    ):
        """Save validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Get layer range for filename
        if validations:
            layers = [v.layer for v in validations]
            layer_range = f"L{min(layers)}-{max(layers)}"
        else:
            layer_range = "Lnone"

        # Save validations
        validation_data = {
            'model': self.model_name,
            'timestamp': timestamp,
            'n_layers_tested': len(validations),
            'n_causal_layers': sum(1 for v in validations if v.is_causal),
            'validations': [asdict(v) for v in validations]
        }

        val_path = self.output_dir / f"layer_validations_{self.model_name}_{layer_range}_{timestamp}.json"
        with open(val_path, 'w') as f:
            json.dump(validation_data, f, indent=2, cls=NumpyEncoder)
        print(f"\nValidations saved: {val_path}")

        # Save detailed results
        results_data = {
            'model': self.model_name,
            'timestamp': timestamp,
            'n_results': len(results),
            'results': [asdict(r) for r in results]
        }

        results_path = self.output_dir / f"steering_results_{self.model_name}_{layer_range}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, cls=NumpyEncoder)
        print(f"Results saved: {results_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Phase 2: Direct Steering Validation')
    parser.add_argument('--model', type=str, default='llama',
                       choices=['llama', 'gemma'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vectors', type=str, default=None,
                       help='Path to steering vectors from Phase 1')
    parser.add_argument('--prompts', type=str, default=None,
                       help='Path to test prompts')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers or range (e.g., "0,5,10" or "0-15")')

    args = parser.parse_args()

    # Parse layers argument
    target_layers = None
    if args.layers:
        if '-' in args.layers and ',' not in args.layers:
            # Range format: "0-15"
            start, end = map(int, args.layers.split('-'))
            target_layers = list(range(start, end + 1))
        else:
            # Comma-separated: "0,5,10,15"
            target_layers = [int(l) for l in args.layers.split(',')]

    phase2 = Phase2DirectSteering(
        model_name=args.model,
        gpu_id=args.gpu,
        config_path=args.config
    )

    phase2.run(vectors_path=args.vectors, prompts_path=args.prompts, target_layers=target_layers)


if __name__ == '__main__':
    main()
