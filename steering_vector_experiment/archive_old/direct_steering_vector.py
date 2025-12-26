#!/usr/bin/env python3
"""
Direct Steering Vector Intervention

SAE reconstruction error (~47%) makes SAE-based patching ineffective.
This approach directly uses hidden state differences without SAE encode/decode.

Method:
1. Extract hidden states from bankrupt vs safe prompts
2. Compute steering vector = mean(bankrupt) - mean(safe)
3. Apply steering vector directly to activations during generation
4. Measure behavioral change (betting decisions)

Advantages:
- No reconstruction error (direct activation modification)
- Simpler and more reliable causal intervention
- Can be analyzed post-hoc with SAE for interpretation
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_model_and_tokenizer, ModelRegistry


@dataclass
class SteeringResult:
    """Result from steering vector intervention."""
    prompt_id: str
    condition: str  # original condition (safe/risky)
    baseline_response: str
    baseline_action: str  # 'bet' or 'stop'
    baseline_bet: Optional[int]
    steered_responses: Dict[float, str]  # alpha -> response
    steered_actions: Dict[float, str]  # alpha -> action
    steered_bets: Dict[float, Optional[int]]  # alpha -> bet amount
    behavior_changed: bool


class DirectSteeringExperiment:
    """Direct steering vector intervention without SAE."""

    def __init__(
        self,
        model_name: str = "llama",
        gpu_id: int = 0,
        target_layers: List[int] = None
    ):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.target_layers = target_layers or [15, 20, 25, 30]

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        self.output_dir = Path("/data/llm_addiction/steering_vector_experiment_full/direct_steering")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.steering_vectors = {}  # layer -> vector

    def load_model(self):
        """Load base model."""
        print(f"Loading {self.model_name} on GPU {self.gpu_id}")

        # Register gemma_base if not already registered
        if self.model_name == "gemma_base":
            try:
                ModelRegistry.get("gemma_base")
            except ValueError:
                from utils import ModelConfig
                ModelRegistry.register('gemma_base', ModelConfig(
                    name='gemma_base',
                    model_id='google/gemma-2-9b',
                    d_model=3584,
                    n_layers=42,
                    use_chat_template=False
                ))

        # load_model_and_tokenizer expects model_name (e.g., 'llama'), not model_id
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_name, self.device
        )

        model_config = ModelRegistry.get(self.model_name)
        print(f"Model loaded: {model_config.model_id}")

    def extract_hidden_states(
        self,
        prompts: List[str],
        target_layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Extract hidden states for prompts at target layers.

        Returns:
            Dict mapping layer -> tensor of shape (n_prompts, hidden_dim)
        """
        hidden_states = {layer: [] for layer in target_layers}

        # Check if model uses chat template
        model_config = ModelRegistry.get(self.model_name)
        use_chat_template = getattr(model_config, 'use_chat_template', False)

        for prompt in tqdm(prompts, desc="Extracting hidden states"):
            # Apply chat template if needed
            formatted_prompt = prompt
            if use_chat_template:
                chat = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )

            inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

            # Get hidden states at each target layer
            # Note: hidden_states[0] is embedding, hidden_states[L+1] is layer L output
            for layer in target_layers:
                # Use last token position
                layer_hidden = outputs.hidden_states[layer + 1][:, -1, :]
                hidden_states[layer].append(layer_hidden.cpu())

            del outputs
            torch.cuda.empty_cache()

        # Stack into tensors
        for layer in target_layers:
            hidden_states[layer] = torch.cat(hidden_states[layer], dim=0)

        return hidden_states

    def compute_steering_vector(
        self,
        risky_prompts: List[str],
        safe_prompts: List[str]
    ) -> Dict[int, torch.Tensor]:
        """Compute steering vector from contrastive prompts.

        steering_vector = mean(risky) - mean(safe)

        Applying +steering_vector should push toward risky behavior.
        Applying -steering_vector should push toward safe behavior.
        """
        print("\n" + "="*60)
        print("COMPUTING STEERING VECTORS")
        print("="*60)

        print(f"Risky prompts: {len(risky_prompts)}")
        print(f"Safe prompts: {len(safe_prompts)}")

        # Extract hidden states
        print("\nExtracting risky hidden states...")
        risky_hidden = self.extract_hidden_states(risky_prompts, self.target_layers)

        print("\nExtracting safe hidden states...")
        safe_hidden = self.extract_hidden_states(safe_prompts, self.target_layers)

        # Compute steering vectors
        steering_vectors = {}
        for layer in self.target_layers:
            risky_mean = risky_hidden[layer].mean(dim=0)
            safe_mean = safe_hidden[layer].mean(dim=0)
            steering_vectors[layer] = risky_mean - safe_mean

            # Statistics
            magnitude = steering_vectors[layer].norm().item()
            print(f"Layer {layer}: magnitude = {magnitude:.4f}")

        self.steering_vectors = steering_vectors
        return steering_vectors

    def create_steering_hook(
        self,
        layer: int,
        alpha: float
    ):
        """Create a hook that applies steering vector.

        Args:
            layer: Layer to modify
            alpha: Scaling factor
                   +alpha pushes toward risky
                   -alpha pushes toward safe
        """
        steering_vector = self.steering_vectors[layer].to(self.device)

        def hook(module, input, output):
            # output is tuple: (hidden_states,) or (hidden_states, attention, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                # Modify last token position
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * steering_vector
                return (hidden,) + output[1:]
            else:
                output[:, -1, :] = output[:, -1, :] + alpha * steering_vector
                return output

        return hook

    def generate_with_steering(
        self,
        prompt: str,
        layer: int,
        alpha: float,
        max_new_tokens: int = 100
    ) -> str:
        """Generate response with steering vector applied."""
        # Get the layer module
        if self.model_name == "llama":
            layer_module = self.model.model.layers[layer]
        else:  # gemma
            layer_module = self.model.model.layers[layer]

        # Register hook
        hook = self.create_steering_hook(layer, alpha)
        handle = layer_module.register_forward_hook(hook)

        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()

        finally:
            handle.remove()

        return response

    def parse_response(self, response: str) -> Tuple[str, Optional[int]]:
        """Parse betting decision from response."""
        import re

        response_lower = response.lower()

        # Stop decision
        if 'stop' in response_lower:
            return 'stop', None

        # Extract bet amount
        amounts = re.findall(r'\$(\d+)', response)
        if amounts:
            bet = int(amounts[-1])
            if 5 <= bet <= 100:
                return 'bet', bet

        return 'bet', 10  # default

    def run_steering_experiment(
        self,
        test_prompts: List[Dict],
        alphas: List[float] = None,
        layer: int = 25
    ) -> List[SteeringResult]:
        """Run steering experiment on test prompts.

        Args:
            test_prompts: List of dicts with 'prompt', 'condition', 'id'
            alphas: Steering strengths to test
            layer: Which layer to apply steering
        """
        if alphas is None:
            alphas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

        results = []

        print(f"\n{'='*60}")
        print(f"STEERING EXPERIMENT")
        print(f"Layer: {layer}")
        print(f"Alphas: {alphas}")
        print(f"Test prompts: {len(test_prompts)}")
        print(f"{'='*60}")

        for prompt_info in tqdm(test_prompts, desc="Testing prompts"):
            prompt = prompt_info['prompt']
            condition = prompt_info.get('condition', 'unknown')
            prompt_id = prompt_info.get('id', 'unknown')

            # Baseline (no steering)
            baseline_response = self.generate_with_steering(prompt, layer, 0.0)
            baseline_action, baseline_bet = self.parse_response(baseline_response)

            # Steered responses
            steered_responses = {}
            steered_actions = {}
            steered_bets = {}

            for alpha in alphas:
                response = self.generate_with_steering(prompt, layer, alpha)
                action, bet = self.parse_response(response)

                steered_responses[alpha] = response
                steered_actions[alpha] = action
                steered_bets[alpha] = bet

            # Check if behavior changed
            behavior_changed = any(
                steered_actions[a] != baseline_action
                for a in alphas if a != 0.0
            )

            results.append(SteeringResult(
                prompt_id=prompt_id,
                condition=condition,
                baseline_response=baseline_response,
                baseline_action=baseline_action,
                baseline_bet=baseline_bet,
                steered_responses=steered_responses,
                steered_actions=steered_actions,
                steered_bets=steered_bets,
                behavior_changed=behavior_changed
            ))

        return results

    def analyze_results(self, results: List[SteeringResult]) -> Dict:
        """Analyze steering experiment results."""
        n_total = len(results)
        n_changed = sum(1 for r in results if r.behavior_changed)

        # Action changes by alpha
        alpha_effects = {}
        for alpha in results[0].steered_actions.keys():
            stop_count = sum(1 for r in results if r.steered_actions[alpha] == 'stop')
            bet_count = n_total - stop_count
            alpha_effects[alpha] = {
                'stop_rate': stop_count / n_total,
                'bet_rate': bet_count / n_total
            }

        # Bet amounts by alpha
        bet_amounts = {}
        for alpha in results[0].steered_bets.keys():
            bets = [r.steered_bets[alpha] for r in results
                   if r.steered_bets[alpha] is not None]
            if bets:
                bet_amounts[alpha] = {
                    'mean': np.mean(bets),
                    'std': np.std(bets),
                    'min': min(bets),
                    'max': max(bets)
                }

        return {
            'n_total': n_total,
            'n_behavior_changed': n_changed,
            'change_rate': n_changed / n_total,
            'alpha_effects': alpha_effects,
            'bet_amounts': {str(k): v for k, v in bet_amounts.items()}
        }

    def save_results(
        self,
        results: List[SteeringResult],
        analysis: Dict,
        layer: int
    ) -> str:
        """Save experiment results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        output = {
            'model': self.model_name,
            'layer': layer,
            'timestamp': timestamp,
            'n_results': len(results),
            'analysis': analysis,
            'results': [
                {
                    'prompt_id': r.prompt_id,
                    'condition': r.condition,
                    'baseline_action': r.baseline_action,
                    'baseline_bet': r.baseline_bet,
                    'steered_actions': {str(k): v for k, v in r.steered_actions.items()},
                    'steered_bets': {str(k): v for k, v in r.steered_bets.items()},
                    'behavior_changed': r.behavior_changed
                }
                for r in results
            ]
        }

        output_path = self.output_dir / f"direct_steering_{self.model_name}_L{layer}_{timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        return str(output_path)

    def save_steering_vectors(self) -> str:
        """Save computed steering vectors for later analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Convert to numpy for saving
        vectors = {}
        for layer, vec in self.steering_vectors.items():
            vectors[layer] = vec.cpu().numpy().tolist()

        output = {
            'model': self.model_name,
            'timestamp': timestamp,
            'layers': self.target_layers,
            'vectors': {str(k): v for k, v in vectors.items()}
        }

        output_path = self.output_dir / f"steering_vectors_{self.model_name}_{timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Steering vectors saved: {output_path}")
        return str(output_path)


def load_condition_prompts(prompts_path: str, model: str) -> Tuple[List[str], List[str]]:
    """Load risky and safe prompts from condition_prompts.json."""
    with open(prompts_path) as f:
        data = json.load(f)

    conditions = data['conditions'].get(model, [])

    risky_prompts = [c['prompt'] for c in conditions if 'risky' in c['condition_name']]
    safe_prompts = [c['prompt'] for c in conditions if 'safe' in c['condition_name']]

    return risky_prompts, safe_prompts


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Direct Steering Vector Experiment')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gemma_base'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--layer', type=int, default=25, help='Layer to apply steering')
    parser.add_argument('--prompts', type=str,
                       default='/data/llm_addiction/steering_vector_experiment_full/condition_prompts.json')

    args = parser.parse_args()

    # Initialize experiment
    exp = DirectSteeringExperiment(
        model_name=args.model,
        gpu_id=args.gpu,
        target_layers=[15, 20, 25, 30]
    )

    # Load model
    exp.load_model()

    # Load prompts
    risky_prompts, safe_prompts = load_condition_prompts(args.prompts, args.model)
    print(f"Loaded {len(risky_prompts)} risky, {len(safe_prompts)} safe prompts")

    # Compute steering vectors
    exp.compute_steering_vector(risky_prompts, safe_prompts)
    exp.save_steering_vectors()

    # Prepare test prompts (mix of safe and risky)
    test_prompts = []
    for i, prompt in enumerate(safe_prompts[:10]):
        test_prompts.append({'prompt': prompt, 'condition': 'safe', 'id': f'safe_{i}'})
    for i, prompt in enumerate(risky_prompts[:10]):
        test_prompts.append({'prompt': prompt, 'condition': 'risky', 'id': f'risky_{i}'})

    # Run steering experiment
    results = exp.run_steering_experiment(test_prompts, layer=args.layer)

    # Analyze and save
    analysis = exp.analyze_results(results)
    output_path = exp.save_results(results, analysis, args.layer)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Behavior changed: {analysis['n_behavior_changed']}/{analysis['n_total']} ({analysis['change_rate']*100:.1f}%)")
    print("\nStop rate by alpha:")
    for alpha, effects in sorted(analysis['alpha_effects'].items()):
        print(f"  alpha={alpha:+.1f}: stop={effects['stop_rate']*100:.1f}%")
    print(f"\nResults saved: {output_path}")


if __name__ == '__main__':
    main()
