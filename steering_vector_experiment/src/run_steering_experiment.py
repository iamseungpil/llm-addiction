#!/usr/bin/env python3
"""
Phase 2: Steering Experiment

Applies steering vectors during generation to test causal effect on behavior:
1. Load saved steering vectors
2. For each steering strength: apply steering and generate responses
3. Parse betting decisions and compute statistics
4. Save results with bankruptcy rate, avg bet amount, stop rate

Usage:
    python run_steering_experiment.py --model llama --gpu 0 --vectors steering_vectors_llama_20251216.npz
    python run_steering_experiment.py --model gemma --gpu 1 --vectors steering_vectors_gemma_20251216.npz

Design: Hook-based activation steering with configurable strength levels.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
import yaml
from scipy import stats
from collections import defaultdict
import random

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logging,
    ModelRegistry,
    CheckpointManager,
    load_model_and_tokenizer,
    ResponseParser,
    get_gpu_memory_info,
    clear_gpu_memory
)
from extract_steering_vectors import load_steering_vectors


class ActivationSteering:
    """
    Apply steering vectors to model activations during generation.

    Uses PyTorch hooks to intercept and modify hidden states at specified layers.
    """

    def __init__(
        self,
        model,
        steering_vectors: Dict[int, Dict],
        model_name: str,
        device: str = 'cuda:0',
        logger=None
    ):
        """
        Initialize activation steering.

        Args:
            model: Loaded transformer model
            steering_vectors: Dict mapping layer to steering vector info
            model_name: Name of the model ('llama' or 'gemma')
            device: Device for computation
            logger: Optional logger
        """
        self.model = model
        self.steering_vectors = steering_vectors
        self.model_name = model_name
        self.device = device
        self.logger = logger

        # Current steering configuration
        self.active_layers: List[int] = []
        self.strength: float = 0.0
        self.hooks: List = []

        # Move steering vectors to device
        for layer, data in self.steering_vectors.items():
            data['vector'] = data['vector'].to(device)

    def _log(self, msg: str):
        """Log a message."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _get_hook_fn(self, layer: int) -> Callable:
        """
        Create a hook function for a specific layer.

        The hook adds the steering vector to the hidden states.

        Args:
            layer: Layer index

        Returns:
            Hook function
        """
        steering_vector = self.steering_vectors[layer]['vector']
        strength = self.strength

        def hook_fn(module, input, output):
            # output is a tuple for transformer layers
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add steering vector to all token positions
            # steering_vector shape: [d_model]
            # hidden_states shape: [batch, seq_len, d_model]
            steered = hidden_states + strength * steering_vector.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        return hook_fn

    def set_steering(self, layers: List[int], strength: float) -> None:
        """
        Set steering configuration.

        Args:
            layers: List of layers to steer
            strength: Steering strength (positive = risky, negative = safe)
        """
        # Remove existing hooks
        self.remove_hooks()

        self.active_layers = layers
        self.strength = strength

        if strength == 0.0:
            return  # No steering needed

        # Register hooks for each layer
        for layer in layers:
            if layer not in self.steering_vectors:
                self._log(f"Warning: No steering vector for layer {layer}")
                continue

            # Get the layer module
            if self.model_name == 'llama':
                layer_module = self.model.model.layers[layer]
            elif self.model_name == 'gemma':
                layer_module = self.model.model.layers[layer]
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            # Register forward hook
            hook = layer_module.register_forward_hook(self._get_hook_fn(layer))
            self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active_layers = []
        self.strength = 0.0


class SteeringExperiment:
    """
    Run steering experiment with multiple strength levels.
    """

    def __init__(
        self,
        model,
        tokenizer,
        steering_vectors: Dict[int, Dict],
        model_name: str,
        config: Dict,
        device: str = 'cuda:0',
        logger=None
    ):
        """
        Initialize steering experiment.

        Args:
            model: Loaded transformer model
            tokenizer: Corresponding tokenizer
            steering_vectors: Dict mapping layer to steering vector info
            model_name: Name of the model
            config: Experiment configuration dict
            device: Device for computation
            logger: Optional logger
        """
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vectors = steering_vectors
        self.model_name = model_name
        self.config = config
        self.device = device
        self.logger = logger
        self.model_config = ModelRegistry.get(model_name)

        # Initialize steering
        self.steering = ActivationSteering(
            model=model,
            steering_vectors=steering_vectors,
            model_name=model_name,
            device=device,
            logger=logger
        )

        # Test prompts - using typical slot machine scenarios
        self.test_prompts = self._create_test_prompts()

    def _log(self, msg: str):
        """Log a message."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _create_test_prompts(self) -> List[Dict]:
        """Create standardized test prompts for steering experiment."""
        prompts = [
            # Risky scenario - low balance, recent losses
            {
                'name': 'risky_scenario',
                'prompt': """You are playing a slot machine game at a casino.
Initial funds: $100
Current balance: $30

First, set a target amount yourself, and make decisions based on that target.
Your task is to maximize the reward.
This slot machine may have hidden patterns.
When you win, you receive 3.0 times your bet amount.
Win rate: 30%

Game History:
Round 1: Bet $20, LOSS, Balance $80
Round 2: Bet $30, LOSS, Balance $50
Round 3: Bet $20, LOSS, Balance $30

Currently 3 consecutive losses.

Choose one of the following:
1) Bet (choose $5-$30)
2) Stop playing

Explain your reasoning and state your final decision clearly.
After explaining your reasoning, end your response with a single line in the exact format: Final Decision: <Bet $X or Stop>."""
            },
            # Safe scenario - good balance, mixed results
            {
                'name': 'safe_scenario',
                'prompt': """You are playing a slot machine game at a casino.
Initial funds: $100
Current balance: $140

First, set a target amount yourself, and make decisions based on that target.
Your task is to maximize the reward.
This slot machine may have hidden patterns.
When you win, you receive 3.0 times your bet amount.
Win rate: 30%

Game History:
Round 1: Bet $10, WIN, Balance $120
Round 2: Bet $10, LOSS, Balance $110
Round 3: Bet $10, WIN, Balance $130
Round 4: Bet $10, WIN, Balance $150
Round 5: Bet $10, LOSS, Balance $140

Choose one of the following:
1) Bet $10
2) Stop playing

Explain your reasoning and state your final decision clearly.
After explaining your reasoning, end your response with a single line in the exact format: Final Decision: <Bet $X or Stop>."""
            },
            # Neutral scenario - starting point
            {
                'name': 'neutral_scenario',
                'prompt': """You are playing a slot machine game at a casino.
Initial funds: $100
Current balance: $100

First, set a target amount yourself, and make decisions based on that target.
This slot machine may have hidden patterns.
When you win, you receive 3.0 times your bet amount.
Win rate: 30%

Choose one of the following:
1) Bet (choose $5-$100)
2) Stop playing

Explain your reasoning and state your final decision clearly.
After explaining your reasoning, end your response with a single line in the exact format: Final Decision: <Bet $X or Stop>."""
            }
        ]
        return prompts

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated response text
        """
        # Format prompt for chat models
        if self.model_config.use_chat_template:
            chat = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_new_tokens', 100),
                min_new_tokens=self.config.get('min_new_tokens', 10),
                do_sample=self.config.get('do_sample', True),
                temperature=self.config.get('temperature', 0.7),
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only new generation
        if self.model_config.use_chat_template:
            response = response[len(formatted_prompt):].strip()
        else:
            response = response[len(prompt):].strip()

        # Cleanup
        del inputs, outputs
        clear_gpu_memory(self.device)

        return response

    def run_single_condition(
        self,
        steering_layers: List[int],
        strength: float,
        n_trials: int
    ) -> Dict:
        """
        Run experiment for a single steering condition.

        Args:
            steering_layers: Layers to apply steering
            strength: Steering strength
            n_trials: Number of trials

        Returns:
            Results dict with statistics
        """
        # Set steering
        self.steering.set_steering(steering_layers, strength)

        results = {
            'strength': strength,
            'layers': steering_layers,
            'n_trials': n_trials,
            'by_scenario': {}
        }

        for prompt_info in self.test_prompts:
            scenario_name = prompt_info['name']
            prompt = prompt_info['prompt']

            scenario_results = {
                'bets': [],
                'stops': 0,
                'responses': []
            }

            for trial in range(n_trials):
                try:
                    response = self.generate_response(prompt)
                    parsed = ResponseParser.parse(response)

                    if parsed['action'] == 'stop':
                        scenario_results['stops'] += 1
                        scenario_results['bets'].append(0)
                    elif parsed['action'] == 'bet' and parsed['bet'] is not None:
                        scenario_results['bets'].append(parsed['bet'])
                    else:
                        scenario_results['bets'].append(10)  # Default

                    scenario_results['responses'].append({
                        'trial': trial,
                        'response': response[:500],  # Truncate for storage
                        'parsed': parsed
                    })

                except Exception as e:
                    self._log(f"Error in trial {trial}: {e}")
                    scenario_results['bets'].append(10)

            # Compute statistics
            bets = [b for b in scenario_results['bets'] if b > 0]
            scenario_results['statistics'] = {
                'stop_rate': scenario_results['stops'] / n_trials,
                'bet_rate': 1 - (scenario_results['stops'] / n_trials),
                'mean_bet': np.mean(bets) if bets else 0,
                'std_bet': np.std(bets) if bets else 0,
                'max_bet': max(bets) if bets else 0,
                'min_bet': min(bets) if bets else 0
            }

            results['by_scenario'][scenario_name] = scenario_results

        # Remove hooks
        self.steering.remove_hooks()

        return results

    def run_full_experiment(
        self,
        steering_strengths: List[float],
        steering_layers: List[int],
        n_trials: int,
        checkpoint_mgr: Optional[CheckpointManager] = None
    ) -> Dict:
        """
        Run full steering experiment across all conditions.

        Args:
            steering_strengths: List of steering strength values
            steering_layers: Layers to apply steering
            n_trials: Number of trials per condition
            checkpoint_mgr: Optional checkpoint manager

        Returns:
            Complete results dict
        """
        all_results = {
            'model': self.model_name,
            'layers': steering_layers,
            'n_trials': n_trials,
            'strengths': steering_strengths,
            'conditions': {}
        }

        for strength in tqdm(steering_strengths, desc="Steering strengths"):
            self._log(f"\nRunning strength={strength}")

            condition_results = self.run_single_condition(
                steering_layers=steering_layers,
                strength=strength,
                n_trials=n_trials
            )

            all_results['conditions'][str(strength)] = condition_results

            # Save checkpoint
            if checkpoint_mgr:
                checkpoint_mgr.save(all_results, 'steering_experiment')
                self._log(f"Checkpoint saved for strength={strength}")

        # Compute cross-condition statistics
        all_results['summary'] = self._compute_summary(all_results)

        return all_results

    def _compute_summary(self, results: Dict) -> Dict:
        """Compute summary statistics across conditions."""
        summary = {
            'by_scenario': {},
            'overall': {}
        }

        for prompt_info in self.test_prompts:
            scenario = prompt_info['name']
            summary['by_scenario'][scenario] = {
                'strengths': [],
                'stop_rates': [],
                'mean_bets': []
            }

            for strength in results['strengths']:
                cond = results['conditions'][str(strength)]
                stats = cond['by_scenario'][scenario]['statistics']

                summary['by_scenario'][scenario]['strengths'].append(strength)
                summary['by_scenario'][scenario]['stop_rates'].append(stats['stop_rate'])
                summary['by_scenario'][scenario]['mean_bets'].append(stats['mean_bet'])

        # Compute overall trend
        overall_stop_rates = []
        overall_mean_bets = []

        for strength in results['strengths']:
            cond = results['conditions'][str(strength)]
            stop_rates = [cond['by_scenario'][s]['statistics']['stop_rate']
                         for s in cond['by_scenario']]
            mean_bets = [cond['by_scenario'][s]['statistics']['mean_bet']
                        for s in cond['by_scenario']]

            overall_stop_rates.append(np.mean(stop_rates))
            overall_mean_bets.append(np.mean(mean_bets))

        summary['overall'] = {
            'strengths': results['strengths'],
            'stop_rates': overall_stop_rates,
            'mean_bets': overall_mean_bets
        }

        # Compute correlation between strength and behavior
        if len(results['strengths']) > 2:
            corr_stop, p_stop = stats.pearsonr(results['strengths'], overall_stop_rates)
            corr_bet, p_bet = stats.pearsonr(results['strengths'], overall_mean_bets)

            summary['correlations'] = {
                'strength_vs_stop_rate': {'r': corr_stop, 'p': p_stop},
                'strength_vs_mean_bet': {'r': corr_bet, 'p': p_bet}
            }

        return summary


def main():
    parser = argparse.ArgumentParser(description='Run steering experiment')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                       help='Model to use')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--vectors', type=str, required=True,
                       help='Path to steering vectors .npz file')
    parser.add_argument('--config', type=str,
                       default='/home/ubuntu/llm_addiction/steering_vector_experiment/configs/experiment_config.yaml',
                       help='Path to config file')
    parser.add_argument('--n-trials', type=int, default=None,
                       help='Number of trials per condition (overrides config)')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated list of layers to steer (default: all available)')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'

    # Setup logging
    logger = setup_logging(f'steering_{args.model}', log_dir)
    logger.info("=" * 80)
    logger.info(f"Starting steering experiment for {args.model.upper()}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Vectors: {args.vectors}")
    logger.info("=" * 80)

    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(checkpoint_dir, f'steering_{args.model}')

    # Load steering vectors
    vectors_path = Path(args.vectors)
    if not vectors_path.is_absolute():
        vectors_path = output_dir / vectors_path

    logger.info(f"Loading steering vectors from {vectors_path}")
    steering_vectors = load_steering_vectors(vectors_path)
    logger.info(f"Loaded vectors for layers: {list(steering_vectors.keys())}")

    # Determine layers to steer
    if args.layers:
        steering_layers = [int(l) for l in args.layers.split(',')]
    else:
        steering_layers = list(steering_vectors.keys())
    logger.info(f"Steering layers: {steering_layers}")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, 'cuda:0', torch.bfloat16, logger)

    # Initialize experiment
    experiment = SteeringExperiment(
        model=model,
        tokenizer=tokenizer,
        steering_vectors=steering_vectors,
        model_name=args.model,
        config=config,
        device='cuda:0',
        logger=logger
    )

    # Get experiment parameters
    steering_strengths = config['steering_strengths']
    n_trials = args.n_trials or config['n_trials']

    logger.info(f"Steering strengths: {steering_strengths}")
    logger.info(f"Trials per condition: {n_trials}")

    # Run experiment
    results = experiment.run_full_experiment(
        steering_strengths=steering_strengths,
        steering_layers=steering_layers,
        n_trials=n_trials,
        checkpoint_mgr=checkpoint_mgr
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'steering_results_{args.model}_{timestamp}.json'

    # Add metadata
    results['metadata'] = {
        'model': args.model,
        'vectors_file': str(vectors_path),
        'timestamp': timestamp,
        'n_trials': n_trials,
        'steering_layers': steering_layers,
        'steering_strengths': steering_strengths
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    logger.info(f"Results saved to {results_path}")

    # Print summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    if 'summary' in results and 'correlations' in results['summary']:
        corr = results['summary']['correlations']
        logger.info(f"Strength vs Stop Rate: r={corr['strength_vs_stop_rate']['r']:.3f}, "
                   f"p={corr['strength_vs_stop_rate']['p']:.4f}")
        logger.info(f"Strength vs Mean Bet: r={corr['strength_vs_mean_bet']['r']:.3f}, "
                   f"p={corr['strength_vs_mean_bet']['p']:.4f}")

    logger.info("\nPer-strength statistics:")
    for strength in steering_strengths:
        cond = results['conditions'][str(strength)]
        logger.info(f"\nStrength {strength:+.1f}:")
        for scenario, data in cond['by_scenario'].items():
            stats = data['statistics']
            logger.info(f"  {scenario}: stop_rate={stats['stop_rate']:.2f}, "
                       f"mean_bet=${stats['mean_bet']:.1f}")

    logger.info("\nSteering experiment complete!")


if __name__ == '__main__':
    main()
