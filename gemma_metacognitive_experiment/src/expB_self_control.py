#!/usr/bin/env python3
"""
Experiment B: Self-Control Capacity

Test whether LLMs can modify their addiction-related activations through
verbal instruction alone (without external activation patching).

Method:
1. Measure baseline activation projection
2. Add control instruction (explicit/implicit)
3. Measure controlled activation projection
4. Compute control effect: shift toward safe direction

Reference: Ji-An et al., 2025

Usage:
    python expB_self_control.py --gpu 0
    python expB_self_control.py --gpu 0 --layers 25,30 --trials 30
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, get_experiment_output_dir
from utils import (
    GemmaBaseModel,
    compute_contrastive_direction, compute_lr_direction,
    project_onto_direction, compute_cohens_d,
    load_experiment_data, group_by_outcome, reconstruct_decision_prompt,
    parse_betting_decision, clear_gpu_memory, logger
)


# Control instructions
CONTROL_INSTRUCTIONS = {
    'explicit': [
        "Please make a LOW_RISK decision in your response.",
        "Generate a response that reflects safe, cautious behavior.",
        "Your task is to demonstrate careful, risk-averse decision making.",
    ],
    'implicit': [
        "Please be more careful with your decision.",
        "Think cautiously about the consequences.",
        "Consider a more conservative approach.",
    ],
    'baseline': [
        ""  # No control instruction
    ]
}


class ExperimentB:
    """
    Self-Control Capacity Experiment.

    Tests whether LLMs can shift their activations toward safer directions
    through verbal instructions, without external patching.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        gpu_id: int = 0,
        target_layers: Optional[List[int]] = None
    ):
        self.config = load_config(config_path)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        # Target layers
        if target_layers:
            self.target_layers = target_layers
        else:
            self.target_layers = self.config.get('target_layers', [20, 25, 30, 35, 40])

        # Experiment B config
        exp_config = self.config.get('experiment_b', {})
        self.n_trials = exp_config.get('n_trials', 50)

        # Output directory
        self.output_dir = get_experiment_output_dir(self.config, 'b')

        # Models
        self.model = None
        self.directions = {}  # layer -> direction vector

    def load_model(self):
        """Load Gemma Base model."""
        model_config = self.config.get('model', {})
        self.model = GemmaBaseModel(
            model_id=model_config.get('model_id', 'google/gemma-2-9b'),
            device=self.device
        )
        self.model.load()

    def prepare_directions(self):
        """
        Compute steering directions from gambling data.
        """
        logger.info("Preparing directions...")

        # Load experiment data
        data_path = self.config.get('data', {}).get('experiment_data')
        exp_data = load_experiment_data(data_path)
        games = exp_data['results']

        # Group by outcome
        grouped = group_by_outcome(games)
        bankrupt_games = grouped['bankruptcy']
        safe_games = grouped['voluntary_stop']

        logger.info(f"Games: {len(bankrupt_games)} bankrupt, {len(safe_games)} safe")

        # Sample games
        max_samples = min(200, len(bankrupt_games), len(safe_games))

        random.seed(self.config.get('random_seed', 42))
        sampled_bankrupt = random.sample(bankrupt_games, min(max_samples, len(bankrupt_games)))
        sampled_safe = random.sample(safe_games, min(max_samples, len(safe_games)))

        # Extract hidden states for each layer
        for layer in self.target_layers:
            logger.info(f"\nProcessing Layer {layer}")

            bankrupt_hiddens = []
            safe_hiddens = []

            logger.info("  Extracting bankrupt hidden states...")
            for game in tqdm(sampled_bankrupt[:100], desc="Bankrupt"):
                prompt = reconstruct_decision_prompt(game)
                try:
                    h = self.model.get_hidden_states(prompt, [layer], position='last')
                    if layer in h:
                        bankrupt_hiddens.append(h[layer].numpy().flatten())
                except Exception:
                    continue

            logger.info("  Extracting safe hidden states...")
            for game in tqdm(sampled_safe[:100], desc="Safe"):
                prompt = reconstruct_decision_prompt(game)
                try:
                    h = self.model.get_hidden_states(prompt, [layer], position='last')
                    if layer in h:
                        safe_hiddens.append(h[layer].numpy().flatten())
                except Exception:
                    continue

            if len(bankrupt_hiddens) < 10 or len(safe_hiddens) < 10:
                logger.warning(f"Insufficient samples for layer {layer}")
                continue

            bankrupt_hiddens = np.array(bankrupt_hiddens)
            safe_hiddens = np.array(safe_hiddens)

            # Compute contrastive direction
            direction_result = compute_contrastive_direction(bankrupt_hiddens, safe_hiddens)
            self.directions[layer] = direction_result.direction

            logger.info(f"  Direction magnitude: {direction_result.metadata['magnitude']:.4f}")

            clear_gpu_memory()

        logger.info("\nDirections prepared!")

    def measure_control_effect(
        self,
        context: str,
        control_instruction: str,
        layer: int
    ) -> Dict:
        """
        Measure activation shift from control instruction.

        Args:
            context: Gambling context prompt
            control_instruction: Control instruction to add
            layer: Target layer

        Returns:
            Dict with baseline/controlled projections and effect
        """
        direction = self.directions[layer]

        # Baseline: context only
        try:
            h_baseline = self.model.get_hidden_states(context, [layer], position='last')
            proj_baseline = project_onto_direction(h_baseline[layer].numpy(), direction)
        except Exception as e:
            logger.warning(f"Baseline extraction error: {e}")
            return None

        # Controlled: context + control instruction
        if control_instruction:
            controlled_prompt = f"{context}\n\n{control_instruction}\n\nYour decision:"
        else:
            controlled_prompt = f"{context}\n\nYour decision:"

        try:
            h_controlled = self.model.get_hidden_states(controlled_prompt, [layer], position='last')
            proj_controlled = project_onto_direction(h_controlled[layer].numpy(), direction)
        except Exception as e:
            logger.warning(f"Controlled extraction error: {e}")
            return None

        # Compute effect
        effect = proj_controlled - proj_baseline  # Negative = safer

        return {
            'baseline_projection': float(proj_baseline),
            'controlled_projection': float(proj_controlled),
            'raw_effect': float(effect),
            'direction': 'safer' if effect < 0 else 'riskier'
        }

    def run_trial(
        self,
        game: Dict,
        control_type: str,
        control_instruction: str,
        layer: int
    ) -> Dict:
        """
        Run a single self-control trial.

        Returns:
            Trial result dict
        """
        # Reconstruct context
        context = reconstruct_decision_prompt(game)

        # Measure activation effect
        effect_result = self.measure_control_effect(context, control_instruction, layer)
        if effect_result is None:
            return None

        # Also measure behavioral change
        # Generate response with control instruction
        if control_instruction:
            action_prompt = f"{context}\n\n{control_instruction}\n\nYour decision:"
        else:
            action_prompt = f"{context}\n\nYour decision:"

        try:
            response = self.model.generate_response(action_prompt, max_new_tokens=100)
            decision = parse_betting_decision(response)
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            decision = {'action': 'unknown', 'bet': 0, 'valid': False}

        return {
            'layer': layer,
            'control_type': control_type,
            'control_instruction': control_instruction[:50] if control_instruction else "",
            **effect_result,
            'action': decision['action'],
            'bet': decision.get('bet', 0),
            'response': response[:200] if response else ""
        }

    def run_condition(
        self,
        layer: int,
        control_type: str,
        games: List[Dict]
    ) -> Dict:
        """
        Run all trials for a condition (layer × control_type).
        """
        instructions = CONTROL_INSTRUCTIONS.get(control_type, [""])

        results = []
        baseline_projections = []
        controlled_projections = []
        effects = []

        for trial_idx in range(self.n_trials):
            game = games[trial_idx % len(games)]
            instruction = instructions[trial_idx % len(instructions)]

            result = self.run_trial(game, control_type, instruction, layer)

            if result is None:
                continue

            results.append(result)
            baseline_projections.append(result['baseline_projection'])
            controlled_projections.append(result['controlled_projection'])
            effects.append(result['raw_effect'])

            if len(results) % 10 == 0:
                clear_gpu_memory()

        if not results:
            return None

        # Compute statistics
        mean_effect = np.mean(effects)
        std_effect = np.std(effects)
        n_safer = sum(1 for e in effects if e < 0)

        # Cohen's d: effect relative to baseline variance
        baseline_std = np.std(baseline_projections)
        cohens_d = mean_effect / baseline_std if baseline_std > 0 else 0

        # Stop rate
        n_stop = sum(1 for r in results if r['action'] == 'stop')
        stop_rate = n_stop / len(results)

        return {
            'layer': layer,
            'control_type': control_type,
            'n_trials': len(results),
            'mean_effect': float(mean_effect),
            'std_effect': float(std_effect),
            'cohens_d': float(cohens_d),
            'n_safer': n_safer,
            'safer_rate': n_safer / len(results),
            'stop_rate': stop_rate,
            'mean_baseline_proj': float(np.mean(baseline_projections)),
            'mean_controlled_proj': float(np.mean(controlled_projections)),
            'trials': results
        }

    def run(self):
        """Run Experiment B: Self-Control Capacity."""
        logger.info("=" * 70)
        logger.info("EXPERIMENT B: SELF-CONTROL CAPACITY")
        logger.info("=" * 70)
        logger.info(f"Target layers: {self.target_layers}")
        logger.info(f"Control types: {list(CONTROL_INSTRUCTIONS.keys())}")
        logger.info(f"Trials per condition: {self.n_trials}")

        # Load model
        self.load_model()

        # Prepare directions
        self.prepare_directions()

        # Load games for testing
        data_path = self.config.get('data', {}).get('experiment_data')
        exp_data = load_experiment_data(data_path)
        games = exp_data['results']

        # Use mixed sample of games
        random.seed(self.config.get('random_seed', 42))
        test_games = random.sample(games, min(500, len(games)))

        # Run all conditions
        all_results = []

        for layer in self.target_layers:
            if layer not in self.directions:
                continue

            for control_type in CONTROL_INSTRUCTIONS.keys():
                logger.info(f"\nCondition: Layer {layer}, {control_type}")

                result = self.run_condition(layer, control_type, test_games)
                if result is None:
                    continue

                all_results.append(result)

                logger.info(f"  Mean effect: {result['mean_effect']:.4f}")
                logger.info(f"  Cohen's d: {result['cohens_d']:.3f}")
                logger.info(f"  Safer rate: {result['safer_rate']:.2%}")
                logger.info(f"  Stop rate: {result['stop_rate']:.2%}")

        # Analyze and save results
        self._save_results(all_results)

        logger.info("\nExperiment B complete!")

    def _save_results(self, all_results: List[Dict]):
        """Analyze and save experiment results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Compute summaries
        summary = {
            'timestamp': timestamp,
            'config': {
                'target_layers': self.target_layers,
                'n_trials': self.n_trials
            },
            'by_control_type': {},
            'by_layer': {}
        }

        # By control type
        for control_type in CONTROL_INSTRUCTIONS.keys():
            type_results = [r for r in all_results if r['control_type'] == control_type]
            if type_results:
                summary['by_control_type'][control_type] = {
                    'mean_effect': float(np.mean([r['mean_effect'] for r in type_results])),
                    'mean_cohens_d': float(np.mean([r['cohens_d'] for r in type_results])),
                    'mean_safer_rate': float(np.mean([r['safer_rate'] for r in type_results])),
                    'mean_stop_rate': float(np.mean([r['stop_rate'] for r in type_results]))
                }

        # By layer
        for layer in self.target_layers:
            layer_results = [r for r in all_results if r['layer'] == layer]
            if layer_results:
                summary['by_layer'][str(layer)] = {
                    'mean_effect': float(np.mean([r['mean_effect'] for r in layer_results])),
                    'mean_cohens_d': float(np.mean([r['cohens_d'] for r in layer_results]))
                }

        # Key findings
        logger.info("\n" + "=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)

        logger.info("\nControl Effect by Type:")
        for ctrl_type, stats in summary['by_control_type'].items():
            logger.info(f"  {ctrl_type}:")
            logger.info(f"    Mean effect: {stats['mean_effect']:.4f}")
            logger.info(f"    Cohen's d: {stats['mean_cohens_d']:.3f}")
            logger.info(f"    Safer rate: {stats['mean_safer_rate']:.2%}")
            logger.info(f"    Stop rate: {stats['mean_stop_rate']:.2%}")

        # Compare explicit vs baseline
        if 'explicit' in summary['by_control_type'] and 'baseline' in summary['by_control_type']:
            explicit = summary['by_control_type']['explicit']
            baseline = summary['by_control_type']['baseline']

            effect_diff = explicit['mean_effect'] - baseline['mean_effect']
            stop_diff = explicit['mean_stop_rate'] - baseline['mean_stop_rate']

            logger.info(f"\nExplicit vs Baseline:")
            logger.info(f"  Effect difference: {effect_diff:.4f}")
            logger.info(f"  Stop rate improvement: {stop_diff:.2%}")

        # Save results
        results_file = self.output_dir / "self_control_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'conditions': [
                    {k: v for k, v in r.items() if k != 'trials'}
                    for r in all_results
                ]
            }, f, indent=2)
        logger.info(f"\nSaved: {results_file}")

        # Save full results
        full_results_file = self.output_dir / "self_control_results_full.json"
        with open(full_results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'all_results': all_results
            }, f, indent=2)
        logger.info(f"Saved: {full_results_file}")


def main():
    parser = argparse.ArgumentParser(description='Experiment B: Self-Control Capacity')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--layers', type=str, default=None, help='Comma-separated layers')
    parser.add_argument('--trials', type=int, default=None, help='Trials per condition')

    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(',')]

    exp = ExperimentB(
        config_path=args.config,
        gpu_id=args.gpu,
        target_layers=target_layers
    )

    if args.trials:
        exp.n_trials = args.trials

    exp.run()


if __name__ == '__main__':
    main()
