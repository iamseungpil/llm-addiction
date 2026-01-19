#!/usr/bin/env python3
"""
Experiment A: Metacognitive Reporting Accuracy

Test whether LLMs can accurately report their addiction-related activation states
using in-context learning (ICL) neurofeedback paradigm.

Reference: Ji-An et al., 2025 - "Language Models Are Capable of Metacognitive
Monitoring and Control of Their Internal Activations"

Method:
1. Compute neural directions (Contrastive, LR, PCA)
2. Create ICL examples with activation-based labels
3. Test LLM's ability to predict its own activation labels

Usage:
    python expA_metacognitive_reporting.py --gpu 0
    python expA_metacognitive_reporting.py --gpu 0 --n-examples 16 --layers 25,30
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
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, get_experiment_output_dir
from utils import (
    GemmaBaseModel,
    compute_contrastive_direction, compute_lr_direction, compute_pca_directions,
    project_onto_direction, binarize_projection,
    create_neurofeedback_prompt, parse_risk_label,
    load_experiment_data, group_by_outcome, extract_context_description,
    clear_gpu_memory, logger, DirectionResult
)


class ExperimentA:
    """
    Metacognitive Reporting Accuracy Experiment.

    Tests whether LLMs can accurately report their own activation states
    when presented with ICL examples of activation-label pairs.
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

        # Experiment A config
        exp_config = self.config.get('experiment_a', {})
        self.n_examples_list = exp_config.get('n_examples', [0, 4, 16, 64, 256])
        self.n_trials = exp_config.get('n_trials', 50)
        self.direction_types = exp_config.get('direction_types', ['contrastive', 'lr', 'pca'])

        # Output directory
        self.output_dir = get_experiment_output_dir(self.config, 'a')

        # Models and data
        self.model = None
        self.directions = {}  # layer -> direction_type -> DirectionResult
        self.examples_pool = []  # [(context, hidden_state, label)]

    def load_model(self):
        """Load Gemma Base model."""
        model_config = self.config.get('model', {})
        self.model = GemmaBaseModel(
            model_id=model_config.get('model_id', 'google/gemma-2-9b'),
            device=self.device
        )
        self.model.load()

    def prepare_directions_and_examples(self):
        """
        Prepare neural directions and ICL example pool from gambling data.
        """
        logger.info("Preparing directions and examples...")

        # Load experiment data
        data_path = self.config.get('data', {}).get('experiment_data')
        exp_data = load_experiment_data(data_path)
        games = exp_data['results']

        # Group by outcome
        grouped = group_by_outcome(games)
        bankrupt_games = grouped['bankruptcy']
        safe_games = grouped['voluntary_stop']

        logger.info(f"Games: {len(bankrupt_games)} bankrupt, {len(safe_games)} safe")

        # Sample games for hidden state extraction
        max_samples = min(500, len(bankrupt_games), len(safe_games))

        random.seed(self.config.get('random_seed', 42))
        sampled_bankrupt = random.sample(bankrupt_games, min(max_samples, len(bankrupt_games)))
        sampled_safe = random.sample(safe_games, min(max_samples, len(safe_games)))

        # Extract hidden states for each layer
        for layer in self.target_layers:
            logger.info(f"\nProcessing Layer {layer}")

            # Extract hidden states
            bankrupt_hiddens = []
            safe_hiddens = []
            bankrupt_contexts = []
            safe_contexts = []

            logger.info("  Extracting bankrupt hidden states...")
            for game in tqdm(sampled_bankrupt, desc="Bankrupt"):
                context = extract_context_description(game)
                prompt = f"Gambling scenario: {context}. What should I do?"

                try:
                    h = self.model.get_hidden_states(prompt, [layer], position='last')
                    if layer in h:
                        bankrupt_hiddens.append(h[layer].numpy().flatten())
                        bankrupt_contexts.append(context)
                except Exception as e:
                    logger.warning(f"Error: {e}")
                    continue

                if len(bankrupt_hiddens) % 50 == 0:
                    clear_gpu_memory()

            logger.info("  Extracting safe hidden states...")
            for game in tqdm(sampled_safe, desc="Safe"):
                context = extract_context_description(game)
                prompt = f"Gambling scenario: {context}. What should I do?"

                try:
                    h = self.model.get_hidden_states(prompt, [layer], position='last')
                    if layer in h:
                        safe_hiddens.append(h[layer].numpy().flatten())
                        safe_contexts.append(context)
                except Exception as e:
                    logger.warning(f"Error: {e}")
                    continue

                if len(safe_hiddens) % 50 == 0:
                    clear_gpu_memory()

            if len(bankrupt_hiddens) < 10 or len(safe_hiddens) < 10:
                logger.warning(f"Insufficient samples for layer {layer}")
                continue

            bankrupt_hiddens = np.array(bankrupt_hiddens)
            safe_hiddens = np.array(safe_hiddens)

            # Compute directions
            self.directions[layer] = {}

            # 1. Contrastive direction
            if 'contrastive' in self.direction_types:
                contrastive_dir = compute_contrastive_direction(bankrupt_hiddens, safe_hiddens)
                self.directions[layer]['contrastive'] = contrastive_dir
                logger.info(f"  Contrastive: magnitude={contrastive_dir.metadata['magnitude']:.4f}")

            # 2. LR direction
            if 'lr' in self.direction_types:
                all_hiddens = np.vstack([bankrupt_hiddens, safe_hiddens])
                labels = np.array([1] * len(bankrupt_hiddens) + [0] * len(safe_hiddens))
                lr_dir = compute_lr_direction(all_hiddens, labels)
                self.directions[layer]['lr'] = lr_dir
                logger.info(f"  LR: accuracy={lr_dir.metadata['accuracy']:.3f}")

            # 3. PCA direction
            if 'pca' in self.direction_types:
                all_hiddens = np.vstack([bankrupt_hiddens, safe_hiddens])
                _, pca_dir = compute_pca_directions(all_hiddens, n_components=128)
                self.directions[layer]['pca'] = pca_dir
                logger.info(f"  PCA: variance={pca_dir.metadata['total_variance_explained']:.3f}")

            # Create example pool for this layer
            # Use contrastive direction for labeling
            ref_dir = self.directions[layer].get('contrastive', list(self.directions[layer].values())[0])

            layer_examples = []

            for context, hidden in zip(bankrupt_contexts, bankrupt_hiddens):
                proj = project_onto_direction(hidden, ref_dir.direction)
                label = binarize_projection(proj)
                layer_examples.append((context, hidden, label, proj))

            for context, hidden in zip(safe_contexts, safe_hiddens):
                proj = project_onto_direction(hidden, ref_dir.direction)
                label = binarize_projection(proj)
                layer_examples.append((context, hidden, label, proj))

            self.examples_pool.append({
                'layer': layer,
                'examples': layer_examples
            })

            logger.info(f"  Examples: {len(layer_examples)}")

        logger.info("\nDirections and examples prepared!")

    def run_trial(
        self,
        layer: int,
        direction_type: str,
        n_examples: int,
        test_context: str,
        test_hidden: np.ndarray
    ) -> Dict:
        """
        Run a single metacognitive reporting trial.

        Args:
            layer: Target layer
            direction_type: Type of direction to use
            n_examples: Number of ICL examples
            test_context: Test context string
            test_hidden: Test hidden state

        Returns:
            Trial result dict
        """
        direction = self.directions[layer][direction_type]

        # Get true label from projection
        true_projection = project_onto_direction(test_hidden, direction.direction)
        true_label = binarize_projection(true_projection)

        if n_examples == 0:
            # Zero-shot: just ask for prediction
            prompt = f'Predict the risk label for this gambling context.\nContext: "{test_context}"\nRisk Label (HIGH_RISK or LOW_RISK):'
        else:
            # Get examples from pool
            layer_data = next((d for d in self.examples_pool if d['layer'] == layer), None)
            if layer_data is None:
                return None

            # Sample examples (excluding test)
            all_examples = [(ex[0], ex[2]) for ex in layer_data['examples']
                           if ex[0] != test_context]
            random.shuffle(all_examples)
            examples = all_examples[:n_examples]

            # Create prompt
            prompt = create_neurofeedback_prompt(examples, test_context, n_examples)

        # Get model prediction
        try:
            response = self.model.generate_response(prompt, max_new_tokens=20, temperature=0.1)
            predicted_label = parse_risk_label(response)
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return None

        # Check correctness
        correct = predicted_label == true_label if predicted_label else False

        return {
            'layer': layer,
            'direction_type': direction_type,
            'n_examples': n_examples,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': correct,
            'true_projection': float(true_projection),
            'response': response[:200]
        }

    def run_condition(
        self,
        layer: int,
        direction_type: str,
        n_examples: int
    ) -> Dict:
        """
        Run all trials for a condition (layer × direction × n_examples).
        """
        results = []

        # Get test examples
        layer_data = next((d for d in self.examples_pool if d['layer'] == layer), None)
        if layer_data is None:
            return {'trials': [], 'accuracy': 0}

        # Sample test cases
        test_examples = random.sample(layer_data['examples'], min(self.n_trials, len(layer_data['examples'])))

        for context, hidden, _, _ in tqdm(test_examples, desc=f"L{layer}/{direction_type}/N={n_examples}"):
            result = self.run_trial(layer, direction_type, n_examples, context, hidden)
            if result:
                results.append(result)

            if len(results) % 10 == 0:
                clear_gpu_memory()

        # Compute accuracy
        n_correct = sum(1 for r in results if r['correct'])
        accuracy = n_correct / len(results) if results else 0

        return {
            'layer': layer,
            'direction_type': direction_type,
            'n_examples': n_examples,
            'n_trials': len(results),
            'n_correct': n_correct,
            'accuracy': accuracy,
            'trials': results
        }

    def run(self):
        """Run Experiment A: Metacognitive Reporting."""
        logger.info("=" * 70)
        logger.info("EXPERIMENT A: METACOGNITIVE REPORTING ACCURACY")
        logger.info("=" * 70)
        logger.info(f"Target layers: {self.target_layers}")
        logger.info(f"Direction types: {self.direction_types}")
        logger.info(f"N examples: {self.n_examples_list}")
        logger.info(f"Trials per condition: {self.n_trials}")

        # Load model
        self.load_model()

        # Prepare directions and examples
        self.prepare_directions_and_examples()

        # Run all conditions
        all_results = []

        total_conditions = len(self.target_layers) * len(self.direction_types) * len(self.n_examples_list)
        logger.info(f"\nTotal conditions: {total_conditions}")

        for layer in self.target_layers:
            if layer not in self.directions:
                continue

            for direction_type in self.direction_types:
                if direction_type not in self.directions[layer]:
                    continue

                for n_examples in self.n_examples_list:
                    logger.info(f"\nCondition: Layer {layer}, {direction_type}, N={n_examples}")

                    result = self.run_condition(layer, direction_type, n_examples)
                    all_results.append(result)

                    logger.info(f"  Accuracy: {result['accuracy']:.2%}")

        # Analyze and save results
        self._save_results(all_results)

        logger.info("\nExperiment A complete!")

    def _save_results(self, all_results: List[Dict]):
        """Analyze and save experiment results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Organize results
        results_by_condition = defaultdict(list)
        for r in all_results:
            key = (r['direction_type'], r['n_examples'])
            results_by_condition[key].append(r['accuracy'])

        # Compute summaries
        summary = {
            'timestamp': timestamp,
            'config': {
                'target_layers': self.target_layers,
                'direction_types': self.direction_types,
                'n_examples_list': self.n_examples_list,
                'n_trials': self.n_trials
            },
            'by_direction': {},
            'by_n_examples': {},
            'by_layer': {}
        }

        # By direction type
        for dir_type in self.direction_types:
            dir_results = [r for r in all_results if r['direction_type'] == dir_type]
            if dir_results:
                summary['by_direction'][dir_type] = {
                    'mean_accuracy': float(np.mean([r['accuracy'] for r in dir_results])),
                    'std_accuracy': float(np.std([r['accuracy'] for r in dir_results]))
                }

        # By N examples
        for n_ex in self.n_examples_list:
            n_results = [r for r in all_results if r['n_examples'] == n_ex]
            if n_results:
                summary['by_n_examples'][str(n_ex)] = {
                    'mean_accuracy': float(np.mean([r['accuracy'] for r in n_results])),
                    'std_accuracy': float(np.std([r['accuracy'] for r in n_results]))
                }

        # By layer
        for layer in self.target_layers:
            layer_results = [r for r in all_results if r['layer'] == layer]
            if layer_results:
                summary['by_layer'][str(layer)] = {
                    'mean_accuracy': float(np.mean([r['accuracy'] for r in layer_results])),
                    'std_accuracy': float(np.std([r['accuracy'] for r in layer_results]))
                }

        # Key findings
        logger.info("\n" + "=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)

        logger.info("\nAccuracy by Direction Type:")
        for dir_type, stats in summary['by_direction'].items():
            logger.info(f"  {dir_type}: {stats['mean_accuracy']:.2%} (+/- {stats['std_accuracy']:.2%})")

        logger.info("\nAccuracy by N Examples:")
        for n_ex, stats in summary['by_n_examples'].items():
            logger.info(f"  N={n_ex}: {stats['mean_accuracy']:.2%}")

        logger.info("\nAccuracy by Layer:")
        for layer, stats in summary['by_layer'].items():
            logger.info(f"  L{layer}: {stats['mean_accuracy']:.2%}")

        # Save results
        results_file = self.output_dir / "reporting_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': summary,
                'conditions': [
                    {k: v for k, v in r.items() if k != 'trials'}
                    for r in all_results
                ]
            }, f, indent=2)
        logger.info(f"\nSaved: {results_file}")

        # Save full results with trials
        full_results_file = self.output_dir / "reporting_results_full.json"
        with open(full_results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'all_results': all_results
            }, f, indent=2)
        logger.info(f"Saved: {full_results_file}")

        # Save direction metadata
        directions_file = self.output_dir / "directions_metadata.json"
        dir_metadata = {}
        for layer, dirs in self.directions.items():
            dir_metadata[str(layer)] = {
                dt: dr.metadata for dt, dr in dirs.items()
            }
        with open(directions_file, 'w') as f:
            json.dump(dir_metadata, f, indent=2)
        logger.info(f"Saved: {directions_file}")


def main():
    parser = argparse.ArgumentParser(description='Experiment A: Metacognitive Reporting')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--layers', type=str, default=None, help='Comma-separated layers')
    parser.add_argument('--n-examples', type=str, default=None,
                       help='Comma-separated N values for ICL')

    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(',')]

    exp = ExperimentA(
        config_path=args.config,
        gpu_id=args.gpu,
        target_layers=target_layers
    )

    if args.n_examples:
        exp.n_examples_list = [int(n) for n in args.n_examples.split(',')]

    exp.run()


if __name__ == '__main__':
    main()
