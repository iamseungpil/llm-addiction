#!/usr/bin/env python3
"""
LR Classification Experiment Runner

Main script to run the logistic regression classification experiment.
Includes group analysis (bet_type, prompt_combo) and baseline comparisons.

Usage:
    # Run Option B (core) with Gemma
    python run_experiment.py --model gemma --option B --gpu 0

    # Run all options with both models
    python run_experiment.py --model all --option all --gpu 0

    # Quick test with fewer layers
    python run_experiment.py --model gemma --option B --gpu 0 --quick

    # Skip extraction, use cached hidden states
    python run_experiment.py --model gemma --option B --skip-extraction

    # Run baselines only (no hidden state extraction)
    python run_experiment.py --model gemma --option B --baselines-only

Author: LLM Addiction Research Team
Last Updated: 2025-01
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_reconstruction import (
    load_experiment_data, get_prompts_and_labels, get_metadata_features,
    filter_by_condition, get_unique_conditions, GameState
)
from lr_classifier import (
    LRClassifier, ExperimentResult, GroupResult, LayerResult,
    save_experiment_result, print_layer_summary, print_experiment_summary
)
from baselines import (
    run_all_baselines, print_baseline_comparison, AllBaselinesResult
)


def load_config(config_path: str = None) -> dict:
    """Load configuration."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_group_hidden_states(
    hidden_states: Dict[int, np.ndarray],
    indices: List[int]
) -> Dict[int, np.ndarray]:
    """Extract hidden states for specific sample indices."""
    return {
        layer: states[indices]
        for layer, states in hidden_states.items()
    }


def run_group_analysis(
    classifier: LRClassifier,
    hidden_states: Dict[int, np.ndarray],
    labels: np.ndarray,
    states: List[GameState],
    verbose: bool = True
) -> Dict[str, Dict[str, GroupResult]]:
    """
    Run LR classification for all groups (bet_type, prompt_combo).

    Args:
        classifier: LRClassifier instance
        hidden_states: Dict of layer -> array [n_samples, d_model]
        labels: Array of labels
        states: List of GameState objects
        verbose: Print progress

    Returns:
        Dict with 'bet_type' and 'prompt_combo' group results
    """
    conditions = get_unique_conditions(states)

    results = {
        'bet_type': {},
        'prompt_combo': {}
    }

    # Analyze by bet_type
    if verbose:
        print("\n" + "-"*70)
        print("GROUP ANALYSIS: bet_type")
        print("-"*70)

    for bt in conditions['bet_types']:
        # Get indices for this bet_type
        indices = [i for i, s in enumerate(states) if s.bet_type == bt]

        if len(indices) < 10:
            if verbose:
                print(f"\n  {bt}: SKIPPED (only {len(indices)} samples)")
            continue

        group_labels = labels[indices]
        group_hidden = get_group_hidden_states(hidden_states, indices)

        group_result = classifier.run_all_layers(
            group_hidden,
            group_labels,
            group_name="bet_type",
            group_value=bt,
            verbose=verbose
        )
        results['bet_type'][bt] = group_result

    # Analyze by prompt_combo
    if verbose:
        print("\n" + "-"*70)
        print("GROUP ANALYSIS: prompt_combo")
        print("-"*70)

    for pc in conditions['prompt_combos']:
        indices = [i for i, s in enumerate(states) if s.prompt_combo == pc]

        if len(indices) < 10:
            if verbose:
                print(f"\n  {pc}: SKIPPED (only {len(indices)} samples)")
            continue

        # Check class balance
        group_labels = labels[indices]
        n_pos = int(group_labels.sum())
        n_neg = len(group_labels) - n_pos

        if n_pos < 2 or n_neg < 2:
            if verbose:
                print(f"\n  {pc}: SKIPPED (insufficient class balance: pos={n_pos}, neg={n_neg})")
            continue

        group_hidden = get_group_hidden_states(hidden_states, indices)

        group_result = classifier.run_all_layers(
            group_hidden,
            group_labels,
            group_name="prompt_combo",
            group_value=pc,
            verbose=verbose
        )
        results['prompt_combo'][pc] = group_result

    return results


def run_single_experiment(
    model_name: str,
    option: str,
    config: dict,
    gpu_id: int = 0,
    layers: Optional[List[int]] = None,
    skip_extraction: bool = False,
    run_baselines: bool = True,
    run_groups: bool = True,
    baselines_only: bool = False
) -> Dict:
    """
    Run experiment for a single model and option.

    Args:
        model_name: 'gemma' or 'llama'
        option: 'A', 'B', or 'C'
        config: Configuration dict
        gpu_id: GPU to use
        layers: List of layers (None = use config)
        skip_extraction: If True, load cached hidden states
        run_baselines: If True, run baseline comparisons
        run_groups: If True, run bet_type and prompt_combo group analysis
        baselines_only: If True, only run baselines (no hidden state extraction)

    Returns:
        Dict with experiment results
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT: {model_name.upper()} - Option {option}")
    print("="*70)

    # Setup paths
    data_root = Path(config['data']['root'])
    data_path = data_root / config['data'][model_name]
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Hidden states cache path
    hidden_cache_path = output_dir / f"hidden_states_{model_name}_option{option}.npz"

    # 1. Load experiment data and reconstruct prompts
    print(f"\n[1/4] Loading data: {data_path}")
    data = load_experiment_data(str(data_path))
    prompts, labels, states = get_prompts_and_labels(data, option)
    labels_array = np.array(labels)

    n_bankrupt = sum(labels)
    print(f"  Samples: {len(prompts)}")
    print(f"  Bankruptcy: {n_bankrupt} ({n_bankrupt/len(labels)*100:.1f}%)")
    print(f"  Voluntary: {len(labels) - n_bankrupt} ({(len(labels)-n_bankrupt)/len(labels)*100:.1f}%)")

    # Get unique conditions
    conditions = get_unique_conditions(states)
    print(f"  bet_types: {conditions['bet_types']}")
    print(f"  prompt_combos: {len(conditions['prompt_combos'])} types")

    # Get metadata features for baseline
    metadata_features = get_metadata_features(states)

    # If baselines only, run baselines and return
    if baselines_only:
        print("\n[2/4] Hidden state extraction... SKIPPED (baselines only)")
        print("\n[3/4] LR classification... SKIPPED (baselines only)")

        print("\n[4/4] Running baseline comparisons...")
        baselines = run_all_baselines(
            labels=labels_array,
            prompts=prompts,
            metadata_features=metadata_features,
            hidden_states=None,  # No hidden states in baselines-only mode
            test_size=config['experiment']['lr']['test_size'],
            random_state=config['experiment']['lr']['random_state'],
            n_cv_folds=config['experiment']['lr'].get('n_cv_folds', 5),
            verbose=True
        )

        # Save baselines result
        baselines_path = output_dir / f"baselines_{model_name}_option{option}_{timestamp}.json"
        with open(baselines_path, 'w') as f:
            json.dump(baselines.to_dict(), f, indent=2)
        print(f"Saved: {baselines_path}")

        return {'baselines': baselines}

    # 2. Extract or load hidden states
    hidden_states = None
    cached_labels = None

    if skip_extraction and hidden_cache_path.exists():
        print(f"\n[2/4] Loading cached hidden states: {hidden_cache_path}")
        from hidden_state_extractor import load_hidden_states
        hidden_states, cached_labels, metadata = load_hidden_states(str(hidden_cache_path))
        # Verify labels match
        if not np.array_equal(cached_labels, labels_array):
            print("  WARNING: Cached labels don't match! Re-extracting...")
            hidden_states = None

    if hidden_states is None:
        print(f"\n[2/4] Extracting hidden states...")

        # Setup GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        from hidden_state_extractor import HiddenStateExtractor, save_hidden_states

        # Determine layers
        if layers is None:
            layers = config['experiment']['target_layers_quick']

        print(f"  Layers: {layers}")
        print(f"  Batch size: {config['experiment']['batch_size']}")

        # Extract
        extractor = HiddenStateExtractor(model_name, device='cuda:0')
        extractor.load_model()

        hidden_states = extractor.extract_batch(
            prompts,
            layers=layers,
            batch_size=config['experiment']['batch_size'],
            show_progress=True
        )

        extractor.unload_model()

        # Save cache
        save_hidden_states(
            hidden_states,
            labels,
            str(hidden_cache_path),
            metadata={
                'model': model_name,
                'option': option,
                'layers': layers,
                'n_samples': len(labels),
                'timestamp': timestamp
            }
        )

    # 3. Run LR classification
    print(f"\n[3/4] Running LR classification...")

    lr_config = config['experiment']['lr']
    classifier = LRClassifier(
        max_iter=lr_config['max_iter'],
        solver=lr_config['solver'],
        class_weight=lr_config['class_weight'],
        random_state=lr_config['random_state'],
        test_size=lr_config['test_size'],
        n_cv_folds=lr_config.get('n_cv_folds', 5)
    )

    # Main result (all data)
    print("\n  Main (all data):")
    main_result = classifier.run_all_layers(
        hidden_states,
        labels_array,
        group_name="all",
        group_value="all",
        verbose=True
    )

    # Create experiment result
    experiment_result = ExperimentResult(
        model_name=model_name,
        option=option,
        timestamp=timestamp,
        n_total_samples=len(labels),
        n_total_positive=n_bankrupt,
        main_result=main_result
    )

    # Group analysis
    if run_groups:
        group_results = run_group_analysis(
            classifier,
            hidden_states,
            labels_array,
            states,
            verbose=True
        )
        experiment_result.bet_type_results = group_results['bet_type']
        experiment_result.prompt_combo_results = group_results['prompt_combo']

    # Save LR result
    result_path = output_dir / f"lr_result_{model_name}_option{option}_{timestamp}.json"
    save_experiment_result(experiment_result, str(result_path))

    # Print summary
    print_experiment_summary(experiment_result)

    # 4. Run baselines
    baselines = None
    if run_baselines:
        print("\n[4/4] Running baseline comparisons...")
        baselines = run_all_baselines(
            labels=labels_array,
            prompts=prompts,
            metadata_features=metadata_features,
            hidden_states=hidden_states,
            test_size=lr_config['test_size'],
            random_state=lr_config['random_state'],
            n_cv_folds=lr_config.get('n_cv_folds', 5),
            verbose=True
        )

        # Get best LR metrics
        best_lr = main_result.layer_results[main_result.best_layer]
        print_baseline_comparison(
            baselines,
            lr_best_accuracy=best_lr.metrics.accuracy,
            lr_best_auc=best_lr.metrics.auc_roc
        )

        # Save baselines result
        baselines_path = output_dir / f"baselines_{model_name}_option{option}_{timestamp}.json"
        with open(baselines_path, 'w') as f:
            json.dump(baselines.to_dict(), f, indent=2)
        print(f"Saved: {baselines_path}")
    else:
        print("\n[4/4] Baselines... SKIPPED")

    return {
        'experiment': experiment_result,
        'baselines': baselines
    }


def main():
    parser = argparse.ArgumentParser(
        description='LR Classification Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Core experiment (Option B)
    python run_experiment.py --model gemma --option B --gpu 0

    # Quick test with fewer layers
    python run_experiment.py --model gemma --option B --gpu 0 --quick

    # Full analysis with all layers
    python run_experiment.py --model gemma --option B --gpu 0 --full

    # Use cached hidden states
    python run_experiment.py --model gemma --option B --skip-extraction

    # Run baselines only (no GPU needed)
    python run_experiment.py --model gemma --option B --baselines-only

    # All options and models
    python run_experiment.py --model all --option all --gpu 0
        """
    )

    parser.add_argument('--model', type=str, default='gemma',
                       choices=['gemma', 'llama', 'all'],
                       help='Model to use')
    parser.add_argument('--option', type=str, default='B',
                       choices=['A', 'B', 'C', 'all'],
                       help='Experiment option (B=core)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer layers')
    parser.add_argument('--full', action='store_true',
                       help='Full mode: all layers')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer indices')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip extraction, use cached hidden states')
    parser.add_argument('--no-baselines', action='store_true',
                       help='Skip baseline comparisons')
    parser.add_argument('--no-groups', action='store_true',
                       help='Skip group analysis (bet_type, prompt_combo)')
    parser.add_argument('--baselines-only', action='store_true',
                       help='Run baselines only (no hidden state extraction)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine layers
    if args.layers:
        layers = [int(x) for x in args.layers.split(',')]
    elif args.quick:
        layers = config['experiment']['target_layers_quick']
    elif args.full:
        layers = None  # Will be set per model
    else:
        layers = config['experiment']['target_layers_quick']

    # Determine models and options
    models = ['gemma', 'llama'] if args.model == 'all' else [args.model]
    options = ['A', 'B', 'C'] if args.option == 'all' else [args.option]

    print("="*70)
    print("LR CLASSIFICATION EXPERIMENT")
    print("="*70)
    print(f"Models: {models}")
    print(f"Options: {options}")
    print(f"GPU: {args.gpu}")
    print(f"Layers: {layers if layers else 'all'}")
    print(f"Skip extraction: {args.skip_extraction}")
    print(f"Run baselines: {not args.no_baselines}")
    print(f"Run groups: {not args.no_groups}")
    print(f"Baselines only: {args.baselines_only}")

    # Run experiments
    all_results = []
    for model in models:
        # Set all layers for full mode
        model_layers = layers
        if args.full:
            n_layers = config['models'][model]['n_layers']
            model_layers = list(range(0, n_layers, 2))  # Every other layer

        for option in options:
            try:
                result = run_single_experiment(
                    model_name=model,
                    option=option,
                    config=config,
                    gpu_id=args.gpu,
                    layers=model_layers,
                    skip_extraction=args.skip_extraction,
                    run_baselines=not args.no_baselines,
                    run_groups=not args.no_groups,
                    baselines_only=args.baselines_only
                )
                all_results.append((model, option, result))
            except Exception as e:
                print(f"\nERROR in {model}/{option}: {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for model, option, result in all_results:
        if 'experiment' in result and result['experiment']:
            exp = result['experiment']
            print(f"{model.upper()} Option {option}: "
                  f"Best Layer {exp.main_result.best_layer}, "
                  f"Accuracy {exp.main_result.best_accuracy:.3f}")
        elif 'baselines' in result:
            bl = result['baselines']
            print(f"{model.upper()} Option {option}: "
                  f"Chance={bl.chance.accuracy:.3f}, "
                  f"Text={bl.text_only.accuracy if bl.text_only else 'N/A':.3f}, "
                  f"Meta={bl.metadata.accuracy if bl.metadata else 'N/A':.3f}")

    print("="*70)
    print("Experiment complete!")


if __name__ == '__main__':
    main()
