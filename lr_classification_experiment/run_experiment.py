#!/usr/bin/env python3
"""
LR Classification Experiment Runner

Main script to run the logistic regression classification experiment.

Usage:
    # Run Option B (core) with Gemma
    python run_experiment.py --model gemma --option B --gpu 0

    # Run all options with both models
    python run_experiment.py --model all --option all --gpu 0

    # Quick test with fewer layers
    python run_experiment.py --model gemma --option B --gpu 0 --quick
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_reconstruction import load_experiment_data, get_prompts_and_labels
from hidden_state_extractor import HiddenStateExtractor, save_hidden_states, load_hidden_states
from lr_classifier import LRClassifier, save_experiment_result, print_summary


def load_config(config_path: str = None) -> dict:
    """Load configuration."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_single_experiment(
    model_name: str,
    option: str,
    config: dict,
    gpu_id: int = 0,
    layers: list = None,
    skip_extraction: bool = False
):
    """
    Run experiment for a single model and option.

    Args:
        model_name: 'gemma' or 'llama'
        option: 'A', 'B', or 'C'
        config: Configuration dict
        gpu_id: GPU to use
        layers: List of layers (None = use config)
        skip_extraction: If True, load cached hidden states
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
    print(f"\n[1/3] Loading data: {data_path}")
    data = load_experiment_data(str(data_path))
    prompts, labels, states = get_prompts_and_labels(data, option)

    n_bankrupt = sum(labels)
    print(f"  Samples: {len(prompts)}")
    print(f"  Bankruptcy: {n_bankrupt} ({n_bankrupt/len(labels)*100:.1f}%)")
    print(f"  Voluntary: {len(labels) - n_bankrupt} ({(len(labels)-n_bankrupt)/len(labels)*100:.1f}%)")

    # 2. Extract or load hidden states
    if skip_extraction and hidden_cache_path.exists():
        print(f"\n[2/3] Loading cached hidden states: {hidden_cache_path}")
        hidden_states, cached_labels, metadata = load_hidden_states(str(hidden_cache_path))
        # Verify labels match
        if not all(cached_labels == labels):
            print("  WARNING: Cached labels don't match! Re-extracting...")
            skip_extraction = False

    if not skip_extraction or not hidden_cache_path.exists():
        print(f"\n[2/3] Extracting hidden states...")

        # Setup GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

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
    print(f"\n[3/3] Running LR classification...")

    lr_config = config['experiment']['lr']
    classifier = LRClassifier(
        max_iter=lr_config['max_iter'],
        solver=lr_config['solver'],
        class_weight=lr_config['class_weight'],
        random_state=lr_config['random_state'],
        test_size=lr_config['test_size']
    )

    import numpy as np
    labels_array = np.array(labels)

    result = classifier.run_all_layers(
        hidden_states,
        labels_array,
        model_name,
        option
    )

    # Save result
    result_path = output_dir / f"lr_result_{model_name}_option{option}_{timestamp}.json"
    save_experiment_result(result, str(result_path))

    # Print summary
    print_summary(result)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='LR Classification Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_experiment.py --model gemma --option B --gpu 0
    python run_experiment.py --model all --option all --gpu 0
    python run_experiment.py --model gemma --option B --gpu 0 --quick
    python run_experiment.py --model gemma --option B --skip-extraction
        """
    )

    parser.add_argument('--model', type=str, default='gemma',
                       choices=['gemma', 'llama', 'all'],
                       help='Model to use')
    parser.add_argument('--option', type=str, default='B',
                       choices=['A', 'B', 'C', 'all'],
                       help='Experiment option')
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

    # Run experiments
    results = []
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
                    skip_extraction=args.skip_extraction
                )
                results.append(result)
            except Exception as e:
                print(f"ERROR in {model}/{option}: {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for r in results:
        print(f"{r.model_name.upper()} Option {r.option}: "
              f"Best Layer {r.best_layer}, Accuracy {r.best_accuracy:.3f}")

    print("="*70)
    print("Experiment complete!")


if __name__ == '__main__':
    main()
