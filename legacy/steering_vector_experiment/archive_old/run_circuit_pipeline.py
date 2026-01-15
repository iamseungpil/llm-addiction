#!/usr/bin/env python3
"""
Main Pipeline: Sparse Feature Circuits + Self-Explaining Features

This script orchestrates the complete circuit discovery pipeline:
1. Prepare prompt pairs from experiment data
2. Discover sparse feature circuits via attribution patching
3. Generate self-explanations for causal features
4. (Optional) Validate with steering vector experiments

Usage:
    # Full pipeline for LLaMA
    python run_circuit_pipeline.py --model llama --gpu 0

    # Full pipeline for Gemma
    python run_circuit_pipeline.py --model gemma --gpu 1

    # Skip to specific phase
    python run_circuit_pipeline.py --model llama --gpu 0 --start-phase 2 --pairs prompt_pairs_llama.json
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import yaml
import subprocess

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging, get_default_config_path


def run_phase(cmd: list, logger, description: str) -> bool:
    """Run a phase command and log output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE: {description}")
    logger.info(f"{'='*60}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent)
        )

        if result.stdout:
            for line in result.stdout.split('\n')[-20:]:  # Last 20 lines
                logger.info(line)

        if result.returncode != 0:
            logger.error(f"Phase failed with return code {result.returncode}")
            if result.stderr:
                logger.error(result.stderr)
            return False

        logger.info(f"Phase completed successfully")
        return True

    except Exception as e:
        logger.error(f"Phase failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run full circuit discovery pipeline')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                       help='Model to analyze')
    parser.add_argument('--gpu', type=int, required=True,
                       help='GPU ID to use')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to config file (default: auto-detect)')
    parser.add_argument('--start-phase', type=int, default=1, choices=[1, 2, 3, 4],
                       help='Phase to start from (1=pairs, 2=circuits, 3=explain, 4=validate)')
    parser.add_argument('--pairs', type=str, default=None,
                       help='Existing prompt pairs file (for --start-phase >= 2)')
    parser.add_argument('--circuit', type=str, default=None,
                       help='Existing circuit file (for --start-phase >= 3)')
    parser.add_argument('--node-threshold', type=float, default=0.1,
                       help='Node threshold for circuit discovery')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k features per layer')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip steering vector validation phase')

    args = parser.parse_args()

    # Load config
    config_path = args.config or str(get_default_config_path())
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(__file__).parent

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(f'pipeline_{args.model}_{timestamp}', output_dir / 'logs')

    logger.info("=" * 70)
    logger.info(f"CIRCUIT DISCOVERY PIPELINE - {args.model.upper()}")
    logger.info("=" * 70)
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Start phase: {args.start_phase}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Node threshold: {args.node_threshold}")
    logger.info(f"Top-k per layer: {args.top_k}")

    # Track files across phases
    pairs_file = args.pairs
    circuit_file = args.circuit

    # =========================================================================
    # Phase 1: Prepare Prompt Pairs
    # =========================================================================
    if args.start_phase <= 1:
        pairs_file = f'prompt_pairs_{args.model}.json'

        cmd = [
            sys.executable,
            str(src_dir / 'prepare_prompt_pairs.py'),
            '--model', args.model,
            '--output', pairs_file,
            '--match-conditions'
        ]

        success = run_phase(cmd, logger, "Prepare Prompt Pairs")
        if not success:
            logger.error("Pipeline failed at Phase 1")
            return 1

    if pairs_file is None:
        pairs_file = f'prompt_pairs_{args.model}.json'
    logger.info(f"Using prompt pairs: {pairs_file}")

    # =========================================================================
    # Phase 2: Discover Sparse Feature Circuits
    # =========================================================================
    if args.start_phase <= 2:
        cmd = [
            sys.executable,
            str(src_dir / 'sparse_feature_circuits.py'),
            '--model', args.model,
            '--gpu', str(args.gpu),
            '--pairs', pairs_file,
            '--config', args.config,
            '--node-threshold', str(args.node_threshold),
            '--top-k', str(args.top_k)
        ]

        success = run_phase(cmd, logger, "Discover Sparse Feature Circuits")
        if not success:
            logger.error("Pipeline failed at Phase 2")
            return 1

        # Find the latest circuit file
        circuit_files = sorted(output_dir.glob(f'circuit_{args.model}_*.json'))
        if circuit_files:
            circuit_file = circuit_files[-1].name
            logger.info(f"Circuit file: {circuit_file}")
        else:
            logger.error("No circuit file generated")
            return 1

    if circuit_file is None:
        circuit_files = sorted(output_dir.glob(f'circuit_{args.model}_*.json'))
        if circuit_files:
            circuit_file = circuit_files[-1].name
        else:
            logger.error("No circuit file found")
            return 1
    logger.info(f"Using circuit: {circuit_file}")

    # =========================================================================
    # Phase 3: Self-Explaining Features
    # =========================================================================
    if args.start_phase <= 3:
        cmd = [
            sys.executable,
            str(src_dir / 'self_explaining_features.py'),
            '--model', args.model,
            '--gpu', str(args.gpu),
            '--circuit', circuit_file,
            '--config', args.config,
            '--top-k', str(args.top_k)
        ]

        success = run_phase(cmd, logger, "Self-Explaining Features")
        if not success:
            logger.error("Pipeline failed at Phase 3")
            return 1

    # =========================================================================
    # Phase 4: Steering Vector Validation (Optional)
    # =========================================================================
    if args.start_phase <= 4 and not args.skip_validation:
        # Check if steering experiment script exists
        steering_script = src_dir / 'run_steering_experiment.py'
        if steering_script.exists():
            cmd = [
                sys.executable,
                str(steering_script),
                '--model', args.model,
                '--gpu', str(args.gpu),
                '--config', args.config,
                '--n-trials', '20',  # Reduced for validation
                '--strengths', '-1.0,0.0,1.0'  # Reduced for validation
            ]

            success = run_phase(cmd, logger, "Steering Vector Validation")
            if not success:
                logger.warning("Steering validation failed (non-critical)")
        else:
            logger.info("Skipping steering validation (script not found)")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    # Load and summarize results
    circuit_path = output_dir / circuit_file
    if circuit_path.exists():
        with open(circuit_path, 'r') as f:
            circuit = json.load(f)

        logger.info(f"\nCircuit Summary for {args.model.upper()}:")
        logger.info(f"  Total causal features: {circuit['summary']['total_causal_features']}")
        logger.info(f"  Safe features: {circuit['summary']['safe_features']}")
        logger.info(f"  Risky features: {circuit['summary']['risky_features']}")
        logger.info(f"\n  Layer distribution:")
        for layer, dist in circuit['layer_distribution'].items():
            logger.info(f"    Layer {layer}: {dist['safe']} safe, {dist['risky']} risky")

    # List generated files
    logger.info("\nGenerated files:")
    for f in sorted(output_dir.glob(f'*_{args.model}_*.json')):
        if f.name.startswith(('prompt_pairs', 'circuit', 'explanations', 'steering')):
            logger.info(f"  {f.name}")

    logger.info("\nPipeline complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
