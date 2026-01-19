#!/usr/bin/env python3
"""
Gemma Metacognitive Experiment Runner

Run metacognitive experiments based on:
"Language Models Are Capable of Metacognitive Monitoring and Control"
(Ji-An et al., 2025)

Experiments:
    A: Metacognitive Reporting Accuracy
    B: Self-Control Capacity
    C: Awareness-Behavior Gap Analysis

Usage:
    # Run all experiments
    python run_experiments.py --gpu 0 --experiments all

    # Run specific experiments
    python run_experiments.py --gpu 0 --experiments a,b

    # Quick test with subset
    python run_experiments.py --gpu 0 --experiments a --layers 25,30 --trials 10
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config import load_config, get_output_dir
from utils import logger


def run_experiment_a(config_path: str, gpu_id: int, layers: Optional[List[int]] = None):
    """Run Experiment A: Metacognitive Reporting."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING EXPERIMENT A: METACOGNITIVE REPORTING")
    logger.info("=" * 70)

    from expA_metacognitive_reporting import ExperimentA

    exp = ExperimentA(
        config_path=config_path,
        gpu_id=gpu_id,
        target_layers=layers
    )
    exp.run()


def run_experiment_b(config_path: str, gpu_id: int, layers: Optional[List[int]] = None):
    """Run Experiment B: Self-Control Capacity."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING EXPERIMENT B: SELF-CONTROL CAPACITY")
    logger.info("=" * 70)

    from expB_self_control import ExperimentB

    exp = ExperimentB(
        config_path=config_path,
        gpu_id=gpu_id,
        target_layers=layers
    )
    exp.run()


def run_experiment_c(config_path: str, gpu_id: int, layers: Optional[List[int]] = None):
    """Run Experiment C: Awareness-Behavior Gap."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING EXPERIMENT C: AWARENESS-BEHAVIOR GAP")
    logger.info("=" * 70)

    from expC_gap_analysis import ExperimentC

    exp = ExperimentC(
        config_path=config_path,
        gpu_id=gpu_id,
        target_layers=layers
    )
    exp.run()


def main():
    parser = argparse.ArgumentParser(
        description='Gemma Metacognitive Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments
    python run_experiments.py --gpu 0 --experiments all

    # Run experiments A and B only
    python run_experiments.py --gpu 0 --experiments a,b

    # Quick test with subset of layers
    python run_experiments.py --gpu 0 --experiments a --layers 25,30

    # Run with custom config
    python run_experiments.py --gpu 0 --config /path/to/config.yaml
        """
    )

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--experiments', type=str, default='all',
                       help='Experiments to run: "all" or comma-separated (e.g., "a,b,c")')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers (e.g., "25,30,35")')

    args = parser.parse_args()

    # Parse experiments
    if args.experiments.lower() == 'all':
        experiments = ['a', 'b', 'c']
    else:
        experiments = [e.strip().lower() for e in args.experiments.split(',')]

    # Validate experiments
    valid_experiments = {'a', 'b', 'c'}
    for exp in experiments:
        if exp not in valid_experiments:
            logger.error(f"Invalid experiment: {exp}. Valid: a, b, c")
            return

    # Parse layers
    target_layers = None
    if args.layers:
        target_layers = [int(l.strip()) for l in args.layers.split(',')]

    # Config path
    if args.config is None:
        config_path = str(Path(__file__).parent / "configs" / "experiment_config.yaml")
    else:
        config_path = args.config

    # Log start
    logger.info("=" * 70)
    logger.info("GEMMA METACOGNITIVE EXPERIMENTS")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Experiments: {experiments}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Config: {config_path}")
    if target_layers:
        logger.info(f"Target layers: {target_layers}")

    # Run experiments
    start_time = datetime.now()

    try:
        for exp in experiments:
            exp_start = datetime.now()

            if exp == 'a':
                run_experiment_a(config_path, args.gpu, target_layers)
            elif exp == 'b':
                run_experiment_b(config_path, args.gpu, target_layers)
            elif exp == 'c':
                run_experiment_c(config_path, args.gpu, target_layers)

            exp_time = datetime.now() - exp_start
            logger.info(f"Experiment {exp.upper()} completed in {exp_time}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Log completion
    total_time = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print output locations
    config = load_config(config_path)
    output_dir = get_output_dir(config)
    logger.info(f"\nOutput directory: {output_dir}")
    for exp in experiments:
        exp_dir = output_dir / f"experiment_{exp}"
        logger.info(f"  Experiment {exp.upper()}: {exp_dir}")


if __name__ == '__main__':
    main()
