#!/usr/bin/env python3
"""
Gemma SAE Experiment Pipeline Runner

Run complete SAE interpretability pipeline for Gemma Base model.

Phases:
    0: SAE Boost - Train Residual SAE (MUST run before Phase 1 if using boost)
    1: SAE Feature Extraction (supports --use-boost)
    2: Correlation Analysis (Bankrupt vs Safe)
    3: Steering Vector Extraction (CAA)
    4: SAE Interpretation
    5: Steering Experiment

Usage:
    # Run all phases with SAE Boost
    python run_pipeline.py --gpu 0 --phases all --use-boost

    # Run without SAE Boost (Base SAE only)
    python run_pipeline.py --gpu 0 --phases 1,2,3,4,5

    # Run only SAE Boost (Phase 0)
    python run_pipeline.py --gpu 0 --phases 0

    # Quick test with subset of layers
    python run_pipeline.py --gpu 0 --phases 0,1,2 --layers 25,30,35 --use-boost
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from config import load_config, get_phase_output_dir
from utils import logger


def run_phase0(config_path: str, gpu_id: int, layers: Optional[List[int]] = None):
    """Run Phase 0: SAE Boost (Residual SAE Training)."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING PHASE 0: SAE BOOST")
    logger.info("=" * 70)

    from phase0_sae_boost import Phase0SAEBoost

    phase0 = Phase0SAEBoost(
        config_path=config_path,
        gpu_id=gpu_id,
        target_layers=layers
    )
    phase0.run()


def run_phase1(
    config_path: str,
    gpu_id: int,
    layers: Optional[List[int]] = None,
    use_boost: bool = False
):
    """Run Phase 1: SAE Feature Extraction."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING PHASE 1: SAE FEATURE EXTRACTION")
    if use_boost:
        logger.info("  (Using Boosted SAE)")
    logger.info("=" * 70)

    from phase1_feature_extraction import Phase1FeatureExtraction

    phase1 = Phase1FeatureExtraction(
        config_path=config_path,
        gpu_id=gpu_id,
        target_layers=layers,
        use_boost=use_boost
    )
    phase1.run(save_hidden_states=True)


def run_phase2(config_path: str, layers: Optional[List[int]] = None):
    """Run Phase 2: Correlation Analysis."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING PHASE 2: CORRELATION ANALYSIS")
    logger.info("=" * 70)

    from phase2_correlation_analysis import Phase2CorrelationAnalysis

    phase2 = Phase2CorrelationAnalysis(
        config_path=config_path,
        target_layers=layers
    )
    phase2.run()


def run_phase3(config_path: str, gpu_id: int, layers: Optional[List[int]] = None):
    """Run Phase 3: Steering Vector Extraction."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING PHASE 3: STEERING VECTOR EXTRACTION")
    logger.info("=" * 70)

    from phase3_steering_vector import Phase3SteeringVector

    phase3 = Phase3SteeringVector(
        config_path=config_path,
        gpu_id=gpu_id,
        target_layers=layers
    )
    phase3.run()


def run_phase4(config_path: str, gpu_id: int, layers: Optional[List[int]] = None):
    """Run Phase 4: SAE Interpretation."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING PHASE 4: SAE INTERPRETATION")
    logger.info("=" * 70)

    from phase4_sae_interpretation import Phase4SAEInterpretation

    phase4 = Phase4SAEInterpretation(
        config_path=config_path,
        gpu_id=gpu_id,
        target_layers=layers
    )
    phase4.run()


def run_phase5(
    config_path: str,
    gpu_id: int,
    layers: Optional[List[int]] = None,
    n_trials: Optional[int] = None
):
    """Run Phase 5: Steering Experiment."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING PHASE 5: STEERING EXPERIMENT")
    logger.info("=" * 70)

    from phase5_steering_experiment import Phase5SteeringExperiment

    phase5 = Phase5SteeringExperiment(
        config_path=config_path,
        gpu_id=gpu_id,
        target_layers=layers,
        n_trials=n_trials
    )
    phase5.run()


def check_phase_dependencies(phases: List[int], config_path: str, use_boost: bool) -> bool:
    """
    Check if required inputs exist for each phase.

    Returns:
        True if all dependencies are satisfied
    """
    config = load_config(config_path)

    # Dependency map
    dependencies = {
        0: [],   # Phase 0 uses raw experiment data
        1: [],   # Phase 1 uses raw experiment data (+ Phase 0 if use_boost)
        2: [1],  # Phase 2 needs Phase 1
        3: [],   # Phase 3 uses raw experiment data
        4: [2, 3],  # Phase 4 needs Phase 2 and 3
        5: [3],  # Phase 5 needs Phase 3
    }

    # Add Phase 0 dependency for Phase 1 if using boost
    if use_boost and 1 in phases and 0 not in phases:
        dependencies[1] = [0]

    for phase in phases:
        if phase == 0 or phase == 1 or phase == 3:
            # Check experiment data exists
            data_path = config.get('data', {}).get('experiment_data')
            if data_path and not Path(data_path).exists():
                logger.error(f"Experiment data not found: {data_path}")
                return False
            continue

        required = dependencies.get(phase, [])

        for req in required:
            req_dir = get_phase_output_dir(config, req)
            if not req_dir.exists() or not any(req_dir.iterdir()):
                logger.error(f"Phase {phase} requires Phase {req} outputs")
                logger.error(f"Missing: {req_dir}")
                return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Gemma SAE Experiment Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all phases with SAE Boost
    python run_pipeline.py --gpu 0 --phases all --use-boost

    # Run without SAE Boost (Base SAE only)
    python run_pipeline.py --gpu 0 --phases 1,2,3,4,5

    # Run only Phase 0 (SAE Boost training)
    python run_pipeline.py --gpu 0 --phases 0

    # Quick test with subset of layers
    python run_pipeline.py --gpu 0 --phases 0,1,2 --layers 25,30,35 --use-boost
        """
    )

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--phases', type=str, default='all',
                       help='Phases to run: "all" or comma-separated (e.g., "0,1,2,3")')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers (e.g., "25,30,35")')
    parser.add_argument('--trials', type=int, default=None,
                       help='Number of trials for Phase 5')
    parser.add_argument('--use-boost', action='store_true',
                       help='Use Boosted SAE (Phase 0 must be run first)')
    parser.add_argument('--skip-check', action='store_true',
                       help='Skip dependency check')

    args = parser.parse_args()

    # Parse phases
    if args.phases.lower() == 'all':
        if args.use_boost:
            phases = [0, 1, 2, 3, 4, 5]
        else:
            phases = [1, 2, 3, 4, 5]
    else:
        phases = [int(p.strip()) for p in args.phases.split(',')]

    # Validate phases
    valid_phases = {0, 1, 2, 3, 4, 5}
    for p in phases:
        if p not in valid_phases:
            logger.error(f"Invalid phase: {p}. Valid phases: 0-5")
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

    # Check dependencies
    if not args.skip_check:
        if not check_phase_dependencies(phases, config_path, args.use_boost):
            logger.error("Dependency check failed. Use --skip-check to bypass.")
            return

    # Log start
    logger.info("=" * 70)
    logger.info("GEMMA SAE EXPERIMENT PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Phases to run: {phases}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Use Boost: {args.use_boost}")
    if target_layers:
        logger.info(f"Target layers: {target_layers}")

    # Run phases
    start_time = datetime.now()

    try:
        for phase in phases:
            phase_start = datetime.now()

            if phase == 0:
                run_phase0(config_path, args.gpu, target_layers)
            elif phase == 1:
                run_phase1(config_path, args.gpu, target_layers, args.use_boost)
            elif phase == 2:
                run_phase2(config_path, target_layers)
            elif phase == 3:
                run_phase3(config_path, args.gpu, target_layers)
            elif phase == 4:
                run_phase4(config_path, args.gpu, target_layers)
            elif phase == 5:
                run_phase5(config_path, args.gpu, target_layers, args.trials)

            phase_time = datetime.now() - phase_start
            logger.info(f"Phase {phase} completed in {phase_time}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Log completion
    total_time = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print output locations
    config = load_config(config_path)
    logger.info("\nOutput locations:")
    for phase in phases:
        output_dir = get_phase_output_dir(config, phase)
        logger.info(f"  Phase {phase}: {output_dir}")


if __name__ == '__main__':
    main()
