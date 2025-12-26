#!/usr/bin/env python3
"""
Full Pipeline Orchestrator: 5-Phase Causal Analysis

Orchestrates the complete A+B+C+D causal analysis pipeline:
  Phase 1 (A): Steering Vector Extraction - Extract hidden states, compute steering vectors
  Phase 2 (C): SAE Feature Projection - Project steering vectors through SAE
  Phase 3 (B): Soft Interpolation Patching - Validate features via dose-response
  Phase 4 (D): Head Patching - Identify causal attention heads
  Phase 5: Gambling-Context Interpretation - Generate behavioral explanations

Usage:
    # Full pipeline
    python run_full_pipeline.py --model llama --gpu 0

    # Start from specific phase
    python run_full_pipeline.py --model llama --gpu 0 --start-phase 3

    # Run specific phases only
    python run_full_pipeline.py --model llama --gpu 0 --phases 1,2,5

Design: Modular orchestration with checkpointing and resume capability.
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logging,
    load_config,
    get_default_config_path,
    CheckpointManager
)


@dataclass
class PhaseInfo:
    """Information about a pipeline phase."""
    phase_id: int
    name: str
    script: str
    description: str
    input_files: List[str]  # Required input file patterns
    output_pattern: str  # Output file pattern


# Define all phases
PHASES = {
    1: PhaseInfo(
        phase_id=1,
        name='steering_extraction',
        script='extract_steering_vectors.py',
        description='Extract hidden states and compute steering vectors',
        input_files=[],  # Uses raw experiment data
        output_pattern='steering_vectors_{model}_*.npz'
    ),
    2: PhaseInfo(
        phase_id=2,
        name='sae_projection',
        script='sae_feature_projection.py',
        description='Project steering vectors through SAE to find candidate features',
        input_files=['steering_vectors_{model}_*.npz'],
        output_pattern='candidate_features_{model}_*.json'
    ),
    3: PhaseInfo(
        phase_id=3,
        name='soft_interpolation',
        script='soft_interpolation_patching.py',
        description='Validate features via dose-response patching',
        input_files=['candidate_features_{model}_*.json'],
        output_pattern='validated_features_{model}_*.json'
    ),
    4: PhaseInfo(
        phase_id=4,
        name='head_patching',
        script='head_patching.py',
        description='Identify causal attention heads',
        input_files=['validated_features_{model}_*.json'],
        output_pattern='causal_heads_{model}_*.json'
    ),
    5: PhaseInfo(
        phase_id=5,
        name='interpretation',
        script='gambling_interpretation.py',
        description='Generate gambling-context interpretations',
        input_files=['validated_features_{model}_*.json'],
        output_pattern='interpretations_{model}_*.json'
    )
}


class PipelineOrchestrator:
    """
    Orchestrate the full 5-phase causal analysis pipeline.

    Handles phase execution, input/output file management,
    checkpointing, and error recovery.
    """

    def __init__(
        self,
        model_name: str,
        gpu_id: int,
        config: Dict,
        config_path: str,
        logger=None
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            model_name: 'llama' or 'gemma'
            gpu_id: GPU ID to use
            config: Configuration dictionary
            config_path: Path to config file
            logger: Optional logger
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.config = config
        self.config_path = config_path
        self.logger = logger

        self.output_dir = Path(config['output_dir'])
        self.src_dir = Path(__file__).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track completed phases and their outputs
        self.phase_outputs = {}

    def _log(self, msg: str):
        """Log a message."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def find_latest_file(self, pattern: str) -> Optional[Path]:
        """Find the latest file matching a pattern."""
        formatted_pattern = pattern.format(model=self.model_name)
        matches = sorted(self.output_dir.glob(formatted_pattern))
        if matches:
            return matches[-1]
        return None

    def run_phase(
        self,
        phase_id: int,
        extra_args: Optional[Dict] = None
    ) -> bool:
        """
        Run a single pipeline phase.

        Args:
            phase_id: Phase ID (1-5)
            extra_args: Optional extra command-line arguments

        Returns:
            True if successful, False otherwise
        """
        if phase_id not in PHASES:
            self._log(f"Unknown phase: {phase_id}")
            return False

        phase = PHASES[phase_id]
        self._log("")
        self._log("=" * 70)
        self._log(f"PHASE {phase_id}: {phase.name.upper()}")
        self._log(f"Description: {phase.description}")
        self._log("=" * 70)

        # Check for required input files
        for input_pattern in phase.input_files:
            input_file = self.find_latest_file(input_pattern)
            if input_file is None:
                # Check if we have it from previous phase
                prev_phase_id = phase_id - 1
                if prev_phase_id in self.phase_outputs:
                    input_file = self.phase_outputs[prev_phase_id]
                else:
                    self._log(f"ERROR: Required input not found: {input_pattern}")
                    return False
            self._log(f"Input: {input_file}")

        # Build command
        script_path = self.src_dir / phase.script

        if not script_path.exists():
            # Try alternate naming
            alternate_scripts = {
                'sae_feature_projection.py': 'analyze_steering_with_sae.py'
            }
            if phase.script in alternate_scripts:
                script_path = self.src_dir / alternate_scripts[phase.script]

            if not script_path.exists():
                self._log(f"ERROR: Script not found: {phase.script}")
                return False

        cmd = [
            sys.executable,
            str(script_path),
            '--model', self.model_name,
            '--gpu', str(self.gpu_id),
            '--config', self.config_path
        ]

        # Add phase-specific input arguments
        if phase_id == 2:
            # SAE projection needs steering vectors
            vectors_file = self.find_latest_file('steering_vectors_{model}_*.npz')
            if vectors_file:
                cmd.extend(['--vectors', str(vectors_file)])

        elif phase_id == 3:
            # Soft interpolation needs candidate features
            candidates_file = self.find_latest_file('candidate_features_{model}_*.json')
            if candidates_file is None:
                # Fall back to SAE analysis output
                candidates_file = self.find_latest_file('sae_analysis_{model}_*.json')
            if candidates_file is None:
                self._log("ERROR: No candidate features file found for Phase 3")
                self._log("Run Phase 2 (SAE projection) first, or provide --candidates argument")
                return False
            cmd.extend(['--candidates', str(candidates_file)])

            # Add real prompts file if available (from Step 0)
            prompts_file = self.find_latest_file('condition_prompts_*.json')
            if prompts_file:
                cmd.extend(['--prompts', str(prompts_file)])
                self._log(f"Using real prompts: {prompts_file}")

        elif phase_id == 4:
            # Head patching needs validated features
            validated_file = self.find_latest_file('validated_features_{model}_*.json')
            if validated_file is None:
                self._log("ERROR: No validated features file found for Phase 4")
                self._log("Run Phase 3 (soft interpolation) first, or provide --validated argument")
                return False
            cmd.extend(['--validated', str(validated_file)])

        elif phase_id == 5:
            # Interpretation needs validated features
            validated_file = self.find_latest_file('validated_features_{model}_*.json')
            if validated_file is None:
                self._log("ERROR: No validated features file found for Phase 5")
                self._log("Run Phase 3 (soft interpolation) first, or provide --validated argument")
                return False
            cmd.extend(['--validated', str(validated_file)])

        # Add extra args
        if extra_args:
            for key, value in extra_args.items():
                cmd.extend([f'--{key}', str(value)])

        self._log(f"Command: {' '.join(cmd)}")

        # Run the phase with explicit CUDA_VISIBLE_DEVICES
        # This ensures the subprocess uses the correct GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self._log(f"Setting CUDA_VISIBLE_DEVICES={self.gpu_id}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.src_dir),
                env=env
            )

            # Log output
            if result.stdout:
                for line in result.stdout.split('\n')[-30:]:
                    if line.strip():
                        self._log(f"  {line}")

            if result.returncode != 0:
                self._log(f"ERROR: Phase {phase_id} failed with return code {result.returncode}")
                if result.stderr:
                    self._log(f"STDERR: {result.stderr[-500:]}")
                return False

            # Find output file
            output_file = self.find_latest_file(phase.output_pattern)
            if output_file:
                self.phase_outputs[phase_id] = output_file
                self._log(f"Output: {output_file}")

            self._log(f"Phase {phase_id} completed successfully")
            return True

        except Exception as e:
            self._log(f"ERROR: Phase {phase_id} failed with exception: {e}")
            return False

    def run_pipeline(
        self,
        phases_to_run: Optional[List[int]] = None,
        start_phase: int = 1,
        strict_mode: bool = True
    ) -> Dict:
        """
        Run the complete pipeline or specified phases.

        Args:
            phases_to_run: List of specific phases to run (overrides start_phase)
            start_phase: Phase to start from (1-5)
            strict_mode: If True, stop on any phase failure

        Returns:
            Dict with pipeline results
        """
        self._log("")
        self._log("=" * 70)
        self._log("5-PHASE CAUSAL ANALYSIS PIPELINE")
        self._log("=" * 70)
        self._log(f"Model: {self.model_name.upper()}")
        self._log(f"GPU: {self.gpu_id}")
        self._log(f"Output: {self.output_dir}")
        self._log(f"Start phase: {start_phase}")
        self._log(f"Strict mode: {strict_mode}")

        # Determine phases to run
        if phases_to_run is None:
            phases_to_run = list(range(start_phase, 6))

        self._log(f"Phases to run: {phases_to_run}")

        results = {
            'model': self.model_name,
            'phases_attempted': [],
            'phases_completed': [],
            'phases_failed': [],
            'outputs': {}
        }

        # Run each phase
        for phase_id in phases_to_run:
            results['phases_attempted'].append(phase_id)

            success = self.run_phase(phase_id)

            if success:
                results['phases_completed'].append(phase_id)
                if phase_id in self.phase_outputs:
                    results['outputs'][phase_id] = str(self.phase_outputs[phase_id])
            else:
                results['phases_failed'].append(phase_id)
                if strict_mode:
                    self._log(f"Stopping pipeline due to phase {phase_id} failure")
                    break

        # Print summary
        self._log("")
        self._log("=" * 70)
        self._log("PIPELINE SUMMARY")
        self._log("=" * 70)
        self._log(f"Completed: {results['phases_completed']}")
        self._log(f"Failed: {results['phases_failed']}")

        if results['outputs']:
            self._log("\nOutput files:")
            for phase_id, path in results['outputs'].items():
                self._log(f"  Phase {phase_id}: {Path(path).name}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Run the full 5-phase causal analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  1. Steering Vector Extraction (A) - Extract hidden states, compute steering vectors
  2. SAE Feature Projection (C) - Project through SAE, find candidate features
  3. Soft Interpolation Patching (B) - Validate via dose-response
  4. Head Patching (D) - Find causal attention heads
  5. Interpretation - Generate gambling-context explanations

Examples:
  # Run full pipeline
  python run_full_pipeline.py --model llama --gpu 0

  # Start from phase 3
  python run_full_pipeline.py --model llama --gpu 0 --start-phase 3

  # Run only phases 1, 2, 5
  python run_full_pipeline.py --model llama --gpu 0 --phases 1,2,5
        """
    )

    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                       help='Model to analyze')
    parser.add_argument('--gpu', type=int, required=True,
                       help='GPU ID to use')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: auto-detect)')
    parser.add_argument('--start-phase', type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help='Phase to start from')
    parser.add_argument('--phases', type=str, default=None,
                       help='Comma-separated list of specific phases to run (e.g., "1,2,5")')
    parser.add_argument('--no-strict', action='store_true',
                       help='Continue on phase failures (default: stop on failure)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be run without executing')

    args = parser.parse_args()

    # Load config
    config_path = args.config or str(get_default_config_path())
    config = load_config(config_path)

    # Setup paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(f'pipeline_{args.model}_{timestamp}', log_dir)

    # Parse phases
    phases_to_run = None
    if args.phases:
        phases_to_run = [int(p) for p in args.phases.split(',')]

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - Not executing")
        logger.info(f"Would run phases: {phases_to_run or list(range(args.start_phase, 6))}")
        logger.info(f"Model: {args.model}")
        logger.info(f"GPU: {args.gpu}")
        logger.info(f"Config: {config_path}")
        return 0

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        model_name=args.model,
        gpu_id=args.gpu,
        config=config,
        config_path=config_path,
        logger=logger
    )

    # Run pipeline
    results = orchestrator.run_pipeline(
        phases_to_run=phases_to_run,
        start_phase=args.start_phase,
        strict_mode=not args.no_strict
    )

    # Save results
    results_path = output_dir / f'pipeline_results_{args.model}_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nPipeline results saved to {results_path}")

    # Return exit code based on success
    if results['phases_failed']:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
