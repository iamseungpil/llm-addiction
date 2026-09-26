#!/usr/bin/env python3
"""
Parallel Soft Interpolation Patching

Splits remaining features across multiple GPUs and runs in parallel.
Supports resuming from checkpoint.

Usage:
    python parallel_soft_interpolation.py --model llama --gpus 4,5,6
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import time

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path("/data/llm_addiction/steering_vector_experiment_full")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def get_latest_checkpoint(model: str) -> Tuple[int, List[Dict], List[Dict]]:
    """
    Get the latest checkpoint for the model.

    Returns:
        Tuple of (progress, validated_results, rejected_results)
    """
    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob(f"soft_interp_{model}_soft_interpolation_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not checkpoint_files:
        return 0, [], []

    latest = checkpoint_files[0]
    print(f"Loading checkpoint: {latest.name}")

    with open(latest) as f:
        data = json.load(f)

    progress = data.get('progress', 0)
    validated = data.get('nominal_validated', [])
    rejected = data.get('nominal_rejected', [])

    print(f"  Progress: {progress}")
    print(f"  Validated: {len(validated)}")
    print(f"  Rejected: {len(rejected)}")

    return progress, validated, rejected


def load_candidate_features(candidates_path: str) -> List[Dict]:
    """Load all candidate features."""
    with open(candidates_path) as f:
        data = json.load(f)
    return data['candidates']


def split_features(
    all_features: List[Dict],
    start_idx: int,
    n_gpus: int
) -> List[List[Dict]]:
    """
    Split remaining features across GPUs.

    Args:
        all_features: All candidate features
        start_idx: Index to start from (after checkpoint)
        n_gpus: Number of GPUs to use

    Returns:
        List of feature lists, one per GPU
    """
    remaining = all_features[start_idx:]
    n_remaining = len(remaining)
    chunk_size = n_remaining // n_gpus

    chunks = []
    for i in range(n_gpus):
        start = i * chunk_size
        if i == n_gpus - 1:
            # Last chunk gets the remainder
            end = n_remaining
        else:
            end = start + chunk_size
        chunks.append(remaining[start:end])

    return chunks


def create_chunk_files(
    chunks: List[List[Dict]],
    model: str,
    original_data: Dict
) -> List[str]:
    """
    Create temporary candidate files for each GPU.

    Returns:
        List of file paths
    """
    chunk_files = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i, chunk in enumerate(chunks):
        chunk_data = {
            'model': model,
            'total_candidates': len(chunk),
            'total_layers': original_data.get('total_layers', 32),
            'source_files': original_data.get('source_files', []),
            'candidates': chunk,
            'layers': original_data.get('layers', []),
            'chunk_info': {
                'chunk_index': i,
                'total_chunks': len(chunks),
                'timestamp': timestamp
            }
        }

        chunk_path = OUTPUT_DIR / f"candidates_{model}_chunk{i}_{timestamp}.json"
        with open(chunk_path, 'w') as f:
            json.dump(chunk_data, f, indent=2)

        chunk_files.append(str(chunk_path))
        print(f"Created chunk {i}: {len(chunk)} features -> {chunk_path.name}")

    return chunk_files


def run_parallel_validation(
    chunk_files: List[str],
    gpus: List[int],
    model: str,
    config_path: str,
    prompts_path: str
) -> List[subprocess.Popen]:
    """
    Start parallel validation processes.

    Returns:
        List of subprocess handles
    """
    script_path = BASE_DIR / "src" / "soft_interpolation_patching.py"
    processes = []

    for i, (chunk_file, gpu) in enumerate(zip(chunk_files, gpus)):
        cmd = [
            "python", str(script_path),
            "--model", model,
            "--gpu", str(gpu),
            "--candidates", chunk_file,
            "--prompts", prompts_path,
            "--config", config_path
        ]

        log_file = OUTPUT_DIR / f"logs" / f"parallel_{model}_gpu{gpu}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting GPU {gpu} with {chunk_file}")
        print(f"  Log: {log_file}")

        with open(log_file, 'w') as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR)
            )

        processes.append((proc, gpu, log_file))
        print(f"  PID: {proc.pid}")

    return processes


def monitor_processes(processes: List[Tuple]) -> None:
    """Monitor running processes until completion."""
    print("\n" + "="*60)
    print("MONITORING PARALLEL PROCESSES")
    print("="*60)

    while True:
        all_done = True
        status_lines = []

        for proc, gpu, log_file in processes:
            poll = proc.poll()
            if poll is None:
                all_done = False
                # Get last line of log
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        last_line = lines[-1].strip() if lines else "Starting..."
                        # Extract progress if present
                        if "Validating features:" in last_line:
                            status_lines.append(f"GPU {gpu}: {last_line.split('Validating features:')[1][:30]}")
                        else:
                            status_lines.append(f"GPU {gpu}: Running...")
                except:
                    status_lines.append(f"GPU {gpu}: Running...")
            else:
                status_lines.append(f"GPU {gpu}: DONE (exit code {poll})")

        # Print status
        print(f"\r{' | '.join(status_lines)}", end='', flush=True)

        if all_done:
            print("\n\nAll processes completed!")
            break

        time.sleep(30)


def merge_results(
    model: str,
    checkpoint_validated: List[Dict],
    checkpoint_rejected: List[Dict]
) -> str:
    """
    Merge results from checkpoint and all parallel runs.

    Returns:
        Path to merged results file
    """
    print("\n" + "="*60)
    print("MERGING RESULTS")
    print("="*60)

    # Find all result files from parallel runs
    result_files = sorted(
        OUTPUT_DIR.glob(f"validated_features_{model}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    all_validated = list(checkpoint_validated)
    all_rejected = list(checkpoint_rejected)

    # Get unique results by (layer, feature_id)
    seen = set()
    for v in all_validated:
        key = (v.get('layer'), v.get('feature_id'))
        seen.add(key)
    for r in all_rejected:
        key = (r.get('layer'), r.get('feature_id'))
        seen.add(key)

    # Load results from each parallel run
    for result_file in result_files[:10]:  # Check recent files
        try:
            with open(result_file) as f:
                data = json.load(f)

            for v in data.get('validated_features', []):
                key = (v.get('layer'), v.get('feature_id'))
                if key not in seen:
                    all_validated.append(v)
                    seen.add(key)

            for r in data.get('rejected_features', []):
                key = (r.get('layer'), r.get('feature_id'))
                if key not in seen:
                    all_rejected.append(r)
                    seen.add(key)

            print(f"  Loaded: {result_file.name}")
        except Exception as e:
            print(f"  Error loading {result_file.name}: {e}")

    # Create merged result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    merged_path = OUTPUT_DIR / f"validated_features_{model}_merged_{timestamp}.json"

    merged_data = {
        'model': model,
        'n_candidates': len(all_validated) + len(all_rejected),
        'n_validated': len(all_validated),
        'validation_rate': len(all_validated) / (len(all_validated) + len(all_rejected)) if (all_validated or all_rejected) else 0,
        'validated_features': all_validated,
        'rejected_features': all_rejected[:50],  # Limit rejected
        'merge_info': {
            'timestamp': timestamp,
            'n_from_checkpoint': len(checkpoint_validated) + len(checkpoint_rejected),
            'n_from_parallel': len(all_validated) + len(all_rejected) - len(checkpoint_validated) - len(checkpoint_rejected)
        }
    }

    with open(merged_path, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"\nMerged results saved to: {merged_path}")
    print(f"  Total validated: {len(all_validated)}")
    print(f"  Total rejected: {len(all_rejected)}")

    return str(merged_path)


def main():
    parser = argparse.ArgumentParser(description='Parallel soft interpolation patching')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'])
    parser.add_argument('--gpus', type=str, required=True, help='Comma-separated GPU IDs (e.g., 4,5,6)')
    parser.add_argument('--candidates', type=str, default=None, help='Path to candidates file')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--prompts', type=str, default=None, help='Path to prompts file')
    parser.add_argument('--no-monitor', action='store_true', help='Do not monitor processes')

    args = parser.parse_args()

    # Parse GPUs
    gpus = [int(g.strip()) for g in args.gpus.split(',')]
    n_gpus = len(gpus)
    print(f"Using {n_gpus} GPUs: {gpus}")

    # Set default paths
    if args.candidates is None:
        args.candidates = str(OUTPUT_DIR / f"candidate_features_{args.model}_merged_20251221_183032.json")
    if args.config is None:
        args.config = str(BASE_DIR / "configs" / "experiment_config_full_layers.yaml")
    if args.prompts is None:
        args.prompts = str(OUTPUT_DIR / "condition_prompts.json")

    # Load checkpoint
    progress, validated, rejected = get_latest_checkpoint(args.model)

    # Load all features
    print(f"\nLoading candidates from: {args.candidates}")
    with open(args.candidates) as f:
        original_data = json.load(f)
    all_features = original_data['candidates']
    print(f"  Total features: {len(all_features)}")
    print(f"  Already processed: {progress}")
    print(f"  Remaining: {len(all_features) - progress}")

    # Split remaining features
    chunks = split_features(all_features, progress, n_gpus)
    print(f"\nSplit into {n_gpus} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i} (GPU {gpus[i]}): {len(chunk)} features")

    # Create chunk files
    chunk_files = create_chunk_files(chunks, args.model, original_data)

    # Run parallel validation
    processes = run_parallel_validation(
        chunk_files, gpus, args.model,
        args.config, args.prompts
    )

    if not args.no_monitor:
        # Monitor until completion
        monitor_processes(processes)

        # Merge results
        merge_results(args.model, validated, rejected)
    else:
        print("\nProcesses started in background. Check logs for progress.")
        print("Run with --merge-only to merge results when complete.")


if __name__ == '__main__':
    main()
