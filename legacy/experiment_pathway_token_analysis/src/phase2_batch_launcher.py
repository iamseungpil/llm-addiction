#!/usr/bin/env python3
"""
Phase 2 Batch Launcher
Distributes feature-feature correlation analysis across GPUs 4-7
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict
import argparse

def load_feature_list(causal_file: Path) -> List[Dict]:
    """Load all 2,787 causal features"""
    with open(causal_file, 'r') as f:
        data = json.load(f)

    features = []
    for feat in data['features']:
        layer = feat['layer']
        feature_id = feat['feature_id']
        feature_name = f"L{layer}-{feature_id}"
        features.append({
            'name': feature_name,
            'layer': layer,
            'feature_id': feature_id
        })

    return features

def launch_phase2_job(
    patching_file: Path,
    target_feature: str,
    condition: str,
    output_file: Path,
    gpu_id: int,
    dry_run: bool = False
):
    """Launch a single Phase 2 analysis job"""

    cmd = [
        'python3',
        'src/phase2_patching_correlations.py',
        '--patching-file', str(patching_file),
        '--target-feature', target_feature,
        '--condition', condition,
        '--output', str(output_file)
    ]

    if dry_run:
        print(f"[DRY RUN] GPU {gpu_id}: {target_feature} / {condition}")
        return None

    # Run with CUDA_VISIBLE_DEVICES (inherit parent environment)
    import os
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per job
        )

        if result.returncode == 0:
            return {'success': True, 'feature': target_feature, 'condition': condition}
        else:
            return {
                'success': False,
                'feature': target_feature,
                'condition': condition,
                'error': result.stderr
            }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'feature': target_feature,
            'condition': condition,
            'error': 'Timeout after 10 minutes'
        }
    except Exception as e:
        return {
            'success': False,
            'feature': target_feature,
            'condition': condition,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID (4, 5, 6, or 7)')
    parser.add_argument('--offset', type=int, default=0, help='Feature offset')
    parser.add_argument('--limit', type=int, default=None, help='Max features to process')
    parser.add_argument('--dry-run', action='store_true', help='Print jobs without executing')
    parser.add_argument('--causal-features', type=str,
                       default='/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_list.json')
    args = parser.parse_args()

    # Paths
    base_dir = Path('/data/llm_addiction/experiment_pathway_token_analysis')
    patching_file = base_dir / f'results/phase1_patching_full/phase1_patching_multifeature_gpu{args.gpu_id}.jsonl'
    output_dir = base_dir / 'results/phase2_correlations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Conditions
    conditions = [
        'safe_mean_safe',
        'safe_mean_risky',
        'risky_mean_safe',
        'risky_mean_risky'
    ]

    # Load features
    all_features = load_feature_list(Path(args.causal_features))

    # Apply offset/limit
    if args.limit:
        features_to_process = all_features[args.offset:args.offset+args.limit]
    elif args.offset > 0:
        features_to_process = all_features[args.offset:]
    else:
        features_to_process = all_features

    total_jobs = len(features_to_process) * len(conditions)

    print(f"=== Phase 2 Batch Launcher ===")
    print(f"GPU: {args.gpu_id}")
    print(f"Features: {len(features_to_process)} (offset={args.offset}, limit={args.limit})")
    print(f"Conditions: {len(conditions)}")
    print(f"Total jobs: {total_jobs}")
    print(f"Patching file: {patching_file}")
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No actual execution]")

    # Execute jobs
    results = []
    completed = 0
    failed = 0

    for i, feat in enumerate(features_to_process, 1):
        for cond in conditions:
            output_file = output_dir / f"correlations_gpu{args.gpu_id}_{feat['name']}_{cond}.jsonl"

            # Skip if already exists
            if output_file.exists() and not args.dry_run:
                print(f"[{i}/{len(features_to_process)}] SKIP: {feat['name']} / {cond} (already exists)")
                completed += 1
                continue

            print(f"[{i}/{len(features_to_process)}] Running: {feat['name']} / {cond}")

            result = launch_phase2_job(
                patching_file=patching_file,
                target_feature=feat['name'],
                condition=cond,
                output_file=output_file,
                gpu_id=args.gpu_id,
                dry_run=args.dry_run
            )

            if result and result['success']:
                completed += 1
            elif result:
                failed += 1
                print(f"  ❌ FAILED: {result.get('error', 'Unknown error')}")

            results.append(result)

    # Summary
    print(f"\n=== Phase 2 Batch Summary ===")
    print(f"Total jobs: {total_jobs}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*completed/total_jobs if total_jobs > 0 else 0:.1f}%")

if __name__ == '__main__':
    main()
