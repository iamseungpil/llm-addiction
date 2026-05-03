"""Upload §4 paper-canonical results + GroupKFold sweep to HuggingFace dataset.

Target: llm-addiction-research/llm-addiction (dataset)
Path on HF: sae_v3_analysis/results/...
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
from huggingface_hub import HfApi

TOKEN = os.environ.get('HF_TOKEN')
if not TOKEN:
    print('Set HF_TOKEN env var', file=sys.stderr)
    sys.exit(1)

REPO = 'llm-addiction-research/llm-addiction'
LOCAL_RESULTS = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results')
LOCAL_SRC = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/src')

# Paper-canonical files first
CANONICAL = [
    'README.md',
    'table1_groupkfold_L8.json',
    'table1_groupkfold_L12.json',
    'table1_groupkfold_L22.json',
    'table1_groupkfold_L25.json',
    'table1_groupkfold_L30.json',
    'table1_perm_null.json',
    'condition_modulation_groupkfold_L22.json',
    'condition_modulation_continuous_ilc_L22.json',
    'iba_cross_task_transfer.json',
    'rq2_audit_consistent_layer.json',
    'headline_robustness.json',
]

# Pre-GroupKFold sweep (kept for traceability)
SWEEP_DIRS = [
    'sweep_3metrics',
    'sweep_3metrics_continuous_ilc',
]

# Scripts that produced the canonical results
SCRIPTS = [
    'run_groupkfold_recompute.py',
    'run_groupkfold_layer_sweep.py',
    'run_table1_perm_null.py',
    'run_perm_null_ilc.py',
    'run_comprehensive_robustness.py',
    'layer_sweep_3metrics.py',
    'layer_sweep_continuous_ilc.py',
]


def main():
    api = HfApi(token=TOKEN)
    print(f'target: {REPO} (dataset)')

    # 1. Canonical result JSONs
    for fname in CANONICAL:
        local = LOCAL_RESULTS / fname
        if not local.exists():
            print(f'  SKIP missing: {fname}')
            continue
        remote = f'sae_v3_analysis/results/{fname}'
        print(f'  uploading: {remote}  ({local.stat().st_size} B)', flush=True)
        api.upload_file(
            path_or_fileobj=str(local), path_in_repo=remote,
            repo_id=REPO, repo_type='dataset',
            commit_message=f'§4 paper-canonical: {fname}',
        )

    # 2. Pre-GroupKFold sweep dirs (entire folders)
    for dname in SWEEP_DIRS:
        local_dir = LOCAL_RESULTS / dname
        if not local_dir.exists():
            print(f'  SKIP missing dir: {dname}')
            continue
        print(f'  uploading folder: sae_v3_analysis/results/{dname}/', flush=True)
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=f'sae_v3_analysis/results/{dname}',
            repo_id=REPO, repo_type='dataset',
            commit_message=f'pre-GroupKFold sweep snapshot: {dname}',
        )

    # 3. Scripts
    for fname in SCRIPTS:
        local = LOCAL_SRC / fname
        if not local.exists():
            print(f'  SKIP missing: src/{fname}')
            continue
        remote = f'sae_v3_analysis/src/{fname}'
        print(f'  uploading: {remote}', flush=True)
        api.upload_file(
            path_or_fileobj=str(local), path_in_repo=remote,
            repo_id=REPO, repo_type='dataset',
            commit_message=f'§4 script: {fname}',
        )

    print('\nDONE')


if __name__ == '__main__':
    main()
