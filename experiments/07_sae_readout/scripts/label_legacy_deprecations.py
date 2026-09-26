#!/usr/bin/env python3
"""Upload the `legacy/` deprecation labels to the HF dataset. Adds files only.

Companion to `relocate_legacy_to_hf.py`, which moved the retired files into
`legacy/<category>/` and wrote one `README.md` per category. Two of those
categories -- `v17_leaky_pipeline/` and `pre_groupkfold_sweep/` -- carry a
do-not-cite README but not the `DEPRECATION_WARNING.md` filename the release uses
everywhere else, so a reader grepping the release by that filename finds
`sae_patching/`, `slot_machine/gemma/` and `slot_machine/llama/` and concludes the
two leaky-pipeline directories are clean. This script closes that gap and adds a
`legacy/README.md` at the folder root explaining the convention.

Unlike `relocate_legacy_to_hf.py` this script performs NO copies and NO deletes. It
uploads three markdown files and touches nothing else, so it is safe to re-run.

The files are kept as real files under `experiments/07_sae_readout/release_labels/`, mirroring
their destination paths, so they are readable and greppable in this repository
without running anything.

    python3 experiments/07_sae_readout/scripts/label_legacy_deprecations.py          # dry run
    HF_TOKEN=... python3 experiments/07_sae_readout/scripts/label_legacy_deprecations.py --push
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = 'llm-addiction-research/llm-addiction'
LABELS_ROOT = Path(__file__).resolve().parents[1] / 'release_labels'

# local file under release_labels/  ->  path in the dataset repo
UPLOADS = {
    'legacy/README.md': 'legacy/README.md',
    'legacy/v17_leaky_pipeline/DEPRECATION_WARNING.md':
        'legacy/v17_leaky_pipeline/DEPRECATION_WARNING.md',
    'legacy/pre_groupkfold_sweep/DEPRECATION_WARNING.md':
        'legacy/pre_groupkfold_sweep/DEPRECATION_WARNING.md',
}

COMMIT_MESSAGE = (
    'legacy: add DEPRECATION_WARNING.md to v17_leaky_pipeline and '
    'pre_groupkfold_sweep, plus legacy/README.md documenting the convention'
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--push', action='store_true',
                        help='actually upload; without it this only reports what would go')
    args = parser.parse_args()

    missing = [src for src in UPLOADS if not (LABELS_ROOT / src).is_file()]
    if missing:
        print('Missing label files under release_labels/:', file=sys.stderr)
        for src in missing:
            print(f'  {src}', file=sys.stderr)
        return 1

    print(f'Repo: {REPO} (dataset)')
    print(f'Adding {len(UPLOADS)} files; moving 0, deleting 0.\n')
    for src, dst in UPLOADS.items():
        size = (LABELS_ROOT / src).stat().st_size
        print(f'  {dst:58s} {size:>6,} bytes')

    if not args.push:
        print('\nDry run. Re-run with --push (and HF_TOKEN set) to upload.')
        return 0

    token = os.environ.get('HF_TOKEN')
    if not token:
        print('\nSet HF_TOKEN to push.', file=sys.stderr)
        return 1

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    operations = [
        CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(LABELS_ROOT / src))
        for src, dst in UPLOADS.items()
    ]
    api.create_commit(
        repo_id=REPO,
        repo_type='dataset',
        operations=operations,
        commit_message=COMMIT_MESSAGE,
    )
    print('\nUploaded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
