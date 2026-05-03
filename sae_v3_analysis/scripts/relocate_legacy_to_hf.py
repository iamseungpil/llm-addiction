"""Phase D: relocate stale/invalidated files on HF dataset to legacy/ subdirs.

Move (= copy + delete) within the dataset repo using HF Hub API operations.
Adds a README.md per legacy/ subdir explaining what's there and why moved.

Categories:
  legacy/v12_steering_invalidated/  — V12 steering, prompts mismatch (2026-04-14)
  legacy/v14_steering/                — V14 steering, paper does not cite
  legacy/v16_steering/                — V16 multilayer steering, paper does not cite
  legacy/v17_leaky_pipeline/          — V17 RF deconfound leakage, do not cite

NO files are deleted — only relocated. Original paths become 404 to
prevent accidental citation; legacy/ paths remain readable.
"""
from __future__ import annotations
import os
import sys
from collections import defaultdict
from huggingface_hub import (
    HfApi, CommitOperationCopy, CommitOperationDelete, CommitOperationAdd,
)

TOKEN = os.environ.get('HF_TOKEN')
if not TOKEN:
    print('Set HF_TOKEN', file=sys.stderr); sys.exit(1)

REPO = 'llm-addiction-research/llm-addiction'
api = HfApi(token=TOKEN)

# Per-category README content
README_V12 = """# V12 Steering — INVALIDATED (do not cite)

Files in this directory are V12 (cross-domain dose-response) steering experiment
outputs from 2026-03-15 to 2026-04-14. They were used in earlier ICLR
preparation but **invalidated 2026-04-14** because the steering prompts did
not match the §3 behavioural-experiment prompt distribution (no
ROLE_INSTRUCTION / no G,M variants).

The paper (NeurIPS 2026 + EMNLP 2026 submission) does NOT cite these files.
For the corrected, paper-canonical pipeline, see:
  sae_v3_analysis/results/table1_groupkfold_*.json
  sae_v3_analysis/results/condition_modulation_groupkfold_L22.json

For the methodology transition record, see:
  sae_v3_analysis/results/README.md (§4 paper-canonical pipeline section)

Archived for reproducibility audit only.
"""

README_V14 = """# V14 Steering — Stale (paper does not cite)

V14 steering follow-up experiments (2026-03-31 to 2026-04-09) running on
behavioural prompts but at the SAE feature level rather than direction level.
Mostly null at corrected scale.

The paper does NOT cite V14. The decision to omit was made when V11 zero-
ablation showed individual neurons NULL → V12 direction steering attempted
→ Plan v5 corrected reproduction NS at scale → all steering claims removed
from paper §4 (commit 4646999, 2026-05-03).

For paper-canonical results, see:
  sae_v3_analysis/results/table1_groupkfold_*.json

Archived for reproducibility audit only.
"""

README_V16 = """# V16 Steering — Stale (paper does not cite)

V16 multilayer steering experiments (2026-04-08 to 2026-04-17) targeting
multiple layers simultaneously. Paper does NOT cite V16 results because:
  1. Steering claims removed from paper (commit 4646999, 2026-05-03)
  2. Multilayer geometry collapses to direction steering at scale

For paper-canonical results, see:
  sae_v3_analysis/results/table1_groupkfold_*.json

Archived for reproducibility audit only.
"""

README_V17 = """# V17 Leaky Pipeline — DO NOT CITE

These files are V17 SAE-readout outputs produced with a leaky RF deconfound:
the RandomForest is fit on the full target before CV split, leaking test-fold
labels into the train residual. This inflates I_LC for some cells (e.g.,
LLaMA/MW L22 from 0.293 strict-CV to 0.779 leaky).

  paper_neural_audit.json     — V17 binary I_LC pipeline output
  v17_nonlinear_deconfound.txt   — leaky text report
  v17_nonlinear_deconfound_REFERENCE.txt — second copy
  build_paper_neural_audit.py — script that produced the above

The paper (NeurIPS / EMNLP 2026 submission) cites only the strict-CV
GroupKFold pipeline — see:
  sae_v3_analysis/results/table1_groupkfold_*.json
  sae_v3_analysis/src/run_groupkfold_recompute.py
  sae_v3_analysis/src/run_perm_null_ilc.py    (strict within-fold deconfound)

Appendix C.1 of the paper documents this transition explicitly
(Tables tab:appendix-sweep-verification + tab:appendix-sweep-peak-mismatch).

Archived for reproducibility audit only — do not use for new claims.
"""

CATEGORY_README = {
    'legacy/v12_steering_invalidated/README.md': README_V12,
    'legacy/v14_steering/README.md': README_V14,
    'legacy/v16_steering/README.md': README_V16,
    'legacy/v17_leaky_pipeline/README.md': README_V17,
}


def categorize(files):
    """Return dict[target_dir, list[old_path]]."""
    out = defaultdict(list)
    for f in files:
        fl = f.lower()
        bn = f.split('/')[-1].lower()
        if ('v12_' in bn or '/v12_' in fl) and 'invalidation' not in fl:
            out['legacy/v12_steering_invalidated/'].append(f)
        elif 'v14_' in bn or '/v14_' in fl:
            out['legacy/v14_steering/'].append(f)
        elif 'v16_' in bn or '/v16_' in fl:
            out['legacy/v16_steering/'].append(f)
        elif 'v17_nonlinear' in fl or 'paper_neural_audit' in fl or 'build_paper_neural_audit' in fl:
            out['legacy/v17_leaky_pipeline/'].append(f)
    return out


def main():
    print('Listing repo files ...')
    files = api.list_repo_files(REPO, repo_type='dataset')
    print(f'  total: {len(files):,}')

    cat = categorize(files)
    total = sum(len(v) for v in cat.values())
    print(f'\nMoving {total} files into legacy/:')
    for tgt, lst in cat.items():
        print(f'  {tgt:50s} {len(lst):>3} files')

    # Build operations: per category one commit (avoid 1 mega-commit)
    for tgt_dir, file_list in cat.items():
        ops = []
        # Add the README.md
        readme_remote = tgt_dir + 'README.md'
        readme_text = CATEGORY_README[readme_remote]
        ops.append(CommitOperationAdd(
            path_in_repo=readme_remote,
            path_or_fileobj=readme_text.encode('utf-8'),
        ))
        # Copy + delete each file
        for f in file_list:
            new_path = tgt_dir + f.split('/')[-1]
            ops.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=new_path))
            ops.append(CommitOperationDelete(path_in_repo=f))
        print(f'\n  Committing {tgt_dir}: {len(ops)} ops ...')
        api.create_commit(
            repo_id=REPO, repo_type='dataset',
            operations=ops,
            commit_message=f'Phase D: relocate {len(file_list)} stale files to {tgt_dir}',
        )
        print(f'  OK')

    print('\n=== Phase D complete ===')


if __name__ == '__main__':
    main()
