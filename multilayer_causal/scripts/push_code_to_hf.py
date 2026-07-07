#!/usr/bin/env python3
"""Tar the harness and upload LATEST to the HF dataset (same path, overwritten)."""
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "llm-addiction-research/llm-addiction"
DEST = "experiments/multilayer_causal/code/multilayer_causal.tar.gz"
PATHS = ["multilayer_causal"]          # self-contained for E1/E2/E3a

# The sec4 w8extract / w9icextract nodes fetch the LLaMA phase_a extractors from
# HF sae_v3_analysis/src/ (they are NOT inside the multilayer_causal tarball).
# Their staleness guards abort unless the fetched copies carry the additive-
# provenance write, so push the latest source alongside the tarball (w9icextract
# needs extract_llama_ic.py to see the provenance write). Overwriting these source
# files with the (superset-schema) latest is safe; it never touches any
# sae_features_v3 data path.
SRC_FILES = [
    "sae_v3_analysis/src/extract_llama_sm.py",
    "sae_v3_analysis/src/extract_llama_mw.py",
    "sae_v3_analysis/src/extract_llama_ic.py",
    "sae_v3_analysis/src/prompt_reconstruction.py",
]


def main():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / "multilayer_causal.tar.gz"
        subprocess.run(["tar", "czf", str(tar),
                        "--exclude", "multilayer_causal/out",
                        "--exclude", "__pycache__",
                        "--exclude", ".pytest_cache",
                        # rendered yamls carry the real HF token — never package
                        "--exclude", "*.rendered.yaml",
                        *PATHS],
                       cwd=REPO_ROOT, check=True)
        api.upload_file(
            path_or_fileobj=str(tar), path_in_repo=DEST,
            repo_id=HF_REPO, repo_type="dataset",
            commit_message="code: multilayer_causal latest")
        print(f"pushed {tar.stat().st_size / 1e6:.1f}MB -> {DEST}")
    for rel in SRC_FILES:
        local = REPO_ROOT / rel
        if not local.exists():
            print(f"skip (missing local) {rel}")
            continue
        api.upload_file(
            path_or_fileobj=str(local), path_in_repo=rel,
            repo_id=HF_REPO, repo_type="dataset",
            commit_message="code: sae_v3_analysis extractor latest")
        print(f"pushed {local.stat().st_size / 1e3:.1f}KB -> {rel}")


if __name__ == "__main__":
    main()
