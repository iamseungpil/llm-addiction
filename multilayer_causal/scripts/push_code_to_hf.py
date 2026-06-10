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


def main():
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
        from huggingface_hub import HfApi
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=str(tar), path_in_repo=DEST,
            repo_id=HF_REPO, repo_type="dataset",
            commit_message="code: multilayer_causal latest")
        print(f"pushed {tar.stat().st_size / 1e6:.1f}MB -> {DEST}")


if __name__ == "__main__":
    main()
