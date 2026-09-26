#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

_running = True


def _stop(signum, frame):
    global _running
    _running = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def _walk_results(base: Path) -> dict[Path, float]:
    out: dict[Path, float] = {}
    if not base.exists():
        return out
    for p in base.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".npz", ".csv", ".log", ".md"}:
            try:
                out[p] = p.stat().st_mtime
            except OSError:
                continue
    return out


def push_once(api: HfApi, repo: str, base: Path, last_mtimes: dict[Path, float]) -> dict[Path, float]:
    current = _walk_results(base)
    pushed = 0
    for path, mtime in current.items():
        if last_mtimes.get(path) == mtime:
            continue
        rel = path.relative_to(base).as_posix()
        target = f"results/{rel}"
        for attempt in range(3):
            try:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=target,
                    repo_id=repo,
                    repo_type="dataset",
                    commit_message=f"sync {rel} @ {datetime.now(timezone.utc).isoformat()}",
                )
                pushed += 1
                break
            except HfHubHTTPError as e:
                wait = 2 ** attempt
                print(f"[push] retry {attempt+1}/3 for {rel}: {e} (sleep {wait}s)", flush=True)
                time.sleep(wait)
    if pushed:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[push] {ts} synced {pushed} files", flush=True)
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--hf_repo", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[push] no HF_TOKEN; exiting", flush=True)
        return 0

    api = HfApi(token=token)
    base = Path(args.base_dir)
    last: dict[Path, float] = {}

    while _running:
        try:
            last = push_once(api, args.hf_repo, base, last)
        except Exception as e:
            print(f"[push] error: {e}", flush=True)
        for _ in range(args.interval):
            if not _running:
                break
            time.sleep(1)

    print("[push] graceful exit", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
