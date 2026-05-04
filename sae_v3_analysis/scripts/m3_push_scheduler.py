"""10-minute HF push scheduler for M3 patching experiments.

Watches /results/v19_multi_patching/ for changed JSONL files and uploads
incrementally to HF every 10 minutes.

Resume-safe: idempotent (HF dedupe via path-based upload).
Background-safe: independent of main experiment process; can run as cron or
    plain background loop.

Usage:
  HF_TOKEN=... python m3_push_scheduler.py [--interval 600] [--once]
"""
from __future__ import annotations
import argparse, hashlib, json, os, signal, sys, time
from datetime import datetime
from pathlib import Path

REPO = 'llm-addiction-research/llm-addiction'
LOCAL_ROOT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/v19_multi_patching')
HF_PREFIX = 'sae_v3_analysis/results/v19_multi_patching'
PUSH_LOG = LOCAL_ROOT / '.push_log.json'
INTERVAL_DEFAULT = 600  # 10 minutes


def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_push_log() -> dict:
    if PUSH_LOG.exists():
        try:
            return json.load(open(PUSH_LOG))
        except Exception:
            pass
    return {'pushes': [], 'file_hashes': {}}


def save_push_log(log: dict) -> None:
    PUSH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PUSH_LOG, 'w') as f:
        json.dump(log, f, indent=2)


def find_changed(log: dict) -> list[Path]:
    """Find JSONL/JSON/MD files that changed since last push."""
    if not LOCAL_ROOT.exists():
        return []
    out = []
    for p in LOCAL_ROOT.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix not in {'.jsonl', '.json', '.md', '.log', '.txt'}:
            continue
        rel = str(p.relative_to(LOCAL_ROOT))
        if rel.startswith('.push_log'):
            continue
        h = file_md5(p)
        if log['file_hashes'].get(rel) != h:
            out.append(p)
            log['file_hashes'][rel] = h
    return out


def push_once(token: str) -> tuple[int, list[str]]:
    """Push all changed files. Returns (n_pushed, list_of_paths)."""
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token)

    log = load_push_log()
    changed = find_changed(log)
    if not changed:
        return 0, []

    ops = []
    paths_pushed = []
    for p in changed:
        rel = str(p.relative_to(LOCAL_ROOT))
        remote = f'{HF_PREFIX}/{rel}'
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(p)))
        paths_pushed.append(remote)

    api.create_commit(
        repo_id=REPO, repo_type='dataset',
        operations=ops,
        commit_message=f'M3 push @ {datetime.now().isoformat()} ({len(ops)} files)',
    )

    log['pushes'].append({
        'timestamp': datetime.now().isoformat(),
        'n_files': len(ops),
        'paths': paths_pushed[:10],  # log first 10 only
    })
    save_push_log(log)
    return len(ops), paths_pushed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=int, default=INTERVAL_DEFAULT,
                    help='seconds between pushes (default 600 = 10min)')
    ap.add_argument('--once', action='store_true', help='push once and exit')
    ap.add_argument('--max-iters', type=int, default=0, help='0 = forever')
    args = ap.parse_args()

    token = os.environ.get('HF_TOKEN')
    if not token:
        print('Set HF_TOKEN env var', file=sys.stderr)
        sys.exit(1)

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    def _term(*_):
        print(f'[scheduler] received signal; exiting at {datetime.now()}', flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    print(f'[scheduler] watching {LOCAL_ROOT}, push every {args.interval}s, repo={REPO}',
          flush=True)

    iter_n = 0
    while True:
        try:
            n, _paths = push_once(token)
            ts = datetime.now().strftime('%H:%M:%S')
            if n > 0:
                print(f'[{ts}] pushed {n} file(s)', flush=True)
            else:
                print(f'[{ts}] no changes', flush=True)
        except Exception as e:
            print(f'[push error] {type(e).__name__}: {str(e)[:200]}', flush=True)

        if args.once:
            break
        iter_n += 1
        if args.max_iters and iter_n >= args.max_iters:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
