"""AMLT progress monitor for the four M3' / Track-B nodes.

Every interval (default 600s = 10 min):
  - Calls `amlt status <exp_name>` for each of the 4 experiments
  - Parses the status (preparing / queued / running / paused / failed / passed)
  - Pulls the latest progress.json from HF (count of completed trials per
    cell) so we know whether the GPU work is actually advancing
  - Auto-resume any paused job (using the same monkey-patch wrapper)
  - Logs to /home/v-seungplee/llm-addiction/sae_v3_analysis/logs/amlt_monitor.log

Designed to be left running with `nohup`. Stateless — no SSH required since
we observe progress via HF, which the remote launcher pushes every 10 min.

Usage:
    nohup python amlt_progress_monitor.py > monitor.out 2>&1 &
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis')
LOG_DIR = ROOT / 'logs'
LOG_FILE = LOG_DIR / 'amlt_monitor.log'

# 4 submitted experiments (auto-generated names from amlt run)
EXPERIMENTS = [
    {'exp': 'peaceful-lemming', 'job': 'm3prime_gemma_sm_n1',
     'role': 'gemma_sm_iba',
     'expected_cells': ['gemma_sm_alpha-2', 'gemma_sm_alpha-1', 'gemma_sm_alpha+0',
                        'gemma_sm_alpha+1', 'gemma_sm_alpha+2', 'gemma_sm_alpha+3',
                        'gemma_sm_random', 'gemma_sm_L8', 'gemma_sm_ILC']},
    {'exp': 'healthy-barnacle', 'job': 'm3prime_llama_sm_n1',
     'role': 'llama_sm_iba',
     'expected_cells': ['llama_sm_alpha-2', 'llama_sm_alpha-1', 'llama_sm_alpha+0',
                        'llama_sm_alpha+1', 'llama_sm_alpha+2', 'llama_sm_alpha+3',
                        'llama_sm_random', 'llama_sm_L8', 'llama_sm_ILC']},
    {'exp': 'free-louse', 'job': 'm3prime_gemma_xtask_n1',
     'role': 'gemma_xtask',
     'expected_cells': ['gemma_mw_alpha-1', 'gemma_mw_alpha+0', 'gemma_mw_alpha+2',
                        'gemma_ic_alpha-1', 'gemma_ic_alpha+0', 'gemma_ic_alpha+2']},
    {'exp': 'helpful-scorpion', 'job': 'm3prime_llama_xtask_n1',
     'role': 'llama_xtask',
     'expected_cells': ['llama_mw_alpha-1', 'llama_mw_alpha+0', 'llama_mw_alpha+2',
                        'llama_ic_alpha-1', 'llama_ic_alpha+0', 'llama_ic_alpha+2']},
]

INTERVAL_S = int(os.environ.get('MONITOR_INTERVAL_S', 600))
HF_REPO = 'llm-addiction-research/llm-addiction'
WRAPPER = ROOT / 'scripts' / 'amlt_submit_no_tty.py'


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def amlt_status(exp_name: str) -> dict:
    """Returns dict with status fields for the experiment."""
    try:
        out = subprocess.run(
            ['amlt', 'status', exp_name],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {'error': 'timeout', 'raw': ''}
    raw = out.stdout + out.stderr
    # Parse "STATUS" column from the table
    status = 'unknown'
    for line in raw.splitlines():
        s = line.strip()
        for token in ['preparing', 'queued', 'running', 'paused',
                      'failed', 'pass', 'killed', 'starting', 'cancelled']:
            if token in s.lower() and ':' in s:
                status = token
                break
        if status != 'unknown':
            break
    return {'status': status, 'raw': raw[-2000:]}


def fetch_hf_progress(role: str) -> dict:
    """Pull the M3prime_indicator_steering trial counts from HF for this role."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get('HF_TOKEN'))
        files = api.list_repo_files(HF_REPO, repo_type='dataset')
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}

    cells = {}
    prefix = 'sae_v3_analysis/results/v19_multi_patching/M3prime_indicator_steering/'
    for f in files:
        if not f.startswith(prefix):
            continue
        if 'trials.jsonl' not in f:
            continue
        # Extract cell name from path: .../{model}_{task}_{condition}_n{N}/trials.jsonl
        parts = f[len(prefix):].split('/')
        if len(parts) < 2:
            continue
        cell = parts[0]
        cells[cell] = f
    return {'cells_with_trials': sorted(cells.keys())}


def auto_resume_if_paused(exp_name: str) -> bool:
    """If experiment is paused, attempt to resume via amlt resume."""
    log(f'  attempting auto-resume on {exp_name}')
    try:
        out = subprocess.run(
            ['amlt', 'resume', exp_name],
            capture_output=True, text=True, timeout=180,
            stdin=subprocess.DEVNULL,
        )
        log(f'  resume output: {(out.stdout + out.stderr)[-500:]}')
        return True
    except Exception as e:
        log(f'  resume failed: {type(e).__name__}: {e}')
        return False


def one_cycle() -> None:
    log('==== monitor cycle ====')
    hf_progress = fetch_hf_progress('all')
    if 'error' in hf_progress:
        log(f'  HF progress fetch error: {hf_progress["error"]}')
        cells_present = set()
    else:
        cells_present = set(hf_progress.get('cells_with_trials', []))
        log(f'  HF cells with trials: {len(cells_present)}')

    for entry in EXPERIMENTS:
        st = amlt_status(entry['exp'])
        prog_count = sum(
            1 for c in entry['expected_cells']
            if any(c in cp for cp in cells_present)
        )
        total = len(entry['expected_cells'])
        log(f'  [{entry["exp"]:20s} role={entry["role"]:14s}] status={st.get("status","?"):10s} '
            f'progress={prog_count}/{total} cells with HF trials')
        if st.get('status') == 'paused':
            auto_resume_if_paused(entry['exp'])
        if st.get('status') in ('failed', 'killed', 'cancelled'):
            log(f'  ⚠ {entry["exp"]} is {st["status"]}; manual review recommended')


def main():
    log(f'==== monitor start (interval={INTERVAL_S}s) ====')
    while True:
        try:
            one_cycle()
        except Exception as e:
            log(f'  cycle exception: {type(e).__name__}: {e}')
        time.sleep(INTERVAL_S)


if __name__ == '__main__':
    main()
