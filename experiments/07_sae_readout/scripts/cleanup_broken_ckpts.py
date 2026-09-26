#!/usr/bin/env python3
"""Remove broken α entries (bk+stop < gate) from Phase 1 checkpoints.

Codex found: ckpt_A_llama_{sm,mw}.json + ckpt_C_llama_sm.json have α >= 0
entries where all games failed (bk=0, stop=0, n=200). These are artifacts
of the earlier ModuleNotFoundError run (llama_gemma_experiment missing).

This script filters `completed_alphas` to keep only entries where
`bk_count + stop_count > gate` (default 50), so rerun picks up clean alphas
only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def clean_ckpt(path: Path, gate: int = 50) -> tuple[int, int]:
    """Return (n_before, n_after). Rewrites the file in place."""
    d = json.load(open(path))
    alphas = d.get('completed_alphas', [])
    n_before = len(alphas)
    kept = []
    dropped = []
    for a in alphas:
        bk = a.get('bk_count', 0) or 0
        stop = a.get('stop_count', 0) or 0
        n_games = a.get('n_games', 0) or 0
        completion = bk + stop
        if n_games > 0 and completion > gate:
            kept.append(a)
        else:
            dropped.append((a.get('alpha'), completion, n_games))
    d['completed_alphas'] = kept
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    return n_before, len(kept), dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', default='/home/v-seungplee/llm-addiction/sae_v3_analysis/results/checkpoints')
    p.add_argument('--gate', type=int, default=50,
                   help='Minimum bk+stop count to consider α entry clean')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    print(f'Scanning {ckpt_dir} (gate={args.gate})...')
    total_dropped = 0
    for f in sorted(ckpt_dir.glob('ckpt_*.json')):
        d = json.load(open(f))
        alphas = d.get('completed_alphas', [])
        n_before = len(alphas)
        dropped_here = []
        for a in alphas:
            bk = a.get('bk_count', 0) or 0
            stop = a.get('stop_count', 0) or 0
            if bk + stop <= args.gate:
                dropped_here.append((a.get('alpha'), bk + stop, a.get('n_games')))
        if dropped_here:
            print(f'\n{f.name}: {n_before} alphas → dropping {len(dropped_here)}:')
            for alpha, completion, ng in dropped_here:
                print(f'  α={alpha:>+5.1f}: only {completion}/{ng} games completed')
            if not args.dry_run:
                n0, n1, dropped = clean_ckpt(f, args.gate)
                print(f'  → saved with {n1} alphas (was {n0})')
            total_dropped += len(dropped_here)
        else:
            print(f'{f.name}: clean ({n_before} alphas) — no change')
    if args.dry_run:
        print(f'\n[DRY RUN] Would drop {total_dropped} entries across broken ckpts')
    else:
        print(f'\nDropped {total_dropped} broken entries across all ckpts')


if __name__ == '__main__':
    main()
