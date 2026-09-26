#!/usr/bin/env python3
"""
Exp2 Simple Feature Associations

Purpose
- From stored Exp2 results, identify features most associated with:
  1) Stop/bankruptcy proxy (stop_rate, 1 - stop_rate)
  2) Bet amount (avg_bet)

Inputs
- Prefers: /data/llm_addiction/results/patching_population_mean_final_*.json
- Fallback: latest /data/llm_addiction/results/patching_intermediate_*.json (uses sample_results only)

Outputs (under analysis/)
- exp2_assoc_bet_topK.csv
- exp2_assoc_stop_topK.csv
- exp2_assoc_bankrupt_topK.csv

Notes
- Uses Spearman correlation between scale and metric per prompt (risky/safe),
  then takes max absolute correlation across prompts as the feature's score.
"""

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr


def find_results() -> Tuple[Path, str]:
    finals = sorted(glob('/data/llm_addiction/results/patching_population_mean_final_*.json'))
    if finals:
        return Path(finals[-1]), 'final'
    inters = sorted(glob('/data/llm_addiction/results/patching_intermediate_*.json'))
    if inters:
        return Path(inters[-1]), 'intermediate'
    raise SystemExit('No Exp2 results found under /data/llm_addiction/results')


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def load_records(path: Path, kind: str) -> List[Dict]:
    with open(path, 'r') as f:
        data = json.load(f)
    if kind == 'final':
        return data.get('all_results', [])
    # intermediate: only last 100 sample results
    return data.get('sample_results', [])


def group_by_feature(records: List[Dict]) -> Dict[Tuple[int, int], Dict[str, List[Tuple[float, float, float]]]]:
    # returns {(layer, feature_id): {prompt_type: [(scale, avg_bet, stop_rate), ...]}}
    out: Dict[Tuple[int, int], Dict[str, List[Tuple[float, float, float]]]] = {}
    for r in records:
        try:
            key = (int(r['layer']), int(r['feature_id']))
            pt = str(r['prompt_type'])
            s = float(r['scale'])
            ab = float(r['avg_bet'])
            sr = float(r['stop_rate'])
        except Exception:
            continue
        out.setdefault(key, {}).setdefault(pt, []).append((s, ab, sr))
    # sort by scale
    for key in out:
        for pt in out[key]:
            out[key][pt].sort(key=lambda t: t[0])
    return out


def corr_over_prompts(vals_by_prompt: Dict[str, List[Tuple[float, float, float]]], metric: str) -> float:
    # metric in {'bet','stop','bankrupt'}
    best = 0.0
    for pt, arr in vals_by_prompt.items():
        if len(arr) < 3:
            continue
        scales = [a[0] for a in arr]
        if metric == 'bet':
            ys = [a[1] for a in arr]
        elif metric == 'stop':
            ys = [a[2] for a in arr]
        else:  # bankrupt proxy
            ys = [1.0 - a[2] for a in arr]
        try:
            rho, _ = spearmanr(scales, ys)
            if rho is not None:
                best = max(best, abs(float(rho)))
        except Exception:
            pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=str, default='', help='Path to Exp2 results JSON (default: latest final or intermediate)')
    ap.add_argument('--top_k', type=int, default=30)
    args = ap.parse_args()

    if args.results:
        path = Path(args.results)
        kind = 'final'
        if not path.exists():
            raise SystemExit(f'Not found: {path}')
    else:
        path, kind = find_results()

    records = load_records(path, kind)
    if not records:
        raise SystemExit('No records found in results file')

    grouped = group_by_feature(records)

    rows_bet: List[Tuple[int, int, float]] = []
    rows_stop: List[Tuple[int, int, float]] = []
    rows_bank: List[Tuple[int, int, float]] = []

    for (layer, fid), vals_by_prompt in grouped.items():
        bet_score = corr_over_prompts(vals_by_prompt, 'bet')
        stop_score = corr_over_prompts(vals_by_prompt, 'stop')
        bank_score = corr_over_prompts(vals_by_prompt, 'bankrupt')
        rows_bet.append((layer, fid, bet_score))
        rows_stop.append((layer, fid, stop_score))
        rows_bank.append((layer, fid, bank_score))

    rows_bet.sort(key=lambda x: x[2], reverse=True)
    rows_stop.sort(key=lambda x: x[2], reverse=True)
    rows_bank.sort(key=lambda x: x[2], reverse=True)

    out_dir = Path('analysis')
    bet_path = out_dir / 'exp2_assoc_bet_topK.csv'
    stop_path = out_dir / 'exp2_assoc_stop_topK.csv'
    bank_path = out_dir / 'exp2_assoc_bankrupt_topK.csv'
    ensure_dir(bet_path)
    ensure_dir(stop_path)
    ensure_dir(bank_path)

    def write_top(path: Path, rows: List[Tuple[int, int, float]]):
        with open(path, 'w') as f:
            f.write('layer,feature_id,score_spearman_abs\n')
            for layer, fid, score in rows[: args.top_k]:
                f.write(f'{layer},{fid},{score:.3f}\n')

    write_top(bet_path, rows_bet)
    write_top(stop_path, rows_stop)
    write_top(bank_path, rows_bank)

    print('Done.')
    print(f'- Input: {path} ({kind})')
    print(f'- Outputs:')
    print(f'  {bet_path}')
    print(f'  {stop_path}')
    print(f'  {bank_path}')


if __name__ == '__main__':
    main()

