#!/usr/bin/env python3
"""
Exp2 Revalidation Runner

Purpose
- Revalidate features marked causal in Exp2 final results with denser scales and
  invalid-trial exclusion, and log per-trial responses for audit.

Outputs (under analysis/)
- exp2_revalidation_summary.csv        # Per-feature revalidation metrics
- exp2_revalidation_suspects.csv       # Suspect features requiring further audit
- exp2_revalidation_trials.jsonl       # Per-trial logs (response, parse info)

CLI
  python analysis/exp2_revalidate.py \
    --results /data/llm_addiction/results/patching_population_mean_final_*.json \
    --top_k 50 --trials 30 --gpu 4 --seed 42

Notes
- Does not modify existing Exp2 code or results. Loads model once and reuses it.
- Uses PopulationMeanPatchingExperiment methods for feature extraction/generation.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

# Ensure we can import the experiment class
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'causal_feature_discovery' / 'src'))
from experiment_patching_population_mean import PopulationMeanPatchingExperiment  # type: ignore


def find_latest_final(path_hint: str) -> Path:
    if path_hint and Path(path_hint).exists():
        return Path(path_hint)
    pattern = '/data/llm_addiction/results/patching_population_mean_final_*.json'
    paths = sorted(glob(pattern))
    if not paths:
        raise SystemExit(f'No final results found matching: {pattern}')
    return Path(paths[-1])


def dedupe_features(causal_bet: List[Dict], causal_stop: List[Dict], top_k: int) -> List[Tuple[int, int]]:
    # Prefer larger effects and combine bet/stop lists
    merged = []
    for x in causal_bet:
        merged.append((x.get('bet_effect', 0.0), (int(x['layer']), int(x['feature_id'])), 'bet'))
    for x in causal_stop:
        merged.append((x.get('stop_effect', 0.0), (int(x['layer']), int(x['feature_id'])), 'stop'))
    # Sort by effect descending
    merged.sort(key=lambda t: t[0], reverse=True)
    seen = set()
    out: List[Tuple[int, int]] = []
    for _, key, _ in merged:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= top_k:
            break
    return out


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class TrialLog:
    layer: int
    feature_id: int
    prompt_type: str
    scale: float
    trial_index: int
    response: str
    decision: str
    bet: int
    source: str
    valid: bool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=str, default='', help='Path to Exp2 final results JSON (default: latest)')
    ap.add_argument('--top_k', type=int, default=50, help='Top-K causal features to revalidate')
    ap.add_argument('--trials', type=int, default=30, help='Trials per (feature, prompt, scale)')
    ap.add_argument('--gpu', type=str, default=None, help='GPU id to use (e.g., 4 or 5)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--scales', type=str, default='0.3,0.5,0.75,1.0,1.25,1.5,2.0')
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)

    results_path = find_latest_final(args.results)
    with open(results_path, 'r') as f:
        final_data = json.load(f)

    causal_bet = final_data.get('causal_features_bet', [])
    causal_stop = final_data.get('causal_features_stop', [])
    selected = dedupe_features(causal_bet, causal_stop, args.top_k)

    # Prepare experiment (reuse model/SAE)
    exp = PopulationMeanPatchingExperiment()
    exp.exclude_invalid = True  # enforce invalid exclusion
    # Override scales and trials
    scales = [float(s) for s in args.scales.split(',') if s]
    exp.scales = scales
    exp.n_trials = int(args.trials)

    exp.load_models()
    features_all = exp.load_features()
    # Index features by (layer, feature_id)
    idx = {(f['layer'], f['feature_id']): f for f in features_all}

    # Outputs
    out_dir = Path('analysis')
    trials_path = out_dir / 'exp2_revalidation_trials.jsonl'
    summary_path = out_dir / 'exp2_revalidation_summary.csv'
    suspects_path = out_dir / 'exp2_revalidation_suspects.csv'
    ensure_dir(trials_path)
    ensure_dir(summary_path)
    ensure_dir(suspects_path)

    # CSV headers
    with open(summary_path, 'w') as sf:
        sf.write('layer,feature_id,bet_range_risky,bet_range_safe,stop_range_risky,stop_range_safe,'
                 'spearman_bet_risky,spearman_bet_safe,spearman_stop_risky,spearman_stop_safe,'
                 'invalid_rate_risky,invalid_rate_safe,valid_trials_risky,valid_trials_safe\n')
    with open(suspects_path, 'w') as sf:
        sf.write('layer,feature_id,reason\n')

    # Revalidate each selected feature
    for (layer, fid) in selected:
        meta = idx.get((layer, fid))
        if not meta:
            print(f'[WARN] Feature missing in NPZ: L{layer}-{fid}, skipping')
            continue

        # Recompute original value on base prompts
        risky_prompt = exp.risky_prompt
        safe_prompt = exp.safe_prompt

        try:
            orig_risky = exp.extract_original_feature(risky_prompt, layer, fid)
        except Exception:
            orig_risky = float('nan')
        try:
            orig_safe = exp.extract_original_feature(safe_prompt, layer, fid)
        except Exception:
            orig_safe = float('nan')

        # Helper to compute patched value per scale (same rule as main exp)
        def patched_val(original_value: float, scale: float, bankrupt_mean: float, safe_mean: float) -> float:
            if scale < 1.0:
                return max(0.0, safe_mean + scale * (original_value - safe_mean))
            else:
                adj = scale - 1.0
                return max(0.0, original_value + adj * (bankrupt_mean - original_value))

        def run_prompt(prompt_type: str) -> Tuple[List[float], List[float], int, int]:
            prompt = risky_prompt if prompt_type == 'risky' else safe_prompt
            original_value = orig_risky if prompt_type == 'risky' else orig_safe
            bankrupt_mean = meta['bankrupt_mean']
            safe_mean = meta['safe_mean']

            avg_bets: List[float] = []
            stop_rates: List[float] = []
            total_valid = 0
            total_trials = 0

            for s in scales:
                pv = patched_val(float(original_value), float(s), float(bankrupt_mean), float(safe_mean))
                bets = []
                stops = 0
                valid_trials = 0
                total_trials += exp.n_trials

                for t in range(exp.n_trials):
                    try:
                        resp = exp.generate_with_patching(prompt, layer, fid, pv)
                        parsed = exp.parse_response(resp)
                        # trial log
                        tl = TrialLog(layer=layer, feature_id=fid, prompt_type=prompt_type, scale=float(s),
                                      trial_index=t, response=resp, decision=parsed['decision'], bet=int(parsed['bet']),
                                      source=parsed.get('source', 'unknown'), valid=bool(parsed.get('valid', True)))
                        with open(trials_path, 'a') as tf:
                            tf.write(json.dumps(tl.__dict__, ensure_ascii=False) + '\n')

                        if exp.exclude_invalid and not parsed.get('valid', True):
                            continue
                        valid_trials += 1
                        if parsed['decision'] == 'stop':
                            stops += 1
                            bets.append(0)
                        else:
                            bets.append(int(parsed['bet']))
                    except Exception as e:
                        # Log exception as invalid trial
                        tl = TrialLog(layer=layer, feature_id=fid, prompt_type=prompt_type, scale=float(s),
                                      trial_index=t, response=f'__EXC__:{e}', decision='invalid', bet=10,
                                      source='exception', valid=False)
                        with open(trials_path, 'a') as tf:
                            tf.write(json.dumps(tl.__dict__, ensure_ascii=False) + '\n')
                        continue

                total_valid += valid_trials
                if valid_trials > 0:
                    avg_bets.append(float(np.mean(bets)))
                    stop_rates.append(float(stops / valid_trials))
                else:
                    avg_bets.append(0.0)
                    stop_rates.append(0.0)

            return avg_bets, stop_rates, total_valid, total_trials

        # Run for both prompts
        risky_bets, risky_stops, risky_valid, risky_total = run_prompt('risky')
        safe_bets, safe_stops, safe_valid, safe_total = run_prompt('safe')

        # Compute monotonicity and effects
        def corr_and_range(xs: List[float]) -> Tuple[float, float]:
            try:
                rho, _ = spearmanr(scales, xs)
                rng = max(xs) - min(xs)
                return float(rho), float(rng)
            except Exception:
                return 0.0, 0.0

        rho_br, range_br = corr_and_range(risky_bets)
        rho_bs, range_bs = corr_and_range(safe_bets)
        rho_sr, range_sr = corr_and_range(risky_stops)
        rho_ss, range_ss = corr_and_range(safe_stops)

        inv_rate_r = 1.0 - (risky_valid / risky_total) if risky_total else 0.0
        inv_rate_s = 1.0 - (safe_valid / safe_total) if safe_total else 0.0

        with open(summary_path, 'a') as sf:
            sf.write(
                f"{layer},{fid},{range_br:.3f},{range_bs:.3f},{range_sr:.3f},{range_ss:.3f},"
                f"{rho_br:.3f},{rho_bs:.3f},{rho_sr:.3f},{rho_ss:.3f},"
                f"{inv_rate_r:.3f},{inv_rate_s:.3f},{risky_valid},{safe_valid}\n"
            )

        # Suspect heuristics
        reasons = []
        if max(abs(rho_br), abs(rho_bs)) < 0.3 and max(range_br, range_bs) < 5.0:
            reasons.append('weak_bet_effect')
        if max(abs(rho_sr), abs(rho_ss)) < 0.3 and max(range_sr, range_ss) < 0.05:
            reasons.append('weak_stop_effect')
        if inv_rate_r > 0.2 or inv_rate_s > 0.2:
            reasons.append('high_invalid_rate')
        if reasons:
            with open(suspects_path, 'a') as sf:
                sf.write(f"{layer},{fid},{'|'.join(reasons)}\n")

    print(f"\nRevalidation complete.\n- Summary: {summary_path}\n- Suspects: {suspects_path}\n- Trials:   {trials_path}")


if __name__ == '__main__':
    main()

