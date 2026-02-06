#!/usr/bin/env python3
"""
Re-evaluate Experiment 2 features using binary outcomes (stop vs continue).

Definitions
- stop: bet == 0
- continue: bet > 0

Comparisons (one-sided tests)
- Safe prompt (risk-increasing detection): safe_risky vs safe_safe
  H1: stop_rate(safe_risky) < stop_rate(safe_safe)  [continue↑]

- Risky prompt (risk-decreasing detection): risky_safe vs risky_risky
  H1: stop_rate(risky_safe) > stop_rate(risky_risky) [stop↑]

Criteria
- Significance: p < alpha (default 0.05)
- Effect size threshold: |Δ_stop| >= delta_threshold (default 0.30 i.e., 30 percentage points)

Inputs
- Defaults to latest /data/llm_addiction/results/exp2_corrected_final_*.json
- Or pass --input <path>

Outputs
- analysis/exp2_binary_causality_summary_<timestamp>.json
- analysis/exp2_binary_causality_summary_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple


def find_latest_final_json(default_dir: str) -> str | None:
    paths = sorted(glob(os.path.join(default_dir, 'exp2_corrected_final_*.json')))
    return paths[-1] if paths else None


def stop_counts(bets: List[float]) -> Tuple[int, int]:
    stops = sum(1 for b in bets if (isinstance(b, (int, float)) and b == 0))
    n = len(bets)
    return stops, n


def two_prop_z_test(k1: int, n1: int, k2: int, n2: int, alternative: str) -> float:
    """One-sided two-proportion z-test p-value.
    alternative in {'less','greater'} tests p1 < p2 or p1 > p2.
    Returns p-value.
    """
    if n1 == 0 or n2 == 0:
        return 1.0
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2) + 1e-12)
    if denom == 0:
        return 1.0
    z = (p1 - p2) / denom

    # Normal CDF approximation
    try:
        # Using error function for CDF
        def phi(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        if alternative == 'less':
            return phi(z)
        elif alternative == 'greater':
            return 1.0 - phi(z)
        else:
            # two-sided
            p = 2.0 * min(phi(z), 1.0 - phi(z))
            return p
    except Exception:
        return 1.0


@dataclass
class FeatureBinaryResult:
    layer: int
    feature_id: int
    original_direction: str | None
    stop_rate_safe_safe: float
    stop_rate_safe_risky: float
    stop_rate_risky_safe: float
    stop_rate_risky_risky: float
    delta_stop_safe: float  # safe_risky - safe_safe (expect negative for risk-increasing)
    delta_stop_risky: float  # risky_safe - risky_risky (expect positive for risk-decreasing)
    p_safe: float
    p_risky: float
    risk_increasing_by_binary: bool
    risk_decreasing_by_binary: bool


def evaluate_feature_binary(feature: Dict, alpha: float, delta_thr: float) -> FeatureBinaryResult:
    res = feature['results']
    s_s_bets = res['safe_safe']['bets']
    s_r_bets = res['safe_risky']['bets']
    r_s_bets = res['risky_safe']['bets']
    r_r_bets = res['risky_risky']['bets']

    k_ss, n_ss = stop_counts(s_s_bets)
    k_sr, n_sr = stop_counts(s_r_bets)
    k_rs, n_rs = stop_counts(r_s_bets)
    k_rr, n_rr = stop_counts(r_r_bets)

    p_ss = k_ss / n_ss if n_ss else 0.0
    p_sr = k_sr / n_sr if n_sr else 0.0
    p_rs = k_rs / n_rs if n_rs else 0.0
    p_rr = k_rr / n_rr if n_rr else 0.0

    # One-sided tests
    pval_safe = two_prop_z_test(k_sr, n_sr, k_ss, n_ss, alternative='less')  # stop(safe_risky) < stop(safe_safe)
    pval_risky = two_prop_z_test(k_rs, n_rs, k_rr, n_rr, alternative='greater')  # stop(risky_safe) > stop(risky_risky)

    delta_safe = p_sr - p_ss
    delta_risky = p_rs - p_rr

    inc = (delta_safe <= -delta_thr) and (pval_safe < alpha)
    dec = (delta_risky >= delta_thr) and (pval_risky < alpha)

    return FeatureBinaryResult(
        layer=int(feature.get('layer')),
        feature_id=int(feature.get('feature_id')),
        original_direction=feature.get('direction') or feature.get('original_direction'),
        stop_rate_safe_safe=p_ss,
        stop_rate_safe_risky=p_sr,
        stop_rate_risky_safe=p_rs,
        stop_rate_risky_risky=p_rr,
        delta_stop_safe=delta_safe,
        delta_stop_risky=delta_risky,
        p_safe=pval_safe,
        p_risky=pval_risky,
        risk_increasing_by_binary=inc,
        risk_decreasing_by_binary=dec,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--delta_stop', type=float, default=0.30, help='Minimum absolute stop-rate change (e.g., 0.30 for 30pp)')
    ap.add_argument('--output_prefix', type=str, default='analysis/exp2_binary_causality_summary')
    args = ap.parse_args()

    input_path = args.input
    if not input_path:
        input_path = find_latest_final_json('/data/llm_addiction/results')
        if not input_path:
            raise SystemExit('No exp2_corrected_final_*.json found under /data/llm_addiction/results')

    with open(input_path, 'r') as f:
        data = json.load(f)

    results = data.get('all_results') or []
    if not results:
        raise SystemExit('No all_results in input JSON')

    out: List[FeatureBinaryResult] = []
    for feat in results:
        out.append(evaluate_feature_binary(feat, alpha=args.alpha, delta_thr=args.delta_stop))

    # Aggregates
    inc = [o for o in out if o.risk_increasing_by_binary]
    dec = [o for o in out if o.risk_decreasing_by_binary]

    def layer_counts(items: List[FeatureBinaryResult]) -> Dict[int, int]:
        d: Dict[int, int] = {}
        for it in items:
            d[it.layer] = d.get(it.layer, 0) + 1
        return d

    summary = {
        'input_file': input_path,
        'alpha': args.alpha,
        'delta_stop_threshold': args.delta_stop,
        'n_features': len(out),
        'risk_increasing_n': len(inc),
        'risk_decreasing_n': len(dec),
        'risk_increasing_by_layer': layer_counts(inc),
        'risk_decreasing_by_layer': layer_counts(dec),
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_out = f"{args.output_prefix}_{ts}.json"
    csv_out = f"{args.output_prefix}_{ts}.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(json_out), exist_ok=True)

    # Write JSON
    with open(json_out, 'w') as f:
        json.dump({
            'summary': summary,
            'features': [asdict(o) for o in out],
            'top_risk_increasing_by_delta_safe': [
                asdict(o) for o in sorted(inc, key=lambda x: x.delta_stop_safe)[:10]
            ],
            'top_risk_decreasing_by_delta_risky': [
                asdict(o) for o in sorted(dec, key=lambda x: -x.delta_stop_risky)[:10]
            ],
        }, f, indent=2)

    # Write CSV
    import csv
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'layer', 'feature_id', 'original_direction',
            'stop_rate_safe_safe', 'stop_rate_safe_risky', 'delta_stop_safe', 'p_safe', 'risk_increasing_by_binary',
            'stop_rate_risky_safe', 'stop_rate_risky_risky', 'delta_stop_risky', 'p_risky', 'risk_decreasing_by_binary'
        ])
        for o in out:
            w.writerow([
                o.layer, o.feature_id, o.original_direction,
                f"{o.stop_rate_safe_safe:.3f}", f"{o.stop_rate_safe_risky:.3f}", f"{o.delta_stop_safe:.3f}", f"{o.p_safe:.4f}", o.risk_increasing_by_binary,
                f"{o.stop_rate_risky_safe:.3f}", f"{o.stop_rate_risky_risky:.3f}", f"{o.delta_stop_risky:.3f}", f"{o.p_risky:.4f}", o.risk_decreasing_by_binary,
            ])

    print("\nBinary-outcome re-evaluation complete.")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved JSON: {json_out}")
    print(f"Saved CSV : {csv_out}")


if __name__ == '__main__':
    main()
