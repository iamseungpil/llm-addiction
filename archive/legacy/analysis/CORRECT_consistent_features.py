#!/usr/bin/env python3
"""
올바른 일관성 기준:
Safe feature는 양쪽 patch 모두 '안전' 방향으로 효과
Risky feature는 양쪽 patch 모두 '위험' 방향으로 효과
"""

import pandas as pd
from pathlib import Path

def main():
    csv_path = Path("/home/ubuntu/llm_addiction/analysis/exp2_L1_31_ALL_LAYERS_feature_group_summary_BASELINE.csv")
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("올바른 일관성 기준 (CORRECT Consistency)")
    print("=" * 80)

    print("\n=== Safe Feature 정의 (일관되게 '안전') ===")
    print("Safe patch:")
    print("  - Safe context: stop↑ (safe_stop_delta > 0)")
    print("  - Risky context: bankruptcy↓ (safe_bankruptcy_delta < 0)")
    print("Risky patch:")
    print("  - Safe context: stop↑ (risky_stop_delta > 0)")
    print("  - Risky context: bankruptcy↓ (risky_bankruptcy_delta < 0)")
    print("\n→ 두 patch 모두 '안전' 방향으로 일관")

    print("\n=== Risky Feature 정의 (일관되게 '위험') ===")
    print("Safe patch:")
    print("  - Safe context: stop↓ (safe_stop_delta < 0)")
    print("  - Risky context: bankruptcy↑ (safe_bankruptcy_delta > 0)")
    print("Risky patch:")
    print("  - Safe context: stop↓ (risky_stop_delta < 0)")
    print("  - Risky context: bankruptcy↑ (risky_bankruptcy_delta > 0)")
    print("\n→ 두 patch 모두 '위험' 방향으로 일관")

    # CORRECT Safe features
    correct_safe = df[
        (df['safe_stop_delta'] > 0) &           # safe patch → stop↑ (안전)
        (df['safe_bankruptcy_delta'] < 0) &     # safe patch → bankruptcy↓ (안전)
        (df['risky_stop_delta'] > 0) &          # risky patch → stop↑ (안전)
        (df['risky_bankruptcy_delta'] < 0)      # risky patch → bankruptcy↓ (안전)
    ].copy()

    # CORRECT Risky features
    correct_risky = df[
        (df['safe_stop_delta'] < 0) &           # safe patch → stop↓ (위험)
        (df['safe_bankruptcy_delta'] > 0) &     # safe patch → bankruptcy↑ (위험)
        (df['risky_stop_delta'] < 0) &          # risky patch → stop↓ (위험)
        (df['risky_bankruptcy_delta'] > 0)      # risky patch → bankruptcy↑ (위험)
    ].copy()

    print(f"\n{'=' * 80}")
    print(f"결과:")
    print(f"  CORRECT Safe features: {len(correct_safe)}")
    print(f"  CORRECT Risky features: {len(correct_risky)}")
    print(f"  Total: {len(correct_safe) + len(correct_risky)}")
    print(f"{'=' * 80}")

    # Layer별 분포 (전체)
    def get_layer(feature_str):
        return int(feature_str.split('-')[0][1:])

    correct_safe['layer'] = correct_safe['feature'].apply(get_layer)
    correct_risky['layer'] = correct_risky['feature'].apply(get_layer)

    safe_counts = correct_safe['layer'].value_counts().sort_index()
    risky_counts = correct_risky['layer'].value_counts().sort_index()

    all_layers = sorted(set(safe_counts.index) | set(risky_counts.index))

    print(f"\n=== Layer별 분포 (전체 L1-31) ===")
    for layer in all_layers:
        s = safe_counts.get(layer, 0)
        r = risky_counts.get(layer, 0)
        print(f"L{layer:2d}: safe={s:3d}, risky={r:3d}, total={s+r:3d}")

    # L25-31만
    print(f"\n=== Layer별 분포 (L25-31만) ===")
    for layer in range(25, 32):
        s = safe_counts.get(layer, 0)
        r = risky_counts.get(layer, 0)
        if s > 0 or r > 0:
            print(f"L{layer}: safe={s:3d}, risky={r:3d}, total={s+r:3d}")

    # 평균 효과
    print(f"\n=== CORRECT Safe Features 평균 효과 (n={len(correct_safe)}) ===")
    if len(correct_safe) > 0:
        print(f"Safe patch:")
        print(f"  safe_stop_delta: {correct_safe['safe_stop_delta'].mean():+.4f}")
        print(f"  safe_bankruptcy_delta: {correct_safe['safe_bankruptcy_delta'].mean():+.4f}")
        print(f"Risky patch:")
        print(f"  risky_stop_delta: {correct_safe['risky_stop_delta'].mean():+.4f}")
        print(f"  risky_bankruptcy_delta: {correct_safe['risky_bankruptcy_delta'].mean():+.4f}")

    print(f"\n=== CORRECT Risky Features 평균 효과 (n={len(correct_risky)}) ===")
    if len(correct_risky) > 0:
        print(f"Safe patch:")
        print(f"  safe_stop_delta: {correct_risky['safe_stop_delta'].mean():+.4f}")
        print(f"  safe_bankruptcy_delta: {correct_risky['safe_bankruptcy_delta'].mean():+.4f}")
        print(f"Risky patch:")
        print(f"  risky_stop_delta: {correct_risky['risky_stop_delta'].mean():+.4f}")
        print(f"  risky_bankruptcy_delta: {correct_risky['risky_bankruptcy_delta'].mean():+.4f}")

    # CSV 저장
    output_safe = Path("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv")
    output_risky = Path("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv")

    correct_safe.to_csv(output_safe, index=False)
    correct_risky.to_csv(output_risky, index=False)

    print(f"\n✅ Saved:")
    print(f"   {output_safe}")
    print(f"   {output_risky}")

    print(f"\n{'=' * 80}")
    print(f"최종 요약:")
    print(f"  CORRECT Safe (양쪽 patch 모두 안전 효과): {len(correct_safe)}")
    print(f"  CORRECT Risky (양쪽 patch 모두 위험 효과): {len(correct_risky)}")
    print(f"  Total: {len(correct_safe) + len(correct_risky)}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
