#!/usr/bin/env python3
"""
진정한 4-way 일관성 분석:
Safe/Risky feature는 두 prompt (safe/risky) × 두 patch (safe/risky) 모두에서 일관된 효과
"""

import pandas as pd
from pathlib import Path

def main():
    csv_path = Path("/home/ubuntu/llm_addiction/analysis/exp2_L1_31_ALL_LAYERS_feature_group_summary_BASELINE.csv")
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("진정한 4-WAY 일관성 분석 (True 4-way Consistency)")
    print("=" * 80)

    print("\n=== 측정 대상 재확인 ===")
    print("Safe prompt (Fixed betting, balance=100):")
    print("  - safe_stop_delta: safe patch → stop rate 변화")
    print("  - risky_stop_delta: risky patch → stop rate 변화")
    print("\nRisky prompt (Variable betting, balance=30):")
    print("  - safe_bankruptcy_delta: safe patch → bankruptcy rate 변화")
    print("  - risky_bankruptcy_delta: risky patch → bankruptcy rate 변화")

    # 진정한 Safe feature 기준
    print("\n=== TRUE Safe Feature (4-way 일관성) ===")
    print("1. Safe prompt에서:")
    print("   - safe patch → stop↑ (safe_stop_delta > 0)")
    print("   - risky patch → stop↓ (risky_stop_delta < 0)")
    print("2. Risky prompt에서:")
    print("   - safe patch → bankruptcy↓ (safe_bankruptcy_delta < 0)")
    print("   - risky patch → bankruptcy↑ (risky_bankruptcy_delta > 0)")

    print("\n=== TRUE Risky Feature (4-way 일관성) ===")
    print("1. Safe prompt에서:")
    print("   - safe patch → stop↓ (safe_stop_delta < 0)")
    print("   - risky patch → stop↑ (risky_stop_delta > 0)")
    print("2. Risky prompt에서:")
    print("   - safe patch → bankruptcy↑ (safe_bankruptcy_delta > 0)")
    print("   - risky patch → bankruptcy↓ (risky_bankruptcy_delta < 0)")

    # TRUE Safe feature 분류
    true_safe = df[
        (df['safe_stop_delta'] > 0) &           # safe patch → safe prompt에서 stop 증가
        (df['risky_stop_delta'] < 0) &          # risky patch → safe prompt에서 stop 감소
        (df['safe_bankruptcy_delta'] < 0) &     # safe patch → risky prompt에서 bankruptcy 감소
        (df['risky_bankruptcy_delta'] > 0)      # risky patch → risky prompt에서 bankruptcy 증가
    ].copy()

    # TRUE Risky feature 분류
    true_risky = df[
        (df['safe_stop_delta'] < 0) &           # safe patch → safe prompt에서 stop 감소
        (df['risky_stop_delta'] > 0) &          # risky patch → safe prompt에서 stop 증가
        (df['safe_bankruptcy_delta'] > 0) &     # safe patch → risky prompt에서 bankruptcy 증가
        (df['risky_bankruptcy_delta'] < 0)      # risky patch → risky prompt에서 bankruptcy 감소
    ].copy()

    # 기존 분류 비교
    original_safe = len(df[df['classified_as'] == 'safe'])
    original_risky = len(df[df['classified_as'] == 'risky'])

    print(f"\n=== 결과 비교 ===")
    print(f"기존 Safe features: {original_safe}")
    print(f"TRUE Safe features (4-way): {len(true_safe)} ({len(true_safe) - original_safe:+d})")

    print(f"\n기존 Risky features: {original_risky}")
    print(f"TRUE Risky features (4-way): {len(true_risky)} ({len(true_risky) - original_risky:+d})")

    print(f"\n기존 Total causal: {original_safe + original_risky}")
    print(f"TRUE Total causal (4-way): {len(true_safe) + len(true_risky)}")
    print(f"차이: {(len(true_safe) + len(true_risky)) - (original_safe + original_risky):+d}")

    # Layer별 분석 (L25-31)
    print(f"\n=== Layer별 TRUE 4-way 분류 (L25-31) ===")
    for layer in range(25, 32):
        layer_true_safe = true_safe[true_safe['feature'].str.startswith(f'L{layer}-')]
        layer_true_risky = true_risky[true_risky['feature'].str.startswith(f'L{layer}-')]
        layer_total = len(layer_true_safe) + len(layer_true_risky)

        # 기존 분류
        layer_df = df[df['feature'].str.startswith(f'L{layer}-')]
        layer_orig_safe = len(layer_df[layer_df['classified_as'] == 'safe'])
        layer_orig_risky = len(layer_df[layer_df['classified_as'] == 'risky'])

        print(f"L{layer}: TRUE_safe={len(layer_true_safe):3d} (원래 {layer_orig_safe:3d}), "
              f"TRUE_risky={len(layer_true_risky):3d} (원래 {layer_orig_risky:3d}), "
              f"TRUE_total={layer_total:3d}")

    # 전체 L1-31
    print(f"\n=== 전체 L1-31 TRUE 4-way 분류 ===")
    for layer in range(1, 32):
        layer_true_safe = true_safe[true_safe['feature'].str.startswith(f'L{layer}-')]
        layer_true_risky = true_risky[true_risky['feature'].str.startswith(f'L{layer}-')]
        layer_total = len(layer_true_safe) + len(layer_true_risky)

        if layer_total > 0:
            print(f"L{layer:2d}: safe={len(layer_true_safe):3d}, risky={len(layer_true_risky):3d}, "
                  f"total={layer_total:3d}")

    # 효과 크기 분석
    print(f"\n=== TRUE Safe Features 평균 효과 ===")
    if len(true_safe) > 0:
        print(f"Safe prompt (fixed betting):")
        print(f"  safe_stop_delta: {true_safe['safe_stop_delta'].mean():+.4f}")
        print(f"  risky_stop_delta: {true_safe['risky_stop_delta'].mean():+.4f}")
        print(f"Risky prompt (variable betting):")
        print(f"  safe_bankruptcy_delta: {true_safe['safe_bankruptcy_delta'].mean():+.4f}")
        print(f"  risky_bankruptcy_delta: {true_safe['risky_bankruptcy_delta'].mean():+.4f}")

    print(f"\n=== TRUE Risky Features 평균 효과 ===")
    if len(true_risky) > 0:
        print(f"Safe prompt (fixed betting):")
        print(f"  safe_stop_delta: {true_risky['safe_stop_delta'].mean():+.4f}")
        print(f"  risky_stop_delta: {true_risky['risky_stop_delta'].mean():+.4f}")
        print(f"Risky prompt (variable betting):")
        print(f"  safe_bankruptcy_delta: {true_risky['safe_bankruptcy_delta'].mean():+.4f}")
        print(f"  risky_bankruptcy_delta: {true_risky['risky_bankruptcy_delta'].mean():+.4f}")

    # 부분 일관성 분석
    print(f"\n=== 부분 일관성 Features ===")

    # Safe prompt에서만 일관성 (2-way)
    safe_prompt_only = df[
        (df['safe_stop_delta'] > 0) &
        (df['risky_stop_delta'] < 0) &
        ~df['feature'].isin(true_safe['feature'])
    ]

    # Risky prompt에서만 일관성 (2-way)
    risky_prompt_only = df[
        (df['safe_bankruptcy_delta'] < 0) &
        (df['risky_bankruptcy_delta'] > 0) &
        ~df['feature'].isin(true_safe['feature'])
    ]

    print(f"Safe prompt only (stop 일관성): {len(safe_prompt_only)}")
    print(f"Risky prompt only (bankruptcy 일관성): {len(risky_prompt_only)}")

    # CSV 저장
    output_csv_safe = Path("/home/ubuntu/llm_addiction/analysis/TRUE_4way_safe_features.csv")
    output_csv_risky = Path("/home/ubuntu/llm_addiction/analysis/TRUE_4way_risky_features.csv")

    true_safe.to_csv(output_csv_safe, index=False)
    true_risky.to_csv(output_csv_risky, index=False)

    print(f"\n✅ TRUE 4-way feature lists saved:")
    print(f"   Safe: {output_csv_safe}")
    print(f"   Risky: {output_csv_risky}")

    # 최종 요약
    print(f"\n{'=' * 80}")
    print(f"최종 요약:")
    print(f"  기존 causal features: {original_safe + original_risky}")
    print(f"  TRUE 4-way causal features: {len(true_safe) + len(true_risky)}")
    print(f"  차이: {(len(true_safe) + len(true_risky)) - (original_safe + original_risky):+d}")
    print(f"\n  TRUE Safe (4-way 일관성): {len(true_safe)}")
    print(f"  TRUE Risky (4-way 일관성): {len(true_risky)}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
