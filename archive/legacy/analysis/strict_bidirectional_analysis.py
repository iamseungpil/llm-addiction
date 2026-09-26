#!/usr/bin/env python3
"""
엄격한 양방향 일관성 분석:
Safe/Risky feature는 두 patch 모두에서 일관된 방향성을 보여야 함
"""

import pandas as pd
from pathlib import Path

def main():
    csv_path = Path("/home/ubuntu/llm_addiction/analysis/exp2_L1_31_ALL_LAYERS_feature_group_summary_BASELINE.csv")
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("엄격한 양방향 일관성 분석 (Strict Bidirectional Consistency)")
    print("=" * 80)

    # 기존 분류
    original_safe = len(df[df['classified_as'] == 'safe'])
    original_risky = len(df[df['classified_as'] == 'risky'])

    print(f"\n=== 기존 분류 (현재 로직) ===")
    print(f"Safe features: {original_safe}")
    print(f"Risky features: {original_risky}")
    print(f"Total causal: {original_safe + original_risky}")

    # 새로운 엄격한 기준
    print(f"\n=== 새로운 엄격한 기준 ===")
    print(f"\n✅ STRICT Safe Feature (양방향 일관성):")
    print(f"   1. safe_bankruptcy_delta < 0 (safe patch → bankruptcy 감소)")
    print(f"   2. risky_bankruptcy_delta > 0 (risky patch → bankruptcy 증가)")
    print(f"   → 즉, 두 patch가 반대 방향으로 bankruptcy에 영향")

    print(f"\n✅ STRICT Risky Feature (양방향 일관성):")
    print(f"   1. safe_bankruptcy_delta > 0 (safe patch → bankruptcy 증가)")
    print(f"   2. risky_bankruptcy_delta < 0 (risky patch → bankruptcy 감소)")
    print(f"   → 즉, safe feature와 정반대 패턴")

    # 새로운 분류 적용
    strict_safe = df[
        (df['safe_bankruptcy_delta'] < 0) &  # safe patch → bankruptcy 감소
        (df['risky_bankruptcy_delta'] > 0)   # risky patch → bankruptcy 증가
    ].copy()

    strict_risky = df[
        (df['safe_bankruptcy_delta'] > 0) &  # safe patch → bankruptcy 증가
        (df['risky_bankruptcy_delta'] < 0)   # risky patch → bankruptcy 감소
    ].copy()

    print(f"\n=== 엄격한 기준 적용 결과 ===")
    print(f"STRICT Safe features: {len(strict_safe)} (기존 {original_safe}에서 {len(strict_safe) - original_safe:+d})")
    print(f"STRICT Risky features: {len(strict_risky)} (기존 {original_risky}에서 {len(strict_risky) - original_risky:+d})")
    print(f"Total STRICT causal: {len(strict_safe) + len(strict_risky)}")

    # 추가 분석: 단방향만 효과 있는 features
    unidirectional_safe = df[
        (df['classified_as'] == 'safe') &
        ~df['feature'].isin(strict_safe['feature'])
    ]

    unidirectional_risky = df[
        (df['classified_as'] == 'risky') &
        ~df['feature'].isin(strict_risky['feature'])
    ]

    print(f"\n=== 단방향 효과만 있는 Features ===")
    print(f"Safe patch만 효과: {len(unidirectional_safe)}")
    print(f"Risky patch만 효과: {len(unidirectional_risky)}")

    # Layer-wise 분석
    print(f"\n=== Layer별 STRICT 분류 (L25-31) ===")
    for layer in range(25, 32):
        layer_strict_safe = strict_safe[strict_safe['feature'].str.startswith(f'L{layer}-')]
        layer_strict_risky = strict_risky[strict_risky['feature'].str.startswith(f'L{layer}-')]
        layer_total = len(layer_strict_safe) + len(layer_strict_risky)

        # 기존 분류
        layer_df = df[df['feature'].str.startswith(f'L{layer}-')]
        layer_orig_safe = len(layer_df[layer_df['classified_as'] == 'safe'])
        layer_orig_risky = len(layer_df[layer_df['classified_as'] == 'risky'])

        print(f"L{layer}: strict_safe={len(layer_strict_safe):3d} (원래 {layer_orig_safe:3d}), "
              f"strict_risky={len(layer_strict_risky):3d} (원래 {layer_orig_risky:3d}), "
              f"strict_total={layer_total:3d}")

    # 전체 L1-31 summary
    print(f"\n=== 전체 L1-31 STRICT 분류 ===")
    for layer in range(1, 32):
        layer_strict_safe = strict_safe[strict_safe['feature'].str.startswith(f'L{layer}-')]
        layer_strict_risky = strict_risky[strict_risky['feature'].str.startswith(f'L{layer}-')]
        layer_total = len(layer_strict_safe) + len(layer_strict_risky)

        if layer_total > 0:
            print(f"L{layer:2d}: safe={len(layer_strict_safe):3d}, risky={len(layer_strict_risky):3d}, "
                  f"total={layer_total:3d}")

    # 통계 분석
    print(f"\n=== 효과 크기 분석 ===")
    print(f"\nSTRICT Safe features (평균 효과):")
    print(f"  safe_bankruptcy_delta: {strict_safe['safe_bankruptcy_delta'].mean():.4f}")
    print(f"  risky_bankruptcy_delta: {strict_safe['risky_bankruptcy_delta'].mean():.4f}")
    print(f"  safe_stop_delta: {strict_safe['safe_stop_delta'].mean():.4f}")
    print(f"  risky_stop_delta: {strict_safe['risky_stop_delta'].mean():.4f}")

    print(f"\nSTRICT Risky features (평균 효과):")
    print(f"  safe_bankruptcy_delta: {strict_risky['safe_bankruptcy_delta'].mean():.4f}")
    print(f"  risky_bankruptcy_delta: {strict_risky['risky_bankruptcy_delta'].mean():.4f}")
    print(f"  safe_stop_delta: {strict_risky['safe_stop_delta'].mean():.4f}")
    print(f"  risky_stop_delta: {strict_risky['risky_stop_delta'].mean():.4f}")

    # CSV 저장
    output_csv_safe = Path("/home/ubuntu/llm_addiction/analysis/STRICT_safe_features_bidirectional.csv")
    output_csv_risky = Path("/home/ubuntu/llm_addiction/analysis/STRICT_risky_features_bidirectional.csv")

    strict_safe.to_csv(output_csv_safe, index=False)
    strict_risky.to_csv(output_csv_risky, index=False)

    print(f"\n✅ STRICT feature lists saved:")
    print(f"   Safe: {output_csv_safe}")
    print(f"   Risky: {output_csv_risky}")

    # 최종 요약
    print(f"\n{'=' * 80}")
    print(f"최종 요약:")
    print(f"  기존 causal features: {original_safe + original_risky}")
    print(f"  STRICT causal features (양방향 일관성): {len(strict_safe) + len(strict_risky)}")
    print(f"  차이: {(len(strict_safe) + len(strict_risky)) - (original_safe + original_risky):+d}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
