#!/usr/bin/env python3
"""
Word-Feature Risky/Safe Association Analysis
어떤 단어들이 risky feature vs safe feature와 연관되어 있는지 분석
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_phase5_classifications():
    """Phase 5에서 risky/safe features 분류"""
    print("=== Phase 5: Risky/Safe Feature 분류 ===\n")

    risky_features = set()
    safe_features = set()

    # GPU 4-7의 Phase 5 데이터 로드
    for gpu_id in [4, 5, 6, 7]:
        file_path = f'/data/llm_addiction/experiment_pathway_token_analysis/results/phase5_prompt_feature_full/prompt_feature_correlation_gpu{gpu_id}.json'

        with open(file_path, 'r') as f:
            data = json.load(f)

        for comp in data['feature_comparisons']:
            # 통계적으로 유의미하고 (p < 0.05)
            if comp['p_value'] < 0.05:
                if comp['cohens_d'] > 0.2:  # Risky: Cohen's d > 0.2
                    risky_features.add(comp['feature'])
                elif comp['cohens_d'] < -0.2:  # Safe: Cohen's d < -0.2
                    safe_features.add(comp['feature'])

    print(f"총 Risky features: {len(risky_features)}개 (Cohen's d > 0.2, p < 0.05)")
    print(f"총 Safe features: {len(safe_features)}개 (Cohen's d < -0.2, p < 0.05)")
    print()

    return risky_features, safe_features

def load_phase4_word_associations():
    """Phase 4에서 단어-feature 연관관계 로드"""
    print("=== Phase 4: 단어-Feature 연관관계 로드 ===\n")

    word_feature_associations = []

    # GPU 4-7의 Phase 4 데이터 로드
    for gpu_id in [4, 5, 6, 7]:
        file_path = f'/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_feature_full/word_feature_correlation_gpu{gpu_id}.json'

        with open(file_path, 'r') as f:
            data = json.load(f)

        # word_feature_correlations에서 연관관계 추출
        for corr in data['word_feature_correlations']:
            word_feature_associations.append({
                'word': corr['word'],
                'feature': corr['feature'],
                'mean_activation': corr['mean_activation'],
                'std_activation': corr['std_activation'],
                'word_count': corr['word_count']
            })

    print(f"총 단어-feature 연관관계: {len(word_feature_associations):,}개")
    print()

    return word_feature_associations

def analyze_word_risky_safe_associations(word_feature_assocs, risky_features, safe_features):
    """단어별 risky/safe feature 연관도 분석"""
    print("=== 단어별 Risky/Safe Feature 연관도 분석 ===\n")

    # 단어별 risky/safe feature 연관 강도 집계
    word_risky_scores = defaultdict(list)
    word_safe_scores = defaultdict(list)

    for assoc in word_feature_assocs:
        word = assoc['word']
        feature = assoc['feature']
        activation = assoc['mean_activation']

        if feature in risky_features:
            word_risky_scores[word].append(activation)
        elif feature in safe_features:
            word_safe_scores[word].append(activation)

    # 평균 activation 계산
    word_risky_summary = {
        word: {
            'mean_activation': np.mean(scores),
            'count': len(scores)
        }
        for word, scores in word_risky_scores.items()
    }

    word_safe_summary = {
        word: {
            'mean_activation': np.mean(scores),
            'count': len(scores)
        }
        for word, scores in word_safe_scores.items()
    }

    return word_risky_summary, word_safe_summary

def print_top_words(word_summary, label, top_n=50):
    """상위 N개 단어 출력"""
    print(f"\n=== {label} 상위 {top_n}개 단어 ===")
    print(f"(mean_activation 높은 순서)\n")

    # mean_activation 높은 순으로 정렬
    sorted_words = sorted(
        word_summary.items(),
        key=lambda x: x[1]['mean_activation'],
        reverse=True
    )[:top_n]

    for i, (word, stats) in enumerate(sorted_words, 1):
        print(f"{i:2d}. '{word}' "
              f"(activation={stats['mean_activation']:.4f}, "
              f"feature 개수={stats['count']})")

    return sorted_words

def analyze_risky_safe_word_differences(word_risky_summary, word_safe_summary):
    """Risky와 Safe에 공통으로 나타나는 단어들의 차이 분석"""
    print("\n\n=== Risky vs Safe 공통 단어 차이 분석 ===\n")

    common_words = set(word_risky_summary.keys()) & set(word_safe_summary.keys())
    print(f"Risky와 Safe 모두에 나타나는 단어: {len(common_words)}개\n")

    # Risky vs Safe activation 차이 계산
    word_differences = []
    for word in common_words:
        risky_act = word_risky_summary[word]['mean_activation']
        safe_act = word_safe_summary[word]['mean_activation']
        diff = risky_act - safe_act

        word_differences.append({
            'word': word,
            'risky_activation': risky_act,
            'safe_activation': safe_act,
            'difference': diff
        })

    # Risky에 더 강한 단어들 (difference > 0)
    risky_biased = sorted([w for w in word_differences if w['difference'] > 0],
                         key=lambda x: x['difference'], reverse=True)[:30]

    print("Risky Feature에 더 강하게 연관된 단어 (상위 30개):")
    for i, w in enumerate(risky_biased, 1):
        print(f"{i:2d}. '{w['word']}' "
              f"(risky={w['risky_activation']:.4f}, "
              f"safe={w['safe_activation']:.4f}, "
              f"차이={w['difference']:.4f})")

    # Safe에 더 강한 단어들 (difference < 0)
    safe_biased = sorted([w for w in word_differences if w['difference'] < 0],
                        key=lambda x: x['difference'])[:30]

    print("\n\nSafe Feature에 더 강하게 연관된 단어 (상위 30개):")
    for i, w in enumerate(safe_biased, 1):
        print(f"{i:2d}. '{w['word']}' "
              f"(risky={w['risky_activation']:.4f}, "
              f"safe={w['safe_activation']:.4f}, "
              f"차이={w['difference']:.4f})")

def main():
    print("================================================================================")
    print("단어-Feature Risky/Safe 연관관계 분석")
    print("================================================================================\n")

    # Step 1: Phase 5에서 risky/safe features 분류
    risky_features, safe_features = load_phase5_classifications()

    # Step 2: Phase 4에서 단어-feature 연관관계 로드
    word_feature_assocs = load_phase4_word_associations()

    # Step 3: 단어별 risky/safe 연관도 분석
    word_risky_summary, word_safe_summary = analyze_word_risky_safe_associations(
        word_feature_assocs, risky_features, safe_features
    )

    print(f"Risky features와 연관된 고유 단어: {len(word_risky_summary):,}개")
    print(f"Safe features와 연관된 고유 단어: {len(word_safe_summary):,}개")

    # Step 4: 상위 단어들 출력
    top_risky_words = print_top_words(word_risky_summary, "Risky Feature 연관 단어", 50)
    top_safe_words = print_top_words(word_safe_summary, "Safe Feature 연관 단어", 50)

    # Step 5: Risky vs Safe 차이 분석
    analyze_risky_safe_word_differences(word_risky_summary, word_safe_summary)

    # Step 6: 결과 저장
    output_file = Path("./word_risky_safe_analysis_report.txt")
    print(f"\n\n결과 저장: {output_file}")

if __name__ == '__main__':
    main()
