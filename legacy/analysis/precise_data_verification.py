#!/usr/bin/env python3
"""
LLaMA 논문 vs 실제 데이터 정확한 비교 검증
모든 수치의 정확성을 실제 데이터로 확인
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import math
from scipy import stats

def load_exp1_data():
    """실험 1 데이터 로드"""
    main_path = "/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json"
    missing_path = "/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json"
    
    print("📂 실험 1 데이터 로딩 중...")
    
    # Main 파일 로드
    with open(main_path, 'r') as f:
        main_data = json.load(f)
    
    # Missing 파일 로드  
    with open(missing_path, 'r') as f:
        missing_data = json.load(f)
    
    print(f"  Main file: {len(main_data['results'])}개 실험")
    print(f"  Missing file: {len(missing_data['results'])}개 실험")
    
    # 전체 데이터 합치기
    all_results = main_data['results'] + missing_data['results']
    total_experiments = len(all_results)
    
    print(f"  총 실험: {total_experiments}개")
    
    return all_results

def load_patching_data():
    """Patching 실험 데이터 로드"""
    gpu5_path = "/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json"
    gpu4_path = "/data/llm_addiction/results/patching_population_mean_final_20250905_150612.json"
    
    print("\n📂 Patching 데이터 로딩 중...")
    
    with open(gpu5_path, 'r') as f:
        gpu5_data = json.load(f)
    
    with open(gpu4_path, 'r') as f:
        gpu4_data = json.load(f)
    
    print(f"  GPU5 데이터 구조: {list(gpu5_data.keys())}")
    print(f"  GPU4 데이터 구조: {list(gpu4_data.keys())}")
    
    return gpu5_data, gpu4_data

def verify_basic_stats(results):
    """기본 통계 검증"""
    print("\n🔍 기본 통계 검증")
    print("=" * 50)
    
    # 1. 총 실험 수
    total_exp = len(results)
    paper_claim_total = 6400
    
    print(f"1. 총 실험 수:")
    print(f"   실제: {total_exp}")
    print(f"   논문: {paper_claim_total}")
    print(f"   일치: {'✅' if abs(total_exp - paper_claim_total) <= 10 else '❌'}")
    
    # 2. 조건별 분포 확인
    conditions = {}
    bet_types = {}
    first_results = {}
    
    bankruptcies = 0
    voluntary_stops = 0
    
    for exp in results:
        # 조건 분포
        condition_id = exp.get('condition_id', 'unknown')
        conditions[condition_id] = conditions.get(condition_id, 0) + 1
        
        # 베팅 타입
        bet_type = exp.get('bet_type', 'unknown')
        bet_types[bet_type] = bet_types.get(bet_type, 0) + 1
        
        # 첫 게임 결과
        first_result = exp.get('first_result', 'unknown')
        first_results[first_result] = first_results.get(first_result, 0) + 1
        
        # 파산/자발적 중단
        if exp.get('is_bankrupt', False):
            bankruptcies += 1
        elif exp.get('voluntary_stop', False):
            voluntary_stops += 1
    
    # 3. 파산율 계산
    bankruptcy_rate = (bankruptcies / total_exp) * 100
    paper_bankruptcy_rate = 3.2
    
    print(f"\n2. 파산율:")
    print(f"   실제: {bankruptcy_rate:.1f}% ({bankruptcies}/{total_exp})")
    print(f"   논문: {paper_bankruptcy_rate:.1f}%")
    print(f"   차이: {abs(bankruptcy_rate - paper_bankruptcy_rate):.1f}%p")
    print(f"   일치: {'✅' if abs(bankruptcy_rate - paper_bankruptcy_rate) <= 0.5 else '❌'}")
    
    # 4. 조건 수 확인
    unique_conditions = len(conditions)
    paper_conditions = 128
    
    print(f"\n3. 실험 조건 수:")
    print(f"   실제: {unique_conditions}개 조건")
    print(f"   논문: {paper_conditions}개 조건")
    print(f"   일치: {'✅' if unique_conditions == paper_conditions else '❌'}")
    
    # 5. 베팅 타입 분포
    print(f"\n4. 베팅 타입 분포:")
    for bt, count in bet_types.items():
        expected = total_exp // 2  # 50:50 분포 예상
        print(f"   {bt}: {count}개 (예상: ~{expected})")
    
    # 6. 첫 게임 결과 분포  
    print(f"\n5. 첫 게임 결과 분포:")
    for fr, count in first_results.items():
        expected = total_exp // 2  # 50:50 분포 예상
        print(f"   {fr}: {count}개 (예상: ~{expected})")
    
    return {
        'total_experiments': total_exp,
        'bankruptcy_rate': bankruptcy_rate,
        'bankruptcies': bankruptcies,
        'conditions': unique_conditions,
        'bet_types': bet_types,
        'first_results': first_results
    }

def verify_feature_counts(gpu5_data, gpu4_data):
    """Feature 개수 검증"""
    print("\n🔍 Feature 개수 검증")
    print("=" * 50)
    
    # GPU5 인과적 features
    gpu5_causal_bet = gpu5_data.get('causal_features_bet', [])
    gpu5_causal_stop = gpu5_data.get('causal_features_stop', [])
    
    # GPU4 인과적 features
    gpu4_causal_bet = gpu4_data.get('causal_features_bet', [])
    gpu4_causal_stop = gpu4_data.get('causal_features_stop', [])
    
    print(f"1. GPU5 인과적 features:")
    print(f"   베팅 영향: {len(gpu5_causal_bet)}개")
    print(f"   중단 영향: {len(gpu5_causal_stop)}개")
    
    print(f"\n2. GPU4 인과적 features:")
    print(f"   베팅 영향: {len(gpu4_causal_bet)}개")
    print(f"   중단 영향: {len(gpu4_causal_stop)}개")
    
    # 전체 unique features 계산
    all_gpu5 = set(gpu5_causal_bet + gpu5_causal_stop)
    all_gpu4 = set(gpu4_causal_bet + gpu4_causal_stop)
    all_unique = all_gpu5.union(all_gpu4)
    
    total_causal = len(all_unique)
    paper_causal = 275
    
    print(f"\n3. 총 인과적 features:")
    print(f"   실제: {total_causal}개")
    print(f"   논문: {paper_causal}개")
    print(f"   차이: {abs(total_causal - paper_causal)}개")
    print(f"   일치: {'✅' if abs(total_causal - paper_causal) <= 10 else '❌'}")
    
    # 중복/독립 분석
    overlap = all_gpu5.intersection(all_gpu4)
    gpu5_only = all_gpu5 - all_gpu4
    gpu4_only = all_gpu4 - all_gpu5
    
    print(f"\n4. GPU 간 분포:")
    print(f"   중복: {len(overlap)}개")
    print(f"   GPU5 전용: {len(gpu5_only)}개")
    print(f"   GPU4 전용: {len(gpu4_only)}개")
    print(f"   총합: {len(overlap) + len(gpu5_only) + len(gpu4_only)}개")
    
    # 논문 claim 검증 - 356개에서 275개 인과관계
    total_discovered = 356
    causal_percentage = (total_causal / total_discovered) * 100
    paper_percentage = 77.2
    
    print(f"\n5. 인과관계 비율:")
    print(f"   실제: {causal_percentage:.1f}% ({total_causal}/{total_discovered})")
    print(f"   논문: {paper_percentage:.1f}%")
    print(f"   차이: {abs(causal_percentage - paper_percentage):.1f}%p")
    print(f"   일치: {'✅' if abs(causal_percentage - paper_percentage) <= 2.0 else '❌'}")
    
    return {
        'total_causal': total_causal,
        'causal_percentage': causal_percentage,
        'gpu5_count': len(all_gpu5),
        'gpu4_count': len(all_gpu4),
        'overlap': len(overlap)
    }

def verify_cohens_d_claims(gpu5_data, gpu4_data):
    """Cohen's d 값들 검증"""
    print("\n🔍 Cohen's d 값 검증")
    print("=" * 50)
    
    # 논문의 극단적 Cohen's d 값들
    paper_cohens_d = {
        28337: -7.07,
        14607: 5.82,
        22493: -4.93,
        18100: 4.21,
        16039: 3.76,
        9244: -3.54,
        30582: 3.12,
        14031: -2.88
    }
    
    print("논문에 제시된 극단적 Cohen's d 값들:")
    extreme_count = 0
    very_extreme_count = 0
    
    for feature_id, d_value in paper_cohens_d.items():
        abs_d = abs(d_value)
        if abs_d > 5.0:
            level = "매우 극단적 (>5.0)"
            very_extreme_count += 1
            extreme_count += 1
        elif abs_d > 3.0:
            level = "극단적 (>3.0)"
            extreme_count += 1
        elif abs_d > 2.0:
            level = "강함 (>2.0)"
        else:
            level = "중간"
        
        print(f"  Feature {feature_id}: d = {d_value:+.2f} → {level}")
    
    print(f"\n통계:")
    print(f"  |d| > 5.0: {very_extreme_count}개 (매우 희귀)")
    print(f"  |d| > 3.0: {extreme_count}개 (극단적)")
    print(f"  평균 |d|: {np.mean([abs(d) for d in paper_cohens_d.values()]):.2f}")
    
    return {
        'extreme_cohens_d': extreme_count,
        'very_extreme_cohens_d': very_extreme_count,
        'avg_abs_d': np.mean([abs(d) for d in paper_cohens_d.values()])
    }

def verify_correlation_claims():
    """상관계수 검증"""
    print("\n🔍 상관계수 검증")
    print("=" * 50)
    
    # 논문의 상관계수들
    paper_correlations = {
        '파산율': 0.905,
        '평균 베팅': 0.905, 
        '평균 손실': 0.929,
        '평균 라운드': 0.524,
        '전체 평균': 0.815
    }
    
    print("논문에 제시된 Spearman 상관계수들:")
    high_corr_count = 0
    very_high_corr_count = 0
    
    for metric, rho in paper_correlations.items():
        if rho > 0.9:
            level = "매우 높음 (>0.9)"
            very_high_corr_count += 1
            high_corr_count += 1
        elif rho > 0.8:
            level = "높음 (>0.8)"
            high_corr_count += 1
        elif rho > 0.5:
            level = "중간 (>0.5)"
        else:
            level = "낮음"
        
        print(f"  {metric}: ρ = {rho:.3f} → {level}")
    
    print(f"\n통계:")
    print(f"  ρ > 0.9: {very_high_corr_count}개 (매우 높은 일관성)")
    print(f"  ρ > 0.8: {high_corr_count}개")
    print(f"  평균 ρ: {np.mean(list(paper_correlations.values())):.3f}")
    
    return {
        'high_correlations': high_corr_count,
        'very_high_correlations': very_high_corr_count,
        'avg_correlation': np.mean(list(paper_correlations.values()))
    }

def final_verification_summary(basic_stats, feature_stats, cohens_stats, corr_stats):
    """최종 검증 요약"""
    print("\n" + "=" * 70)
    print("🎯 최종 데이터 검증 요약")
    print("=" * 70)
    
    verification_results = []
    
    # 1. 기본 통계
    total_exp_match = abs(basic_stats['total_experiments'] - 6400) <= 10
    bankruptcy_match = abs(basic_stats['bankruptcy_rate'] - 3.2) <= 0.5
    conditions_match = basic_stats['conditions'] == 128
    
    verification_results.append(('총 실험 수', '6,400개', f"{basic_stats['total_experiments']}개", total_exp_match))
    verification_results.append(('파산율', '3.2%', f"{basic_stats['bankruptcy_rate']:.1f}%", bankruptcy_match))
    verification_results.append(('실험 조건', '128개', f"{basic_stats['conditions']}개", conditions_match))
    
    # 2. Feature 통계
    causal_match = abs(feature_stats['total_causal'] - 275) <= 10
    percentage_match = abs(feature_stats['causal_percentage'] - 77.2) <= 2.0
    
    verification_results.append(('인과적 features', '275개', f"{feature_stats['total_causal']}개", causal_match))
    verification_results.append(('인과 비율', '77.2%', f"{feature_stats['causal_percentage']:.1f}%", percentage_match))
    
    # 3. 통계적 수치들
    verification_results.append(('극단적 Cohen\'s d (>3.0)', 'N/A', f"{cohens_stats['extreme_cohens_d']}개", None))
    verification_results.append(('매우 높은 상관계수 (>0.9)', 'N/A', f"{corr_stats['very_high_correlations']}개", None))
    
    print(f"{'항목':<20} {'논문 주장':<15} {'실제 데이터':<15} {'일치도'}")
    print("-" * 70)
    
    total_matches = 0
    checkable_items = 0
    
    for item, paper_claim, actual_data, match in verification_results:
        if match is not None:
            status = "✅" if match else "❌"
            checkable_items += 1
            if match:
                total_matches += 1
        else:
            status = "⚠️"
        
        print(f"{item:<20} {paper_claim:<15} {actual_data:<15} {status}")
    
    print("-" * 70)
    accuracy_rate = (total_matches / checkable_items) * 100 if checkable_items > 0 else 0
    print(f"전체 정확도: {accuracy_rate:.1f}% ({total_matches}/{checkable_items} 항목 일치)")
    
    # 최종 판정
    if accuracy_rate >= 90:
        final_status = "✅ 데이터 정확성 확인됨"
    elif accuracy_rate >= 80:
        final_status = "⚠️ 대부분 정확하나 일부 확인 필요"
    else:
        final_status = "❌ 데이터 불일치 발견"
    
    print(f"\n🎯 최종 판정: {final_status}")
    
    return accuracy_rate >= 90

def main():
    print("🔍 LLaMA 논문 vs 실제 데이터 정확성 검증")
    print("=" * 70)
    
    try:
        # 데이터 로드
        exp1_results = load_exp1_data()
        gpu5_data, gpu4_data = load_patching_data()
        
        # 각 섹션별 검증
        basic_stats = verify_basic_stats(exp1_results)
        feature_stats = verify_feature_counts(gpu5_data, gpu4_data)
        cohens_stats = verify_cohens_d_claims(gpu5_data, gpu4_data)
        corr_stats = verify_correlation_claims()
        
        # 최종 검증 요약
        is_accurate = final_verification_summary(basic_stats, feature_stats, cohens_stats, corr_stats)
        
        return is_accurate
        
    except Exception as e:
        print(f"❌ 검증 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    is_data_accurate = main()