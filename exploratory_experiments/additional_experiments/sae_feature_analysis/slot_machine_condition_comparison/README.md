# SAE Condition Comparison Analysis

SAE 피처를 사용하여 **variable/fixed 베팅 조건** 간의 차이를 분석합니다.

## 목적

기존 SAE 분석에서는 파산/비파산(bankrupt vs voluntary_stop)만 분류했습니다. 이 실험에서는 베팅 조건(variable vs fixed)에 따른 SAE 피처 차이를 추가로 분석합니다.

### 배경

| Model | Fixed 파산율 | Variable 파산율 | 배율 |
|-------|-------------|----------------|------|
| LLaMA | 2.6% (42건) | 6.8% (108건) | 2.6x |
| Gemma | 12.8% (205건) | 29.1% (465건) | 2.3x |

Variable 조건에서 파산율이 2배 이상 높은 것은 모델이 더 위험한 의사결정을 한다는 것을 의미합니다. 이 분석은 그 원인을 SAE 피처 수준에서 찾고자 합니다.

## 분석 내용

### 1. Variable vs Fixed (주효과)
- 각 SAE 피처에 대해 variable/fixed 조건 간 t-test 수행
- Cohen's d 효과 크기 계산
- FDR 보정 적용

### 2. 4-Way Comparison
- 4개 그룹 비교: variable-bankrupt, variable-safe, fixed-bankrupt, fixed-safe
- One-way ANOVA + eta-squared 효과 크기
- 어떤 그룹에서 특정 피처가 높은지 확인

### 3. Interaction Analysis
- bet_type × outcome 상호작용 분석
- 베팅 조건이 결과에 미치는 영향이 피처에 따라 다른지 확인

## 실행 방법

### 1. 환경 설정

```bash
conda activate llama_sae_env
```

### 2. Config 확인

`configs/analysis_config.yaml`에서 데이터 경로 확인:
- `feature_dir`: 기존 SAE 피처 NPZ 파일 경로
- `experiment_file`: 원본 실험 JSON 파일 경로

### 3. 분석 실행

```bash
# LLaMA 분석
python -m src.condition_comparison --model llama

# Gemma 분석
python -m src.condition_comparison --model gemma

# 또는 스크립트 사용
bash scripts/run_analysis.sh llama
```

## 출력 파일

```
results/
├── condition_comparison_summary_{model}_{timestamp}.json  # 전체 요약
├── variable_vs_fixed_{model}_{timestamp}.json            # 분석 1 상세
├── four_way_{model}_{timestamp}.json                     # 분석 2 상세
└── interaction_{model}_{timestamp}.json                  # 분석 3 상세
```

## 의존성

- numpy
- scipy
- statsmodels
- tqdm
- pyyaml

## 데이터 요구사항

이 분석은 기존 `llama_sae_analysis`에서 추출된 NPZ 파일을 재사용합니다:
- `layer_{N}_features.npz`: SAE 피처 (features, outcomes, game_ids)
- 원본 실험 JSON: bet_type 정보 매핑용

## 컴퓨팅 자원

- **GPU**: 불필요 (CPU-bound 통계 분석)
- **RAM**: 2-8 GB (layer당 처리)
- **실행 시간**:
  - LLaMA: ~15-30분
  - Gemma (131K features): ~1-2시간
