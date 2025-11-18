# L1-31 Multilayer Patching Experiment Analysis

## 개요

이 폴더는 **Experiment 2 (L1-31 Multilayer Patching)**의 일관된 양방향 효과(Consistent Bidirectional Effects) 분석 결과를 시각화합니다.

## 데이터 소스

### 원본 실험 데이터
```
/data/llm_addiction/experiment_2_multilayer_patching/response_logs/
├── responses_L1_*.json
├── responses_L2_*.json
├── ...
└── responses_L31_*.json
```
- **총 199개 response log 파일**
- **1,026,971개 valid trials**
- **9,300개 unique features** (L1-31, 각 레이어당 300개)

### 재분석 결과 (Baseline Comparison)
```
/home/ubuntu/llm_addiction/analysis/exp2_L1_31_ALL_LAYERS_feature_group_summary_BASELINE.csv
```
- Baseline 비교 방식으로 재분석한 전체 9,300개 features
- 각 feature의 safe/risky patch 효과 측정

### CORRECT Consistent Features 분류
```
/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv
/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv
```
- **Safe features: 640개**
- **Risky features: 2,147개**
- **Total: 2,787개** (양방향 일관성 검증 완료)

## 분석 기준

### CORRECT Safe Feature (640개)
**양쪽 patch 모두 일관되게 '안전' 방향으로 효과**:
1. Safe patch: `safe context → stop↑, risky context → bankruptcy↓`
2. Risky patch: `safe context → stop↑, risky context → bankruptcy↓`
→ 두 patch 모두 중단율을 높이고 파산율을 낮춤

### CORRECT Risky Feature (2,147개)
**양쪽 patch 모두 일관되게 '위험' 방향으로 효과**:
1. Safe patch: `safe context → stop↓, risky context → bankruptcy↑`
2. Risky patch: `safe context → stop↓, risky context → bankruptcy↑`
→ 두 patch 모두 중단율을 낮추고 파산율을 높임

## 생성된 Figure

### Figure 1: Average Effects
```
figures/L1_31_causal_patching_average_effects.png
figures/L1_31_causal_patching_average_effects.pdf
```

**Safe Features (n=640):**
- Safe context (stop rate): `+9.1% (safe patch), +8.7% (risky patch)`
- Risky context (bankruptcy): `-19.6% (safe patch), -19.4% (risky patch)`

**Risky Features (n=2,147):**
- Safe context (stop rate): `-41.3% (safe patch), -41.5% (risky patch)`
- Risky context (bankruptcy): `+17.0% (safe patch), +17.0% (risky patch)`

### Figure 2: Layer Distribution
```
figures/L1_31_causal_features_layer_distribution.png
figures/L1_31_causal_features_layer_distribution.pdf
```

**Layer별 분포 (Top 5):**
- L9: 272개 (safe=0, risky=272)
- L13: 205개 (safe=0, risky=205)
- L14: 200개 (safe=0, risky=200)
- L12: 192개 (safe=0, risky=192)
- L17: 191개 (safe=1, risky=190)

## 실행 방법

### Figure 생성
```bash
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/analysis
python create_figures_L1_31.py
```

### 재분석 (필요 시)
```bash
cd /home/ubuntu/llm_addiction/analysis

# 1. Baseline comparison 재분석
python exp2_L1_31_reanalysis_baseline.py

# 2. CORRECT consistent features 분류
python CORRECT_consistent_features.py

# 3. Figure 생성
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/analysis
python create_figures_L1_31.py
```

## 원본 441개 분석과 비교

### 원본 (GPU 4/5, Experiment 2 Final)
- **Layers**: L25-31만
- **Features tested**: ~3,365개
- **Causal features**: 441개
- **Trials**: 50 per condition
- **결과 위치**: `/home/ubuntu/llm_addiction/writing/figures/`

### L1-31 (이번 분석)
- **Layers**: L1-31 전체
- **Features tested**: 9,300개
- **CORRECT consistent causal**: 2,787개
- **Trials**: 30 per condition
- **결과 위치**: `./figures/`

## 주요 차이점

1. **Layer 범위**: L1-31 전체 (원본은 L25-31만)
2. **Feature 선택**: Top 300 by |Cohen's d| per layer
3. **일관성 기준**: 양방향 일관성 (원본은 2-way)
4. **결과**: 양방향 모두 같은 방향 효과를 보이는 2,787개 선별

## 데이터 Hallucination 방지

✅ **NO HARDCODING**: 모든 수치는 실제 CSV 데이터에서 계산
✅ **NO HALLUCINATION**: 존재하지 않는 데이터 사용 안 함
✅ **실제 데이터만 사용**: CORRECT_consistent_*.csv 파일에서 직접 로드

## 관련 파일

### 분석 스크립트
- `create_figures_L1_31.py`: Figure 생성 (이 폴더)
- `/home/ubuntu/llm_addiction/analysis/exp2_L1_31_reanalysis_baseline.py`: Baseline 재분석
- `/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_features.py`: 양방향 일관성 분류

### 데이터 파일
- `/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv`: Safe features (640개)
- `/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv`: Risky features (2,147개)
- `/home/ubuntu/llm_addiction/analysis/exp2_L1_31_ALL_LAYERS_feature_group_summary_BASELINE.csv`: 전체 9,300개

### 원본 실험 코드
- `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py`

---

**생성일**: 2025-10-22
**실험**: Experiment 2 - Multilayer Activation Patching (L1-31)
**방법론**: CORRECT Consistent Bidirectional Effects (양방향 일관성)
