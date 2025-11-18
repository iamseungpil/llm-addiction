# 시각화 이미지 완전 요약

생성 일시: 2025-11-08
저장 위치: `/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/images/`

---

## ✅ 생성 완료된 이미지 (총 9개 PNG + 6개 PDF)

### 📊 Image 1: Phase 5 - Risky vs Safe Feature Distribution

**파일:**
- `01_phase5_risky_safe_distribution.png` (1.0 MB)
- `01_phase5_risky_safe_distribution.pdf` (364 KB)

**내용:**
- 3,425개 통계적으로 유의미한 features (p < 0.05)
- Layer별 risky/safe feature 분포
- 4개 subplot:
  1. Scatter plot: Layer vs Cohen's d (p-value 색상 구분)
  2. Layer-wise count distribution (risky vs safe)
  3. Effect size distribution histogram
  4. Top 5 risky/safe features table

**주요 발견:**
- Risky features: 1,701개 (49.7%)
- Safe features: 1,724개 (50.3%)
- Layer 9에서 최다 features (503개)
- Layer 13도 높은 분포 (426개)

---

### 📊 Image 2: Word-Feature Association Heatmap

**파일:**
- `02_word_feature_association_heatmap.png` (908 KB)
- `02_word_feature_association_heatmap.pdf` (51 KB)

**내용:**
- 7,366,041개 word-feature correlations 분석
- 3개 subplot:
  1. Top 30 risky-associated words heatmap
  2. Top 30 safe-associated words heatmap
  3. Differential activation (risky - safe)

**주요 발견:**
- Top risky words: 'bik', 'bikik', 'baltos', 'amid', 'day', '165'
- Top safe words: 'anywhere', 'beware', '$138', 'bilset', 'attempt', 'around'
- Risky features: 62개 (Cohen's d > 0.2)
- Safe features: 82개 (Cohen's d < -0.2)

---

### 📊 Image 3: Phase 2 - Feature-Feature Correlation Network

**파일 (기존 시각화 복사):**
- `phase2_correlation_distribution.png` (748 KB)
- `phase2_layer_interaction_heatmap.png` (222 KB)
- `phase2_strong_correlations_summary.png` (429 KB)

**내용:**
- 272,351개 feature-feature correlations 분석
- Correlation distribution, layer interaction heatmap, strong correlations summary

**주요 발견:**
- Mean Pearson r: +0.8964
- Strong correlations (|r| > 0.7): 272,351개 (100%)
- Same-layer: 13,599개 (mean_r=+0.8906)
- Cross-layer: 258,752개 (mean_r=+0.8967)
- Top correlation: r=1.0000 (여러 feature 쌍)

---

### 📊 Image 4: Layer-wise Feature Evolution

**파일:**
- `04_layer_evolution.png` (682 KB)
- `04_layer_evolution.pdf` (49 KB)

**내용:**
- Experiment 1 데이터 (6,400 experiments processed)
- 31 layers 분석
- 87,012개 significant features
- 4개 subplot:
  1. Significant features per layer (bar chart)
  2. Bankrupt vs Safe features (grouped bar chart)
  3. Cohen's d evolution across layers (line plot)
  4. Layer statistics table (every 3rd layer)

**주요 발견:**
- Layer 1: 2,195개 significant (53.6% of 4,096 total)
- Layer 10: 2,193개 significant (53.5%)
- 전체 평균: ~47% significance rate
- Cohen's d는 layer에 따라 변동

---

### 📊 Image 5: Multi-round Patching Effect Timeline

**파일:**
- `05_multiround_patching_timeline.png` (794 KB)
- `05_multiround_patching_timeline.pdf` (45 KB)

**내용:**
- Experiment 5 데이터 (441 features)
- 39,690 trials (safe patch + risky patch)
- 100 rounds 분석
- 4개 subplot:
  1. Bet amount: safe vs risky patching
  2. Balance evolution: safe vs risky
  3. Cumulative bankruptcies comparison
  4. Active trials per round

**주요 발견:**
- Safe patch trials: 692,458개
- Risky patch trials: 1,287,282개
- Safe patching이 더 낮은 bet amount 유도
- Safe patching이 더 높은 balance 유지
- Safe patching이 더 낮은 bankruptcy rate

---

### 📊 Image 6: Comprehensive Pipeline Overview

**파일:**
- `06a_pipeline_flowchart.png` (434 KB)
- `06a_pipeline_flowchart.pdf` (57 KB)
- `06b_pipeline_statistics.png` (365 KB)
- `06b_pipeline_statistics.pdf` (41 KB)

**내용:**
- 06a: 5단계 파이프라인 flowchart
  1. Experiment 1 Feature Discovery
  2. Phase 1 Activation Patching
  3. Phase 5 Prompt Correlation
  4. Phase 4 Word Association
  5. Final Classification
- 06b: Phase별 통계 요약 (4개 quadrant)

**주요 흐름:**
- 6,400 experiments → 2,787 causal features
- 334,440 patching tests → 3,425 significant features
- 7.3M word-feature correlations
- Final: 1,701 risky + 1,724 safe features

---

## 📈 통계 요약

### 전체 데이터 규모:
- **Experiment 1**: 6,400 실험, 87,012 significant features (31 layers)
- **Phase 1**: 2,787 causal features, 334,440 patching tests
- **Phase 2**: 272,351 feature-feature correlations
- **Phase 4**: 7,366,041 word-feature correlations
- **Phase 5**: 3,425 significant features (p<0.05)
- **Experiment 5**: 441 features, 39,690 trials, 100 rounds

### 주요 발견:
1. **Risky vs Safe 균형**: 거의 1:1 비율 (49.7% vs 50.3%)
2. **Layer 분포**: Mid layers (9-13)에서 가장 많은 features
3. **Word associations**: Risky는 숫자/공격적 단어, Safe는 보수적 판단 단어
4. **Feature correlations**: 매우 높은 상관관계 (mean r=+0.8964)
5. **Patching 효과**: Safe patching이 명확히 안전한 행동 유도

---

## 🎯 사용 목적별 추천:

### 논문용 (Academic Paper):
- Image 1 (Phase 5 Distribution) - Main finding
- Image 4 (Layer Evolution) - Layer analysis
- Image 6b (Pipeline Statistics) - Methodology

### 프레젠테이션용:
- Image 6a (Pipeline Flowchart) - Overview
- Image 1 (Phase 5 Distribution) - Key results
- Image 5 (Multiround Patching) - Causal effects

### 상세 분석용:
- Image 2 (Word Association) - Interpretability
- Image 3 (Phase 2 Network) - Feature relationships
- Image 4 (Layer Evolution) - Technical details

---

## 📁 파일 구조:

```
/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/
├── images/
│   ├── 01_phase5_risky_safe_distribution.png (1.0 MB)
│   ├── 01_phase5_risky_safe_distribution.pdf (364 KB)
│   ├── 02_word_feature_association_heatmap.png (908 KB)
│   ├── 02_word_feature_association_heatmap.pdf (51 KB)
│   ├── 04_layer_evolution.png (682 KB)
│   ├── 04_layer_evolution.pdf (49 KB)
│   ├── 05_multiround_patching_timeline.png (794 KB)
│   ├── 05_multiround_patching_timeline.pdf (45 KB)
│   ├── 06a_pipeline_flowchart.png (434 KB)
│   ├── 06a_pipeline_flowchart.pdf (57 KB)
│   ├── 06b_pipeline_statistics.png (365 KB)
│   ├── 06b_pipeline_statistics.pdf (41 KB)
│   ├── phase2_correlation_distribution.png (748 KB)
│   ├── phase2_layer_interaction_heatmap.png (222 KB)
│   └── phase2_strong_correlations_summary.png (429 KB)
└── scripts/
    ├── visualize_phase5_distribution.py
    ├── visualize_word_feature_heatmap.py
    ├── visualize_layer_evolution_fixed.py
    ├── visualize_multiround_patching_fixed.py
    └── visualize_pipeline_overview.py
```

---

## ✅ 완료 체크리스트:

- [x] Image 1: Phase 5 Distribution (NEW)
- [x] Image 2: Word Association Heatmap (NEW)
- [x] Image 3: Phase 2 Network (EXISTING - 복사 완료)
- [x] Image 4: Layer Evolution (NEW - FIXED)
- [x] Image 5: Multiround Patching Timeline (NEW - FIXED)
- [x] Image 6: Pipeline Overview (NEW)

**총 9개 PNG + 6개 PDF = 15개 파일 생성/복사 완료**

---

생성 스크립트: `/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/scripts/`
