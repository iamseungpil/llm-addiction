# SAE Condition Comparison — Brief Figure Guide

본 문서는 SAE 조건 비교 분석의 핵심 Figure인 **Figure 1 (Heatmap)**과 **Figure 3 (Scatter Plot)**에 대한 간략한 해석 가이드이다.

---

## ⚠️ 중요: 이 분석은 관찰 연구(Observational Study)

본 분석의 모든 통계(t-test, ANOVA, Cohen's d, η²)는 **상관관계**를 측정한다. 따라서:

- ✅ **알 수 있는 것**: 어떤 feature가 조건/결과와 연관되는가
- ❌ **알 수 없는 것**: 그 feature가 행동을 유발하는가, 결과를 사후 기록하는가

**인과성 검증**은 별도의 실험(Causal Patching, Steering)이 필요하다.

---

## Figure 1: SAE Feature Activation Heatmap

**파일**: `fig1_improved_heatmap_llama.png`, `fig1_improved_heatmap_gemma.png`

### 무엇을 보여주는가

- Four-Way ANOVA에서 η² (effect size)가 가장 높은 **상위 20개 feature**
- 4개 조건별 활성화 패턴: Variable-Bankrupt, Variable-Safe, Fixed-Bankrupt, Fixed-Safe
- Z-score 정규화: 각 feature 내에서 어느 조건이 상대적으로 높은지 표시

### 색상 해석

- **빨간색**: 해당 조건에서 평균보다 높은 활성화
- **파란색**: 해당 조건에서 평균보다 낮은 활성화
- **흰색**: 평균 수준

### 핵심 패턴

#### LLaMA-3.1-8B
```
Variable Betting | Fixed Betting
파란색    파란색  | 빨간색  빨간색
```
- 점선(Variable/Fixed 경계)을 기준으로 색이 명확히 구분
- **해석**: 상위 feature들이 **베팅 조건**(Variable vs Fixed)을 인코딩
- 파산 여부는 거의 영향 없음

#### Gemma-2-9B-IT
```
Variable Betting | Fixed Betting
빨간색    파란색  | 빨간색  파란색
```
- Bankrupt(1, 3열) vs Safe(2, 4열)로 색이 구분
- **해석**: 상위 feature들이 **게임 결과**(Bankrupt vs Safe)를 인코딩
- 베팅 조건은 거의 영향 없음
- 활성화 차이가 극단적 (50~100배)

### 왜 중요한가

동일한 행동적 결과(Variable에서 2.4배 높은 파산율)가 **완전히 다른 신경 표상 전략**에서 발생한다:
- LLaMA: "이것이 어떤 환경인가" (베팅 조건)
- Gemma: "어떤 결과가 나왔는가" (파산 여부)

---

## Figure 3: Bet Type vs Outcome Effect Scatter

**파일**: `fig3_improved_scatter_llama.png`, `fig3_improved_scatter_gemma.png`

### 무엇을 보여주는가

각 SAE feature가 "베팅 조건"과 "게임 결과" 중 무엇을 더 강하게 인코딩하는지 2차원 공간에 표시.

### 축의 의미

**X축 (Outcome Effect)**:
```
|평균(Bankrupt) - 평균(Safe)|
```
→ 값이 클수록 해당 feature가 파산 여부를 강하게 구분

**Y축 (Bet Type Effect)**:
```
|평균(Variable) - 평균(Fixed)|
```
→ 값이 클수록 해당 feature가 베팅 조건을 강하게 구분

### 대각선의 의미

- **대각선 위쪽 (보라색 영역)**: Bet Type Effect > Outcome Effect
  - 베팅 조건이 더 중요
- **대각선 아래쪽 (주황색 영역)**: Outcome Effect > Bet Type Effect
  - 게임 결과가 더 중요

### 점의 속성

- **크기**: η² (효과 크기)에 비례 → 큰 점일수록 중요한 feature
- **색상**: η² 값 (노란색 = 높음, 보라색 = 낮음)

### 핵심 패턴

#### LLaMA-3.1-8B
- **거의 모든 점이 대각선 위쪽**
- Y축 범위: 0.07~0.40, X축 범위: 0.00~0.14
- Y축이 X축보다 약 **3배 넓음**
- **결론**: Feature들이 압도적으로 **베팅 조건**을 인코딩

#### Gemma-2-9B-IT
- **거의 모든 점이 대각선 아래쪽**
- X축 범위: 5~50, Y축 범위: 0~5
- X축이 Y축보다 약 **10배 넓음**
- **결론**: Feature들이 압도적으로 **게임 결과**를 인코딩

### 왜 중요한가

한 장의 그래프로 두 모델의 완전히 반대되는 인코딩 전략을 명확히 보여준다:
- LLaMA는 "환경 표상 우선" 모델
- Gemma는 "결과 표상 우선" 모델

---

## 논문에서의 활용

### Figure 배치 권장사항

**Main Text**:
- **Figure 3** (Scatter): 핵심 메시지를 가장 명확히 전달
  - Panel A: LLaMA
  - Panel B: Gemma

**Supplementary 또는 본문 보조**:
- **Figure 1** (Heatmap): 구체적인 활성화 패턴 상세 제공
  - Panel A: LLaMA
  - Panel B: Gemma

### Figure Caption 작성 시 주의사항

1. **관찰적 발견임을 명시**:
   - ❌ "Features **cause** differential behavior"
   - ✅ "Features **are associated with** differential behavior"
   - ✅ "Features **encode** condition/outcome information"

2. **인과성 실험 참조**:
   - "These associative patterns are validated through causal patching experiments (Section 4)"

3. **효과 크기 기준 명시**:
   - "Top 20 features ranked by η² from Four-Way ANOVA"
   - "Cohen's d > 0.3 and FDR-corrected p < 0.05"

---

## 통계 요약

### 데이터

- **LLaMA**: 3,200 games (Variable: 1,600, Fixed: 1,600)
  - Variable 파산율: 6.75% (108/1600)
  - Fixed 파산율: 2.63% (42/1600)

- **Gemma**: 3,200 games (Variable: 1,600, Fixed: 1,600)
  - Variable 파산율: 29.06% (465/1600)
  - Fixed 파산율: 12.81% (205/1600)

### 분석 방법

1. **Analysis 1 (Variable vs Fixed)**: Welch's t-test + Cohen's d
2. **Analysis 2 (Four-Way ANOVA)**: 4개 그룹 비교 + η² (eta-squared)
3. **FDR 보정**: Benjamini-Hochberg method (α = 0.05)

### 효과 크기 기준

- **Cohen's d**: 0.3 (minimum), 0.8 (large), 2.0 (very large)
- **η²**: 0.01 (small), 0.06 (medium), 0.14 (large)

---

## 해석 시 주의사항

1. **축 스케일의 차이**
   - LLaMA와 Gemma의 Figure 3에서 축 범위가 크게 다름 (LLaMA: 0~0.4, Gemma: 0~50)
   - 이는 SAE 학습 설정 차이로 인한 raw activation 값의 차이
   - 비교 시 절댓값이 아닌 **비율**(Bet/Outcome)로 해석

2. **Z-score 정규화의 의미**
   - Figure 1은 행(feature) 단위 정규화
   - 따라서 "절대적으로 얼마나 활성화되는가"가 아닌 "4개 조건 중 어디서 상대적으로 높은가"를 보여줌
   - 다른 feature 간 절댓값 비교 불가

3. **상관관계 vs 인과관계**
   - 본 분석은 어디까지나 **가설 생성**의 역할
   - 인과적 주장은 Causal Patching/Steering 실험 결과와 결합해야 함

---

*문서 생성일: 2026-01-31*
*분석 데이터: additional_experiments/sae_condition_comparison/results/*
*전체 가이드: SAE_Figure_Analysis_Guide.md*
