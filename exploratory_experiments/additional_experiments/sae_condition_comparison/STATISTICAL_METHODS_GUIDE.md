# Statistical Methods Learning Guide

## SAE Condition Comparison 실험의 통계 분석 이해하기

이 문서는 `sae_condition_comparison` 실험에서 사용된 통계 분석 기법들을 체계적으로 학습할 수 있도록 작성되었습니다. 수학적 기초부터 실제 코드 구현까지 단계별로 설명합니다.

---

## 목차

1. [기초 통계 개념](#1-기초-통계-개념)
2. [가설 검정의 기본](#2-가설-검정의-기본)
3. [t-test와 Cohen's d](#3-t-test와-cohens-d)
4. [ANOVA와 Eta-squared](#4-anova와-eta-squared)
5. [다중 비교 문제와 FDR 보정](#5-다중-비교-문제와-fdr-보정)
6. [Two-Way ANOVA와 상호작용](#6-two-way-anova와-상호작용)
7. [Z-score Normalization](#7-z-score-normalization)
8. [우리 실험에서의 적용](#8-우리-실험에서의-적용)
9. [추가 학습 자료](#9-추가-학습-자료)

---

## 1. 기초 통계 개념

### 1.1 평균 (Mean)

데이터의 중심 경향을 나타내는 가장 기본적인 지표입니다.

```
평균 = Σ(x_i) / n
```

**예시:**
- 데이터: [10, 20, 30, 40, 50]
- 평균 = (10+20+30+40+50) / 5 = 30

**코드에서:** `np.mean(array)`

---

### 1.2 분산 (Variance)과 표준편차 (Standard Deviation)

**분산:** 데이터가 평균으로부터 얼마나 퍼져있는지 측정합니다.

```
분산 = Σ(x_i - mean)² / (n - 1)
```

- `n-1`로 나누는 이유: **표본 분산**을 구할 때는 자유도(degree of freedom)를 고려하여 `n-1`을 사용합니다 (Bessel's correction).

**표준편차:** 분산의 제곱근으로, 원래 데이터와 같은 단위를 가집니다.

```
표준편차 (SD) = √분산
```

**예시:**
- 데이터: [10, 20, 30, 40, 50]
- 평균 = 30
- 편차: [-20, -10, 0, 10, 20]
- 편차 제곱: [400, 100, 0, 100, 400]
- 분산 = (400+100+0+100+400) / 4 = 250
- 표준편차 = √250 ≈ 15.81

**코드에서:**
```python
np.var(array, ddof=1)  # 표본 분산 (n-1로 나눔)
np.std(array, ddof=1)  # 표본 표준편차
```

**왜 중요한가?**
- 분산/표준편차가 크면 데이터가 평균으로부터 멀리 퍼져있음 (변동성이 큼)
- 작으면 데이터가 평균 근처에 밀집되어 있음

---

### 1.3 Sum of Squares (SS) - 제곱합

통계 분석의 핵심 개념으로, 분산을 분해하는 데 사용됩니다.

**Total Sum of Squares (SS_total):**
```
SS_total = Σ(x_i - grand_mean)²
```
전체 데이터의 변동성을 나타냅니다.

**Between-group Sum of Squares (SS_between):**
```
SS_between = Σ n_j × (group_mean_j - grand_mean)²
```
그룹 간 차이로 설명되는 변동성입니다.

**Within-group Sum of Squares (SS_within):**
```
SS_within = Σ Σ (x_ij - group_mean_j)²
```
그룹 내부의 변동성 (오차)입니다.

**관계:**
```
SS_total = SS_between + SS_within
```

이 분해가 ANOVA의 핵심입니다!

---

## 2. 가설 검정의 기본

### 2.1 귀무가설 (Null Hypothesis, H₀)과 대립가설 (Alternative Hypothesis, H₁)

**귀무가설 (H₀):** "차이가 없다" 또는 "효과가 없다"를 주장
- 예: "Variable 조건과 Fixed 조건의 평균이 같다"

**대립가설 (H₁):** "차이가 있다" 또는 "효과가 있다"를 주장
- 예: "Variable 조건과 Fixed 조건의 평균이 다르다"

---

### 2.2 p-value (유의확률)

**정의:** 귀무가설이 참일 때, 관찰된 결과만큼 극단적이거나 더 극단적인 결과가 나올 확률

**해석:**
- p-value가 **작을수록** 귀무가설을 기각할 증거가 강함
- 관례적으로 p < 0.05를 유의수준으로 사용
  - p < 0.05: "통계적으로 유의하다" (귀무가설 기각)
  - p ≥ 0.05: "통계적으로 유의하지 않다" (귀무가설 기각 실패)

**주의사항:**
- p-value는 "효과의 크기"가 아니라 "통계적 확실성"을 나타냄
- p < 0.05라고 해서 실용적으로 중요한 차이라는 의미는 아님
- **효과 크기(effect size)**를 함께 봐야 함!

---

### 2.3 유의수준 (Significance Level, α)

귀무가설을 기각하는 기준 확률입니다.

- 일반적으로 **α = 0.05** (5%) 사용
- α = 0.05의 의미: "귀무가설이 참일 때, 잘못 기각할 확률을 5% 이하로 제한"

**Type I Error (제1종 오류):**
- 귀무가설이 참인데 기각하는 오류 (거짓 양성, False Positive)
- 확률 = α

**Type II Error (제2종 오류):**
- 귀무가설이 거짓인데 기각하지 못하는 오류 (거짓 음성, False Negative)
- 확률 = β

---

## 3. t-test와 Cohen's d

### 3.1 Independent Samples t-test (독립표본 t-검정)

**목적:** 두 독립적인 그룹의 평균이 다른지 검정

**가설:**
- H₀: μ₁ = μ₂ (두 그룹의 평균이 같다)
- H₁: μ₁ ≠ μ₂ (두 그룹의 평균이 다르다)

**t-statistic 공식 (Welch's t-test):**

우리 코드에서 사용하는 Welch's t-test는 두 그룹의 분산이 다를 수 있다고 가정합니다.

```
t = (mean₁ - mean₂) / √(s₁²/n₁ + s₂²/n₂)
```

여기서:
- `mean₁, mean₂`: 각 그룹의 평균
- `s₁², s₂²`: 각 그룹의 분산
- `n₁, n₂`: 각 그룹의 샘플 수

**자유도 (degrees of freedom):**

Welch-Satterthwaite 방정식으로 근사합니다 (복잡하므로 scipy가 자동 계산).

**해석:**
- t-statistic의 절댓값이 클수록 두 그룹의 차이가 큼
- p-value가 작으면 (< 0.05) 두 그룹의 평균이 통계적으로 유의하게 다름

---

### 3.2 Cohen's d (효과 크기)

**정의:** 두 그룹의 평균 차이를 표준화한 효과 크기 지표

```
Cohen's d = (mean₁ - mean₂) / pooled_SD

pooled_SD = √[((n₁-1)×s₁² + (n₂-1)×s₂²) / (n₁+n₂-2)]
```

**해석 가이드라인:**
- |d| = 0.2: 작은 효과 (small effect)
- |d| = 0.5: 중간 효과 (medium effect)
- |d| = 0.8: 큰 효과 (large effect)

**부호:**
- d > 0: 그룹 1의 평균이 더 큼 (우리 코드: Variable이 더 높음)
- d < 0: 그룹 2의 평균이 더 큼 (우리 코드: Fixed가 더 높음)

**왜 필요한가?**
- p-value는 샘플 크기에 영향을 받음
- Cohen's d는 순수하게 **효과의 실질적 크기**를 측정
- 샘플이 많으면 작은 차이도 유의하게 나올 수 있지만, d가 작으면 실용적으로는 의미 없음

**우리 코드에서:**
```python
def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

---

### 3.3 우리 실험에서의 적용 (Analysis 1: Variable vs Fixed)

**목적:** Variable 조건과 Fixed 조건에서 SAE feature 활성화 값이 다른지 검정

**절차:**
1. 각 layer의 각 feature에 대해 두 그룹의 데이터 추출
   - `v_vals`: Variable 조건의 activation values
   - `f_vals`: Fixed 조건의 activation values

2. **Welch's t-test** 수행
   - H₀: Variable과 Fixed의 평균 activation이 같다
   - H₁: Variable과 Fixed의 평균 activation이 다르다

3. **Cohen's d** 계산
   - 효과의 실질적 크기 측정
   - `min_cohens_d = 0.3` (medium effect 이상만 선택)

4. **FDR 보정** (뒤에서 설명)

**코드 위치:** `condition_comparison.py:59-106`

---

## 4. ANOVA와 Eta-squared

### 4.1 One-Way ANOVA (일원 분산분석)

**목적:** 3개 이상 그룹의 평균이 모두 같은지 검정

**가설:**
- H₀: μ₁ = μ₂ = μ₃ = ... (모든 그룹의 평균이 같다)
- H₁: 적어도 한 그룹의 평균이 다르다

**핵심 아이디어:**

전체 변동성(SS_total)을 두 부분으로 분해합니다:
```
SS_total = SS_between + SS_within
```

- **SS_between**: 그룹 간 차이로 설명되는 변동
- **SS_within**: 그룹 내 오차로 남는 변동

**F-statistic:**

```
F = (SS_between / df_between) / (SS_within / df_within)
  = MS_between / MS_within
```

여기서:
- `df_between = k - 1` (k = 그룹 수)
- `df_within = N - k` (N = 전체 샘플 수)
- `MS` (Mean Square) = SS / df

**해석:**
- F가 클수록 그룹 간 차이가 큼 (그룹 간 변동 > 그룹 내 변동)
- p-value < 0.05이면 "적어도 한 그룹이 다르다"고 결론

**주의:** ANOVA는 "어느 그룹이 다른지"는 알려주지 않음 → post-hoc 검정 필요

---

### 4.2 Eta-squared (η²) - ANOVA의 효과 크기

**정의:** 전체 변동성 중 그룹 간 차이로 설명되는 비율

```
η² = SS_between / SS_total
```

**해석:**
- η² = 0.01: 작은 효과 (1% 설명)
- η² = 0.06: 중간 효과 (6% 설명)
- η² = 0.14: 큰 효과 (14% 설명)

**범위:** 0 ≤ η² ≤ 1
- 0에 가까우면: 그룹 간 차이가 전체 변동의 극히 일부만 설명
- 1에 가까우면: 거의 모든 변동이 그룹 간 차이로 설명됨

**우리 코드에서:**
```python
def compute_eta_squared(groups):
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)

    # Between-group SS
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

    # Total SS
    ss_total = np.sum((all_data - grand_mean)**2)

    return ss_between / ss_total
```

---

### 4.3 우리 실험에서의 적용 (Analysis 2: Four-Way Comparison)

**목적:** 4개 그룹(Variable-Bankrupt, Variable-Safe, Fixed-Bankrupt, Fixed-Safe)의 평균이 다른지 검정

**절차:**
1. 각 feature에 대해 4개 그룹의 데이터 추출

2. **One-Way ANOVA** 수행
   - H₀: 4개 그룹의 평균이 모두 같다
   - H₁: 적어도 한 그룹의 평균이 다르다

3. **Eta-squared** 계산
   - 효과의 실질적 크기 측정
   - `min_eta_squared = 0.01` (작은 효과 이상만 선택)

4. **FDR 보정**

**코드 위치:** `condition_comparison.py:108-154`

---

## 5. 다중 비교 문제와 FDR 보정

### 5.1 다중 비교 문제 (Multiple Comparisons Problem)

**문제:** 많은 가설 검정을 동시에 수행하면 거짓 양성(Type I Error)이 누적됨

**예시:**
- 가설 검정 1개: α = 0.05 → 5% 확률로 거짓 양성
- 가설 검정 100개 (독립적): 적어도 1개가 거짓 양성일 확률 = 1 - (0.95)^100 ≈ 99.4%!

**우리 실험에서:**
- LLaMA: 7 layers × 131,072 features = **917,504개 가설 검정**
- 보정 없이 p < 0.05를 사용하면 → 약 45,875개의 거짓 양성 예상!

---

### 5.2 Bonferroni 보정 (가장 보수적)

**방법:** 각 검정의 유의수준을 전체 개수로 나눔

```
α_adjusted = α / m

여기서 m = 가설 검정 개수
```

**예시:**
- m = 100, α = 0.05
- α_adjusted = 0.05 / 100 = 0.0005

**장점:** 매우 보수적 (Type I Error를 강하게 통제)
**단점:** 너무 보수적이라 진짜 효과도 놓칠 수 있음 (Type II Error 증가)

---

### 5.3 FDR 보정 (Benjamini-Hochberg) - 우리가 사용하는 방법

**False Discovery Rate (FDR):**
- 유의하다고 판단한 결과 중 **거짓 양성의 비율**을 제어

**예시:**
- 100개를 유의하다고 판단했을 때
- FDR = 0.05 → 그 중 평균 5개 정도는 거짓 양성
- Bonferroni는 전체 실험에서 5% 확률로 거짓 양성 1개 이상 발생

**Benjamini-Hochberg 절차:**

1. 모든 p-value를 오름차순 정렬: p₁ ≤ p₂ ≤ ... ≤ p_m

2. 다음 조건을 만족하는 가장 큰 k를 찾음:
   ```
   p_k ≤ (k/m) × α
   ```

3. p₁, p₂, ..., p_k를 유의하다고 판단

**장점:**
- Bonferroni보다 검정력이 높음 (진짜 효과를 더 잘 찾음)
- 대규모 다중 검정(genomics, neuroimaging 등)에서 표준

**우리 코드에서:**
```python
from statsmodels.stats.multitest import multipletests

def apply_fdr_correction(p_values, alpha=0.05):
    rejected, corrected_p, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method='fdr_bh'  # Benjamini-Hochberg
    )
    return rejected, corrected_p
```

**사용 예:**
```python
# 모든 feature의 p-value 수집
p_values = np.array([r['p_value'] for r in all_results])

# FDR 보정
fdr_rejected, fdr_pvals = apply_fdr_correction(p_values, alpha=0.05)

# 보정된 p-value와 유의성 판단 저장
for i, result in enumerate(all_results):
    result['p_fdr'] = fdr_pvals[i]
    result['fdr_significant'] = fdr_rejected[i]
```

---

### 5.4 Bonferroni vs FDR 비교

| 측면 | Bonferroni | FDR (Benjamini-Hochberg) |
|------|------------|---------------------------|
| 제어 대상 | Family-Wise Error Rate (FWER) | False Discovery Rate (FDR) |
| 보수성 | 매우 보수적 | 적당히 보수적 |
| 검정력 | 낮음 (많이 놓침) | 높음 (잘 찾아냄) |
| 적용 분야 | 소수의 중요한 검정 | 대규모 탐색적 연구 |
| 우리 실험 | 부적합 (너무 보수적) | **적합** (90만개 feature 분석) |

---

## 6. Two-Way ANOVA와 상호작용

### 6.1 Two-Way ANOVA (이원 분산분석)

**목적:** 두 개의 독립변수(factor)가 종속변수에 미치는 영향과 상호작용 검정

**우리 실험:**
- **Factor 1 (Bet Type):** Variable vs Fixed
- **Factor 2 (Outcome):** Bankruptcy vs Voluntary Stop
- **종속변수:** SAE feature activation values

**가설:**
1. **Main effect of Bet Type:**
   - H₀: Variable과 Fixed의 평균이 같다 (Outcome 평균 후)

2. **Main effect of Outcome:**
   - H₀: Bankruptcy와 Voluntary Stop의 평균이 같다 (Bet Type 평균 후)

3. **Interaction effect:**
   - H₀: Bet Type의 효과가 Outcome에 따라 달라지지 않는다
   - H₁: Bet Type의 효과가 Outcome에 따라 달라진다 (상호작용 존재)

---

### 6.2 상호작용 (Interaction) 이해하기

**상호작용이란?**

한 독립변수의 효과가 다른 독립변수의 수준에 따라 달라지는 현상

**예시 1: 상호작용 없음 (Additive)**

| | Bankruptcy | Voluntary Stop | 차이 |
|---|------------|----------------|------|
| Variable | 10 | 20 | +10 |
| Fixed | 5 | 15 | +10 |

- Variable과 Fixed의 차이가 Outcome에 관계없이 일정 (+10)
- 두 요인이 독립적으로 작용

**예시 2: 상호작용 있음 (Non-additive)**

| | Bankruptcy | Voluntary Stop | 차이 |
|---|------------|----------------|------|
| Variable | 10 | 25 | +15 |
| Fixed | 5 | 10 | +5 |

- Variable-Fixed 차이가 Outcome에 따라 다름
  - Bankruptcy에서: 10-5 = +5
  - Voluntary Stop에서: 25-10 = +15
- 두 요인이 결합하여 추가적인 효과 발생

**시각적 이해:**

상호작용 없음 (평행선):
```
Activation
   |
25 |                    Variable
   |                  /
20 |                /
15 |              /  Fixed
10 |            /
 5 |          /
   |________/________________
      Bankrupt    Vol.Stop
```

상호작용 있음 (교차/비평행):
```
Activation
   |
25 |                Variable
   |              /
20 |            /
15 |          /
10 |  Fixed /
 5 |______/
   |________________________
      Bankrupt    Vol.Stop
```

---

### 6.3 Two-Way ANOVA의 분산 분해

```
SS_total = SS_factor1 + SS_factor2 + SS_interaction + SS_error
```

**각 성분의 F-statistic:**

```
F_factor1 = MS_factor1 / MS_error
F_factor2 = MS_factor2 / MS_error
F_interaction = MS_interaction / MS_error
```

**각 성분의 Eta-squared:**

```
η²_factor1 = SS_factor1 / SS_total
η²_factor2 = SS_factor2 / SS_total
η²_interaction = SS_interaction / SS_total
```

---

### 6.4 우리 코드의 Simplified Two-Way ANOVA

**중요:** 우리 코드는 **근사값(approximation)**을 사용합니다!

**이유:**
- 정확한 Two-Way ANOVA는 statsmodels의 `ols()` + `anova_lm()` 필요
- 917,504개 feature에 대해 정확한 ANOVA를 돌리면 **계산 비용이 너무 큼**
- 탐색적 분석에서는 빠른 근사로 충분

**방법 (utils.py:294-391):**

1. **Main effect 1 (Bet Type):**
   - Variable 그룹 전체 vs Fixed 그룹 전체를 One-Way ANOVA로 비교

2. **Main effect 2 (Outcome):**
   - Bankruptcy 그룹 전체 vs Voluntary Stop 그룹 전체를 One-Way ANOVA로 비교

3. **Interaction:**
   - Additive model의 예측값과 실제 값의 차이로 추정
   - "Difference of differences" 접근법

**코드 주석에 명시:**
```python
"""
Simplified 2x2 factorial analysis using separate one-way ANOVAs.
Returns main effects and interaction estimates.

Note: For proper 2-way ANOVA with interaction, use statsmodels.
This is a simplified version for computational efficiency.
"""
```

**언제 정확한 분석이 필요한가?**
- Top 100 features를 선별한 후
- 논문에 포함할 주요 결과는 statsmodels로 재검증 권장
- `ANALYSIS_ISSUES_REPORT.md` 참조

---

### 6.5 우리 실험에서의 적용 (Analysis 3: Interaction)

**목적:** Bet Type과 Outcome의 상호작용 효과 탐색

**절차:**
1. 각 feature에 대해 2×2 design의 데이터 준비
   - Variable-Bankruptcy
   - Variable-Voluntary Stop
   - Fixed-Bankruptcy
   - Fixed-Voluntary Stop

2. **Simplified Two-Way ANOVA** 수행
   - Main effect of Bet Type
   - Main effect of Outcome
   - **Interaction effect**

3. **Interaction Eta-squared** 계산
   - 상호작용 효과의 크기 측정
   - `min_eta_squared = 0.01`

4. **FDR 보정** (interaction p-value에 대해)

**코드 위치:** `condition_comparison.py:156-192`

**주의사항 (CRITICAL):**
- Sparse features (activation < 1%)는 interaction 분석에서 **허위 결과** 생성
- Interaction eta ≈ 1.0인 feature의 92%가 sparse feature artifact
- 반드시 `INTERACTION_ETA_PROBLEM_EXPLAINED.md` 참조
- Minimum activation threshold 필터링 필수:
  ```python
  min_activation_rate = 0.01  # 1% of samples must be active
  min_mean = 0.001
  ```

---

## 7. Z-score Normalization

### 7.1 Z-score란?

**정의:** 데이터를 평균 0, 표준편차 1로 표준화

```
z = (x - mean) / std
```

**예시:**
- 데이터: [10, 20, 30, 40, 50]
- 평균 = 30, 표준편차 ≈ 15.81
- Z-scores: [-1.27, -0.63, 0, 0.63, 1.27]

**해석:**
- z = 0: 평균과 같음
- z = 1: 평균보다 1 표준편차 위
- z = -1: 평균보다 1 표준편차 아래
- |z| > 2: 상위/하위 약 5% (이상치 판단 기준)
- |z| > 3: 상위/하위 약 0.3% (강한 이상치)

---

### 7.2 왜 Z-score Normalization이 필요한가?

**1. 서로 다른 스케일의 변수 비교**

예시: 키(cm)와 몸무게(kg)를 함께 분석할 때
- 키: 150~190 (범위 40)
- 몸무게: 50~90 (범위 40)
- Z-score 후: 둘 다 -3~3 사이로 표준화

**2. SAE features의 경우:**
- Feature마다 activation 범위가 다름
  - Feature 100: 0~0.1
  - Feature 200: 0~10
- Z-score 후: 모든 feature를 동일한 척도로 비교 가능

**3. 통계 검정의 가정 충족:**
- 많은 parametric test는 데이터가 정규분포를 따른다고 가정
- Z-score는 분포의 모양을 바꾸지 않지만, 중심과 척도를 표준화
- Outlier의 영향을 줄임

---

### 7.3 우리 코드에서 Z-score 사용 여부

**현재 코드:**
- Raw activation values를 직접 사용 (Z-score 미적용)

**이유:**
- SAE features는 이미 L1 penalty로 sparse하게 학습됨
- Feature별로 분석하므로 절대값이 의미 있음
- Cohen's d, Eta-squared는 이미 표준화된 효과 크기

**Z-score가 필요한 경우:**
- 여러 feature를 합쳐서 분석할 때 (예: PCA, clustering)
- Cross-layer 비교 시 (layer마다 activation 범위가 다를 수 있음)

**적용 방법 (필요시):**
```python
from scipy.stats import zscore

# Per-feature z-score normalization
features_normalized = np.apply_along_axis(zscore, axis=0, arr=features)

# 또는 numpy로:
features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)
```

---

## 8. 우리 실험에서의 적용

### 8.1 전체 분석 파이프라인

```
Input:
├── SAE features (NPZ files): layer_N_features.npz
├── Experiment metadata (JSON): experiment results with bet_type

Analysis 1: Variable vs Fixed
├── Welch's t-test (two-sample comparison)
├── Cohen's d (effect size)
└── FDR correction (Benjamini-Hochberg)

Analysis 2: Four-Way Comparison
├── One-Way ANOVA (4 groups)
├── Eta-squared (effect size)
└── FDR correction

Analysis 3: Interaction Analysis
├── Simplified Two-Way ANOVA (2×2 design)
├── Main effects + Interaction
├── Eta-squared for each component
├── FDR correction (interaction p-values)
└── ⚠️ Sparse feature filtering required!

Output:
├── Top features ranked by effect size
├── FDR-corrected p-values
└── Summary statistics
```

---

### 8.2 통계 지표 해석 가이드

**Analysis 1: Variable vs Fixed**

| 지표 | 의미 | 기준 |
|------|------|------|
| `p_fdr` | FDR-보정된 유의확률 | < 0.05: 유의 |
| `cohens_d` | 효과 크기 | \|d\| ≥ 0.3 (medium) |
| `direction` | 어느 조건이 높은가 | higher_in_variable / higher_in_fixed |

**Analysis 2: Four-Way**

| 지표 | 의미 | 기준 |
|------|------|------|
| `p_fdr` | FDR-보정된 유의확률 | < 0.05: 유의 |
| `eta_squared` | 효과 크기 | η² ≥ 0.01 (small) |
| `group_means` | 4개 그룹의 평균 | 패턴 해석 |

**Analysis 3: Interaction**

| 지표 | 의미 | 기준 |
|------|------|------|
| `bet_type_f`, `bet_type_p` | Bet Type 주효과 | p < 0.05: 유의 |
| `outcome_f`, `outcome_p` | Outcome 주효과 | p < 0.05: 유의 |
| `interaction_f`, `interaction_p_fdr` | 상호작용 효과 | p < 0.05: 유의 |
| `interaction_eta` | 상호작용 효과 크기 | ⚠️ Sparse 문제 주의! |

---

### 8.3 결과 신뢰도 순위

**신뢰도 계층:**

1. **Analysis 1 (Variable vs Fixed)** - 가장 신뢰
   - 단순한 two-sample 비교
   - Welch's t-test는 robust
   - Cohen's d는 표준화된 지표

2. **Analysis 2 (Four-Way ANOVA)** - 신뢰
   - One-Way ANOVA는 검증된 방법
   - Eta-squared는 직관적
   - 단, 어느 그룹이 다른지는 post-hoc 필요

3. **Analysis 3 (Interaction)** - 주의 필요
   - Simplified 방법 (근사값)
   - Sparse feature artifact 문제
   - **Top features는 statsmodels로 재검증 필수**

---

### 8.4 실제 결과 예시 해석

**예시 1: Analysis 1 결과**

```json
{
  "layer": 12,
  "feature_id": 45678,
  "t_stat": 8.32,
  "p_value": 1.2e-15,
  "p_fdr": 3.4e-12,
  "cohens_d": 0.52,
  "variable_mean": 2.34,
  "fixed_mean": 1.89,
  "direction": "higher_in_variable"
}
```

**해석:**
- Layer 12의 Feature 45678
- Variable 조건에서 **유의하게 높음** (p_fdr < 0.001)
- **중간 크기 효과** (d = 0.52)
- Variable: 평균 2.34, Fixed: 평균 1.89
- 이 feature는 Variable betting과 관련된 개념을 encode할 가능성

---

**예시 2: Analysis 3 Interaction 결과**

```json
{
  "layer": 15,
  "feature_id": 89012,
  "bet_type_f": 12.5,
  "bet_type_p": 4.3e-4,
  "bet_type_eta": 0.03,
  "outcome_f": 3.2,
  "outcome_p": 0.073,
  "outcome_eta": 0.008,
  "interaction_f": 18.7,
  "interaction_p": 1.5e-5,
  "interaction_p_fdr": 8.2e-4,
  "interaction_eta": 0.045
}
```

**해석:**
- **Bet Type 주효과**: 유의 (p = 0.0004), small effect (η² = 0.03)
- **Outcome 주효과**: 비유의 (p = 0.073)
- **상호작용 효과**: 유의 (p_fdr = 0.0008), small-medium effect (η² = 0.045)
- **결론**: Bet Type의 효과가 Outcome에 따라 달라짐
- **주의**: Sparse feature 확인 필요!

---

## 9. 추가 학습 자료

### 9.1 기초 통계학

**입문 (한국어):**
- "통계학, 빅데이터를 잡다" (프리렉, 서민, 권오상)
- Khan Academy 통계학 강의 (한글 자막)
- "세상에서 가장 쉬운 통계학 입문" (고지마 히로유키)

**표준 교재 (영어):**
- "Statistics" by Freedman, Pisani, Purves (비전공자용 명저)
- "The Analysis of Biological Data" by Whitlock & Schluter (생명과학 통계)

**온라인 강의:**
- Coursera: "Statistics with R Specialization" (Duke University)
- edX: "Introduction to Probability and Data" (Duke University)

---

### 9.2 ANOVA와 실험 설계

**책:**
- "Design and Analysis of Experiments" by Montgomery
- "Experimental Design and Analysis" by Howard J. Seltman (무료 PDF)

**온라인:**
- Penn State STAT 502: "Analysis of Variance and Design of Experiments"
  - https://online.stat.psu.edu/stat502/
- UCLA IDRE: "Introduction to SAS/R/Stata - ANOVA"
  - https://stats.oarc.ucla.edu/

---

### 9.3 효과 크기 (Effect Size)

**논문:**
- Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences"
- Lakens, D. (2013). "Calculating and reporting effect sizes to facilitate cumulative science"

**웹 자료:**
- Effect Size Calculator: https://www.psychometrica.de/effect_size.html
- "The Essential Guide to Effect Sizes" by Paul Ellis

---

### 9.4 다중 비교 문제

**입문:**
- "Multiple Comparisons: Theory and methods" by Hochberg & Tamhane

**리뷰 논문:**
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate"
- Noble, W. S. (2009). "How does multiple testing correction work?" (Nature Biotech)

**Interactive Tutorial:**
- http://www.biostathandbook.com/multiplecomparisons.html

---

### 9.5 Python 통계 라이브러리

**SciPy:**
- Documentation: https://docs.scipy.org/doc/scipy/reference/stats.html
- Tutorial: "Statistical functions (scipy.stats)"

**Statsmodels:**
- Documentation: https://www.statsmodels.org/
- ANOVA tutorial: https://www.statsmodels.org/stable/anova.html

**Pingouin (추천):**
- 생명과학/심리학 특화 통계 라이브러리
- https://pingouin-stats.org/
- ANOVA, post-hoc, effect size 등 통합 제공

---

### 9.6 우리 실험 관련 특화 자료

**Sparse Autoencoder (SAE):**
- "Towards Monosemanticity" (Anthropic, 2023)
- "Scaling Monosemanticity" (Anthropic, 2024)

**Multiple Testing in Neuroimaging:**
- Nichols & Hayasaka (2003). "Controlling the familywise error rate in functional neuroimaging"
- Genovese et al. (2002). "Thresholding of statistical maps in functional neuroimaging using FDR"

**Feature Analysis in Neural Networks:**
- "Interpretability in the Wild" (Anthropic)
- GemmaScope / LlamaScope technical reports

---

## 부록 A: 통계 검정 선택 가이드

```
데이터 형태에 따른 검정 선택:

1. 두 그룹 비교
   ├── 독립 표본 → Independent t-test (Welch's)
   └── 대응 표본 → Paired t-test

2. 세 그룹 이상 비교
   ├── 독립 표본 → One-Way ANOVA
   ├── 대응 표본 → Repeated Measures ANOVA
   └── 두 개 이상 요인 → Two-Way / Factorial ANOVA

3. 비모수적 방법 (정규성 가정 위배 시)
   ├── 두 그룹 → Mann-Whitney U test
   ├── 세 그룹 이상 → Kruskal-Wallis test
   └── 대응 표본 → Wilcoxon signed-rank test

4. 우리 실험
   ├── Analysis 1: Independent t-test (2 groups)
   ├── Analysis 2: One-Way ANOVA (4 groups)
   └── Analysis 3: Two-Way ANOVA (2×2 factorial)
```

---

## 부록 B: 효과 크기 비교표

| 검정 방법 | 효과 크기 지표 | Small | Medium | Large |
|-----------|---------------|-------|--------|-------|
| t-test | Cohen's d | 0.2 | 0.5 | 0.8 |
| ANOVA | Eta-squared (η²) | 0.01 | 0.06 | 0.14 |
| ANOVA | Partial eta-squared (ηₚ²) | 0.01 | 0.06 | 0.14 |
| 상관분석 | Pearson's r | 0.1 | 0.3 | 0.5 |
| 회귀분석 | R² | 0.02 | 0.13 | 0.26 |

**출처:** Cohen (1988), modified for ANOVA by Richardson (2011)

---

## 부록 C: 우리 실험의 통계 검정력 (Statistical Power)

**샘플 크기:**
- Variable: ~1,600 games
- Fixed: ~1,600 games
- Total: 3,200 games

**검정력 분석 (Power Analysis):**

Analysis 1 (t-test):
- Effect size: d = 0.3 (medium)
- α = 0.05 (FDR-corrected)
- n₁ = n₂ = 1,600
- **Power ≈ 0.999** (거의 완벽한 검정력)

→ 우리 샘플 크기는 medium effect를 탐지하기에 충분!

Analysis 3 (Two-Way ANOVA):
- Effect size: η² = 0.01 (small)
- α = 0.05
- n = 3,200
- **Power ≈ 0.98** (높은 검정력)

→ 작은 효과도 탐지 가능

**결론:** False negatives (Type II Error)는 거의 문제가 안 됨. 오히려 **다중 비교로 인한 false positives (Type I Error)**가 주요 이슈 → FDR 보정 필수!

---

## 부록 D: 자주 묻는 질문 (FAQ)

**Q1: p-value가 작으면 효과가 큰 건가요?**

A: 아닙니다! p-value는 "통계적 확실성"이지 "효과 크기"가 아닙니다.
- 샘플이 많으면 작은 효과도 p < 0.001이 될 수 있음
- 반드시 Cohen's d나 Eta-squared를 함께 확인!

---

**Q2: FDR 보정 후에도 수천 개의 유의한 feature가 나옵니다. 정상인가요?**

A: 네, 정상입니다!
- 우리는 90만 개 이상의 feature를 검정
- FDR = 0.05 → 유의한 결과 중 5%는 거짓 양성
- 예: 10,000개가 유의하면 → 약 500개는 거짓 양성, 9,500개는 진짜
- 추가 필터링: effect size 기준 (d ≥ 0.3, η² ≥ 0.01)

---

**Q3: Interaction eta가 1.0에 가까운 feature가 많은데요?**

A: **Sparse feature artifact**입니다!
- SAE features는 대부분 sparse (L1 penalty)
- Activation rate < 1%인 feature는 interaction 분석에 부적합
- 해결: minimum activation threshold 필터링
- 자세한 내용: `INTERACTION_ETA_PROBLEM_EXPLAINED.md`

---

**Q4: Simplified Two-Way ANOVA는 얼마나 정확한가요?**

A: 탐색적 분석에는 충분하지만, 주요 결과는 재검증 필요합니다.
- Main effects는 정확 (One-Way ANOVA 사용)
- Interaction은 근사값 (10-20% 오차 가능)
- Top 100 features → statsmodels `ols()` + `anova_lm()`으로 재계산 권장

---

**Q5: 왜 Welch's t-test를 쓰나요? Student's t-test는 왜 안 쓰나요?**

A: Welch's t-test가 더 robust합니다.
- Student's t-test: 두 그룹의 분산이 같다고 가정 (equal variance assumption)
- Welch's t-test: 분산이 달라도 됨 (더 안전)
- 우리 데이터: Variable과 Fixed의 분산이 다를 가능성 높음
- Welch's test는 Student's test보다 나쁠 게 없음 (보수적)

---

## 부록 E: Quick Reference - 코드 스니펫

**1. Welch's t-test + Cohen's d**

```python
from scipy.stats import ttest_ind

# Welch's t-test
t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

# Cohen's d
n1, n2 = len(group1), len(group2)
var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
```

---

**2. One-Way ANOVA + Eta-squared**

```python
from scipy.stats import f_oneway

# One-Way ANOVA
f_stat, p_value = f_oneway(group1, group2, group3, group4)

# Eta-squared
all_data = np.concatenate([group1, group2, group3, group4])
grand_mean = np.mean(all_data)
ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2
                 for g in [group1, group2, group3, group4])
ss_total = np.sum((all_data - grand_mean)**2)
eta_squared = ss_between / ss_total
```

---

**3. FDR Correction**

```python
from statsmodels.stats.multitest import multipletests

# p-values: array of p-values from multiple tests
rejected, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# rejected: boolean array (True if significant after FDR)
# p_fdr: adjusted p-values
```

---

**4. Proper Two-Way ANOVA (statsmodels)**

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Prepare data
df = pd.DataFrame({
    'feature': feature_values,
    'bet_type': bet_types,  # ['variable', 'fixed', ...]
    'outcome': outcomes     # ['bankruptcy', 'voluntary_stop', ...]
})

# Fit model
model = ols('feature ~ C(bet_type) * C(outcome)', data=df).fit()

# ANOVA table
anova_table = anova_lm(model, typ=2)  # Type II SS
print(anova_table)

# Extract eta-squared
ss_bet = anova_table.loc['C(bet_type)', 'sum_sq']
ss_outcome = anova_table.loc['C(outcome)', 'sum_sq']
ss_int = anova_table.loc['C(bet_type):C(outcome)', 'sum_sq']
ss_total = anova_table['sum_sq'].sum()

eta_bet = ss_bet / ss_total
eta_outcome = ss_outcome / ss_total
eta_int = ss_int / ss_total
```

---

## 마치며

이 문서는 SAE Condition Comparison 실험의 통계 분석 방법을 이해하기 위한 학습 가이드입니다.

**학습 순서 추천:**
1. 기초 통계 개념 (Section 1-2)
2. t-test와 Cohen's d (Section 3)
3. ANOVA와 Eta-squared (Section 4)
4. 다중 비교와 FDR (Section 5) ← 매우 중요!
5. Two-Way ANOVA와 상호작용 (Section 6)
6. 우리 실험 적용 (Section 8)

**실습 권장:**
- Python으로 간단한 데이터 생성 → 각 통계 기법 적용
- SciPy 예제 따라하기
- 우리 결과 파일(JSON) 직접 분석해보기

**추가 질문이나 설명이 필요한 부분이 있으면 언제든지 물어보세요!**

---

**Document Version:** 1.0
**Last Updated:** 2026-02-02
**Author:** Claude Code (claude.ai/code)
**Related Files:**
- `condition_comparison.py`
- `utils.py`
- `ANALYSIS_ISSUES_REPORT.md`
- `INTERACTION_ETA_PROBLEM_EXPLAINED.md`
