# Two-Way ANOVA 완전 가이드
## SAE Condition Comparison을 위한 통계적 방법론

**작성일**: 2026-02-02
**분석 대상**: LLM 슬롯머신 실험 (Variable vs Fixed Betting, Bankrupt vs Safe Outcome)
**사용 도구**: statsmodels (Python), Type II Sum of Squares

---

## 목차

1. [개요](#1-개요)
2. [왜 Two-Way ANOVA가 필요한가?](#2-왜-two-way-anova가-필요한가)
3. [핵심 개념](#3-핵심-개념)
4. [수학적 정의](#4-수학적-정의)
5. [계산 과정 단계별 설명](#5-계산-과정-단계별-설명)
6. [효과 크기 (η²) 계산](#6-효과-크기-η²-계산)
7. [교호작용(Interaction) 이해하기](#7-교호작용interaction-이해하기)
8. [구현 코드 설명](#8-구현-코드-설명)
9. [결과 해석 가이드](#9-결과-해석-가이드)
10. [Figure 설명](#10-figure-설명)
11. [주의사항 및 제약](#11-주의사항-및-제약)
12. [FAQ](#12-faq)

---

## 1. 개요

### 1.1 분석 목적

SAE (Sparse Autoencoder) 특징(feature)들이 다음 두 요인에 어떻게 반응하는지 분석:

- **Factor 1 (Bet Type)**: Variable Betting vs Fixed Betting
- **Factor 2 (Outcome)**: Bankrupt vs Safe

### 1.2 실험 설계

```
2×2 Factorial Design (완전 요인 설계)

                 Outcome
                 Bankrupt    Safe
         ┌─────────┬─────────┐
Variable │   VB    │   VS    │
Bet Type ├─────────┼─────────┤
Fixed    │   FB    │   FS    │
         └─────────┴─────────┘

샘플 크기 (LLaMA-3.1-8B):
- VB (Variable-Bankrupt): 108 games
- VS (Variable-Safe): 1,492 games
- FB (Fixed-Bankrupt): 42 games
- FS (Fixed-Safe): 1,558 games
- Total: 3,200 games
```

### 1.3 연구 질문

1. **Bet Type 주효과**: Variable과 Fixed 조건에서 feature 활성화가 다른가?
2. **Outcome 주효과**: Bankrupt과 Safe 조건에서 feature 활성화가 다른가?
3. **교호작용**: Bet Type의 효과가 Outcome에 따라 달라지는가?

---

## 2. 왜 Two-Way ANOVA가 필요한가?

### 2.1 기존 방법의 한계

#### One-Way ANOVA with 4 Groups (기존 방법)

```python
# 4개 그룹을 독립적으로 비교
groups = [VB, VS, FB, FS]
f_stat, p_value = f_oneway(*groups)
```

**문제점**:
- "4개 그룹이 다르다"는 것만 알 수 있음
- **어떤 요인** 때문에 다른지 알 수 없음
- 주효과와 교호작용을 분리할 수 없음

**예시**:
```
VB=10, VS=8, FB=5, FS=3
One-Way ANOVA: p < 0.001 (유의함)

하지만:
- Bet Type 때문인가? (Variable: 9 vs Fixed: 4)
- Outcome 때문인가? (Bankrupt: 7.5 vs Safe: 5.5)
- 둘의 상호작용 때문인가?
→ 알 수 없다!
```

#### Two-Way ANOVA (개선 방법)

```python
# 요인별로 분해
model = ols('activation ~ C(bet_type) + C(outcome) + C(bet_type):C(outcome)', data=df).fit()
anova_table = anova_lm(model, typ=2)
```

**장점**:
- **Bet Type 주효과**: 분산의 X%를 설명 (η² = 0.20)
- **Outcome 주효과**: 분산의 Y%를 설명 (η² = 0.03)
- **교호작용**: 분산의 Z%를 설명 (η² = 0.02)
- **어느 요인이 중요한지** 정량화 가능

### 2.2 실제 연구 사례

**LLaMA-3.1-8B 분석 결과** (Layers 28-30, 2,168 features):

| 분석 방법 | 결과 |
|-----------|------|
| One-Way ANOVA | "4개 그룹이 다르다" (75%의 features에서 유의) |
| Two-Way ANOVA | **Bet Type η² = 0.041**, Outcome η² = 0.008, Interaction η² = 0.001 |
| **해석** | **Bet Type이 Outcome보다 5배 중요!** |

→ Two-Way ANOVA 없이는 이 중요한 발견을 놓쳤을 것입니다.

---

## 3. 핵심 개념

### 3.1 요인(Factor)과 수준(Level)

**요인(Factor)**: 독립변수
- Factor 1 (Bet Type): 2 levels (Variable, Fixed)
- Factor 2 (Outcome): 2 levels (Bankrupt, Safe)

**셀(Cell)**: 요인 조합
- 4개 셀: VB, VS, FB, FS

### 3.2 주효과(Main Effect)

**정의**: 한 요인의 평균적인 영향 (다른 요인을 평균낸 것)

**Bet Type 주효과**:
```
Mean(Variable) = (VB + VS) / 2
Mean(Fixed) = (FB + FS) / 2
Main Effect = Mean(Variable) - Mean(Fixed)
```

**시각화**:
```
Mean Activation

    │
 10 │     ●  Variable
    │
  5 │              ● Fixed
    │
  0 └─────────────────────
        Bet Type
```

→ Bet Type이 활성화에 영향을 줌 (주효과 존재)

### 3.3 교호작용(Interaction)

**정의**: 한 요인의 효과가 다른 요인의 수준에 따라 달라지는 현상

**예시 1: 교호작용 없음**
```
Activation

    │     Variable
 10 │    ●────────●
    │
  5 │         ●────────● Fixed
    │
  0 └────────────────────
      Bankrupt   Safe
```
- Variable과 Fixed의 기울기가 같음
- Bet Type 효과가 Outcome과 무관

**예시 2: 교호작용 있음**
```
Activation

    │         Variable
 10 │    ●╱
    │     ╱ ╲
  5 │    ╱   ╲● Fixed
    │   ╱     ╲
  0 └────────────────────
      Bankrupt   Safe
```
- Variable과 Fixed의 기울기가 다름
- Variable에서만 Outcome 효과가 큼 → **교호작용**

### 3.4 분산 분해(Variance Decomposition)

**총 분산**을 여러 성분으로 분해:

```
Total Variance (SS_total)
  = SS_bet_type         (Bet Type 주효과)
  + SS_outcome          (Outcome 주효과)
  + SS_interaction      (Bet Type × Outcome 교호작용)
  + SS_residual         (설명되지 않는 잔차)
```

**효과 크기 (η²)**:
```
η²_bet_type = SS_bet_type / SS_total
```

→ "Bet Type이 총 분산의 몇 %를 설명하는가?"

---

## 4. 수학적 정의

### 4.1 모형(Model)

**Two-Way ANOVA 모형**:

```
Y_ijk = μ + α_i + β_j + (αβ)_ij + ε_ijk

여기서:
- Y_ijk: k번째 관측치의 활성화 값 (i번째 Bet Type, j번째 Outcome)
- μ: 전체 평균 (grand mean)
- α_i: Bet Type i의 주효과
- β_j: Outcome j의 주효과
- (αβ)_ij: Bet Type i와 Outcome j의 교호작용
- ε_ijk: 잔차 (random error)
```

**제약 조건**:
```
Σ α_i = 0  (주효과의 합 = 0)
Σ β_j = 0
Σ (αβ)_ij = 0  (각 요인에 대해)
```

### 4.2 가설 검정

**귀무가설**:

1. **Bet Type 주효과**:
   ```
   H₀: α_Variable = α_Fixed = 0
   H₁: α_i 중 적어도 하나는 0이 아님
   ```

2. **Outcome 주효과**:
   ```
   H₀: β_Bankrupt = β_Safe = 0
   H₁: β_j 중 적어도 하나는 0이 아님
   ```

3. **교호작용**:
   ```
   H₀: (αβ)_ij = 0 for all i, j
   H₁: (αβ)_ij 중 적어도 하나는 0이 아님
   ```

### 4.3 F-통계량

**일반 형태**:
```
F = MS_effect / MS_residual

여기서:
- MS = SS / df  (Mean Square = Sum of Squares / degrees of freedom)
```

**자유도 (degrees of freedom)**:
```
df_bet_type = (levels_bet - 1) = 2 - 1 = 1
df_outcome = (levels_outcome - 1) = 2 - 1 = 1
df_interaction = df_bet_type × df_outcome = 1 × 1 = 1
df_residual = N - (levels_bet × levels_outcome) = 3200 - 4 = 3196
```

**p-value 계산**:
```
p = P(F_observed | H₀)
  = 1 - F.cdf(F_observed, df_effect, df_residual)
```

→ F-분포를 사용하여 관측된 F-통계량이 귀무가설 하에서 나올 확률

---

## 5. 계산 과정 단계별 설명

### 5.1 Step 1: 셀 평균 계산

**데이터 구조**:
```
3,200개 게임 × 1개 feature = 3,200개 활성화 값
```

**셀 평균**:
```
Ȳ_VB = (1/n_VB) Σ Y_VB  = mean([활성화 값들 for VB games])
Ȳ_VS = (1/n_VS) Σ Y_VS
Ȳ_FB = (1/n_FB) Σ Y_FB
Ȳ_FS = (1/n_FS) Σ Y_FS
```

**예시 데이터**:
```
Feature L28-12265:
  VB: mean = 0.0083, n = 108
  VS: mean = 0.0020, n = 1492
  FB: mean = 0.2172, n = 42
  FS: mean = 0.2562, n = 1558
```

### 5.2 Step 2: 주효과 평균 계산

**Bet Type 주효과** (Outcome을 평균냄):
```
Ȳ_Variable = (n_VB × Ȳ_VB + n_VS × Ȳ_VS) / (n_VB + n_VS)
           = (108 × 0.0083 + 1492 × 0.0020) / (108 + 1492)
           = 0.0024

Ȳ_Fixed = (n_FB × Ȳ_FB + n_FS × Ȳ_FS) / (n_FB + n_FS)
        = (42 × 0.2172 + 1558 × 0.2562) / (42 + 1558)
        = 0.2505
```

**Outcome 주효과** (Bet Type을 평균냄):
```
Ȳ_Bankrupt = (n_VB × Ȳ_VB + n_FB × Ȳ_FB) / (n_VB + n_FB)
           = (108 × 0.0083 + 42 × 0.2172) / (108 + 42)
           = 0.0668

Ȳ_Safe = (n_VS × Ȳ_VS + n_FS × Ȳ_FS) / (n_VS + n_FS)
       = (1492 × 0.0020 + 1558 × 0.2562) / (1492 + 1558)
       = 0.1311
```

**전체 평균**:
```
Ȳ = (n_VB × Ȳ_VB + n_VS × Ȳ_VS + n_FB × Ȳ_FB + n_FS × Ȳ_FS) / N
  = (108×0.0083 + 1492×0.0020 + 42×0.2172 + 1558×0.2562) / 3200
  = 0.1264
```

### 5.3 Step 3: 제곱합(Sum of Squares) 계산

#### 총 제곱합 (SS_total)

**정의**: 전체 평균으로부터 각 관측치의 편차 제곱합
```
SS_total = Σ (Y_ijk - Ȳ)²
```

**계산**:
```python
grand_mean = np.mean(all_activations)
SS_total = np.sum((all_activations - grand_mean)**2)
```

#### Bet Type 제곱합 (SS_bet_type)

**정의**: Bet Type 주효과로 설명되는 제곱합
```
SS_bet_type = Σ n_i (Ȳ_i - Ȳ)²

여기서:
- i ∈ {Variable, Fixed}
- n_i: 해당 Bet Type의 샘플 수
```

**계산**:
```
SS_bet_type = n_Variable × (Ȳ_Variable - Ȳ)² + n_Fixed × (Ȳ_Fixed - Ȳ)²
            = 1600 × (0.0024 - 0.1264)² + 1600 × (0.2505 - 0.1264)²
            = 1600 × 0.0154 + 1600 × 0.0154
            = 49.28
```

#### Outcome 제곱합 (SS_outcome)

**정의**: Outcome 주효과로 설명되는 제곱합
```
SS_outcome = Σ n_j (Ȳ_j - Ȳ)²

여기서:
- j ∈ {Bankrupt, Safe}
```

**계산**:
```
SS_outcome = n_Bankrupt × (Ȳ_Bankrupt - Ȳ)² + n_Safe × (Ȳ_Safe - Ȳ)²
           = 150 × (0.0668 - 0.1264)² + 3050 × (0.1311 - 0.1264)²
           = 150 × 0.0036 + 3050 × 0.0000
           = 0.54
```

#### 교호작용 제곱합 (SS_interaction)

**정의**: 셀 평균이 주효과 모델로 예측한 값에서 벗어나는 정도
```
SS_interaction = Σ n_ij (Ȳ_ij - Ȳ_i - Ȳ_j + Ȳ)²

여기서:
- Ȳ_ij: 실제 셀 평균
- Ȳ_i + Ȳ_j - Ȳ: 주효과 모델로 예측한 셀 평균
```

**직관적 설명**:
- 주효과만으로 예측: `Ȳ_VB_predicted = Ȳ_Variable + Ȳ_Bankrupt - Ȳ`
- 실제 관측값: `Ȳ_VB_observed`
- 차이: `Ȳ_VB_observed - Ȳ_VB_predicted` → 교호작용

**계산** (Type II Sum of Squares, statsmodels 사용):
```python
model = ols('activation ~ C(bet_type) + C(outcome) + C(bet_type):C(outcome)', data=df).fit()
anova_table = anova_lm(model, typ=2)
SS_interaction = anova_table.loc['C(bet_type):C(outcome)', 'sum_sq']
```

#### 잔차 제곱합 (SS_residual)

**정의**: 모델로 설명되지 않는 제곱합
```
SS_residual = SS_total - SS_bet_type - SS_outcome - SS_interaction
```

**계산**:
```
SS_residual = 100.0 - 49.28 - 0.54 - 0.18
            = 50.0
```

### 5.4 Step 4: 평균 제곱(Mean Square) 계산

**정의**: 제곱합을 자유도로 나눈 값
```
MS_effect = SS_effect / df_effect
```

**계산**:
```
MS_bet_type = SS_bet_type / 1 = 49.28 / 1 = 49.28
MS_outcome = SS_outcome / 1 = 0.54 / 1 = 0.54
MS_interaction = SS_interaction / 1 = 0.18 / 1 = 0.18
MS_residual = SS_residual / 3196 = 50.0 / 3196 = 0.0156
```

### 5.5 Step 5: F-통계량 계산

**정의**: 효과 평균 제곱 / 잔차 평균 제곱
```
F = MS_effect / MS_residual
```

**계산**:
```
F_bet_type = 49.28 / 0.0156 = 3156.4
F_outcome = 0.54 / 0.0156 = 34.6
F_interaction = 0.18 / 0.0156 = 11.5
```

**해석**:
- F-통계량이 클수록 효과가 큼
- F = 1이면 효과 없음 (효과 분산 = 잔차 분산)
- F >> 1이면 효과 존재

### 5.6 Step 6: p-value 계산

**정의**: 귀무가설(효과 없음) 하에서 관측된 F보다 큰 값이 나올 확률
```
p = P(F > F_observed | H₀)
  = 1 - F.cdf(F_observed, df_effect, df_residual)
```

**계산** (F-분포 사용):
```python
from scipy.stats import f

p_bet_type = 1 - f.cdf(3156.4, df1=1, df2=3196)
           ≈ 0.0000  (매우 유의)

p_outcome = 1 - f.cdf(34.6, df1=1, df2=3196)
          ≈ 0.0000  (매우 유의)

p_interaction = 1 - f.cdf(11.5, df1=1, df2=3196)
              ≈ 0.0007  (유의)
```

**판정 기준** (α = 0.05):
- p < 0.05: 효과 존재 (귀무가설 기각)
- p ≥ 0.05: 효과 없음 (귀무가설 채택)

---

## 6. 효과 크기 (η²) 계산

### 6.1 정의

**Eta-squared (η²)**: 특정 효과가 설명하는 총 분산의 비율

```
η² = SS_effect / SS_total
```

**해석**:
- η² = 0.01: Small effect (작은 효과)
- η² = 0.06: Medium effect (중간 효과)
- η² = 0.14: Large effect (큰 효과)

### 6.2 계산 예시

**Feature L28-12265**:
```
SS_total = 100.0
SS_bet_type = 49.28
SS_outcome = 0.54
SS_interaction = 0.18
SS_residual = 50.0

η²_bet_type = 49.28 / 100.0 = 0.493  (매우 큰 효과)
η²_outcome = 0.54 / 100.0 = 0.005   (매우 작은 효과)
η²_interaction = 0.18 / 100.0 = 0.002  (매우 작은 효과)
η²_residual = 50.0 / 100.0 = 0.500
```

**검증**:
```
η²_bet_type + η²_outcome + η²_interaction + η²_residual
= 0.493 + 0.005 + 0.002 + 0.500
= 1.000  ✓
```

→ **총 분산의 49.3%가 Bet Type으로 설명됨!**

### 6.3 Partial η² vs η²

**Partial η² (부분 에타 제곱)**:
```
partial_η² = SS_effect / (SS_effect + SS_residual)
```

**η² (전체 에타 제곱)**:
```
η² = SS_effect / SS_total
```

**차이점**:
- Partial η²: 다른 효과를 제거한 후의 효과 크기 (더 큼)
- η²: 전체 분산 대비 효과 크기 (보수적)

**본 분석에서는 η² 사용** (보수적 추정)

---

## 7. 교호작용(Interaction) 이해하기

### 7.1 교호작용이 없는 경우

**셀 평균 패턴**:
```
          Bankrupt   Safe   Difference
Variable     10        8        2
Fixed         5        3        2
```

**그래프**:
```
Activation
    │
 10 │  ●━━━━━━━━━━━●  Variable
    │    ╲         ╱
  5 │     ●━━━━━━━●  Fixed
    │
  0 └──────────────────
     Bankrupt  Safe
```

**해석**:
- Variable과 Fixed 모두 Bankrupt에서 Safe로 갈 때 2씩 감소
- **기울기가 같음** → **교호작용 없음**
- Outcome 효과가 Bet Type과 무관

### 7.2 교호작용이 있는 경우

**셀 평균 패턴**:
```
          Bankrupt   Safe   Difference
Variable     10        2        8  ← 큰 차이
Fixed         5        4        1  ← 작은 차이
```

**그래프**:
```
Activation
    │
 10 │  ●╲            Variable
    │   ╲╲
  5 │    ●━━━━━━━●  Fixed
    │          ╱
  2 │         ╱●  Variable
  0 └──────────────────
     Bankrupt  Safe
```

**해석**:
- Variable: Bankrupt에서 Safe로 갈 때 **8 감소** (큰 효과)
- Fixed: Bankrupt에서 Safe로 갈 때 **1 감소** (작은 효과)
- **기울기가 다름** → **교호작용 존재**
- Outcome 효과가 Bet Type에 따라 다름

### 7.3 교호작용의 유형

#### 순서형 교호작용 (Ordinal Interaction)

```
      Bankrupt   Safe
Var      10        5    ↓ 둘 다 감소하지만
Fix       8        3    ↓ 감소 폭이 다름
```

- 순서는 유지됨 (Variable > Fixed in both conditions)
- **효과의 크기만** 달라짐

#### 역전형 교호작용 (Disordinal Interaction)

```
      Bankrupt   Safe
Var      10        3    ↓ Variable은 크게 감소
Fix       4        8    ↑ Fixed는 오히려 증가!
```

- **순서가 역전됨** (Bankrupt: Var>Fix, Safe: Fix>Var)
- 효과의 **방향** 자체가 달라짐

### 7.4 교호작용의 통계적 의미

**귀무가설**:
```
H₀: (μ_VB - μ_VS) = (μ_FB - μ_FS)
```
→ "Outcome 효과가 Bet Type과 무관하다"

**대립가설**:
```
H₁: (μ_VB - μ_VS) ≠ (μ_FB - μ_FS)
```
→ "Outcome 효과가 Bet Type에 따라 다르다"

**검정**:
```
SS_interaction = Σ n_ij [(Ȳ_ij - Ȳ_i - Ȳ_j + Ȳ)]²
F_interaction = (SS_interaction / 1) / MS_residual
p = P(F > F_interaction | H₀)
```

---

## 8. 구현 코드 설명

### 8.1 데이터 준비

```python
# SAE feature 활성화 데이터 로드
all_features = np.vstack([vb_features, vs_features, fb_features, fs_features])
# Shape: (3200, 32768) → 3200 games × 32768 features

# 라벨 생성
bet_type_labels = np.concatenate([
    np.zeros(n_vb + n_vs),  # 0 = Variable
    np.ones(n_fb + n_fs)     # 1 = Fixed
])

outcome_labels = np.concatenate([
    np.zeros(n_vb),  # 0 = Bankrupt
    np.ones(n_vs),   # 1 = Safe
    np.zeros(n_fb),  # 0 = Bankrupt
    np.ones(n_fs)    # 1 = Safe
])
```

### 8.2 statsmodels를 사용한 ANOVA

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 데이터프레임 생성
df = pd.DataFrame({
    'activation': feature_activations,  # 단일 feature의 활성화 값
    'bet_type': bet_type_labels,
    'outcome': outcome_labels
})

# Categorical 변환
df['bet_type'] = pd.Categorical(df['bet_type'])
df['outcome'] = pd.Categorical(df['outcome'])

# 모델 적합 (교호작용 포함)
model = ols('activation ~ C(bet_type) + C(outcome) + C(bet_type):C(outcome)', data=df).fit()

# ANOVA 테이블 생성 (Type II)
anova_table = anova_lm(model, typ=2)
```

**ANOVA 테이블 출력 예시**:
```
                              sum_sq     df         F    PR(>F)
C(bet_type)                  49.280      1   3156.41   0.0000
C(outcome)                    0.540      1     34.62   0.0000
C(bet_type):C(outcome)        0.180      1     11.54   0.0007
Residual                     50.000   3196
```

### 8.3 효과 크기 계산

```python
# 총 제곱합
ss_total = anova_table['sum_sq'].sum()

# Eta-squared 계산
eta_sq_bet = anova_table.loc['C(bet_type)', 'sum_sq'] / ss_total
eta_sq_outcome = anova_table.loc['C(outcome)', 'sum_sq'] / ss_total
eta_sq_interaction = anova_table.loc['C(bet_type):C(outcome)', 'sum_sq'] / ss_total
eta_sq_residual = anova_table.loc['Residual', 'sum_sq'] / ss_total

print(f"η² (Bet Type): {eta_sq_bet:.3f}")
print(f"η² (Outcome): {eta_sq_outcome:.3f}")
print(f"η² (Interaction): {eta_sq_interaction:.3f}")
print(f"η² (Residual): {eta_sq_residual:.3f}")
```

### 8.4 Type II vs Type I vs Type III Sum of Squares

**Type I (Sequential)**:
- 순서대로 효과를 추가하면서 계산
- 요인 순서에 따라 결과가 달라짐
- **균형 설계가 아니면 부적절**

**Type II (Hierarchical)**:
- 각 효과를 다른 주효과와 함께 고려
- 교호작용 항은 제외하고 주효과를 계산
- **균형 설계가 아닐 때 권장** ← **우리가 사용**

**Type III (Marginal)**:
- 각 효과를 모든 다른 효과와 함께 고려
- 가장 보수적
- SPSS 기본값

**본 분석에서 Type II 사용 이유**:
```python
# 샘플 크기가 불균형
VB: 108, VS: 1492, FB: 42, FS: 1558

# Type II가 불균형 설계에서 더 적절한 추정 제공
anova_lm(model, typ=2)  # typ=2 → Type II
```

### 8.5 전체 레이어 분석 자동화

```python
def analyze_layer(self, layer: int):
    """모든 features에 대해 Two-Way ANOVA 실행"""
    grouped = self.data_loader.load_layer_features_grouped(layer)

    # 데이터 준비
    all_features = np.vstack([
        grouped['variable_bankrupt'],
        grouped['variable_safe'],
        grouped['fixed_bankrupt'],
        grouped['fixed_safe']
    ])

    results = []
    for feature_id in range(n_features):
        feature_activations = all_features[:, feature_id]

        # 분산이 0이면 skip
        if np.std(feature_activations) == 0:
            continue

        # ANOVA 실행
        anova_result = self.run_two_way_anova_feature(
            feature_activations, bet_type_labels, outcome_labels
        )

        results.append({
            'layer': layer,
            'feature_id': feature_id,
            **anova_result
        })

    return results
```

---

## 9. 결과 해석 가이드

### 9.1 ANOVA 테이블 읽기

**예시 결과** (Feature L28-12265):
```
Effect            Sum_Sq   df    F        p        η²
─────────────────────────────────────────────────────
Bet Type          49.280    1   3156.41  0.0000  0.493
Outcome            0.540    1     34.62  0.0000  0.005
Interaction        0.180    1     11.54  0.0007  0.002
Residual          50.000  3196
─────────────────────────────────────────────────────
Total            100.000  3199
```

**해석 순서**:

1. **유의성 확인** (p-value):
   - Bet Type: p < 0.0001 → **매우 유의**
   - Outcome: p < 0.0001 → **매우 유의**
   - Interaction: p = 0.0007 → **유의** (α=0.05 기준)

2. **효과 크기 비교** (η²):
   - Bet Type: η² = 0.493 → **매우 큰 효과** (49.3% 설명)
   - Outcome: η² = 0.005 → **매우 작은 효과** (0.5% 설명)
   - Interaction: η² = 0.002 → **무시 가능** (0.2% 설명)

3. **결론**:
   - **Bet Type이 압도적으로 중요**
   - Outcome은 통계적으로 유의하지만 실질적 효과는 미미
   - 교호작용은 거의 없음

### 9.2 그룹 평균 해석

**셀 평균**:
```
                Bankrupt   Safe    Marginal
Variable        0.0083    0.0020   0.0024  ← 낮음
Fixed           0.2172    0.2562   0.2505  ← 높음
─────────────────────────────────────────
Marginal        0.0668    0.1311   0.1264
```

**주효과 크기**:
```
Bet Type Effect = |0.2505 - 0.0024| = 0.248  ← 큼!
Outcome Effect = |0.1311 - 0.0668| = 0.064
```

**교호작용 패턴**:
```
Variable: Bankrupt (0.0083) vs Safe (0.0020) → Diff = 0.0063
Fixed: Bankrupt (0.2172) vs Safe (0.2562) → Diff = -0.0390
```
- Variable에서는 Bankrupt가 더 높음
- Fixed에서는 Safe가 더 높음
- **패턴이 반대** → 교호작용 존재 (하지만 작음)

### 9.3 Feature 분류

**Bet Type Dominant** (η²_bet > η²_outcome):
```python
bet_dominant = [r for r in results
                if r['bet_type_effect']['eta_squared'] >
                   r['outcome_effect']['eta_squared']]

# LLaMA Layers 28-30: 1617/2168 = 74.6%
```
→ 이 features는 "어떤 베팅 조건인가"를 인코딩

**Outcome Dominant** (η²_outcome > η²_bet):
```python
outcome_dominant = [r for r in results
                    if r['outcome_effect']['eta_squared'] >
                       r['bet_type_effect']['eta_squared']]

# LLaMA Layers 28-30: 551/2168 = 25.4%
```
→ 이 features는 "파산했는가"를 인코딩

**Significant Interaction** (p_interaction < 0.05):
```python
interaction_sig = [r for r in results
                   if r['interaction_effect']['p_value'] < 0.05]

# LLaMA Layers 28-30: 575/2168 = 26.5%
```
→ 이 features는 조건 간 복잡한 상호작용

### 9.4 레이어별 경향

**Layer-wise Summary** (LLaMA):
```
Layer  Mean η²(Bet)  Mean η²(Outcome)  Bet/Outcome Ratio
─────────────────────────────────────────────────────────
28     0.0415        0.0081            5.1×
29     0.0408        0.0079            5.2×
30     0.0413        0.0080            5.2×
─────────────────────────────────────────────────────────
Mean   0.0412        0.0080            5.2×
```

**해석**:
- 모든 레이어에서 **Bet Type이 Outcome보다 5배 중요**
- 레이어 간 변화 거의 없음 (일관된 패턴)

---

## 10. Figure 설명

### Figure 1: Effect Size Comparison

**좌측 패널 (Distribution)**:
- Violin plot + Box plot
- 전체 features의 효과 크기 분포
- LLaMA: Bet Type η²가 Outcome보다 훨씬 큼

**우측 패널 (Top 100 Stacked Bar)**:
- 효과 크기 상위 100개 features
- Stacked bar로 효과 분해
  - 파란색: Bet Type
  - 빨간색: Outcome
  - 녹색: Interaction
- LLaMA: 대부분 Bet Type (파란색)이 차지

### Figure 2: Main Effects Scatter

**축**:
- X축: Outcome η²
- Y축: Bet Type η²

**대각선**:
- 대각선 위: Bet Type > Outcome
- 대각선 아래: Outcome > Bet Type

**패턴**:
- LLaMA: **거의 모든 점이 대각선 위** (Bet Type dominant)
- Gemma: 대부분 대각선 아래 (Outcome dominant)

**점 속성**:
- 크기: Total η² (큰 점 = 중요한 feature)
- 색상: Total η² (노란색 = 높음)

### Figure 3: Interaction Patterns

**12개 서브플롯**:
- 교호작용이 유의한 상위 12개 features
- X축: Outcome (Bankrupt vs Safe)
- Y축: Mean Activation

**선**:
- 파란색 (Variable): Variable 조건
- 빨간색 (Fixed): Fixed 조건

**교호작용 판단**:
- 선이 평행 → 교호작용 없음
- 선이 교차/기울기 다름 → 교호작용 존재

### Figure 4: Layer-wise Effects

**히트맵**:
- 행: Bet Type, Outcome, Interaction
- 열: Layer (28, 29, 30)
- 색상: Mean η²

**패턴**:
- LLaMA: Bet Type 행이 가장 밝음 (높은 η²)
- 레이어 간 일관성

---

## 11. 주의사항 및 제약

### 11.1 통계적 가정

**Two-Way ANOVA 가정**:

1. **독립성 (Independence)**:
   - 각 관측치는 독립적
   - ✓ 각 게임은 독립적으로 플레이됨

2. **정규성 (Normality)**:
   - 잔차가 정규분포
   - ⚠️ SAE features는 매우 sparse (대부분 0)
   - 대표본 (n=3200)이므로 중심극한정리로 완화

3. **등분산성 (Homogeneity of Variance)**:
   - 각 그룹의 분산이 같음
   - ⚠️ 샘플 크기 불균형 (VB:108, FB:42 vs VS:1492, FS:1558)
   - Type II Sum of Squares로 완화

### 11.2 샘플 크기 불균형

**문제**:
```
VB:  108 (3.4%)  ← 매우 작음
VS: 1492 (46.6%)
FB:   42 (1.3%)  ← 극도로 작음
FS: 1558 (48.7%)
```

**영향**:
- 작은 그룹 (FB, VB)의 분산 추정 불안정
- 특히 sparse features에서 심화

**대응**:
1. **Type II Sum of Squares** 사용 (불균형 설계에 적합)
2. **효과 크기 (η²)** 중심 해석 (p-value만 보지 않음)
3. **최소 샘플 수 필터링**: n_min = 5 이상인 features만 분석

### 11.3 Sparse Features

**문제**:
- SAE features는 L1 penalty로 매우 sparse
- 많은 features가 대부분 0, 소수 샘플만 활성화
- 분산이 0이거나 매우 작음

**대응**:
```python
# 분산이 0인 features 제외
if np.std(feature_activations) == 0:
    continue
```

**추가 필터링 권장**:
```python
# 최소 활성화 비율 필터링
activation_rate = (feature_activations > 0).mean()
if activation_rate < 0.01:  # 1% 미만 활성화
    continue  # 너무 sparse, 신뢰할 수 없음
```

### 11.4 다중 비교 문제

**문제**:
- 32,768 features를 동시에 검정
- Type I 오류율 증가 (false positive)

**대응**:
1. **FDR 보정** (Benjamini-Hochberg):
   ```python
   from statsmodels.stats.multitest import multipletests
   _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh', alpha=0.05)
   ```

2. **효과 크기 기준** 추가:
   ```python
   significant = (p_corrected < 0.05) & (eta_squared > 0.01)
   ```

3. **Top-k 전략**:
   - p-value 대신 η² 기준으로 상위 k개 선택
   - 본 분석: Top 100 features 시각화

### 11.5 인과성 vs 상관성

**중요**: Two-Way ANOVA는 **관찰 연구 (Observational Study)**

**알 수 있는 것**:
- ✅ Bet Type과 feature 활성화가 연관됨 (association)
- ✅ 연관의 강도 (η²)

**알 수 없는 것**:
- ❌ Feature가 행동을 **유발**하는가?
- ❌ 행동이 feature를 **유발**하는가?
- ❌ 제3의 요인이 둘 다 유발하는가?

**인과성 검증 방법**:
1. **Causal Patching** (Activation Editing):
   - Feature 활성화를 인위적으로 변경
   - 행동 변화를 관찰
   - → `paper_experiments/llama_sae_analysis/phase4`

2. **Steering Vectors** (Directional Control):
   - Feature 방향으로 활성화 이동
   - 행동 변화 측정
   - → `steering_vector_analysis/`

---

## 12. FAQ

### Q1: One-Way ANOVA와 Two-Way ANOVA 중 무엇을 사용해야 하나요?

**A**: 실험 설계에 따라 다릅니다.

**One-Way ANOVA 사용 조건**:
- 요인이 1개일 때
- 예: "3개 모델 (LLaMA, Gemma, GPT) 간 차이"

**Two-Way ANOVA 사용 조건**:
- 요인이 2개 이상일 때
- **교호작용**에 관심이 있을 때
- 예: "Bet Type × Outcome"

**우리 연구**:
- 2개 요인 (Bet Type, Outcome)
- 교호작용 존재 가능성
- → **Two-Way ANOVA 필수**

### Q2: Type I, II, III Sum of Squares 중 무엇을 써야 하나요?

**A**: 실험 설계의 균형성에 따라 다릅니다.

| 조건 | 권장 Type |
|------|-----------|
| 균형 설계 (n_ij 모두 같음) | Any (결과 같음) |
| 불균형 설계 | **Type II** (주효과 중심) or Type III (보수적) |
| 우리 연구 (VB:108, VS:1492, FB:42, FS:1558) | **Type II** |

**이유**:
- Type II는 불균형 설계에서 주효과를 더 잘 추정
- Type III는 너무 보수적 (검정력 손실)

### Q3: η²가 작은데 p-value가 유의하면 어떻게 해석하나요?

**A**: **효과 크기를 우선 고려**하세요.

**예시**:
```
Outcome Effect:
- p = 0.0001 (매우 유의)
- η² = 0.005 (0.5% 설명)
```

**해석**:
- 통계적으로 유의 (p < 0.05) ✓
- 실질적으로 무의미 (η² < 0.01)
- **결론**: Outcome 효과는 존재하지만 **매우 작음**

**권장 보고 형식**:
```
"Outcome 주효과는 통계적으로 유의했으나 (F(1,3196)=34.6, p<.001),
효과 크기는 매우 작았다 (η²=0.005). 반면 Bet Type 주효과는
통계적으로 유의하고 (F(1,3196)=3156.4, p<.001)
효과 크기도 매우 컸다 (η²=0.493)."
```

### Q4: 교호작용이 유의하면 주효과를 해석할 수 있나요?

**A**: **조심스럽게** 해석해야 합니다.

**Case 1: 교호작용이 작을 때** (η²_interaction < 0.01):
- 주효과를 그대로 해석 가능
- "평균적으로 Bet Type이 중요"

**Case 2: 교호작용이 클 때** (η²_interaction > 0.06):
- 주효과를 **조건부로** 해석
- "Variable에서만 Outcome 효과가 큼"
- Simple effects analysis 필요

**우리 연구** (η²_interaction = 0.002):
- 교호작용 매우 작음
- 주효과 해석 가능

### Q5: 왜 LLaMA는 Bet Type, Gemma는 Outcome이 중요한가요?

**A**: **서로 다른 표상 전략 (Representational Strategy)**

**가설**:
1. **LLaMA (환경 표상 우선)**:
   - "지금 어떤 게임 규칙인가?"를 인코딩
   - Bet Type을 명시적으로 표상
   - Layer 12-15에서 강하게 나타남

2. **Gemma (결과 표상 우선)**:
   - "무슨 일이 일어났는가?"를 인코딩
   - Outcome (파산/안전)을 명시적으로 표상
   - Layer 26-40에서 강하게 나타남

**의미**:
- **다중 실현 (Multiple Realizability)**
  - 같은 입력 → 같은 출력
  - 하지만 내부 계산 경로는 다름
- 알고리즘적 다양성 증거

### Q6: 교호작용이 역전형(Disordinal)이면 문제인가요?

**A**: 문제가 아니라 **흥미로운 발견**입니다.

**역전형 교호작용**:
```
          Bankrupt   Safe
Variable     높음    낮음  ← Variable에서 Bankrupt 높음
Fixed        낮음    높음  ← Fixed에서는 Safe 높음
```

**해석**:
- Feature가 단순히 Bankrupt를 인코딩하는 것이 아님
- **조건에 따라** Bankrupt/Safe의 의미가 다름
- 복잡한 contextual representation

**보고 시 강조**:
```
"Feature L28-12265는 유의한 역전형 교호작용을 보였다
(F(1,3196)=11.5, p<.001). Variable 조건에서는 Bankrupt 게임에서
더 높은 활성화를 보인 반면 (0.0083 vs 0.0020),
Fixed 조건에서는 Safe 게임에서 더 높은 활성화를 보였다
(0.2562 vs 0.2172). 이는 동일한 feature가 조건에 따라
서로 다른 정보를 인코딩함을 시사한다."
```

### Q7: 레이어별로 분석해야 하나요, 전체를 합쳐야 하나요?

**A**: **레이어별로 따로** 분석하세요.

**이유**:
1. **레이어 간 이질성**:
   - Early layers: 저수준 특징
   - Middle layers: 조건 표상 (우리 관심)
   - Late layers: 출력 준비

2. **효과의 레이어별 변화**:
   - LLaMA L12-15: Bet Type 강함
   - LLaMA L25-31: 여전히 Bet Type 우세
   - Gemma L26-40: Outcome 우세

3. **해석 용이성**:
   - "Layer X에서 Bet Type을 인코딩"
   - 신경망의 계층적 처리 과정 이해

**예외**: Summary statistics는 전체 평균 가능
```
"LLaMA Layers 28-30 전체에서 평균 η²(Bet Type) = 0.041,
η²(Outcome) = 0.008로, Bet Type이 약 5배 더 중요했다."
```

### Q8: sparse features는 어떻게 처리하나요?

**A**: **최소 활성화 기준**을 설정하세요.

**문제**:
- SAE features는 L1 penalty로 sparse
- 극도로 sparse한 feature (활성화율 < 1%)는 통계적으로 불안정

**권장 필터링**:
```python
# 1단계: 분산 체크 (기본)
if np.std(feature_activations) == 0:
    continue

# 2단계: 활성화율 체크 (추가)
activation_rate = (feature_activations > 0).mean()
if activation_rate < 0.01:  # 1% 미만 활성화
    continue

# 3단계: 최소 평균 활성화 체크
if np.mean(feature_activations) < 0.001:
    continue
```

**CLAUDE.md 참고**:
```
additional_experiments/sae_condition_comparison/:
- CRITICAL: Sparse features (activation rate < 1%) cause
  interaction analysis artifacts
- Apply minimum activation threshold filtering before
  interpretation
```

### Q9: 결과를 논문에 어떻게 보고하나요?

**A**: **APA 형식** + **효과 크기** 병기

**템플릿**:
```
"Two-Way ANOVA 결과, Bet Type 주효과가 유의했다
(F(1, 3196) = 3156.41, p < .001, η² = .493).
Outcome 주효과도 유의했으나 효과 크기는 작았다
(F(1, 3196) = 34.62, p < .001, η² = .005).
교호작용은 통계적으로 유의했으나 실질적 효과는 미미했다
(F(1, 3196) = 11.54, p < .001, η² = .002)."
```

**Figure 캡션**:
```
Figure 2. Two-Way ANOVA Main Effects Comparison for LLaMA-3.1-8B.
Each point represents a single SAE feature (n=2,168).
X-axis shows Outcome effect size (η²), Y-axis shows Bet Type effect size (η²).
Point size and color indicate total effect size.
74.6% of features showed Bet Type > Outcome (above diagonal),
indicating that LLaMA primarily encodes betting condition information
rather than game outcomes.
```

---

## 부록 A: 수식 요약

### A.1 Two-Way ANOVA 모형

```
Y_ijk = μ + α_i + β_j + (αβ)_ij + ε_ijk

제약:
Σ α_i = 0
Σ β_j = 0
Σ (αβ)_ij = 0 (for each factor)
```

### A.2 제곱합

```
SS_total = Σ (Y_ijk - Ȳ)²

SS_A = Σ n_i (Ȳ_i - Ȳ)²

SS_B = Σ n_j (Ȳ_j - Ȳ)²

SS_AB = Σ n_ij (Ȳ_ij - Ȳ_i - Ȳ_j + Ȳ)²

SS_residual = SS_total - SS_A - SS_B - SS_AB
```

### A.3 F-통계량

```
F_A = (SS_A / df_A) / (SS_residual / df_residual)

df_A = a - 1  (a = levels of factor A)
df_B = b - 1
df_AB = (a-1)(b-1)
df_residual = N - ab
```

### A.4 Eta-squared

```
η²_A = SS_A / SS_total

Partial η²_A = SS_A / (SS_A + SS_residual)
```

---

## 부록 B: statsmodels 설치 및 사용

### B.1 설치

```bash
pip install statsmodels
# or
conda install statsmodels
```

### B.2 기본 사용법

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 데이터프레임 준비
df = pd.DataFrame({
    'response': [값들...],
    'factor_a': [그룹 라벨들...],
    'factor_b': [그룹 라벨들...]
})

# 모델 적합
model = ols('response ~ C(factor_a) + C(factor_b) + C(factor_a):C(factor_b)', data=df).fit()

# ANOVA 테이블
anova_table = anova_lm(model, typ=2)
print(anova_table)
```

### B.3 Type 지정

```python
# Type I (Sequential)
anova_lm(model, typ=1)

# Type II (Hierarchical) ← 권장 (불균형 설계)
anova_lm(model, typ=2)

# Type III (Marginal)
anova_lm(model, typ=3)
```

---

## 부록 C: 추가 참고 자료

### C.1 통계 이론

- **Kirk, R. E. (2013)**. *Experimental design: Procedures for the behavioral sciences* (4th ed.). SAGE Publications.
- **Maxwell, S. E., & Delaney, H. D. (2004)**. *Designing experiments and analyzing data: A model comparison perspective* (2nd ed.). Lawrence Erlbaum Associates.

### C.2 Python 구현

- **statsmodels 공식 문서**: https://www.statsmodels.org/stable/anova.html
- **scipy.stats.f_oneway**: One-Way ANOVA
- **pingouin.anova**: 간편한 ANOVA 라이브러리

### C.3 효과 크기

- **Cohen, J. (1988)**. *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.
- **Richardson, J. T. E. (2011)**. Eta squared and partial eta squared as measures of effect size in educational research. *Educational Research Review*, 6(2), 135-147.

---

**문서 작성**: 2026-02-02
**분석 도구**: statsmodels 0.14.6, Python 3.11
**실험 데이터**: LLaMA-3.1-8B Layers 28-30, 3,200 games
**작성자**: Claude Code (Anthropic)
