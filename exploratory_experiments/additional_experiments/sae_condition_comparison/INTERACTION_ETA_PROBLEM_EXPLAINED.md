# Interaction Eta = 1.0 문제 상세 설명

## TL;DR

**문제**: Sparse SAE 피처에서 interaction_eta ≈ 1.0이 대량 발생
**원인**: 극소수 샘플만 활성화 → Additive model 실패 → 전체 변동이 "interaction"으로 잘못 분류됨
**해결**: Minimum activation threshold 필터링 필요

---

## 실제 데이터로 재현

### L1-3679 피처 (Interaction eta = 0.9999)

#### 기본 통계
```
Total samples: 3,200
Non-zero count: 4 (0.125%)
Zero count: 3,196 (99.875%)

Mean: 0.000000556
Std:  0.000015704
Max:  0.000444442
```

#### 4개 그룹별 분포
```
VB (Variable-Bankrupt):      n=108,  mean=0,  non-zero=0
VS (Variable-Safe):          n=1492, mean=0,  non-zero=0
FB (Fixed-Bankrupt):         n=42,   mean=0,  non-zero=0
FS (Fixed-Safe):             n=1558, mean=1.14e-6, non-zero=4  ← 모든 활성화가 여기!
```

**핵심**: 3,200개 게임 중 단 4개 게임에서만 활성화, 그것도 모두 같은 그룹(FS)에 집중

---

## 왜 Interaction Eta ≈ 1.0이 나오는가?

### 단계별 분석

#### 1단계: 4-Way ANOVA Eta-squared (정상)
```
SS_total:   7.89e-07
SS_between: 1.04e-09  (0.13% of total)
SS_within:  7.88e-07  (99.87% of total)

Eta-squared = SS_between / SS_total = 0.0013
```

**해석**: 4개 그룹 간 차이는 거의 없음 (eta=0.0013). 이것이 **정상적인 결과**.

#### 2단계: Interaction 계산 (문제 발생!)

**Additive Model 예측**:
```
Marginal means:
  Fixed: 1.11e-6
  Voluntary_stop: 5.83e-7

Predicted value (for FS group):
  = Fixed + Voluntary_stop - Grand_mean
  = 1.11e-6 + 5.83e-7 - 5.56e-7
  = 1.14e-6
```

**실제 값 vs 예측 값**:
```
Non-zero 위치 (모두 FS 그룹):
  Index 280: actual=4.44e-4, predicted=1.14e-6, residual=4.43e-4
  Index 284: actual=4.44e-4, predicted=1.14e-6, residual=4.43e-4
  Index 285: actual=4.44e-4, predicted=1.14e-6, residual=4.43e-4
  Index 292: actual=4.44e-4, predicted=1.14e-6, residual=4.43e-4
```

**Residual이 실제 값의 400배!**

#### 3단계: Interaction SS 계산

```python
SS_interaction = Σ(features - predicted_additive - grand_mean)²
               = Σ(residuals)²
               ≈ 4 × (4.43e-4)²
               = 7.89e-07
```

**결과**:
```
SS_interaction: 7.89e-07
SS_total:       7.89e-07

Interaction eta = SS_interaction / SS_total = 0.9999 ≈ 1.0 !!!
```

---

## 왜 이것이 문제인가?

### 통계적 의미의 왜곡

**Interaction eta = 1.0의 일반적 의미**:
> "베팅 조건의 효과가 결과에 따라 완전히 달라진다"
> "Bet type과 Outcome 간에 강력한 상호작용이 있다"

**하지만 실제 상황**:
- 단 **4개 샘플**로 결정된 값
- 4개 모두 **같은 그룹**에 몰려있음
- 나머지 3,196개는 모두 0

이것은 진정한 상호작용이 아니라 **수치 오류 (numerical artifact)**입니다.

### 왜 Additive Model이 실패했는가?

**정상 케이스** (충분한 샘플):
```
FS 그룹에 1,558개 샘플:
  - 대부분: 정상적인 분포
  - Marginal means로 예측 가능
  - Residual 작음
```

**Sparse 케이스** (극소수 샘플):
```
FS 그룹에 4개만 non-zero:
  - 4개가 우연히 높은 값 (4.44e-4)
  - Marginal mean은 1.14e-6 (1,558개 평균)
  - 4개 값과 평균의 괴리 = 400배
  - Residual = 거의 전체 값
```

**핵심**: Marginal mean이 극소수 outlier를 대표하지 못함

---

## 수학적 원리

### Eta-squared 정의

```
η² = SS_between / SS_total

where:
  SS_total = Σ(x_i - grand_mean)²
  SS_between = Σ n_k (x̄_k - grand_mean)²
  SS_within = Σ Σ (x_ij - x̄_k)²
```

### Sparse Feature의 특수성

**대부분이 0인 경우**:

1. **Grand mean ≈ 0**
   - 3,196개가 0이므로

2. **SS_total = 4개 non-zero 값으로만 결정**
   ```
   SS_total ≈ Σ(non-zero values)²
   ```

3. **SS_within ≈ SS_total**
   - 4개가 모두 같은 그룹 → group mean ≈ 4개의 평균
   - 하지만 전체 1,558개의 평균은 거의 0
   - Within-group variance가 전체 variance를 차지

4. **Additive model prediction ≈ 0**
   - Marginal means가 모두 거의 0
   - Predicted ≈ 0 + 0 - 0 = 0

5. **Residual ≈ Actual value**
   ```
   Residual = Actual - Predicted - Grand_mean
            ≈ 4.44e-4 - 0 - 0
            = 4.44e-4
   ```

6. **SS_interaction ≈ SS_total**
   ```
   SS_interaction = Σ(Residual)²
                  ≈ 4 × (4.44e-4)²
                  = SS_total
   ```

7. **Interaction eta ≈ 1.0**

---

## 왜 이것이 2,413개나 발생했는가?

### SAE의 본질적 특성

**Sparse Autoencoder**는 의도적으로 sparse한 표상을 학습합니다:
- L1 penalty로 대부분의 feature를 0으로 억제
- 특정 입력에만 선택적으로 활성화

**결과**:
```
LLaMA Layer 1:
  - Total features: 32,768
  - Extremely sparse (<1% active): 32,479 (99.1%)
  - Near-zero mean: 32,493 (99.2%)
```

### 왜 현재 코드가 이를 걸러내지 못했나?

**현재 필터 (condition_comparison.py:79-80)**:
```python
# Skip if no variance
if np.std(v_vals) == 0 and np.std(f_vals) == 0:
    continue
```

**문제**: 이 조건은 **모든 값이 완전히 같을 때만** 제외합니다.

**L1-3679의 경우**:
```
VB: std=0, VS: std=0, FB: std=0, FS: std=0.0000225

→ FS에서 std > 0이므로 필터 통과!
```

하지만 실제로는:
- FS의 1,554개는 0
- 단 4개만 4.44e-4
- Std > 0이지만 **통계적으로 무의미**

---

## 실제 영향

### 분석 결과에서

**Interaction 분석 (Analysis 3)**:
```
Total features: 1,015,808
Significant interactions (eta >= 0.01): 2,616

이 중:
  - Interaction eta > 0.99: 2,413개 (92%)
  - 대부분이 이런 sparse artifact
```

**논문에 기재된 상위 피처들**:
```
L16-19395: int_eta=1.000
L21-4648:  int_eta=1.000
L8-20026:  int_eta=1.000
...

→ 모두 신뢰 불가!
```

---

## 해결 방법

### 1. Minimum Activation Threshold (권장)

```python
def filter_sparse_features(features, min_activation_rate=0.01, min_mean=0.001):
    """
    Remove extremely sparse features before analysis.

    Args:
        min_activation_rate: 최소 활성화 비율 (default: 1%)
        min_mean: 최소 평균 활성화 (default: 0.001)
    """
    activation_rate = np.count_nonzero(features, axis=0) / features.shape[0]
    mean_activation = np.mean(features, axis=0)

    valid_mask = (activation_rate >= min_activation_rate) & (mean_activation >= min_mean)
    return features[:, valid_mask], np.where(valid_mask)[0]
```

**적용 시 효과**:
- L1-3679: activation_rate = 0.125% < 1% → 제외 ✓
- 정상 피처: activation_rate > 1% → 유지 ✓

### 2. Minimum Sample Size per Group

```python
def check_group_sample_size(grouped, min_samples=10):
    """각 그룹에 최소 샘플 수 확인"""
    for name, arr in grouped.items():
        non_zero_count = np.count_nonzero(arr, axis=0)
        if non_zero_count.min() < min_samples:
            # Skip this feature
            continue
```

### 3. Proper 2-Way ANOVA with statsmodels

```python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# For top features only (computational efficiency)
model = ols('feature ~ C(bet_type) * C(outcome)', data=df).fit()
anova_table = anova_lm(model, typ=2)
```

---

## 권장 조치

### 즉시 필요
1. ✅ **Sparse feature 필터링 추가**
   - `min_activation_rate = 0.01` (1% 이상만 유지)
   - `min_mean = 0.001` (평균 활성화 임계값)

2. ⚠️ **Interaction 분석 재실행**
   - 필터링 후 결과가 완전히 달라질 것

3. ❌ **현재 Interaction 결과 사용 금지**
   - 논문 Main Figure/Table에서 제외
   - Supplementary에도 "preliminary"로만 표기

### 논문 작성 시
- Limitations section에 sparse feature 문제 명시
- "Interaction analysis requires further validation with minimum activation filtering"

---

## 요약

| 항목 | 값 |
|------|-----|
| **문제 피처** | L1-3679 |
| **활성화** | 4/3,200 games (0.125%) |
| **그룹 분포** | 모두 FS에 집중 |
| **4-way eta** | 0.0013 (정상) |
| **Interaction eta** | 0.9999 (오류!) |
| **원인** | Additive model 실패 (400배 예측 오차) |
| **영향** | 2,413개 피처에서 유사 문제 발생 |
| **해결** | Activation rate < 1% 필터링 |

**핵심 교훈**: Sparse data에서 복잡한 통계 모델(interaction ANOVA)은 샘플 크기 검증 필수!
