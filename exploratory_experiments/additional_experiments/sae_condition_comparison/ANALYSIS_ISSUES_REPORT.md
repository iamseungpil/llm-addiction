# SAE Condition Comparison Analysis - 잠재적 이슈 보고서

생성일: 2026-02-01

## 요약

SAE 조건 비교 분석의 데이터 파싱 및 통계 분석을 검증한 결과, **데이터 로딩은 정확하나 통계 분석에 중요한 이슈들이 발견됨**.

---

## ✅ 정상 작동 확인

### 1. 데이터 로딩 및 매핑
- **NPZ ↔ JSON 매핑**: 정확함
  - game_ids가 JSON 인덱스와 1:1 대응
  - outcomes 일치 검증 완료
  - bet_type 필드 정상 존재 및 매핑

- **샘플 크기**: 예상과 일치
  ```
  Total: 3,200 games
  - Variable: 1,600 (Variable-Bankrupt: 108, Variable-Safe: 1,492)
  - Fixed: 1,600 (Fixed-Bankrupt: 42, Fixed-Safe: 1,558)
  ```

### 2. Analysis 1 (Variable vs Fixed t-test)
- Cohen's d 계산: 정확함
- Pooled standard deviation 사용: 적절함
- FDR 보정: 올바르게 적용됨

### 3. Analysis 2 (Four-Way ANOVA)
- 그룹 분류: 정확함
- Eta-squared 계산: 올바름
- 주요 발견 (LLaMA L14-12265):
  ```
  eta^2 = 0.850 (매우 강한 효과)
  Group means:
    VB: 0.0083, VS: 0.0020, FB: 0.2172, FS: 0.2562
  → Fixed 조건에서 명확히 높은 활성화
  ```

---

## ⚠️ 발견된 이슈

### 이슈 1: Interaction Analysis의 통계적 아티팩트 ⚠️⚠️

**문제**: interaction_eta = 0.999~1.000인 피처가 2,413개 (전체의 92%)

**원인**: 극도로 sparse한 피처들
```python
L1-3679 (interaction_eta=0.9999):
  Mean: 0.000001
  Std: 0.000016
  Non-zero: 4 / 3,200 games (99.88% zeros)
  Group means: 모든 그룹 ≈ 0 (하나만 1e-6)
```

**왜 문제인가?**
- **샘플 크기**: 4개 게임에서만 활성화 → ANOVA 전제 조건 위반
- **분산 극소**: 거의 모든 값이 0 → 수치적 불안정성
- **eta=1.0의 의미**: 실제 상호작용이 아닌 **수치 오류**

**영향 범위**:
- Interaction 분석의 상위 피처 대부분이 신뢰 불가
- 논문 Table/Figure에 사용 시 오해 유발 가능

**해결 방법**:
1. **Sparsity threshold 추가**: 활성화율 < 1% 피처 제외
2. **Minimum activation threshold**: mean < 0.01 제외
3. **재분석 필요**: 필터링 후 interaction 재계산

---

### 이슈 2: 샘플 크기 불균형 (통계적 검정력 문제) ⚠️

**문제**: Bankrupt 그룹의 샘플 크기가 매우 작음
```
LLaMA:
  Variable-Bankrupt: 108 (6.8%)
  Fixed-Bankrupt: 42 (2.6%) ← 특히 작음

Gemma:
  Variable-Bankrupt: 465 (29.1%)
  Fixed-Bankrupt: 205 (12.8%)
```

**왜 문제인가?**
- **검정력 부족**: Fixed-Bankrupt 42건으로는 안정적인 통계 추론 어려움
- **효과 크기 과대추정 위험**: 소표본에서 우연히 큰 차이 발생 가능
- **Four-Way ANOVA**: 4개 그룹 중 하나가 n=42 → 불균형 설계

**완화 요소**:
- Analysis 1 (t-test)은 전체 Variable vs Fixed 비교 → 충분한 샘플 (1,600 vs 1,600)
- FDR 보정으로 다중 비교 보정됨

**권장 사항**:
- 논문에서 샘플 크기 명시적으로 기재
- Bootstrap confidence intervals로 안정성 검증 고려

---

### 이슈 3: Two-Way ANOVA 구현의 근사 계산 ⚠️

**문제**: `utils.py:294-391`의 `two_way_anova_simple()`이 **진정한 2-way ANOVA가 아님**

**현재 구현**:
```python
# Separate one-way ANOVAs for main effects
main_bet = one_way_anova([variable_all, fixed_all])
main_outcome = one_way_anova([bankrupt_all, safe_all])

# Interaction estimated via "difference of differences"
interaction_effect = (cell[0,0] - cell[0,1]) - (cell[1,0] - cell[1,1])
```

**진정한 2-way ANOVA와의 차이**:
- **올바른 방법**: `statsmodels.formula.api.ols()` + `anova_lm()`
  - 모든 효과를 동시에 추정
  - 올바른 자유도와 F-통계량
- **현재 방법**: 근사치
  - Main effects는 정확하지만, interaction은 approximation
  - 주석에도 명시: `"This is a simplified version for computational efficiency"`

**왜 이렇게 했나?**
- 계산 효율성: 1,015,808개 피처 × statsmodels → 매우 느림
- 대부분의 경우 근사값이 충분히 정확함

**검증**:
```python
# 상위 피처 L14-12265로 statsmodels과 비교 필요
```

**권장 사항**:
- 상위 100개 피처에 대해 statsmodels로 재계산
- 차이가 크면 논문에 명시 필요

---

### 이슈 4: Layer별 피처 수 차이 (문서화 문제)

**관찰**: 로그에서 레이어별 분석된 피처 수가 다름
```
Layer 1: 299 features
Layer 2: 485 features
Layer 12: 979 features
Layer 31: 1,171 features
```

**질문**:
- 각 레이어에 32,768개 피처가 있는데 왜 일부만 분석?
- **원인 추정**: `if np.std(v_vals) == 0 and np.std(f_vals) == 0: continue`
  - 분산이 0인 피처(dead features) 제외
  - 이것은 **올바른 처리**

**확인 필요**:
- Dead feature 비율이 레이어마다 다른가?
- Gemma는 131K 피처/layer인데 실제 분석된 수는?

---

### 이슈 5: Gemma "극단적 활성화" 패턴의 해석 주의 ⚠️

**관찰**: Gemma의 Bankrupt vs Safe 활성화 차이가 50~100배
```
L40-108098:
  VB: 33.03, VS: 0.41, FB: 28.33, FS: 0.64
  → Bankrupt에서 50배 높음
```

**잠재적 오해**:
- 이것이 "Gemma가 파산을 더 강하게 인코딩"을 의미하는가?
- 아니면 단순히 **SAE 학습 설정의 차이**인가?

**비교**:
- LlamaScope SAE: Activation 범위 0~2
- GemmaScope SAE: Activation 범위 0~50
- 두 SAE의 학습 목표, L1 penalty, sparsity 설정이 다름

**주의점**:
- 절대값 비교는 무의미 (SAE마다 스케일 다름)
- **상대적 패턴**만 의미 있음 (Bankrupt vs Safe의 비율)
- 논문에서 "Gemma가 더 강하게 인코딩"이라고 주장하려면 정규화 필요

---

## 🔍 추가 검증 필요 사항

### 1. Sparse Feature Filtering
```python
# 제안 코드
def filter_sparse_features(features, min_activation_rate=0.01, min_mean=0.001):
    """Remove extremely sparse features before analysis"""
    activation_rate = np.count_nonzero(features, axis=0) / features.shape[0]
    mean_activation = np.mean(features, axis=0)

    valid_mask = (activation_rate >= min_activation_rate) & (mean_activation >= min_mean)
    return features[:, valid_mask], np.where(valid_mask)[0]
```

### 2. Statsmodels 검증 (상위 피처)
```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Top 100 features: Compare current vs statsmodels
```

### 3. Bootstrap 신뢰 구간
```python
# For small sample groups (Fixed-Bankrupt n=42)
from scipy.stats import bootstrap
```

### 4. Dead Feature 분석
```python
# Count dead features per layer
for layer in layers:
    dead_count = count_zero_variance_features(layer)
```

---

## 📊 분석 결과 신뢰도 평가

| 분석 | 신뢰도 | 비고 |
|------|--------|------|
| **Analysis 1 (Variable vs Fixed)** | ✅ **높음** | 충분한 샘플, 올바른 통계 |
| **Analysis 2 (Four-Way ANOVA)** | ⚠️ **중간** | 샘플 불균형, 하지만 FDR 보정됨 |
| **Analysis 3 (Interaction)** | ❌ **낮음** | Sparse feature artifact, 재분석 필요 |

---

## 권장 조치

### 즉시 필요
1. ✅ **Sparse feature 필터링 추가**
   - Activation rate < 1% 제외
   - Mean activation < 0.001 제외
   - Interaction 분석 재실행

2. ⚠️ **논문에서 Interaction 결과 사용 자제**
   - 현재 결과는 통계적 아티팩트 포함
   - 필터링 후 재분석 전까지는 Supplementary에만 배치

### 개선 고려
3. 📊 **상위 피처 statsmodels 검증**
   - Top 100 features에 대해 정확한 2-way ANOVA 재계산
   - 논문 Figure에 사용할 피처는 반드시 검증

4. 📝 **문서화 강화**
   - 샘플 크기 불균형 명시
   - SAE 스케일 차이 설명
   - Dead feature 제외 기준 기재

---

## 결론

**핵심 발견은 여전히 유효함**:
- ✅ LLaMA는 베팅 조건을, Gemma는 결과를 인코딩 (Analysis 1, 2에서 확인)
- ✅ 레이어 분포 차이 존재 (LLaMA L12-15 vs Gemma L26-40)
- ✅ 효과 크기 차이 존재 (Cohen's d 최대 4.75 vs 3.67)

**주의해야 할 점**:
- ⚠️ Interaction 분석은 sparse feature 필터링 후 재실행 필요
- ⚠️ 절대 활성화 값은 SAE마다 다르므로 비교 주의
- ⚠️ 소표본 그룹(Fixed-Bankrupt n=42)의 결과는 신중히 해석

**논문 작성 시**:
- Analysis 1과 2의 결과는 충분히 신뢰 가능 → Main Figure로 사용 가능
- Analysis 3은 필터링 후 재분석 → 현재는 Supplementary에만
- Limitations section에 샘플 불균형과 SAE 스케일 차이 명시
