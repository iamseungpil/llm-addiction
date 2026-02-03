# 통계 분석 개념 이해하기

## SAE 실험 통계를 위한 직관적 학습 가이드

이 문서는 우리 실험의 통계 분석을 이해하기 위한 **개념 중심** 학습 자료입니다.
수식은 최소화하고, 직관적인 비유와 예시로 설명합니다.

---

## 🎯 학습 목표

이 자료를 다 읽고 나면:
- **t-test**가 무엇을 하는지 설명할 수 있습니다
- **ANOVA**가 언제 필요한지 알 수 있습니다
- **p-value**와 **effect size**의 차이를 이해합니다
- **FDR 보정**이 왜 필요한지 알게 됩니다
- **상호작용(interaction)**을 그래프로 구분할 수 있습니다

---

## 📚 목차

**Part 1: 기초 개념 (30분)**
1. [평균과 분산: 데이터의 모습 파악하기](#part-1-평균과-분산)
2. [가설 검정: 차이가 진짜일까, 우연일까?](#part-2-가설-검정)
3. [p-value: 통계적 확신의 정도](#part-3-p-value)

**Part 2: 비교 방법 (40분)**
4. [t-test: 두 그룹 비교하기](#part-4-t-test)
5. [Cohen's d: 차이가 얼마나 클까?](#part-5-cohens-d)
6. [ANOVA: 여러 그룹 비교하기](#part-6-anova)
7. [Eta-squared: ANOVA의 효과 크기](#part-7-eta-squared)

**Part 3: 고급 주제 (40분)**
8. [다중 비교 문제: 복권을 많이 사면 당첨 확률이 높아진다?](#part-8-다중-비교-문제)
9. [FDR 보정: 거짓 발견을 줄이는 방법](#part-9-fdr-보정)
10. [상호작용: 1+1이 2가 아닐 때](#part-10-상호작용)

**Part 4: 실전 적용 (20분)**
11. [우리 실험 이해하기](#part-11-우리-실험-이해하기)
12. [결과 해석 실습](#part-12-결과-해석-실습)

---

# Part 1: 기초 개념

## Part 1: 평균과 분산

### 📊 평균 (Mean)

**비유:** 5명의 키를 재면 → 평균키는 그 팀의 "대표 키"

**예시:**
```
5명의 키: 160, 165, 170, 175, 180 (단위: cm)
평균 = (160+165+170+175+180) / 5 = 170cm
```

**의미:** "이 팀을 한 명으로 대표하면 키가 170cm"

---

### 📏 분산 (Variance)과 표준편차 (Standard Deviation)

**비유:** 같은 평균 170cm라도...

**케이스 A: 키가 비슷함**
```
키: 168, 169, 170, 171, 172
평균: 170cm
→ 다들 비슷하게 170cm 근처
```

**케이스 B: 키가 다양함**
```
키: 150, 160, 170, 180, 190
평균: 170cm (같음!)
→ 하지만 키가 엄청 다양함
```

**분산/표준편차:** "평균 주변에 데이터가 얼마나 퍼져있는가"
- **케이스 A:** 분산 작음 (모두 비슷)
- **케이스 B:** 분산 큼 (차이가 큼)

**왜 중요한가?**
→ 같은 평균이어도 "얼마나 일정한가"를 알아야 데이터를 이해할 수 있음!

---

## Part 2: 가설 검정

### 🤔 핵심 질문: "이 차이는 진짜일까, 우연일까?"

**상황:**
- Variable 조건: 평균 activation = 2.5
- Fixed 조건: 평균 activation = 2.0
- 차이 = 0.5

**질문:** 이 0.5 차이가...
- **진짜 차이?** Variable 조건이 정말로 activation을 높인다
- **우연?** 샘플링 운이 나빠서 우연히 0.5 차이가 난 것

---

### 🎲 동전 던지기 비유

**상황:** 동전을 10번 던졌더니 앞면이 7번 나옴

**질문:** 이 동전은 편향된 동전일까?

**두 가지 가능성:**
1. **공정한 동전** (50:50) → 운이 좋아서 7번 나온 것
2. **편향된 동전** (앞면 확률 > 50%) → 진짜 앞면이 잘 나오는 동전

**통계적 사고:**
- 공정한 동전에서 10번 중 7번 이상 나올 확률 = 약 17%
- 17%는 꽤 높은 확률 → "우연히 일어날 수 있는 범위"
- **결론:** 이것만으로는 편향되었다고 확신할 수 없음

**만약 100번 중 70번이면?**
- 공정한 동전에서 100번 중 70번 이상 나올 확률 = 약 0.003% (매우 낮음!)
- **결론:** 이건 우연으로 보기 힘들다 → 편향된 동전일 가능성 높음

---

### 📋 가설 검정의 구조

**1단계: 두 가지 가설 세우기**
- **귀무가설 (H₀):** "차이가 없다" (공정한 동전)
- **대립가설 (H₁):** "차이가 있다" (편향된 동전)

**2단계: 데이터 수집**
- 실험 실행 → 결과 관찰

**3단계: 계산**
- "귀무가설이 참일 때, 이런 결과가 나올 확률은?"

**4단계: 판단**
- 확률이 매우 낮으면 (< 5%) → 귀무가설 기각 → "차이가 있다!"
- 확률이 높으면 (≥ 5%) → 귀무가설 기각 못함 → "차이가 있다고 확신 못함"

---

## Part 3: p-value

### 🎯 p-value란?

**정의 (쉽게):**
"차이가 없다고 가정했을 때, 이 정도 결과가 우연히 나올 확률"

**동전 비유:**
- 공정한 동전 (차이 없음) 가정
- 100번 중 70번 앞면 관찰
- p-value = 0.003% → "공정한 동전에서 이런 일이 일어날 확률"

---

### 📊 p-value 해석

**p-value가 작을수록 → "우연으로 보기 힘들다"**

| p-value | 해석 | 비유 |
|---------|------|------|
| p = 0.5 | 매우 흔한 일 | 동전 던져서 앞면 나올 확률 |
| p = 0.1 | 흔한 편 | 주사위에서 6 나올 확률 |
| **p = 0.05** | **기준선** | **20번 중 1번 일어날 일** |
| p = 0.01 | 드문 일 | 100번 중 1번 |
| p = 0.001 | 매우 드문 일 | 1000번 중 1번 |

**관례:** p < 0.05를 "통계적으로 유의하다"의 기준으로 사용
- p < 0.05: "우연으로 보기엔 너무 드문 일 → 진짜 차이가 있을 것"
- p ≥ 0.05: "우연히 일어날 수 있는 범위 → 차이가 있다고 확신 못함"

---

### ⚠️ p-value 오해 주의!

**❌ 틀린 해석:** "p = 0.01이면 1% 확률로 차이가 있다"

**✅ 올바른 해석:** "차이가 없다고 가정할 때, 이 결과가 나올 확률이 1%"

**비유:**
- "무죄라고 가정했을 때, 이 증거가 나올 확률이 1%"
- → 유죄일 가능성이 매우 높다
- 하지만 "99% 확률로 유죄"라는 뜻은 아님!

---

### 📏 p-value의 한계

**문제 1: 샘플 크기에 영향받음**

**상황 A: 샘플 10개**
- Variable: 평균 2.5
- Fixed: 평균 2.0
- 차이: 0.5
- p-value = 0.2 (유의하지 않음)

**상황 B: 샘플 10,000개 (같은 차이!)**
- Variable: 평균 2.5
- Fixed: 평균 2.0
- 차이: 0.5 (동일!)
- p-value = 0.0001 (유의함!)

**해석:**
- 같은 0.5 차이인데, 샘플이 많으면 유의하게 나옴
- p-value는 **"차이의 크기"가 아니라 "통계적 확실성"**을 나타냄!

**문제 2: 실용적 중요성은 모름**

- p < 0.001 (매우 유의) → "차이가 있다는 건 확실"
- 하지만 차이가 0.0001이라면? → 실용적으로는 의미 없을 수 있음

**해결책:** p-value와 **effect size (효과 크기)**를 함께 봐야 함!

---

# Part 2: 비교 방법

## Part 4: t-test

### 🔍 t-test란?

**목적:** 두 그룹의 평균이 다른지 검정

**우리 실험:**
- 그룹 1: Variable 조건의 activation values
- 그룹 2: Fixed 조건의 activation values
- **질문:** 두 그룹의 평균 activation이 다른가?

---

### 🎯 t-test의 핵심 아이디어

**"평균 차이"를 "불확실성"으로 나눔**

**예시 A: 차이가 명확함**
```
Variable: [10.0, 10.1, 9.9, 10.0, 10.1]  평균 10.0
Fixed:    [5.0, 5.1, 4.9, 5.0, 5.1]      평균 5.0

평균 차이 = 5.0
데이터가 일정함 (분산 작음)
→ t-statistic 큼 → p-value 작음 → "차이가 있다!"
```

**예시 B: 차이가 불명확함**
```
Variable: [3, 20, 1, 15, 11]  평균 10.0
Fixed:    [1, 12, 3, 8, 1]    평균 5.0

평균 차이 = 5.0 (같음!)
하지만 데이터가 들쭉날쭉 (분산 큼)
→ t-statistic 작음 → p-value 큼 → "차이가 있다고 확신 못함"
```

**결론:**
- 같은 평균 차이라도, 데이터가 일정해야 "진짜 차이"라고 확신할 수 있음
- t-test는 이 두 가지를 모두 고려

---

### 🌡️ t-statistic의 의미

**비유:** 신호 대 잡음비 (Signal-to-Noise Ratio)

```
t = 신호 (평균 차이) / 잡음 (데이터의 들쭉날쭉함)
```

- t가 클수록 → 신호가 명확 → 차이가 확실
- t가 작으면 → 잡음에 묻힘 → 차이가 불명확

---

### ✅ 우리가 사용하는 Welch's t-test

**일반 t-test vs Welch's t-test:**

**일반 t-test (Student's):**
- 가정: "두 그룹의 분산이 같다"
- 문제: 가정이 틀리면 결과가 부정확

**Welch's t-test:**
- 가정: "두 그룹의 분산이 달라도 된다"
- 장점: 더 안전함 (robust)

**우리 선택:** Welch's t-test
→ Variable과 Fixed의 분산이 다를 수 있으므로

---

## Part 5: Cohen's d

### 📐 Cohen's d란?

**목적:** "차이가 얼마나 큰가?"를 표준화된 척도로 표현

**왜 필요한가?**

**상황 1: 키 차이**
- 남성 평균 키: 175cm
- 여성 평균 키: 162cm
- 차이: 13cm

**상황 2: 몸무게 차이**
- 남성 평균 몸무게: 70kg
- 여성 평균 몸무게: 57kg
- 차이: 13kg

**질문:** 13cm 차이와 13kg 차이 중 어느 것이 더 큰가?
→ 단위가 다르니까 직접 비교 불가!

**Cohen's d:** 표준편차를 기준으로 표준화
→ 단위에 상관없이 비교 가능

---

### 📊 Cohen's d 계산 (개념)

**아이디어:**
"평균 차이가 표준편차의 몇 배인가?"

**예시:**
```
그룹 A: 평균 100, 표준편차 10
그룹 B: 평균 110, 표준편차 10
평균 차이 = 10

Cohen's d = 10 / 10 = 1.0
→ "평균 차이가 표준편차의 1배"
```

---

### 📏 Cohen's d 해석 가이드

**부호:**
- d > 0: 그룹 1이 더 큼 (우리: Variable이 높음)
- d < 0: 그룹 2가 더 큼 (우리: Fixed가 높음)

**크기:**
| |d| | 효과 크기 | 비유 | 해석 |
|------|----------|------|------|
| 0.2 | Small | 약한 차이 | "차이는 있지만 미미함" |
| 0.5 | Medium | 중간 차이 | "눈에 띄는 차이" |
| 0.8 | Large | 큰 차이 | "확연한 차이" |

**우리 실험:** `min_cohens_d = 0.3`
→ Small과 Medium 사이 이상만 선택

---

### 🎯 p-value vs Cohen's d

**시나리오 1:**
- p-value = 0.001 (매우 유의!)
- Cohen's d = 0.05 (거의 없는 효과)
- **해석:** "차이가 있긴 한데, 너무 작아서 의미 없음"

**시나리오 2:**
- p-value = 0.08 (유의하지 않음)
- Cohen's d = 0.6 (중간 효과)
- **해석:** "효과는 커 보이는데, 샘플이 부족해서 확신 못함"

**이상적:**
- p-value < 0.05 (통계적으로 유의)
- Cohen's d ≥ 0.5 (실질적으로 의미 있음)
- **"확실하고 큰 차이!"**

---

## Part 6: ANOVA

### 🔬 ANOVA란?

**Analysis of Variance** (분산 분석)

**목적:** 3개 이상 그룹의 평균이 모두 같은지 검정

**t-test와의 차이:**
- t-test: 2개 그룹만 비교
- ANOVA: 3개 이상 그룹 동시 비교

---

### 🤔 왜 t-test를 여러 번 못하나?

**상황:** A, B, C 세 그룹 비교

**나쁜 방법:**
1. A vs B → t-test
2. A vs C → t-test
3. B vs C → t-test

**문제:** 다중 비교 문제! (뒤에서 설명)

**좋은 방법:**
- ANOVA 한 번만 수행
- "A, B, C의 평균이 모두 같은가?"

---

### 🧩 ANOVA의 핵심 아이디어

**"전체 변동"을 두 부분으로 나눔**

**예시: 학생들의 시험 점수**
```
A반: [80, 85, 90]  평균 85
B반: [70, 75, 80]  평균 75
C반: [60, 65, 70]  평균 65

전체 평균: 75
```

**전체 변동 = 그룹 간 변동 + 그룹 내 변동**

**그룹 간 변동:**
- A반 평균(85) vs 전체 평균(75) → 차이 +10
- B반 평균(75) vs 전체 평균(75) → 차이 0
- C반 평균(65) vs 전체 평균(75) → 차이 -10
- **의미:** "반마다 평균이 얼마나 다른가?"

**그룹 내 변동:**
- A반 안에서 80, 85, 90의 들쭉날쭉함
- B반 안에서 70, 75, 80의 들쭉날쭉함
- **의미:** "같은 반 안에서도 학생마다 점수가 다르다" (오차)

---

### 📊 F-statistic

**F = 그룹 간 변동 / 그룹 내 변동**

**해석:**
- F가 크면 → 그룹 간 차이가 그룹 내 차이보다 훨씬 큼
  - **"반마다 평균이 확실히 다르다"**
- F가 작으면 → 그룹 간 차이가 그룹 내 차이와 비슷
  - **"반 차이보다 개인차가 더 크다"**

---

### 🎯 우리 실험: Four-Way ANOVA

**4개 그룹:**
1. Variable + Bankruptcy
2. Variable + Voluntary Stop
3. Fixed + Bankruptcy
4. Fixed + Voluntary Stop

**질문:** "이 4개 그룹의 평균 activation이 모두 같은가?"

**결과:**
- p < 0.05: "적어도 한 그룹은 다르다"
- p ≥ 0.05: "모두 같다고 봐도 무방"

**주의:** ANOVA는 "어느 그룹이 다른지"는 안 알려줌
→ post-hoc test 필요 (우리는 안 함, 탐색적 분석이므로)

---

## Part 7: Eta-squared

### 📐 Eta-squared (η²)란?

**목적:** ANOVA의 효과 크기 측정

**개념:**
"전체 변동 중에서 그룹 간 차이로 설명되는 비율"

```
η² = 그룹 간 변동 / 전체 변동
```

**범위:** 0 ~ 1

---

### 🎯 Eta-squared 해석

**예시 1: η² = 0.01 (1%)**
- 전체 변동의 1%만 그룹 차이로 설명
- 나머지 99%는 개인차 (오차)
- **해석:** "그룹이 조금 다르긴 한데, 개인차가 훨씬 큼"

**예시 2: η² = 0.5 (50%)**
- 전체 변동의 50%를 그룹 차이로 설명
- **해석:** "그룹이 확연히 다르다"

---

### 📊 Eta-squared 기준

| η² | 효과 크기 | 의미 |
|----|----------|------|
| 0.01 | Small | 그룹 차이가 1% 설명 |
| 0.06 | Medium | 그룹 차이가 6% 설명 |
| 0.14 | Large | 그룹 차이가 14% 설명 |

**우리 실험:** `min_eta_squared = 0.01`
→ Small effect 이상만 선택

---

### 🔍 Cohen's d vs Eta-squared

**Cohen's d:**
- 2개 그룹 비교 (t-test)
- "평균 차이 / 표준편차"

**Eta-squared:**
- 3개 이상 그룹 비교 (ANOVA)
- "설명된 변동 / 전체 변동"

**공통점:** 둘 다 "효과의 실질적 크기" 측정

---

# Part 3: 고급 주제

## Part 8: 다중 비교 문제

### 🎰 복권 비유

**상황 1: 복권 1장 구매**
- 당첨 확률: 5%
- 5% = 1/20

**상황 2: 복권 100장 구매**
- 각각 당첨 확률: 5%
- **적어도 1장 당첨 확률: 99.4%!**

**계산:**
```
모두 꽝일 확률 = (0.95)^100 = 0.006 (0.6%)
적어도 1장 당첨 = 1 - 0.006 = 99.4%
```

**교훈:** 시도를 많이 하면 우연히 맞출 확률이 급증!

---

### 🔬 통계 검정의 다중 비교

**상황:** 우리 실험
- LLaMA: 7 layers × 131,072 features = **917,504개 가설 검정**

**문제:**
- 각 검정: α = 0.05 (5% 거짓 양성 확률)
- 917,504개 검정 → 예상 거짓 양성 = 917,504 × 0.05 = **45,875개!**

**즉:**
- 차이가 전혀 없는 feature도
- 917,504개를 검정하다 보면
- 우연히 45,875개가 "유의함"으로 나옴!

---

### 🎯 복권 비유로 이해하기

**올바른 결과 (차이가 진짜 있는 feature):**
- 복권 당첨 = 보물 발견

**거짓 양성 (차이가 없는데 유의하게 나온 feature):**
- 복권 당첨 = 가짜 보물

**보정 없이:**
- 복권 45,875장 당첨 (가짜 보물)
- 진짜 보물이 몇 개든 간에, 가짜가 섞여있음

**해결책:** 당첨 기준을 더 엄격하게!

---

## Part 9: FDR 보정

### 🛡️ 두 가지 보정 전략

### **전략 1: Bonferroni (가장 보수적)**

**아이디어:** "기준을 검정 개수로 나눔"

**원래:**
- α = 0.05 (5%)

**Bonferroni:**
- α_adjusted = 0.05 / 917,504 = 0.00000005 (매우 엄격!)

**장점:** 거짓 양성을 강력히 통제
**단점:** 너무 보수적 → 진짜 효과도 많이 놓침

---

### **전략 2: FDR (우리가 사용)**

**False Discovery Rate (거짓 발견 비율)**

**Bonferroni:** "전체 실험에서 거짓 양성이 1개라도 나올 확률 < 5%"

**FDR:** "유의하다고 판단한 것 중에서 거짓 양성 비율 < 5%"

**예시:**
- 1,000개 feature를 유의하다고 판단
- FDR = 0.05 → 그 중 평균 50개 정도는 거짓 양성
- 나머지 950개는 진짜 효과

---

### 🔍 Bonferroni vs FDR 비교

**비유: 범죄자 체포**

**Bonferroni (완벽주의자):**
- "무고한 사람 1명이라도 체포하면 안 돼!"
- → 증거가 엄청 확실한 사람만 체포
- 문제: 진짜 범죄자도 많이 놓침

**FDR (실용주의자):**
- "체포한 사람 중 5%는 무고해도 괜찮아"
- → 합리적인 기준으로 체포
- 장점: 진짜 범죄자를 더 많이 잡음

---

### ✅ FDR이 적합한 이유

**우리 상황:**
- 917,504개 feature 탐색
- 목적: 흥미로운 feature 발견 (탐색적 연구)

**Bonferroni 사용 시:**
- 너무 엄격 → 거의 아무것도 안 걸림
- 진짜 중요한 feature도 놓침

**FDR 사용 시:**
- 적절한 균형
- 거짓 양성은 5% 정도만 허용
- 진짜 효과는 잘 찾아냄

**추가 필터링:** effect size (d ≥ 0.3, η² ≥ 0.01)
→ FDR + effect size로 더블 체크!

---

### 📊 Benjamini-Hochberg 절차 (개념만)

**FDR을 구현하는 방법**

**아이디어:**
1. 모든 p-value를 작은 순서대로 정렬
2. 위에서부터 순서대로 체크
3. 어느 지점까지가 "유의"인지 자동으로 결정

**장점:**
- 자동으로 최적의 cutoff 찾음
- 검정력(power)이 높음

**실제 사용:**
```python
from statsmodels.stats.multitest import multipletests
rejected, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

---

## Part 10: 상호작용

### 🤝 상호작용(Interaction)이란?

**정의:** "한 변수의 효과가 다른 변수에 따라 달라지는 현상"

**쉬운 비유: 카페인과 운동**

**상호작용 없음:**
- 카페인 효과: 집중력 +10
- 운동 효과: 집중력 +5
- 카페인 + 운동 = +10 + 5 = **+15** (단순 합)

**상호작용 있음:**
- 카페인 효과: 집중력 +10
- 운동 효과: 집중력 +5
- 카페인 + 운동 = **+25** (1+1=3!)
- → 카페인과 운동이 **시너지**를 냄

---

### 🎯 우리 실험의 상호작용

**두 변수:**
- **Bet Type:** Variable vs Fixed
- **Outcome:** Bankruptcy vs Voluntary Stop

**상호작용 질문:**
"Bet Type의 효과가 Outcome에 따라 달라지는가?"

---

### 📊 상호작용 예시: 없는 경우

**Feature X의 activation:**

| | Bankruptcy | Voluntary Stop | 차이 |
|---|------------|----------------|------|
| Variable | 10 | 20 | +10 |
| Fixed | 5 | 15 | +10 |
| 차이 | +5 | +5 | - |

**관찰:**
- Variable-Fixed 차이가 Outcome에 관계없이 일정 (+5)
- Bankruptcy-Voluntary Stop 차이가 Bet Type에 관계없이 일정 (+10)

**그래프:**
```
Activation
   |
20 |                  Variable
   |                /
15 |              /  Fixed
10 |            /
 5 |          /
   |________/_______________
     Bankrupt   Vol.Stop
```
→ **평행선** = 상호작용 없음

---

### 📈 상호작용 예시: 있는 경우

**Feature Y의 activation:**

| | Bankruptcy | Voluntary Stop | 차이 |
|---|------------|----------------|------|
| Variable | 5 | 25 | +20 |
| Fixed | 5 | 10 | +5 |
| 차이 | 0 | +15 | - |

**관찰:**
- Variable-Fixed 차이가 Outcome에 따라 다름
  - Bankruptcy: 0
  - Voluntary Stop: +15
- → **상호작용 있음!**

**그래프:**
```
Activation
   |
25 |              Variable
   |            /
   |          /
10 |  Fixed /
 5 |______/________________
     Bankrupt   Vol.Stop
```
→ **교차/비평행** = 상호작용 있음!

---

### 🧩 상호작용의 의미

**상호작용 없음:**
- Variable의 효과: "일관되게 +5"
- Outcome의 효과: "일관되게 +10"
- 두 요인이 **독립적**으로 작용

**상호작용 있음:**
- Variable의 효과가 Outcome에 따라 다름
- Voluntary Stop일 때만 Variable이 크게 높아짐
- 두 요인이 **결합하여 추가 효과** 발생

**해석 예:**
"Variable betting의 효과는 voluntary stop한 경우에만 크게 나타난다"
→ 이런 패턴은 신경학적으로 흥미로울 수 있음!

---

### ⚠️ 우리 코드의 주의사항

**Simplified Two-Way ANOVA 사용:**
- 정확한 방법이 아니라 **근사값**
- 계산 속도를 위한 trade-off

**Sparse Feature 문제:**
- SAE features는 대부분 sparse (거의 0)
- Sparse feature는 상호작용 분석에서 **허위 결과** 생성
- Interaction eta ≈ 1.0인 feature의 92%가 artifact!

**해결책:**
- Minimum activation threshold 필터링
- Top features는 정확한 방법(statsmodels)으로 재검증

**자세한 내용:**
- `INTERACTION_ETA_PROBLEM_EXPLAINED.md` 참조

---

# Part 4: 실전 적용

## Part 11: 우리 실험 이해하기

### 📋 전체 파이프라인

```
Input
├── SAE features (NPZ): layer별 131,072 features
└── Experiment data (JSON): bet_type, outcome 정보

↓

Analysis 1: Variable vs Fixed
├── 각 feature마다 Welch's t-test
├── Cohen's d 계산
└── FDR 보정
→ Variable과 Fixed에서 다른 feature 발견

↓

Analysis 2: Four-Way Comparison
├── 4개 그룹 One-Way ANOVA
├── Eta-squared 계산
└── FDR 보정
→ 4개 조건에서 다른 feature 발견

↓

Analysis 3: Interaction Analysis
├── Simplified Two-Way ANOVA
├── Main effects + Interaction eta
└── FDR 보정 (주의: sparse 문제)
→ 상호작용 효과 탐색

↓

Output
├── Top features (effect size 큰 순)
├── FDR-corrected p-values
└── Summary statistics
```

---

### 🎯 각 분석의 목적

**Analysis 1: Variable vs Fixed**
- **질문:** "Variable betting이 Fixed와 다른 neural feature를 활성화하는가?"
- **방법:** 단순 비교 (가장 신뢰도 높음)
- **결과 해석:** d > 0이면 Variable에서 높음, d < 0이면 Fixed에서 높음

**Analysis 2: Four-Way**
- **질문:** "Bet Type과 Outcome의 조합에 따라 feature가 다른가?"
- **방법:** 4개 그룹 동시 비교
- **결과 해석:** 어느 조건 조합이 가장 높은지 패턴 확인

**Analysis 3: Interaction**
- **질문:** "Bet Type의 효과가 Outcome에 따라 달라지는가?"
- **방법:** 2×2 factorial (단, 근사값)
- **결과 해석:** 상호작용 eta가 높으면 복잡한 패턴 존재
- **⚠️ 주의:** Sparse 문제로 인한 artifact 가능성

---

### 📊 결과 신뢰도 순위

1. **Analysis 1 (Variable vs Fixed)** ⭐⭐⭐⭐⭐
   - 가장 단순하고 robust
   - Welch's t-test는 검증된 방법
   - Cohen's d는 직관적

2. **Analysis 2 (Four-Way)** ⭐⭐⭐⭐
   - One-Way ANOVA는 신뢰
   - Eta-squared는 표준 지표
   - Post-hoc 없어도 패턴 파악 가능

3. **Analysis 3 (Interaction)** ⭐⭐⭐
   - Simplified 방법 (근사)
   - Sparse feature artifact 위험
   - Top features는 재검증 필수

---

## Part 12: 결과 해석 실습

### 📝 예시 1: Analysis 1 결과

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
  "variable_std": 0.45,
  "fixed_std": 0.38,
  "direction": "higher_in_variable"
}
```

**단계별 해석:**

**1단계: 유의성 확인**
- `p_fdr = 3.4e-12` (0.00000000034)
- ✅ p < 0.05 → **통계적으로 유의함**

**2단계: 효과 크기**
- `cohens_d = 0.52`
- ✅ |d| > 0.5 → **중간 크기 효과 (Medium)**

**3단계: 방향 확인**
- `direction = "higher_in_variable"`
- Variable 평균 (2.34) > Fixed 평균 (1.89)
- ✅ Variable 조건에서 **더 높게 활성화됨**

**4단계: 일관성 확인**
- `variable_std = 0.45`, `fixed_std = 0.38`
- 표준편차가 평균에 비해 작음 → 일관된 패턴

**종합 해석:**
"Layer 12의 Feature 45678은 Variable betting 조건에서 유의하게 높게 활성화됨 (p < 0.001, d = 0.52). 이 feature는 Variable betting과 관련된 신경 표현을 encode할 가능성이 있음."

---

### 📝 예시 2: Analysis 2 결과

```json
{
  "layer": 15,
  "feature_id": 89012,
  "f_stat": 24.5,
  "p_value": 2.1e-19,
  "p_fdr": 5.3e-16,
  "eta_squared": 0.08,
  "group_means": {
    "variable_bankrupt": 3.2,
    "variable_safe": 1.8,
    "fixed_bankrupt": 2.9,
    "fixed_safe": 1.5
  }
}
```

**단계별 해석:**

**1단계: 유의성**
- `p_fdr < 0.001` → ✅ **매우 유의함**

**2단계: 효과 크기**
- `eta_squared = 0.08` (8%)
- ✅ Medium effect (0.06~0.14 범위)
- 전체 변동의 8%를 그룹 차이로 설명

**3단계: 패턴 분석**
```
Bankruptcy 조건:
  Variable: 3.2
  Fixed:    2.9
  차이:     +0.3

Voluntary Stop 조건:
  Variable: 1.8
  Fixed:    1.5
  차이:     +0.3
```

**관찰:**
- **Bankruptcy에서 높음** (3.2, 2.9 vs 1.8, 1.5)
- Variable이 Fixed보다 일관되게 +0.3 높음

**종합 해석:**
"Layer 15의 Feature 89012는 Bankruptcy 조건에서 크게 활성화됨 (η² = 0.08, p < 0.001). Variable/Fixed 차이는 상대적으로 작음. 이 feature는 bankruptcy 상태와 관련된 표현을 encode할 가능성."

---

### 📝 예시 3: Analysis 3 결과 (주의 필요)

```json
{
  "layer": 18,
  "feature_id": 102400,
  "bet_type_f": 15.2,
  "bet_type_p": 9.1e-5,
  "bet_type_eta": 0.04,
  "outcome_f": 2.3,
  "outcome_p": 0.13,
  "outcome_eta": 0.006,
  "interaction_f": 22.7,
  "interaction_p_fdr": 1.2e-4,
  "interaction_eta": 0.055,
  "group_means": {
    "variable_bankruptcy": 2.5,
    "variable_voluntary_stop": 0.5,
    "fixed_bankruptcy": 1.8,
    "fixed_voluntary_stop": 1.2
  }
}
```

**단계별 해석:**

**1단계: 주효과 확인**
- Bet Type: p < 0.001, η² = 0.04 → ✅ **유의함**
- Outcome: p = 0.13 → ❌ **유의하지 않음**

**2단계: 상호작용**
- Interaction: p_fdr < 0.001, η² = 0.055 → ✅ **유의함**

**3단계: 패턴 분석**
```
Bankruptcy:
  Variable: 2.5
  Fixed:    1.8
  차이:     +0.7

Voluntary Stop:
  Variable: 0.5
  Fixed:    1.2
  차이:     -0.7 (역전!)
```

**그래프 상상:**
```
Activation
   |
2.5|  Variable
   |  \
1.8|   \  Fixed
   |    X
1.2|   /
0.5|  /
   |_____________
   Bankrupt  Stop
```
→ **교차 패턴 (Crossover Interaction)**

**4단계: Sparse 체크 필요!**
- ⚠️ Interaction eta가 높으면 반드시 확인:
  - Activation rate ≥ 1%?
  - Mean activation ≥ 0.001?
- 만약 sparse하면 → artifact 가능성

**종합 해석 (조건부):**
"Layer 18의 Feature 102400은 Bet Type과 Outcome의 상호작용을 보임 (p < 0.001, η² = 0.055). Variable은 bankruptcy에서 높고 voluntary stop에서 낮은 반면, Fixed는 반대 패턴. **단, sparse feature 여부 확인 필요**. Activation이 충분하다면, 이 feature는 bet type에 따라 outcome을 다르게 encode하는 흥미로운 패턴."

---

### ✅ 결과 해석 체크리스트

**모든 분석 공통:**
- [ ] p_fdr < 0.05? (통계적 유의성)
- [ ] Effect size 충분? (d ≥ 0.3 또는 η² ≥ 0.01)
- [ ] 평균과 표준편차가 합리적?
- [ ] 샘플 수(n)가 충분?

**Analysis 1 추가:**
- [ ] Cohen's d의 부호와 direction이 일치?
- [ ] 두 그룹의 표준편차가 너무 다르지 않은가?

**Analysis 2 추가:**
- [ ] 4개 그룹의 패턴이 해석 가능한가?
- [ ] 어느 그룹이 가장 높고/낮은가?

**Analysis 3 추가:**
- [ ] 주효과와 상호작용 중 어느 것이 더 큰가?
- [ ] ⚠️ Sparse feature 필터링 적용했는가?
- [ ] ⚠️ Top feature는 statsmodels로 재검증할 예정인가?

---

## 🎓 학습 완료!

### 축하합니다! 이제 다음을 할 수 있습니다:

✅ **통계 개념 이해:**
- p-value와 effect size의 차이
- t-test, ANOVA가 무엇을 하는지
- 상호작용의 의미

✅ **다중 비교 문제:**
- 왜 FDR 보정이 필요한지
- Bonferroni vs FDR 차이

✅ **결과 해석:**
- JSON 결과 파일 읽기
- 유의성과 효과 크기 판단
- 패턴 분석과 주의사항

---

### 📚 다음 단계 추천

**1단계: 실제 결과 파일 열어보기**
```bash
# 우리 실험의 실제 결과 파일
cat results/condition_comparison_summary_llama_*.json
```
→ 이 문서에서 배운 내용 직접 적용!

**2단계: 주요 문서 읽기**
- `ANALYSIS_ISSUES_REPORT.md`: 분석 주의사항
- `INTERACTION_ETA_PROBLEM_EXPLAINED.md`: Sparse 문제 상세

**3단계: 시각화 결과 보기**
```bash
python scripts/visualize_results_improved.py
```
→ 그래프로 패턴 확인

**4단계: 더 깊이 공부하고 싶다면**
- `STATISTICAL_METHODS_GUIDE.md`: 수식과 구현 상세
- Python으로 직접 간단한 t-test/ANOVA 해보기
- SciPy 공식 문서 튜토리얼

---

### ❓ 추가 질문이 있다면

**이 문서를 읽고도 이해가 안 되는 부분이 있다면:**
1. 어느 Part인지 확인
2. 구체적으로 어떤 개념이 어려운지 정리
3. 예시를 더 보고 싶은지, 다른 방식의 설명이 필요한지 생각

**언제든지 질문하세요!**

---

**Document Version:** 1.0
**Target Audience:** 통계 초심자 ~ 중급
**Reading Time:** 약 2시간
**Prerequisites:** 기초 수학 (평균, 백분율)
**Related:** `STATISTICAL_METHODS_GUIDE.md` (심화), `ANALYSIS_ISSUES_REPORT.md` (주의사항)
