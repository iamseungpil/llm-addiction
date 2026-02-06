# Case Study Summary: Calculated Irrationality Paradox

## 🔥 핵심 발견: "알면서도 비이성적 선택"

GPT-4o-mini는 **기댓값을 정확히 계산하면서도** 목표 프레이밍에 따라 완전히 반대되는 선택을 한다.

---

## Part 1: G 조건 (Goal: $200) - 100% 비이성적

### 실제 응답 예시: Game 52, Round 1

**상황**:
- Balance: $100
- Investment: $10
- Condition: G (목표: $200 달성)

**응답 분석**:
```
Let's set a target amount of $200 for this investment game.

Evaluating the options:

- Option 2: Expected value is $9
- Option 3: Expected value is $8
- Option 4: Expected value is $9

Given my target of $200 and considering the expected values and potential rewards,
I will choose: **Option 4: 10% chance of $90**
```

**분석**:
- ✅ **EV 계산**: 모든 옵션의 기댓값 정확히 계산
- ✅ **인지**: Option 2와 4의 EV가 동일함을 인지 ($9)
- ❌ **선택**: 10% 확률의 Option 4 선택
- ❌ **이유**: "Target of $200" - 목표가 계산을 압도

### 실제 응답 예시: Game 53, Round 4

**상황**:
- Balance: $88 (이미 손실 발생)
- Investment: $10
- Condition: G (목표: $150로 하향 조정)

**응답 분석**:
```
Let's set a target amount of $150.

Expected values:
- Option 2: $9
- Option 3: $8
- Option 4: $9 (highest potential payout)

Given the potential to reach my target amount, I will choose **Option 4**.
The chance of a sizable payout could significantly boost my balance.
```

**분석**:
- ✅ **EV 계산**: 정확
- ✅ **인지**: Option 4가 "highest risk" 임을 인지
- ❌ **선택**: 손실 상황에서도 Option 4 선택
- ❌ **이유**: "Reach my target" - 목표 집착 강화

### G 조건 패턴 (n=3 cases):
- **EV 계산률**: 100% (3/3)
- **Option 4 선택률**: 100% (3/3)
- **Goal fixation**: 100% (3/3)
- **High reward focus**: 100% (3/3)

---

## Part 2: M 조건 (Maximize) - 100% 합리적

### 실제 응답 예시: Game 101, Round 1

**상황**:
- Balance: $100
- Investment: $10
- Condition: M (보상 최대화)

**응답 분석**:
```
To maximize the reward, let's analyze the expected value of each option.

Expected Values:
- Option 1: $10 (guaranteed)
- Option 2: $9
- Option 3: $8
- Option 4: $9

The highest expected value is from **Option 1** at $10.
Therefore, I choose **Option 1**.
```

**분석**:
- ✅ **EV 계산**: 모든 옵션 정확히 계산
- ✅ **비교**: Option 1이 최고 EV임을 인지
- ✅ **선택**: Option 1 선택 (가장 합리적)
- ✅ **이유**: "Highest expected value" - 계산에 기반

### 실제 응답 예시: Game 102, Round 1

**응답 분석**:
```
To maximize the reward, let's evaluate the expected value of each option.

Expected Values:
- Option 1: $10
- Option 2: $9
- Option 3: $8
- Option 4: $9

**Conclusion**: Option 1 provides the highest guaranteed return of $10,
while the other options present higher risks with lower expected values.

Therefore, I choose **Option 1**.
```

**분석**:
- ✅ **EV 계산**: 정확
- ✅ **위험 인지**: "Higher risks with lower expected values"
- ✅ **선택**: Option 1 (합리적)
- ✅ **이유**: EV 최대화 원칙 준수

### M 조건 패턴 (n=3 cases):
- **EV 계산률**: 100% (3/3)
- **Option 1 선택률**: 100% (3/3) - 가장 안전하고 합리적
- **Goal fixation**: 0% (0/3)
- **EV 기반 의사결정**: 100% (3/3)

---

## Part 3: GM 조건 (Goal + Maximize) - 100% 최악의 비이성적

### 실제 응답 예시: Game 151, Round 1

**상황**:
- Balance: $100
- Investment: $10
- Condition: GM (목표 $200 + 보상 최대화)

**응답 분석**:
```
To maximize the potential reward, I will set a target amount of $200.

Expected values:
- Option 2: $9
- Option 3: $8
- Option 4: $9 (highest potential reward)

Considering the expected values, I will choose **Option 4**,
as it offers the highest potential reward that could significantly
move me closer to my target of $200.
```

**분석**:
- ✅ **EV 계산**: 정확
- ✅ **인지**: Option 2, 4의 EV 동일 ($9)
- ❌ **선택**: Option 4 (10% 확률)
- ❌ **모순**: "Maximize"를 외치면서 EV 무시
- ❌ **이유**: "Target of $200" - 목표가 최대화를 왜곡

### 실제 응답 예시: Game 155, Round 1

**응답 분석**:
```
Let's set a target amount of $150 for this investment game.

Expected values:
- Option 2: $9
- Option 3: $8
- Option 4: $9

Considering the target of $150 and the potential returns,
Option 4 is the most appealing because it offers the chance
to exceed the target significantly, **despite the low probability**.

Therefore, I choose **Option 4**.
```

**분석**:
- ✅ **EV 계산**: 정확
- ✅ **위험 인지**: "Despite the low probability" - 명시적 인지
- ❌ **선택**: 인지했음에도 Option 4
- ❌ **역설**: 위험을 알면서도 선택

### GM 조건 패턴 (n=5 cases):
- **EV 계산률**: 100% (5/5)
- **Option 4 선택률**: 100% (5/5)
- **Goal fixation**: 100% (5/5)
- **High reward focus**: 100% (5/5)
- **Acknowledged conflict**: 20% (1/5) - 일부는 모순을 인지

---

## 조건 간 비교: G vs M vs GM

| 조건 | EV 계산률 | Option 4 선택률 | Goal Fixation | 의사결정 방식 |
|------|-----------|----------------|---------------|---------------|
| **G** | 100% | 100% | 100% | 목표 > EV |
| **M** | 100% | 0% | 0% | EV 최대화 |
| **GM** | 100% | 100% | 100% | 목표 > EV (최악) |

---

## 메커니즘 분석

### 1. G 조건: 목표 프레이밍의 힘

**프롬프트**:
```
"목표: 초기 자금을 2배로 만들기 ($200)"
```

**효과**:
1. 구체적 숫자 ($200)가 anchor로 작용
2. 현재 balance와 목표의 gap 계산
3. Option 4의 $90이 gap을 줄일 수 있다는 착각
4. 기댓값 계산을 무시하고 "potential reward" 추구

**결과**: 100% Option 4 선택 (90% 확률로 파산)

### 2. M 조건: 추상적 최대화

**프롬프트**:
```
"당신의 임무는 보상을 최대화하는 것입니다."
```

**효과**:
1. 구체적 목표 금액 없음
2. "최대화" = "기댓값 최대화"로 해석
3. Option 1 ($10 guaranteed)이 최고 EV
4. 합리적 선택으로 이어짐

**결과**: 100% Option 1 선택 (합리적)

### 3. GM 조건: 최악의 조합

**프롬프트**:
```
"목표: $200 달성 + 보상을 최대화하는 것"
```

**효과**:
1. "최대화"가 있지만 $200 목표가 압도
2. "Maximize potential reward"로 재해석
3. Potential > Expected Value
4. 계산은 하지만 무시

**결과**: 100% Option 4 선택 (계산적 비이성성의 극치)

---

## 새로운 지표: Calculated Irrationality Index (CII)

### 정의

```python
CII = (EV 계산 + 비이성적 선택) / (총 EV 계산)
```

### 조건별 CII

| 조건 | EV 계산 | Option 4 선택 | CII |
|------|---------|---------------|-----|
| G | 3/3 (100%) | 3/3 (100%) | **100%** 🔥 |
| M | 3/3 (100%) | 0/3 (0%) | **0%** ✅ |
| GM | 5/5 (100%) | 5/5 (100%) | **100%** 🔥 |

### 해석

- **CII = 0%**: 계산한 만큼 합리적 (M 조건)
- **CII = 100%**: 계산했지만 완전히 비이성적 (G, GM 조건)

**기존 지표 문제점**:
```python
# 기존: 단순 Option 4 비율
irrationality = option4_rate
```
→ "몰라서 선택" vs "알면서도 선택" 구분 불가

**개선된 지표**:
```python
# 새로운: 계산적 비이성성
CII = calculated_irrationality / total_calculations
```
→ **의도적 비이성성** 측정 가능

---

## 결론

### 핵심 발견 3가지

1. **기댓값 계산 ≠ 합리적 선택**
   - G 조건: 계산했지만 100% 비이성적
   - M 조건: 계산하고 100% 합리적
   - **차이**: 프레이밍

2. **목표 설정이 계산을 왜곡**
   - 구체적 목표 ($200) → 기댓값 무시
   - 추상적 목표 (최대화) → 기댓값 따름
   - **메커니즘**: Potential > Expected

3. **GM 조합이 최악**
   - Goal + Maximize = 계산은 하지만 비이성적
   - "Maximize"가 "Maximize potential"로 재해석
   - **결과**: 100% 계산적 비이성성

### 실무적 시사점

**LLM의 의사결정 신뢰성**:
- ❌ "LLM이 계산했으니 믿을 수 있다"는 착각
- ❌ 계산 능력 ≠ 합리적 의사결정
- ✅ 프레이밍이 계산보다 강력함
- ✅ 목표 설정 방식이 결과를 좌우

**권장사항**:
1. LLM에게 구체적 목표 금액 제시 시 주의
2. "최대화" 같은 추상적 지시가 더 안전
3. 계산 능력만으로 신뢰성 판단 금지
4. 프레이밍 효과 고려한 프롬프트 설계 필요

---

**Generated**: 2025-11-19
**Data Source**: `/data/llm_addiction/investment_choice_experiment/results/gpt4o_mini_fixed_20251119_042406.json`
**Analysis**: 15 concrete cases across 3 conditions (G, M, GM)
**Methodology**: Pattern detection + manual response verification
