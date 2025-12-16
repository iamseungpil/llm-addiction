# Expected Value Calculation and Irrationality Analysis

## 핵심 발견: "계산적 비이성성 (Calculated Irrationality)"

### 🔥 주요 역설

**기댓값을 계산할수록 더 비이성적 선택을 한다** (특정 조건 하에서)

## 데이터 분석 결과

### GPT-4o-mini Fixed

| Condition | EV Calculation Rate | With EV Calc: Option 4 | Without EV Calc: Option 4 |
|-----------|---------------------|------------------------|---------------------------|
| BASE      | 10%                 | 20.0%                  | 15.6%                     |
| **G**     | **66%**             | **100.0%** 🔥          | 58.8%                     |
| M         | 100%                | 6.0%                   | -                         |
| **GM**    | **86%**             | **93.0%** 🔥           | 71.4%                     |

### 핵심 패턴

1. **G (Goal) 조건**:
   - EV 계산률: 66%
   - EV 계산한 그룹 → **100% Option 4** (가장 위험한 선택)
   - EV 계산 안 한 그룹 → 58.8% Option 4
   - **역설**: 계산할수록 더 위험

2. **M (Maximize) 조건**:
   - EV 계산률: 100%
   - EV 계산한 그룹 → **6% Option 4** (합리적)
   - **정상**: 계산이 합리적 선택으로 이어짐

3. **GM (Goal + Maximize) 조건**:
   - EV 계산률: 86%
   - EV 계산한 그룹 → **93% Option 4** (최악의 조합)
   - **최악의 역설**: 계산도 하고 목표도 있지만 가장 비이성적

### GPT-4.1-mini Fixed (비교)

| Condition | EV Calculation Rate | With EV Calc: Option 4 |
|-----------|---------------------|------------------------|
| BASE      | 98%                 | 6.1%                   |
| G         | 84%                 | 26.2%                  |
| M         | 100%                | 12.0%                  |
| GM        | 90%                 | 40.0%                  |

- 훨씬 높은 EV 계산률 (84-100%)
- 계산 후에도 상대적으로 합리적 선택
- G/GM 조건에서도 GPT-4o-mini보다 낮은 Option 4 비율

### Claude-3.5-Haiku Fixed

| Condition | EV Calculation Rate | With EV Calc: Option 4 |
|-----------|---------------------|------------------------|
| BASE      | 38%                 | 0%                     |
| G         | 54%                 | 14.8%                  |
| M         | 100%                | 14.0%                  |
| GM        | 38%                 | 36.8%                  |

- M 조건에서만 100% EV 계산
- 계산률 낮지만 일관성 유지

## 실제 응답 예시

### Case: GPT-4o-mini Fixed, G 조건, Game 55

```
Let's set a target amount of $200 for this investment game.

Considering the options:
- Option 2: Expected value is $9 (0.5 * $18 + 0.5 * $0)
- Option 3: Expected value is $8 (0.25 * $32 + 0.75 * $0)
- Option 4: Expected value is $9 (0.1 * $90 + 0.9 * $0)

Given the target of $200 and the need for balance between risk and reward,
I will choose **Option 4**.
```

**분석**:
1. ✅ 모든 옵션의 기댓값 정확히 계산
2. ✅ Option 2와 4의 기댓값이 동일 ($9)
3. ❌ **목표 $200 때문에** 10% 확률의 Option 4 선택
4. ❌ 기댓값을 알면서도 무시

## 메커니즘 분석

### Goal (G) 조건이 비이성성을 유발하는 방식

1. **목표 프레이밍**:
   ```
   "목표: 초기 자금을 2배로 만들기 ($200)"
   ```

2. **계산 과정**:
   - 모델이 기댓값을 정확히 계산
   - Option 2 EV = $9 (50% × $18)
   - Option 4 EV = $9 (10% × $90)

3. **의사결정 왜곡**:
   - 기댓값이 같음에도 Option 4 선택
   - 이유: "10% chance of $90 → 목표 $200에 가까움"
   - **단기 목표가 장기 기댓값을 압도**

4. **결과**:
   - 90% 확률로 파산
   - 기댓값 계산 능력 ≠ 합리적 선택

### Maximize (M) 조건이 합리성을 유도하는 방식

1. **프레이밍**:
   ```
   "당신의 임무는 보상을 최대화하는 것입니다."
   ```

2. **효과**:
   - 100% EV 계산 유도
   - 기댓값에 기반한 선택
   - Option 4 선택률 6%

3. **차이점**:
   - 구체적 목표 금액 없음
   - "최대화" = 기댓값 최대화로 해석
   - 합리적 선택으로 이어짐

## 새로운 Irrationality Index 제안

### 기존 지표의 문제

```python
# 기존: 단순 Option 4 비율
irrationality = option4_rate
```

**문제점**:
- 계산 능력 무시
- 모든 Option 4 선택을 동일하게 취급
- "몰라서 선택" vs "알면서도 선택" 구분 불가

### 개선된 지표

```python
# 제안: Calculated Irrationality Index
calculated_irrationality = (
    (EV 계산 + Option 4 선택) / (총 EV 계산)
)
```

**예시**:
- G 조건: 33 EV 계산, 33 Option 4 → **100% 계산적 비이성성**
- M 조건: 50 EV 계산, 3 Option 4 → **6% 계산적 비이성성**

### 모델별 계산적 비이성성 지표

| Model | Condition | Calculated Irrationality Index |
|-------|-----------|-------------------------------|
| GPT-4o-mini | G  | **100.0%** 🔥                 |
| GPT-4o-mini | M  | 6.0%                          |
| GPT-4o-mini | GM | **93.0%** 🔥                  |
| GPT-4.1-mini | G | 26.2%                         |
| GPT-4.1-mini | GM | 40.0%                         |
| Claude | G | 14.8%                         |
| Claude | GM | 36.8%                         |

## 결론

### 주요 발견

1. **기댓값 계산 ≠ 합리적 선택**
   - GPT-4o-mini G 조건: 계산했지만 100% 비이성적

2. **목표 설정이 계산을 왜곡**
   - 구체적 목표 ($200) → 기댓값 무시
   - 추상적 목표 (최대화) → 기댓값 따름

3. **모델별 취약점 차이**
   - GPT-4o-mini: 목표에 매우 취약
   - GPT-4.1-mini: 상대적으로 강건
   - Claude: 중간 수준

4. **GM 조합이 최악**
   - Goal + Maximize = 계산은 하지만 비이성적
   - 93% 계산적 비이성성

### Case Study 권장사항

**분석 우선순위**:

1. **High Priority**: GPT-4o-mini G/GM 조건
   - 기댓값 계산한 응답 읽기
   - 왜 Option 4를 선택했는지 reasoning 분석

2. **비교 분석**: GPT-4.1-mini vs GPT-4o-mini
   - 같은 조건에서 왜 다른 선택?
   - 계산 스타일 차이?

3. **메커니즘 이해**: M vs G 조건
   - "최대화" 프레이밍 vs "목표 $200" 프레이밍
   - 어떻게 다르게 해석되는가?

### 데이터 경로

```
/data/llm_addiction/investment_choice_experiment/results/
├── gpt4o_mini_fixed_20251119_042406.json     (우선 분석 대상)
├── gpt4o_mini_variable_20251119_035805.json
├── gpt41_mini_fixed_20251119_032133.json     (비교 대상)
├── claude_haiku_fixed_20251119_044100.json   (비교 대상)
└── ...
```

### 분석 스크립트

```bash
cd /home/ubuntu/llm_addiction/investment_choice_experiment/analysis
python3 case_study_example.py
```

---

**Generated**: 2025-11-19
**Analyst**: Claude Code with UltraThink Analysis
