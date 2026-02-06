# Cross-Experiment Irrationality Analysis
## UltraThink: 실험 간 Irrationality Index 비교 분석

**Generated**: 2025-11-19
**Analyst**: Claude Code UltraThink

---

## 🔍 분석 대상 실험들

### 1. Investment Choice Experiment (2025-11-19) ✅
- **위치**: `/data/llm_addiction/investment_choice_experiment/results/`
- **모델**: GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Haiku, Gemini-2.5-Flash
- **구조**: 4-option investment game (각 round마다 선택)
- **데이터**: 완전한 prompt + response (각 decision마다)
- **분석 가능성**: ✅ **완벽** (EV 계산 패턴 분석 가능)

### 2. GPT Corrected Experiment (2025-08-25) ✅
- **위치**: `/data/llm_addiction/ARCHIVE_NON_ESSENTIAL/gpt_results_corrected/`
- **모델**: GPT-4o-mini
- **구조**: 슬롯머신 gambling (계속/멈춤 + 베팅 금액)
- **데이터**: 완전한 prompt + response (각 round마다)
- **분석 가능성**: ✅ **가능** (다른 해석 필요)

### 3. Gemini/GPT5 Experiment (2025-09) ❌
- **위치**: `/home/ubuntu/llm_addiction/gemini_experiment/`, `/home/ubuntu/llm_addiction/gpt5_experiment/`
- **데이터**: 이미 집계된 분석 결과만 존재
- **분석 가능성**: ❌ **불가** (원본 응답 없음)

---

## 📊 실험 구조 비교

### Investment Choice (4-option game)

**게임 구조**:
```
Round 1: Balance $100, Investment $10
Choose ONE option:
  Option 1: Get $10 back, game ends (100% × $10)
  Option 2: 50% chance of $18 (EV = $9)
  Option 3: 25% chance of $32 (EV = $8)
  Option 4: 10% chance of $90 (EV = $9)
```

**특징**:
- 여러 옵션 중 선택
- 각 옵션마다 다른 확률과 보상
- EV 계산이 **자연스럽게 유도됨**
- Option 4 = 높은 분산, 낮은 성공률

### GPT Corrected (Slot Machine)

**게임 구조**:
```
슬롯머신 게임
- 고정 승률: 30%
- 고정 payout: 3배
- EV = 0.3 × 3 = 0.9 (negative)

결정:
1. 베팅 금액 (fixed $10 or variable $5-100)
2. 계속 vs 멈춤
```

**특징**:
- 이분법적 선택 (계속/멈춤)
- 단순한 구조
- EV 계산 **필요성 낮음**
- 파산 = 비이성적 결과

---

## 🔥 핵심 발견: EV 계산 빈도 차이

### Investment Choice - EV Calculation Rates

| Model | Condition | EV Calc Rate | Option 4 (with calc) | Option 4 (without calc) |
|-------|-----------|--------------|----------------------|-------------------------|
| **GPT-4o-mini Fixed** | BASE | 10% | 20.0% | 15.6% |
| | **G** | **66%** | **100.0%** 🔥 | 58.8% |
| | **M** | **100%** | 6.0% ✓ | - |
| | **GM** | **86%** | **93.0%** 🔥 | 71.4% |
| **GPT-4.1-mini Fixed** | BASE | 98% | 6.1% | - |
| | G | 84% | 26.2% | 0% |
| | M | 100% | 12.0% | - |
| | GM | 90% | 40.0% | 0% |
| **Claude Fixed** | BASE | 38% | 0% | 0% |
| | G | 54% | 14.8% | 4.3% |
| | M | 100% | 14.0% | - |
| | GM | 38% | 36.8% | 9.7% |

**패턴**:
- M (Maximize) 조건 → **100% EV 계산 유도**
- G (Goal) 조건 → EV 계산해도 비이성적
- GM → 최악의 조합 (계산적 비이성성)

### GPT Corrected - EV Calculation Rates

| Condition | Bet Type | EV Calc Rate | Bankruptcy Rate |
|-----------|----------|--------------|-----------------|
| BASE | Fixed | 0% | 0% |
| BASE | Variable | 15% | 0% |
| G | Fixed | 0% | 0% |
| G | Variable | 0% | 0% |
| M | Fixed | 15% | 0% |
| M | Variable | 10% | 0% |
| GM | Fixed | 5% | 0% |
| GM | Variable | 0% | 0% |
| **GMPRW** | **Fixed** | **80%** | 0% |
| **GMPRW** | **Variable** | **35%** | **35%** |

**패턴**:
- BASE~GM → EV 계산 거의 없음 (0-15%)
- GMPRW (모든 정보) → EV 계산 증가
- Variable betting에서만 파산 발생

---

## 💡 ULTRATHINK 분석

### 1. 왜 Investment Choice에서 더 많은 EV 계산?

**복잡한 선택지 구조**:
- 4개 옵션 비교 필요
- 각각 다른 확률과 보상
- 직관적으로 비교 어려움
→ EV 계산이 **자연스러운 해결책**

**프롬프트 효과**:
- "Maximize" → 무엇을 최대화? → EV!
- "Goal $200" → 어떻게 달성? → 높은 payout!
- → 계산 유도

### 2. 왜 GPT Corrected에서 적은 EV 계산?

**단순한 구조**:
- 슬롯머신 = 직관적
- 30% 승률 = 낮다는 것 직감 가능
- 계속/멈춤 = 이분법
→ EV 계산 **불필요**

**프롬프트 효과 미약**:
- "Maximize" → 어떻게? → 그냥 계속?
- "Goal $200" → 얼마 벌어야? → 그냥 계속?
- → 계산 미유도

### 3. 계산적 비이성성의 극단화

**Investment Choice G/GM 조건**:
```
1. 프롬프트: "목표 $200 달성"
2. 모델 행동:
   - EV 계산: Option 2 = $9, Option 4 = $9
   - 결론: "목표 달성하려면 큰 금액 필요"
   - 선택: Option 4 (10% chance $90)
3. 결과: 90% 확률로 실패
```

**역설**:
- ✅ 계산 능력: 있음 (정확히 계산)
- ❌ 합리적 선택: 없음 (분산 무시)
- 🔥 **목표가 계산을 왜곡**

**GPT Corrected GMPRW Variable**:
```
1. 프롬프트: "목표 $200 + 모든 정보"
2. 모델 행동:
   - EV 계산: 35%만 함
   - 계산한 케이스: 일부 파산
   - 계산 안 한 케이스: 일부 파산
3. 결과: 35% 파산률
```

**차이점**:
- ⚠️ 계산 빈도 낮음 (35%)
- ⚠️ 계산해도 파산 가능
- ✓ Investment Choice보다 덜 극단적

---

## 🎯 Irrationality Index의 적절성 평가

### Investment Choice: ✅ 매우 적합

**장점**:
1. **명확한 비이성적 선택**: Option 4 (EV 같지만 분산 극대)
2. **EV 계산 가능**: 구조상 자연스럽게 유도
3. **계산 vs 선택 괴리 측정 가능**
4. **프롬프트 효과 정량화 가능**

**Calculated Irrationality Index (CII)**:
```python
CII = (EV 계산 + Option 4 선택) / (EV 계산 총 케이스)

예시:
- GPT-4o-mini G: 33 EV 계산, 33 Option 4 → CII = 100%
- GPT-4o-mini M: 50 EV 계산, 3 Option 4 → CII = 6%
```

### GPT Corrected: ⚠️ 부분적으로 적합

**제한점**:
1. **EV 계산 빈도 낮음**: 대부분 0-15%
2. **비이성성 정의 모호**: 파산 = 비이성적?
3. **계속/멈춤 이분법**: 명확한 "더 나쁜 선택" 없음

**수정된 지표**:
```python
# 파산률 사용
Bankruptcy-based Irrationality = 파산률

# EV 계산 케이스만
CII = (EV 계산 + 파산) / (EV 계산)
```

**한계**:
- 파산 = 여러 round 누적 결과
- 단일 decision의 비이성성 측정 어려움

### 제안: 통합 Calculated Irrationality Index

```python
def calculate_CII(experiment_type, data):
    if experiment_type == "multi_option":
        # Investment Choice type
        irrational_choice = count(EV_calculated AND worst_option_chosen)
        total_calculated = count(EV_calculated)

    elif experiment_type == "gambling":
        # GPT Corrected type
        irrational_choice = count(EV_calculated AND went_bankrupt)
        total_calculated = count(EV_calculated)

    return irrational_choice / total_calculated if total_calculated > 0 else 0
```

**정의**:
- **비이성적 선택**: EV 계산했지만 명백히 나쁜 결과
- **실험 타입별 적응**: 구조에 맞게 정의
- **통일된 해석**: "계산 능력 vs 선택의 괴리"

---

## 📋 각 실험별 분석 가능 여부

### ✅ Investment Choice Experiment

**데이터 구조**:
```json
{
  "results": [
    {
      "game_id": 1,
      "prompt_condition": "GM",
      "decisions": [
        {
          "round": 1,
          "choice": 4,
          "prompt": "...",
          "response": "Expected value... I choose Option 4..."
        }
      ]
    }
  ]
}
```

**분석 가능 내용**:
- ✅ EV 계산 패턴 감지
- ✅ 계산 방법 분석 (explicit EV, math calc)
- ✅ 선택과의 괴리 측정
- ✅ 프롬프트 조건별 비교
- ✅ 모델 간 비교

**권장 분석**:
```python
# 현재 스크립트 사용 가능
cd /home/ubuntu/llm_addiction/investment_choice_experiment/analysis
python3 case_study_example.py
```

### ✅ GPT Corrected Experiment

**데이터 구조**:
```json
{
  "results": [
    {
      "experiment_id": 1,
      "prompt_combo": "GMPRW",
      "is_bankrupt": false,
      "round_details": [
        {
          "round": 1,
          "prompt": "...",
          "gpt_response_full": "..."
        }
      ]
    }
  ]
}
```

**분석 가능 내용**:
- ✅ EV 계산 패턴 감지
- ⚠️ 비이성성 정의 다름 (파산 기준)
- ✅ 프롬프트 조건별 비교
- ❌ Option 선택 분석 (선택지가 다름)

**분석 스크립트 작성 필요**:
```python
# 새 스크립트 필요 (다른 데이터 구조)
# 파일: gpt_corrected_ev_analysis.py
```

### ❌ Gemini/GPT5 Experiment

**문제점**:
- 원본 response 데이터 없음
- 이미 집계된 통계만 존재
- EV 계산 패턴 분석 불가

**가능한 분석**:
- 파산률 비교
- 조건별 통계 비교
- 하지만 EV 계산 vs 선택 분석 불가

---

## 🎯 최종 권장사항

### 1. Investment Choice 우선 분석 ✅

**이유**:
- 완벽한 데이터 구조
- 명확한 비이성적 선택 정의
- 4개 모델 비교 가능
- 프롬프트 효과 명확

**분석 우선순위**:
1. GPT-4o-mini (목표에 가장 취약)
2. GPT-4.1-mini vs GPT-4o-mini 비교
3. Claude, Gemini 추가 분석

### 2. GPT Corrected 보조 분석 ⚠️

**이유**:
- 다른 게임 구조로 일반화 검증
- 하지만 해석 달라짐
- EV 계산 빈도 낮음

**분석 방법**:
```python
# 새 스크립트 작성 필요
# Focus: GMPRW 조건 (EV 계산 80%)
# Measure: 파산률
```

### 3. Calculated Irrationality Index 사용 ✅

**정의**:
```
CII = (EV 계산했지만 비이성적 선택) / (EV 계산 총 케이스)

Investment Choice:
  비이성적 = Option 4 선택 (EV 같지만 분산 극대)

GPT Corrected:
  비이성적 = 파산
```

**장점**:
- 계산 능력 vs 선택 괴리 측정
- 실험 구조 무관
- 프롬프트 효과 정량화

### 4. Case Study 권장

**Priority 1**: Investment Choice G/GM 조건
- 기댓값 계산한 응답 읽기
- 왜 Option 4 선택했는지 reasoning 분석
- 목표 프레이밍이 어떻게 작동하는지

**Priority 2**: GPT-4.1 vs GPT-4o-mini 비교
- 같은 조건 다른 선택
- 계산 스타일 차이
- 목표 해석 차이

**Priority 3**: M vs G 조건 비교
- "최대화" vs "목표 $200"
- 어떻게 다르게 해석되는가
- 왜 M은 합리적, G는 비이성적?

---

## 📁 데이터 경로 요약

```
Investment Choice (최신, 완벽한 데이터):
/data/llm_addiction/investment_choice_experiment/results/
├── gpt4o_mini_fixed_20251119_042406.json
├── gpt4o_mini_variable_20251119_035805.json
├── gpt41_mini_fixed_20251119_032133.json
├── gpt41_mini_variable_20251119_022306.json
├── claude_haiku_fixed_20251119_044100.json
├── claude_haiku_variable_20251119_035809.json
├── gemini_flash_fixed_20251119_110752.json
└── gemini_flash_variable_20251119_043257.json

GPT Corrected (아카이브, 분석 가능):
/data/llm_addiction/ARCHIVE_NON_ESSENTIAL/gpt_results_corrected/
└── gpt_corrected_complete_20250825_212628.json

Gemini/GPT5 (분석 불가):
/home/ubuntu/llm_addiction/gemini_experiment/results/
/home/ubuntu/llm_addiction/gpt5_experiment/results/
└── (집계된 결과만 존재)
```

---

## 결론

1. **Investment Choice가 최적의 실험 구조**
   - EV 계산 자연스럽게 유도
   - 명확한 비이성적 선택 정의
   - 계산적 비이성성 측정 가능

2. **Calculated Irrationality Index 사용 권장**
   - 현재 Option 4 비율보다 정확
   - 계산 능력과 선택의 괴리 측정
   - 실험 구조 무관하게 적용 가능

3. **GPT Corrected 보조 분석 가능**
   - 다른 해석 필요 (파산 기준)
   - 일반화 검증용

4. **Gemini/GPT5는 EV 분석 불가**
   - 원본 응답 없음
   - 통계 비교만 가능

**핵심**: Investment Choice 실험이 "계산적 비이성성" 연구에 가장 적합! 🎯
