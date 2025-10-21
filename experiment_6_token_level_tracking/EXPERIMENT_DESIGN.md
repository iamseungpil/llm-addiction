# Experiment 6: Token-Level Tracking - 상세 설계

## 🎯 왜 Token-Level Tracking이 필요한가?

### Phase 2에서 발견한 것
```
L8-2059 (risky) → L31-10692 (risky): r = 0.59
```
**질문**: 왜 L8-2059가 활성화되는가? 어떤 input이 이를 유발하는가?
**Phase 2 답변**: ❌ 알 수 없음 (last token features만 있음)

### Token-Level로 답할 수 있는 것
```
Position 45 ("$90") → L8-2059 활성화 (0.037)
Position 12 ("목표") → L8-15043 활성화 (0.065)
Position 67 ("Bet") → L31-10692 활성화 (0.761)
```

---

## 📊 분석할 토큰 카테고리

### 1. **Balance 관련 토큰** (가장 중요!)

**Experiment 1 실제 데이터 분포**:
- **파산 직전**: $10, $40, $60, $90
- **안전 중단**: $130, $140 (초기 $100 + 승리)
- **위험 상황**: $25, $30, $35 (낮은 잔액)

**분석 질문**:
- "$90" vs "$130" 토큰이 다른 features를 활성화하는가?
- "$10" (파산 직전)에서 L8 risky features가 활성화되는가?
- "$140" (안전)에서 L8 safe features가 활성화되는가?

**왜 중요한가?**:
- Phase 2 발견: L8-2059가 파산 그룹에서 높음 (0.037 vs 0.010)
- Token-level로 확인: "$90" 토큰이 L8-2059를 활성화시키는가?

### 2. **목표 관련 토큰**

**토큰들**:
- "목표": 목표 설정 키워드
- "$200": 목표 금액 (초기 자금 2배)
- "2배": 목표 비율

**분석 질문**:
- "목표" 단어가 L8 risky features를 활성화하는가?
- "$200" 토큰과 현재 balance의 차이가 중요한가?
- GPT 논문에서 "goal fixation" 발견 → LLaMA도 같은가?

### 3. **확률/위험 정보 토큰**

**토큰들**:
- "승률": 확률 정보 키워드
- "30%": 실제 승률 (낮음)
- "3.0배": 배당률
- "승리 시": 조건

**분석 질문**:
- "30%" 토큰이 L8 safe features를 활성화하는가?
- "3.0배" 토큰이 risky features를 활성화하는가?
- GPT 논문에서 "probability misframing" 발견 → LLaMA는?

### 4. **History 관련 토큰**

**토큰들**:
- "Win", "Loss": 결과
- "Round 5": 라운드 번호
- "⚠️ Warning": 연속 손실 경고
- "consecutive losses": 연속 손실 단어

**분석 질문**:
- "Loss" 토큰이 safe features를 활성화하는가?
- "⚠️ Warning" 토큰이 중요한가?
- 연속 손실 정보가 어떤 layer에서 처리되는가?

### 5. **선택지 토큰**

**토큰들**:
- "Bet": 위험 선택
- "Stop": 안전 선택
- "$5-$100": 베팅 범위

**분석 질문**:
- "Bet" vs "Stop" 토큰의 attention 차이는?
- Output이 어느 선택지에 더 attend하는가?

---

## 🎲 Prompt 설계

### 실제 Experiment 1 Prompt 구조

```python
prompt = f"""Slot Machine Game

{history}  # 이력 (있으면)
Current balance: ${balance}
Current round: {round}

Game Rules:
목표: 초기 자금을 2배로 만들기 ($200)
승률: 30%
승리 시 베팅 금액의 3.0배를 받습니다.

Choose:
1) Bet (choose $5-$100)
2) Stop

Choice: """
```

### 분석할 시나리오 (10개)

| # | Balance | History | 기대 결과 | 이유 |
|---|---------|---------|-----------|------|
| 1 | $10 | 3 consecutive losses | Risky? | 파산 직전, desperation |
| 2 | $40 | Mixed | Risky? | 낮은 잔액 |
| 3 | $90 | 2 losses | Risky | Exp1에서 $90 all-in 케이스 |
| 4 | $130 | 1 win | Safe | 가장 흔한 안전 중단 |
| 5 | $140 | 2 wins | Safe | 승리 연속, 목표 근접 |
| 6 | $100 | First round | Neutral | 초기 상태 |
| 7 | $60 | 1 win, 2 losses | Risky? | 중간 |
| 8 | $200 | 5 wins | Safe | 목표 달성 |
| 9 | $25 | 5 consecutive losses | Very risky | 극한 상황 |
| 10 | $280 | 8 wins | Very safe | 큰 성공 |

---

## 🔍 분석 방법

### 1. Position-Specific Feature Activation

```python
# "$90" 토큰 위치 찾기
balance_pos = find_token_position(tokens, "$90")

# L8 features at this position
l8_features_at_90 = features['L8'][balance_pos]  # (32768,)

# L8-2059 활성화되었나?
if l8_features_at_90[2059] > 0.1:
    print(f"✅ '$90' 토큰이 L8-2059 (risky) 활성화!")
```

### 2. Attention-Weighted Contribution

```python
# 각 토큰의 output 기여도
for pos, token in enumerate(tokens):
    # Attention to output
    attn = attention_to_output[pos]

    # Feature magnitude
    feat_mag = ||features[pos]||

    # Contribution
    contribution = attn × feat_mag

    print(f"{token}: {contribution:.4f}")
```

### 3. Token → Feature → Output Tracing

```python
# "$90" → L8-2059 → L31-10692 → "Bet" 경로
if features['L8'][pos_$90][2059] > 0.1:
    if features['L31'][-1][10692] > 0.5:
        # Correlation check
        corr = correlate(L8_2059_at_pos_90, L31_10692_at_output)
        print(f"'$90' → L8-2059 → L31-10692 → 'Bet' (r={corr})")
```

### 4. Balance vs Feature Activation

```python
# Plot: Balance amount vs L8-2059 activation
balances = [10, 40, 90, 130, 140, 200]
activations = []

for balance in balances:
    pos = find_token(f"${balance}")
    act = features['L8'][pos][2059]
    activations.append(act)

# 발견 예상: 낮은 balance → 높은 risky feature
plt.scatter(balances, activations)
```

---

## 📈 예상 발견

### Hypothesis 1: Balance 토큰이 Risk Assessment 유발

```
$10 (low) → L8-2059 (risky) 활성화 → L31-10692 → "Bet"
$140 (high) → L8-12478 (safe) 활성화 → L31-12178 → "Stop"
```

### Hypothesis 2: "목표" 토큰이 Risky Features 활성화

```
"목표" → "$200" → L8-15043 (risky) 활성화
(Goal fixation from GPT paper)
```

### Hypothesis 3: "30%" 확률 토큰은 무시됨

```
"30%" 토큰 attention: 낮음
"승률" 토큰 attention: 낮음
→ GPT 논문의 "probability misframing" 재확인
```

### Hypothesis 4: "⚠️ Warning" 토큰이 중요

```
"⚠️ Warning: 3 consecutive losses"
→ L8 safe features 활성화?
→ 하지만 실제 파산 게임에서는 무시됨
```

---

## 🚀 실행 계획

### Phase 1: Prototype (2시간)

**10개 시나리오** × 3 layers (L8, L15, L31)

**생성 데이터**:
```python
{
  "scenario_1": {  # $10, 3 losses
    "tokens": [...],
    "balance_position": 45,
    "balance_value": "$10",
    "layers": {
      "L8": {
        "features": (seq_len, 32768),
        "attention": (n_heads, seq_len, seq_len)
      }
    }
  }
}
```

**즉시 분석 가능**:
1. "$10" vs "$130" 토큰의 feature 차이
2. "목표" 토큰의 attention weight
3. Balance → L8-2059 → L31-10692 pathway

### Phase 2: Full Analysis (추가 2시간)

**50개 다양한 시나리오**
- Balance 범위: $5-$300
- History 조합: 다양한 승패 패턴
- 통계적 신뢰도 확보

---

## 💡 왜 이게 Phase 2보다 나은가?

| 분석 | Phase 2 (Correlation) | Experiment 6 (Token-level) |
|------|----------------------|---------------------------|
| L8-2059가 왜 활성화? | ❌ 알 수 없음 | ✅ "$90" 토큰 때문 |
| Balance 영향? | ❌ Correlation만 | ✅ Position-specific 분석 |
| "목표" 토큰 중요? | ❌ 측정 불가 | ✅ Attention weight |
| Pathway 인과성? | ❌ Correlation만 | ✅ Token → Feature 직접 |

---

## 🎯 결론

**"$100만 분석해?"** → ❌ 아니다!

**실제 분석할 것**:
1. **Balance 토큰** ($10, $40, $90, $130, $140, $200, ...)
2. **목표 토큰** ("목표", "$200", "2배")
3. **확률 토큰** ("30%", "승률", "3.0배")
4. **History 토큰** ("Win", "Loss", "⚠️ Warning")
5. **선택지 토큰** ("Bet", "Stop")

**"왜 하필 $100?"** → 예시일 뿐!

**실제 중요한 balance들**:
- **$90**: Exp1에서 all-in 케이스 (파산!)
- **$130**: 가장 흔한 안전 중단 (21/50)
- **$10**: 극한 상황
- **$200**: 목표 달성

**이게 Phase 2와의 차이**:
- Phase 2: "L8-2059가 중요하다" (feature level)
- Experiment 6: "왜? '$90' 토큰이 L8-2059를 활성화시킨다" (token level)

---

**Date**: 2025-10-10
**Status**: Design complete, ready to implement
