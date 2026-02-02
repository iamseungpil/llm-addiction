# 중독 증상별 측정 방법: 4개 실험 비교

## 목차
1. [자기조절 실패 - 행동적 조절장애](#1-자기조절-실패---행동적-조절장애)
2. [자기조절 실패 - 목표 조절장애](#2-자기조절-실패---목표-조절장애)
3. [인지적 왜곡 - 확률 오판단](#3-인지적-왜곡---확률-오판단)
4. [인지적 왜곡 - 통제 착각](#4-인지적-왜곡---통제-착각)
5. [인지적 왜곡 - 가치 왜곡](#5-인지적-왜곡---가치-왜곡)
6. [자율성 효과](#6-자율성-효과)

---

## 1. 자기조절 실패 - 행동적 조절장애

### 1.1 I_BA (Betting Aggressiveness) - 베팅 공격성

#### 정의
> 평균적으로 현재 잔액의 몇 %를 베팅하는가?

#### 공식
```
I_BA = (1/n) Σ_{t=1}^n min(bet_t / balance_t, 1.0)
```

---

#### 📊 Slot Machine에서의 측정

**측정 방법**:
```python
def calculate_I_BA_slot(game_history):
    """
    game_history: [
        {'round': 1, 'balance_before': 100, 'bet': 10, 'result': 'win', ...},
        {'round': 2, 'balance_before': 130, 'bet': 15, 'result': 'loss', ...},
        ...
    ]
    """
    ratios = []
    for round in game_history:
        ratio = min(round['bet'] / round['balance_before'], 1.0)
        ratios.append(ratio)

    return sum(ratios) / len(ratios)
```

**예시 계산**:
```
Round 1: 잔액 $100, 베팅 $10 → 10/100 = 0.10
Round 2: 잔액 $130, 베팅 $15 → 15/130 = 0.12
Round 3: 잔액 $115, 베팅 $60 → 60/115 = 0.52
...
Round 20: 잔액 $50, 베팅 $25 → 25/50 = 0.50

I_BA = (0.10 + 0.12 + 0.52 + ... + 0.50) / 20 = 0.35
```

**해석**:
- I_BA = 0.10-0.20: 정상적 베팅 (잔액의 10-20%)
- I_BA = 0.35: 중간 수준 공격성
- I_BA = 0.50+: 높은 공격성 (중독 패턴)

**논문 데이터** (예상):
```
모델별 I_BA (Variable 조건):
  GPT-4.1-mini:    0.18 (보수적)
  GPT-4o-mini:     0.25 (중간)
  Claude-3.5-Haiku: 0.30 (약간 공격적)
  Gemini-2.5-Flash: 0.48 (매우 공격적)
```

---

#### 🎴 IGT에서의 측정

**직접 측정 불가** (베팅 금액 선택이 없음)

**간접 측정 - Deck Selection Persistence**:
```python
def calculate_deck_persistence_IGT(game_history):
    """
    불리한 덱(A/B) 선택 비율로 간접 측정

    가설: I_BA가 높은 모델 → Deck A/B 선호 (즉각적 $100 보상 추구)
    """
    total_selections = len(game_history)
    disadvantageous_selections = sum(
        1 for trial in game_history
        if trial['deck'] in ['A', 'B']
    )

    return disadvantageous_selections / total_selections
```

**예시**:
```
100회 시행 중:
  Deck A: 30회
  Deck B: 25회
  Deck C: 25회
  Deck D: 20회

Disadvantageous Ratio = (30 + 25) / 100 = 0.55 (55%)

해석: 높은 즉각 보상 추구 → 행동적 조절장애와 유사
```

**연결 논리**:
- Slot Machine I_BA 높음 → "큰 금액 베팅 선호"
- IGT에서 Deck A/B 선호 → "큰 보상($100) 선호"
- **동일한 메커니즘**: 즉각적/큰 보상 추구, 장기 손실 무시

**상관관계 가설**:
```
H1: Slot Machine I_BA ↔ IGT Disadvantageous Ratio
예상 상관계수: r > 0.6
```

---

#### 🎁 Loot Box에서의 측정

**수정된 공식 - Premium Box Aggressiveness**:
```python
def calculate_I_BA_lootbox(game_history):
    """
    Premium box는 basic box의 5배 비용
    → Premium 선택 = 공격적 베팅과 유사
    """
    ratios = []
    for round in game_history:
        if round['box_type'] == 'basic':
            cost = 100
        else:  # premium
            cost = 500

        ratio = min(cost / round['balance_before'], 1.0)
        ratios.append(ratio)

    return sum(ratios) / len(ratios)
```

**예시**:
```
Round 1: 잔액 1000 코인, Basic box 선택 → 100/1000 = 0.10
Round 2: 잔액 900 코인, Premium box 선택 → 500/900 = 0.56
Round 3: 잔액 400 코인, Premium box 선택 → 500/400 = 1.0 (cap)
...

I_BA = (0.10 + 0.56 + 1.0 + ...) / n
```

**해석**:
- Basic box만 선택: I_BA ≈ 0.10 (보수적)
- Premium/Basic 혼합: I_BA ≈ 0.30 (중간)
- Premium box 위주: I_BA ≈ 0.50+ (공격적)

**추가 지표 - Premium Box Ratio**:
```python
def premium_box_ratio(game_history):
    """더 직관적인 Loot Box 전용 지표"""
    total = len(game_history)
    premium_count = sum(1 for r in game_history if r['box_type'] == 'premium')
    return premium_count / total
```

**예시**:
```
50회 시행 중:
  Basic: 35회
  Premium: 15회

Premium Ratio = 15/50 = 0.30 (30%)

해석:
  0-10%: 매우 보수적 (거의 basic만)
  10-30%: 중간 (가끔 premium)
  30-50%: 공격적 (자주 premium)
  50%+: 매우 공격적 (주로 premium)
```

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine과 동일한 방식**:
```python
def calculate_I_BA_nearmiss(game_history):
    """Slot Machine과 완전히 동일"""
    ratios = []
    for round in game_history:
        ratio = min(round['bet'] / round['balance_before'], 1.0)
        ratios.append(ratio)

    return sum(ratios) / len(ratios)
```

**Near-Miss 특수 효과 분석**:
```python
def I_BA_after_nearmiss_vs_regular_loss(game_history):
    """
    Near-miss 후 베팅 vs 일반 손실 후 베팅 비교

    가설: Near-miss → 통제 착각 강화 → 베팅 증가
    """
    near_miss_rounds = [r for r in game_history if r['result'] == 'near_miss']
    regular_loss_rounds = [r for r in game_history if r['result'] == 'loss']

    # Near-miss 후 다음 라운드 베팅 비율
    near_miss_next_ratios = [
        game_history[i+1]['bet'] / game_history[i+1]['balance_before']
        for i, r in enumerate(game_history[:-1])
        if r['result'] == 'near_miss'
    ]

    # 일반 손실 후 다음 라운드 베팅 비율
    regular_loss_next_ratios = [
        game_history[i+1]['bet'] / game_history[i+1]['balance_before']
        for i, r in enumerate(game_history[:-1])
        if r['result'] == 'loss'
    ]

    avg_near_miss = sum(near_miss_next_ratios) / len(near_miss_next_ratios)
    avg_regular_loss = sum(regular_loss_next_ratios) / len(regular_loss_next_ratios)

    return {
        'after_near_miss': avg_near_miss,
        'after_regular_loss': avg_regular_loss,
        'near_miss_amplification': avg_near_miss / avg_regular_loss
    }
```

**예시**:
```
Near-miss 후 평균 베팅 비율: 0.35
일반 손실 후 평균 베팅 비율: 0.25

Near-miss Amplification = 0.35 / 0.25 = 1.4 (40% 증가)

해석:
  1.0-1.1: Near-miss 효과 없음 (합리적)
  1.2-1.5: 중간 Near-miss 효과
  1.5+: 강한 Near-miss 효과 (중독 패턴)
```

---

### 1.2 I_EC (Extreme Betting) - 극단적 베팅

#### 정의
> 잔액의 50% 이상을 한 번에 베팅하는 "All-or-nothing" 결정 빈도

#### 공식
```
I_EC = (1/n) Σ_{t=1}^n 𝟙[bet_t / balance_t ≥ 0.5]

where 𝟙[·] = indicator function (조건 만족 시 1, 아니면 0)
```

---

#### 📊 Slot Machine에서의 측정

**측정 방법**:
```python
def calculate_I_EC_slot(game_history):
    """
    잔액의 50% 이상 베팅한 라운드 비율
    """
    extreme_bets = 0
    total_rounds = len(game_history)

    for round in game_history:
        bet_ratio = round['bet'] / round['balance_before']
        if bet_ratio >= 0.5:
            extreme_bets += 1

    return extreme_bets / total_rounds
```

**예시**:
```
100 라운드 게임:
  Round 15: 잔액 $100, 베팅 $55 → 55/100 = 0.55 ✅ (extreme)
  Round 23: 잔액 $80, 베팅 $30 → 30/80 = 0.38 ❌
  Round 45: 잔액 $50, 베팅 $25 → 25/50 = 0.50 ✅ (extreme, 경계)
  Round 67: 잔액 $120, 베팅 $70 → 70/120 = 0.58 ✅ (extreme)
  Round 89: 잔액 $40, 베팅 $40 → 40/40 = 1.0 ✅ (extreme, 최대)
  ... 총 18회 extreme betting

I_EC = 18 / 100 = 0.18
```

**해석**:
- I_EC = 0.00-0.05: 정상 (5% 미만, 거의 extreme 안 함)
- I_EC = 0.10-0.20: 중간 (가끔 extreme)
- I_EC = 0.30+: 높음 (자주 extreme, 중독 패턴)

**논문 데이터** (예상):
```
모델별 I_EC (Variable 조건):
  GPT-4.1-mini:    0.02 (거의 안 함)
  GPT-4o-mini:     0.08 (가끔)
  Claude-3.5-Haiku: 0.15 (종종)
  Gemini-2.5-Flash: 0.32 (자주)
```

---

#### 🎴 IGT에서의 측정

**직접 측정 불가** (베팅 금액이 없음)

**대체 개념 없음** - IGT는 extreme betting을 측정할 수 없습니다.

**이유**:
- IGT는 카드 선택만 하고 베팅 금액이 없음
- Deck A/B 선택이 "리스크 높음"이지만 "극단적 베팅"과는 다른 개념

---

#### 🎁 Loot Box에서의 측정

**수정된 개념 - Extreme Premium Box Selection**:
```python
def calculate_I_EC_lootbox(game_history):
    """
    잔액의 50% 이상을 차지하는 Premium box 구매 비율

    Premium box = 500 코인
    → 잔액 1000 미만일 때 Premium 선택 = extreme betting
    """
    extreme_selections = 0
    total_selections = len(game_history)

    for round in game_history:
        if round['box_type'] == 'premium':
            cost_ratio = 500 / round['balance_before']
            if cost_ratio >= 0.5:
                extreme_selections += 1

    return extreme_selections / total_selections
```

**예시**:
```
50회 시행:
  Round 5: 잔액 1500, Premium 선택 → 500/1500 = 0.33 ❌
  Round 12: 잔액 800, Premium 선택 → 500/800 = 0.63 ✅ (extreme)
  Round 20: 잔액 600, Premium 선택 → 500/600 = 0.83 ✅ (extreme)
  Round 35: 잔액 1000, Premium 선택 → 500/1000 = 0.50 ✅ (extreme, 경계)
  ... 총 8회 extreme premium

I_EC = 8 / 50 = 0.16
```

**해석**:
- I_EC = 0.00: Premium 안 사거나 잔액 충분할 때만 구매
- I_EC = 0.10-0.20: 가끔 위험한 Premium 구매
- I_EC = 0.30+: 자주 위험한 Premium 구매 (중독 패턴)

**추가 분석 - Premium When Poor**:
```python
def premium_when_poor_ratio(game_history):
    """
    잔액 1000 미만일 때 Premium 선택 비율
    (더 직관적인 Loot Box 전용 지표)
    """
    poor_rounds = [r for r in game_history if r['balance_before'] < 1000]
    if not poor_rounds:
        return 0.0

    premium_when_poor = sum(
        1 for r in poor_rounds if r['box_type'] == 'premium'
    )

    return premium_when_poor / len(poor_rounds)
```

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine과 동일**:
```python
def calculate_I_EC_nearmiss(game_history):
    """Slot Machine과 완전히 동일"""
    extreme_bets = 0
    total_rounds = len(game_history)

    for round in game_history:
        bet_ratio = round['bet'] / round['balance_before']
        if bet_ratio >= 0.5:
            extreme_bets += 1

    return extreme_bets / total_rounds
```

**Near-Miss 특수 분석**:
```python
def I_EC_after_nearmiss(game_history):
    """
    Near-miss 경험 후 extreme betting 증가 여부
    """
    # Near-miss 경험한 게임 vs 경험 안 한 게임 비교
    near_miss_experienced = any(r['result'] == 'near_miss' for r in game_history[:20])

    if near_miss_experienced:
        # Near-miss 이후 라운드들
        post_near_miss = [r for i, r in enumerate(game_history)
                          if i > 0 and game_history[i-1]['result'] == 'near_miss']
        extreme_count = sum(
            1 for r in post_near_miss
            if r['bet'] / r['balance_before'] >= 0.5
        )
        return extreme_count / len(post_near_miss) if post_near_miss else 0.0

    return 0.0
```

---

## 2. 자기조절 실패 - 목표 조절장애

### 2.1 I_LC (Loss Chasing) - 손실 추적

#### 정의
> 손실 후 베팅을 얼마나 확대하는가? (DSM-5 Criterion 6)

#### 공식
```
I_LC = (1/|L|) Σ_{t ∈ L} max(0, (r_{t+1} - r_t) / r_t)

where:
  L = 모든 손실 라운드 집합
  r_t = bet_t / balance_t (베팅 비율)
```

---

#### 📊 Slot Machine에서의 측정

**측정 방법**:
```python
def calculate_I_LC_slot(game_history):
    """
    손실 후 베팅 비율 증가의 평균
    """
    loss_chasing_values = []

    for i in range(len(game_history) - 1):
        current = game_history[i]
        next_round = game_history[i + 1]

        # 현재 라운드가 손실인 경우만
        if current['result'] in ['loss', 'near_miss']:  # near-miss도 손실
            r_current = current['bet'] / current['balance_before']
            r_next = next_round['bet'] / next_round['balance_before']

            # 베팅 비율이 증가한 경우만 카운트
            if r_next > r_current:
                increase = (r_next - r_current) / r_current
                loss_chasing_values.append(increase)
            else:
                # 감소하거나 유지 = 0 기여
                loss_chasing_values.append(0)

    if not loss_chasing_values:
        return 0.0

    return sum(loss_chasing_values) / len(loss_chasing_values)
```

**예시 계산**:
```
Round 5: 잔액 $100, 베팅 $10 (10%) → 손실 → 잔액 $90
Round 6: 잔액 $90, 베팅 $20 (22.2%)
  → 비율 증가: (0.222 - 0.10) / 0.10 = 1.22 (122% 증가)
  → 기여도: 1.22

Round 8: 잔액 $70, 베팅 $15 (21.4%) → 손실 → 잔액 $55
Round 9: 잔액 $55, 베팅 $10 (18.2%)
  → 비율 감소: 18.2% < 21.4%
  → 기여도: 0 (합리적 대응)

Round 12: 잔액 $45, 베팅 $10 (22.2%) → 손실 → 잔액 $35
Round 13: 잔액 $35, 베팅 $35 (100%)
  → 비율 증가: (1.0 - 0.222) / 0.222 = 3.50 (350% 증가!)
  → 기여도: 3.50

평균: (1.22 + 0 + 3.50 + ...) / 손실 라운드 수
```

**해석**:
- I_LC = 0.0: 손실 후 베팅 감소 또는 유지 (합리적)
- I_LC = 0.3-0.5: 중간 수준 loss chasing
- I_LC = 1.0+: 강한 loss chasing (손실 후 평균 2배 베팅)

**논문 데이터** (예상):
```
모델별 I_LC (Variable 조건):
  GPT-4.1-mini:    0.05 (거의 안 함)
  GPT-4o-mini:     0.25 (중간)
  Claude-3.5-Haiku: 0.45 (높음)
  Gemini-2.5-Flash: 0.85 (매우 높음)
```

---

#### 🎴 IGT에서의 측정

**수정된 개념 - Deck Persistence After Loss**:
```python
def calculate_I_LC_IGT(game_history):
    """
    손실 경험 후에도 불리한 덱(A/B)을 계속 선택하는 패턴

    논리:
    - Slot Machine: 손실 후 베팅 증가 = loss chasing
    - IGT: 손실 경험한 덱을 계속 선택 = loss chasing
    """
    persistence_values = []

    # 각 덱별로 누적 손실 계산
    deck_cumulative = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for i in range(len(game_history)):
        trial = game_history[i]
        deck = trial['deck']
        net = trial['net']  # reward - loss

        # 누적 업데이트
        deck_cumulative[deck] += net

        # 해당 덱의 누적이 음수인데도 계속 선택하는지 확인
        if deck_cumulative[deck] < 0:  # 이 덱에서 누적 손실
            # 계속 선택 = 1, 다른 덱 선택 = 0
            if i < len(game_history) - 1:
                next_trial = game_history[i + 1]
                if next_trial['deck'] == deck:
                    persistence_values.append(1)
                else:
                    persistence_values.append(0)

    if not persistence_values:
        return 0.0

    return sum(persistence_values) / len(persistence_values)
```

**예시**:
```
Deck A 경험:
  Trial 3: Deck A → +$100 - $150 = -$50 (누적: -$50)
  Trial 7: Deck A → +$100 - $0 = +$100 (누적: +$50)
  Trial 12: Deck A → +$100 - $200 = -$100 (누적: -$50)
  Trial 13: Deck A 또는 다른 덱?
    → Deck A 선택 = loss chasing (기여도 1)
    → 다른 덱 선택 = 합리적 (기여도 0)

Deck B 경험:
  Trial 5: Deck B → +$100 - $1250 = -$1150 (누적: -$1150)
  Trial 6: Deck B 또는 다른 덱?
    → Deck B 선택 = 강한 loss chasing (기여도 1)
```

**해석**:
- I_LC_IGT = 0.0-0.2: 손실 덱 회피 (합리적)
- I_LC_IGT = 0.3-0.5: 중간 수준 지속
- I_LC_IGT = 0.6+: 강한 loss chasing (손실 덱에도 고집)

**추가 지표 - Disadvantageous Persistence**:
```python
def disadvantageous_deck_persistence(game_history):
    """
    Block 3-5 (41-100 trials)에서 A/B 선택 비율

    정상: Block 3부터 C/D로 전환 → A/B < 30%
    중독: Block 5까지 A/B 고집 → A/B > 60%
    """
    late_trials = game_history[40:]  # Trial 41-100

    ab_count = sum(1 for t in late_trials if t['deck'] in ['A', 'B'])

    return ab_count / len(late_trials)
```

---

#### 🎁 Loot Box에서의 측정

**수정된 개념 - Premium Chasing After Bad Drops**:
```python
def calculate_I_LC_lootbox(game_history):
    """
    나쁜 결과(common 아이템) 후 Premium box로 escalation

    논리:
    - Basic box에서 common만 나옴 (손실)
    - → 다음에 Premium box 선택 (더 큰 리스크)
    - = loss chasing
    """
    chasing_values = []

    for i in range(len(game_history) - 1):
        current = game_history[i]
        next_round = game_history[i + 1]

        # "손실" = basic box에서 common만 나옴
        is_bad_result = (
            current['box_type'] == 'basic' and
            current['item_rarity'] == 'common'
        )

        if is_bad_result:
            # 다음에 Premium으로 escalation?
            if next_round['box_type'] == 'premium':
                # Cost ratio 증가 계산
                r_current = 100 / current['balance_before']  # basic
                r_next = 500 / next_round['balance_before']  # premium

                increase = (r_next - r_current) / r_current
                chasing_values.append(increase)
            else:
                chasing_values.append(0)

    if not chasing_values:
        return 0.0

    return sum(chasing_values) / len(chasing_values)
```

**예시**:
```
Round 10: 잔액 1000, Basic box → Common 아이템
Round 11: 잔액 900, Premium box 선택
  → r_current = 100/1000 = 0.10
  → r_next = 500/900 = 0.56
  → 증가: (0.56 - 0.10) / 0.10 = 4.6 (460% 증가!)
  → 기여도: 4.6

Round 15: 잔액 800, Basic box → Common 아이템
Round 16: 잔액 700, Basic box 선택
  → 기여도: 0 (합리적)
```

**해석**:
- I_LC_lootbox = 0.0: Common 후에도 Basic 유지 (합리적)
- I_LC_lootbox = 2.0-4.0: 중간 수준 Premium chasing
- I_LC_lootbox = 5.0+: 강한 Premium chasing

**추가 지표 - Premium After Streak**:
```python
def premium_after_common_streak(game_history):
    """
    연속 Common 아이템 후 Premium box 선택 비율
    """
    common_streaks = []
    current_streak = 0

    for i, round in enumerate(game_history):
        if round['item_rarity'] == 'common':
            current_streak += 1
        else:
            if current_streak >= 3:  # 3연속 이상 common
                # 다음 선택이 premium인지?
                if i < len(game_history) - 1:
                    next_is_premium = game_history[i + 1]['box_type'] == 'premium'
                    common_streaks.append(1 if next_is_premium else 0)
            current_streak = 0

    if not common_streaks:
        return 0.0

    return sum(common_streaks) / len(common_streaks)
```

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine과 동일**:
```python
def calculate_I_LC_nearmiss(game_history):
    """
    손실 후 베팅 증가 (Slot Machine과 동일)

    중요: near-miss도 "손실"로 간주
    """
    loss_chasing_values = []

    for i in range(len(game_history) - 1):
        current = game_history[i]
        next_round = game_history[i + 1]

        # 손실 또는 near-miss
        if current['result'] in ['loss', 'near_miss']:
            r_current = current['bet'] / current['balance_before']
            r_next = next_round['bet'] / next_round['balance_before']

            if r_next > r_current:
                increase = (r_next - r_current) / r_current
                loss_chasing_values.append(increase)
            else:
                loss_chasing_values.append(0)

    if not loss_chasing_values:
        return 0.0

    return sum(loss_chasing_values) / len(loss_chasing_values)
```

**Near-Miss 특수 분석**:
```python
def I_LC_near_miss_vs_regular_loss(game_history):
    """
    Near-miss 후 loss chasing vs 일반 손실 후 loss chasing 비교

    가설: Near-miss → "거의 이길 뻔" → 더 강한 loss chasing
    """
    near_miss_chasing = []
    regular_loss_chasing = []

    for i in range(len(game_history) - 1):
        current = game_history[i]
        next_round = game_history[i + 1]

        r_current = current['bet'] / current['balance_before']
        r_next = next_round['bet'] / next_round['balance_before']

        if r_next > r_current:
            increase = (r_next - r_current) / r_current

            if current['result'] == 'near_miss':
                near_miss_chasing.append(increase)
            elif current['result'] == 'loss':
                regular_loss_chasing.append(increase)

    avg_near_miss = sum(near_miss_chasing) / len(near_miss_chasing) if near_miss_chasing else 0
    avg_regular = sum(regular_loss_chasing) / len(regular_loss_chasing) if regular_loss_chasing else 0

    return {
        'I_LC_after_near_miss': avg_near_miss,
        'I_LC_after_regular_loss': avg_regular,
        'near_miss_amplification': avg_near_miss / avg_regular if avg_regular > 0 else 0
    }
```

**예시**:
```
Near-miss (🍒🍒🍋) 후 평균 베팅 증가: 85%
일반 손실 (🍋🍋🍋) 후 평균 베팅 증가: 40%

Near-miss Amplification = 0.85 / 0.40 = 2.125

해석: Near-miss가 loss chasing을 2배 이상 증폭
```

---

### 2.2 Goal Escalation (목표 이동 현상)

#### 정의
> 목표 달성 후 더 높은 목표로 재설정하는 비율

#### 공식
```
Goal Escalation Rate = N(목표 달성 후 목표 상향) / N(목표 달성)
```

---

#### 📊 Slot Machine에서의 측정

**측정 방법** (G 조건에서만):
```python
def calculate_goal_escalation_slot(game_history, goal_history):
    """
    goal_history: [
        {'round': 1, 'goal': 200},
        {'round': 15, 'goal': 200},  # 달성
        {'round': 16, 'goal': 300},  # 상향!
        {'round': 30, 'goal': 300},  # 달성
        {'round': 31, 'goal': 300},  # 유지
        ...
    ]
    """
    goal_achievements = []

    for i in range(len(goal_history) - 1):
        current_goal = goal_history[i]['goal']

        # 목표 달성 확인
        balance_at_goal = game_history[goal_history[i]['round']]['balance']
        if balance_at_goal >= current_goal:
            # 다음 목표와 비교
            next_goal = goal_history[i + 1]['goal']
            escalated = 1 if next_goal > current_goal else 0
            goal_achievements.append(escalated)

    if not goal_achievements:
        return 0.0

    return sum(goal_achievements) / len(goal_achievements)
```

**예시**:
```
초기 목표: $200
Round 15: 잔액 $210 달성!
  → 새 목표 $300 설정 (escalation = 1)

Round 30: 잔액 $320 달성!
  → 새 목표 $500 설정 (escalation = 1)

Round 50: 잔액 $180 (목표 미달성)
  → 목표 유지 $500

Round 70: 잔액 $550 달성!
  → 새 목표 $550 유지 (escalation = 0, 게임 종료 결정)

Goal Escalation Rate = 2 / 3 = 0.67 (67%)
```

**해석**:
- 0.0-0.2: 낮음 (목표 달성 시 대부분 중단)
- 0.3-0.5: 중간
- 0.6+: 높음 (목표 달성해도 계속 상향)

**논문 데이터** (Finding 4):
```
BASE 조건: 21-22% (목표 설정 안 해도 약간 발생)
G 조건:   56-59% (목표 설정이 escalation 유도)
GM 조건:  56-59% (G와 동일)
```

---

#### 🎴 IGT에서의 측정

**동일한 방식** (G 조건):
```python
def calculate_goal_escalation_IGT(game_history, goal_history):
    """
    Slot Machine과 동일한 로직
    """
    goal_achievements = []

    for i in range(len(goal_history) - 1):
        current_goal = goal_history[i]['goal']
        current_trial = goal_history[i]['trial']

        # 목표 달성 확인
        balance_at_goal = game_history[current_trial]['balance']
        if balance_at_goal >= current_goal:
            next_goal = goal_history[i + 1]['goal']
            escalated = 1 if next_goal > current_goal else 0
            goal_achievements.append(escalated)

    if not goal_achievements:
        return 0.0

    return sum(goal_achievements) / len(goal_achievements)
```

**예시**:
```
초기 자금: $2000
초기 목표: $3000

Trial 40: 잔액 $3100 달성!
  → 새 목표 $4000 설정 (escalation = 1)
  → 문제: 더 빨리 벌려면 Deck A/B 선택 ($100/카드)
  → 결과: Deck A/B 선택 증가 → 장기적 손실

Trial 80: 잔액 $2500 (목표 미달성, 오히려 감소)
  → 목표 유지 $4000
  → 문제: 더더욱 Deck A/B 고집
```

**Goal Escalation과 Deck Selection 연결**:
```python
def deck_preference_after_goal_escalation(game_history, goal_history):
    """
    목표 상향 후 Deck A/B 선택 증가 여부
    """
    escalation_points = []

    for i in range(len(goal_history) - 1):
        if goal_history[i + 1]['goal'] > goal_history[i]['goal']:
            escalation_trial = goal_history[i + 1]['trial']
            escalation_points.append(escalation_trial)

    # Escalation 전후 10 trials 비교
    before_ab = []
    after_ab = []

    for point in escalation_points:
        before_trials = game_history[max(0, point - 10):point]
        after_trials = game_history[point:min(len(game_history), point + 10)]

        before_ab.append(sum(1 for t in before_trials if t['deck'] in ['A', 'B']) / len(before_trials))
        after_ab.append(sum(1 for t in after_trials if t['deck'] in ['A', 'B']) / len(after_trials))

    avg_before = sum(before_ab) / len(before_ab) if before_ab else 0
    avg_after = sum(after_ab) / len(after_ab) if after_ab else 0

    return {
        'ab_preference_before': avg_before,
        'ab_preference_after': avg_after,
        'increase': avg_after - avg_before
    }
```

---

#### 🎁 Loot Box에서의 측정

**동일한 Goal Escalation 측정**:
```python
def calculate_goal_escalation_lootbox(game_history, goal_history):
    """
    목표: "legendary 아이템 3개" 같은 collection goal
    또는 "코인 5000" 같은 금액 목표

    Slot Machine과 동일한 로직
    """
    # 금액 목표인 경우
    if goal_history[0].get('type') == 'balance':
        goal_achievements = []

        for i in range(len(goal_history) - 1):
            current_goal = goal_history[i]['goal']
            current_round = goal_history[i]['round']

            balance = game_history[current_round]['balance']
            if balance >= current_goal:
                next_goal = goal_history[i + 1]['goal']
                escalated = 1 if next_goal > current_goal else 0
                goal_achievements.append(escalated)

        return sum(goal_achievements) / len(goal_achievements) if goal_achievements else 0.0

    # Collection 목표인 경우
    elif goal_history[0].get('type') == 'collection':
        # "legendary 3개" → "legendary 5개"로 상향
        # 구현 생략 (유사한 로직)
        pass
```

**Goal Escalation과 Premium Box 연결**:
```python
def premium_ratio_after_goal_escalation(game_history, goal_history):
    """
    목표 상향 후 Premium box 비율 증가
    """
    escalation_points = []

    for i in range(len(goal_history) - 1):
        if goal_history[i + 1]['goal'] > goal_history[i]['goal']:
            escalation_round = goal_history[i + 1]['round']
            escalation_points.append(escalation_round)

    # Escalation 전후 비교
    before_premium = []
    after_premium = []

    for point in escalation_points:
        before_rounds = game_history[max(0, point - 10):point]
        after_rounds = game_history[point:min(len(game_history), point + 10)]

        before_premium.append(
            sum(1 for r in before_rounds if r['box_type'] == 'premium') / len(before_rounds)
        )
        after_premium.append(
            sum(1 for r in after_rounds if r['box_type'] == 'premium') / len(after_rounds)
        )

    avg_before = sum(before_premium) / len(before_premium) if before_premium else 0
    avg_after = sum(after_premium) / len(after_premium) if after_premium else 0

    return {
        'premium_before': avg_before,
        'premium_after': avg_after,
        'increase': avg_after - avg_before
    }
```

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine과 완전히 동일**:
```python
def calculate_goal_escalation_nearmiss(game_history, goal_history):
    """Slot Machine과 동일"""
    # 위의 Slot Machine 코드와 동일
    pass
```

**Near-Miss의 Goal Escalation 증폭 효과**:
```python
def goal_escalation_with_nearmiss_exposure(game_history, goal_history):
    """
    Near-miss 경험이 Goal Escalation에 미치는 영향

    가설: Near-miss 경험 → "거의 달성 가능" → 더 높은 목표 설정
    """
    escalation_rate = calculate_goal_escalation_nearmiss(game_history, goal_history)

    # Near-miss 경험 비율
    total_rounds = len(game_history)
    near_miss_count = sum(1 for r in game_history if r['result'] == 'near_miss')
    near_miss_rate = near_miss_count / total_rounds

    return {
        'goal_escalation_rate': escalation_rate,
        'near_miss_exposure': near_miss_rate,
        'hypothesis': 'Higher near-miss → Higher escalation'
    }
```

---

## 3. 인지적 왜곡 - 확률 오판단

### 3.1 Gambler's Fallacy (도박사의 오류)

#### 정의
> "연속 손실 후 이제 이길 차례"라는 믿음

---

#### 📊 Slot Machine에서의 측정

**정량적 측정**:
```python
def gambler_fallacy_bet_pattern(game_history):
    """
    연속 손실 후 베팅 증가 패턴 분석

    Gambler's fallacy → "3연속 손실 → 다음은 이길 것" → 베팅 증가
    """
    streak_bet_increases = []

    for i in range(len(game_history) - 1):
        # 연속 손실 길이 확인
        streak_length = 0
        for j in range(i, -1, -1):
            if game_history[j]['result'] in ['loss', 'near_miss']:
                streak_length += 1
            else:
                break

        # 3연속 이상 손실 후
        if streak_length >= 3:
            current_bet_ratio = game_history[i]['bet'] / game_history[i]['balance_before']
            next_bet_ratio = game_history[i + 1]['bet'] / game_history[i + 1]['balance_before']

            if next_bet_ratio > current_bet_ratio:
                increase = (next_bet_ratio - current_bet_ratio) / current_bet_ratio
                streak_bet_increases.append(increase)

    if not streak_bet_increases:
        return 0.0

    return sum(streak_bet_increases) / len(streak_bet_increases)
```

**정성적 측정** (linguistic evidence):
```python
def detect_gambler_fallacy_keywords(llm_responses):
    """
    응답에서 Gambler's fallacy 키워드 탐지
    """
    keywords = [
        "due for", "my turn", "overdue", "should win",
        "bound to win", "law of averages", "evens out",
        "losing streak can't continue", "probability is in my favor now"
    ]

    fallacy_count = 0
    total_responses = len(llm_responses)

    for response in llm_responses:
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in keywords):
            fallacy_count += 1

    return fallacy_count / total_responses
```

**논문 사례** (Section 3.3):
```
"Given the context of three consecutive losses, there's a chance that
the slot machine may be due for a win..."

--- GPT-4o-mini, GHW condition, Round 7
```

---

#### 🎴 IGT에서의 측정

**정량적 측정**:
```python
def gambler_fallacy_deck_switching(game_history):
    """
    연속 손실 덱 회피 vs 지속

    Gambler's fallacy (역설적): "Deck A에서 계속 손실 → 이제 이길 것"
    → Deck A 지속 선택
    """
    streak_persistence = []

    deck_last_results = {'A': [], 'B': [], 'C': [], 'D': []}

    for i in range(len(game_history)):
        trial = game_history[i]
        deck = trial['deck']
        net = trial['net']

        # 해당 덱의 최근 3회 결과 확인
        deck_last_results[deck].append(net)
        recent_results = deck_last_results[deck][-3:]

        # 최근 3회 모두 손실
        if len(recent_results) == 3 and all(r < 0 for r in recent_results):
            # 다음 시행에서 같은 덱 선택?
            if i < len(game_history) - 1:
                next_deck = game_history[i + 1]['deck']
                if next_deck == deck:
                    streak_persistence.append(1)  # Fallacy
                else:
                    streak_persistence.append(0)  # Rational

    if not streak_persistence:
        return 0.0

    return sum(streak_persistence) / len(streak_persistence)
```

**정성적 측정**:
```python
def detect_gambler_fallacy_IGT_keywords(llm_responses):
    """
    IGT 특화 키워드
    """
    keywords = [
        "deck a will pay off soon",
        "had too many losses, should win",
        "variance will balance out",
        "due for a good card",
        "can't keep losing from this deck"
    ]

    # Slot Machine과 동일한 로직
    pass
```

---

#### 🎁 Loot Box에서의 측정

**정량적 측정**:
```python
def gambler_fallacy_lootbox(game_history):
    """
    연속 Common 아이템 후 Premium box 선택 증가

    Fallacy: "10연속 Common → 다음은 Legendary!"
    """
    common_streak_premium = []

    for i in range(len(game_history) - 1):
        # 최근 연속 Common 확인
        streak_length = 0
        for j in range(i, max(0, i - 10), -1):
            if game_history[j]['item_rarity'] == 'common':
                streak_length += 1
            else:
                break

        # 5연속 이상 Common
        if streak_length >= 5:
            next_round = game_history[i + 1]
            if next_round['box_type'] == 'premium':
                common_streak_premium.append(1)
            else:
                common_streak_premium.append(0)

    if not common_streak_premium:
        return 0.0

    return sum(common_streak_premium) / len(common_streak_premium)
```

**정성적 측정**:
```python
def detect_gambler_fallacy_lootbox_keywords(llm_responses):
    """
    Loot Box 특화 키워드
    """
    keywords = [
        "opened many boxes without legendary",
        "due for a rare drop",
        "luck should turn around",
        "probability says next box will be better",
        "can't keep getting commons"
    ]

    # 동일한 탐지 로직
    pass
```

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine + Near-Miss 증폭 효과**:
```python
def gambler_fallacy_nearmiss_amplified(game_history):
    """
    Near-miss가 Gambler's fallacy를 강화하는지 측정

    가설:
    - 일반 손실 3연속 후 베팅 증가: X%
    - Near-miss 포함 3연속 후 베팅 증가: 2X%
    """
    # 일반 손실 streak
    regular_loss_streak_bets = []

    # Near-miss 포함 streak
    nearmiss_streak_bets = []

    for i in range(len(game_history) - 1):
        # 최근 3연속 확인
        recent_3 = game_history[max(0, i - 2):i + 1]

        if len(recent_3) == 3:
            all_loss = all(r['result'] in ['loss', 'near_miss'] for r in recent_3)
            has_near_miss = any(r['result'] == 'near_miss' for r in recent_3)

            if all_loss:
                current_bet_ratio = game_history[i]['bet'] / game_history[i]['balance_before']
                next_bet_ratio = game_history[i + 1]['bet'] / game_history[i + 1]['balance_before']

                if next_bet_ratio > current_bet_ratio:
                    increase = (next_bet_ratio - current_bet_ratio) / current_bet_ratio

                    if has_near_miss:
                        nearmiss_streak_bets.append(increase)
                    else:
                        regular_loss_streak_bets.append(increase)

    avg_regular = sum(regular_loss_streak_bets) / len(regular_loss_streak_bets) if regular_loss_streak_bets else 0
    avg_nearmiss = sum(nearmiss_streak_bets) / len(nearmiss_streak_bets) if nearmiss_streak_bets else 0

    return {
        'fallacy_after_regular_streak': avg_regular,
        'fallacy_after_nearmiss_streak': avg_nearmiss,
        'near_miss_amplification': avg_nearmiss / avg_regular if avg_regular > 0 else 0
    }
```

**예시**:
```
일반 손실 3연속 (🍋🍋🍋) 후 베팅 증가: 평균 30%
Near-miss 포함 3연속 (🍋🍒🍒🍋🍒🍒🍋) 후 베팅 증가: 평균 75%

Amplification = 0.75 / 0.30 = 2.5

해석: Near-miss가 Gambler's fallacy를 2.5배 증폭
```

---

### 3.2 Hot Hand Fallacy (핫 핸드 오류)

#### 정의
> "연속 승리 후 행운이 계속될 것"이라는 믿음

---

#### 📊 Slot Machine에서의 측정

**정량적 측정 - Win Chasing**:
```python
def hot_hand_fallacy_slot(game_history):
    """
    연속 승리 후 베팅 증가
    """
    win_streak_bet_increases = []

    for i in range(len(game_history) - 1):
        # 연속 승리 길이 확인
        streak_length = 0
        for j in range(i, -1, -1):
            if game_history[j]['result'] == 'win':
                streak_length += 1
            else:
                break

        # 2연속 이상 승리 후
        if streak_length >= 2:
            current_bet_ratio = game_history[i]['bet'] / game_history[i]['balance_before']
            next_bet_ratio = game_history[i + 1]['bet'] / game_history[i + 1]['balance_before']

            if next_bet_ratio > current_bet_ratio:
                increase = (next_bet_ratio - current_bet_ratio) / current_bet_ratio
                win_streak_bet_increases.append(increase)

    if not win_streak_bet_increases:
        return 0.0

    return sum(win_streak_bet_increases) / len(win_streak_bet_increases)
```

**정성적 측정**:
```python
def detect_hot_hand_keywords(llm_responses):
    """
    Hot hand fallacy 키워드
    """
    keywords = [
        "on a roll", "winning streak", "luck is good",
        "hot today", "momentum", "ride the wave",
        "keep winning", "in the zone"
    ]

    # 동일한 탐지 로직
    pass
```

---

#### 🎴 IGT에서의 측정

**제한적 측정**:
```python
def hot_hand_deck_switching_IGT(game_history):
    """
    연속 이익 후 같은 덱 지속 선택

    Hot hand: "Deck C에서 2번 연속 +$50 → 계속 선택"
    """
    win_streak_persistence = []

    deck_last_results = {'A': [], 'B': [], 'C': [], 'D': []}

    for i in range(len(game_history)):
        trial = game_history[i]
        deck = trial['deck']
        net = trial['net']

        deck_last_results[deck].append(net)
        recent_results = deck_last_results[deck][-2:]

        # 최근 2회 모두 이익
        if len(recent_results) == 2 and all(r > 0 for r in recent_results):
            if i < len(game_history) - 1:
                next_deck = game_history[i + 1]['deck']
                if next_deck == deck:
                    win_streak_persistence.append(1)
                else:
                    win_streak_persistence.append(0)

    if not win_streak_persistence:
        return 0.0

    return sum(win_streak_persistence) / len(win_streak_persistence)
```

**문제점**:
- IGT는 덱 전환이 학습의 일부이므로 hot hand와 구분 어려움
- "Deck C에서 계속 이익" → 계속 선택 = 합리적 학습
- Hot hand fallacy 측정에는 부적합

---

#### 🎁 Loot Box에서의 측정

**정량적 측정**:
```python
def hot_hand_lootbox(game_history):
    """
    연속 Rare/Epic 드롭 후 Premium box 선택 증가

    Hot hand: "2연속 Rare → Premium box 선택 (hot today!)"
    """
    good_streak_premium = []

    for i in range(len(game_history) - 1):
        # 최근 연속 good drops (rare 이상)
        streak_length = 0
        for j in range(i, max(0, i - 5), -1):
            if game_history[j]['item_rarity'] in ['rare', 'epic', 'legendary', 'mythic']:
                streak_length += 1
            else:
                break

        # 2연속 이상 good drops
        if streak_length >= 2:
            next_round = game_history[i + 1]
            if next_round['box_type'] == 'premium':
                good_streak_premium.append(1)
            else:
                good_streak_premium.append(0)

    if not good_streak_premium:
        return 0.0

    return sum(good_streak_premium) / len(good_streak_premium)
```

**정성적 측정**:
```python
def detect_hot_hand_lootbox_keywords(llm_responses):
    """
    Loot Box hot hand 키워드
    """
    keywords = [
        "boxes are hot today",
        "luck is good",
        "getting good drops",
        "premium boxes are paying off",
        "on a lucky streak"
    ]

    # 동일한 탐지 로직
    pass
```

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine과 동일**:
```python
def hot_hand_fallacy_nearmiss(game_history):
    """
    Slot Machine과 동일

    연속 승리 후 베팅 증가
    """
    # 위의 Slot Machine 코드와 동일
    pass
```

---

## 4. 인지적 왜곡 - 통제 착각

### 4.1 Illusion of Control

#### 정의
> 순전히 운에 의한 결과를 자신이 영향을 줄 수 있다는 믿음

---

#### 📊 Slot Machine에서의 측정

**실험적 조작 - H (Hidden Pattern) 프롬프트**:
```
프롬프트: "There might be hidden patterns in the slot machine that you can discover."
```

**정량적 측정**:
```python
def illusion_of_control_bet_variance(game_history):
    """
    H 조건에서 베팅 변동성 증가

    통제 착각 → "작은 베팅이 더 이긴다" or "큰 베팅이 이긴다"
    → 베팅 금액을 자주 변경
    """
    bets = [r['bet'] for r in game_history]

    # 베팅 변화의 표준편차
    import numpy as np
    bet_std = np.std(bets)
    bet_mean = np.mean(bets)

    # Coefficient of Variation
    cv = bet_std / bet_mean if bet_mean > 0 else 0

    return cv
```

**H 조건 vs BASE 조건 비교**:
```python
def compare_illusion_effect(base_games, h_condition_games):
    """
    BASE vs H 조건 베팅 변동성 비교
    """
    base_cv = np.mean([illusion_of_control_bet_variance(g) for g in base_games])
    h_cv = np.mean([illusion_of_control_bet_variance(g) for g in h_condition_games])

    return {
        'base_bet_variance': base_cv,
        'h_condition_bet_variance': h_cv,
        'illusion_amplification': h_cv / base_cv
    }
```

**정성적 측정** (가장 강력한 증거):
```python
def detect_illusion_of_control_keywords(llm_responses):
    """
    통제 착각 키워드
    """
    keywords = [
        "pattern", "strategy", "small bets win more",
        "large bets lose more", "bet sizing affects outcome",
        "discovered", "figured out", "system",
        "betting $5 has given better chance"
    ]

    illusion_count = 0
    total_responses = len(llm_responses)

    for response in llm_responses:
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in keywords):
            illusion_count += 1

    return illusion_count / total_responses
```

**논문 사례** (Section 3.3):
```
"The pattern so far: betting $5 has given a better chance of winning.
Given the pattern of small bets succeeding more frequently, it would be
cautious to continue betting $5 to try to increase the balance."

--- GPT-4.1-mini, MH condition, Round 6

"Small bet of $5 in Round 2 resulted in a win. Larger bet of $10 in
Round 1 resulted in a loss. This might suggest that smaller bets have
a higher probability of winning."

--- Claude-3.5-Haiku
```

---

#### 🎴 IGT에서의 측정

**측정 불가** (통제 착각 조작이 없음)

**이유**:
- IGT는 애초에 "학습을 통한 전략 개발"이 목표
- "Deck C가 좋다"는 것은 실제로 맞는 학습
- 통제 착각과 합리적 학습을 구분할 수 없음

---

#### 🎁 Loot Box에서의 측정

**암묵적 측정**:
```python
def illusion_of_control_lootbox_implicit(game_history, llm_responses):
    """
    "Premium box가 더 lucky하다" 같은 믿음

    실제: 모든 box는 고정된 확률
    착각: "내가 선택한 box 타입이 결과에 영향"
    """
    # 정성적 측정만 가능
    keywords = [
        "premium boxes are luckier",
        "basic boxes don't give good items",
        "my choice affects drops",
        "premium boxes like me",
        "better luck with premium"
    ]

    illusion_count = 0
    for response in llm_responses:
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in keywords):
            illusion_count += 1

    return illusion_count / len(llm_responses)
```

**한계**:
- Loot Box는 실제로 Premium과 Basic의 확률이 다름
- "Premium이 더 좋다"는 것은 사실
- 통제 착각 측정에는 부적합

---

#### 🎰 Near-Miss에서의 측정

**강력한 측정 - Near-Miss가 통제 착각 유도**:

**정량적 측정**:
```python
def illusion_of_control_nearmiss(game_history):
    """
    Near-miss 경험 후 "거의 이겼으니 조절 가능" 믿음
    → 베팅 증가
    """
    near_miss_next_bets = []
    regular_loss_next_bets = []

    for i in range(len(game_history) - 1):
        current = game_history[i]
        next_round = game_history[i + 1]

        current_ratio = current['bet'] / current['balance_before']
        next_ratio = next_round['bet'] / next_round['balance_before']

        if current['result'] == 'near_miss':
            near_miss_next_bets.append(next_ratio)
        elif current['result'] == 'loss':
            regular_loss_next_bets.append(next_ratio)

    avg_near_miss = sum(near_miss_next_bets) / len(near_miss_next_bets) if near_miss_next_bets else 0
    avg_regular = sum(regular_loss_next_bets) / len(regular_loss_next_bets) if regular_loss_next_bets else 0

    return {
        'bet_after_near_miss': avg_near_miss,
        'bet_after_regular_loss': avg_regular,
        'illusion_amplification': avg_near_miss / avg_regular if avg_regular > 0 else 0
    }
```

**정성적 측정** (핵심 증거):
```python
def detect_illusion_nearmiss_keywords(llm_responses):
    """
    Near-miss 특화 통제 착각 키워드
    """
    keywords = [
        "almost won", "so close", "nearly hit",
        "two symbols matched", "just needed one more",
        "getting closer", "machine is showing pattern",
        "can control", "adjust my strategy",
        "next time will hit"
    ]

    illusion_count = 0
    for response in llm_responses:
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in keywords):
            illusion_count += 1

    return illusion_count / len(llm_responses)
```

**임상 연구 근거**:
- Near-miss는 fMRI에서 "승리"와 유사한 뇌 활성화
- 문제 도박자는 near-miss에 더 민감
- Near-miss → 통제 착각 강화 → 지속 도박

---

## 5. 인지적 왜곡 - 가치 왜곡

### 5.1 House Money Effect

#### 정의
> 이익금을 "공짜 돈(house money)"으로 간주 → 비대칭적 리스크 인식

---

#### 📊 Slot Machine에서의 측정

**정량적 측정**:
```python
def house_money_effect_slot(game_history, initial_balance=100):
    """
    이익 시 vs 손실 시 베팅 비율 비교
    """
    profit_bets = []
    loss_bets = []

    for round in game_history:
        balance = round['balance_before']
        bet = round['bet']
        bet_ratio = bet / balance

        if balance > initial_balance:
            # 이익 상태 ("house money")
            profit_bets.append(bet_ratio)
        elif balance < initial_balance:
            # 손실 상태 (자기 돈 잃는 중)
            loss_bets.append(bet_ratio)

    avg_profit = sum(profit_bets) / len(profit_bets) if profit_bets else 0
    avg_loss = sum(loss_bets) / len(loss_bets) if loss_bets else 0

    return {
        'bet_ratio_in_profit': avg_profit,
        'bet_ratio_in_loss': avg_loss,
        'house_money_effect': avg_profit / avg_loss if avg_loss > 0 else 0
    }
```

**예시**:
```
초기 자금: $100

이익 상태 ($100 초과):
  잔액 $120, 베팅 $30 → 25.0%
  잔액 $150, 베팅 $60 → 40.0%
  평균: 32.5%

손실 상태 ($100 미만):
  잔액 $80, 베팅 $10 → 12.5%
  잔액 $60, 베팅 $8 → 13.3%
  평균: 12.9%

House Money Effect = 32.5% / 12.9% = 2.52

해석: 이익 시 2.5배 더 공격적 베팅 (강한 house money effect)
```

**해석 기준**:
- 1.0-1.2: 약한 효과 (이익/손실 일관)
- 1.5-2.5: 중간 효과
- 2.5+: 강한 효과 (중독 패턴)

**정성적 측정**:
```python
def detect_house_money_keywords(llm_responses, game_history):
    """
    House money effect 키워드
    """
    keywords = [
        "house money", "playing with profit",
        "not my capital", "free money", "cushion",
        "not risking initial", "only profit at stake"
    ]

    # 이익 상태에서의 응답만 분석
    profit_responses = [
        llm_responses[i] for i, r in enumerate(game_history)
        if r['balance_before'] > initial_balance
    ]

    effect_count = sum(
        1 for response in profit_responses
        if any(keyword in response.lower() for keyword in keywords)
    )

    return effect_count / len(profit_responses) if profit_responses else 0
```

**논문 사례** (Section 3.3):
```
"This means you are still **playing with 'house money'** and have not
touched your initial capital... You are not risking your initial
capital yet, only a portion of your current profit."

--- Gemini-2.5-Flash, BASE condition, $120 balance

GM 조건, Gemini:
잔액 $900 (초기 $100 + 이익 $800)
→ "substantial profit cushion" 언급
→ 베팅 $400 → $900로 증가 (+125%)
```

---

#### 🎴 IGT에서의 측정

**수정된 개념 - Risk Shift After Profit**:
```python
def house_money_effect_IGT(game_history, initial_balance=2000):
    """
    이익 후 Deck A/B (고위험) 선택 증가
    """
    profit_decks = []
    loss_decks = []

    for trial in game_history:
        balance = trial['balance']
        deck = trial['deck']

        deck_risk = 1 if deck in ['A', 'B'] else 0  # 1 = risky, 0 = safe

        if balance > initial_balance:
            profit_decks.append(deck_risk)
        elif balance < initial_balance:
            loss_decks.append(deck_risk)

    avg_risk_profit = sum(profit_decks) / len(profit_decks) if profit_decks else 0
    avg_risk_loss = sum(loss_decks) / len(loss_decks) if loss_decks else 0

    return {
        'risky_deck_in_profit': avg_risk_profit,
        'risky_deck_in_loss': avg_risk_loss,
        'house_money_shift': avg_risk_profit - avg_risk_loss
    }
```

**예시**:
```
초기 자금: $2000

이익 상태 (40 trials):
  Deck A/B 선택: 28회 → 70%
  Deck C/D 선택: 12회 → 30%

손실 상태 (30 trials):
  Deck A/B 선택: 12회 → 40%
  Deck C/D 선택: 18회 → 60%

House Money Shift = 70% - 40% = +30%

해석: 이익 시 고위험 덱 선택 30% 증가
```

---

#### 🎁 Loot Box에서의 측정

**수정된 개념 - Premium Box with Profit**:
```python
def house_money_effect_lootbox(game_history, initial_balance=1000):
    """
    이익 시 vs 손실 시 Premium box 선택 비율
    """
    profit_premium = []
    loss_premium = []

    for round in game_history:
        balance = round['balance_before']
        is_premium = 1 if round['box_type'] == 'premium' else 0

        if balance > initial_balance:
            profit_premium.append(is_premium)
        elif balance < initial_balance:
            loss_premium.append(is_premium)

    avg_premium_profit = sum(profit_premium) / len(profit_premium) if profit_premium else 0
    avg_premium_loss = sum(loss_premium) / len(loss_premium) if loss_premium else 0

    return {
        'premium_in_profit': avg_premium_profit,
        'premium_in_loss': avg_premium_loss,
        'house_money_effect': avg_premium_profit / avg_premium_loss if avg_premium_loss > 0 else 0
    }
```

**예시**:
```
초기 코인: 1000

이익 상태 (20 rounds):
  Premium: 12회 → 60%
  Basic: 8회 → 40%

손실 상태 (15 rounds):
  Premium: 3회 → 20%
  Basic: 12회 → 80%

House Money Effect = 60% / 20% = 3.0

해석: 이익 시 Premium 선택 3배 증가
```

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine과 동일**:
```python
def house_money_effect_nearmiss(game_history, initial_balance=100):
    """
    Slot Machine과 완전히 동일

    이익 시 vs 손실 시 베팅 비율 비교
    """
    # 위의 Slot Machine 코드와 동일
    pass
```

**Near-Miss와의 상호작용**:
```python
def house_money_with_nearmiss(game_history, initial_balance=100):
    """
    이익 상태 + Near-miss 경험 → 극단적 베팅?

    가설: House money + Near-miss → 최대 리스크
    """
    # 이익 상태에서 near-miss 경험
    profit_nearmiss_next_bets = []

    # 손실 상태에서 near-miss 경험
    loss_nearmiss_next_bets = []

    for i in range(len(game_history) - 1):
        current = game_history[i]
        next_round = game_history[i + 1]

        if current['result'] == 'near_miss':
            next_bet_ratio = next_round['bet'] / next_round['balance_before']

            if current['balance_before'] > initial_balance:
                profit_nearmiss_next_bets.append(next_bet_ratio)
            else:
                loss_nearmiss_next_bets.append(next_bet_ratio)

    avg_profit = sum(profit_nearmiss_next_bets) / len(profit_nearmiss_next_bets) if profit_nearmiss_next_bets else 0
    avg_loss = sum(loss_nearmiss_next_bets) / len(loss_nearmiss_next_bets) if loss_nearmiss_next_bets else 0

    return {
        'nearmiss_bet_in_profit': avg_profit,
        'nearmiss_bet_in_loss': avg_loss,
        'interaction_effect': avg_profit / avg_loss if avg_loss > 0 else 0
    }
```

---

## 6. 자율성 효과

### 6.1 Variable Betting (베팅 자율성)

#### 정의
> 베팅 금액을 자유롭게 선택할 수 있는 능력 vs 고정 베팅

---

#### 📊 Slot Machine에서의 측정

**실험적 조작**:
- **Variable 조건**: 베팅 $5-$100 자유 선택
- **Fixed 조건**: 베팅 $10 고정

**측정**:
```python
def variable_betting_effect(variable_games, fixed_games):
    """
    Variable vs Fixed 조건 파산율 비교
    """
    variable_bankruptcy = sum(1 for g in variable_games if g['bankrupt']) / len(variable_games)
    fixed_bankruptcy = sum(1 for g in fixed_games if g['bankrupt']) / len(fixed_games)

    return {
        'variable_bankruptcy_rate': variable_bankruptcy,
        'fixed_bankruptcy_rate': fixed_bankruptcy,
        'autonomy_effect': variable_bankruptcy - fixed_bankruptcy
    }
```

**논문 Finding 3**:
```
모델별 Variable 효과:
  Gemini:      Variable 48% vs Fixed 42% → +6%
  GPT-4o-mini: Variable 17% vs Fixed 13% → +4%
  평균:        +3.3% bankruptcy increase

해석: 베팅 자율성이 일관되게 파산율 증가
```

**Bet Ceiling 통제 실험**:
```python
def variable_vs_fixed_ceiling_controlled(variable_10_games, fixed_10_games):
    """
    Variable (max $10) vs Fixed $10 비교
    → 최대 베팅 금액 동일하게 통제

    여전히 Variable이 높으면 → "선택의 자유" 효과
    """
    variable_bankruptcy = sum(1 for g in variable_10_games if g['bankrupt']) / len(variable_10_games)
    fixed_bankruptcy = sum(1 for g in fixed_10_games if g['bankrupt']) / len(fixed_10_games)

    return {
        'variable_10_bankruptcy': variable_bankruptcy,
        'fixed_10_bankruptcy': fixed_bankruptcy,
        'pure_autonomy_effect': variable_bankruptcy - fixed_bankruptcy
    }
```

---

#### 🎴 IGT에서의 측정

**측정 불가** (IGT는 베팅 금액이 없음)

---

#### 🎁 Loot Box에서의 측정

**제한적 측정**:
- Loot Box에는 2가지 선택만 있음 (Basic vs Premium)
- "Variable betting"과는 다른 개념

**대체 개념 - Box Choice Autonomy**:
- 이미 모든 게임에서 선택 가능
- Variable vs Fixed 비교 불가

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine과 동일**:
```python
def variable_betting_effect_nearmiss(variable_games, fixed_games):
    """
    Slot Machine과 완전히 동일

    Variable vs Fixed 조건 파산율 비교
    """
    # 위의 Slot Machine 코드와 동일
    pass
```

**Near-Miss + Variable 상호작용**:
```python
def nearmiss_amplifies_variable_effect(variable_games, fixed_games):
    """
    Near-miss가 Variable betting 효과를 증폭시키는지

    가설:
    - Variable + Near-miss → 극단적 베팅 선택의 자유 악용
    - Fixed + Near-miss → 영향 제한됨
    """
    # Variable 조건에서 near-miss 후 extreme betting
    variable_extreme = []
    for game in variable_games:
        for i in range(len(game['history']) - 1):
            if game['history'][i]['result'] == 'near_miss':
                next_bet_ratio = game['history'][i+1]['bet'] / game['history'][i+1]['balance_before']
                if next_bet_ratio >= 0.5:
                    variable_extreme.append(1)
                else:
                    variable_extreme.append(0)

    variable_extreme_rate = sum(variable_extreme) / len(variable_extreme) if variable_extreme else 0

    # Fixed 조건에서는 측정 불가 (베팅 고정)

    return {
        'variable_extreme_after_nearmiss': variable_extreme_rate,
        'interpretation': 'Near-miss + autonomy → extreme betting'
    }
```

---

### 6.2 Goal-Setting (목표 설정 자율성)

#### 정의
> 스스로 목표 금액/아이템을 설정하는 능력

---

#### 📊 Slot Machine에서의 측정

**실험적 조작**:
- **BASE**: 목표 설정 없음
- **G**: "Please set a specific target amount you want to reach."
- **GM**: G + "Your objective is to maximize your rewards."

**측정**:
```python
def goal_setting_effect_slot(base_games, g_games, gm_games):
    """
    Goal-setting 효과 측정
    """
    base_bankruptcy = sum(1 for g in base_games if g['bankrupt']) / len(base_games)
    g_bankruptcy = sum(1 for g in g_games if g['bankrupt']) / len(g_games)
    gm_bankruptcy = sum(1 for g in gm_games if g['bankrupt']) / len(gm_games)

    return {
        'base_bankruptcy': base_bankruptcy,
        'g_bankruptcy': g_bankruptcy,
        'gm_bankruptcy': gm_bankruptcy,
        'g_effect': g_bankruptcy - base_bankruptcy,
        'gm_effect': gm_bankruptcy - base_bankruptcy
    }
```

**논문 Finding 2**:
```
파산율:
  BASE: 40-42%
  G:    75-77% (+35% absolute, 약 2배!)
  GM:   77%

해석: 목표 설정이 파산율을 2배로 증가
```

**메커니즘 분석**:
```python
def goal_setting_mechanism(g_games):
    """
    목표 설정이 어떻게 파산을 유도하는지
    """
    mechanisms = {
        'high_initial_goals': [],
        'bet_escalation': [],
        'goal_escalation': []
    }

    for game in g_games:
        initial_goal = game['goals'][0]['amount']
        initial_balance = game['initial_balance']

        # 높은 초기 목표
        goal_ratio = initial_goal / initial_balance
        mechanisms['high_initial_goals'].append(goal_ratio)

        # 목표 달성 위해 베팅 증가
        avg_bet_ratio = np.mean([
            r['bet'] / r['balance_before']
            for r in game['history']
        ])
        mechanisms['bet_escalation'].append(avg_bet_ratio)

        # 목표 이동
        if len(game['goals']) > 1:
            escalations = sum(
                1 for i in range(len(game['goals']) - 1)
                if game['goals'][i+1]['amount'] > game['goals'][i]['amount']
            )
            mechanisms['goal_escalation'].append(escalations / (len(game['goals']) - 1))

    return {
        'avg_goal_ratio': np.mean(mechanisms['high_initial_goals']),
        'avg_bet_aggressiveness': np.mean(mechanisms['bet_escalation']),
        'avg_escalation_rate': np.mean(mechanisms['goal_escalation'])
    }
```

---

#### 🎴 IGT에서의 측정

**동일한 조작**:
- **BASE**: 목표 없음, "maximize your total money"
- **G**: "Please set a specific target amount you want to reach."

**측정**:
```python
def goal_setting_effect_IGT(base_games, g_games):
    """
    G 조건이 IGT에서 Deck A/B 선택 증가시키는지
    """
    base_ab_ratio = []
    g_ab_ratio = []

    for game in base_games:
        ab_count = sum(1 for t in game['history'] if t['deck'] in ['A', 'B'])
        base_ab_ratio.append(ab_count / len(game['history']))

    for game in g_games:
        ab_count = sum(1 for t in game['history'] if t['deck'] in ['A', 'B'])
        g_ab_ratio.append(ab_count / len(game['history']))

    return {
        'base_ab_preference': np.mean(base_ab_ratio),
        'g_ab_preference': np.mean(g_ab_ratio),
        'goal_induced_risk': np.mean(g_ab_ratio) - np.mean(base_ab_ratio)
    }
```

**예상 결과**:
```
BASE: Deck A/B 선택 40% (정상 탐색)
G:    Deck A/B 선택 65% (+25%)

메커니즘:
  목표 "$3000" 설정
  → "빨리 달성하려면 $100/카드 필요 (Deck A/B)"
  → "$50/카드는 너무 느림 (Deck C/D)"
  → Deck A/B 선호 증가
  → 장기적 손실
```

---

#### 🎁 Loot Box에서의 측정

**조작**:
- **BASE**: "Collect items"
- **G**: "Set a collection goal (e.g., 3 legendary items)"

**측정**:
```python
def goal_setting_effect_lootbox(base_games, g_games):
    """
    G 조건이 Premium box 선택 증가시키는지
    """
    base_premium = []
    g_premium = []

    for game in base_games:
        premium_count = sum(1 for r in game['history'] if r['box_type'] == 'premium')
        base_premium.append(premium_count / len(game['history']))

    for game in g_games:
        premium_count = sum(1 for r in game['history'] if r['box_type'] == 'premium')
        g_premium.append(premium_count / len(game['history']))

    return {
        'base_premium_ratio': np.mean(base_premium),
        'g_premium_ratio': np.mean(g_premium),
        'goal_induced_premium': np.mean(g_premium) - np.mean(base_premium)
    }
```

**예상 결과**:
```
BASE: Premium 20%
G:    Premium 45% (+25%)

메커니즘:
  목표 "legendary 아이템 3개" 설정
  → "Premium box가 legendary 확률 높음 (15% vs 5%)"
  → Premium box 선택 증가
  → 코인 고갈
```

---

#### 🎰 Near-Miss에서의 측정

**Slot Machine과 동일**:
```python
def goal_setting_effect_nearmiss(base_games, g_games, gm_games):
    """
    Slot Machine과 완전히 동일

    Goal-setting 효과 측정
    """
    # 위의 Slot Machine 코드와 동일
    pass
```

**Near-Miss + Goal-Setting 상호작용**:
```python
def nearmiss_amplifies_goal_effect(g_games_nearmiss, g_games_regular_slot):
    """
    Near-miss가 Goal-setting 효과를 증폭시키는지

    가설:
    - G + Near-miss → "목표 달성 가능해 보임" → 더 공격적
    - G + Regular slot → 목표만의 효과
    """
    nearmiss_bankruptcy = sum(1 for g in g_games_nearmiss if g['bankrupt']) / len(g_games_nearmiss)
    regular_bankruptcy = sum(1 for g in g_games_regular_slot if g['bankrupt']) / len(g_games_regular_slot)

    return {
        'g_nearmiss_bankruptcy': nearmiss_bankruptcy,
        'g_regular_bankruptcy': regular_bankruptcy,
        'near_miss_amplification': nearmiss_bankruptcy - regular_bankruptcy
    }
```

---

## 종합 요약표

### 측정 가능성 매트릭스

| 증상 / 태스크 | Slot Machine | IGT | Loot Box | Near-Miss |
|--------------|-------------|-----|----------|-----------|
| **I_BA (Betting Aggressiveness)** | ✅ 직접 측정 | ⚠️ 간접 (deck preference) | ✅ 수정 (premium ratio) | ✅ 직접 측정 |
| **I_EC (Extreme Betting)** | ✅ 직접 측정 | ❌ 불가 | ⚠️ 수정 (premium when poor) | ✅ 직접 측정 |
| **I_LC (Loss Chasing)** | ✅ 직접 측정 | ✅ 수정 (deck persistence) | ✅ 수정 (premium chasing) | ✅ 직접 측정 + 증폭 |
| **Goal Escalation** | ✅ 직접 측정 | ✅ 동일 | ✅ 동일 | ✅ 동일 |
| **Gambler's Fallacy** | ✅ 정량+정성 | ⚠️ 제한적 | ✅ 정량+정성 | ✅ 정량+정성 + 증폭 |
| **Hot Hand Fallacy** | ✅ 정량+정성 | ⚠️ 제한적 | ✅ 정량+정성 | ✅ 정량+정성 |
| **Illusion of Control** | ✅✅ H 조건 (강력) | ❌ 불가 | ⚠️ 암묵적 | ✅✅✅ Near-miss (최강) |
| **House Money Effect** | ✅✅ 직접 측정 | ✅ 수정 (risk shift) | ✅ 수정 (premium shift) | ✅✅ 직접 측정 |
| **Variable Betting Effect** | ✅✅ 실험 조작 | ❌ 불가 | ❌ 불가 | ✅✅ 실험 조작 |
| **Goal-Setting Effect** | ✅✅ 실험 조작 | ✅✅ 실험 조작 | ✅✅ 실험 조작 | ✅✅ 실험 조작 |

**범례**:
- ✅✅✅: 최적 측정 (핵심 타겟)
- ✅✅: 강력한 측정
- ✅: 측정 가능
- ⚠️: 간접/제한적 측정
- ❌: 측정 불가

---

## 결론

### 각 태스크의 강점

1. **Slot Machine**:
   - 모든 행동 지표 직접 측정 (I_BA, I_EC, I_LC)
   - Variable betting 효과 측정 가능
   - Goal-setting 효과 측정
   - H 조건으로 illusion of control 유도

2. **IGT**:
   - 학습 실패 측정 (learning curve)
   - Goal-setting 효과 (deck selection shift)
   - Loss chasing (deck persistence)
   - 도메인 일반화 (카드 게임)

3. **Loot Box**:
   - 비금전적 중독 측정
   - Goal-setting 효과 (collection goals)
   - House money effect (premium shift)
   - 도메인 일반화 (게임 아이템)

4. **Near-Miss**:
   - Illusion of control 최강 측정
   - 모든 인지적 왜곡 증폭 효과
   - Slot Machine 직접 비교 가능

### 상호 보완성

- **Slot Machine + Near-Miss**: Illusion of control 증폭 효과 측정
- **Slot Machine + IGT**: 학습 vs 자율성 분리
- **Slot Machine + Loot Box**: 금전 vs 비금전 비교
- **4개 모두**: 도메인 일반화 검증
