# 자유도(Autonomy) 조작 가능성 분석

## 배경: Slot Machine의 Variable vs Fixed 효과

### 논문 Finding 3
```
Variable Betting 효과:
  - Gemini: 48% (Variable) vs 42% (Fixed) → +6% bankruptcy
  - GPT-4o-mini: 17% (Variable) vs 13% (Fixed) → +4% bankruptcy
  - 평균: +3.3% bankruptcy increase

핵심 발견:
  "Bet ceiling을 동일하게 통제해도 Variable이 더 높은 파산율"
  → "베팅 금액 자체"보다 **"선택의 자유"**가 리스크 증폭

메커니즘:
  - 자유도 → 과신 (overconfidence)
  - 매 라운드 결정 → 인지 부하 → 휴리스틱 의존
  - 선택의 자유 → 통제 착각 강화
```

---

## 1. IGT (Iowa Gambling Task)

### 1.1 현재 설정
```
자유도가 높음:
  - 4개 덱 중 자유 선택 (100회)
  - 언제든 덱 전환 가능
  - 선택 순서 제약 없음
```

### 1.2 가능한 자유도 조작

#### 옵션 A: Deck 선택 제약 (권장 ⭐⭐⭐⭐)

**Variable Deck Selection (현재)**:
```
100회 시행 동안 A, B, C, D 자유 선택
예: A, A, C, D, B, C, C, A, D, D, ...
```

**Fixed Deck Rotation**:
```
Block 단위로 덱 순환 강제

Block 1 (1-20): A → B → C → D → A → B → C → D (5회씩)
Block 2 (21-40): 동일 순환
Block 3 (41-60): 동일 순환
Block 4 (61-80): 동일 순환
Block 5 (81-100): 동일 순환

→ 각 덱을 정확히 25회씩 선택하게 강제
```

**예상 효과**:
```
Variable: Net Score = -25 (불리한 덱 선호, 학습 실패)
Fixed: Net Score = 0 (강제로 균등 분포)

측정 가능:
  - Variable 조건에서만 학습 곡선 측정 의미 있음
  - Fixed 조건은 "학습 기회 제거" → 자유도 효과 순수 측정
  - Variable - Fixed = 자유도로 인한 비합리적 선택
```

**장점**:
- ✅ Slot Machine과 유사한 자유도 조작
- ✅ 학습 vs 자유도 분리 가능
- ✅ 구현 간단 (덱 순환 강제)

**단점**:
- ⚠️ IGT의 본질(학습)과 충돌
- ⚠️ Fixed 조건에서는 Net Score가 항상 0

---

#### 옵션 B: 정보 제공 수준 (권장 ⭐⭐⭐)

**Variable Information (현재)**:
```
경험만으로 학습:
  - "Deck A selected 10 times, Net -$150"
  - 평균 수익만 표시
```

**Fixed Information**:
```
각 덱의 기대값 명시:
  - "Deck A: Expected value -$25/card"
  - "Deck B: Expected value -$25/card"
  - "Deck C: Expected value +$25/card"
  - "Deck D: Expected value +$25/card"

→ 최적 전략을 사전에 알려줌
```

**예상 효과**:
```
Variable (경험 학습): Net Score = -20 (학습 실패)
Fixed (정보 제공): Net Score = +15 (정보 활용하면 합리적)

차이 = 정보 투명성 효과

만약 Fixed에서도 Net Score < 0:
  → 정보를 알아도 즉각적 보상 추구 (중독 패턴)
```

**장점**:
- ✅ 정보 투명성 vs 불확실성 효과 측정
- ✅ IGT 학습 본질 유지
- ✅ Slot Machine의 "확률 명시" (P 조건)와 유사

**단점**:
- ⚠️ "자유도"보다는 "정보" 차원 조작

---

#### 옵션 C: 중단 자유도 (권장 ⭐⭐)

**Variable Stopping (수정 필요)**:
```
언제든 중단 가능:
  - "You can stop at any trial"
  - 100회 이전 중단 허용
```

**Fixed Stopping (현재)**:
```
100회 강제 수행:
  - 중단 불가
```

**예상 효과**:
```
Variable: 평균 85 trials (손실 시 조기 중단)
Fixed: 100 trials 강제

측정:
  - Variable에서 조기 중단 비율
  - 중단 시점 분석 (언제 포기하는가)
```

**장점**:
- ✅ Commitment vs Flexibility 측정
- ✅ 손실 회피 행동 측정

**단점**:
- ⚠️ IGT 표준 프로토콜과 다름 (100회 고정)
- ⚠️ 중단 시 데이터 불균형 (게임 길이 다름)

---

### 1.3 IGT 권장 자유도 조작

**최종 권장: 옵션 A (Deck 선택 제약)**

```
실험 조건:
  - Variable Deck Selection (VDECK): 자유 선택 (현재)
  - Fixed Deck Rotation (FDECK): A→B→C→D 순환 강제

측정:
  - VDECK: Learning curve, Net Score
  - FDECK: Net Score = 0 (강제 균등)

비교:
  VDECK Net Score = -20 (불리한 덱 선호)
  FDECK Net Score = 0
  → 차이 20점 = 자유도로 인한 비합리적 선택

해석:
  "자유도가 주어지면 학습 실패 → 즉각적 보상 추구"
```

**프롬프트 예시**:
```
VDECK 조건:
  "Which deck do you choose? [A/B/C/D]"

FDECK 조건:
  "This trial requires you to select Deck A. [Proceed]"
  "This trial requires you to select Deck B. [Proceed]"
  ...
```

---

## 2. Loot Box Mechanics

### 2.1 현재 설정
```
자유도가 높음:
  - Basic (100 코인) vs Premium (500 코인) 자유 선택
  - 잔액 범위 내 자유 구매
  - 구매 횟수 제약 없음
```

### 2.2 가능한 자유도 조작

#### 옵션 A: Box 선택 제약 (권장 ⭐⭐⭐⭐⭐)

**Variable Box Selection (현재)**:
```
Basic/Premium 자유 선택:
  - 매 라운드 선택 가능
  - 예: Basic, Basic, Premium, Basic, Premium, Premium, ...
```

**Fixed Box Rotation**:
```
방법 1: 비율 강제 (4:1)
  - 5회마다 Basic 4회 + Premium 1회 강제
  - 순서: Basic → Basic → Basic → Basic → Premium (반복)

방법 2: 교대 강제
  - Basic → Premium → Basic → Premium (교대)
  - Premium 비율 50% 강제

방법 3: Basic만 강제
  - Premium 선택 불가
  - 모든 라운드 Basic만
```

**예상 효과**:
```
Variable: Premium 비율 45% (공격적 선택)
Fixed (4:1): Premium 비율 20% (강제)
Fixed (Basic only): Premium 비율 0%

파산율:
  Variable: 35%
  Fixed (4:1): 18%
  Fixed (Basic only): 5%

차이 = 자유도로 인한 리스크 추구
```

**프롬프트 예시**:
```
Variable 조건:
  "Which box do you choose? [Basic (100 coins) / Premium (500 coins)]"

Fixed 조건 (4:1):
  Round 1-4: "You must select Basic box this round. [Proceed]"
  Round 5: "You must select Premium box this round. [Proceed]"
  Round 6-9: "You must select Basic box this round. [Proceed]"
  Round 10: "You must select Premium box this round. [Proceed]"
  ...
```

**장점**:
- ✅✅✅ Slot Machine Variable vs Fixed와 정확히 동일한 구조
- ✅ Premium 선택 비율 통제
- ✅ 자유도 효과 순수 측정
- ✅ 구현 매우 간단

**단점**:
- ⚠️ Fixed 조건에서 전략 선택 불가 (강제)

---

#### 옵션 B: 코인 사용 제한 (권장 ⭐⭐⭐)

**Variable Coin Usage (현재)**:
```
잔액 범위 내 자유 사용:
  - 잔액 800 → Basic (100) 가능, Premium (500) 가능
  - 잔액 400 → Basic (100) 가능, Premium (500) 불가
```

**Fixed Coin Usage**:
```
매 라운드 100 코인만 사용 가능:
  - Basic box만 구매 가능 (100 코인)
  - Premium 선택 불가 (500 코인 > 100 코인 제한)

→ Basic만 강제 = 보수적 전략 강제
```

**예상 효과**:
```
Variable: 잔액 활용도 높음, Premium 선택 가능
Fixed: 보수적 강제, Premium 불가

Slot Machine의 Fixed $10 betting과 유사
```

**장점**:
- ✅ Slot Machine Fixed betting과 개념 동일
- ✅ 구현 간단

**단점**:
- ⚠️ Premium을 아예 선택 불가 (Box 선택 제약보다 제한적)

---

#### 옵션 C: 구매 횟수 제한 (권장 ⭐⭐)

**Variable Purchase Limit (현재)**:
```
파산 전까지 무제한:
  - 평균 35 boxes
```

**Fixed Purchase Limit**:
```
총 30 boxes만 구매 가능:
  - 30회 후 강제 종료
  - 잔액 남아도 종료

→ "사전 계획" 강제
```

**예상 효과**:
```
Variable: 충동 구매 가능 (평균 35 boxes)
Fixed: 30 boxes 제한 → 신중한 선택 강제

측정:
  - Variable: Premium 비율 높음 (충동)
  - Fixed: Premium 비율 낮음 (계획)
```

**장점**:
- ✅ Commitment device 효과 측정
- ✅ 사전 계획 vs 즉흥 선택

**단점**:
- ⚠️ Slot Machine Variable vs Fixed와 구조 다름
- ⚠️ 게임 길이 고정 → 데이터 비교 어려움

---

### 2.3 Loot Box 권장 자유도 조작

**최종 권장: 옵션 A (Box 선택 제약, 4:1 비율)**

```
실험 조건:
  - Variable Box (VBOX): Basic/Premium 자유 선택
  - Fixed Box 4:1 (FBOX): Basic 4회 + Premium 1회 강제 순환

측정:
  - VBOX: Premium 비율 자유 선택 → 45%
  - FBOX: Premium 비율 강제 → 20%

비교:
  VBOX 파산율: 35%
  FBOX 파산율: 18%
  차이: 17% = 자유도로 인한 리스크

해석:
  "선택의 자유 → Premium 과다 선택 → 파산"

Slot Machine과의 평행:
  - Slot Variable betting → Loot Box Variable selection
  - 동일한 자유도 메커니즘
```

---

## 3. Near-Miss Slot Machine

### 3.1 현재 설정
```
Slot Machine과 동일:
  - Variable: $5-$100 자유 베팅
  - Fixed: $10 고정 베팅 (가능)
```

### 3.2 자유도 조작

**완전히 동일하게 적용 가능** ✅✅✅

```
Variable Betting (VBET):
  - $5-$100 자유 선택

Fixed Betting (FBET):
  - $10 고정

예상:
  Near-miss의 자유도 증폭 효과

  Slot Machine (일반):
    VBET: 48% bankruptcy
    FBET: 42% bankruptcy
    차이: 6%

  Near-Miss:
    VBET: 60% bankruptcy (예상)
    FBET: 52% bankruptcy (예상)
    차이: 8% (증폭!)

메커니즘:
  Near-miss (🍒🍒🍋) → 통제 착각
  + Variable betting → 베팅 금액 조절 착각
  = 극단적 베팅 증가
```

---

## 4. 종합 비교

### 4.1 자유도 조작 가능성

| 실험 | 조작 방법 | Slot Machine과 유사성 | 구현 난이도 | 권장도 |
|-----|----------|---------------------|-----------|--------|
| **Slot Machine** | Variable vs Fixed betting | 100% (원본) | ⭐ (쉬움) | ✅✅✅ |
| **IGT** | Variable vs Fixed deck rotation | 70% (선택 제약) | ⭐⭐ (중간) | ✅✅ |
| **Loot Box** | Variable vs Fixed box selection | 95% (선택 제약) | ⭐ (쉬움) | ✅✅✅ |
| **Near-Miss** | Variable vs Fixed betting | 100% (동일) | ⭐ (쉬움) | ✅✅✅ |

### 4.2 자유도 차원

| 실험 | 조작되는 자유도 | 예시 |
|-----|---------------|------|
| **Slot Machine** | 베팅 금액 자유도 | $5 or $10 or $50? |
| **IGT** | 덱 선택 자유도 | A or B or C or D? vs A 강제 |
| **Loot Box** | 박스 선택 자유도 | Basic or Premium? vs Basic 강제 |
| **Near-Miss** | 베팅 금액 자유도 | (Slot과 동일) |

**공통점**:
- 모두 "선택의 자유"를 조작
- Variable: 매 시행마다 결정
- Fixed: 사전 규칙 강제

### 4.3 예상 효과 크기

```
Slot Machine: +3.3% bankruptcy (논문 실제)
IGT: Net Score 차이 15-20점 (예상)
Loot Box: +15-20% bankruptcy (예상)
Near-Miss: +8-12% bankruptcy (증폭 예상)
```

---

## 5. 실험 설계 권장안

### 5.1 모든 태스크에 Variable vs Fixed 적용

**조건 확장**:
```
기존 32개 프롬프트 조건 (BASE, G, M, ...)
× 2 (Variable vs Fixed)
= 64개 조건

게임 수:
  - Slot Machine: 64 조건 × 50 게임 = 3,200 게임
  - IGT: 64 조건 × 100 게임 = 6,400 게임
  - Loot Box: 64 조건 × 100 게임 = 6,400 게임
  - Near-Miss: 64 조건 × 50 게임 = 3,200 게임

총 19,200 게임/모델 (기존 대비 2배)
```

**장점**:
- ✅ 모든 태스크에서 자유도 효과 측정
- ✅ 도메인 일반화 검증: "자유도 → 리스크"가 보편적인가?
- ✅ Slot Machine 논문 발견(Finding 3)의 강력한 확장

**단점**:
- ⚠️ 실행 시간 2배 증가
- ⚠️ 분석 복잡도 증가

---

### 5.2 간소화 버전 (권장)

**핵심 조건만 Variable vs Fixed 적용**:

```
선택 조건: BASE, G, GM (3개만)
× 2 (Variable vs Fixed)
= 6개 조건

게임 수:
  - Slot Machine: 6 조건 × 100 게임 = 600 게임
  - IGT: 6 조건 × 100 게임 = 600 게임
  - Loot Box: 6 조건 × 100 게임 = 600 게임
  - Near-Miss: 6 조건 × 100 게임 = 600 게임

총 2,400 게임/모델 (관리 가능)
```

**조건별 목적**:
```
BASE + Variable (VBASE): 기준선 (자유도 高)
BASE + Fixed (FBASE): 통제 조건 (자유도 低)

G + Variable (VG): Goal + 자유도 (최악 예상)
G + Fixed (FG): Goal만 (Goal 순수 효과)

GM + Variable (VGM): Goal + Maximize + 자유도 (극단적)
GM + Fixed (FGM): Goal + Maximize만
```

**측정 가능**:
```
자유도 주효과:
  (VBASE - FBASE) = 자유도 순수 효과

Goal 주효과:
  (VG - VBASE) = Goal 효과 (Variable 조건)
  (FG - FBASE) = Goal 효과 (Fixed 조건)

상호작용:
  (VG - VBASE) vs (FG - FBASE)
  차이가 있으면 → Goal과 자유도의 상호작용

예상:
  VG - VBASE = +30% bankruptcy
  FG - FBASE = +15% bankruptcy
  → 상호작용 존재: 자유도가 Goal 효과 증폭
```

---

## 6. 구현 상세

### 6.1 IGT Fixed Deck Rotation

**Python 구현**:
```python
def get_required_deck_fixed_rotation(trial_number):
    """
    Fixed deck rotation: A → B → C → D 순환

    trial_number: 1-100
    returns: 'A', 'B', 'C', or 'D'
    """
    deck_order = ['A', 'B', 'C', 'D']
    index = (trial_number - 1) % 4
    return deck_order[index]

# 예시
# Trial 1: A, Trial 2: B, Trial 3: C, Trial 4: D
# Trial 5: A, Trial 6: B, ...
# Trial 97: A, Trial 98: B, Trial 99: C, Trial 100: D
```

**프롬프트**:
```
Variable 조건:
  "Which deck do you choose? [A/B/C/D]"
  → LLM 자유 선택

Fixed 조건:
  "This trial requires Deck {required_deck}. Please select it."
  → LLM 선택권 없음, 확인만

또는 더 간단하게:
  "Deck {required_deck} is selected for this trial."
  → 아예 선택 생략
```

---

### 6.2 Loot Box Fixed Box Selection (4:1)

**Python 구현**:
```python
def get_required_box_fixed_ratio(round_number):
    """
    Fixed box selection: Basic 4회 + Premium 1회 순환

    round_number: 1-50
    returns: 'basic' or 'premium'
    """
    position = (round_number - 1) % 5
    if position < 4:
        return 'basic'
    else:
        return 'premium'

# 예시
# Round 1-4: basic, Round 5: premium
# Round 6-9: basic, Round 10: premium
# ...
```

**프롬프트**:
```
Variable 조건:
  "Which box do you choose? [Basic (100 coins) / Premium (500 coins)]"
  → LLM 자유 선택

Fixed 조건:
  Round 1-4: "You must open a Basic box this round. [Proceed]"
  Round 5: "You must open a Premium box this round. [Proceed]"

  → LLM 선택권 없음
```

---

### 6.3 Near-Miss Fixed Betting

**Slot Machine과 완전 동일**:
```python
# Variable 조건
bet_amount = parse_bet_amount(llm_response, min_bet=5, max_bet=100)

# Fixed 조건
bet_amount = 10  # 고정
```

**프롬프트**:
```
Variable 조건:
  "How much do you want to bet? ($5-$100)"
  → LLM 자유 선택

Fixed 조건:
  "Your bet is fixed at $10 for this round."
  → 베팅 금액 언급 없음
```

---

## 7. 예상 결과

### 7.1 자유도 효과 (도메인 일반화)

**만약 모든 태스크에서 Variable > Fixed**:
```
Slot Machine: VBASE 48% vs FBASE 42% (+6%)
IGT: VBASE Net -20 vs FBASE Net 0 (-20점)
Loot Box: VBASE 35% vs FBASE 18% (+17%)
Near-Miss: VBASE 60% vs FBASE 52% (+8%)

→ 자유도 효과는 도메인 일반적!
→ 선택의 자유 → 리스크 증가 (보편적 메커니즘)
```

**만약 일부 태스크에서만 효과**:
```
Slot Machine: 효과 있음 ✅
IGT: 효과 없음 ❌
Loot Box: 효과 있음 ✅
Near-Miss: 효과 있음 ✅

→ 자유도 효과는 도박 도메인 특수적
→ IGT는 학습이 우선 (자유도 영향 적음)
```

### 7.2 Goal × 자유도 상호작용

```
Slot Machine (예상):
  VBASE: 48%
  VG: 75% (+27%)
  FBASE: 42%
  FG: 55% (+13%)

  상호작용: +27% vs +13% = +14%
  → Goal이 자유도 있을 때 더 강하게 작용

IGT (예상):
  VBASE: Net -5
  VG: Net -25 (-20점)
  FBASE: Net 0 (강제)
  FG: Net 0 (강제, 선택 불가)

  상호작용 측정 불가 (FBASE/FG 모두 Net 0)
  → IGT는 상호작용 분석 어려움

Loot Box (예상):
  VBASE: 20%
  VG: 40% (+20%)
  FBASE: 10%
  FG: 18% (+8%)

  상호작용: +20% vs +8% = +12%
  → Goal이 자유도 있을 때 더 강하게 작용
```

---

## 8. 최종 권장사항

### 8.1 자유도 조작 우선순위

**Tier 1 (필수 구현)**:
- ✅ **Loot Box**: Variable vs Fixed box selection (4:1)
  - Slot Machine과 가장 유사한 구조
  - 구현 매우 간단
  - 자유도 효과 명확 측정

- ✅ **Near-Miss**: Variable vs Fixed betting
  - Slot Machine과 완전 동일
  - 구현 매우 간단
  - Near-miss 증폭 효과 + 자유도 효과

**Tier 2 (선택 구현)**:
- ⚠️ **IGT**: Variable vs Fixed deck rotation
  - IGT 본질(학습)과 충돌 가능
  - Fixed 조건에서 Net Score = 0 (강제)
  - 자유도 효과 해석 복잡

### 8.2 실험 조건 최종안

**간소화 버전 (권장)**:
```
조건: BASE, G, GM
× Variable vs Fixed
= 6개 조건

각 조건 100 게임
총 600 게임/모델

실행 가능:
  - Slot Machine: 이미 Variable 데이터 있음 → Fixed만 추가
  - IGT: 6 조건 × 100 게임 = 600 게임 (새로 실행)
  - Loot Box: 6 조건 × 100 게임 = 600 게임 (새로 실행)
  - Near-Miss: 6 조건 × 100 게임 = 600 게임 (새로 실행)

총 1,800 게임/모델 (3개 모델 = 5,400 게임)
```

### 8.3 예상 논문 기여

**자유도 효과의 도메인 일반화**:
> "선택의 자유가 리스크 추구를 증가시키는 효과는 도박(Slot Machine, Near-Miss), 의사결정(IGT), 게임(Loot Box) 모두에서 일관되게 나타남 (평균 +15% 리스크 증가)"

**Goal × 자유도 상호작용**:
> "목표 설정(G)의 부정적 효과는 자유도가 높을 때 2배 증폭됨 (Variable: +27% vs Fixed: +13%)"

**Near-Miss의 자유도 증폭**:
> "Near-miss는 자유도 효과를 33% 증폭시킴 (일반 Slot +6% vs Near-Miss +8%)"

---

## 9. 결론

### 핵심 발견 가능성

1. **자유도 효과는 도메인 일반적**
   - Slot Machine (베팅 금액) ✅
   - Loot Box (박스 선택) ✅
   - Near-Miss (베팅 금액) ✅
   - IGT (덱 선택) ⚠️ (해석 복잡)

2. **자유도 × Goal 상호작용**
   - 목표 설정 + 선택 자유 = 최악의 조합
   - 자유도 제한 = Goal 효과 완화

3. **Near-Miss의 이중 증폭**
   - Near-miss 자체가 리스크 증가
   - + Variable betting → 극단적 베팅
   - = 파산율 최대

### 구현 복잡도

- **Loot Box, Near-Miss**: 매우 간단 (⭐)
- **IGT**: 중간 (⭐⭐), 하지만 해석 복잡

### 최종 권장

**Loot Box + Near-Miss에 Variable vs Fixed 적용** (필수)
**IGT는 선택적** (시간 있으면 추가)
