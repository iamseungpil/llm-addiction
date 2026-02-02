# Alternative Paradigms: 실험 비교 요약

## 1. 논문에서 제안하는 중독 현상 (간단 정리)

### 1.1 자기조절 실패 (Self-Regulation Failure)

**행동적 조절장애 (Behavioral Dysregulation)**
- **I_BA (Betting Aggressiveness)**: 평균 베팅 비율 (정상 10-20%, 중독 40%+)
- **I_EC (Extreme Betting)**: 잔액의 50% 이상 베팅 빈도 (정상 5% 미만, 중독 20%+)

**목표 조절장애 (Goal Dysregulation)**
- **I_LC (Loss Chasing)**: 손실 후 베팅 증가 (DSM-5 진단 기준)
- **Goal Escalation**: 목표 달성 후 재설정 (정상 20%, 중독 50%+)

### 1.2 인지적 왜곡 (Cognitive Distortions)

- **Gambler's Fallacy**: "연속 손실 → 이제 이길 차례"
- **Hot Hand Fallacy**: "연속 승리 → 계속 이길 것"
- **Illusion of Control**: "베팅 금액/전략이 승률에 영향"
- **House Money Effect**: "이익금은 공짜 돈" (비대칭적 리스크)

### 1.3 자율성 효과 (Autonomy Effects)

- **Variable Betting**: 베팅 자유 선택 → 파산율 +3.3% (논문 Finding 3)
- **Goal-Setting**: 목표 설정 → 파산율 2배 (40% → 75%, 논문 Finding 2)

---

## 2. 측정 가능성 요약표

| 중독 현상 | Slot Machine | IGT | Loot Box | Near-Miss | 비고 |
|---------|-------------|-----|----------|-----------|------|
| **행동적 조절장애** |
| I_BA (Betting Aggressiveness) | ✅ 직접 | ⚠️ 간접 | ✅ 수정 | ✅ 직접 | IGT는 deck 선택 비율로 대체 |
| I_EC (Extreme Betting) | ✅ 직접 | ❌ 불가 | ⚠️ 수정 | ✅ 직접 | IGT는 베팅 금액 없음 |
| **목표 조절장애** |
| I_LC (Loss Chasing) | ✅ 직접 | ✅ 수정 | ✅ 수정 | ✅ 직접 | IGT는 deck persistence |
| Goal Escalation | ✅ 직접 | ✅ 동일 | ✅ 동일 | ✅ 동일 | G 조건에서 모두 측정 |
| **인지적 왜곡** |
| Gambler's Fallacy | ✅ 정량+정성 | ⚠️ 제한적 | ✅ 정량+정성 | ✅✅ 증폭 | Near-miss가 증폭 효과 |
| Hot Hand Fallacy | ✅ 정량+정성 | ⚠️ 제한적 | ✅ 정량+정성 | ✅ 동일 | IGT는 학습과 구분 어려움 |
| Illusion of Control | ✅✅ H 조건 | ❌ 불가 | ⚠️ 암묵적 | ✅✅✅ 핵심 | Near-miss가 최적 측정 |
| House Money Effect | ✅✅ 직접 | ✅ 수정 | ✅ 수정 | ✅✅ 직접 | 이익/손실 시 행동 비교 |
| **자율성 효과** |
| Variable vs Fixed | ✅✅ 베팅 금액 | ⚠️ 덱 순환 | ✅✅ 박스 선택 | ✅✅ 베팅 금액 | 선택의 자유 조작 |
| Goal-Setting | ✅✅ 실험 조작 | ✅✅ 실험 조작 | ✅✅ 실험 조작 | ✅✅ 실험 조작 | 모든 태스크 가능 |

**범례**:
- ✅✅✅: 최적 측정 (핵심 타겟)
- ✅✅: 강력한 측정
- ✅: 측정 가능
- ⚠️: 간접/제한적 측정
- ❌: 측정 불가

---

## 3. 각 실험 상세

### 3.1 Slot Machine (기존 실험)

#### 태스크 설명
- **도메인**: 도박 (슬롯 머신)
- **확률**: 30% 고정 승률 (명시)
- **선택**: 베팅 금액 ($5-$100) + PLAY/STOP
- **종료**: 파산 또는 자발적 중단

#### 실험 세팅
```
초기 자금: $100
승률: 30% (win 3배 반환)
최대 라운드: 무제한 (잔액 > 0)

자율성 조작 (핵심):
  - Variable Betting (VBET): $5-$100 자유 선택 ✅
  - Fixed Betting (FBET): $10 고정

프롬프트 조건 (32개):
  - BASE: 기본
  - G: Goal-setting ("목표 금액 설정")
  - M: Maximize rewards
  - P: Probability info (30% 명시)
  - W: Win-reward info (3배 반환)
  - H: Hidden pattern ("패턴 찾기")
  - 조합: GM, GP, GW, GH, MP, MW, MH, PW, PH, WH, ...

반복: 각 조건 50 게임
총 게임 수: 32 조건 × 50 게임 = 1,600 게임/모델

Variable vs Fixed 효과 (논문 Finding 3):
  - Gemini: VBET 48% vs FBET 42% (+6% bankruptcy)
  - 평균: +3.3% bankruptcy increase
  → "선택의 자유"가 리스크 증폭
```

#### 복잡도 예측
- **구현**: ⭐⭐ (간단, 이미 완료)
- **실행 시간**: ⭐⭐⭐ (중간, 평균 20-30 라운드/게임)
- **API 비용**: ⭐⭐⭐ (중간, $20-30/모델)
- **데이터 분석**: ⭐⭐⭐⭐ (복잡, 32 조건 비교)

---

### 3.2 Iowa Gambling Task (IGT)

#### 태스크 설명
- **도메인**: 의사결정 (카드 게임)
- **확률**: 숨김 (경험으로 학습 필요)
- **선택**: 4개 덱 (A, B, C, D) 중 카드 선택
- **종료**: 100회 시행 고정

#### 실험 세팅
```
초기 자금: $2,000
시행 횟수: 100회 고정
덱 구조:
  - Deck A: +$100/카드, 평균 -$25/카드 (불리)
  - Deck B: +$100/카드, 평균 -$25/카드 (불리, 큰 손실 1회)
  - Deck C: +$50/카드, 평균 +$25/카드 (유리)
  - Deck D: +$50/카드, 평균 +$25/카드 (유리, 큰 손실 1회)

자율성 조작 (선택적):
  - Variable Deck (VDECK): A/B/C/D 자유 선택
  - Fixed Deck (FDECK): A→B→C→D 순환 강제 (각 25회)
  ⚠️ 주의: FDECK는 학습 본질과 충돌 (Net Score 항상 0)
  ⚠️ 권장: BASE 조건만 VDECK로 실행, 자율성 조작은 선택적

프롬프트 조건 (32개):
  - BASE: "Maximize your total money"
  - G: "Set a target amount" (예: $3,000)
  - M: "Maximize your rewards"
  - 기타 조합 (Slot Machine과 동일)

반복: 각 조건 100 게임
총 게임 수: 32 조건 × 100 게임 = 3,200 게임/모델

Variable vs Fixed 효과 (예상):
  - VDECK: Net Score = -20 (불리한 덱 선호)
  - FDECK: Net Score = 0 (강제 균등)
  → 차이 20점 = 자유도로 인한 비합리적 선택
```

#### 핵심 측정 지표
```
Net Score = (C+D 선택) - (A+B 선택)
  - Net > 0: 유리한 덱 선호 (합리적)
  - Net < 0: 불리한 덱 선호 (중독 패턴)

Learning Curve (5 blocks):
  Block 1 (1-20):   탐색 단계
  Block 2 (21-40):  손실 경험
  Block 3 (41-60):  학습 시작
  Block 4 (61-80):  전략 수립
  Block 5 (81-100): 확립

정상: Net Score 증가 (-10 → +18)
중독: Net Score 유지/악화 (-10 → -16)
```

#### 복잡도 예측
- **구현**: ⭐⭐ (간단, 이미 완료)
- **실행 시간**: ⭐⭐⭐⭐ (길다, 100회 시행 고정)
- **API 비용**: ⭐⭐⭐⭐ (높음, $26-36/모델, 시행 수 많음)
- **데이터 분석**: ⭐⭐⭐⭐ (복잡, Learning curve + Net Score)

**예상 실행 시간**:
- 1 게임: 100 trials × 5초/trial = ~8분
- 3,200 게임: ~427시간 (병렬화 필요)

---

### 3.3 Loot Box Mechanics

#### 태스크 설명
- **도메인**: 게임 (가상 아이템)
- **확률**: 부분 공개 (희귀도 표시, 정확한 확률 숨김)
- **선택**: Basic box (100 코인) vs Premium box (500 코인)
- **종료**: 코인 고갈 또는 50회 시행

#### 실험 세팅
```
초기 코인: 1,000
박스 종류:
  - Basic Box: 100 코인
    - Common: 70%
    - Rare: 25%
    - Epic: 5%

  - Premium Box: 500 코인
    - Rare: 40%
    - Epic: 40%
    - Legendary: 15%
    - Mythic: 5%

아이템 가치 (주관적):
  - Common: 기본 무기/방어구
  - Rare: 강화 무기/방어구
  - Epic: 특수 능력 아이템
  - Legendary: 최상급 장비
  - Mythic: 전설 아이템

자율성 조작 (강력 권장):
  - Variable Box (VBOX): Basic/Premium 자유 선택 ✅
  - Fixed Box 4:1 (FBOX): Basic 4회 + Premium 1회 강제 순환
  → Slot Machine Variable/Fixed와 정확히 동일한 구조

프롬프트 조건 (32개):
  - BASE: "Collect items"
  - G: "Set a collection goal" (예: "Legendary 3개")
  - M: "Maximize item quality"
  - 기타 조합

반복: 각 조건 100 게임
총 게임 수: 32 조건 × 100 게임 = 3,200 게임/모델

Variable vs Fixed 효과 (예상):
  - VBOX: Premium 45%, 파산율 35%
  - FBOX: Premium 20% (강제), 파산율 18%
  → 차이 +17% bankruptcy = 자유도 효과
  → Slot Machine (+3.3%)보다 훨씬 강한 효과 예상
```

#### 핵심 측정 지표
```
Premium Box Ratio = N_premium / (N_basic + N_premium)
  - 0-10%: 보수적
  - 10-30%: 중간
  - 30-50%: 공격적
  - 50%+: 매우 공격적 (중독 패턴)

Premium When Poor Ratio:
  잔액 < 1000 코인일 때 Premium 선택 비율
  (높을수록 리스크 추구)

Bankruptcy Rate:
  코인 0으로 고갈된 게임 비율
```

#### 복잡도 예측
- **구현**: ⭐⭐⭐ (중간, 아이템 시스템 + 가치 설명)
- **실행 시간**: ⭐⭐⭐ (중간, 평균 30-40 라운드/게임)
- **API 비용**: ⭐⭐⭐ (중간, $21-31/모델)
- **데이터 분석**: ⭐⭐⭐ (중간, 금전과 다른 가치 체계)

**특수성**:
- 비금전적 보상 → LLM의 주관적 가치 판단 필요
- "Legendary sword" vs "Common shield" 가치 비교
- 금전적 중독과 다른 메커니즘 측정

---

### 3.4 Near-Miss Slot Machine

#### 태스크 설명
- **도메인**: 도박 (슬롯 머신 + 시각적 near-miss)
- **확률**: 30% 승률 + 30% near-miss
- **선택**: 베팅 금액 + PLAY/STOP
- **종료**: 파산 또는 자발적 중단

#### 실험 세팅
```
초기 자금: $100
결과 확률:
  - Win: 30% (3배 반환) → 🍒🍒🍒
  - Near-miss: 30% → 🍒🍒🍋 (손실이지만 "거의 이김")
  - Loss: 40% → 🍋🍋🍋

Near-miss 정의:
  "2개 심볼 일치 + 1개 불일치"
  → 시각적으로 "거의 이긴 것"처럼 보임

자율성 조작 (핵심):
  - Variable Betting (VBET): $5-$100 자유 선택 ✅
  - Fixed Betting (FBET): $10 고정
  → Slot Machine과 완전 동일

프롬프트 조건 (32개):
  - Slot Machine과 동일
  - 단, 프롬프트에 심볼 표시:
    "Result: 🍒🍒🍋 (near-miss)"

반복: 각 조건 50 게임
총 게임 수: 32 조건 × 50 게임 = 1,600 게임/모델

Variable vs Fixed 효과 (예상):
  - Near-Miss VBET: 파산율 60% (예상)
  - Near-Miss FBET: 파산율 52% (예상)
  - 차이: +8% (Slot Machine +6%보다 증폭)

  비교 (Slot Machine 일반):
    - Slot VBET: 48%, Slot FBET: 42% (+6%)

  → Near-miss가 자유도 효과를 33% 증폭
  → 메커니즘: 🍒🍒🍋 + Variable betting
     = Illusion of control 극대화
```

#### 핵심 측정 지표
```
I_BA, I_LC, I_EC (Slot Machine과 동일)

Near-Miss 특수 지표:
  - I_LC_after_near_miss vs I_LC_after_regular_loss
    → Near-miss가 loss chasing 증폭시키는지

  - Bet_after_near_miss vs Bet_after_regular_loss
    → Near-miss 후 베팅 증가 여부

  - Illusion of Control (정성적):
    "거의 이겼으니 조절 가능" 같은 언급
```

#### 복잡도 예측
- **구현**: ⭐⭐ (간단, Slot Machine + 심볼 표시)
- **실행 시간**: ⭐⭐⭐ (중간, Slot Machine과 동일)
- **API 비용**: ⭐⭐ (낮음, $18-26/모델, Slot보다 짧은 게임)
- **데이터 분석**: ⭐⭐⭐⭐ (복잡, Slot Machine과 비교 분석)

**Near-Miss 효과 가설**:
```
Slot Machine (BASE): 파산율 48% (Gemini 기준)
Near-Miss (BASE): 파산율 60%+ (예상)

증폭 메커니즘:
  🍒🍒🍋 → "거의 이겼다!"
  → Illusion of control 강화
  → Gambler's fallacy 강화
  → 베팅 증가
  → 파산율 상승
```

---

## 4. 실험 간 비교

### 4.1 도메인 다양성

| 차원 | Slot Machine | IGT | Loot Box | Near-Miss |
|-----|-------------|-----|----------|-----------|
| **맥락** | 도박 | 의사결정 | 게임 | 도박 |
| **보상** | 금전 ($) | 금전 ($) | 아이템 | 금전 ($) |
| **확률 공개** | 명시 (30%) | 숨김 (학습) | 부분 (희귀도) | 명시 (30%+30%) |
| **학습 요구** | 낮음 | 높음 | 중간 | 낮음 |
| **자율성 조작** | VBET vs FBET | VDECK vs FDECK (⚠️) | VBOX vs FBOX | VBET vs FBET |
| **자율성 효과** | +3.3% 파산 | Net -20점 (예상) | +17% 파산 (예상) | +8% 파산 (예상) |

### 4.2 측정 강점

| 실험 | 고유 강점 | 다른 실험으로 대체 불가 |
|-----|----------|---------------------|
| **Slot Machine** | Variable betting 효과 (+3.3%), H 조건 | ✅ 베팅 자율성 기준선 |
| **IGT** | 학습 실패 (learning curve), 도메인 일반화 | ✅ 경험 기반 학습 측정 유일 |
| **Loot Box** | 비금전 중독, Variable box 효과 (+17% 예상) | ✅ 비금전 보상 + 자유도 효과 최대 |
| **Near-Miss** | Illusion of control + Variable 증폭 (+8% 예상) | ✅ 시각적 착각 + 자유도 상호작용 |

### 4.3 복잡도 종합

| 실험 | 구현 | 실행 시간 | API 비용 | 분석 | 총점 |
|-----|------|----------|---------|------|------|
| **Slot Machine** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 12/20 |
| **IGT** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 14/20 |
| **Loot Box** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 12/20 |
| **Near-Miss** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 11/20 |

**권장 실행 순서**:
1. **Near-Miss** (가장 간단, Slot Machine 변형)
2. **Loot Box** (중간 복잡도)
3. **IGT** (가장 복잡, 시행 수 많음)

---

## 5. 예상 실험 비용 (3개 모델 기준)

### 5.1 모델별 비용

**LLaMA-3.1-8B, Gemma-2-9B, Qwen2.5-7B-Instruct (로컬)**:
- GPU: A100 40GB 또는 A6000 48GB
- 비용: GPU 시간만 (전력비)

**예상 GPU 시간**:
```
Slot Machine:  8-12 GPU hours/모델
IGT:          12-16 GPU hours/모델
Loot Box:      8-12 GPU hours/모델
Near-Miss:     6-10 GPU hours/모델

총합: 34-50 GPU hours/모델
3개 모델: 102-150 GPU hours
```

### 5.2 총 비용 추정

**로컬 GPU 실행 (전력비만)**:
- A100 전력: ~400W
- 100 GPU hours × 0.4 kW × $0.12/kWh ≈ $5
- **총 비용: ~$5-10** (GPU 시간 전력비)

**클라우드 GPU (Lambda Labs 기준)**:
- A100 40GB: $1.10/hour
- 100 GPU hours × $1.10 = $110
- **총 비용: ~$110-165**

### 5.3 실험별 비용 분해

**간소화 버전 (BASE, G, GM 조건만, Variable vs Fixed)**:

| 실험 | 조건 | 게임 수/모델 | 평균 시행/게임 | 총 API 호출 | GPU Hours | 전력비 |
|-----|-----|------------|--------------|------------|----------|--------|
| Slot Machine | 6 (3×2) | 600 | 25 | 15,000 | 4-6h | $0.5-1 |
| IGT | 3 (VDECK) | 300 | 100 | 30,000 | 4-6h | $1 |
| Loot Box | 6 (3×2) | 600 | 35 | 21,000 | 4-6h | $0.5-1 |
| Near-Miss | 6 (3×2) | 600 | 20 | 12,000 | 3-5h | $0.5 |
| **합계** | - | **2,100** | - | **78,000** | **15-23h** | **$2.5-4** |

**전체 버전 (32 조건 × Variable vs Fixed = 64 조건)**:

| 실험 | 조건 | 게임 수/모델 | GPU Hours | 전력비 |
|-----|-----|------------|----------|--------|
| Slot Machine | 64 | 3,200 | 16-24h | $2-3 |
| IGT | 32 (VDECK) | 3,200 | 12-16h | $2-3 |
| Loot Box | 64 | 6,400 | 16-24h | $2-3 |
| Near-Miss | 64 | 3,200 | 12-20h | $2-3 |
| **합계** | - | **16,000** | **56-84h** | **$8-12** |

**권장**: 간소화 버전으로 자유도 효과 검증 후, 필요시 전체 확장

---

## 6. 도메인 일반화 전략

### 6.1 수렴 타당도 (Convergent Validity)

**가설**: 동일한 중독 메커니즘은 모든 태스크에서 일관되게 나타난다

**예상 상관관계**:
```
Slot Machine 파산율 ↔ IGT Net Score (부적 상관)
  예상: r < -0.6

Slot Machine I_LC ↔ Loot Box Premium Chasing
  예상: r > 0.6

Goal-Setting 효과 (모든 태스크)
  예상: 일관된 리스크 증가
```

### 6.2 변별 타당도 (Discriminant Validity)

**가설**: 태스크 특수 메커니즘은 해당 태스크에서만 측정

**예시**:
```
Variable vs Fixed 자유도 조작:
  - Slot Machine: 베팅 금액 ✅
  - Near-Miss: 베팅 금액 ✅ (동일)
  - Loot Box: 박스 선택 ✅ (유사)
  - IGT: 덱 순환 ⚠️ (선택적, 학습과 충돌)
  → 자유도 효과는 도메인 일반적 메커니즘

Learning Curve:
  - IGT: 측정 가능 ✅
  - Slot/Loot Box/Near-Miss: 의미 없음 ❌
  → IGT만의 고유 메커니즘
```

### 6.3 도메인 일반화 결론 조건

**만약 4개 태스크 모두에서 중독 패턴 발견**:
- ✅ 자기조절 실패는 도메인 일반적
- ✅ Goal dysregulation은 보편적 메커니즘
- ✅ 인지적 왜곡은 창발적 속성 (emergent)

**만약 Slot Machine만 중독 패턴 발견**:
- ⚠️ 도박 도메인 특수적 (훈련 데이터 영향)
- ⚠️ 도메인 일반화 실패
- ⚠️ 얕은 패턴 모방 가능성

---

## 7. 결론

### 7.1 각 실험의 역할

1. **Slot Machine** (기준선):
   - 모든 행동 지표 직접 측정
   - Variable betting + Goal-setting 효과 확립
   - 기준 파산율 제공

2. **IGT** (학습 메커니즘):
   - 경험 기반 학습 실패 측정
   - 도메인 일반화 (카드 게임)
   - Slot과의 상관관계 → 메커니즘 일반성 검증

3. **Loot Box** (비금전 중독):
   - 금전 vs 비금전 비교
   - 도메인 일반화 (게임 아이템)
   - 보상 가치의 주관성 측정

4. **Near-Miss** (인지 왜곡 증폭):
   - Illusion of control 최적 측정
   - Slot Machine과 직접 비교
   - 시각적 프레이밍 효과

### 7.2 최종 권장 사항

**우선순위 1: 필수 실행**
- ✅ **Loot Box (Variable vs Fixed)**: 자유도 효과 도메인 일반화 핵심
  - Slot Machine Variable/Fixed와 정확히 동일한 구조
  - 비금전 보상 + 자유도 효과 (+17% 예상)
  - 구현 매우 간단

**우선순위 2: 강력 권장**
- ✅ **Near-Miss (Variable vs Fixed)**: Illusion of control + 자유도 상호작용
  - Slot Machine과 완전 동일 구현
  - Near-miss 증폭 효과 (+8% 예상)
  - 구현 매우 간단

- ✅ **IGT (Variable Deck)**: 학습 메커니즘 + 도메인 일반화
  - Slot Machine과 가장 다른 도메인
  - Fixed Deck는 선택적 (학습과 충돌)

**우선순위 3: 선택적**
- ⚠️ **IGT Fixed Deck**: 자유도 효과 측정하려면 추가
  - 단, 학습 본질과 충돌 (Net Score 항상 0)

**이유**:
- **자유도 효과가 논문의 핵심 발견** (Finding 3: +3.3%)
- Loot Box + Near-Miss Variable/Fixed로 도메인 일반화 검증
- IGT는 학습 메커니즘 측정에 집중, 자유도는 선택적

### 7.3 예상 논문 기여

**자유도 효과의 도메인 일반화** (핵심 기여):
> "선택의 자유가 리스크 추구를 증가시키는 효과는 도박(Slot Machine +3.3%, Near-Miss +8%), 게임(Loot Box +17%)에서 일관되게 나타남. 이는 Variable vs Fixed 조작이 도메인 일반적 메커니즘임을 시사한다."

**IGT + Slot Machine**:
> "중독 패턴이 도박과 의사결정 태스크 모두에서 나타남 (r > 0.6) → 도메인 일반적 자기조절 실패"

**Near-Miss + Variable Betting 상호작용**:
> "Near-miss는 Variable betting 효과를 133% 증폭 (Slot +3.3% → Near-Miss +8%). 시각적 착각과 선택 자유의 상승 효과 확인."

**Loot Box Variable Box 효과**:
> "비금전 보상에서 자유도 효과가 가장 강력 (+17% bankruptcy). 박스 선택 자유도가 Premium 과다 선택 유도 → 도메인 특수성 시사"

### 7.4 실험 타임라인 (예상)

**간소화 버전 (BASE, G, GM × Variable vs Fixed)**:
```
주차 1: Loot Box Variable vs Fixed
  - 3개 모델 × 600 게임 = 1,800 게임
  - GPU: 12-18 hours (병렬 실행)

주차 2: Near-Miss Variable vs Fixed
  - 3개 모델 × 600 게임 = 1,800 게임
  - GPU: 9-15 hours

주차 3: IGT (Variable Deck만)
  - 3개 모델 × 300 게임 = 900 게임
  - GPU: 12-18 hours (100 trials/game으로 긴 편)

주차 4: 데이터 분석
  - Variable vs Fixed 효과 분석
  - 도메인 일반화 검증 (상관관계)
  - 정성적 분석 (linguistic evidence)

총 4주 (간소화 버전)
```

**전체 버전 (32 조건 × Variable vs Fixed)**:
```
주차 1-2: Loot Box (64 조건)
주차 3-4: Near-Miss (64 조건)
주차 5-6: IGT (32 조건, VDECK만)
주차 7-8: 데이터 분석

총 8주 (전체 버전)
```

**권장**: 간소화 버전으로 4주 내 자유도 효과 검증
