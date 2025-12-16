# Variable vs Fixed Betting Risk Analysis Proposal

## 제안하는 분석 방향들

---

## **분석 A: Option 선택 분포 분석**

### 목적
각 모델이 어떤 Option을 얼마나 선택하는지 파악

### 분석 내용
- Option 1/2/3/4 선택 비율 (모델별, 베팅 타입별)
- 프롬프트 조건별 차이 (BASE/G/M/GM)
- "안전 탈출(Option 1)" vs "위험 추구(Option 2-4)" 경향

### 시각화
- Stacked bar chart: 모델별 × 베팅타입별 Option 분포
- Heatmap: 모델 × 프롬프트 조건 × Option 선택률

### 한계
- Variable betting에서 "Bet Amount"를 고려하지 않음
- Option 선택만으로는 실제 위험을 완전히 설명 못함

---

## **분석 B: Bet Amount 분포 분석 (Variable Betting Only)**

### 목적
Variable betting에서 실제로 얼마를 베팅하는지 분포 파악

### 분석 내용
1. **절대 금액 분포**
   - 평균/중간값/표준편차 베팅 금액
   - 최소/최대 베팅 금액

2. **잔액 대비 비율 분포**
   - Bet / Balance 비율 계산
   - "보수적 (<20%)" vs "공격적 (>50%)" vs "올인 (100%)" 비율

3. **Option별 베팅 금액**
   - Option 4 선택 시 평균 베팅 금액
   - Option 2 선택 시 평균 베팅 금액
   - 높은 위험 옵션 + 높은 베팅 금액 조합 여부

### 시각화
- Histogram: 베팅 금액 분포
- Box plot: 모델별 × Option별 베팅 금액
- Scatter plot: Balance vs Bet Amount (각 decision)

### 핵심 질문
- "Gemini가 Option 4를 많이 선택하는데, 작은 금액만 베팅하는가?"
- "Claude가 Option 4를 거의 안 선택하는데, 다른 옵션에 큰 금액 베팅하는가?"

---

## **분석 C: 실제 위험 계산 (Realized Variance)**

### 목적
각 decision의 **실제 분산**을 계산하여 진짜 위험 비교

### 분석 내용
1. **Decision-level variance**
   ```
   For each decision:
     variance = calculate_variance(option, bet_amount)

   Option 2, $10 bet → variance = 81
   Option 4, $10 bet → variance = 729
   Option 2, $50 bet → variance = 2,025
   Option 4, $1 bet → variance = 7.29
   ```

2. **Game-level total variance**
   - 한 게임에서 모든 라운드의 variance 합산
   - 모델별 평균 total variance 비교

3. **Risk-adjusted metrics**
   - Variance per dollar bet
   - Sharpe ratio (if possible)

### 시각화
- Distribution plot: 각 모델의 decision variance 분포
- Box plot: Game-level total variance 비교
- Scatter: Option 4 선택률 vs Actual variance

### 핵심 질문
- "Option 4 선택률과 실제 위험(variance)이 일치하는가?"
- "Claude의 낮은 Option 4 선택률이 실제로 낮은 위험을 의미하는가?"

---

## **분석 D: Context-Dependent Betting (승패 후 행동 변화)**

### 목적
이전 라운드 결과가 다음 베팅에 미치는 영향

### 분석 내용
1. **승리 후 (Win) vs 패배 후 (Loss)**
   - Win 후 베팅 금액 변화
   - Loss 후 베팅 금액 변화
   - Win 후 Option 선택 변화
   - Loss 후 Option 선택 변화

2. **Loss-chasing 행동**
   - 패배 후 베팅 금액 증가하는가?
   - 패배 후 더 위험한 Option 선택하는가?
   - Variable betting에서만 가능한 "올인" 행동

3. **Winning streak vs Losing streak**
   - 연속 승리/패배 시 베팅 패턴 변화

### 시각화
- Before-after plot: Win/Loss 전후 베팅 금액
- Violin plot: Win context vs Loss context 베팅 분포
- Line plot: Streak length vs Average bet amount

### 핵심 질문
- "Gambler's fallacy: 연속 패배 후 더 큰 베팅?"
- "Hot hand fallacy: 연속 승리 후 더 큰 베팅?"

---

## **분석 E: 잔액 반응성 (Balance Reactivity)**

### 목적
현재 잔액이 베팅 행동에 미치는 영향

### 분석 내용
1. **Bet / Balance 비율 추이**
   - 잔액이 증가하면 비율이 어떻게 변하는가?
   - 잔액이 감소하면 비율이 어떻게 변하는가?

2. **절망 베팅 (Desperation betting)**
   - 잔액 < $30 일 때 올인 비율
   - 잔액이 낮아질수록 위험 증가하는가?

3. **House money effect**
   - 초기 잔액($100) 이상일 때 vs 이하일 때
   - "번 돈"으로 더 공격적 베팅하는가?

### 시각화
- Scatter plot: Balance vs Bet/Balance ratio (각 decision)
- Line plot: Balance quantiles vs Average bet ratio
- Histogram: 잔액 구간별 베팅 행동

### 핵심 질문
- "잔액이 증가하면 더 공격적 베팅 (house money effect)?"
- "잔액이 감소하면 절망적 올인 (desperation)?"

---

## **분석 F: 통합 위험 프로파일 (Comprehensive Risk Profile)**

### 목적
여러 지표를 종합하여 각 모델의 "위험 프로파일" 생성

### 분석 내용
1. **Multi-dimensional risk metrics**
   - Option 4 선택률 (preference for high variance)
   - Average bet amount (capital at risk)
   - Bet/Balance ratio (proportional risk)
   - Realized variance (actual risk)
   - Loss-chasing intensity (context reactivity)
   - Bankruptcy rate (ultimate outcome)

2. **Risk score 계산**
   - 각 지표를 normalize (0-1)
   - Weighted sum으로 종합 risk score
   - 모델별 ranking

3. **Fixed vs Variable 비교**
   - 같은 모델이 베팅 타입에 따라 위험이 어떻게 변하는가?
   - Variable이 실제로 더 위험한가?

### 시각화
- Radar chart: 각 모델의 6개 위험 지표
- Heatmap: 모델 × 베팅타입 × 위험 지표
- Ranking table: 종합 위험 순위

### 핵심 질문
- "어떤 모델이 가장 위험한가? (종합적으로)"
- "Variable betting이 Fixed보다 정말 위험한가?"
- "Option 4 선택률이 실제 위험의 좋은 proxy인가?"

---

## **제안 요약**

| 분석 | 초점 | 장점 | 단점 |
|------|------|------|------|
| **A. Option 분포** | 선택 경향 | 간단, 직관적 | Variable에서 불완전 |
| **B. Bet Amount** | 베팅 금액 | Variable의 핵심 | Fixed에 적용 불가 |
| **C. Realized Variance** | 실제 위험 | 가장 정확한 위험 측정 | 계산 복잡 |
| **D. Context-Dependent** | 맥락 반응성 | Loss-chasing 발견 | 인과관계 약함 |
| **E. Balance Reactivity** | 잔액 의존성 | House money effect | 상관관계만 |
| **F. 통합 프로파일** | 종합 평가 | 전체 그림 | 지표 선택 주관적 |

---

## **제 추천**

사용자의 목적에 따라 다릅니다:

1. **"실제로 어떤 모델이 가장 위험한가?"** → **분석 C + F**
2. **"Variable betting의 특성 이해"** → **분석 B + E**
3. **"Option 4 선택률의 타당성 검증"** → **분석 A + C**
4. **"인지 편향 패턴 발견"** → **분석 D + E**

또는 여러 분석을 순차적으로 진행할 수도 있습니다.

어떤 분석 방향이 가장 관심 있으신지 말씀해주시면, 그 방향으로 깊게 파고들겠습니다!
