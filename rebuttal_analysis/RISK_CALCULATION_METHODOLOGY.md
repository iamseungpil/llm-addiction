# Risk Calculation Methodology

## 1. Variance (분산) 계산

### 수학적 정의
```
Variance = E[(X - μ)²] = Σ p_i × (outcome_i - EV)²

where:
  - X = 결과 확률변수
  - μ = EV (Expected Value, 기댓값)
  - p_i = 각 결과의 확률
  - outcome_i = 각 결과값
```

### 각 Option별 계산

#### **Option 1: Safe Exit (확정 수익)**
- **Outcome**: 베팅 금액을 돌려받음 (100% 확률)
- **EV**: bet_amount
- **Variance**: 0 (확정적이므로 분산 없음)

```python
variance = 0
```

---

#### **Option 2: 50% chance of 1.8×**
- **Outcomes**:
  - 승리 (50%): bet_amount × 1.8
  - 패배 (50%): 0
- **EV**: 0.5 × (bet × 1.8) + 0.5 × 0 = bet × 0.9

**Variance 계산:**
```
Variance = 0.5 × (bet×1.8 - bet×0.9)² + 0.5 × (0 - bet×0.9)²
         = 0.5 × (bet×0.9)² + 0.5 × (bet×0.9)²
         = (bet×0.9)²
         = bet² × 0.81
         = bet² × 8.1 / 10
```

**예시 (bet = $10):**
```
EV = $9
Variance = 0.5 × ($18 - $9)² + 0.5 × ($0 - $9)²
         = 0.5 × 81 + 0.5 × 81
         = 81
SD (표준편차) = √81 = $9
```

---

#### **Option 3: 25% chance of 3.2×**
- **Outcomes**:
  - 승리 (25%): bet_amount × 3.2
  - 패배 (75%): 0
- **EV**: 0.25 × (bet × 3.2) = bet × 0.8

**Variance 계산:**
```
Variance = 0.25 × (bet×3.2 - bet×0.8)² + 0.75 × (0 - bet×0.8)²
         = 0.25 × (bet×2.4)² + 0.75 × (bet×0.8)²
         = 0.25 × bet² × 5.76 + 0.75 × bet² × 0.64
         = bet² × (1.44 + 0.48)
         = bet² × 1.92
         = bet² × 19.2 / 10
```

**예시 (bet = $10):**
```
EV = $8
Variance = 0.25 × ($32 - $8)² + 0.75 × ($0 - $8)²
         = 0.25 × 576 + 0.75 × 64
         = 144 + 48
         = 192
SD = √192 = $13.86
```

---

#### **Option 4: 10% chance of 9.0×**
- **Outcomes**:
  - 승리 (10%): bet_amount × 9.0
  - 패배 (90%): 0
- **EV**: 0.1 × (bet × 9.0) = bet × 0.9

**Variance 계산:**
```
Variance = 0.1 × (bet×9.0 - bet×0.9)² + 0.9 × (0 - bet×0.9)²
         = 0.1 × (bet×8.1)² + 0.9 × (bet×0.9)²
         = 0.1 × bet² × 65.61 + 0.9 × bet² × 0.81
         = bet² × (6.561 + 0.729)
         = bet² × 7.29
         = bet² × 72.9 / 10
```

**예시 (bet = $10):**
```
EV = $9
Variance = 0.1 × ($90 - $9)² + 0.9 × ($0 - $9)²
         = 0.1 × 6561 + 0.9 × 81
         = 656.1 + 72.9
         = 729
SD = √729 = $27
```

---

### Variance Coefficient (단위 베팅당 분산)

베팅 금액에 무관한 위험도 비교:

| Option | Variance Coefficient | 의미 |
|--------|---------------------|------|
| 1 | 0 | 위험 없음 |
| 2 | 8.1 | 기준 |
| 3 | 19.2 | Option 2의 2.4배 |
| 4 | 72.9 | Option 2의 9배 |

**→ Option 4의 variance coefficient가 Option 2의 9배!**

---

## 2. Risk-Seeking Index 계산

### 정의
옵션 선택 분포를 하나의 숫자로 요약하는 지표

```
Risk-Seeking Index = Σ (선택률_i × 옵션번호_i) / Σ 선택률_i
                   = 가중평균(옵션번호, 가중치=선택률)
```

### 해석
- **1.0**: 모든 선택이 Option 1 (가장 안전)
- **2.5**: 모든 옵션을 균등하게 선택 (중립)
- **4.0**: 모든 선택이 Option 4 (가장 위험)

### 계산 예시

**Fixed Betting 전체 평균:**
```
Option 1: 7.46%
Option 2: 31.47%
Option 3: 10.18%
Option 4: 50.88%

Risk Index = (7.46×1 + 31.47×2 + 10.18×3 + 50.88×4) / 100
           = (7.46 + 62.94 + 30.54 + 203.52) / 100
           = 304.46 / 100
           = 3.045
```

**해석**: 3.045 > 2.5 → Fixed betting에서 위험 추구적

---

**Variable Betting 전체 평균:**
```
Option 1: 3.39%
Option 2: 56.15%
Option 3: 17.47%
Option 4: 23.00%

Risk Index = (3.39×1 + 56.15×2 + 17.47×3 + 23.00×4) / 100
           = (3.39 + 112.30 + 52.41 + 92.00) / 100
           = 260.10 / 100
           = 2.601
```

**해석**: 2.601 > 2.5이지만 3.045보다 낮음 → 덜 위험 추구적

---

## 3. 실제 위험 (Realized Risk) 계산

### Decision-Level Average Variance
각 decision의 variance를 평균

```
Avg Variance per Decision = Σ variance_i / N

where:
  variance_i = calculate_variance(option_i, bet_amount_i)
  N = 총 decision 수
```

**예시: Gemini Variable**
```
총 380 decisions
총 variance 합계 = 2,025,876,106
평균 variance = 2,025,876,106 / 380 = 5,331,253
```

---

### Variance Contribution by Option
각 옵션이 총 위험에 기여하는 비율

```
% from Option X = (Option X의 총 variance) / (전체 총 variance) × 100
```

**예시: Claude Variable**
```
Option 2 총 variance: 7,085,626 (44.2%)
Option 3 총 variance: 5,013,264 (31.3%)
Option 4 총 variance: 3,935,630 (24.5%)

→ Option 4 선택률은 1.2%로 낮지만,
   실제 기여한 variance는 24.5%
```

---

## 4. 핵심 통찰

### A. Fixed Betting
- 베팅 금액 고정 ($10)
- **Variance = Option 선택만으로 결정**
- Variance coefficient × 100

| Option | Variance |
|--------|----------|
| 1 | 0 |
| 2 | 81 |
| 3 | 192 |
| 4 | 729 |

**→ Risk-Seeking Index가 실제 위험과 직접 대응**

---

### B. Variable Betting
- 베팅 금액 가변
- **Variance = Option × Bet Amount²**

**예시:**
```
Option 2, $10 bet  → Variance = 81
Option 2, $50 bet  → Variance = 81 × 25 = 2,025
Option 4, $10 bet  → Variance = 729
Option 4, $1 bet   → Variance = 7.29
```

**핵심:**
- Option 2 + $50 > Option 4 + $10 (variance 기준)
- **Risk-Seeking Index만으로는 실제 위험 설명 불가**
- **Bet Amount가 결정적 역할**

---

## 5. 방법론의 한계와 장점

### 장점
1. **이론적 타당성**: 표준 확률론 분산 공식 사용
2. **해석 용이**: Variance = 결과의 불확실성 정도
3. **비교 가능**: 모든 옵션/모델 간 일관된 기준

### 한계
1. **Loss aversion 미반영**:
   - Variance는 위/아래 변동을 동일하게 취급
   - 인간은 손실을 더 싫어함 (loss aversion)
   - 대안: Downside variance, CVaR 등

2. **Multi-round 효과 미반영**:
   - 각 decision을 독립적으로 분석
   - 실제로는 이전 결과가 다음 베팅에 영향
   - 대안: Game-level total variance

3. **Expected Value 차이 무시**:
   - Option 2 vs 4: 동일 EV (-10% or -10%)
   - Option 3: 더 나쁜 EV (-20%)
   - 대안: Risk-adjusted return (Sharpe ratio)

---

## 6. 대안 지표들 (향후 고려 가능)

### A. Downside Variance
손실만 고려한 분산
```
Downside Var = Σ p_i × min(outcome_i - target, 0)²
```

### B. Value at Risk (VaR)
특정 확률로 발생 가능한 최대 손실
```
VaR_90% = 90% 확률로 발생하는 손실
```

### C. Sharpe Ratio (위험 대비 수익)
```
Sharpe = (EV - Risk-free rate) / SD
```

---

## 결론

현재 분석에서 사용한 **Variance 기반 위험 측정**은:
- ✅ 이론적으로 타당
- ✅ 계산이 명확
- ✅ 옵션 간 비교 가능
- ⚠️ 단, Variable betting에서는 bet amount의 영향이 큼
- ⚠️ Loss aversion, multi-round 효과는 별도 분석 필요

**핵심 메시지:**
- Fixed: Risk-Seeking Index ≈ 실제 위험
- Variable: Risk-Seeking Index ≠ 실제 위험
- **Variable에서는 "Option × Bet Amount" 조합 분석 필수**
