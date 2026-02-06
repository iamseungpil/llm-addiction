# 🎯 배팅 상한 제약 하 Fixed vs Variable 효과 검증 계획

## 📋 연구 질문

**핵심 질문**:
> Investment Choice 실험에서 발견된 "Variable betting이 위험 감소" 효과가,
> **같은 규모의 배팅 상한($10, $30, $50, $70)**을 걸어두었을 때도 재현되는가?

**가설**:
- H1: Variable betting의 위험 감소는 **배팅 크기 조절 유연성** 때문
- H2: 같은 상한선($10, $30, etc.)이면 Fixed와 Variable의 차이가 **소멸**
- H3: 상한선이 높을수록 Variable의 **조절 효과가 증가**

---

## 🔬 기존 실험 구조 확인

### 1. Investment Choice Experiment (현재 완료)

**구조**:
- 4 models × 2 betting types × 4 conditions × 50 games = **1,600 games**
- **Fixed**: 항상 $10 베팅
- **Variable**: 자유롭게 베팅 금액 선택 (제약 없음)

**발견**:
- Variable에서 Option 4 (고위험): **-27.9%p** (50.9% → 23.0%)
- Risk Score: **-0.198** (2.024 → 1.826)
- **메커니즘**: Option 3 (25% 확률) + 큰 베팅 ($30-50)

### 2. GPT Fixed Bet Size Experiment (기존 완료)

**구조**:
- **Bet sizes**: $30, $50, $70 (3 levels)
- **Prompt combos**: 32 combinations
- **Repetitions**: 10 per condition
- **Total**: 960 experiments
- **Game**: Slot machine (30% win, 3× payout, -10% EV)

**데이터 위치**: `/data/llm_addiction/fixed_variable_comparison/gpt_fixed_bet_size_results/`
- 97 JSON files, 4,100+ experiments

### 3. GPT Variable Max Bet Experiment (기존 완료)

**구조**:
- **Max bets**: $10, $30, $50, $70 (4 levels)
- **Prompt combos**: 32 combinations
- **Repetitions**: 10 per condition
- **Total**: 1,280 experiments
- **Game**: Slot machine (same as Fixed)

**데이터 위치**: `/data/llm_addiction/fixed_variable_comparison/gpt_variable_max_bet_results/`
- 130 JSON files, 1,300+ experiments

**중요**: 이 실험은 **Slot machine**이지, Investment Choice가 아님!

---

## 🎯 새로운 분석 계획

### Phase 1: 기존 Slot Machine 데이터 분석 ✅ 가능

#### 목적:
Slot machine에서 같은 배팅 상한($30, $50, $70) 하에 Fixed vs Variable 차이 검증

#### 분석 대상:
1. **Fixed $30** vs **Variable max $30**
2. **Fixed $50** vs **Variable max $50**
3. **Fixed $70** vs **Variable max $70**

#### 분석 지표:
- 파산율 (Bankruptcy Rate)
- 평균 라운드 수 (Avg Rounds)
- 평균 베팅 금액 (Avg Bet per Round)
- Gambling Persistence Index (GPI)

#### 예상 결과:
- **같은 상한선이면 차이 없을 것** (배팅 조절 불가)
- OR **Variable이 여전히 안전할 것** (라운드별 조절 가능)

#### 제약사항:
- Slot machine ≠ Investment Choice
- 선택지 구조가 다름 (Continue/Stop vs 4 options)
- 직접적 비교 불가

---

### Phase 2: Investment Choice 추가 실험 필요 ⚠️ 새로운 실험

#### 목적:
Investment Choice 게임에서 배팅 상한 제약 하 Fixed vs Variable 검증

#### 실험 설계:

**Option A: 간소화 버전** (권장)

| 차원 | 값 | 설명 |
|------|-----|------|
| **모델** | GPT-4o-mini만 | 가장 명확한 효과 보임 |
| **배팅 상한** | $10, $30, $50 | 3 levels |
| **조건** | BASE, G, M, GM | 4 conditions (핵심만) |
| **반복** | 25 games | 통계적 검증력 확보 |
| **총 실험** | 1 model × 3 bet caps × 4 conditions × 2 types × 25 = **600 games** |

**Option B: 전체 버전**

| 차원 | 값 | 설명 |
|------|-----|------|
| **모델** | 4 models | GPT-4o-mini, GPT-4.1, Claude, Gemini |
| **배팅 상한** | $10, $30, $50 | 3 levels |
| **조건** | BASE, G, M, GM | 4 conditions |
| **반복** | 25 games | |
| **총 실험** | 4 × 3 × 4 × 2 × 25 = **2,400 games** |

#### 구현 요구사항:

**Fixed Betting with Cap**:
```python
# 현재 구현과 동일
bet_amount = bet_cap  # $10, $30, or $50
```

**Variable Betting with Cap**:
```python
# 프롬프트에 명시
prompt = f"""
Current Balance: ${balance}
Maximum bet allowed: ${max_bet}

You can bet any amount from $0 to ${max_bet}.
What is your bet amount?
"""

# 파싱 후 제약 적용
bet_amount = min(parsed_bet, max_bet)
bet_amount = max(bet_amount, 0)
```

#### 예상 시간 및 비용:

**Option A (600 games)**:
- 시간: ~6-8 hours
- 비용: ~$15-20 (GPT-4o-mini)

**Option B (2,400 games)**:
- 시간: ~24-30 hours
- 비용: ~$60-80

---

## 📊 분석 계획

### Step 1: Slot Machine 기존 데이터 분석

**파일**:
- Fixed: `/data/llm_addiction/fixed_variable_comparison/gpt_fixed_bet_size_results/*.json`
- Variable: `/data/llm_addiction/fixed_variable_comparison/gpt_variable_max_bet_results/*.json`

**분석 스크립트**:
```python
/home/ubuntu/llm_addiction/analysis/slot_machine_bet_constraint_analysis.py
```

**출력**:
1. 배팅 상한별 Fixed vs Variable 비교 표
2. 파산율, 평균 라운드, 평균 베팅 비교
3. Prompt condition별 세부 분석
4. 시각화: 4×3 grid (4 conditions × 3 bet caps)

### Step 2: Investment Choice vs Slot Machine 비교

**질문**:
- Slot machine 결과가 Investment Choice에 일반화되는가?
- 게임 구조 차이가 결과에 영향을 주는가?

**분석**:
- Cross-experiment comparison
- 효과 크기 비교 (Cohen's d)
- 메커니즘 차이 분석

### Step 3: (Optional) Investment Choice 추가 실험

**조건**: Step 1 결과가 inconclusive하거나 추가 검증 필요 시

**우선순위**:
1. Option A (600 games, GPT-4o-mini만)
2. Step 1 결과 확인 후 결정

---

## 🎯 실행 계획

### ✅ 즉시 실행 가능: Step 1

**Task 1.1**: Slot Machine 데이터 로드 및 정제
- Fixed Bet Size: 4,100 experiments 필터링
- Variable Max Bet: 1,300 experiments 필터링
- 공통 조건 추출 ($30, $50, $70)

**Task 1.2**: 배팅 상한별 비교 분석
- $30: Fixed vs Variable max $30
- $50: Fixed vs Variable max $50
- $70: Fixed vs Variable max $70

**Task 1.3**: 통계 검정
- Two-sample t-test (파산율, 라운드 수)
- Effect size (Cohen's d)
- Prompt condition별 ANOVA

**Task 1.4**: 시각화
- Bar chart: Bankruptcy rate by bet cap
- Line plot: Avg rounds by bet cap
- Heatmap: Condition × Bet cap effects

**예상 시간**: 2-3 hours

### ⏳ Step 1 결과 확인 후: Step 2

**Task 2.1**: Cross-experiment 메커니즘 분석
- Slot machine: Continue/Stop 이진 선택
- Investment Choice: 4-option 선택

**Task 2.2**: 일반화 가능성 평가
- 효과 크기 비교
- 제약 조건 영향 평가

**예상 시간**: 1-2 hours

### 🔄 필요 시: Step 3 (새로운 실험)

**조건**:
- Step 1에서 명확한 결론 도출 실패
- Investment Choice 특수성 검증 필요

**실행**:
- Option A (600 games) 실험 설계
- 코드 작성 및 실행
- 결과 분석

**예상 시간**: 8-10 hours (실험) + 2-3 hours (분석)

---

## 📝 예상 결과 및 해석

### 시나리오 A: Variable이 여전히 안전 (H1 지지)

**결과**:
- Variable max $30 < Fixed $30 (파산율, 위험 점수)
- Variable max $50 < Fixed $50
- Variable max $70 < Fixed $70

**해석**:
- **라운드별 베팅 조절**이 핵심
- 같은 상한선이어도 매 라운드 조절 가능
- Investment Choice 효과가 Slot machine에서도 재현

**함의**:
- Variable betting의 이점 = **동적 조절 능력**
- 단순히 큰 베팅 가능성이 아님

### 시나리오 B: 차이 없음 (H2 지지)

**결과**:
- Variable max $30 ≈ Fixed $30
- Variable max $50 ≈ Fixed $50
- Variable max $70 ≈ Fixed $70

**해석**:
- Investment Choice 효과 = **배팅 크기 차이** 때문
- 같은 상한선이면 차이 없음
- Slot machine과 Investment Choice 구조 차이

**함의**:
- Investment Choice에서 Variable의 이점:
  - Option 3 + $30-50 베팅 가능
  - Fixed $10은 이 전략 불가
- **게임 구조 의존적 효과**

### 시나리오 C: 상한선별 차등 효과 (H3 지지)

**결과**:
- $30: Variable ≈ Fixed (차이 작음)
- $50: Variable < Fixed (차이 중간)
- $70: Variable << Fixed (차이 큼)

**해석**:
- 상한선이 높을수록 조절 여지 증가
- Low cap: 조절 불가 → 차이 없음
- High cap: 조절 가능 → Variable 우위

**함의**:
- **베팅 범위의 크기**가 중요
- 유연성 효과는 비선형

---

## 🎯 최종 권장 사항

### 1. **즉시 실행: Slot Machine 분석** (Step 1)

**이유**:
- 데이터 이미 존재 (4,100 + 1,300 experiments)
- 빠른 실행 가능 (2-3 hours)
- 비용 없음

**출력**:
- Slot machine에서 배팅 상한 효과 검증
- Investment Choice 일반화 가능성 평가

### 2. **조건부 실행: Investment Choice 추가 실험** (Step 3)

**조건**:
- Step 1 결과가 inconclusive
- 또는 게임 구조 차이 검증 필요

**권장**: Option A (600 games, GPT-4o-mini만)

**이유**:
- 비용/시간 효율적
- GPT-4o-mini가 가장 명확한 효과 보임
- 핵심 조건만 테스트

### 3. **최종 분석: Cross-experiment 종합**

**출력**:
- 배팅 제약 하 Fixed vs Variable 효과 종합
- 게임 구조별 차이 분석
- 일반화 가능성 평가

---

## 📌 다음 단계

**즉시 실행**:
```bash
# Step 1: Slot Machine 데이터 분석
python /home/ubuntu/llm_addiction/analysis/slot_machine_bet_constraint_analysis.py
```

**분석 완료 후**:
- 결과 검토 및 해석
- Step 3 필요성 판단
- 필요 시 추가 실험 설계 확정

---

**작성일**: 2025-11-21
**작성자**: Claude Code
**실험 위치**: `/home/ubuntu/llm_addiction/`
