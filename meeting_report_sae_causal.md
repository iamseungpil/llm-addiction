# SAE Condition Comparison 분석 및 인과관계 실험 제안

## 연구 질문: 자율성이 파산에 영향을 주는가?

**행동 데이터 (상관관계)**
- Variable betting → 파산율 2.6배 높음 (LLaMA: 6.8% vs 2.6%, Gemma: 29.1% vs 12.8%)
- **한계**: 상관관계일 뿐, 인과관계 불명확

**SAE Condition Comparison 분석 완료** (Variable vs Fixed 베팅 조건)
- 11,999개 유의한 features 발견 (Cohen's d ≥ 0.3, FDR corrected)
- Variable-associated features가 신경 수준에서 존재함을 확인
- **한계**: 상관관계만 확인, 인과관계 검증 없음

## 기존 성공한 인과 실험 (Experiment 2)

**Experiment 2 결과** - Outcome 차원 (Safe vs Risky)
- **방법**: SAE-level patching (feature 값 직접 설정)
- **성공**: 361 safe features (+29.6% stopping), 80 risky features (-6.4% stopping)
- **프롬프트**: 2개 고정 시나리오 (safe_prompt, risky_prompt)
- **측정**: Betting amount, stopping rate

## 제안: Experiment 2 방법론을 Variable/Fixed 차원에 적용

**Experiment 2와의 비교**

| 측면 | Experiment 2 (기존) | 제안 실험 (신규) |
|------|-------------------|---------------|
| **Patching 방식** | SAE-level (동일) | SAE-level (동일) |
| **Feature 구분** | Outcome (Safe/Risky) | **Bet Type (Variable/Fixed)** |
| **Patch 값** | Population mean | Population mean (동일) |
| **프롬프트** | 2개 고정 시나리오 | **3200개 실제 게임** |
| **Feature 수** | 361 + 80 = 441개 | **11,999개** (더 많음) |

**4가지 실험 (Experiment 2 방법론 적용)**

### 실험 1: Direct Manipulation
- **가설**: Variable feature mean 주입 → Fixed 게임에서 Variable-like 행동
- **방법**: Fixed 게임 프롬프트 + Variable feature mean 값으로 SAE patching
- **측정**: Stop rate 감소? Reasoning에서 더 큰 베팅 언급?

### 실험 2: Cross-Condition Transfer
- **가설**: Fixed feature mean 주입 → Variable-Bankrupt 게임이 안전하게 변화
- **방법**: Variable-Bankrupt 게임(108개) + Fixed-Safe feature mean 주입
- **측정**: 파산율 감소? (108 → ~50개?), 베팅 금액 감소?

### 실험 3: Multi-Feature Intervention
- **가설**: 여러 features 동시 주입 → 더 강한 효과
- **방법**: Top 5 Variable features를 Fixed-Safe 게임에 동시 주입
- **측정**: 단일 feature보다 stop rate 더 큰 감소?

### 실험 4: Ablation
- **가설**: Variable features 제거 → Variable 게임이 Fixed-like 행동
- **방법**: Variable 게임에서 Top 10 Variable features를 0으로 설정
- **측정**: 파산율 감소? (6.8% → 3%?), 평균 베팅 금액 감소?

**기대 효과**: Variable-associated features가 실제로 Variable-like 행동을 유발하는지 검증 (Experiment 2 성공 재현)

---

## 왜 이 제안이 "자율성 → 파산" 인과관계를 보이는가?

### 인과 경로 (Causal Chain)

```
1. Neural level:   Variable features 존재 (발견 완료)
                           ↓
2. Behavioral level: Variable features → Variable-like 행동 (우리 실험으로 검증)
                           ↓
3. Outcome level:   Variable 행동 → 파산 2.6배 (행동 데이터로 이미 확인됨)
                           ↓
결론:              Variable features → 파산 (간접 인과관계 입증)
```

### 타당성 근거

**1. 성공 사례 존재 (Experiment 2)**
- Safe/Risky features 발견 (상관관계)
- Features 조작 → 행동 변화 확인 (+29.6% stopping)
- **결론**: Features가 행동을 제어한다 (인과관계 입증)

**2. 동일한 논리를 Bet Type 차원에 적용**
- Variable/Fixed features 발견 (상관관계)
- Features 조작 → 행동 변화? (우리 실험)
- **기대**: Variable features가 Variable-like 행동을 유발

**3. 왜 이게 "자율성 → 파산" 인과를 보이는가?**
- Variable features가 정말 "자율성(베팅 금액 자유 선택)" 관련이라면
- 이 features를 조작 → Variable-like 행동 유발
- Variable 행동 = 더 큰 베팅, 더 위험한 선택 → 파산율 증가
- **따라서**: Neural substrate (Variable features) → Behavioral outcome (파산)

### Experiment 2와의 유사성

| 단계 | Experiment 2 | 우리 제안 |
|------|-------------|----------|
| **발견** | Safe/Risky features | Variable/Fixed features |
| **조작** | Features 주입/제거 | Features 주입/제거 (동일) |
| **행동 변화** | Stopping rate 변화 | Stop rate, 베팅 금액, 파산율 변화 |
| **인과 입증** | Features → Safe/Risky 행동 | **Features → Variable 행동 → 파산** |

**차별점**: Experiment 2는 "결과"를 예측하는 features, 우리는 "원인(자율성)"을 나타내는 features
