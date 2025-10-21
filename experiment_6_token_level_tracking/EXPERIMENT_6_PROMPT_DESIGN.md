# Experiment 6 Extension: Prompt Component Analysis

## 목표
잔액 정보 외에 **prompt 구성 요소**(G, M, P, R, W)가 token generation 중 feature 활성화에 미치는 영향 분석

## 현재 상태 vs 제안

### 현재 Experiment 6
- **고정 프롬프트**: G + P + W (모든 시나리오 동일)
- **변수**: 잔액 ($25, $90, $130)
- **발견**: 잔액 → L31-10692 활성화 → 의사결정

### 제안: Experiment 6B - Prompt Component Analysis

**설계**:
```
시나리오: Bankruptcy_90 ($90) 고정
프롬프트 조합: 8가지
- BASE (no components)
- G (Goal only)
- P (Probability only)
- M (Maximize only)
- GP (Goal + Probability)
- GM (Goal + Maximize)
- PM (Probability + Maximize)
- GMP (Goal + Maximize + Probability)

각 조합당: 30 trials
총: 240 trials
```

### 분석할 질문들

1. **Goal 정보가 feature 활성화에 미치는 영향**
   - BASE vs G: L31-10692 차이?
   - "목표 $200" 토큰이 generation에 영향?

2. **Probability 정보의 영향**
   - BASE vs P: "30%" 정보가 risk feature 활성화?
   - P만 있을 때 vs GP일 때 차이?

3. **Maximize instruction의 영향**
   - M 추가 시 L31-10692 증가?
   - "보상을 최대화" → 더 위험한 선택?

4. **조합 효과**
   - GMP가 G+M+P의 단순 합?
   - 시너지 또는 억제 효과?

### 예상 결과

**가설 1**: Goal이 있으면 L31-10692 증가
- BASE: Bet 50%, L31-10692 = 0.05
- G: Bet 70%, L31-10692 = 0.15

**가설 2**: Probability가 위험 인식 증가
- BASE: 무지성 Bet
- P: 신중한 Bet (30% 인식)

**가설 3**: Maximize가 가장 강력
- M: Bet 80%, L31-10692 = 0.30
- GM: Bet 85%, L31-10692 = 0.40

### 실험 실행 계획

**Phase 1** (빠른 검증):
```python
# 4가지 핵심 조합만
prompts = ['BASE', 'G', 'M', 'GMP']
trials_per_prompt = 20
total = 4 × 20 = 80 trials
예상 시간: 3분
```

**Phase 2** (전체):
```python
# 8가지 전체
prompts = ['BASE', 'G', 'P', 'M', 'GP', 'GM', 'PM', 'GMP']
trials_per_prompt = 30
total = 8 × 30 = 240 trials
예상 시간: 10분
```

### 분석 방법

1. **Feature 활성화 비교**
   ```
   BASE: L31-10692 mean = X
   G:    L31-10692 mean = Y
   Effect size = Y - X
   ```

2. **Prompt 토큰 attention**
   - Generation 중 어느 prompt 토큰에 주목?
   - "목표", "승률", "최대화" 중 어느 것?

3. **Decision pathway**
   - Goal 있을 때: "$200" → L8-2083 → L31-10692
   - Maximize 있을 때: "최대화" → 다른 feature?

## 기대 효과

1. **Prompt engineering 인사이트**
   - 어떤 요소가 위험 행동 유도?
   - LLM gambling addiction 메커니즘

2. **Feature interpretation**
   - L31-10692가 정말 "goal fixation"?
   - L8-2059가 "probability neglect"?

3. **논문 기여**
   - Token-level causality
   - Prompt → Feature → Behavior pathway

---

# 다음 단계

## 즉시 실행 가능
```bash
# Phase 1: 4가지 핵심 조합 (3분)
python experiment_6_prompt_components.py --quick

# Phase 2: 전체 8가지 (10분)
python experiment_6_prompt_components.py --full
```

## 코드 작성 필요
- `experiment_6_prompt_components.py` 생성
- Bankruptcy_90 시나리오 고정
- 8가지 prompt 조합 생성
- 동일한 feature extraction

---

**진행할까요?**
