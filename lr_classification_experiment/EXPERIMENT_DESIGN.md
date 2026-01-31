# LR Classification Experiment Design

## 핵심 연구 질문

> **SAE 없이 raw activation만으로 bankruptcy vs voluntary_stop 상태를 구분할 수 있는가?**

### 배경
- 기존 연구: SAE를 통해 feature를 추출하여 분석
- 문제점: 범용 SAE 사용 → task-specific하지 않음, 시간/비용 높음
- 새 접근: Raw hidden state를 직접 사용 → 더 직접적인 분석

---

## 실험 데이터

### 데이터 소스
| Model | Path | Games | Bankruptcy Rate |
|-------|------|-------|-----------------|
| Gemma-2-9B | `slot_machine/gemma/final_gemma_20251004_172426.json` | 3,200 | ~20.9% |
| LLaMA-3.1-8B | `slot_machine/llama/final_llama_20251004_021106.json` | 3,200 | TBD |

### 실험 조건 (64 conditions)
- **bet_type**: fixed, variable (2)
- **prompt_combo**: BASE + 31 combinations (32)
  - G: Goal setting
  - M: Maximize reward
  - R: Hidden patterns
  - W: Win multiplier (3.0x)
  - P: Win rate (30%)

---

## Option B: 종료 직전 상태 (핵심 실험)

### 정의
"게임의 마지막 결정 직전 상태"의 hidden state를 추출

### 세부 정의

#### Case 1: Bankruptcy
```
Round 1: balance=$100 → bet $90 → LOSS → balance=$10
Round 2: balance=$10  → bet $10 → LOSS → balance=$0 (파산)

Option B 추출 시점: Round 2 직전
- balance: $10
- history: [Round 1 정보]
- 프롬프트: Round 2에서 모델이 보는 프롬프트
```

#### Case 2: Voluntary Stop
```
Round 1: balance=$100 → bet $10 → WIN → balance=$120
Round 2: balance=$120 → bet $10 → LOSS → balance=$110
Round 3: STOP (자발적 중단)

Option B 추출 시점: Round 3 직전 (STOP 결정 직전)
- balance: $110
- history: [Round 1, Round 2 정보]
- 프롬프트: STOP 결정할 때 모델이 보는 프롬프트
```

#### Case 3: Immediate Stop (Round 0에서 바로 중단)
```
Round 0: STOP (첫 프롬프트에서 바로 중단)

Option B 추출 시점: Round 0
- balance: $100
- history: []
- 프롬프트: 초기 프롬프트
```

### 코드 구현 확인사항
```python
# 현재 구현 (prompt_reconstruction.py)
if game['outcome'] == 'bankruptcy':
    # 파산 직전 상태 (마지막 베팅 전)
    balance = history[-2]['balance'] if len(history) >= 2 else 100
    hist_for_prompt = history[:-1]
else:
    # 자발적 중단: 최종 상태
    balance = game['final_balance']
    hist_for_prompt = history
```

---

## 분석 그룹 설계

### 1. 전체 분석 (Main)
- 모든 데이터를 합쳐서 분석
- N = 3,200 (per model)

### 2. bet_type별 분석
| Group | N (예상) | 설명 |
|-------|----------|------|
| fixed | 1,600 | 고정 베팅 ($10) |
| variable | 1,600 | 가변 베팅 ($5-$balance) |

### 3. prompt_combo별 분석
| Group | N (예상) | 설명 |
|-------|----------|------|
| BASE | 100 | 기본 프롬프트 |
| M | 100 | Maximize만 |
| GM | 100 | Goal + Maximize |
| ... | ... | ... |
| GMRWP | 100 | 모든 컴포넌트 |

### 4. bet_type × prompt_combo 교차 분석
- 64 conditions × 50 repetitions
- 각 condition별로 충분한 샘플 확보 여부 확인 필요

---

## 평가 지표

### 1. Accuracy (정확도)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- 의미: 전체 예측 중 맞춘 비율
- 범위: 0-1 (1이 최고)
- 주의: 클래스 불균형 시 misleading할 수 있음

### 2. AUC-ROC (Area Under ROC Curve)
```
ROC Curve: True Positive Rate vs False Positive Rate
AUC: 곡선 아래 면적
```
- 의미: 분류기의 전반적 성능 (threshold 무관)
- 범위: 0-1 (0.5 = random, 1 = perfect)
- 장점: 클래스 불균형에 robust

### 3. Precision (정밀도)
```
Precision = TP / (TP + FP)
```
- 의미: "bankruptcy"로 예측한 것 중 실제 bankruptcy 비율
- 범위: 0-1
- 중요: False alarm이 문제일 때

### 4. Recall (재현율)
```
Recall = TP / (TP + FN)
```
- 의미: 실제 bankruptcy 중 맞춘 비율
- 범위: 0-1
- 중요: 놓치면 안 될 때 (위험 탐지)

### 5. F1 Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- 의미: Precision과 Recall의 조화 평균
- 범위: 0-1
- 용도: 불균형 데이터에서 단일 지표로 사용

### 6. Cross-Validation Score
```
5-Fold CV: 데이터를 5등분하여 교차 검증
```
- 의미: 일반화 성능 추정
- 보고: mean ± std

---

## Baseline 비교

### Baseline 1: Chance Level
```
Accuracy = max(P(bankruptcy), P(safe))
```
- Gemma 기준: ~79.1% (majority class = safe)
- 의미: "항상 safe 예측"보다 나은가?

### Baseline 2: Input Text Only (Bag of Words)
```
프롬프트 텍스트 → TF-IDF → LR
```
- Hidden state 없이 입력 텍스트만으로 분류
- 의미: Hidden state가 텍스트 이상의 정보를 담는가?

### Baseline 3: Metadata Only
```
[balance, n_rounds, n_wins, n_losses, consecutive_losses] → LR
```
- Hidden state 없이 게임 상태 메타데이터만 사용
- 의미: Hidden state가 메타데이터 이상의 정보를 담는가?

### Baseline 4: Random Projection
```
Hidden state → Random matrix → LR
```
- Hidden state의 차원을 랜덤하게 축소
- 의미: LR 성능이 hidden state의 구조적 정보 때문인지 확인

---

## 실험 매트릭스

### Phase 1: Main Experiment (Option B)

| 실험 ID | Model | bet_type | prompt_combo | 비교 |
|---------|-------|----------|--------------|------|
| 1.1 | Gemma | all | all | Main |
| 1.2 | LLaMA | all | all | Main |
| 1.3 | Gemma | fixed | all | bet_type |
| 1.4 | Gemma | variable | all | bet_type |
| 1.5 | LLaMA | fixed | all | bet_type |
| 1.6 | LLaMA | variable | all | bet_type |

### Phase 2: prompt_combo별 분석

| 실험 ID | Model | prompt_combo | N |
|---------|-------|--------------|---|
| 2.1 | Gemma | BASE | ~100 |
| 2.2 | Gemma | M | ~100 |
| ... | ... | ... | ... |

### Phase 3: Baseline 비교

| 실험 ID | Method | 비교 대상 |
|---------|--------|----------|
| 3.1 | Chance Level | 1.1, 1.2 |
| 3.2 | Text Only (TF-IDF) | 1.1, 1.2 |
| 3.3 | Metadata Only | 1.1, 1.2 |
| 3.4 | Random Projection | 1.1, 1.2 |

---

## 예상 결과 해석

### 시나리오 1: LR Accuracy >> Chance Level
```
예: Accuracy = 75%, Chance = 50%
```
- **해석**: Hidden state가 결과를 예측하는 정보를 담고 있음
- **의미**: 모델이 "위험 상태"를 내부적으로 인코딩함

### 시나리오 2: LR Accuracy ≈ Chance Level
```
예: Accuracy = 52%, Chance = 50%
```
- **해석**: Hidden state만으로는 구분 불가
- **의미**: 결과가 random하거나, 다른 표현 공간 필요

### 시나리오 3: LR Accuracy >> Metadata Baseline
```
예: LR = 75%, Metadata = 60%
```
- **해석**: Hidden state가 메타데이터 이상의 정보 포함
- **의미**: 모델이 명시적 상태 외의 "잠재적 상태"를 인코딩

### 시나리오 4: Later layers > Earlier layers
```
예: Layer 35 = 75%, Layer 10 = 55%
```
- **해석**: 후반 레이어에서 더 추상적인 표현 형성
- **의미**: 결정 관련 정보가 깊은 레이어에 집중

---

## 구현 체크리스트

- [ ] Option B 정의 재확인 및 코드 검증
- [ ] bet_type별, prompt_combo별 그룹 분리 구현
- [ ] 전체 분석 + 그룹별 분석 파이프라인
- [ ] 모든 평가 지표 계산 함수
- [ ] Baseline 비교 구현
  - [ ] Chance level
  - [ ] Text only (TF-IDF)
  - [ ] Metadata only
  - [ ] Random projection
- [ ] 결과 저장 및 시각화

---

## 다음 단계

1. 이 설계 문서 확인 및 피드백
2. 코드 업데이트 (baseline 추가 등)
3. GPU 환경 확보 후 실험 실행
