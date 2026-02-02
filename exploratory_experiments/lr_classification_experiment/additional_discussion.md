# LR Classification 실험: 핵심 아이디어와 향후 계획

> 이 문서는 실험의 이론적 배경, 예상 결과 해석, 그리고 후속 연구 방향에 대한 심층 논의를 담고 있습니다.

---

## 1. 현재 실험이 밝히고자 하는 것

### 핵심 연구 질문

> **LLM의 internal activation에 "위험 상태(risky state)"가 선형적으로 인코딩되어 있는가?**

이 질문을 더 구체적으로 분해하면:

1. **Bankruptcy로 끝나는 게임**과 **Voluntary Stop으로 끝나는 게임**의 hidden state가 activation space에서 **선형적으로 분리 가능한가?**

2. 만약 분리 가능하다면, 이것은 **모델이 자신의 "운명"을 이미 내부적으로 알고 있다**는 증거인가?

3. 이 "위험 상태"의 표상이 **어느 레이어에서 형성되는가?** (초기 레이어 vs 후기 레이어)

### 왜 이것이 중요한가?

**기존 SAE 접근의 한계:**
- SAE는 범용적으로 학습됨 → task-specific feature를 놓칠 수 있음
- Sparse coding의 가정이 항상 맞지 않을 수 있음
- 계산 비용이 높고, feature 해석이 주관적

**LR 접근의 장점:**
- **직접적**: raw activation에서 바로 분류
- **해석 가능**: LR 가중치 = "위험 방향" 벡터
- **효율적**: SAE 학습 없이 바로 분석 가능
- **이론적 의미**: 선형 분리 가능성 = 일관된 방향으로의 인코딩

### Option B가 특별한 이유

Option B는 **"결정 직전 상태"**를 분석합니다:

```
Bankruptcy 케이스:
  Round N-1: balance=$50 → bet $50 → LOSS → balance=$0 (파산)
  ↑ 우리가 보는 시점: Round N-1 직전의 hidden state

Voluntary Stop 케이스:
  Round N: balance=$150 → STOP 결정
  ↑ 우리가 보는 시점: STOP 결정 직전의 hidden state
```

이 시점에서:
- 모델은 **아직 최종 결과를 "모름"** (결정을 내리기 전)
- 하지만 hidden state에 **결과를 예측할 정보가 있다면**?
- 이는 모델 내부에 **"위험 경향성"이 인코딩**되어 있다는 강력한 증거

---

## 2. 예상 결과 시나리오별 해석

### 시나리오 A: LR 성능이 높음 (Accuracy 70%+, AUC 0.75+)

**해석:**
- Hidden state가 bankruptcy vs safe를 **선형적으로 구분**할 수 있음
- 모델 내부에 **"위험 상태 표상"**이 존재함
- 이 표상은 **일관된 방향**(LR 가중치 방향)으로 인코딩됨

**의미:**
- 모델이 gambling 상황에서 **자기 상태를 "알고" 있음**
- 이 방향을 조작하면 **행동을 바꿀 수 있을 가능성**
- SAE 없이도 meaningful한 direction을 찾을 수 있음

### 시나리오 B: LR 성능이 Metadata Baseline과 비슷함

**해석:**
- Hidden state의 정보가 **명시적 게임 상태(balance, rounds, wins)와 동등**
- 모델이 "추가적인 잠재 상태"를 인코딩하지 않음

**의미:**
- 결과 예측은 **단순히 현재 상태의 반영**일 뿐
- "위험 성향"보다는 **합리적 상태 추적**에 가까움
- 비선형 분석이나 다른 접근이 필요할 수 있음

### 시나리오 C: LR 성능이 Chance Level에 가까움 (Accuracy ~50%)

**해석:**
- Raw activation으로는 **선형적 구분 불가**
- 결과가 truly random하거나, **다른 표현 공간**에 정보가 있음

**의미:**
- 비선형 분류기 시도 필요 (MLP, SVM-RBF)
- 다른 token position 분석 필요
- 또는 gambling 결과가 **내부 상태와 무관**할 수 있음

### 시나리오 D: Later Layers >> Earlier Layers

**해석:**
- 후반 레이어에서 **추상적 "위험" 표상**이 형성됨
- 초기 레이어는 단순 입력 처리, 후기 레이어에서 의사결정 관련 표상

**의미:**
- **계층적 표상 학습**의 증거
- Steering intervention은 **후반 레이어**에서 더 효과적일 가능성
- Mechanistic interpretability 연구의 타겟 레이어 식별

---

## 3. 이 실험 이후 가능한 연구 방향

### Phase 2: 결과 심화 분석

#### 2.1 LR 가중치 분석
```
LR이 학습한 가중치 w ∈ R^d_model
→ 이것이 "bankruptcy direction" 벡터
→ 이 방향으로 projection하면 "위험 점수" 계산 가능
```

**질문들:**
- 이 방향이 **semantic하게 해석 가능**한가?
- 어떤 **뉴런/차원**이 가장 큰 가중치를 갖는가?
- 이 방향과 **SAE feature**의 상관관계는?

#### 2.2 레이어별 Dynamics
- 각 레이어의 LR 가중치를 비교
- "위험 방향"이 레이어를 거치며 **어떻게 변화**하는가?
- **정보 흐름** 추적

#### 2.3 Condition별 심화 분석
- **bet_type별**: fixed vs variable에서 "위험 인코딩"이 다른가?
- **prompt_combo별**: 특정 프롬프트가 더 명확한 분리를 만드는가?
- 이 차이가 **gambling 행동의 차이**와 연결되는가?

---

### Phase 3: Steering & Intervention

#### 3.1 Activation Steering
```
기존 hidden state: h
Steering: h' = h + α * w  (w = LR 가중치, α = steering 강도)
```

**실험:**
- Bankruptcy-prone state → Safe direction으로 steering
- **실제로 행동이 바뀌는가?** (Bet → Stop으로 변경)
- 최적의 **α 값**과 **레이어** 탐색

#### 3.2 Causal Intervention
- Activation patching: 특정 위치의 activation을 swap
- "위험 상태"의 activation을 "안전 상태"로 교체
- **Causal effect** 측정: 결과가 실제로 바뀌는가?

#### 3.3 Real-time 위험 감지
- Hidden state를 실시간 모니터링
- LR 점수가 threshold 초과 시 **경고/intervention**
- 실용적 application 가능성 탐색

---

### Phase 4: Cross-Domain 확장

#### 4.1 Cross-Model 비교
| 비교 | 질문 |
|------|------|
| Gemma vs LLaMA | 같은 "위험 방향"이 존재하는가? |
| Transfer | Gemma에서 학습한 LR이 LLaMA에서 작동하는가? |
| Universality | "위험 인코딩"이 모델 아키텍처에 독립적인가? |

#### 4.2 Cross-Task 비교
- **Slot Machine** vs **Investment Choice**
- 같은 "위험" 개념이 다른 task에서도 일관되게 인코딩되는가?
- **범용 "위험 감지" 메커니즘**이 있는가?

#### 4.3 다른 도메인으로 확장
- **금융 의사결정**: 투자, 대출 결정
- **건강 관련 결정**: 위험한 선택 vs 안전한 선택
- **일반 의사결정**: 즉각적 보상 vs 지연된 보상

---

### Phase 5: Mechanistic Interpretability

#### 5.1 Attention Pattern 분석
- LR 성능이 높은 레이어에서 **attention head** 분석
- 어떤 head가 "위험 상태"를 인코딩하는가?
- **Balance 숫자**, **history 패턴** 등에 attention하는가?

#### 5.2 Circuit Discovery
- "위험 → 행동" pathway 추적
- 특정 **MLP 뉴런**이나 **attention head**가 핵심인가?
- **Ablation study**: 해당 component 제거 시 행동 변화

#### 5.3 SAE와의 통합
- LR 가중치 방향과 **SAE feature 방향** 비교
- LR이 implicitly하게 특정 SAE feature를 사용하는가?
- **SAE-free direction**과 **SAE feature**의 관계

---

## 4. 연구 로드맵

```
현재 (Phase 1)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  LR Classification 실험 실행                      │
│  - Option B (핵심)                               │
│  - 전체 + 그룹별 분석                             │
│  - Baseline 비교                                 │
└─────────────────────────────────────────────────┘
    │
    ▼
    ┌─────────┬─────────┬─────────┐
    │         │         │         │
    ▼         ▼         ▼         ▼
  높은 성능  중간 성능  낮은 성능  Metadata와 유사
    │         │         │         │
    ▼         ▼         ▼         ▼
Phase 2a   Phase 2b   Phase 2c   Phase 2d
가중치분석  레이어분석  비선형시도  Residual분석
    │         │         │         │
    └────┬────┴────┬────┘         │
         │         │              │
         ▼         ▼              ▼
     Phase 3: Steering/Intervention
         │
         ▼
     Phase 4: Cross-Model/Cross-Task
         │
         ▼
     Phase 5: Mechanistic Interpretability
```

---

## 5. 논문 기여도 예상

### 만약 실험이 성공적이라면 (시나리오 A):

1. **방법론적 기여**: SAE 없이 raw activation에서 직접 meaningful direction 추출
2. **실증적 발견**: LLM 내부에 "위험 상태 표상" 존재 확인
3. **응용 가능성**: Steering을 통한 gambling 행동 modification
4. **이론적 통찰**: LLM의 의사결정 메커니즘에 대한 이해

### 논문 제목 후보:
- "Linear Probing Reveals Risk-State Representations in LLMs During Gambling Tasks"
- "Can We Detect When LLMs Are About to Go Bankrupt? A Linear Classification Approach"
- "From Hidden States to Risk States: Probing LLM Decision-Making in Gambling Scenarios"

---

## 6. 핵심 요약

| 항목 | 내용 |
|------|------|
| **핵심 질문** | Raw activation으로 bankruptcy vs safe 구분 가능한가? |
| **왜 중요한가** | 모델 내부의 "위험 상태 표상" 존재 여부 확인 |
| **왜 LR인가** | 선형 분리 가능성 = 일관된 방향으로의 인코딩 |
| **성공 시** | Steering, intervention, 실용적 위험 감지 가능 |
| **실패 시** | 비선형 분석, 다른 token position, 다른 접근 필요 |
| **장기 목표** | LLM 의사결정 메커니즘 이해 및 제어 |

---

## 7. 관련 참고문헌

- **Ji-An et al. (2025)**: "Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations" (arXiv:2505.13763)
  - LR을 사용하여 activation space에서 의미있는 방향을 식별
  - Neurofeedback paradigm을 통한 모델 자기 인식 연구

- **Linear Probing 관련**:
  - Alain & Bengio (2016): "Understanding intermediate layers using linear classifier probes"
  - Belinkov (2022): "Probing Classifiers: Promises, Shortcomings, and Advances"

- **Steering/Intervention 관련**:
  - Turner et al. (2023): "Activation Addition: Steering Language Models Without Optimization"
  - Li et al. (2024): "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"

---

## 8. 실험 실행 전 체크리스트

- [ ] GPU 환경 확보 (최소 22GB VRAM for Gemma, 19GB for LLaMA)
- [ ] 데이터 경로 확인 (`/mnt/c/Users/oollccddss/git/data/llm-addiction/`)
- [ ] config.yaml 설정 확인
- [ ] bf16 precision 사용 확인 (quantization 금지)
- [ ] 결과 저장 디렉토리 확인

---

*Last Updated: 2025-01*
*Author: LLM Addiction Research Team*
