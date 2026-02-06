# 실험 설계 비교 분석

> 기존 Paper Experiments와 새로 설계한 Gemma 실험들의 연구 목적, 방법론, 기대 효과 비교

---

## 연구 흐름 개요

```
[기존 연구: 외부 관찰자 관점]
    │
    ├── Section 3: 행동 현상 관찰
    │   ├── Slot Machine 6-Models: "LLM이 중독적 행동을 보이는가?"
    │   └── Investment Choice: "구조화된 선택지에서도 동일한가?"
    │
    ├── Section 4: 신경 메커니즘 분석
    │   └── LLaMA SAE Analysis: "어떤 features가 원인인가?"
    │
    └── Section 5: 시간적/언어적 분석
        └── Pathway Token Analysis: "결정 과정이 어떻게 전개되는가?"

                    ▼

[새 연구: 자기 인식 관점]
    │
    ├── gemma_metacognitive_experiment
    │   └── "LLM이 자신의 상태를 인식하고 통제할 수 있는가?"
    │
    └── gemma_sae_experiment
        └── "Domain-specific SAE로 해석력을 개선할 수 있는가?"
```

---

## 1. 기존 실험 요약

### 1.1 Slot Machine 6-Models (Section 3)

| 항목 | 내용 |
|------|------|
| **연구 질문** | LLM이 도박 상황에서 중독과 유사한 행동을 보이는가? |
| **방법** | 6개 모델 × 64 조건 × 50 반복 = 19,200 게임 |
| **핵심 발견** | Variable betting이 파산율을 급격히 증가시킴 (0% → 6-48%) |
| **의의** | LLM의 비합리적 위험 추구 행동 최초 정량화 |

### 1.2 Investment Choice (Section 3b)

| 항목 | 내용 |
|------|------|
| **연구 질문** | 구조화된 3-option 선택에서도 위험 추구 패턴이 나타나는가? |
| **방법** | Safe Exit / Moderate Risk / High Risk 선택 분석 |
| **핵심 발견** | 모델별 위험 선호도 차이, 라운드별 위험 escalation 패턴 |
| **의의** | Slot Machine 결과의 일반화 가능성 검증 |

### 1.3 LLaMA SAE Analysis (Section 4)

| 항목 | 내용 |
|------|------|
| **연구 질문** | 위험 추구 행동의 신경 수준 원인은 무엇인가? |
| **방법** | 6,400 게임 → SAE feature 추출 → Activation Patching |
| **핵심 발견** | 361개 safe features, 80개 risky features 발견 |
| **의의** | 행동의 인과적 신경 메커니즘 규명 |

### 1.4 Pathway Token Analysis (Section 5)

| 항목 | 내용 |
|------|------|
| **연구 질문** | 위험 결정이 시간적으로 어떻게 전개되는가? |
| **방법** | Token-level feature tracking + Word-feature correlation |
| **핵심 발견** | 결정 시점별 feature 진화, 언어적 위험 신호 패턴 |
| **의의** | 결정 과정의 시간적 구조 해명 |

---

## 2. 새 실험 설계

### 2.1 gemma_metacognitive_experiment

#### 연구 배경

기존 연구는 **외부 관찰자** 시점에서 LLM 행동을 분석했다. 그러나 AI Safety 관점에서 더 중요한 질문은:

> "LLM이 자신의 위험 상태를 **스스로 인식**하고 **통제**할 수 있는가?"

#### 이론적 기반

**Ji-An et al., 2025** - *"Language Models Are Capable of Metacognitive Monitoring and Control"*

이 연구는 LLM이 자신의 내부 activation을 읽고 조절할 수 있음을 보였다. 우리는 이를 gambling domain에 적용한다.

#### 세 가지 실험

| 실험 | 연구 질문 | 측정 변수 |
|------|----------|----------|
| **Exp A** | 자신의 risk activation을 정확히 보고할 수 있는가? | Reporting accuracy |
| **Exp B** | 언어적 지시만으로 safer 방향으로 이동할 수 있는가? | Activation shift, Cohen's d |
| **Exp C** | Self-report와 실제 행동/activation 간 gap은? | Deception rate, Correlation |

#### 기대 결과 및 함의

**Exp A - Metacognitive Reporting**:
- 예상: ICL examples 증가 → accuracy 향상
- 함의: LLM이 자기 상태 학습 가능 → self-monitoring 구현 가능성

**Exp B - Self-Control Capacity**:
- 예상: Explicit instruction > Implicit > Baseline 순으로 효과
- 함의: 언어적 개입만으로 행동 교정 가능 → 외부 patching 없이 안전성 확보

**Exp C - Awareness-Behavior Gap**:
- 예상: 일정 비율의 "self-deception" (Low risk 보고 + Risky 행동)
- 함의: Gap이 크면 → 외부 감시 필수, Gap이 작으면 → 자율적 안전 가능

#### 기존 연구와의 관계

```
Section 4 (SAE Analysis)     →    gemma_metacognitive
"Features가 행동을 결정"          "그 features를 LLM이 인식 가능한가?"
     (3인칭 분석)                      (1인칭 분석)
```

---

### 2.2 gemma_sae_experiment

#### 연구 배경

기존 LLaMA SAE Analysis의 한계:
- Base SAE의 reconstruction error로 인한 feature 해석 부정확
- General-purpose SAE가 gambling domain에 최적화되지 않음

#### 이론적 기반

**arXiv:2507.12990** - *"Teach Old SAEs New Domain Tricks with Boosting"*

Domain-specific residual SAE를 학습하여 reconstruction error를 줄이는 방법론.

#### 핵심 기법: SAE Boost

```
문제: Base SAE의 reconstruction error = x - x̂
해결: Residual SAE로 error를 예측하여 보정

Boosted reconstruction = Base reconstruction + Residual prediction
```

#### 6-Phase 파이프라인

| Phase | 목적 | 기존 대비 차이 |
|-------|------|---------------|
| **0** | Residual SAE 학습 | **신규** - gambling data로 domain adaptation |
| **1** | Feature 추출 | Boosted SAE 옵션 추가 |
| **2** | Correlation 분석 | 동일 |
| **3** | Steering Vector | 동일 |
| **4** | SAE 해석 | 더 정확한 feature 분해 기대 |
| **5** | Steering 실험 | 동일 |

#### 기대 결과 및 함의

**Phase 0 결과**:
- 예상: MSE 15-30% 감소
- 함의: Domain-specific adaptation의 효과 정량화

**전체 파이프라인 결과**:
- 예상: Boosted features가 더 선명한 risky/safe 분리 제공
- 함의: 더 정확한 causal features 식별 → 더 효과적인 intervention

#### 기존 연구와의 관계

```
Section 4 (LLaMA SAE)        →    gemma_sae_experiment
"Base SAE로 features 분석"        "Boosted SAE로 정확도 향상"
     (일반 도구)                      (domain 최적화)
```

---

## 3. 전체 연구 맥락에서의 위치

### 3.1 연구 질문의 진화

| 단계 | 질문 | 실험 |
|------|------|------|
| **1단계** | LLM이 중독적 행동을 보이는가? | Slot Machine, Investment Choice |
| **2단계** | 그 행동의 신경적 원인은? | LLaMA SAE Analysis |
| **3단계** | 결정 과정은 어떻게 전개되는가? | Pathway Token Analysis |
| **4단계** | LLM이 스스로 인식하고 통제 가능한가? | **gemma_metacognitive** |
| **4단계'** | 분석 도구를 개선할 수 있는가? | **gemma_sae** |

### 3.2 AI Safety 관점

```
기존 연구                         새 연구
────────                         ────────
"위험 행동 이해"                   "자율적 안전 가능성"
     │                                │
     │                                ├── 자기 인식 가능? (Exp A)
     ▼                                ├── 자기 통제 가능? (Exp B)
"외부 개입으로 교정"                  └── 인식-행동 일치? (Exp C)
(Activation Patching)                     │
                                          ▼
                                    "외부 감시 vs 자율 통제"
                                    정책 결정 근거 제공
```

### 3.3 방법론적 확장

| 측면 | 기존 | 새 실험 |
|------|------|--------|
| **관찰 시점** | 3인칭 (연구자가 관찰) | 1인칭 (LLM 자기 보고) |
| **개입 방식** | Activation patching | 언어적 지시 (verbal) |
| **SAE 활용** | Off-the-shelf | Domain-adapted (Boost) |
| **분석 대상** | 행동 + 내부 상태 | 행동 + 내부 상태 + **자기 인식** |

---

## 4. 예상되는 연구 기여

### 4.1 학술적 기여

1. **메타인지 연구 확장**: Ji-An et al.의 방법론을 gambling domain에 최초 적용
2. **Self-deception 정량화**: LLM의 자기 보고 신뢰성에 대한 실증적 증거
3. **SAE Boost 효과 검증**: Domain adaptation이 interpretability에 미치는 영향

### 4.2 실용적 기여

1. **AI 안전 정책**: 외부 감시 필요성 판단 근거
2. **Intervention 설계**: 언어적 개입 vs activation patching 효과 비교
3. **해석 도구 개선**: Gambling 외 다른 domain에도 SAE Boost 적용 가능성

### 4.3 논문 구조에서의 위치

```
[현재 논문]
Section 3: 행동 관찰
Section 4: 메커니즘 분석
Section 5: 시간적/언어적 분석

[확장 가능성]
Section 6: 자기 인식과 통제 (gemma_metacognitive)
    또는
Appendix: 개선된 SAE 분석 (gemma_sae)
```

---

## 5. 핵심 차이점 요약

| 구분 | 기존 실험 | 새 실험 |
|------|----------|--------|
| **핵심 질문** | "왜 이런 행동을 하는가?" | "스스로 인식/통제 가능한가?" |
| **관점** | 외부 관찰자 | 자기 인식 |
| **개입** | 신경 수준 (patching) | 언어 수준 (instruction) |
| **SAE** | General-purpose | Domain-adapted |
| **AI Safety 함의** | 위험 이해 | 자율적 안전 가능성 |

---

*마지막 업데이트: 2025-01-20*
