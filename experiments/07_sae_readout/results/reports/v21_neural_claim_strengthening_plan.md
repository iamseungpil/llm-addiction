# V21: Neural Claim Strengthening Plan

**Date**: 2026-04-09  
**Status**: Drafted from current codebase + literature-grounded design review

---

## 1. 목표

한국어 논문의 신경 섹션은 현재 다음의 강한 주장까지는 노릴 수 있다.

> 자율성이 증폭시키는 비합리적 행동은 단순한 프롬프트 표면 현상으로 환원되지 않으며, 모델 내부에서 측정 가능한 표상 구조와 제한적이지만 직접 개입 가능한 상태 변수를 동반한다.

다만 아래와 같은 더 강한 주장까지는 아직 닫히지 않았다.

- 단일한 범용 회로가 세 패러다임 전체를 관통한다
- 현재 steering 결과만으로 강한 인과 메커니즘이 확립되었다
- feature-level 설명이 곧 causal abstraction 수준의 설명이다

따라서 본 계획의 목적은 **주장의 톤은 유지하되, 그 주장을 실제로 지지하는 증거 패키지를 더 정교하게 보강**하는 것이다.

---

## 2. 현재 증거와 남은 빈틈

### 이미 있는 것

- 행동 수준:
  - 25,600게임에서 가변 베팅과 자기 목표 설정이 비합리성을 증폭
- 신경 수준 RQ1:
  - 비선형 잔액 통제 후 I_LC가 6개 model×task 조합 모두에서 유의
  - I_BA는 SM/MW에서 유의
  - 초기 layer보다 중간-후반 layer에서 신호가 강함
- 신경 수준 RQ3:
  - Gemma SM에서 G-prompt가 I_LC 신호를 증폭
  - Fixed에서는 신호가 사실상 소멸
- 직접 개입:
  - LLaMA SM L22 steering은 monotonic effect를 보이며 permutation p=0.048

### 아직 부족한 것

- probing 결과가 nuisance-matched control task보다 얼마나 선택적인지
- 현재 steering이 “generic perturbation”이 아니라 “특정 상태 변수 개입”인지
- RQ2를 feature overlap 실패 이상의 더 직접적인 반사실적 실험으로 보강한 증거

---

## 3. 문헌 기반 후보 실험

### 후보 A. Probe Selectivity / Control Task

- 의도:
  - 현재 RQ1이 단순히 더 유연한 supervised fitting이 아님을 보여준다.
- 핵심 차별점:
  - 기존 random feature baseline보다 강하다.
  - nuisance 구조를 보존한 control target과 직접 비교한다.
- 이론적 연결:
  - probing literature의 control-task / selectivity 논리와 정합적이다.

### 후보 B. Matched Interchange Intervention

- 의도:
  - 같은 라운드, 비슷한 잔액, 같은 프롬프트 조건에서 내부 상태만 바꿨을 때 다음 행동이 바뀌는지 본다.
- 핵심 차별점:
  - 기존 global alpha steering보다 훨씬 직접적인 counterfactual test다.
  - “surface prompt가 아니라 latent state가 행동을 바꾼다”는 주장과 가장 직접적으로 연결된다.
- 이론적 연결:
  - causal abstraction / interchange intervention 계열 논리와 맞닿아 있다.

### 후보 C. Causal Specificity / Scrubbing-lite

- 의도:
  - 효과가 특정 subspace에 집중되는지, 아니면 아무 perturbation이나 먹히는지 확인한다.
- 핵심 차별점:
  - random direction보다 더 강한 specificity test다.
- 리스크:
  - 구현 복잡도가 높고, 본 프로젝트의 현재 로그/데이터 구조와 가장 덜 맞는다.

### 후보 D. DAS-lite Cross-Paradigm Alignment

- 의도:
  - feature-level transfer 실패가 곧 shared low-rank structure 부재를 뜻하는지 확인한다.
- 핵심 차별점:
  - RQ2를 “공유 회로 없음”이 아니라 “shared low-rank factor가 있는지/없는지”로 더 정밀하게 판정한다.
- 리스크:
  - 새 contribution처럼 보이기 쉽고, 짧은 일정에서 구현/해석 부담이 크다.

---

## 4. 선택

### 지금 바로 진행할 실험

#### 실험 1. Group-held-out Probe Selectivity

- 의도:
  - RQ1의 핵심 신호가 game-specific leakage나 nuisance-matched pseudo-task보다 실제로 더 선택적인지 확인한다.
- 가설:
  - GroupKFold(game_id 기준)로 게임 단위 holdout을 하더라도 실제 I_LC/I_BA 예측은 양의 성능을 유지한다.
  - 같은 nuisance 구조를 보존한 control target에서는 성능이 크게 낮아진다.
- 검증 방법:
  - fold 내부에서만 RF deconfound, feature selection, scaling을 수행
  - real target과 nuisance-matched shuffled target을 같은 pipeline으로 비교
  - 지표:
    - `real_r2`
    - `control_r2_mean`
    - `selectivity_gap = real_r2 - control_r2_mean`
    - `p_selectivity`
- 왜 필요한가:
  - 현재 random feature / permutation null보다 probing 비판에 더 직접적으로 답한다.
- 구현 비용:
  - CPU 위주, 기존 코드 재사용 가능

#### 실험 2. Matched Interchange Intervention (SM 우선)

- 의도:
  - strong claim을 가장 직접적으로 보강한다.
- 가설:
  - 같은 round / balance bin / prompt condition / previous outcome을 맞춘 쌍에서, high-risk donor state를 low-risk recipient에 patch하면 다음 행동의 risk가 증가한다.
- 검증 방법:
  - target metric:
    - immediate next action continuation
    - bet_ratio
    - downstream bankruptcy
  - controls:
    - same-class donor patch
    - random donor patch
    - wrong-layer patch
    - random direction add-on
- 왜 필요한가:
  - 기존 steering보다 “specific internal state substitution”에 가깝다.
- 구현 비용:
  - GPU 필요, round-level prompt/state alignment 필요
  - 현재 자산 기준으로는 **LLaMA SM 우선**이 현실적이다. Gemma는 round metadata는 정리되어 있으나 donor-recipient 수준의 raw hidden-state 재구성이 더 필요할 수 있다.

### 이번 라운드에서는 보류할 실험

- Causal specificity / scrubbing-lite:
  - 실익은 있으나 구현 대비 위험이 큼
- DAS-lite:
  - 일정상 contribution scope를 불필요하게 넓힐 수 있음

---

## 5. 실행 순서

### 단계 1. 논문 가독성 정리

- 신경 섹션 첫머리에 RQ1/RQ2/RQ3의 역할과 지표 의미를 먼저 설명
- 표본 수와 분석 범위를 최신 값(16,000 game, 약 196K rounds)으로 통일
- direct causal evidence는 strong but bounded tone으로 정리

### 단계 2. 실험 1 코드 정비

- 기존 strict deconfound 구현 재사용
- GroupKFold + nuisance-matched control target 추가
- smoke test
- self-critique 후 수정

### 단계 3. 실험 1 실행

- representative config:
  - Gemma SM L24 I_LC
  - LLaMA SM L16 I_LC
  - Gemma MW L24 I_BA
  - LLaMA MW L16 I_BA
- 결과가 깔끔하면 본문 또는 appendix에 반영

### 단계 4. 실험 2 설계 확정

- round-level prompt/state asset inventory 확정
- raw hidden-state donor snapshot을 바로 쓸 수 있는 모델/과제부터 제한
- donor-recipient matching rule 확정
- immediate-action metric 우선 구현
- smoke test 뒤 GPU가 비면 실행

---

## 6. 논문 반영 원칙

### 본문에서 유지할 주장

- 자율성은 행동적으로 비합리성을 증폭시킨다
- 이 효과는 내부 표상 수준의 대응 신호를 동반한다
- 그 표상은 과제 전반에 동일한 회로로 공유되기보다 과제별로 분기한다

### 본문에서 피할 주장

- 현재 증거만으로 범용 단일 회로가 입증되었다
- single-direction steering 하나로 causal mechanism이 완전히 확립되었다
- feature overlap 실패만으로 모든 shared structure 부재가 증명되었다

---

## 7. 성공 기준

이번 계획은 아래가 충족되면 성공으로 본다.

- 신경 섹션이 definition-first 구조로 읽히고, 약어 없이도 흐름을 따라갈 수 있음
- Methods / Results / Discussion의 표본 수와 증거 수준이 서로 일치함
- 새 robustness 실험에서 real signal이 nuisance-matched control보다 분명히 높음
- 이후 GPU 실험은 matched interchange처럼 현재 주장과 직접 연결되는 것만 남김

---

## 8. 판단

현 시점에서 가장 효율적인 수순은 다음과 같다.

1. 논문 신경 섹션을 먼저 정돈한다.
2. CPU 기반 selectivity 실험으로 RQ1을 더 단단하게 만든다.
3. 그 다음에만 matched interchange intervention으로 stronger causal claim을 노린다.

이 순서는 현재 코드베이스와 가장 일관적이고, 실험 실패 시에도 논문 품질을 떨어뜨리지 않는다.
