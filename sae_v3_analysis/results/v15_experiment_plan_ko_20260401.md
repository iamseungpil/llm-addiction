# V15 추가 실험 계획서

날짜: 2026-04-01  
기준 문서: 한국어판 논문 `LLM_Addiction_NMT_KOR`  
실험 코드 기준: `sae_v3_analysis/src/run_v12_all_steering.py`, `run_v14_experiments.py`, `run_v14_parallel.py`

## 1. 목적

이 계획의 목적은 한국어판 논문의 인과성 파트를 현재 증거 수준에 맞게 강화하는 것이다.  
핵심 방향은 다음과 같다.

- 가장 강한 인과성 주장은 `LLaMA SM`에 둔다.
- `LLaMA IC`는 보조 인과성 증거로 평가한다.
- `LLaMA MW`, `cross-domain`, `Gemma MW`는 V14 최종 결과에 따라 제한적으로만 확장한다.
- 추가 실험은 기존 V12/V14와 같은 프롬프트, 같은 게임 로직, 같은 parsing 규칙을 유지해야 한다.

## 2. 현재 판단

중간 로그 기준으로 현재 결과는 다음과 같이 해석된다.

- 강한 축:
  - `Exp1` LLaMA SM
  - `Exp2a` LLaMA IC
- 약하거나 불안정한 축:
  - `Exp2b` LLaMA MW
  - `Exp3` cross-domain
- 아직 미완료:
  - `Exp4` Gemma MW

따라서 지금 단계에서 옳은 수순은:

1. V14를 끝까지 완료한다.
2. 한국어판 논문은 우선 `SM 중심의 직접 인과성` 구조로 정리한다.
3. MW 또는 cross-domain을 강하게 쓰고 싶을 때만 추가 실험을 수행한다.

## 2.1 NeurIPS 기준 기반 평가 규칙

NeurIPS 기준에서 현재 원고를 판단할 때 핵심은 ``실험을 많이 했는가''보다 ``핵심 claim과 직접 증거가 정확히 맞물리는가''이다. 특히 기존 방법이나 모델에 대한 깊이 있는 분석도 기여로 인정되므로, 현재 프로젝트는 무리한 일반화보다 정교한 해석과 재현성 쪽이 더 중요하다.

따라서 V14 종료 후 평가는 아래 네 축으로 수행한다.

1. \textbf{핵심 인과 주장 축}
   - `LLaMA SM`이 여전히 가장 강한 direction-specific causal case인가
   - `LLaMA IC` 또는 `Gemma MW`가 보조 causal evidence로 남는가

2. \textbf{일반화 축}
   - `MW`, `cross-domain`을 본문 핵심으로 올릴 만큼 직접 증거가 충분한가
   - 아니면 classification/shared-subspace 수준의 부분 일반화로만 남겨야 하는가

3. \textbf{재현성 축}
   - 결과 JSON, figure 재생성, seed 규칙, 실험 코드 경로가 카메라 레디 기준으로 설명 가능한가
   - random-control 결과가 한 번의 draw에 과도하게 의존하지 않는가

4. \textbf{서술 적합성 축}
   - 본문 claim이 현재 strongest case보다 앞서 나가지는 않는가
   - 약한 결과가 본문이 아니라 appendix/exploratory evidence로 정리되어 있는가

최종 판단 규칙은 다음처럼 둔다.

- \textbf{추가 실험 없이 진행 가능}
  - `Exp1`이 강하게 유지되고
  - `Exp2a` 또는 `Exp4` 중 하나 이상이 보조 축으로 남으며
  - MW/cross-domain을 약한 보조 증거로 낮춰도 논문의 중심 메시지가 유지될 때

- \textbf{선택적 1개 추가 실험 필요}
  - 본문에서 MW 또는 cross-domain을 살리고 싶지만 직접 증거가 애매할 때
  - 재현성 우려 때문에 strongest case를 한 번 더 고정 seed로 복제할 필요가 있을 때

- \textbf{추가 실험보다 서술 정리가 우선}
  - strongest claim이 이미 `LLaMA SM`으로 충분히 서고
  - 나머지는 appendix/exploratory로 내려도 전체 논문 구조가 유지될 때

- \textbf{추가 실험이 사실상 필요}
  - `Exp1` 외에는 direct causal section이 거의 비어 버려
  - reviewer가 ``single-case mechanism''이라고 볼 가능성이 높을 때

## 3. 비변경 원칙

다음 항목은 V15에서도 유지한다.

- `run_v12_all_steering.py`의 `play_game`
- `run_v12_all_steering.py`의 `run_condition`
- `run_v12_all_steering.py`의 `compute_bk_direction`
- `build_sm_prompt`, `build_ic_prompt`, `build_mw_prompt`
- `parse_sm_response`, `parse_ic_response`, `parse_mw_response`
- residual stream에 `alpha * direction`을 더하는 steering 방식

즉, 추가 실험은 새 게임을 만드는 것이 아니라 기존 실험 설정 위에서 최소 변경만 가하는 방식이어야 한다.

## 4. V15-A: LLaMA MW layer 재선정

### 목적

- 현재 MW 결과가 약한 이유가 `L22` 선택 때문인지 확인한다.

### 설정

- 모델: `LLaMA-3.1-8B-Instruct`
- 태스크: `MW`
- layer 후보:
  - `L16`
  - `L22`
  - `L25`
  - `L30`
- alpha grid:
  - V14와 동일
- random controls:
  - V14와 동일하거나 소폭 증가
- `n_games`:
  - V14와 동일하거나 소폭 증가

### 성공 기준

- 적어도 한 layer에서:
  - BK direction의 monotonicity가 현재 MW보다 좋아지고
  - random control보다 우위에 있어야 한다.

### 실패 시 해석

- MW는 단일 strong causal case로 쓰지 않고 exploratory 결과로 유지한다.

## 5. V15-B: Cross-domain 단일 pair 재검증

### 목적

- cross-domain 전체를 넓게 주장하는 대신 가장 유망한 한 pair만 강하게 점검한다.

### 우선순위

1. `MW -> SM`
2. `IC -> SM`
3. `MW -> IC`

### 설정

- source/target pair 하나만 선택
- 동일한 target-task prompt/game loop 유지
- `n_games` 증가
- random controls 증가
- direction 추출 방식은 동일하게 유지

### 성공 기준

- BK direction이 random control 분포보다 일관되게 강해야 한다.

### 실패 시 해석

- `RQ2`는 classification transfer / shared subspace 중심으로만 작성하고
- cross-domain causal transfer 주장은 제거하거나 exploratory로 낮춘다.

## 6. V15-C: Multi-layer steering

### 목적

- single-layer가 약한 조건에서 신호가 여러 depth에 분산되어 있는지 확인한다.

### 사용 조건

- within-domain evidence는 강한데
- cross-domain 또는 MW에서 single-layer 결과가 약할 때만 수행한다.

### 후보 조합

- `L22 + L25 + L30`

### 주의

- prompt, parser, game rule은 바꾸지 않는다.
- multi-layer 조합은 transfer signal 확인용 보조 실험으로만 사용한다.

## 7. V15-D: 재현성 강화 복제 실험

### 목적

- strongest claim이 우연한 random-control draw에 좌우되지 않도록 한다.

### 대상

- `Exp1` 스타일의 `LLaMA SM`
- `Exp2a` 스타일의 `LLaMA IC`

### 구현 원칙

- `hash(exp_name)` 대신 고정 seed 사용
- BK direction sweep 후 중간 JSON 저장
- random control 하나가 끝날 때마다 중간 JSON 저장

### 성공 기준

- verdict가 재실행 후에도 유지된다.

## 8. 논문 반영 규칙

### RQ1

- BK-related representation의 존재
- strongest direct causal evidence는 `LLaMA SM`
- `LLaMA IC`가 잘 나오면 supportive causal evidence로 추가

### RQ2

- cross-domain은 완전한 도메인 불변성이 아니라 부분적 공유 구조로 기술
- cross-domain steering은 추가 실험에서 직접 확인된 pair만 제한적으로 주장

### RQ3

- prompt, bet type, constraint가 BK signal을 어떻게 조절하는지 설명
- modulation claim 중심으로 유지

## 9. 실행 순서

1. V14 종료 대기
2. `Exp1`, `Exp2a`, `Exp2b`, `Exp3`, `Exp4` 최종 verdict 정리
3. `Exp1`과 `Exp2a`가 강하면 한국어판 본문 우선 수정
4. MW 또는 cross-domain을 더 강하게 쓰고 싶을 때만 V15-A 또는 V15-B 실행
5. 필요 시 V15-C, V15-D 진행

## 9.1 후속 실험 선택 방법

후속 실험은 ``남는 빈칸을 다 채우는 방식''이 아니라, 아래 순서대로 하나씩 고른다.

1. \textbf{논문에서 꼭 살리고 싶은 claim을 먼저 고른다.}
   - MW를 direct causal case로 살리고 싶은가
   - cross-domain causal transfer를 살리고 싶은가
   - reproducibility를 더 강하게 보여주고 싶은가

2. \textbf{그 claim에 직접 대응하는 실험만 선택한다.}
   - MW를 살리고 싶으면 `V15-A`
   - cross-domain을 살리고 싶으면 `V15-B`
   - strongest case의 안정성을 높이고 싶으면 `V15-D`
   - `V15-C`는 single-layer weakness가 명확할 때만 보조적으로 사용

3. \textbf{한 번에 하나만 실행한다.}
   - 첫 후속 실험 결과로 claim이 충분히 정리되면 거기서 멈춘다.
   - 첫 실험이 실패하면 바로 두 번째를 연쇄적으로 붙이지 않고, 논문 서술을 낮추는 편을 우선 검토한다.

4. \textbf{Gemma SM/IC 재반복은 기본 후보에서 제외한다.}
   - 두 조건은 이미 V12 direct steering에서 ceiling/floor 경계 조건으로 드러났기 때문에
   - 논문 중심 claim을 살리기 위한 1순위 후속 실험으로 두지 않는다.

## 10. 최종 권고

- 지금 당장은 현재 V14를 멈추지 않는다.
- 현재 가장 올바른 전략은:
  - `SM 중심 인과성`으로 논문 구조를 고정하고
  - 약한 주장만 선택적으로 보강하는 것이다.
- 추가 실험은 “논문 전체를 살리기 위해 전부 더 돌리는 방식”이 아니라
  - “강조하고 싶은 주장 하나를 선택해 보강하는 방식”으로 운영해야 한다.
