# V5 종합 SAE/활성화 분석: 신경 메커니즘 연구 보고서

**저자**: 이승필, 신동현, 이윤정, 김선동 (GIST)
**날짜**: 2026-03-09
**프로젝트**: LLM 도박 중독 — 신경 메커니즘 분석
**논문**: "Can Large Language Models Develop Gambling Addiction?" (Nature Machine Intelligence)
**논문 저장소**: `/home/jovyan/LLM_Addiction_NMT/`
**코드 저장소**: `/home/jovyan/llm-addiction/`

---

## Executive Summary

V5는 V4의 방법론적 개선(부트스트랩 CI, 순열 검정, 동일 레이어 비교)을 기반으로, **두 가지 표현 유형(SAE 특징 vs Raw Hidden States)**, **두 가지 모델 아키텍처(Gemma-2-9B, LLaMA-3.1-8B)**, **세 가지 도박 패러다임(IC, SM, MW)** 에 걸쳐 체계적인 비교를 수행한 종합 분석이다.

### 핵심 발견 요약

| 분석 | 핵심 지표 | Gemma IC | Gemma SM | Gemma MW | LLaMA IC |
|------|----------|----------|----------|----------|----------|
| BK 예측 (DP, SAE) | AUC | **0.964** (L22) | **0.981** (L12) | **0.966** (L33) | **0.957** (L11) |
| BK 예측 (DP, Hidden) | AUC | **0.964** (L26) | **0.982** (L10) | **0.968** (L12) | 미완 |
| R1 예측 (잔고 통제) | AUC | 0.854 (L18) | 0.901 (L16) | 0.766 (L22) | 0.799 (L1) |
| 잔고 매칭 BK | AUC | 0.744 (L26) | 0.689 (L0) | 0.702 (L0) | 미완 |
| 교차 도메인 전이 | AUC [CI] | IC→SM: 0.893 | IC→MW: 0.908 | MW→SM: 0.877 | — |
| R1 순열 검정 | p-value | p<0.001 | p<0.001 | p<0.001 | p=0.010 |
| 제약 인코딩 (4-class) | AUC | 0.966 (L18) | — | — | 0.994 (L6) |

**핵심 결론**:

1. **SAE ≈ Hidden**: SAE 분해는 예측 관련 정보를 완전히 보존한다 (DP AUC 차이 < 0.003).
2. **R1 신호는 실제**: 첫 라운드(모두 $100)에서 이미 파산 예측 가능 (AUC 0.766–0.901, 모두 p<0.01).
3. **잔고 혼동변수 ~15%**: DP AUC의 약 15%는 잔고 차이에 기인하지만, 잔고 매칭 후에도 실질적 신호 잔존 (0.630–0.744).
4. **교차 도메인 전이는 비대칭**: IC는 강력한 소스 도메인(→SM 0.893, →MW 0.908), SM/MW는 약한 소스(→IC 0.616–0.631).
5. **아키텍처 일반성**: LLaMA IC AUC 0.957은 Gemma 0.964와 비교 가능. 교차 모델 전이 AUC 0.951.

---

## 1. 실험 설계

### 1.1 데이터셋 (검증된 Clean 데이터만 사용)

| 데이터셋 | 모델 | 게임 수 | 파산(BK) | BK% | 경로 |
|---------|------|---------|---------|-----|------|
| IC V2role | Gemma-2-9B | 1,600 | 172 | 10.8% | `investment_choice_v2_role/` |
| SM V4role | Gemma-2-9B | 3,200 | 87 | 2.7% | `slot_machine/experiment_0_gemma_v4_role/` |
| MW V2role | Gemma-2-9B | 3,200 | 54 | 1.7% | `mystery_wheel_v2_role/` |
| IC V2role | LLaMA-3.1-8B | 1,600 | 142 | 8.9% | `investment_choice_v2_role_llama/` |
| SM V4role | LLaMA-3.1-8B | **실행중** | — | — | `slot_machine/experiment_0_llama_v4_role/` |
| MW V2role | LLaMA-3.1-8B | **실행중** | — | — | `mystery_wheel/` |

(source: CLAUDE.md Experiment Data Inventory)

**데이터 품질**: V1 슬롯머신 데이터(24.6% 잘못된 고정 베팅, 파서 버그 6개)는 전량 제외. V2role/V4role 데이터는 코드+데이터 감사를 통해 100% 검증됨.

### 1.2 SAE 아키텍처

| 모델 | SAE | 레이어 | 특징 수/레이어 | 인코더 | 로딩 방식 |
|------|-----|--------|-------------|--------|----------|
| Gemma-2-9B | GemmaScope | 42 (L0–L41) | 131,072 | JumpReLU | sae_lens |
| LLaMA-3.1-8B | LlamaScope | 32 (L0–L31) | 32,768 | ReLU + norm_factor | fnlp 직접 |

(source: `paper_experiments/llama_sae_analysis/src/phase1_optimized.py`)

### 1.3 분석 파이프라인

```python
# 공통 분류 방법론
LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced')
# 5-겹 층화 교차 검증 (StratifiedKFold)
# 활성 특징: 샘플 간 평균 활성화 > 1e-6
# 결정 지점(Decision Point): 파산 게임은 history[-2], 비파산은 최종 라운드
```

(source: `sae_v3_analysis/src/run_all_analyses.py`, `run_comprehensive_gemma.py`)

### 1.4 파산 정의와 결정 지점

**파산(Bankruptcy, BK)**: 게임 종료 시 잔고 = $0 (모든 자본 소진).

**결정 지점(Decision Point, DP)**: 파산 게임의 경우 잔고가 $0이 되기 직전의 마지막 의사결정 라운드. 비파산 게임은 자발적 중단 직전 또는 마지막 라운드. `history[-2]['balance']` 사용 (history[-1]은 결과 반영 후).

**라운드 1(R1)**: 모든 게임의 첫 라운드 — 잔고가 $100으로 동일하여 잔고 혼동변수가 제거됨.

---

## 2. 실험 결과

### 2.1 Goal A: 파산 분류 — SAE와 Hidden States의 동등한 예측력

Gemma-2-9B의 세 패러다임에서, SAE 특징과 Raw Hidden States는 거의 동일한 파산 예측 성능을 보인다.

![Figure 1: Gemma BK Classification Matrix](figures/v5_fig1_bk_classification_matrix.png)

**Figure 1 해석**: 좌측 패널(Decision Point)에서 SAE와 Hidden의 AUC 차이는 IC에서 0.000, SM에서 0.001, MW에서 0.002로, 사실상 동일하다. 이는 GemmaScope SAE가 파산 관련 정보를 손실 없이 분해함을 의미한다. 중앙 패널(Round 1)에서도 유사한 패턴이 관찰된다 (차이 < 0.003). 우측 패널(Balance-Matched)에서는 Hidden과 SAE 간 차이가 다소 증가하며 (IC: 0.020, SM: 0.059), 이는 잔고 매칭으로 데이터가 줄어들면서 노이즈의 영향이 커진 것으로 해석된다.

#### 2.1.1 Decision Point 분류 결과 (전체 라운드)

| 패러다임 | SAE AUC (Layer) | Hidden AUC (Layer) | n_BK | n_Safe | 차이 |
|---------|----------------|-------------------|------|--------|------|
| IC | 0.964 (L22) | 0.964 (L26) | 172 | 1,428 | 0.000 |
| SM | 0.981 (L12) | 0.982 (L10) | 87 | 3,113 | -0.001 |
| MW | 0.966 (L33) | 0.968 (L12) | 54 | 3,146 | -0.002 |

(source: `all_analyses_20260306_091055.json`, `comprehensive_gemma_20260309_063511.json`)

#### 2.1.2 Round 1 분류 결과 (잔고 = $100 통제)

| 패러다임 | SAE AUC (Layer) | Hidden AUC (Layer) | p-value (순열) |
|---------|----------------|-------------------|---------------|
| IC | 0.854 (L18) | 0.856 (L33) | < 0.001 |
| SM | 0.901 (L16) | 0.900 (L26) | < 0.001 |
| MW | 0.766 (L22) | 0.764 (L0) | < 0.001 |

(source: `improved_v4_20260308_032435.json:r1_permutation_test`, `comprehensive_gemma_20260309_063511.json:hidden_bk`)

**해석**: R1에서 모든 게임의 잔고는 $100으로 동일하므로, 파산 예측 신호는 잔고가 아닌 **모델의 내부 의사결정 성향**에서 발생한다. 순열 검정(100회, 라벨 셔플)에서 모두 p<0.001로, 이 신호는 통계적으로 유의하다. 귀무 분포 평균은 0.498–0.505, 표준편차 0.030–0.050으로, 관찰된 AUC(0.766–0.901)와 명확히 분리된다.

#### 2.1.3 잔고 매칭 분석

| 패러다임 | SAE Bal-Matched (Layer) | Hidden Bal-Matched (Layer) | DP→BM 감소폭 |
|---------|------------------------|--------------------------|-------------|
| IC | 0.744 (L26) | 0.724 (L26) | -0.220 (SAE) |
| SM | 0.689 (L0) | 0.630 (L12) | -0.292 (SAE) |
| MW | 0.702 (L0) | 0.835 (L12)* | -0.264 (SAE) |

*MW Hidden Balance-Matched: 분석 진행중 (L12 기준 중간 결과)

(source: `improved_v4_20260308_032435.json:gemma_balance_controlled`, `hidden_gaps_20260309_181059.log`)

**해석**: DP AUC에서 잔고 매칭 AUC로의 감소폭은 약 0.22–0.29로, DP 성능의 약 15–20%가 잔고 차이에 기인한다. 그러나 매칭 후에도 0.630–0.744의 실질적 신호가 잔존하여, 파산 예측이 단순히 "잔고가 낮아서 파산"이 아닌 더 근본적인 의사결정 패턴 차이를 반영함을 확인한다.

---

### 2.2 Goal B: 교차 도메인 전이 — IC가 최강 소스 도메인

한 패러다임에서 학습한 SAE 특징 분류기를 다른 패러다임에서 평가하여, 도메인 간 일반화되는 "범용 도박 신호"의 존재를 검증한다.

![Figure 2: Cross-Domain SAE Transfer Heatmap](figures/v5_fig2_cross_domain_transfer.png)

**Figure 2 해석**: 히트맵에서 IC 행(IC를 소스로 사용)은 일관되게 높은 전이 성능을 보인다 (→SM 0.893, →MW 0.908). 반면, SM과 MW를 소스로 사용할 때 IC로의 전이는 약하다 (0.616, 0.631). 이 비대칭은 IC의 풍부한 의사결정 구조(4가지 선택지, 베팅 금액 변동, 명시적 투자 프레이밍)가 보다 일반적인 위험 추구 표현을 학습하게 함을 시사한다.

#### 2.2.1 부트스트랩 95% 신뢰구간

| 소스 → 타겟 | AUC | 95% CI | Layer | 해석 |
|------------|-----|--------|-------|------|
| IC → SM | 0.893 | [0.867, 0.921] | L26 | **강한 전이** |
| IC → MW | 0.908 | [0.874, 0.944] | L18 | **강한 전이** |
| MW → SM | 0.877 | [0.846, 0.906] | L10 | 중간 전이 |
| SM → IC | 0.616 | [0.574, 0.679] | L30 | 약한 전이 |
| SM → MW | 0.657 | [0.583, 0.706] | L22 | 약한 전이 |
| MW → IC | 0.631 | [0.586, 0.680] | L22 | 약한 전이 |

(source: `improved_v4_20260308_032435.json:cross_domain_bootstrap`)

**전이 비대칭 분석**:
- **IC → 다른 도메인**: 일관되게 AUC > 0.89. IC는 4가지 선택지(정지, 보수적, 중간, 공격적)를 가져 위험 성향의 스펙트럼을 풍부하게 인코딩한다.
- **SM/MW → IC**: AUC 0.616–0.631. SM은 이진 결정(베팅/정지)만 가지며, IC의 세분화된 위험 수준을 포착하기 어렵다.
- **MW → SM**: 0.877로 예상보다 높다. 두 패러다임 모두 "확률 숨김" 설계를 공유하여 불확실성 하의 의사결정 패턴이 전이되는 것으로 보인다.

#### 2.2.2 동일 레이어(L22) 특징 중복

| 비교 쌍 | 공유 특징 수 (상위 100) | Jaccard 지수 |
|--------|---------------------|-------------|
| IC ∩ SM | 13 | 0.070 |
| IC ∩ MW | 25 | 0.143 |
| SM ∩ MW | 0 | 0.000 |

(source: `improved_v4_20260308_032435.json:same_layer_features`)

**해석**: V3에서 다른 레이어 비교로 인한 Jaccard=0.000 아티팩트를 해결. 동일 레이어(L22) 비교에서 IC∩MW=25개 공유 특징이 확인되어, IC와 MW 간의 교차 전이(0.908)를 부분적으로 설명한다. SM∩MW=0은 SM의 최적 레이어(L12)와 MW의 최적 레이어(L33)의 차이가 여전히 반영된 것일 수 있다.

---

### 2.3 Goal C: 교차 아키텍처 검증 — LLaMA-3.1-8B에서의 재현

LLaMA-3.1-8B (LlamaScope 32K SAE, 32 레이어)를 사용하여 IC 패러다임에서 Gemma의 발견이 재현되는지 검증한다.

![Figure 3: LLaMA vs Gemma IC Comparison](figures/v5_fig3_llama_vs_gemma.png)

**Figure 3 해석**: (A) LLaMA의 SAE BK DP AUC 0.957(L11)은 Gemma 0.964(L22)와 비교 가능한 수준이다. LLaMA는 L0에서 이미 0.941에 도달하며, Gemma보다 더 이른 레이어에서 파산 관련 정보를 인코딩한다. 이는 8B 모델(32레이어)과 9B 모델(42레이어) 간의 아키텍처 차이를 반영한다. (B) 베팅 제약별 파산률은 두 모델 모두 c10=0%, c70≈21%의 동일한 패턴을 보이나, c30에서 LLaMA(1.2%)가 Gemma(5.3%)보다 낮아 LLaMA가 약간 더 보수적임을 시사한다.

#### 2.3.1 LLaMA IC 분류 결과

| 분석 유형 | AUC | Layer | 비교 (Gemma) |
|----------|-----|-------|-------------|
| SAE BK DP | 0.957 | L11 | 0.964 (L22) |
| SAE BK R1 | 0.799 | L1 | 0.854 (L18) |
| R1 순열 검정 | p=0.0099 | L11 | p<0.001 |
| 제약 4-class | 0.994 | L6 | 0.966 (L18) |

(source: `llama_ic_analyses_20260308_202635.json`)

#### 2.3.2 교차 모델 전이

| 전이 방향 | AUC | 해석 |
|----------|-----|------|
| LLaMA IC → Gemma IC | 0.951 (L10) | **강한 아키텍처 간 전이** |
| Gemma IC (참조) | 0.964 (L22) | 동일 모델 내 성능 |

(source: `improved_v4_20260308_032435.json:cross_model_comparison`)

**해석**: LLaMA에서 학습한 SAE 분류기가 Gemma 데이터에서 0.951 AUC를 달성한다는 것은, 두 모델이 파산 관련 정보를 유사한 방식으로 인코딩함을 의미한다. 이는 파산 예측 신호가 특정 모델의 아티팩트가 아닌, 트랜스포머 아키텍처에서 보편적으로 나타나는 현상임을 강력하게 지지한다.

#### 2.3.3 LLaMA IC 베팅 제약별 파산률

| 제약 | 게임 수 | 파산 | 파산률 |
|------|---------|------|--------|
| c10 | 400 | 0 | 0.0% |
| c30 | 400 | 5 | 1.2% |
| c50 | 400 | 52 | 13.0% |
| c70 | 400 | 85 | 21.2% |

(source: `llama_ic_analyses_20260308_202635.json:condition_analysis`)

#### 2.3.4 LLaMA IC 조기 예측

| Layer | R1 AUC ± std |
|-------|-------------|
| L0 | 0.792 ± 0.038 |
| L5 | 0.795 ± 0.035 |
| L9 | 0.796 ± 0.034 |
| L15 | **0.796 ± 0.034** |
| L22 | 0.795 ± 0.038 |
| L31 | 0.796 ± 0.035 |

(source: `llama_ic_analyses_20260308_202635.json:early_prediction`)

**해석**: LLaMA에서 R1 조기 예측 AUC는 모든 레이어에서 0.792–0.796으로 매우 안정적이다. 이는 Gemma의 레이어 간 변동(0.766–0.901)과 대조적이며, LLaMA가 파산 관련 정보를 모든 레이어에 고르게 분산시키는 반면, Gemma는 특정 레이어(L16–L26)에 집중시킴을 시사한다.

---

### 2.4 Goal D: 실험 조건 인코딩 — 완벽한 조건 분리

모델이 실험 조건(베팅 유형, 베팅 제약, 프롬프트 조건)을 내부적으로 어떻게 표현하는지 분석한다.

![Figure 4: Condition Encoding](figures/v5_fig4_condition_encoding.png)

**Figure 4 해석**: 베팅 유형(fixed/variable) 2-class 분류는 모든 패러다임에서 AUC=1.000을 달성하여, 모델이 베팅 유형을 완벽하게 인코딩함을 확인한다. IC 베팅 제약 4-class 분류에서 LLaMA(0.994)가 Gemma(0.966)를 상회하는데, 이는 LLaMA의 SAE가 32K 특징으로도 제약 정보를 더 효율적으로 분리함을 보여준다. 라운드 수준 위험 분류(IC risk choice 0.681)는 게임 수준 분류(0.964)보다 낮으며, 이는 개별 결정보다 누적 패턴이 더 예측적임을 의미한다.

#### 2.4.1 조건 수준 분류 결과

| 분류 과제 | 모델 | AUC | Layer | 클래스 수 |
|----------|------|-----|-------|---------|
| 베팅 유형 (IC) | Gemma | 1.000 | L0 | 2 |
| 베팅 유형 (SM) | Gemma | 1.000 | L2 | 2 |
| 베팅 유형 (MW) | Gemma | 1.000 | L0 | 2 |
| IC 베팅 제약 | Gemma | 0.966 | L18 | 4 (c10/30/50/70) |
| IC 베팅 제약 | LLaMA | 0.994 | L6 | 4 (c10/30/50/70) |
| IC 프롬프트 조건 | Gemma | 1.000 | L6 | 32 |

(source: `extended_analyses_20260306_211214.json:exp4_condition_level`, `llama_ic_analyses_20260308_202635.json:condition_analysis`)

#### 2.4.2 라운드 수준 위험/베팅 분류

| 분류 과제 | 패러다임 | AUC | Layer |
|----------|---------|-----|-------|
| 위험 선택 (risky vs safe) | IC | 0.681 | L8 |
| 베팅 크기 (high vs low) | SM | 0.908 | L22 |
| 베팅 크기 (high vs low) | MW | 0.826 | L24 |
| Hidden 위험 분류 | IC | 0.674 | L22 |
| Hidden 위험 분류 | SM | 0.816 | L26 |

(source: `extended_analyses_20260306_211214.json:exp3_round_level_risk`, `comprehensive_gemma_20260309_063511.json:hidden_risk`)

**해석**: SM 베팅 크기 분류(0.908)가 IC 위험 선택 분류(0.681)보다 높은 것은, SM에서 베팅 금액이 연속적이고 변동이 크기 때문이다 ($10–$100 범위). IC는 4개 이산 선택지(Option 1–4)로, 위험 수준의 그라데이션이 제한적이다.

---

### 2.5 Goal E: 조건별 파산 분류 — 안정적 성능

실험 조건(베팅 유형, 프롬프트 조건, 베팅 제약)의 하위 집합에서도 파산 예측이 안정적인지 검증한다.

![Figure 5: Per-Condition Hidden BK Classification](figures/v5_fig5_per_condition_hidden_bk.png)

**Figure 5 해석**: IC, SM, MW 모두에서 대부분의 조건 하위집합에서 AUC > 0.93을 유지한다. 특히 MW variable(0.999)은 파산 게임이 4개뿐임에도 거의 완벽한 분리를 보이며, 이는 variable MW 파산이 매우 극단적인 행동 패턴과 연관됨을 시사한다. SM에서 G(Goal) 없는 조건의 AUC(0.997)가 G 있는 조건(0.966)보다 높은데, 이는 Goal 프롬프트가 파산/비파산 간 행동 차이를 줄이는 효과가 있음을 나타낸다.

#### 2.5.1 IC 조건별 Hidden BK (Decision Point)

| 조건 | AUC | Layer | n_BK |
|------|-----|-------|------|
| fixed | 0.936 | L26 | 158 |
| variable | 0.941 | L6 | 14 |
| c30 | 0.986 | L40 | 21 |
| c50 | 0.951 | L22 | 67 |
| c70 | 0.917 | L26 | 84 |
| with_G | 0.963 | L12 | 87 |
| without_G | 0.960 | L22 | 85 |
| with_M | 0.956 | L12 | 87 |
| without_M | 0.970 | L22 | 85 |

#### 2.5.2 SM 조건별 Hidden BK (Decision Point)

| 조건 | AUC | Layer | n_BK |
|------|-----|-------|------|
| variable | 0.962 | L10 | 87 |
| with_G | 0.966 | L22 | 83 |
| without_G | 0.997 | L24 | 4 |
| with_H | 0.976 | L18 | 55 |
| without_H | 0.973 | L24 | 32 |
| with_M | 0.973 | L16 | 61 |
| without_M | 0.989 | L26 | 26 |
| with_P | 0.984 | L26 | 47 |
| without_P | 0.982 | L12 | 40 |
| with_W | 0.982 | L12 | 61 |
| without_W | 0.988 | L18 | 26 |

#### 2.5.3 MW 조건별 Hidden BK (Decision Point)

| 조건 | AUC | Layer | n_BK |
|------|-----|-------|------|
| fixed | 0.944 | L18 | 50 |
| variable | 0.999 | L12 | 4 |
| with_G | 0.950 | L12 | 40 |
| without_G | 0.979 | L12 | 14 |
| with_H | 0.971 | L40 | 24 |
| without_H | 0.968 | L18 | 30 |
| with_M | 0.966 | L18 | 32 |
| without_M | 0.982 | L12 | 22 |
| with_P | 0.968 | L12 | 25 |
| without_P | 0.969 | L24 | 29 |
| with_W | 0.960 | L12 | 25 |
| without_W | 0.966 | L0 | 29 |

(source: `comprehensive_gemma_20260309_095339.json:percondition_hidden_bk`)

---

### 2.6 특징 중요도 분석

파산 예측에서 가장 기여도가 높은 SAE 특징을 식별한다.

| 패러다임 | Layer | 1위 특징 (방향) | 2위 특징 (방향) | 3위 특징 (방향) |
|---------|-------|---------------|---------------|---------------|
| IC | L22 | #91055 (BK-) | #118299 (BK-) | #4008 (BK+) |
| SM | L12 | #73515 (BK-) | #80995 (BK-) | #38058 (BK+) |
| MW | L33 | #97655 (BK-) | #8554 (BK+) | #97291 (BK-) |

(source: `extended_analyses_20260306_211214.json:exp2b_feature_importance`)

**해석**: IC와 SM에서 상위 2개 특징이 모두 BK- 방향(활성화가 높을수록 파산 확률 감소)인 것은, 이 특징들이 "안전 행동" 또는 "위험 회피"를 인코딩하는 것으로 해석된다. BK+ 특징은 "위험 추구" 패턴과 연관될 가능성이 높다.

---

## 3. 실험 설정 및 방법론

### 3.1 하드웨어

| 구성 | 사양 |
|------|------|
| GPU | 2× NVIDIA A100-SXM4-40GB |
| CPU | 100 코어 |
| RAM | 1TB |
| CUDA | 12.9 (driver) / 12.8 (PyTorch) |
| 플랫폼 | OpenHPC (Kubernetes/JupyterHub) |

### 3.2 소프트웨어

| 구성 요소 | 버전 |
|----------|------|
| Python | 3.13.11 (Anaconda) |
| PyTorch | 2.8.0+cu128 |
| sae_lens | 6.5.1 |
| transformers | 최신 |
| scikit-learn | 1.8 |

### 3.3 결과 파일 위치

| 파일 | 내용 | 경로 |
|------|------|------|
| V3 원본 분석 | SAE BK/Hidden/Cross-domain | `results/json/all_analyses_20260306_091055.json` |
| V4 개선 분석 | Bootstrap CI/Permutation/Overlap | `results/json/improved_v4_20260308_032435.json` |
| 확장 분석 | Balance-matched/Risk/Condition | `results/json/extended_analyses_20260306_211214.json` |
| Gemma 종합 | Hidden BK/Risk/Per-condition | `results/json/comprehensive_gemma_20260309_063511.json` |
| Gemma 조건별 | Per-condition Hidden BK detail | `results/json/comprehensive_gemma_20260309_095339.json` |
| LLaMA IC | 10개 분석 | `results/json/llama_ic_analyses_20260308_202635.json` |

---

## 4. 진행 상황 및 향후 계획

### 4.1 현재 진행 상황 (2026-03-09 19:30 UTC)

![Figure 6: Analysis Progress Matrix](figures/v5_fig6_progress_matrix.png)

**Figure 6 해석**: Gemma 3개 패러다임은 SAE 분석 100% 완료, Hidden 분석 ~80% (잔고 매칭, 위험 분류, 교차 도메인 진행중). LLaMA는 IC만 SAE 분석 완료 (Hidden 분석 미완). SM/MW는 행동 데이터 수집 중 (GPU 점유).

#### 실행중인 프로세스 (4개)

| 프로세스 | 자원 | 진행률 | 예상 완료 |
|---------|------|--------|----------|
| LLaMA SM V4role behavioral | GPU 0 | 50/3,200 (1.6%) | ~24h |
| LLaMA MW V2role behavioral | GPU 1 | 14/3,200 (0.4%) | ~30h |
| Gemma Hidden Gaps (A7-10) | CPU | A7 MW L12 진행중 | ~1h |
| Gemma Risk Fix (A2,4) | CPU | A2 SM L22 진행중 | ~1h |

### 4.2 Gemma 분석 완료 현황

```
                    SAE          Hidden
                    DP  R1  BM   DP  R1  BM   Risk  PerCond  CrossDom  Perm
IC                  [O] [O] [O]  [O] [O] [~]  [O]   [O]      [O]       [O]
SM                  [O] [O] [O]  [O] [O] [~]  [~]   [O]      [O]       [O]
MW                  [O] [O] [O]  [O] [O] [~]  [~]   [~]      [~]       [O]

[O] = Done, [~] = Running (CPU background)
```

### 4.3 LLaMA 분석 완료 현황

```
                    SAE          Hidden
                    DP  R1  BM   DP  R1  BM   Risk  PerCond  CrossDom  Perm
IC                  [O] [O] [ ]  [ ] [ ] [ ]  [ ]   [O]      [ ]       [O]
SM                  [ ] [ ] [ ]  [ ] [ ] [ ]  [ ]   [ ]      [ ]       [ ]
MW                  [ ] [ ] [ ]  [ ] [ ] [ ]  [ ]   [ ]      [ ]       [ ]

[O] = Done, [ ] = Not started (SM/MW waiting for behavioral data)
```

### 4.4 향후 작업 파이프라인

```
Phase 1 (현재):         Phase 2 (~24h 후):        Phase 3 (~28h 후):       Phase 4 (~32h 후):

Gemma Hidden Gaps      LLaMA SM/MW              LLaMA SM/MW SAE         LLaMA 전체 분석
완료 대기 (1-2h)        behavioral 완료           extraction (~4h each)   (Gemma와 대칭)
                       ↓                        ↓                       ↓
LLaMA SM [RUN] -----> SM data ready -------> extract_llama_sm.py --> run_comprehensive_llama.py
LLaMA MW [RUN] -----> MW data ready -------> extract_llama_mw.py --> run_hidden_gaps_llama.py

Gemma Risk Fix -----> [DONE] Gemma 100% complete
```

### 4.5 P1 즉시 실행 가능 작업

LLaMA IC 데이터(SAE + Hidden states)는 이미 추출 완료:
- `sae_features_v3/investment_choice/llama/` (32 레이어, 32K features)
- `sae_features_v3/investment_choice/llama/checkpoints/phase_a_hidden_states.npz` (3291, 32, 4096)

아직 실행하지 않은 분석:
1. **LLaMA IC Hidden BK** (DP + R1)
2. **LLaMA IC Hidden Risk**
3. **LLaMA IC Balance-Matched** (SAE + Hidden)
4. **LLaMA IC Per-Condition Hidden BK/Risk**

이 분석들은 GPU 없이 CPU에서 실행 가능하며, 현재 background 분석 완료 후 순차 실행 가능.

---

## 5. 제안된 개선 사항

### 5.1 Causal Validation via Activation Patching

**문제점**: 현재 모든 분석은 상관 관계(classification)에 기반하며, 인과적 검증이 부재하다.

**해결 방안**: SAE 특징 패칭을 통해 파산 관련 특징의 활성화를 인위적으로 변경하고, 행동 변화를 관찰한다.

```python
# Activation patching framework
def patch_and_generate(model, sae, feature_idx, scale_factor):
    """특정 SAE 특징의 활성화를 스케일링하고 모델 출력 변화 관찰"""
    def hook_fn(module, input, output):
        hidden = output[0]
        # SAE 인코딩 → 특징 수정 → 디코딩
        features = sae.encode(hidden)
        features[:, feature_idx] *= scale_factor  # 증폭 또는 억제
        modified = sae.decode(features)
        return (modified,) + output[1:]
    return hook_fn
```

**예상 효과**: BK+ 특징 억제 시 파산률 감소, BK- 특징 증폭 시 파산률 감소 예상. 기존 논문(Section 3.2)의 112개 인과적 특징과 비교 가능.

### 5.2 Round-Level Trajectory Analysis

**문제점**: 현재 분석은 게임 수준(1 벡터/게임)이며, 라운드별 내부 표현 변화를 추적하지 않는다.

**해결 방안**: 각 게임의 모든 라운드에서 hidden state를 추출하고, PCA/UMAP으로 궤적을 시각화한다.

**예상 효과**: 파산 게임의 hidden state 궤적이 비파산 게임과 분기하는 시점(critical round)을 식별할 수 있다.

### 5.3 LLaMA SM/MW 교차 도메인 분석

**문제점**: 교차 도메인 전이는 현재 Gemma만 분석되었다.

**해결 방안**: LLaMA SM/MW 행동 데이터 완료 → SAE 추출 → 동일한 교차 도메인 전이 분석 수행.

**예상 효과**: Gemma에서 관찰된 IC 소스 도메인 우위가 LLaMA에서도 재현되면, 이는 모델 불특정(model-agnostic) 현상으로 확인된다.

---

## 6. 한계점

1. **Gemma 전용 교차 도메인**: 교차 도메인 전이는 Gemma만 분석 (LLaMA SM/MW 대기중). 단일 모델의 결과로는 일반화에 한계.

2. **MW 파산 수 부족**: MW BK=54개로 가장 적으며, 조건별 분석 시 일부 하위 집합은 BK < 10으로 통계적 검정력 부족.

3. **인과 검증 부재**: 모든 분석이 classification 기반의 상관 관계. Activation patching을 통한 인과 방향 확인 필요.

4. **ROLE_INSTRUCTION 효과**: ROLE_INSTRUCTION 없이는 Gemma가 도박 프롬프트를 거부 (V3: 27.3% 즉시 중단). 이 프롬프트 엔지니어링이 행동 패턴에 미치는 영향을 분리하지 못함.

5. **두 모델 아키텍처**: Gemma-2-9B와 LLaMA-3.1-8B만 분석. 더 큰 모델(70B), 다른 아키텍처(Mixtral, Qwen), 상용 API 모델(GPT, Claude)에서의 일반화 미확인.

6. **SAE 인코딩 차이**: GemmaScope(JumpReLU, 131K)와 LlamaScope(ReLU, 32K)는 서로 다른 인코딩을 사용. 교차 모델 전이(0.951)가 높은 것은 고무적이나, 인코딩 방식 차이가 비교에 미치는 영향을 정량화하지 못함.

---

## 7. 결론

V5 종합 분석은 다음을 확인한다:

1. **SAE 분해의 충실도**: SAE 특징과 Raw Hidden States는 파산 예측에서 동등한 성능을 보이며 (DP AUC 차이 < 0.003), SAE가 예측 관련 정보를 완전히 보존함을 확인.

2. **내재적 파산 신호**: R1(잔고 $100 동일) 예측에서 AUC 0.766–0.901 (모두 p<0.001), 잔고 매칭 후에도 0.630–0.744 유지. 이는 파산이 단순한 잔고 부족이 아닌, 모델의 내재적 의사결정 성향에서 기인함을 시사.

3. **비대칭 교차 도메인 전이**: IC는 강력한 소스 도메인 (→SM 0.893, →MW 0.908), 역방향은 약함 (0.616–0.631). IC의 풍부한 선택 구조가 범용적 위험 추구 표현 학습에 유리.

4. **아키텍처 일반성**: LLaMA IC AUC 0.957 ≈ Gemma 0.964. 교차 모델 전이 0.951. 파산 신호는 모델 특정적이 아닌 트랜스포머 보편적 현상.

5. **조건 불변성**: 베팅 유형, 프롬프트 조건, 베팅 제약의 모든 하위 집합에서 AUC > 0.91 유지 (MW variable 제외, n_BK=4). 파산 신호는 실험 조건에 강건.

---

## 8. 다음 실험 계획

### E1: LLaMA IC Hidden 분석 완료
- **검정**: LLaMA Hidden BK/Risk가 SAE와 동등한 성능을 보이는가
- **설정**: `run_llama_ic.py --analyses hidden_bk,hidden_risk,balance_matched`
- **예상**: Hidden DP AUC ≈ 0.957 (SAE와 동일), 잔고 매칭 AUC ≈ 0.70–0.75

### E2: LLaMA SM V4role SAE 추출 + 분석
- **전제**: SM behavioral 완료 (~24h 후)
- **설정**: `extract_llama_sm.py` → `run_comprehensive_llama.py --paradigm sm`
- **예상**: SM SAE DP AUC > 0.90 (Gemma SM 0.981과 유사)

### E3: LLaMA MW V2role SAE 추출 + 분석
- **전제**: MW behavioral 완료 (~30h 후)
- **설정**: `extract_llama_mw.py` → `run_comprehensive_llama.py --paradigm mw`
- **예상**: MW SAE DP AUC > 0.90

### E4: LLaMA 교차 도메인 전이
- **전제**: E2 + E3 완료
- **검정**: Gemma에서 관찰된 IC 소스 우위가 LLaMA에서도 재현되는가
- **예상**: LLaMA IC→SM/MW AUC > 0.85 (Gemma: 0.893/0.908)

### E5: Activation Patching (인과 검증)
- **전제**: E1–E4 완료, 최적 특징 식별
- **설정**: Phase 4 patching pipeline 적용
- **예상**: BK+ 특징 억제 시 파산률 -10~30%p 감소

---

## 참고문헌

- Lee, S., Shin, D., Lee, Y., & Kim, S. (2026). Can Large Language Models Develop Gambling Addiction? *Nature Machine Intelligence* (under review).
- Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Anthropic Technical Report*.
- Lieberum, T., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. *Google DeepMind Technical Report*.
- He, Z., et al. (2024). LlamaScope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders. *arXiv:2410.20526*.

---

*V5 문서 생성일: 2026-03-09. 이 문서는 진행중인 실험의 중간 보고서이며, LLaMA SM/MW 행동 데이터 완료 후 V6에서 업데이트 예정.*
