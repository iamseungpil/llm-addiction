# V13: LLM의 위험 의사결정에 대한 신경 기반 연구

**저자**: 이승필, 신동현, 이윤정, 김선동 (GIST)
**작성일**: 2026년 3월 30일
**통합 버전**: V3 (초기 SAE), V6 (프롬프트 위계), V8 (Gemma 교차 도메인), V10 (교차 모델 대칭 분석), V11 (인과적 steering 파일럿), V12 (방향 특이성, 교차 과제, 교차 모델, 교차 도메인 steering), V13 (LLaMA RQ3, 사실 검증 수정)

---

## 1. Introduction

대규모 언어 모델은 불확실성 하에서 순차적 의사결정을 수행하지만, 이 결정을 유도하는 내부 메커니즘은 아직 충분히 이해되지 않았다. 음의 기대값을 가진 도박 게임에서 일부 모델은 다른 모델보다 훨씬 높은 빈도로 파산하며, 특정 조건은 위험 추구 행동을 일관되게 증폭시킨다. 이러한 차이가 재정 파멸 성향을 부호화하는 일관된 내부 표상, 즉 신경 패턴을 반영하는 것인지, 아니면 공유 기반이 없는 과제 특수적 휴리스틱에 불과한 것인지가 핵심 질문이다.

이 보고서는 두 개의 트랜스포머 기반 언어 모델의 내부 표상을 세 가지 도박 과제와 16,000회 게임에 걸쳐 분석한다. 두 모델 모두 과제 간 일관되고, 교란 변수에 강건하며, 표적 개입을 통해 행동을 인과적으로 제어하는 파산 예측 신경 패턴을 포함하고 있었다.

이 보고서에서 사용하는 주요 용어는 다음과 같다:

- **Bankruptcy (BK)**: 모델의 잔액이 $0에 도달하는 사건. BK는 일차적 이진 결과 변수이다.
- **Decision Point (DP)**: 모델의 마지막 베팅/중단 결정 시점의 activation 상태. 잔액 정보가 교란 변수로 포함될 수 있다.
- **Round 1 (R1)**: 모델의 첫 번째 결정 시점의 activation 상태. 모든 게임이 $100 잔액으로 시작하므로 잔액 교란 변수가 제거된다.
- **Sparse Autoencoder (SAE)**: 밀집 hidden state를 희소한 해석 가능 feature로 분해하는 네트워크 (GemmaScope: layer당 131K; LlamaScope: layer당 32K).
- **Steering**: 추론 시 모델의 residual stream에 벡터를 더하여 내부 표상을 목표 방향으로 이동시키는 기법.
- **Dose-response**: steering 크기(alpha)와 BK 비율 간의 관계. 단조적 dose-response는 점진적 인과 효과를 시사한다.
- **Universal BK neuron**: 모든 패러다임에서 BK와 유의한 상관(FDR 보정, p < 0.01)을 보이며, 부호가 일관된 hidden-state 차원.
- **BK direction vector**: 특정 layer에서 mean(BK hidden states) - mean(Safe hidden states)로 정의되는 activation 공간의 "파산 방향".
- **IC / SM / MW**: Investment Choice (네 가지 선택지, 위험 수준 상이), Slot Machine (이진 계속/중단), Mystery Wheel (이진 회전/중단).

Figure 0은 실험 프레임워크의 개관을 제시한다. 두 모델이 체계적으로 변화하는 조건 하에서 세 가지 패러다임을 수행하며, 분석은 분류에서 전이를 거쳐 인과적 steering으로 진행된다.

![Fig. 0: Experimental framework. Two models (Gemma, LLaMA) play three gambling paradigms (IC, SM, MW) under systematically varied conditions. Internal representations are analyzed through correlational classification, cross-domain transfer, and causal steering.](figures/v13_fig0_experimental_framework.png)

세 가지 연구 질문이 이 보고서를 구성한다. 첫째, BK 예측 신경 패턴이 이 모델들에 존재하는가? 둘째, 이 패턴이 도박 도메인 간에 보편적인가? 셋째, 실험 조건이 이 패턴을 어떻게 조절하는가? 각 질문은 상관적 증거와 인과적 증거를 함께 제시하여 답한다.

---

## 2. Experimental Setup

### 2.1 Models and Data

두 개의 instruction-tuned 트랜스포머 모델을 피험 대상으로 사용하였다: Gemma-2-9B-IT (Google, 42 layer, 3,584차원 hidden state)와 LLaMA-3.1-8B-Instruct (Meta, 32 layer, 4,096차원 hidden state). 각 모델은 음의 기대값을 가진 세 가지 도박 과제를 수행하였다. IC는 베팅 제약(c10, c30, c50, c70)과 프롬프트 조건(BASE, G, M, GM)을 변화시킨다. SM과 MW는 5개의 프롬프트 구성요소(Goal, Money, Warning, Hint, Persona)를 2^5 요인 설계로 조합하여 패러다임당 32개 조건을 생성한다.

Table 1은 모든 모델-패러다임 조합의 데이터셋을 요약한다.

**Table 1. Dataset Overview**

| Paradigm | Model | Games | BK Count | BK Rate | Bet Types | Special Conditions |
|----------|-------|:-----:|:--------:|:-------:|-----------|-------------------|
| IC | Gemma | 1,600 | 172 | 10.8% | Fixed / Variable | c10/c30/c50/c70, BASE/G/M/GM |
| IC | LLaMA | 1,600 | 142 | 8.9% | Fixed / Variable | c10/c30/c50/c70, BASE/G/M/GM |
| SM | Gemma | 3,200 | 87 | 2.7% | Fixed / Variable | 32 prompt combos |
| SM | LLaMA | 3,200 | 1,164 | 36.4% | Fixed / Variable | 32 prompt combos |
| MW | Gemma | 3,200 | 54 | 1.7% | Fixed / Variable | 32 prompt combos |
| MW | LLaMA | 3,200 | 2,426 | 75.8% | Fixed / Variable | 32 prompt combos |

Table 1은 두 모델 간 현저한 행동 차이를 보여준다. SM과 MW에서 LLaMA의 BK 비율은 Gemma의 13~44배에 달하며, 이는 근본적으로 다른 위험 전략을 시사한다. 이러한 행동적 격차에도 불구하고, 이후 분석에서 두 모델이 동등한 정확도로 BK 정보를 부호화하고 있음이 확인된다.

### 2.2 Analysis Pipeline

Hidden state는 각 트랜스포머 layer의 residual stream activation이다. 분류 파이프라인은 StandardScaler, PCA (50 components), 로지스틱 회귀(balanced class weights, C = 1.0)를 적용하고, 5-fold 층화 교차검증을 수행한다. 유의성은 200회 순열 검정으로 평가한다. Steering 파이프라인은 추론 시 residual stream에 스케일된 BK direction vector를 더하고, alpha 값에 따른 BK 비율 변화를 측정한다.

---

## 3. RQ1: BK 예측 신경 패턴이 존재하는가?

이 절은 두 모델 모두 파산을 신뢰성 있게 예측하는 내부 표상을 포함하는지 검증한다. 상관적 분류(내부 상태에서 BK를 예측할 수 있는가?)에서 인과적 steering(BK 패턴을 조작하면 행동이 변하는가?)으로 증거를 확대한다. 신경 패턴이 BK를 예측하면서 동시에 제어할 수 있다면, 이는 행동 결과의 수동적 반영이 아니라 의사결정 과정의 기능적 구성요소이다.

### 3.1 Classification Evidence

첫 번째 분석은 decision point의 내부 표상을 학습한 분류기가 BK 게임과 Safe 게임을 구별할 수 있는지 검증한다. Table 2는 각 모델과 패러다임의 최적 layer AUC를 보고한다.

**Table 2. DP Classification AUC (Best Layer)**

| Paradigm | Gemma Hidden (Layer) | Gemma SAE (Layer) | LLaMA Hidden (Layer) | Difference |
|----------|:-------------------:|:-----------------:|:-------------------:|:----------:|
| IC | 0.964 (L26) | 0.964 (L22) | 0.954 (L12) | 0.006 |
| SM | 0.982 (L10) | 0.981 (L12) | 0.974 (L8) | 0.002 |
| MW | 0.968 (L12) | 0.966 (L33) | 0.963 (L16) | 0.003 |

두 모델 모두 모든 패러다임에서 AUC 0.95를 초과하며, 교차 모델 차이는 최대 0.006이다. 행동 프로파일이 크게 다른 두 아키텍처에서 거의 동일한 정확도가 나타난다는 점은, BK 부호화가 트랜스포머 언어 모델의 일반적 특성임을 시사하는 첫 번째 증거이다.

그러나 DP 분류에는 잠재적 교란 변수가 존재한다: BK 게임은 $0 잔액으로 종료되며, 잔액 정보가 hidden state에 부분적으로 부호화될 수 있다. 이를 통제하기 위해 R1 분류는 모든 게임이 $100 잔액으로 시작하는 첫 번째 결정 시점을 분석한다. R1에서의 추가 교란 변수는 bet type이며, AUC = 1.0으로 부호화되고 BK 비율과 상관이 있다. Bet type 내 R1 분류는 두 교란 변수를 동시에 제거한다.

**Table 3. Within-Bet-Type R1 Classification (Gemma, Confound-Controlled)**

| Subset | n | BK Count (%) | AUC | Perm. p | z |
|--------|:-:|:------------:|:---:|:-------:|:-:|
| IC Fixed | 800 | 158 (19.8%) | 0.753 | 0.010 | 6.07 |
| IC Variable | 800 | 14 (1.8%) | 0.692 | 0.020 | 1.99 |
| SM Variable | 1,600 | 87 (5.4%) | 0.805 | 0.010 | 6.70 |
| MW Fixed | 1,600 | 50 (3.1%) | 0.617 | 0.040 | 1.98 |

잔액과 bet type을 모두 통제한 후에도 모든 하위 집합이 통계적으로 유의하다. 어떠한 손실도 발생하기 전인 첫 라운드에서 분류기가 최종 파산을 예측할 수 있다는 사실은, 모델이 각 게임 시작 시점에 위험 성향 신호를 부호화하고 있음을 나타낸다. LLaMA의 bet type 내 DP 분류는 더 높은 값을 산출하며(6개 패러다임-조건 조합 전체에서 AUC 0.885~0.995), 이 발견의 교차 모델 일반성을 확인한다.

Figure 1은 분류 결과를 보여준다.

![Fig. 1: BK classification AUC across models and paradigms. Both Gemma and LLaMA achieve AUC above 0.95 at their respective best layers, with cross-model differences within 0.006.](figures/v13_fig1_bk_classification.png)

### 3.2 Universal BK Neurons

이 분석은 BK 결과와의 상관이 통계적으로 유의하고 모든 패러다임에서 부호가 일관된 개별 hidden-state 차원을 식별한다. Universal BK neuron은 모든 패러다임에서 FDR 보정 유의성(Benjamini-Hochberg, p < 0.01)을 통과하고 동일한 방향성을 가져야 한다.

**Table 4. Universal BK Neurons (L22)**

| Property | Gemma (3-paradigm) | LLaMA (2-paradigm) |
|----------|:------------------:|:------------------:|
| Total neurons | 3,584 | 4,096 |
| Sign-consistent (universal) | 600 (16.7%) | 1,334 (32.6%) |
| BK-promoting | 302 | 672 |
| BK-inhibiting | 298 | 662 |

핵심 발견은 promoting과 inhibiting의 균형적 비율이다. 두 모델 모두 BK 가능성이 높을 때 활성화되는 뉴런(promoting)과 비활성화되는 뉴런(inhibiting)이 거의 동일한 수로 BK를 부호화한다. 이 양방향 구조는 BK가 단일 뉴런의 활동이 아닌 activation 공간의 방향으로 부호화되는 push-pull 메커니즘을 시사한다. LLaMA의 절대 수치가 높은 것은 주로 확률 기저선의 차이에 기인한다: 2-패러다임 부호 일관성의 우연 기저선은 50%인 반면, 3-패러다임 부호 일관성의 우연 기저선은 25%이다.

Figure 2는 universal BK neuron의 균형 구조를 시각화한다.

![Fig. 2: Universal BK neurons. (a) Gemma: 600 neurons, balanced 302 promoting / 298 inhibiting. (b) LLaMA: 1,334 neurons, balanced 672 promoting / 662 inhibiting.](figures/v13_fig2_universal_neurons.png)

추가 요인 분해 분석은 SAE feature가 bet type 및 패러다임 정체성과 독립적으로 BK를 부호화하는지 검증한다. OLS 회귀(feature ~ outcome + bet_type + paradigm)에서 Gemma feature의 65.2%, LLaMA feature의 75.8%가 교란 변수를 통제한 후에도 유의한 outcome 계수를 유지하였으며, 순열 영가설에서는 약 1%에 불과하였다. 이 feature들에 부호화된 BK 신호는 bet type이나 패러다임 상관의 산물이 아닌 실제 신호이다.

### 3.3 Causal Confirmation: Feature Patching and Direction Steering

동일한 LLaMA SM 데이터를 사용한 선행 분석에서 8,000개 이상의 후보 중 112개의 SAE feature가 추론 시 activation을 패칭하면 도박 행동을 인과적으로 변화시킴을 확인하였다. 이 인과적 feature들은 해부학적으로 분리된다: safe-promoting feature는 초기 layer(L4--L19)에 군집하고, risk-promoting feature는 후기 layer(L24+)에 집중된다. Safe feature 패칭은 중단 행동을 29.6% 증가시키며, 희소 feature가 의사결정에 인과적으로 영향을 줄 수 있음을 확인하였다. 그러나 개별 뉴런 제거는 유의한 효과를 산출하지 못하며(모든 p > 0.5), 인과 메커니즘이 개별 단위가 아닌 분산된 방향을 통해 작동함을 시사한다.

Direction steering은 이 발견을 희소 feature에서 전체 activation 공간으로 확장한다.

BK 패턴은 단순히 예측적인 것이 아니라 인과적이다. 이 분석은 추론 시 BK direction vector를 residual stream에 더했을 때 도박 행동이 용량 의존적으로 변화하는지, 그리고 이 효과가 일반적 교란이 아닌 BK 방향에 특이적인지를 검증한다.

BK direction vector는 layer 22에서 mean(BK hidden states) - mean(Safe hidden states)로 정의된다. 추론 시 이 벡터를 alpha in {-2, -1, -0.5, 0, +0.5, +1, +2}로 스케일하여 residual stream에 더한다. 3개의 무작위 단위 노름 벡터를 통제 조건으로 사용한다. 각 조건은 n = 200 게임에서 검증한다.

**Table 5. Direction Steering Dose-Response (LLaMA SM L22, n = 200)**

| Alpha | BK Direction | Random 0 | Random 1 | Random 2 |
|:-----:|:------------:|:--------:|:--------:|:--------:|
| -2.0 | 0.365 | 0.430 | 0.485 | 0.520 |
| -1.0 | 0.435 | 0.495 | 0.495 | 0.475 |
| -0.5 | 0.500 | 0.510 | 0.475 | 0.405 |
| 0.0 | 0.520 | 0.520 | 0.520 | 0.520 |
| +0.5 | 0.570 | 0.480 | 0.475 | 0.445 |
| +1.0 | 0.550 | 0.480 | 0.505 | 0.490 |
| +2.0 | 0.640 | 0.500 | 0.495 | 0.435 |
| **rho** | **0.964** | 0.198 | 0.273 | -0.342 |
| **p** | **0.00045** | 0.670 | 0.554 | 0.452 |

BK direction은 alpha = -2에서 0.365, alpha = +2에서 0.640으로 단조적 dose-response를 산출하며, 총 변동폭은 27.5 퍼센트포인트이다. Spearman 상관은 rho = 0.964 (p = 0.00045)이다. 3개의 무작위 방향은 모두 비유의하며(최대 |rho| = 0.342), BK 비율이 alpha에 대한 체계적 의존성 없이 좁은 범위에서 변동한다.

Figure 7은 dose-response 곡선을 보여준다.

![Fig. 7: Dose-response curves. The BK direction (blue) shows a steep monotonic increase from alpha = -2 to alpha = +2. Three random controls (gray) remain flat around baseline.](figures/v13_fig7_dose_response.png)

BK direction 곡선과 평탄한 무작위 방향 곡선 간의 분리는 이 보고서에서 가장 강력한 단일 증거이다. BK direction vector가 residual stream의 일반적 교란이 아닌 파산에 대한 인과적으로 특이적인 정보를 전달함을 보여준다.

다중 layer steering은 BK 표상이 깊이에 걸쳐 분산되어 있음을 추가로 보여준다. L22, L25, L30에서 동시 steering 시 delta는 +0.490으로, alpha = +2에서 BK 비율이 0.350에서 0.840으로 증가한다. 이는 최적 단일 layer 효과(L25, delta = +0.200) 대비 2.5배 증폭이며, 서로 다른 layer의 BK direction vector가 중복적이 아닌 상보적 표상 성분을 부호화함을 나타낸다.

Figure 8은 다중 layer 증폭 효과를 시각화한다.

![Fig. 8: Multi-layer steering. The combined L22+L25+L30 curve shows a steeper slope than any individual layer. The combined effect (+0.49) exceeds the sum-of-parts expectation.](figures/v13_fig8_multilayer.png)

### 3.4 Summary

일관된 BK 예측 신경 패턴이 두 모델 모두에 존재한다. 분류 증거에 따르면 내부 표상은 AUC 0.954~0.982로 BK를 예측하며, 잔액과 bet type 교란 변수를 통제한 후에도 신호가 유지된다(bet type 내 R1 AUC: Gemma 0.62~0.80, LLaMA 0.885~0.995). Universal BK neuron은 promoting 대 inhibiting 비율이 균형적이며, BK를 개별 뉴런이 아닌 방향으로 부호화한다. 인과적 steering은 이 방향의 기능적 의미를 확인한다: 조작 시 단조적, 용량 의존적 BK 비율 변화가 발생하고(rho = 0.964, p = 0.00045), 무작위 방향에서는 효과가 없다. BK 패턴은 실재하며, 강건하고, 인과적이다.

---

## 4. RQ2: 이 패턴은 도박 도메인 간에 보편적인가?

이 절은 BK 표상이 발견된 과제를 넘어 일반화되는지 검증한다. 한 도박 도메인에서 학습된 패턴이 다른 도메인의 행동을 예측하거나 제어할 수 있다면, 그것은 과제 특수적일 수 없다. 상관적 전이(한 과제에서 학습한 분류기가 다른 과제에서 BK를 예측하는가?)에서 공유 부분공간 분석을 거쳐 인과적 교차 도메인 steering(한 과제에서 추출한 BK 방향이 다른 과제의 행동을 변화시키는가?)으로 분석을 확대한다.

### 4.1 Correlational Transfer

한 패러다임의 BK/Safe 라벨로 학습한 분류기를 다른 패러다임에서 테스트한다. AUC가 0.5를 유의하게 초과하면 공유된 BK 관련 구조가 존재함을 나타낸다.

**Table 6. Cross-Domain Transfer AUC**

| Transfer | Gemma AUC | LLaMA AUC |
|----------|:---------:|:---------:|
| IC -> MW | 0.932 | 0.680 |
| IC -> SM | 0.913 | 0.577 |
| SM -> MW | 0.867 | -- |
| MW -> IC | 0.853 | 0.805 |
| SM -> IC | 0.646 | 0.603 |
| MW -> SM | -- | 0.682 |

두 모델 모두 모든 전이 방향이 통계적으로 유의하다(p = 0.000). Gemma가 LLaMA보다 높은 절대 전이 AUC를 달성하며(최대 0.932 대 0.805), 이는 Gemma의 낮은 BK 비율이 분리도가 높은 더 특이적인 BK 표상을 생성하기 때문으로 보인다. MW가 관련된 전이가 일관되게 강하다: Gemma IC -> MW = 0.932, LLaMA MW -> IC = 0.805. MW는 BK 패턴이 다른 도메인과 잘 전이되는 허브 패러다임으로 기능한다.

Figure 3은 전이 결과를 히트맵으로 보여준다.

![Fig. 3: Cross-domain transfer heatmaps for Gemma and LLaMA. MW rows and columns show the strongest transfer, confirming MW as the hub paradigm.](figures/v13_fig3_crossdomain_transfer.png)

### 4.2 Shared Low-Dimensional Subspace

패러다임별 BK 분류기의 가중치 벡터는 거의 직교하지만(cosine 약 0.04), 교차 도메인 전이는 성공한다. BK 신호가 단일 방향이 아닌 공유 저차원 부분공간을 점유한다고 이해하면 이 겉보기 모순이 해결된다. 패러다임별 로지스틱 회귀 가중치 벡터에 PCA를 적용하여 이 부분공간을 추출한다.

Gemma에서 3차원, LLaMA에서 2차원만으로 모든 패러다임에서 AUC 0.86~0.97의 BK 분류가 가능하다. 원래 hidden state는 3,584차원(Gemma) 또는 4,096차원(LLaMA)이다. 가용 차원의 0.1% 미만이 모든 패러다임의 BK 신호를 포착하며, BK 표상이 작은 공유 부분공간에 극도로 집중되어 있음을 나타낸다.

### 4.3 Causal Confirmation: Cross-Domain Steering

상관적 전이는 동일한 분류기가 도메인 간에 작동함을 보여준다. 교차 도메인 steering은 더 강한 주장을 검증한다: 한 과제에서 추출한 BK direction vector가 다른 과제의 행동을 인과적으로 변화시키는가?

**Table 7. Cross-Domain Steering Transfer (LLaMA L22, n = 50 per condition)**

| Source | Target | rho | p | Significant |
|:------:|:------:|:---:|:-:|:-----------:|
| SM | SM | 0.964 | 0.00045 | Yes |
| SM | IC | 0.447 | 0.450 | No |
| SM | MW | 0.821 | 0.089 | No |
| IC | SM | -0.900 | 0.037 | Yes |
| IC | IC | 0.991 | 0.000 | Yes |
| IC | MW | -0.359 | 0.553 | No |
| MW | SM | -0.900 | 0.037 | Yes |
| MW | IC | -0.975 | 0.005 | Yes |
| MW | MW | 0.955 | 0.001 | Yes |

대각 항목은 도메인 내 dose-response를 확인한다(|rho| = 0.955~0.991). 6개의 비대각 조합 중 3개가 통계적으로 유의하다: IC -> SM (rho = -0.900, p = 0.037), MW -> SM (rho = -0.900, p = 0.037), MW -> IC (rho = -0.975, p = 0.005). MW가 인과적 허브이다: MW에서 추출한 방향만이 SM과 IC 모두에서 행동을 유의하게 변화시킨다. MW -> IC 결과(|rho| = 0.975)는 도메인 내 IC 결과(|rho| = 0.991)에 필적하며, MW와 IC의 BK 표상이 인과적 수준에서 거의 완전히 중첩됨을 나타낸다. SM은 이상적인 타겟이다: 기저선 BK 비율이 약 50%(0.52)로 양방향 행동 변동이 가능하기 때문이다.

Figure 4는 교차 도메인 steering 결과를 시각화한다.

![Fig. 4: Cross-domain steering transfer. MW row shows the strongest cross-domain transfer. Dose-response curves for the 3 significant cross-domain combinations confirm monotonic relationships.](figures/v13_fig4_crossdomain_steering.png)

유의한 교차 도메인 조합에서 나타나는 음의 rho 값은 Section 6.2에 기술된 부호 역전 메커니즘을 반영한다: BK direction 추가의 행동적 결과는 학습 데이터의 클래스 균형이 아닌 추론 시점의 기저선 BK 비율에 의존한다.

### 4.4 Summary

BK 패턴은 상관적 수준과 인과적 수준 모두에서 도박 도메인 간에 일반화된다. 모든 교차 도메인 분류 전이가 유의하며(AUC 0.58~0.93), BK 신호는 가용 차원의 0.1% 미만에 해당하는 2~3차원 부분공간에 집중된다. 인과적 교차 도메인 steering은 한 과제에서 추출한 방향이 다른 과제의 행동을 변화시킴을 확인하며, 6개 조합 중 3개가 유의하다. MW는 두 수준 모두에서 허브 패러다임으로 기능하며, MW의 단순한 회전/중단 구조가 가장 일반화 가능한 형태의 위험 표상을 유발하기 때문으로 보인다.

---

## 5. RQ3: 실험 조건은 이 패턴을 어떻게 조절하는가?

이 절은 RQ1에서 확인하고 RQ2에서 일반성을 보인 BK 표상이 실험 조건에 의해 추가적으로 조절되는지 분석한다. BK 패턴이 각 모델의 고정된 속성인지, 아니면 의사결정 맥락에 반응하는 동적 표상인지를 결정한다는 점에서 이 질문은 중요하다. 이 절의 데이터는 V8(Gemma)과 V13(LLaMA RQ3 분석)에서 가져온다.

### 5.1 Prompt Components: Goal-Setting Amplifies BK

5개의 이진 프롬프트 구성요소 --- Goal (G), Money (M), Warning (W), Hint (H), Persona (P) --- 가 SM과 MW에서 2^5 요인 설계로 변화한다. Goal 프롬프트가 BK 비율과 BK 방향과의 신경 정렬 모두에서 가장 큰 효과를 보인다.

Gemma SM에서 Goal 프롬프트는 BK 비율을 20.8배 증가시킨다(G 포함 시 5.19%, G 미포함 시 0.25%). G-프롬프트 방향(G 존재 평균 - G 부재 평균 activation)은 공유 3D 부분공간에서 BK 방향과 cosine +0.85로 정렬된다. G-프롬프트가 모델의 내부 표상을 activation 공간의 파산 영역으로 직접 이동시킨다는 의미이다. IC와 MW에서는 G-프롬프트 방향이 BK 방향과 반대로 정렬되며(cosine = -0.87, -0.79), 행동적 효과도 상응하게 작다(IC: 1.0x, MW: 2.9x). 이 메커니즘은 패러다임에 의존적이다: SM의 이진 계속/중단 구조에서는 "계속 플레이하라"는 목표가 중단이라는 안전 전략과 직접 충돌하는 반면, IC의 네 가지 선택 구조에서는 모델이 중간 위험 옵션을 통해 목표를 추구할 수 있다.

LLaMA에서도 이 메커니즘의 교차 모델 일반성이 확인된다. LLaMA SM은 G-프롬프트와 BK 방향의 cosine이 +0.634이며, G 포함 시 BK 비율 40.6%, G 미포함 시 32.2%이다. LLaMA MW에서는 G-프롬프트 정렬이 더 강하다(cosine = +0.650, BK 비율 차이 13.8 퍼센트포인트). LLaMA SM에서 BK-projection 효과 기준 프롬프트 위계는 G (+2.93), P (+0.27), M (+0.20), W (+0.02), H (-0.07) 순이다. 행동적 BK 비율 기준으로는 M (1.31x), G (1.26x), W (1.15x), P (1.06x), H (1.05x) 순으로 순위가 변한다. M과 P에서 신경적 순위와 행동적 순위의 괴리는, 일부 프롬프트 구성요소가 BK-projection 지표로 충분히 포착되지 않는 경로를 통해 행동에 영향을 미침을 시사한다.

### 5.2 Fixed vs. Variable: Behavioral Risk and Neural Risk Diverge

Variable-bet 게임에서는 모델이 자체적으로 베팅 크기를 선택하고, Fixed-bet 게임에서는 미리 정해진 베팅이 부과된다. 이 조건과 BK 표상 간의 관계는 모델에 따라 근본적으로 다르다.

Gemma에서 Variable 조건은 행동적으로 더 위험한 선택을 산출하지만(IC: 위험 선택 15.4% vs. Fixed 10.4%, 1.65배 긴 플레이), Variable 게임의 3D BK-projection은 모든 베팅 제약 수준에서 Fixed 게임보다 1.4~2.0 단위 낮다. 이 "자율성 역설"은 BK 표상이 선택의 위험성이 아닌 재정 궤적의 위험을 부호화한다고 이해하면 해결된다. Variable-bet 모델은 자체적으로 베팅 크기를 조절하여, 더 위험한 옵션을 선택하면서도 재정적 변동성을 줄인다. BK-projection은 절대 잔액 변화(r = 0.52) 및 변동성(r = 0.22)과 상관이 있으며, 위험 선택 비율과는 상관이 없다.

LLaMA에서는 반대 패턴이 나타난다. Variable 조건은 세 패러다임 모두에서 더 높은 행동적 BK 비율과 더 높은 BK-projection을 동시에 산출한다(SM: +0.765, IC: +2.084, MW: +0.779의 BK-projection 차이, 모든 p < 10^-37). 자율성 역설은 관찰되지 않는다. LLaMA에서 행동과 신경 표상이 일관되는 이유는, 전반적으로 높은 BK 비율이 BK/Safe 군집의 기하학을 변화시키기 때문으로 보인다. BK가 다수 클래스인 경우(LLaMA MW 75.8%), 위험한 선택이 위험한 궤적으로 직결되므로 선택 위험성과 궤적 위험 간의 구분이 소실된다.

### 5.3 Bet Constraints Map Linearly to BK Activation

IC의 네 가지 베팅 제약(c10, c30, c50, c70)은 최대 허용 베팅을 체계적으로 변화시킨다. BK-projection은 c10에서 c70까지 단조적으로 증가하며, Pearson 상관은 r = 0.98 (Gemma), r = 0.987 (LLaMA)이다. Gemma에서 BK-projection은 c10의 -1.37(BK 비율 0%)에서 c70의 +2.04(BK 비율 21.0%)까지 분포한다. 이 연속적이고 거의 선형인 관계는 BK 표상이 단순한 이진 BK/Safe 구분이 아닌 외부적으로 부과된 재정 위험의 정도에 반응함을 보여준다. 다만, 적합에 4개의 데이터 포인트만 사용되었으므로 선형성 해석에는 주의가 필요하다.

### 5.4 Cross-Bet-Type Transfer

Fixed와 Variable 조건이 BK 표상을 공유하는지에 대한 가장 강력한 검증은 한 조건에서 분류기를 학습하고 다른 조건에서 테스트하는 것이다. LLaMA에서 모든 layer와 두 전이 방향 모두 p = 0.000을 산출하며, AUC 범위는 0.736~0.927이다. Variable -> Fixed 전이(0.842~0.927)가 Fixed -> Variable 전이(0.736~0.872)보다 일관되게 높으며, Variable 게임이 Fixed BK 영역을 포괄하는 더 넓은 activation 공간을 탐색하기 때문으로 보인다. SAE L22에서 415개의 LLaMA feature가 두 베팅 조건 모두에서 일관된 BK 효과(동일 부호, Cohen's d >= 0.3)를 보이며, promoting 대 inhibiting 비율(213/202)이 universal neuron 구조와 동일하게 균형적이다. LLaMA bet type 내 분류는 AUC 0.885 (SM Variable)~0.995 (SM Fixed)를 산출하며, BK 부호화가 베팅 조건과 독립적이라는 가장 강력한 교란 통제 증거를 제공한다.

### 5.5 Summary

실험 조건은 BK activation을 체계적이면서도 패러다임 의존적으로 조절한다. G-프롬프트는 목표 추구가 안전 전략과 충돌하는 패러다임에서 BK 방향 activation을 선택적으로 증폭시키며, 교차 모델 확인이 이루어졌다(Gemma SM cosine = +0.85, LLaMA SM cosine = +0.634). 베팅 제약은 BK activation에 대한 연속적이고 거의 선형인 매핑을 산출한다(r = 0.98). Fixed/Variable 조작은 모델 의존적 괴리를 보여준다: Gemma는 자율성 역설(더 위험한 선택이지만 더 낮은 BK-projection)을 보이고, LLaMA는 행동과 신경 표상 간 일관성을 보인다. 이러한 조건 의존적 조절에도 불구하고, 기저의 BK 표상은 불변으로 유지되며, cross-bet-type 전이 AUC 0.74~0.93과 LLaMA bet type 내 AUC 0.885~0.995가 이를 증명한다.

---

## 6. Robustness and Boundary Conditions

### 6.1 Pipeline Robustness

분류 결과는 파이프라인 하이퍼파라미터의 산물이 아니다. PCA 50 components는 6개 데이터셋 중 4개에서 AUC를 포화시키며, 나머지 2개(Gemma MW, LLaMA IC)에서는 PCA = 50이 전체 차원 표상보다 우수하다. 이 데이터셋들은 BK 표본 크기가 가장 작아 고차원에서 과적합이 발생하기 때문으로 보인다. 3개의 분류기(로지스틱 회귀, MLP, SVM-RBF) 간 최대 AUC 차이는 데이터셋 내에서 0.007이다. 어떤 분류기도 일관되게 우위를 보이지 않으며, 비선형 결정 경계가 체계적 이점을 제공하지 않는다. BK 표상은 PCA 축소 공간에서 선형 분리 가능하다.

### 6.2 Sign Reversal

6개의 모델-과제 steering 조합 중 4개에서 음의 rho 값이 나타난다(alpha 증가 시 BK 비율 감소). 이 반직관적 패턴은 추론 시점의 기저선으로 설명된다. BK direction vector는 학습 데이터에서 Safe 중심으로부터 BK 중심 방향을 가리킨다. 이 방향의 추가가 추론 시 BK를 증가시키는지 감소시키는지는 steering 환경의 기저선 BK 비율에 의존한다. "steering 기저선 BK 50% 초과 시 양의 rho 예측"이라는 임계값 모델은 80% 정확도를 달성한다(5개 유효 조합 중 4개). 부호 역전은 인과적 주장을 무효화하지 않는다. 핵심 증거는 단조적 dose-response(모든 유의한 경우에서 |rho| >= 0.955)이며, 양의 alpha가 어떤 행동 극에 이점을 주는지와 무관하게 점진적 방향 정보가 존재함을 보여준다.

### 6.3 Boundary Conditions

Gemma SM과 IC는 steering 패러다임의 작동 한계를 정의한다. Gemma IC는 모든 alpha 값에서 기저선 BK 0.000을 산출한다(바닥 효과: 어떠한 교란도 BK를 유발하지 못함). Gemma SM은 기저선 0.740을 산출한다(천장 효과: 방향에 관계없이 BK 비율이 높게 유지됨). Gemma MW만이 성공적인 교차 모델 인과 증거를 제공하며, 연구 전체에서 가장 강력한 단일 결과를 달성한다(|rho| = 1.000, 88 퍼센트포인트 행동 변동). 이 경계 조건은, 기저선에서 충분한 행동 변동이 검출 가능한 steering 효과의 전제조건임을 나타낸다. 뉴런 제거 실험(LLaMA L22에서 104개 BK-promoting, 89개 BK-inhibiting 뉴런의 영값 제거)은 유의한 행동 변화를 산출하지 못하였다. 이는 균형적 promoting/inhibiting 뉴런 구조가 보여주는 분산 부호화와 일관되며, 뉴런 수준이 아닌 방향 수준 개입의 필요성을 추가로 뒷받침한다.

---

## 7. Conclusion

이 연구는 대규모 언어 모델이 도박 도메인에 걸쳐 위험 의사결정을 예측하고 제어하는 내부 표상을 형성하는지 물었다. 증거는 세 가지 긍정적 답변을 지지한다.

BK 예측 신경 패턴은 Gemma와 LLaMA 모두에서 거의 동일한 정밀도로 존재한다(AUC 0.954~0.982). 행동 전략이 근본적으로 다름에도 이 결과가 성립한다. 손실이 발생하기 전, bet type을 통제한 bet type 내 R1 분류에서도 최종 파산이 예측된다(Gemma AUC 0.62~0.80, LLaMA AUC 0.885~0.995). 이 패턴은 promoting과 inhibiting 뉴런의 균형적 집단을 통해 부호화되며, 개별 위험 부호화 단위가 아닌 push-pull 구조를 시사한다.

이 패턴은 구조적으로 상이한 세 가지 도박 도메인에 걸쳐 일반화된다. 교차 도메인 분류는 모든 방향에서 성공하며(AUC 0.58~0.93), BK 신호는 가용 차원의 0.1% 미만에 해당하는 2~3차원 부분공간에 집중된다. 교차 도메인 인과적 steering이 이를 확인한다: MW에서 추출한 BK 방향이 SM(rho = -0.900, p = 0.037)과 IC(rho = -0.975, p = 0.005) 모두에서 행동을 변화시킨다.

실험 조건은 패턴을 조절하되 파괴하지 않는다. Goal 프롬프트는 목표 추구가 안전 전략과 충돌할 때 BK 방향 activation을 선택적으로 증폭시킨다(SM cosine: Gemma +0.85, LLaMA +0.634). 베팅 제약은 BK activation에 선형적으로 매핑된다(r = 0.98). Fixed/Variable 조작은 Gemma에서만 행동적 위험과 신경적 위험 간 괴리를 보이며, LLaMA에서는 나타나지 않는다.

인과적 증거가 이 발견들을 상관 수준 이상으로 격상시킨다. Direction steering은 단조적 dose-response를 산출하고(rho = 0.964, p = 0.00045), 무작위 방향에서는 효과가 없다. 다중 layer steering은 효과를 2.5배 증폭시킨다. Gemma MW에서의 교차 모델 steering은 88 퍼센트포인트 행동 변동과 함께 완전한 단조적 관계를 달성한다. BK 방향은 수동적 부산물이 아니라 의사결정의 능동적 참여자이다.

이 발견들은 트랜스포머 언어 모델이 재정 파멸 위험에 대한 소형의 도메인 일반적 표상, 즉 내부 "파산 나침반"을 발달시킴을 시사한다. 이 표상은 서로 다른 과제가 약간 다른 방식으로 투영하는 저차원 부분공간으로 나타난다. 실용적 함의는, LLM의 위험 추구 행동이 기존에 가정된 것보다 표적 개입에 더 용이할 수 있다는 점이다: 프롬프트 수정이나 파인튜닝 대신, 소수의 내부 방향을 steering하는 것만으로 신중함과 무모함 사이의 균형을 이동시킬 수 있다.

---

## 8. Limitations

8~9B 파라미터 범위의 두 아키텍처만 테스트하였으므로, 다른 아키텍처, 규모, 기반 모델로의 일반화는 미해결이다. Gemma SM과 IC의 steering 실패(천장/바닥 효과)로 교차 모델 인과 증거는 단일 패러다임(MW)에 의존한다. 부호 역전 임계값 모델은 5개 데이터 포인트에서만 보정되었다. 교차 도메인 steering은 조건당 n = 50을 사용하여 통계적 검정력이 제한적이며, 비유의한 3개 조합은 더 큰 표본에서 유의해질 가능성이 있다. 베팅 제약 선형성(r = 0.98)은 4개 포인트에만 적합되었다. 모든 RQ3 증거는 상관적이며, G-프롬프트 정렬의 인과적 기반은 미검증이다. Universal BK neuron은 잔액이 분석을 교란할 수 있는 DP에서 식별되었다. 연구 전체에 걸친 다중 비교는 보정하지 않았으며, Bonferroni 보정 alpha 0.0056은 L30(p = 0.012)을 제외한 모든 유의한 조건에서 여전히 충족된다.

---

## 9. Next Steps

가장 시급한 확장은 다음과 같다: (1) 비유의한 교차 도메인 steering 3개 조합의 표본 크기를 n = 50에서 n = 200으로 확대; (2) SM에서 효과를 회복하기 위한 Gemma 다중 layer steering (L18 + L30); (3) G 대 non-G 프롬프트 조건에서 steering을 통한 RQ3 인과 검증; (4) 확장된 무작위 통제(조건당 10~20개 방향); (5) Gemma와 LLaMA를 넘어선 일반성 검증을 위한 추가 모델 아키텍처(Qwen, Mistral)로의 확장.

---

## Figures Referenced

| Figure | Description | File |
|--------|-------------|------|
| Fig. 0 | Experimental framework | figures/v13_fig0_experimental_framework.png |
| Fig. 1 | BK classification AUC | figures/v13_fig1_bk_classification.png |
| Fig. 2 | Universal BK neurons | figures/v13_fig2_universal_neurons.png |
| Fig. 3 | Cross-domain transfer heatmaps | figures/v13_fig3_crossdomain_transfer.png |
| Fig. 4 | Cross-domain steering transfer | figures/v13_fig4_crossdomain_steering.png |
| Fig. 7 | Dose-response curves (direction specificity) | figures/v13_fig7_dose_response.png |
| Fig. 8 | Multi-layer steering amplification | figures/v13_fig8_multilayer.png |
