# 주간 연구 보고서 (2025-10-20)

## 전체 흐름

LLM의 gambling-like behavior 근본 원인을 SAE(Sparse Autoencoder) 기반 mechanistic interpretability로 규명하는 연구를 진행 중이다. 현재까지 1) GPT-4o-mini의 fixed vs variable betting 비교 실험(12,800개), 2) LLaMA vs Gemma 모델 비교(6,400개), 3) L1-31 전체 layer에서 87,012개 significant features 추출, 4) Layer-wise feature pathway 분석, 5) Token-level attention tracking을 완료하였다. 현재 single-round activation patching 실험이 진행 중이며, feature-word attribution 분석이 계획되어 있다.

---

## 완료된 작업

### 1. GPT-4o-mini Fixed vs Variable Betting 실험 (완료)

총 12,800개 실험을 완료하였다(Fixed 6,400개, Variable 6,400개). 각 조건은 4개 betting amounts($10, $30, $50, $70) × 32개 prompt combinations × 50 repetitions로 구성되었다. Loss chasing index(i_lc) 정의를 기존 절대값 기반(손실 후 베팅액 증가량)에서 비율 기반(bet/balance ratio 증가)으로 수정하여 잔고 크기를 정규화하였다.

Variable betting은 Fixed betting 대비 파산율이 최대 18배 증가하였다($70 조건: 18.25% vs 0.38%). 또한 게임 지속성이 현저히 증가하여 Variable에서 평균 15-20 rounds를 플레이한 반면 Fixed에서는 0.3-1.8 rounds에 불과하였다. 흥미롭게도 Variable betting에서 GPT는 max bet의 27.5-62.5%만 실제 사용하였으며, 베팅 상한이 높을수록 더 보수적으로 베팅하는 패턴을 보였다. Composite irrationality index는 Variable에서 평균 2배 이상 높았으며, 프롬프트 복잡도가 증가할수록 Variable의 파산율이 0%에서 30%까지 급증하였다. 이 결과는 choice overload 효과와 loss chasing 행동을 실험적으로 검증하였으며, 6개 figures로 시각화 완료하였다.

### 2. LLaMA-3.1-8B vs Gemma-2-9b-it 비교 실험 (완료)

두 모델의 gambling behavior 차이를 검증하기 위해 각 모델당 3,200개 실험(64 conditions × 50 repetitions)을 완료하였다. 모든 실험은 영어 프롬프트를 사용하였으며 GPT-4o-mini 실험과 동일한 게임 설정(30% win rate, 3× payout)을 적용하였다.

두 모델은 뚜렷한 행동 차이를 보였다. LLaMA는 극도로 보수적인 패턴을 보여 대부분의 게임에서 즉시 중단하였으며, Gemma는 상대적으로 공격적인 betting 패턴을 나타냈다. 이는 모델 아키텍처와 학습 방법론이 risk-taking behavior에 영향을 미침을 시사한다. 분석 결과는 CSV 형식으로 저장되었으며, 모델 간 행동 패턴 차이가 mechanistic level에서 어떻게 발현되는지 추가 분석이 필요하다.

### 3. L1-31 전체 Layer Feature Extraction (완료)

6,400개 LLaMA 실험에서 layers 1-31 전체를 대상으로 SAE feature extraction을 완료하였다. 각 layer당 32,768개 features를 분석하여 총 87,012개 significant features를 선별하였다(p < 0.01, Cohen's d > 0.5). Layer-wise 분포를 보면 early layers(L1-L10)에서 약 2,000-3,000개, middle layers(L11-L20)에서 3,000-4,000개, late layers(L21-L31)에서 2,000-3,000개 features가 유의미하게 나타났다. 이 데이터는 후속 실험(activation patching, feature-word analysis)의 기반이 되며, 전체 network에서 gambling behavior와 관련된 computational pathway를 추적할 수 있는 토대를 마련하였다.

### 4. Layer-wise Feature Pathway Analysis (완료)

Layer 8(early processing)과 Layer 31(final decision) 간 feature correlation을 분석하여 25개 significant pathways를 발견하였다(Pearson r > 0.6, p < 0.01).

발견된 핵심 pathway는 세 가지 유형으로 분류된다. 첫째, risky pathway(L8-26623 → L31-3327, r = 0.84)는 early layer의 risk-taking features가 final decision layer로 직접 전파되는 경로를 보여준다. 둘째, safe inhibition pathway(L8-12478 → L31-3327, r = -0.83)는 safe features가 risky features를 억제하는 inhibition mechanism을 나타낸다. 셋째, three-layer pathway(L8-2059 → L10-5950 → L31-10692, r = 0.59-0.68)는 중간층을 거쳐 정보가 처리되는 computational path를 보여준다. 이는 early layer의 risk assessment features가 final layer의 decision features로 전파되며, safe features가 risky features를 억제하는 inhibition mechanism이 작동함을 의미한다. Multi-layer decision signature analysis에서 L31과 L8이 가장 높은 discriminative power를 보였다(Cohen's d > 2.5).

### 5. Multiround Persistent Patching 실험 (완료, Negative Result)

441개 causal features에 대해 전체 게임 진행 동안 feature 값을 고정하는 persistent patching 방식으로 39,690개 games를 수행하였다(3 conditions × 30 trials × 441 features). 실험 결과 safe mean patching이 예상과 반대로 파산율을 증가시키는 negative result를 얻었다. 이는 persistent patching이 model의 자연스러운 decision dynamics를 방해하여 역효과를 발생시킴을 시사한다. 이 결과는 single-round patching의 필요성을 확인하였으며, 논문에 negative result로 보고할 예정이다.

### 6. Token-Level Feature Tracking (완료)

10개 diverse scenarios(파산 직전, 안전 중단, 초기 상태 등)에 대해 token-level feature activations와 attention patterns를 추출하였다. L1-31 전체 tracking은 모든 31 layers에 대해 NPZ 압축 형식으로 저장되었으며(2-3GB), 메모리 최적화를 통해 layer-wise incremental saving을 구현하여 165MB peak memory로 실행 가능하다. 3-layer tracking은 layers 8, 15, 31에 대한 상세 분석을 완료하였으며 balance, goal, probability, choice tokens의 위치별 feature activations를 확인하였다. 이 데이터는 어떤 input tokens가 최종 decision에 가장 큰 영향을 미치는지, 그리고 attention이 어떻게 흐르는지를 분석하는 기반이 된다.

### 7. Feature-Word Co-occurrence Analysis (완료)

441개 causal features에 대해 6,400개 responses를 분석하여 각 feature의 high/low activation 시 나타나는 단어 패턴을 추출하였다. TF-IDF weighting을 적용하여 feature-specific vocabulary를 확인하였다.

이 분석은 features가 어떤 linguistic patterns과 연관되는지를 규명하며, token-level tracking과 결합하여 "balance" feature가 실제로 balance tokens에 attend하는지 검증할 수 있다.

---

## 진행 중인 작업

### 1. Single-Round Activation Patching (진행 중)

Multiround persistent patching의 문제점을 해결하기 위해 decision 시점에만 patching을 수행하는 single-round 방식으로 전환하였다. Layers 1-31 전체에서 각 layer당 상위 300개 features(총 9,300 features)를 대상으로 실험 진행 중이다. 실험 설계는 safe/risky prompts × safe/risky feature values의 4-condition testing이며 각 조건당 50 trials을 수행한다. Cohen's d와 t-test를 통해 causal effects를 검증한다. GPU 1개당 약 30-40시간 소요 예상이며, 현재 일부 layers에서 진행 중이고 중간 결과는 유의미한 causal features를 확인하고 있다.

---

## 계획된 작업

### 1. Feature-Word Analysis 확장 (L1-31 전체)

441개 causal features에 대한 word analysis가 완료되었으나, 전체 87,012개 significant features로 확장하는 것이 계획되어 있다. 이는 4 GPUs에서 병렬로 약 30-40시간 소요 예상이다. 현재 441개 features의 word patterns가 충분한지, 아니면 전체 network의 linguistic representation을 이해하기 위해 확장이 필요한지 필요성 평가를 검토 중이다.

### 2. Cross-Reference Analysis (Token + Word + Attention)

완료된 token-level tracking, feature-word analysis, attention patterns을 통합하여 feature-token correspondence를 검증한다. 구체적으로 "balance" features가 실제로 balance tokens에 attend하는지, feature의 word associations와 attention targets이 일치하는지, 그리고 layer별로 이러한 correspondence가 어떻게 변화하는지를 분석한다.

---

## 차주 작업 계획

우선순위 최상으로 single-round activation patching 실험을 완료한다. L1-31 전체 9,300 features에 대한 실험을 마무리하고, causal features를 선별하여 효과 크기를 분석하며, Experiment 5의 negative result와 비교 분석을 수행한다. Cross-reference 분석에서는 token tracking, feature-word, attention 데이터를 통합하여 feature-token correspondence를 검증하고 key tokens을 파악하여 pathway와 연결한다. Experiment 0에 대해서는 LLaMA vs Gemma의 behavioral differences를 정량화하고 모델 비교 section의 논문 초안을 작성한다. L1-31 word analysis의 필요성을 평가하여 현재 441개 결과로 충분한지 판단하고 필요 시 실행 계획을 수립한다. 마지막으로 완료된 실험들의 results section을 작성하고 figures를 생성하며 통계를 검증하고, Experiment 5의 negative result를 논문에 포함할 수 있도록 보고서를 작성한다.

---

## 주요 이슈

Multiround persistent patching 실험에서 persistent manipulation이 model dynamics를 방해하여 의도와 반대되는 결과를 초래하였다. 이에 따라 single-round patching으로 방법론을 전환하였다. Computational resource 측면에서는 single-round patching이 9,300 features × 200 trials = 1.86M games로 계산량이 크기 때문에 GPU 분산 실행이 필요하다. L1-31 word analysis 확장과 관련하여 87,012 features 전체 분석이 computational cost 대비 추가 insight를 제공하는지 평가가 필요하다. 모델 비교 실험에서는 LLaMA의 극도로 보수적인 패턴이 model architecture에서 기인하는지 training data에서 비롯되는지 명확한 원인 규명이 필요하다.

---

## 데이터 현황

실험 데이터는 GPT Fixed/Variable 12,800 games, LLaMA vs Gemma 6,400 games, LLaMA Exp1 original 6,400 games가 완료되었으며, multiround patching 39,690 games는 negative result로 완료되었다. Single-round patching은 목표 1.86M games를 향해 진행 중이다. Feature 데이터는 L1-31에서 87,012개 significant features가 추출되었으며, Exp2의 causal features는 현재 선별 중이고, feature-word associations는 441 features에 대해 완료되었다. Token-level 데이터는 3-layer tracking 2.41GB와 L1-31 tracking 2-3GB NPZ 형식으로 저장되었으며, 10 scenarios × 31 layers × 32,768 features 규모를 포함한다. 총 실험 규모는 약 60,000+ games, 87,012 features, 2.5M+ data points에 달한다.

