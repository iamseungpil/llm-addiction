# Causal Feature Discovery in LLM Gambling Behavior: Full-Layer Analysis

## 실험 목적

본 연구는 대규모 언어 모델(LLM)이 도박 시뮬레이션에서 보이는 위험 추구 행동의 인과적 메커니즘을 규명하고자 한다. 구체적으로, Sparse Autoencoder(SAE)를 통해 학습된 해석 가능한 특징(feature) 중 도박 지속 또는 중단 결정에 인과적 영향을 미치는 특징을 식별하고, 이러한 특징이 모델의 행동에 어떻게 기여하는지를 정량적으로 분석한다.

기존 연구에서 LLaMA-3.1-8B와 Gemma-2-9B 모델을 대상으로 각각 3,200회의 슬롯머신 도박 실험을 수행한 결과, LLaMA는 4.7%, Gemma는 20.9%의 파산율을 보였다. 본 실험은 이러한 행동 차이의 신경 표상(neural representation) 수준 원인을 규명하기 위해, 전체 transformer layer(LLaMA 32개, Gemma 42개)에 대한 포괄적 분석을 수행한다.

## 실험 방법론

### Phase 1: Steering Vector 추출

첫 번째 단계에서는 Contrastive Activation Addition(CAA) 방법론(Turner et al., 2023)을 적용하여 파산(bankruptcy)과 자발적 중단(voluntary stop) 두 조건 간의 hidden state 차이를 추출한다. 각 모델의 전체 layer에서 마지막 토큰 위치의 hidden state를 수집하고, 두 조건 간 평균 차이 벡터를 steering vector로 정의한다.

수식으로 표현하면:
```
steering_vector[layer] = E[h_bankruptcy] - E[h_safe]
```

여기서 h는 해당 조건의 hidden state를 의미한다. 이 벡터는 "안전한 결정에서 위험한 결정으로" 이동하는 방향을 나타낸다.

**참고문헌**: Turner, A., et al. (2023). "Activation Addition: Steering Language Models Without Optimization." arXiv:2308.10248

### Phase 2: SAE Feature Projection

두 번째 단계에서는 steering vector를 SAE encoder를 통해 해석 가능한 feature 공간으로 투영한다. SAE는 transformer의 dense hidden state를 sparse하고 해석 가능한 feature로 분해하는 도구로(Cunningham et al., 2023; Bricken et al., 2023), 각 feature는 특정 개념이나 행동 경향성에 대응할 수 있다.

본 연구에서는 LlamaScope(fnlp, 32,768 features/layer)와 GemmaScope(google, 16,384 features/layer)를 사용하여 steering vector의 feature 분해를 수행한다. 각 layer에서 기여도(contribution magnitude) 상위 20개 feature를 후보로 선정하여, LLaMA에서 최대 640개, Gemma에서 최대 840개의 후보 feature를 추출한다.

**참고문헌**:
- Cunningham, H., et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv:2309.08600
- Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic.

### Phase 3: Soft Interpolation Patching (용량-반응 검증)

세 번째 단계는 후보 feature의 인과성을 검증하는 핵심 실험이다. Activation patching(Geiger et al., 2021; Wang et al., 2022)의 연속적 변형인 soft interpolation을 사용하여, feature 값을 5단계(α = 0.0, 0.25, 0.5, 0.75, 1.0)로 조작하고 행동 변화를 측정한다.

```
patched_feature = (1 - α) × safe_value + α × risky_value
```

이 방법의 핵심 이점은 용량-반응 관계(dose-response relationship)의 단조성(monotonicity)을 검증할 수 있다는 점이다. 진정한 인과적 feature라면, α가 증가함에 따라 중단 확률 P(stop)이 단조적으로 감소해야 한다. Spearman 상관계수 |ρ| ≥ 0.8, Cohen's d ≥ 0.3을 검증 기준으로 설정하고, Benjamini-Hochberg FDR 보정을 통해 다중 비교 문제를 해결한다.

**참고문헌**:
- Geiger, A., et al. (2021). "Causal Abstractions of Neural Networks." NeurIPS.
- Wang, K., et al. (2022). "Interpretability in the Wild: A Circuit for Indirect Object Identification." arXiv:2211.00593

### Phase 5: Gambling-Context Interpretation

마지막 단계에서는 검증된 feature의 의미를 해석한다. SelfIE(Chen et al., 2024)와 Patchscopes(Ghandeharioun et al., 2024) 방법론을 참고하여, SAE decoder vector를 모델의 residual stream에 주입하고 모델 자체가 feature의 의미를 자연어로 설명하도록 유도한다.

도박 맥락에 특화된 템플릿을 사용한다:
- "When gambling, if this pattern activates, the player tends to {X}"
- "This pattern represents the concept of '{X}' in risk-taking decisions"

이를 통해 "손실 추격(loss chasing)", "도박사의 오류(gambler's fallacy)", "위험 회피(risk aversion)" 등 도박 심리학 개념과의 연결을 시도한다.

**참고문헌**:
- Chen, Y., et al. (2024). "SelfIE: Self-Interpretation of Large Language Model Embeddings." arXiv:2403.10949
- Ghandeharioun, A., et al. (2024). "Patchscopes: A Unifying Framework for Inspecting Hidden Representations." arXiv:2401.06102

## 프롬프트 선정 및 근거

### 8개 조건 설계

본 연구는 2×2×2 요인 설계를 채택하여 8개 조건에서 분석을 수행한다:

| 조건 | 모델 | 상황 | 베팅 유형 |
|------|------|------|----------|
| 1 | LLaMA | Safe (자발적 중단) | Fixed ($10) |
| 2 | LLaMA | Safe (자발적 중단) | Variable ($5-$100) |
| 3 | LLaMA | Risky (파산 직전) | Fixed ($10) |
| 4 | LLaMA | Risky (파산 직전) | Variable ($5-$100) |
| 5 | Gemma | Safe (자발적 중단) | Fixed ($10) |
| 6 | Gemma | Safe (자발적 중단) | Variable ($5-$100) |
| 7 | Gemma | Risky (파산 직전) | Fixed ($10) |
| 8 | Gemma | Risky (파산 직전) | Variable ($5-$100) |

### 프롬프트 추출 방법

각 조건에서 실제 실험 데이터(3,200개)로부터 대표 프롬프트를 추출한다:

1. **Safe 조건**: 자발적으로 게임을 중단한 케이스에서, 마지막 라운드 후 잔고와 전체 게임 이력을 포함하는 프롬프트를 재구성한다. 이는 모델이 "그만두기로 결정한 순간"의 맥락을 반영한다.

2. **Risky 조건**: 파산한 케이스에서, 마지막(파산을 초래한) 베팅 직전의 상태를 재구성한다. 구체적으로, `history[-2]`의 잔고와 마지막 라운드를 제외한 이력(`history[:-1]`)을 사용한다. 이는 "위험한 베팅을 결정한 순간"의 맥락을 반영한다.

### 선정 근거

1. **생태학적 타당성(Ecological Validity)**: 합성된 시나리오가 아닌 실제 실험에서 발생한 상황을 사용함으로써, 모델이 실제로 경험한 의사결정 맥락에서 feature의 효과를 측정한다.

2. **베팅 유형 분리**: Fixed betting(보수적)과 Variable betting(위험 가능)을 분리하여, feature의 효과가 베팅 유형에 따라 달라지는지 검증한다. 기존 분석에서 Variable betting에서 파산율이 유의하게 높았다(LLaMA: 6.8% vs 2.6%, Gemma: 29.1% vs 12.8%).

3. **모델 간 비교**: LLaMA(보수적, 4.7% 파산)와 Gemma(위험 추구, 20.9% 파산) 간 동일 feature의 효과 차이를 분석하여, 모델 아키텍처나 학습 데이터에 따른 행동 차이의 메커니즘적 원인을 규명한다.

4. **양방향 인과성 검증**: Safe→Risky 방향과 Risky→Safe 방향 모두에서 feature 조작의 효과를 검증하여, 단순 상관이 아닌 양방향 인과관계를 확립한다.

## 예상 결과

본 실험을 통해 다음의 결과를 기대한다:

1. **인과적 Feature 식별**: 전체 layer에서 도박 행동에 인과적 영향을 미치는 SAE feature 목록
2. **용량-반응 곡선**: 각 feature의 활성화 강도와 중단 확률 간의 정량적 관계
3. **조건별 효과 차이**: 8개 조건에서 feature 효과의 이질성(heterogeneity) 분석
4. **해석 가능한 설명**: 각 feature가 나타내는 도박 관련 심리적 개념

## 기술적 세부사항

- **GPU**: GPU 4 (LLaMA), GPU 5 (Gemma)
- **전체 Layer**: LLaMA 32개, Gemma 42개
- **예상 소요 시간**: 약 8시간
- **출력 디렉토리**: `/data/llm_addiction/steering_vector_experiment_full/`

---

*최종 업데이트: 2025-12-21*
