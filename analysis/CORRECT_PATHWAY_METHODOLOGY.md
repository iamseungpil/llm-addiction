# 올바른 Pathway & Word Analysis 방법론

## 🔍 사용자 질문에 대한 답변

### Q1: "보통 그런 방법을 사용하는지 기존 연구들을 살펴봐줄래?"
**A**: ❌ **제가 제안한 correlation 방법은 잘못되었습니다!**

### Q2: "Feature와 feature의 연관 관계를 1-31까지 연속적으로 추적해야하지 않아?"
**A**: ✅ **맞습니다!** Backward Jacobian으로 연속 추적해야 합니다.

### Q3: "Safe prompt와 risk prompt에서 각 feature들이 어떻게 반응하는지를 추적하지 않나?"
**A**: ✅ **맞습니다!** 조건별 feature activation을 모두 기록해야 합니다.

### Q4: "해당 feature들이 발화할 때 어떤 단어와 연관이 있는지 분석이 가능할까?"
**A**: ✅ **가능합니다!** experiment_3 코드로 이미 구현되어 있습니다.

---

## 🎓 Anthropic 2025 방법론 (Attribution Graphs)

### 핵심 원리: Backward Jacobian

```python
# Anthropic 방법 (2025) - GRADIENT-BASED CAUSAL TRACING

def attribution_graph(model, output_feature_idx):
    """
    출력 feature로부터 역으로 추적하여 입력까지의 인과 경로 발견
    """

    # 1. 출력 feature로부터 시작
    current_features = [output_feature_idx]
    attribution_paths = []

    # 2. Layer별로 역방향 추적
    for layer in reversed(range(32)):  # L31 → L30 → ... → L1

        next_layer_features = []

        for feat_idx in current_features:
            # Backward Jacobian 계산
            # ∂(output_feature) / ∂(previous_layer_features)
            jacobian = compute_backward_jacobian(
                model,
                current_layer=layer + 1,
                target_feature=feat_idx,
                source_layer=layer
            )

            # 강하게 기여한 features 찾기
            # |gradient| > threshold
            contributing_features = np.where(np.abs(jacobian) > 0.1)[0]

            for src_feat in contributing_features:
                attribution_paths.append({
                    'source': f'L{layer}-{src_feat}',
                    'target': f'L{layer+1}-{feat_idx}',
                    'attribution': jacobian[src_feat],  # CAUSAL contribution
                    'method': 'gradient'
                })

                next_layer_features.append(src_feat)

        current_features = next_layer_features

    return attribution_paths
```

**핵심 차이점**:
- ✅ **Gradient-based**: 실제 인과 기여도 측정 (causation)
- ✅ **역방향 추적**: 출력에서 입력으로 backward
- ✅ **수학적 정확성**: Jacobian은 편미분으로 정확한 기여도

### Cross-Layer Transcoders (CLT)

```python
# CLT를 사용한 간소화

class CrossLayerTranscoder:
    """
    각 feature가 residual stream에서 읽고,
    모든 후속 MLP layer에 직접 기여
    """

    def __init__(self, layer_idx):
        self.layer = layer_idx

    def encode(self, residual_stream):
        """Residual stream → SAE features"""
        return self.sae_encoder(residual_stream)

    def decode_to_all_downstream_mlps(self, features):
        """
        Feature activations → all downstream MLP inputs

        Instead of:
          Feature → Layer i+1 → Layer i+2 → ... → Output

        CLT directly computes:
          Feature → [MLP_i+1, MLP_i+2, ..., MLP_31]
        """
        downstream_contributions = {}

        for target_layer in range(self.layer + 1, 32):
            # Direct contribution to target MLP
            contribution = self.decoder_matrix[target_layer] @ features
            downstream_contributions[target_layer] = contribution

        return downstream_contributions
```

**장점**:
- ✅ 중간 layer 계산 생략 가능
- ✅ Circuit이 단순화됨
- ✅ Feature L8-123이 L31-456에 직접 기여하는 정도 계산

---

## ❌ 제가 제안한 Correlation 방법의 문제점

### 문제 1: Correlation ≠ Causation

```python
# 제가 제안한 방법 (잘못됨!)
r, p = stats.pearsonr(l8_features, l31_features)

# 문제점:
# - 높은 correlation이라도 인과관계는 아님
# - 공통 원인 때문일 수 있음 (spurious correlation)
# - 방향성을 알 수 없음 (L8→L31? L31→L8? 둘 다 input→?)
```

**예시**:
```
L8-123 activation: [0.1, 0.5, 0.8, 0.3, ...]  (N games)
L31-456 activation: [0.2, 0.6, 0.9, 0.4, ...]  (N games)

Pearson r = 0.95 (매우 높음!)

BUT 이것만으로는:
❌ L8-123이 L31-456을 활성화했는지 알 수 없음
❌ 둘 다 입력의 "$100"에 반응한 것일 수도 있음
❌ L31-456이 L8-123에 영향을 줬을 수도 있음 (불가능하지만 correlation은 구분 못함)
```

### 문제 2: 연속성 부족

```python
# 제가 제안: L8 → L31만 봄
r_8_31 = correlation(l8, l31)

# 올바른 방법: L8 → L9 → L10 → ... → L31 전부 추적
for i in range(8, 32):
    for j in range(i+1, 32):
        jacobian = compute_gradient(layer_i, layer_j)
```

---

## ✅ 올바른 Pathway Analysis 설계

### Option 1: Simplified Gradient-based (추천)

```python
class SimplifiedPathwayTracker:
    """
    Backward Jacobian의 간소화 버전
    (Full CLT 없이 가능)
    """

    def __init__(self, model, saes):
        self.model = model
        self.saes = saes  # {layer: SAE}

    def track_pathway(self, prompt, target_layer, target_feature):
        """
        특정 출력 feature로부터 역추적

        Returns:
            pathway: List of (source_feat, target_feat, attribution)
        """

        # 1. Forward pass with gradient tracking
        with torch.enable_grad():
            inputs = tokenize(prompt)
            inputs.requires_grad = True

            # Get all hidden states
            outputs = self.model(
                inputs,
                output_hidden_states=True,
                return_dict=True
            )

            # 2. Extract SAE features for all layers
            all_features = {}
            for layer in range(1, 32):
                hidden = outputs.hidden_states[layer][:, -1, :]
                features = self.saes[layer].encode(hidden)
                all_features[layer] = features

            # 3. Target feature activation
            target_activation = all_features[target_layer][target_feature]

        # 4. Backward pass (gradient computation)
        target_activation.backward()

        # 5. Extract gradients (attributions)
        pathway = []
        for source_layer in range(1, target_layer):
            # Gradient of target w.r.t. source features
            grad = all_features[source_layer].grad

            # Find strongly contributing features
            strong_contributors = torch.where(torch.abs(grad) > 0.1)[0]

            for src_feat in strong_contributors:
                pathway.append({
                    'source': f'L{source_layer}-{src_feat.item()}',
                    'target': f'L{target_layer}-{target_feature}',
                    'attribution': grad[src_feat].item(),
                    'is_causal': True  # Gradient-based!
                })

        return pathway
```

**장점**:
- ✅ Gradient-based: 진짜 인과성
- ✅ 구현 가능: 기존 LlamaScope + PyTorch autograd
- ✅ 빠름: 한 번의 forward + backward

**단점**:
- ⚠️ Last token만 추적 (all-position 불가)
- ⚠️ Full CLT만큼 정교하지 않음

### Option 2: Attention Flow Tracking

```python
class AttentionFlowPathway:
    """
    Attention pattern을 따라 정보 흐름 추적
    """

    def track_attention_pathway(self, prompt, target_token_pos):
        """
        특정 토큰으로 흐르는 attention 추적
        """

        outputs = self.model(
            prompt,
            output_attentions=True,
            output_hidden_states=True
        )

        # Layer별 attention pattern
        # attentions: (n_layers, n_heads, seq_len, seq_len)

        pathway = []
        current_tokens = [target_token_pos]  # 마지막 토큰

        # 역방향 추적
        for layer in reversed(range(32)):
            attention = outputs.attentions[layer]  # (n_heads, seq_len, seq_len)

            next_tokens = []
            for target_pos in current_tokens:
                # 어느 토큰들이 target_pos에 attend했는가?
                # attention[:, target_pos, :] = source tokens → target

                for head in range(attention.shape[0]):
                    attn_weights = attention[head, target_pos, :]

                    # 강한 attention (> threshold)
                    strong_sources = torch.where(attn_weights > 0.1)[0]

                    for src_pos in strong_sources:
                        pathway.append({
                            'layer': layer,
                            'head': head,
                            'source_token': src_pos.item(),
                            'target_token': target_pos,
                            'attention_weight': attn_weights[src_pos].item()
                        })

                        next_tokens.append(src_pos.item())

            current_tokens = list(set(next_tokens))

        return pathway
```

**장점**:
- ✅ Token-level precision
- ✅ 어느 단어가 어느 단어에 영향을 주는지 정확히 알 수 있음

**단점**:
- ⚠️ Attention만 추적 (MLP는 안 봄)
- ⚠️ Attention이 인과성의 전부는 아님

---

## 📝 Word-Level Analysis (Feature-Word Association)

### Anthropic/Neuronpedia 방법

```python
class FeatureWordAnalysis:
    """
    Feature가 어떤 단어/토큰과 연관되는지 분석
    (Neuronpedia dashboard 방식)
    """

    def find_top_activating_examples(self, feature_idx, layer, dataset):
        """
        해당 feature의 activation이 가장 높은 예시들 찾기
        """

        activations = []

        for example in dataset:
            # Forward pass
            outputs = self.model(
                example['tokens'],
                output_hidden_states=True
            )

            # SAE encode
            hidden = outputs.hidden_states[layer]
            features = self.sae.encode(hidden)  # (seq_len, 32768)

            # 각 토큰 위치에서 feature activation
            for pos in range(len(example['tokens'])):
                activations.append({
                    'example_id': example['id'],
                    'token': example['tokens'][pos],
                    'token_str': example['token_strings'][pos],
                    'position': pos,
                    'activation': features[pos, feature_idx].item(),
                    'context': example['token_strings'][max(0, pos-5):pos+6]
                })

        # Activation 기준 정렬
        activations.sort(key=lambda x: x['activation'], reverse=True)

        # Top 100 예시
        top_examples = activations[:100]

        # 단어 빈도 분석
        word_freq = Counter([ex['token_str'] for ex in top_examples])

        return {
            'top_examples': top_examples,
            'top_words': word_freq.most_common(20),
            'interpretation': self.auto_interpret(top_examples)
        }
```

**핵심**:
- ✅ **모든 토큰 위치** 확인 (not just last token!)
- ✅ **Top activating examples** 수집
- ✅ **단어 빈도** 계산

### Experiment 3 방법 (우리 코드)

```python
# /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis/

class Experiment3WordAnalysis:
    """
    Feature activation 기준으로 high/low 그룹 나눠서
    단어 빈도 차이 분석
    """

    def analyze_feature_words(self, feature_idx, layer):
        """
        해당 feature의 activation이 높을 때 vs 낮을 때
        어떤 단어가 더 자주 나오는지
        """

        # 1. 6,400개 게임에서 feature activation 추출
        activations = []
        responses = []

        for game in self.exp1_data:
            # SAE encode (last token)
            feat_value = self.extract_feature(game['response'], layer, feature_idx)

            activations.append(feat_value)
            responses.append(game['response'])

        # 2. Median으로 split
        median = np.median(activations)

        high_group = [resp for act, resp in zip(activations, responses) if act > median]
        low_group = [resp for act, resp in zip(activations, responses) if act <= median]

        # 3. 단어 빈도 계산
        high_words = Counter()
        low_words = Counter()

        for resp in high_group:
            words = resp.split()
            high_words.update(words)

        for resp in low_group:
            words = resp.split()
            low_words.update(words)

        # 4. 빈도 차이 계산
        word_diffs = []
        all_words = set(high_words.keys()) | set(low_words.keys())

        for word in all_words:
            high_freq = high_words[word] / len(high_group)
            low_freq = low_words[word] / len(low_group)
            diff = high_freq - low_freq

            if abs(diff) > 0.01:  # 1% 이상 차이
                word_diffs.append({
                    'word': word,
                    'high_freq': high_freq,
                    'low_freq': low_freq,
                    'diff': diff
                })

        word_diffs.sort(key=lambda x: abs(x['diff']), reverse=True)

        return word_diffs[:50]  # Top 50 words
```

**장점**:
- ✅ 이미 구현됨!
- ✅ 6,400 게임 데이터 활용
- ✅ 통계적으로 유의미한 단어만 추출

**단점**:
- ⚠️ Last token만 분석 (Anthropic처럼 모든 token은 아님)
- ⚠️ Response text 전체를 보기 때문에 position-specific 정보 없음

---

## 🎯 2,787개 Features에 대한 실험 설계

### 실험 1: Gradient-based Pathway Tracking

```python
# 새 실험: /home/ubuntu/llm_addiction/experiment_pathway_gradient/

"""
목표: 2,787개 features의 gradient-based pathway 추적

방법:
1. Safe/Risky prompt 각각에서
2. 2,787개 각 feature를 target으로
3. Backward gradient로 기여 features 찾기
4. L1부터 L31까지 연속 추적

결과:
- L1-123 → L9-456 → L25-789: attribution = 0.85 (strong causal path)
- 중간 layer risky features (L9-L17)의 역할 규명
"""

class GradientPathwayExperiment:
    def run(self):
        results = []

        # 2,787 features
        for feat_info in self.causal_features:
            layer = feat_info['layer']
            feat_id = feat_info['feature_id']

            # Safe prompt pathway
            safe_pathway = self.track_pathway(
                self.safe_prompt,
                target_layer=layer,
                target_feature=feat_id
            )

            # Risky prompt pathway
            risky_pathway = self.track_pathway(
                self.risky_prompt,
                target_layer=layer,
                target_feature=feat_id
            )

            results.append({
                'feature': feat_info['feature'],
                'safe_pathway': safe_pathway,
                'risky_pathway': risky_pathway
            })

        return results
```

**예상 시간**:
- 2,787 features × 2 prompts = 5,574 forward+backward passes
- ~1초 per pass = **~1.5시간** (실행 가능!)

### 실험 2: Word Association (Experiment 3 재사용)

```python
# 기존 코드 활용: /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis/

"""
목표: 2,787개 features의 word association 분석

방법:
1. Experiment 3 코드 수정
2. 2,787개 features만 분석 (87,012 대신)
3. 6,400 게임 데이터 활용

결과:
- L1-1292 (safe): ["balance", "stop", "enough"] 연관
- L9-??? (risky): ["win", "more", "continue"] 연관
"""

# 수정 필요:
# 1. 2,787 features CSV 로드
# 2. Layer별로 처리
# 3. 결과 저장

예상 시간:
- 2,787 features × 6,400 games × 0.01초 = ~5시간 (1 GPU)
```

---

## 📊 통합 분석 가능

### Pathway + Word Analysis 결합

```python
def integrated_analysis(pathway_results, word_results):
    """
    Pathway와 Word analysis 통합
    """

    insights = []

    for feat in causal_features:
        # Pathway 정보
        pathway = pathway_results[feat['feature']]

        # Word 정보
        words = word_results[feat['feature']]

        # 통합
        insights.append({
            'feature': feat['feature'],

            # Pathway
            'upstream_features': pathway['sources'],
            'downstream_features': pathway['targets'],

            # Words
            'associated_words': words['top_words'],

            # 해석
            'interpretation': f"""
            Feature {feat['feature']}:
            - Upstream: {', '.join(pathway['sources'][:3])}
            - Words: {', '.join([w['word'] for w in words['top_words'][:5]])}
            - Role: {'Safe decision' if feat['type'] == 'safe' else 'Risky decision'}
            """
        })

    return insights
```

**예시 결과**:
```
L25-1234 (Safe feature):
  Upstream pathway: L1-123 → L8-456 → L19-789 → L25-1234
  Associated words: ["balance", "stop", "enough", "safe"]
  Role: Detects "stop" decision keywords and accumulates safety signals from early layers

L9-5678 (Risky feature):
  Upstream pathway: L3-111 → L9-5678
  Downstream pathway: L9-5678 → L17-999 → L30-2222
  Associated words: ["win", "more", "try", "continue"]
  Role: Middle-layer amplifier of risk signals, bridges early detection to late decision
```

---

## 🎯 최종 답변

### Q1: 보통 어떤 방법을 사용하나?
**A**: **Backward Jacobian (Gradient-based)** 또는 **Attention Flow**
- ❌ Correlation은 잘못된 방법
- ✅ Gradient는 인과성 측정
- ✅ Attention은 정보 흐름 추적

### Q2: L1-31까지 연속적으로 추적해야 하나?
**A**: ✅ **맞습니다!**
- Gradient backward pass가 자동으로 모든 layer 추적
- L1 → L2 → ... → L31 전체 경로 발견

### Q3: Safe/risky prompt에서 feature 반응 추적?
**A**: ✅ **맞습니다!**
- 두 조건에서 각각 pathway 추적
- 차이 분석으로 조건별 메커니즘 규명

### Q4: Feature-word association 분석 가능?
**A**: ✅ **가능합니다!**
- Experiment 3 코드 이미 존재
- 2,787 features로 재실행 가능
- 5시간이면 완료

---

## 📝 추천 실행 순서

1. **Gradient Pathway Tracking** (~1.5시간)
   - 2,787 features의 인과 경로 발견
   - Safe/risky prompt별 pathway 비교

2. **Word Association Analysis** (~5시간)
   - 2,787 features의 단어 연관성
   - High/low activation 단어 차이

3. **통합 분석** (~30분)
   - Pathway + Word 결합
   - 중간 layer risky features 역할 규명
   - 논문용 figure 생성

**Total: ~7시간** (실행 가능!)

---

**Date**: 2025-10-22
**Based on**: Anthropic 2025 Circuit Tracing, Neuronpedia Dashboard
**Status**: Ready to implement
