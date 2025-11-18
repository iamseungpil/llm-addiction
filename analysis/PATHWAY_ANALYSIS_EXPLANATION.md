# Pathway Analysis 원리 및 실험 설계

## 📚 Pathway Analysis란?

**Pathway Analysis**는 LLM의 여러 layer에 걸쳐 정보가 어떻게 전달되고 변환되는지를 추적하는 분석 방법입니다.

---

## 🔬 기본 원리

### 1. Feature Correlation을 통한 간접 추론

**핵심 아이디어**:
- 만약 L8의 feature A가 L31의 feature B와 **강한 상관관계**를 보인다면
- L8의 feature A 정보가 L31의 feature B로 **전달되었을 가능성**이 있음

**방법**:
```python
# 동일한 게임들에 대해
l8_features = [game1_l8, game2_l8, ..., gameN_l8]  # (N games,)
l31_features = [game1_l31, game2_l31, ..., gameN_l31]  # (N games,)

# Pearson correlation
r, p_value = stats.pearsonr(l8_features, l31_features)

if r > 0.6 and p_value < 0.01:
    print("Strong pathway: L8 → L31")
```

**한계**:
- ❌ **Correlation ≠ Causation**: 인과관계는 검증 불가
- ❌ **간접 경로 구분 불가**: L8 → L31이 직접인지, L8 → L10 → L31인지 모름
- ❌ **공통 원인 가능성**: 둘 다 입력에 반응하는 것일 수 있음

---

### 2. Multi-hop Pathway Tracing

**3-layer pathway 발견 방법**:

```python
# 가설: L8 → L10 → L31 경로가 존재
for l8_feat in important_l8_features:
    for l10_feat in important_l10_features:
        for l31_feat in important_l31_features:

            r_8_10 = correlation(l8_feat, l10_feat)
            r_10_31 = correlation(l10_feat, l31_feat)
            r_8_31 = correlation(l8_feat, l31_feat)

            # 경로 존재 조건
            if (r_8_10 > 0.6 and r_10_31 > 0.6 and r_8_31 > 0.5):
                print(f"Pathway: {l8_feat} → {l10_feat} → {l31_feat}")
                print(f"  L8→L10: r={r_8_10:.2f}")
                print(f"  L10→L31: r={r_10_31:.2f}")
                print(f"  L8→L31: r={r_8_31:.2f} (direct)")
```

**해석**:
- `r_8_10`과 `r_10_31`이 모두 높음 → L10이 **중간 매개체** 역할
- `r_8_31`도 높음 → **정보 보존** (L10을 거쳐도 정보 손실 적음)

---

### 3. Anthropic 2025 방법론과의 차이

#### Anthropic Attribution Graphs (Causal)
```python
# Anthropic 방법 (2025)
# Token-level causal intervention

def find_token_attribution(model, prompt, output_token):
    """각 input token이 output에 얼마나 기여하는지 측정"""

    attributions = []
    for token_pos in range(len(prompt)):
        # 특정 token의 feature를 ablate
        ablated_output = model.forward_with_ablation(
            prompt,
            ablate_position=token_pos
        )

        # Output 변화 측정
        delta = original_output - ablated_output
        attributions.append(delta)

    return attributions  # Token-level causal attribution
```

**Anthropic이 할 수 있는 것**:
- ✅ Token-level precision: "$100" 토큰이 "bet" 결정에 얼마나 기여
- ✅ Causal validation: Ablation으로 인과성 검증
- ✅ Attention flow: 어느 토큰이 어느 토큰에 attend
- ✅ Cross-Layer Transcoders: MLP를 interpretable features로 분해

#### 우리의 Correlation 방법 (현재)
```python
# 우리 방법 (last token features만)
# Statistical correlation

def find_layer_correlation(l8_activations, l31_activations):
    """Layer 간 feature correlation"""

    correlations = []
    for l8_feat in l8_activations:
        for l31_feat in l31_activations:
            r, p = stats.pearsonr(l8_feat, l31_feat)

            if r > 0.6 and p < 0.01:
                correlations.append({
                    'l8': l8_feat.name,
                    'l31': l31_feat.name,
                    'r': r,
                    'p': p
                })

    return correlations  # Layer-level correlation
```

**우리가 할 수 있는 것**:
- ✅ Layer-level correlation: L8 feature가 L31 feature와 연결
- ❌ Token-level precision: 어느 토큰 때문인지 모름
- ❌ Causal validation: Correlation만 있고 causation 없음
- ❌ Attention flow: Attention pattern 없음

---

## 🎯 현재 상황 분석

### Experiment 2 (Patching) 데이터

**저장된 정보**:
```json
{
  "feature": "L11-1829",
  "condition": "safe_baseline",
  "trial": 0,
  "response": "1\nCurrent balance: $150\n...",
  "parsed": {
    "action": "bet",
    "bet": 10,
    "valid": true
  }
}
```

**없는 정보**:
- ❌ Prompt (재구성 가능하긴 함)
- ❌ Hidden states (layer별 activation)
- ❌ SAE features (모든 32,768개 features)
- ❌ Attention patterns

**결론**:
- ❌ **Pathway tracking 불가능** (hidden states 없음)
- ✅ **BUT**: 새로운 실험으로 데이터 수집 가능

---

## 💡 새로운 Pathway Tracking 실험 설계

### 목표
**2,787개 causal features의 layer 간 정보 흐름 추적**

### 실험 설계

#### Phase 1: Feature Extraction (데이터 수집)

```python
class PathwayTrackingExperiment:
    """
    목표: 2,787개 features의 모든 layer activations 추출
    """

    def __init__(self):
        self.target_features = load_2787_features()
        # Safe: 640, Risky: 2147

        # 6 conditions (Experiment 2와 동일)
        self.conditions = [
            'safe_baseline',
            'safe_with_safe_patch',
            'safe_with_risky_patch',
            'risky_baseline',
            'risky_with_safe_patch',
            'risky_with_risky_patch',
        ]

        # 작은 샘플로 시작 (계산량 고려)
        self.n_trials_per_condition = 30

    def extract_all_layer_features(self, prompt, target_features):
        """
        모든 layer의 feature activations 추출

        Returns:
            {
                'L1': {feat_id: activation},
                'L2': {feat_id: activation},
                ...
                'L31': {feat_id: activation}
            }
        """

        # 1. LLaMA forward pass
        with torch.no_grad():
            outputs = self.model(
                prompt,
                output_hidden_states=True
            )

        # 2. 각 layer의 hidden states → SAE features
        all_layer_features = {}

        for layer in range(1, 32):
            hidden = outputs.hidden_states[layer][:, -1, :]  # Last token

            # SAE encode
            sae = self.load_sae(layer)
            features = sae.encode(hidden)  # (32768,)

            # 해당 layer의 target features만 저장
            layer_target_feats = [
                f for f in target_features
                if f['layer'] == layer
            ]

            all_layer_features[f'L{layer}'] = {
                feat['feature_id']: features[feat['feature_id']].item()
                for feat in layer_target_feats
            }

        return all_layer_features

    def run_experiment(self):
        """
        실험 실행
        """
        results = []

        for condition in self.conditions:
            for trial in range(self.n_trials_per_condition):
                # Prompt 생성
                if 'safe' in condition:
                    prompt = self.safe_prompt
                else:
                    prompt = self.risky_prompt

                # Patching 적용 (필요시)
                if 'patch' in condition:
                    prompt = self.apply_patching(prompt, condition)

                # 모든 layer features 추출
                all_features = self.extract_all_layer_features(
                    prompt,
                    self.target_features
                )

                # Response 생성
                response = self.generate_response(prompt)

                # 저장
                results.append({
                    'condition': condition,
                    'trial': trial,
                    'all_layer_features': all_features,
                    'response': response,
                    'parsed': self.parse_response(response)
                })

        return results
```

**저장 형식**:
```json
{
  "condition": "safe_baseline",
  "trial": 0,
  "all_layer_features": {
    "L1": {
      "1292": 0.0234,
      "1301": 0.0156,
      ...
    },
    "L2": {
      "2035": 0.0412,
      ...
    },
    ...
    "L31": {
      "10692": 0.7612,
      ...
    }
  },
  "response": "1\nCurrent balance: $150\n...",
  "parsed": {
    "action": "bet",
    "bet": 10
  }
}
```

#### Phase 2: Pathway Analysis (분석)

```python
def analyze_pathways(results):
    """
    2,787개 features의 layer 간 correlation 분석
    """

    # 1. Condition별로 grouping
    safe_baseline = [r for r in results if r['condition'] == 'safe_baseline']

    # 2. Feature activation matrix 구성
    # Shape: (n_trials, n_features_per_layer)

    feature_matrices = {}
    for layer in range(1, 32):
        layer_name = f'L{layer}'

        # 모든 trial의 해당 layer features 추출
        layer_matrix = []
        for trial_data in safe_baseline:
            layer_feats = trial_data['all_layer_features'][layer_name]
            layer_matrix.append(list(layer_feats.values()))

        feature_matrices[layer_name] = np.array(layer_matrix)
        # Shape: (30 trials, n_features_in_this_layer)

    # 3. Cross-layer correlation
    pathways = []

    for source_layer in range(1, 31):  # L1-L30
        for target_layer in range(source_layer + 1, 32):  # L2-L31

            source_name = f'L{source_layer}'
            target_name = f'L{target_layer}'

            source_matrix = feature_matrices[source_name]  # (30, n_src)
            target_matrix = feature_matrices[target_name]  # (30, n_tgt)

            # 각 source feature × target feature correlation
            for src_idx in range(source_matrix.shape[1]):
                for tgt_idx in range(target_matrix.shape[1]):

                    r, p = stats.pearsonr(
                        source_matrix[:, src_idx],
                        target_matrix[:, tgt_idx]
                    )

                    if abs(r) > 0.6 and p < 0.01:
                        pathways.append({
                            'source': f'{source_name}-{src_idx}',
                            'target': f'{target_name}-{tgt_idx}',
                            'correlation': r,
                            'p_value': p
                        })

    return pathways
```

#### Phase 3: Multi-hop Pathway Discovery

```python
def find_multihop_pathways(pathways):
    """
    3-layer pathways 발견: L_i → L_j → L_k
    """

    multi_hop = []

    # Pathway를 graph로 변환
    graph = defaultdict(list)
    for p in pathways:
        graph[p['source']].append(p['target'])

    # 3-hop paths 찾기
    for l1_feat in graph.keys():
        l1_layer = int(l1_feat.split('-')[0][1:])

        for l2_feat in graph[l1_feat]:
            l2_layer = int(l2_feat.split('-')[0][1:])

            for l3_feat in graph[l2_feat]:
                l3_layer = int(l3_feat.split('-')[0][1:])

                # Layer 순서 확인
                if l1_layer < l2_layer < l3_layer:

                    # Direct path도 확인
                    direct_corr = get_correlation(l1_feat, l3_feat, pathways)

                    multi_hop.append({
                        'path': f'{l1_feat} → {l2_feat} → {l3_feat}',
                        'layers': f'L{l1_layer} → L{l2_layer} → L{l3_layer}',
                        'hop1_corr': get_correlation(l1_feat, l2_feat, pathways),
                        'hop2_corr': get_correlation(l2_feat, l3_feat, pathways),
                        'direct_corr': direct_corr
                    })

    return multi_hop
```

---

## 📊 예상 결과

### 1. Safe Feature Pathways
```
Early layers (L1-L4) → Late layers (L25-L29)

L1-1292 (safe) → L24-1111 (safe) → L29-??? (safe)
  L1→L24: r = 0.72
  L24→L29: r = 0.68
  Direct L1→L29: r = 0.55
```

### 2. Risky Feature Pathways
```
Middle layers (L9-L17) → Late layers (L30)

L9-??? (risky) → L17-??? (risky) → L30-??? (risky)
  L9→L17: r = 0.81
  L17→L30: r = 0.75
  Direct L9→L30: r = 0.63
```

### 3. Cross-type Inhibition
```
Safe features → Risky features (negative correlation)

L1-1292 (safe) → L9-??? (risky): r = -0.78 (억제)
L24-1111 (safe) → L30-??? (risky): r = -0.82 (억제)
```

---

## 💻 구현 계획

### 계산량 추정

**데이터 수집**:
- 6 conditions × 30 trials = 180 trials
- 31 layers × 2,787 features (평균 ~90 per layer)
- 저장 공간: ~180 trials × 31 layers × 90 features × 8 bytes ≈ **4MB**

**Correlation 계산**:
- 2,787 features × 2,787 features ≈ 7.7M pairs
- Cross-layer만: ~1M pairs (manageable)

**예상 시간**:
- Feature extraction: ~180 trials × 2초/trial = **6분**
- Correlation analysis: ~1M pairs × 0.0001초 = **2분**
- **Total: ~10분** (feasible!)

### 실험 실행

```bash
# 1. 새 실험 코드 작성
/home/ubuntu/llm_addiction/experiment_3_pathway_tracking/
├── pathway_tracking_experiment.py
└── pathway_analysis.py

# 2. 실험 실행 (GPU 필요)
python pathway_tracking_experiment.py --gpu 4

# 3. 분석
python pathway_analysis.py
```

---

## 🎯 핵심 질문에 대한 답변

### Q1: Patching 실험 데이터로 pathway tracking 가능?
**A**: ❌ **불가능** (hidden states 저장 안 됨)

### Q2: 새로운 실험 구성 가능?
**A**: ✅ **가능** (10분 내 완료 가능)

### Q3: LlamaScope 사용 가능?
**A**: ✅ **사용 가능** (모든 layer SAE 로드)

### Q4: Pathway 분석 원리?
**A**: **Feature correlation across layers**
- Layer i의 feature A와 Layer j의 feature B의 상관관계 측정
- 높은 상관관계 = 정보 전달 가능성 (간접 증거)
- 한계: Correlation ≠ Causation

---

## 📚 참고문헌

1. **Anthropic (2025)**: "Attribution Graphs for Computational Pathways"
   - Token-level causal attribution
   - Cross-Layer Transcoders (CLTs)

2. **Experiment 1 Layer Pathway Analysis** (우리)
   - `/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/`
   - L8 → L10 → L31 pathway 발견 (correlation-based)

3. **Pearson Correlation**
   - Statistical measure of linear relationship
   - r > 0.6: strong positive correlation
   - p < 0.01: statistically significant

---

**Date**: 2025-10-22
**Author**: Analysis Documentation
**Purpose**: Pathway Analysis 원리 설명 및 새 실험 설계 제안
