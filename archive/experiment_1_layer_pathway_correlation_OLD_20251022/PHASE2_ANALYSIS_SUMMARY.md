# Phase 2 Analysis Summary: Information Flow in LLaMA Gambling Decisions

## 🎯 연구 질문

현재 데이터(last token features만)로 **"정보가 어떻게 흐르는가"**를 간접적으로 추론할 수 있는가?

## 📊 수행한 분석

### 1. Feature Pathway Tracing
**질문**: L8의 features가 L31의 features와 연결되는가?

**방법**: Cross-layer feature correlation (Pearson r)

**핵심 발견**:

#### 🔗 L8 → L31 Direct Pathways (r > 0.4, p < 0.01)

**강력한 Risky Pathway:**
```
L8-15043 (risky) → L31-3327 (risky): r = 0.80 ✅✅✅
L8-26623 (risky) → L31-3327 (risky): r = 0.84 ✅✅✅
L8-26623 (risky) → L31-12485 (risky): r = 0.76 ✅✅
L8-26623 (risky) → L31-10692 (risky): r = 0.73 ✅✅
```

**강력한 Safe Pathway:**
```
L8-12478 (safe) → L31-3327 (risky): r = -0.83 ✅✅✅ (억제)
L8-12478 (safe) → L31-10692 (risky): r = -0.73 ✅✅ (억제)
L8-12478 (safe) → L31-12178 (safe): r = 0.68 ✅✅
```

#### 🔗 L8 → L10 → L31 Three-Layer Pathways

**발견된 Computational Path:**
```
L8-2059 (risky) → L10-5950 (risky) → L31-10692 (risky)
  L8→L10: r = 0.68
  L10→L31: r = 0.62
  L8→L31: r = 0.59
```

**발견된 Safe Path:**
```
L8-12478 (safe) → L10-27828 (safe) → L31-12178 (safe)
  L8→L10: r = 0.88 ✅✅✅
  L10→L31: r = -0.84 (safe → risky 억제)
  L8→L31: r = -0.73 (direct safe → risky 억제)
```

### 2. Decision Signature Analysis
**질문**: "Bet" vs "Stop" 결정의 multi-layer 패턴은?

**방법**:
- 5개 critical layers (L8, L9, L10, L11, L31) × top 5 features = 25차원 벡터
- PCA, t-SNE로 decision space 시각화

**핵심 발견**:

#### 📈 Most Discriminative Features in Multi-Layer Signature

| Rank | Position | Layer-Feature | Cohen's d | Bet Mean | Stop Mean |
|------|----------|---------------|-----------|----------|-----------|
| 1 | 20 | L31-10692 | 3.502 | 0.761 | 0.510 |
| 2 | 21 | L31-12485 | 3.145 | 1.114 | 0.979 |
| 3 | 22 | L31-13816 | 2.794 | 0.081 | 0.022 |
| 4 | 0 | L8-2059 | 2.608 | 0.037 | 0.010 |
| 5 | 23 | L31-12178 | -2.528 | 0.205 | 0.307 (safe) |

**의미**:
- **L31 features가 가장 강력한 discriminator** (top 3 모두 L31)
- **L8이 4위**: 초기 risk assessment가 중요
- Multi-layer signature는 **L31과 L8이 핵심**

#### 🎨 Decision Space Visualization

**PCA 결과**:
- PC1: 84.0% variance (주요 축: bet vs stop)
- PC2: 3.9% variance
- **Bet/Stop 결정이 명확히 분리됨**

**t-SNE 결과**:
- Bankruptcy와 Safe 게임이 **명확히 cluster 형성**
- **소수의 bankruptcy games가 outlier로 분리**

### 3. Layer Contribution Analysis
**질문**: 각 레이어가 "bet" vs "stop" 결정에 얼마나 기여하는가?

**결과**:

| Layer | Bet Activation | Stop Activation | Difference |
|-------|---------------|-----------------|------------|
| L8 | 0.46 | 0.45 | +0.01 (거의 동일) |
| L9 | 0.43 | 0.44 | -0.01 (거의 동일) |
| **L10** | **0.87** | **0.97** | -0.10 (Stop 높음) |
| L11 | 0.40 | 0.39 | +0.01 (거의 동일) |
| **L31** | **0.51** | **0.39** | **+0.12 (Bet 높음)** |

**의미**:
- **L10**: Safe decisions에서 더 활성화 (억제 기능?)
- **L31**: Risky decisions에서 더 활성화 (최종 출력)
- **L8, L9, L11**: 두 그룹 간 차이 없음 (개별 features만 중요)

---

## 💡 핵심 인사이트

### 1. Information Flow의 간접 증거

**발견한 Computational Pathway:**
```
Input (balance, history)
  ↓
L8 Risk Assessment
  - L8-2059, L8-15043, L8-26623 (risky features)
  - L8-12478, L8-7472 (safe features)
  ↓ (r = 0.6~0.8 correlations)
L10 Intermediate Processing
  - L10-5950 (risky)
  - L10-27828, L10-15785 (safe)
  ↓ (r = 0.6~0.8 correlations)
L31 Final Decision
  - L31-10692, L31-12485, L31-3327 (risky → bet)
  - L31-12178 (safe → stop)
  ↓
Output: "Bet $10" or "Stop"
```

### 2. "왜 L8이 중요한가?" 부분 답변

**Phase 1 발견**: L8이 가장 높은 변별력 (Cohen's d = 0.0234)

**Phase 2 답변**:
1. **L8 features가 L31 features와 강하게 연결됨** (r = 0.6~0.8)
2. **L8-26623 → L31-3327 pathway가 가장 강력** (r = 0.84)
3. **L8의 safe features (12478)가 L31 risky features를 억제** (r = -0.83)

→ **L8은 초기 risk assessment를 수행하고, 그 결과를 L31로 전달하는 "gateway layer"**

### 3. 현재 분석의 한계

**할 수 있는 것**:
- ✅ Layer 간 feature correlation (어떤 feature가 어떤 feature와 연결되는가)
- ✅ Multi-layer decision signature (전체 레이어의 패턴)
- ✅ Layer-wise contribution (어느 레이어가 더 활성화되는가)

**할 수 없는 것**:
- ❌ Token-level attribution (어떤 input token이 영향을 주는가)
- ❌ Causal direction (correlation ≠ causation)
- ❌ Attention flow (attention patterns 없음)
- ❌ Position-specific processing (last token만 추출)

---

## 🔬 Anthropic 2025 방법론과의 비교

### Anthropic Circuit Tracing (2025)

**필요한 데이터**:
1. **All position features**: `(seq_len, 32768)` per layer
2. **Attention patterns**: `(n_heads, seq_len, seq_len)`
3. **Cross-Layer Transcoders (CLTs)**: MLP를 interpretable features로 분해
4. **Token embeddings**: 각 position의 token

**우리가 가진 데이터**:
- ✅ Last token features: `(32768,)` per layer
- ❌ All position features
- ❌ Attention patterns
- ❌ CLTs
- ❌ Token positions

**우리가 수행한 간접 방법**:
1. **Feature correlation** (Anthropic: Attribution graphs)
2. **Multi-layer vectors** (Anthropic: Computational paths)
3. **Layer-wise analysis** (Anthropic: Circuit tracing)

**차이**:
- Anthropic: **Causal graphs with token-level precision**
- 우리: **Correlational analysis with layer-level approximation**

---

## 📁 생성된 파일

1. **feature_network.png**: L8 → L31 feature correlation network
2. **correlation_heatmap.png**: L8 × L31 correlation matrix
3. **decision_space.png**: PCA/t-SNE decision space
4. **layer_contributions.png**: Layer activation by decision type
5. **feature_pathway_results.json**: Correlation data (25 L8→L31 correlations)
6. **decision_signature_results.json**: 25-dim decision vectors

---

## 🎯 답변: "현재 데이터로 정보 흐름 분석이 가능한가?"

### ✅ 가능한 것

1. **Layer 간 feature 연결성**: L8-26623 → L31-3327 같은 pathway 발견
2. **Multi-layer decision pattern**: 25차원 signature로 bet/stop 구분
3. **Layer 기여도**: L31이 최종 결정, L8이 초기 assessment
4. **Computational pathway 추정**: L8 → L10 → L31 경로 간접 확인

### ❌ 불가능한 것

1. **Token-level attribution**: "$100" 토큰이 "bet" 결정에 얼마나 기여하는가
2. **Causal validation**: Correlation만 있고 causation은 검증 불가
3. **Attention flow**: 어떤 토큰이 어떤 토큰에 attend하는가
4. **Position-specific**: 프롬프트의 어느 부분이 중요한가

### 💡 결론

**현재 분석으로 할 수 있는 것**:
- "어느 레이어의 어떤 features가 연결되는가" ✅
- "Multi-layer 패턴이 결정을 구분하는가" ✅
- "L8 → L31 경로가 존재하는가" ✅ (간접 증거)

**근본적 한계**:
- **Token-level tracking 없이는 "왜" 그 feature가 활성화되는지 알 수 없음**
- **Correlation ≠ Causation**: 연결성만 보고 인과성은 검증 불가

**해결책**:
- Token-level tracking 실험 수행 (PROPOSAL_token_level_tracking.md 참조)
- Anthropic 2025 방법론 적용 (CLTs, Attribution graphs)

---

## 📚 참고문헌

1. Anthropic (2025). "Circuit Tracing: Revealing Computational Graphs in Language Models"
   - https://transformer-circuits.pub/2025/attribution-graphs/

2. Pearson Correlation Analysis
   - Feature pathway tracing using statistical correlation
   - 25개 L8→L31 pathways 발견 (all p < 0.01)

3. PCA/t-SNE Dimensionality Reduction
   - Multi-layer decision signature visualization
   - 84% variance captured by first PC

---

**Date**: 2025-10-10
**Analysis**: Phase 2 Advanced Pathway Analysis
**Data**: Experiment 1 Layer Pathway L1-31 (50 games, 2.7GB)
