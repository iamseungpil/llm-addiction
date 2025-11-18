# Proposal: Token-Level Feature Tracking Experiment

## 현재 한계
Experiment 1은 **last token features만** 추출하여 토큰 수준 정보 흐름 추적 불가능

## 제안: Enhanced Pathway Tracking

### 목표
1. **토큰별 feature 추적**: 모든 position의 features 추출
2. **Attention flow 분석**: 어떤 토큰이 어떤 토큰에 영향을 주는가
3. **Circuit tracing**: Anthropic 방법론 적용

### 실험 설계

#### Data Collection
```python
for each gambling decision:
    # 현재: Last token만
    current_data = {
        "L25": [32768]  # shape: (32768,)
    }

    # 제안: 모든 position
    proposed_data = {
        "L25_features": [[f1], [f2], ..., [fn]],  # shape: (seq_len, 32768)
        "L25_attention": [[[...]]]  # shape: (n_heads, seq_len, seq_len)
    }
```

#### 분석 방법

##### 1. Attribution Patching (Neel Nanda 방법론)
```python
# Clean vs Corrupted prompts
clean_prompt = "Current balance: $100. Bet or Stop?"
corrupted_prompt = "Current balance: $10. Bet or Stop?"  # Different state

# Patch specific positions
for layer in [1, 8, 15, 25, 31]:
    for position in range(seq_len):
        patch_activation(clean, corrupted, layer, position)
        measure_decision_change()
```

**발견 가능한 것**:
- "balance: $100" 토큰이 어느 레이어에서 중요한가?
- 과거 손실 기록이 어떤 레이어에서 처리되는가?

##### 2. Circuit Tracing (Anthropic 2025)
```python
# Attribution graphs 생성
attribution_graph = build_graph(
    input_tokens=["balance", "$100", "bet", "stop"],
    target_output="bet $10",
    layers=[1, 5, 10, 15, 20, 25, 31]
)

# Trace backward
critical_path = trace_backward(
    from_output="bet $10",
    to_inputs=["balance", "history"]
)
```

**발견 가능한 것**:
- "bet $10" 결정에 기여한 input tokens
- 각 레이어에서 활성화된 features와 그 연결
- L8 → L15 → L31로 이어지는 computational path

##### 3. Logit Lens Analysis
```python
# 각 레이어에서 예측하는 토큰
for layer in range(1, 32):
    intermediate_logits = unembed(hidden_states[layer])
    predicted_token = argmax(intermediate_logits)

# 발견: "언제부터" 모델이 "bet"을 예측하는가?
```

### 예상 결과

#### Before (현재 Experiment 1)
```
L8 has high discriminability (Cohen's d = 0.0234)
→ 하지만 "왜?"는 알 수 없음
```

#### After (Token-level tracking)
```
L8 discriminability explained:
- Position 12 ("$100") → Feature 1234 activated
- Feature 1234 → L15 Feature 5678
- L15-5678 → L31-9012 → Output "bet $10"

Critical circuit discovered:
  Balance token → L8-Risk Assessment → L15-Decision → L31-Output
```

### 구현 계획

#### Step 1: Minimal Prototype (10 games)
- 10개 게임만 token-level 추출
- L8, L15, L31만 분석 (full 31 layers는 나중)
- Attention patterns 포함

#### Step 2: Circuit Discovery
- Attribution patching으로 critical positions 발견
- Backward tracing으로 정보 흐름 추적

#### Step 3: Full Analysis
- 50 games × token-level × 31 layers
- Cross-layer feature correlation
- Decision circuit 완전 매핑

### 예상 소요

- **데이터 수집**: ~2-3시간 (10 games prototype)
- **분석**: ~5-10시간
- **총**: ~1-2일

### 기대 효과

1. **구체적 메커니즘 이해**:
   - "L8이 중요하다" → "L8의 어떤 feature가 왜 중요한가"

2. **토큰 수준 인과성**:
   - "$100" 토큰이 "bet $10" 결정에 얼마나 기여하는가

3. **Computational pathway**:
   - Input → L8 (risk assessment) → L15 (decision) → L31 (output)

4. **논문 가치**:
   - 기존 연구: Layer-level analysis
   - 우리 연구: Token-level circuit discovery in gambling

## 참고 문헌

1. Anthropic (2025). "Circuit Tracing: Revealing Computational Graphs"
   - https://transformer-circuits.pub/2025/attribution-graphs/

2. Neel Nanda. "Attribution Patching: Activation Patching At Industrial Scale"
   - https://www.neelnanda.io/mechanistic-interpretability/attribution-patching

3. Rai et al. (2024). "A Practical Review of Mechanistic Interpretability"
   - https://arxiv.org/abs/2407.02646

## 결론

**현재 Experiment 1**: "Which layer is important?"에 답변 (L8-L11 발견)
**제안 Experiment**: "How and why is it important?"에 답변

Token-level tracking으로 **실제 decision circuit 발견 가능**.
