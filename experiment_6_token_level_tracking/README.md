# Experiment 6: Token-Level Feature Tracking

## 🎯 목표

Experiment 1의 한계를 극복하여 **token-level attribution** 가능하게 만들기

## ❌ Experiment 1의 한계

```python
# Experiment 1 데이터 구조
{
  "L8": [32768 floats]  # Last token만, shape: (32768,)
}
```

**불가능했던 것**:
1. ❌ Token-level attribution: "$100" 토큰의 영향 불가
2. ❌ Causal validation: Correlation만 가능
3. ❌ Attention flow: Attention patterns 없음
4. ❌ Position-specific: Last token만 추출

## ✅ Experiment 6의 해결책

```python
# Experiment 6 데이터 구조
{
  "tokens": ["Current", "balance", ":", "$", "100", ...],
  "layers": {
    "L8": {
      "features": [[f1], [f2], ..., [fn]],  # shape: (seq_len, 32768) ✅
      "attention": [[[...]]]                 # shape: (n_heads, seq_len, seq_len) ✅
    }
  }
}
```

**가능한 것**:
1. ✅ Token-level attribution: 각 토큰의 contribution 측정
2. ✅ Attention flow: 어떤 토큰이 output에 attend하는지
3. ✅ Position-specific analysis: "$100" 위치의 features 추출
4. ✅ Token → Feature → Output pathway 추적

---

## 📊 실험 설계

### Data Collection

**Critical Layers**: L8, L15, L31 (Phase 1에서 중요하다고 발견)

**각 게임마다 수집**:
1. **Tokens**: 모든 token positions
2. **Features**: `(seq_len, 32768)` per layer
3. **Attention**: `(n_heads, seq_len, seq_len)` per layer

### Sample Size

**Prototype**: 10 games (빠른 검증)
- 예상 소요: ~30분
- 파일 크기: ~100-200MB

**Full**: 50 games (Experiment 1과 동일)
- 예상 소요: ~2-3시간
- 파일 크기: ~1-2GB

---

## 🚀 실행 방법

### 1. Prototype 실행 (10 games)

```bash
cd /home/ubuntu/llm_addiction/experiment_6_token_level_tracking
conda activate llama_sae_env
python experiment_6_token_tracking.py
```

### 2. 분석

```bash
# Token attribution 분석
python analyze_token_attribution.py /data/llm_addiction/experiment_6_token_level/token_level_tracking_*.json
```

---

## 📈 예상 결과

### Token Attribution

```
L8 Token Attribution:
Top 10 tokens contributing to output:
Rank   Position   Token                Importance    Attention     ||Features||
1      45         $100                 0.234567      0.156         1.234
2      12         balance              0.123456      0.089         0.987
3      67         Bet                  0.098765      0.234         0.456
...
```

### Attention Flow

각 레이어에서:
- "$100" 토큰이 output에 얼마나 attend하는지
- "balance" 토큰의 영향
- "Bet" vs "Stop" 선택지의 attention

### Feature Heatmap

Position × Feature 히트맵:
- 어느 position에서 어떤 feature가 활성화되는지
- "$100" 위치에서 L8-2059 같은 risky feature 활성화?

---

## 🔬 Anthropic 2025 방법론 적용

### 현재 가능 (Experiment 6 완료 후)

1. **Attribution Patching**
   ```python
   # "$100" → "$10" 로 바꿔서 feature 변화 측정
   clean_prompt = "Current balance: $100"
   corrupted_prompt = "Current balance: $10"

   # Position 45 ($100) 패칭
   patch_activation(layer=8, position=45, from_prompt=corrupted, to_prompt=clean)
   measure_output_change()
   ```

2. **Attention-weighted Feature Attribution**
   ```python
   # "$100" 토큰의 기여도
   contribution = (
       attention_to_output[pos_$100] *
       feature_magnitude[pos_$100] *
       feature_importance[L8-2059]
   )
   ```

3. **Backward Tracing**
   ```python
   # Output "Bet $10" ← L31-10692 ← L8-2059 ← Position "$100"
   trace_pathway(
       from_output="Bet",
       through_features=[("L31", 10692), ("L8", 2059)],
       to_input_position=45  # "$100"
   )
   ```

### 아직 불가능 (CLTs 필요)

1. **Cross-Layer Transcoders (CLTs)**
   - Anthropic가 따로 학습한 모델
   - Layer 간 feature 연결을 decompose
   - 우리는 correlation으로 근사

2. **Causal Graphs**
   - Anthropic: 완전한 computational graph
   - 우리: Attention + correlation 기반 pathway

---

## 📁 파일 구조

```
experiment_6_token_level_tracking/
├── experiment_6_token_tracking.py    # 메인 실험 코드
├── analyze_token_attribution.py     # 분석 코드
├── README.md                         # 이 파일
└── /data/llm_addiction/experiment_6_token_level/
    └── token_level_tracking_*.json  # 결과 파일
```

---

## 🎯 답변: "불가능한 것을 다시 실험으로 확인 가능한가?"

### ✅ YES! Experiment 6으로 가능

| 항목 | Experiment 1 | Experiment 6 |
|------|--------------|--------------|
| Token-level attribution | ❌ | ✅ Position-specific features |
| Causal validation | ❌ | ✅ Attribution patching 가능 |
| Attention flow | ❌ | ✅ Full attention patterns |
| Position-specific | ❌ | ✅ All positions extracted |

### 실행 가능 분석

1. **"$100" 토큰이 "bet" 결정에 얼마나 기여하는가?**
   - ✅ Attention weight × Feature magnitude로 측정

2. **"balance" 토큰의 영향은?**
   - ✅ Position-specific feature activation 분석

3. **L8의 어떤 position에서 risky features가 활성화되는가?**
   - ✅ Feature heatmap으로 시각화

4. **Token → Feature → Output pathway?**
   - ✅ Attention flow + feature correlation으로 추적

---

## ⏱️ 예상 소요 시간

**Prototype (10 games)**:
- 데이터 수집: ~20-30분
- 분석: ~10분
- 총: ~40분

**Full (50 games)**:
- 데이터 수집: ~2-3시간
- 분석: ~30분
- 총: ~3-4시간

---

## 🚀 즉시 시작 가능

**GPU 2 사용 가능** (0 MiB / 81920 MiB)

```bash
# GPU 2에서 prototype 실행
tmux new -s exp6_token_level
conda activate llama_sae_env
cd /home/ubuntu/llm_addiction/experiment_6_token_level_tracking
python experiment_6_token_tracking.py
```

**완료 후**:
- Token attribution 분석
- Attention flow 시각화
- Experiment 1과 비교
- 논문에 추가 가능한 결과

---

**Date**: 2025-10-10
**Status**: Ready to run
**GPU**: 2 (available)
