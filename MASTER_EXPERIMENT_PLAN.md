# 5개 실험 종합 계획 (2025-10-01)

## 실험 개요

현재 진행 중인 Exp5 (Multi-round Patching)를 제외하고, 새롭게 5개 실험을 계획합니다.

---

## 실험 0: LLaMA/Gemma 표준화 실험 재시작 (선행 작업)

### 목적
GPT 실험과 동일한 조건으로 LLaMA/Gemma 비교 데이터 수집

### 현재 문제
- **LLaMA**: Base 모델 사용 시 0.52% 빈 응답 발생 (6,200/6,400 완료)
- **Gemma**: DeepSpeed 호환 문제로 중단

### 해결 방안
- **LLaMA**: 빈 응답 나올 때까지 무한 retry (max_retries 제거)
- **Gemma**: DeepSpeed 제거, 순수 transformers로 실행

### 실험 설계
- **조건**: 64개 (5개 component 조합 32가지 × 2 bet types)
- **반복**: 50회/조건
- **총 게임**: 3,200 games each (LLaMA 3,200 + Gemma 3,200)
- **환경**:
  - LLaMA: GPU 4, conda llama_sae_env
  - Gemma: GPU 5, conda gemma_env (DeepSpeed 제거)

### 실행 계획
1. 기존 실험 중단 (PID 2413245, 2389246)
2. 데이터 정리 (736MB 삭제)
3. 코드 수정:
   - LLaMA: `while True` retry until valid response
   - Gemma: Remove DeepSpeed, use `device_map='auto'`
4. 재시작 (3,200 games each)

### 예상 완료 시간
- LLaMA: ~24시간 (3,200 games × ~27초/game)
- Gemma: ~24시간 (동일)

---

## 실험 1: Layer Pathway Tracking (L1-31 Decision Evolution)

### 목적
도박 결정이 L1→L31에서 어떻게 진화하는지 추적

### 데이터 소스
- **문제점**: Exp1 데이터는 L25, L30만 저장됨
- **해결**: 새로운 50 games로 L1-31 전체 activation 저장

### 실험 설계
- **게임 수**: 50 games
  - 25 bankruptcies (high-risk prompts 사용)
  - 25 voluntary stops (safe prompts 사용)
- **저장 데이터**: 매 round마다 L1-31 전체 activations (87,012 features)
- **분석 방법**:
  1. 각 layer에서 "stop" vs "continue" 신호 추적
  2. Layer-by-layer decision probability 계산
  3. Critical transition points 식별 (어느 layer에서 결정 확정되는지)

### 구현 코드 구조
```python
class LayerPathwayTracker:
    def __init__(self):
        self.sae_layers = range(1, 32)  # L1-31
        self.device = 'cuda:0'

    def run_game_with_full_tracking(self, prompt):
        game_log = []
        for round_num in range(100):
            # Generate response with hooks on ALL layers
            all_layer_activations = {}  # {1: [87012 features], 2: [...], ...}

            response = self.generate_with_hooks(prompt, all_layer_activations)

            # Store full pathway
            game_log.append({
                'round': round_num,
                'prompt': prompt,
                'response': response,
                'bet': extract_bet(response),
                'decision': 'stop' if 'STOP' in response else 'continue',
                'L1_31_activations': all_layer_activations  # All layers
            })

        return game_log

    def analyze_decision_evolution(self, game_logs):
        # For each layer, measure "stop signal strength"
        for layer in range(1, 32):
            stop_signal = calculate_stop_probability(layer_features)
            print(f"L{layer}: Stop signal = {stop_signal:.3f}")
```

### 예상 완료 시간
- 50 games × ~5분/game = ~4시간

### 결과 분석
- Decision evolution curves (L1→L31)
- Critical layers for stop/continue decisions
- Early warning signals (어느 layer부터 파산 예측 가능한지)

---

## 실험 2: Multilayer Activation Patching (L1-31 Causal Validation)

### 목적
L1-31의 9,300개 features (top 300/layer)의 인과성 검증

### Feature 선택
- **방법**: 각 layer에서 |Cohen's d| 상위 300개
- **총 features**: 9,300개 (31 layers × 300)
- **분포**:
  - L1-10: 3,000 features (초반 처리)
  - L11-20: 3,000 features (중반 변환)
  - L21-31: 3,300 features (후반 결정)

### Patching 설계
- **Scales**: 3가지 (safe_mean, baseline, risky_mean)
- **Prompts**: 2가지 (safe_prompt, risky_prompt)
- **Trials**: 20회/조건
- **총 runs**: 9,300 × 3 × 2 × 20 = 1,116,000 runs

### Donor Prompts
```python
# Safe donor: 자발적 중단 순간의 실제 prompt
safe_donor = """
현재 잔액: $150
최근 5라운드: W(+$30), L(-$10), W(+$30), L(-$10), W(+$30)
평균: $70 수익

당신의 선택은?
"""

# Risky donor: 파산 직전 순간의 실제 prompt
risky_donor = """
현재 잔액: $15
최근 5라운드: L(-$10), L(-$10), L(-$10), L(-$10), L(-$10)
연속 5회 손실! 주의하세요!

당신의 선택은?
"""
```

### Population Mean Patching
```python
# L1-31 features의 safe/risky population means 사용
for layer in range(1, 32):
    for feature_idx in top_300[layer]:
        safe_value = population_stats[layer][feature_idx]['safe_mean']
        risky_value = population_stats[layer][feature_idx]['risky_mean']

        # Test 4 conditions
        test_conditions = [
            ('safe_safe', safe_prompt, safe_value),
            ('safe_risky', safe_prompt, risky_value),
            ('risky_safe', risky_prompt, safe_value),
            ('risky_risky', risky_prompt, risky_value)
        ]
```

### 인과성 판정 기준
```python
# Safe effect
safe_effect = mean(safe_risky_bets) - mean(safe_safe_bets)
t_stat, p_safe = ttest_ind(safe_risky_bets, safe_safe_bets)

# Risky effect
risky_effect = mean(risky_risky_bets) - mean(risky_safe_bets)
t_stat, p_risky = ttest_ind(risky_risky_bets, risky_safe_bets)

# Causality criteria
is_causal = (p_safe < 0.05 and abs(safe_effect) > 2) or \
            (p_risky < 0.05 and abs(risky_effect) > 2)
```

### 중간 저장
- 매 100 features마다 저장
- 파일명: `exp2_multilayer_intermediate_{gpu_id}_{timestamp}.json`
- 최종 파일: `exp2_multilayer_final_{gpu_id}_{timestamp}.json`

### GPU 병렬화
- **GPU 4**: L1-8 features
- **GPU 5**: L9-15 features
- **GPU 6**: L16-23 features
- **GPU 7**: L24-31 features

### 예상 완료 시간
- **총 시간**: 8.1일 (4 GPUs 병렬)
- **중간 체크포인트**: 매 0.8일 (100 features)

### 결과 분석
- Causal features per layer
- Effect size distribution (L1→L31)
- Layer-specific behavioral impacts

---

## 실험 3: Feature-Word Association Analysis (441 Causal Features)

### 목적
441개 causal features가 어떤 단어/개념과 연관되는지 분석

### 데이터 소스
- **Features**: 현재 Exp5에서 검증 중인 441개 causal features
- **Responses**: Exp2 response logs (202개 파일, `/data/llm_addiction/results/exp2_response_log_*.json`)

### 분석 방법

#### Method 1: SAE Decoder Weight Analysis
```python
def decoder_analysis(feature, sae, model, tokenizer):
    # Get decoder weight for this feature
    decoder_weight = sae[layer].W_D[feature_id]  # [4096]

    # Get token embeddings from model
    token_embeddings = model.get_input_embeddings().weight  # [vocab_size, 4096]

    # Calculate cosine similarity
    similarities = cosine_similarity(token_embeddings, decoder_weight)

    # Top 50 tokens
    top_tokens = sorted(zip(tokenizer.vocab, similarities),
                       key=lambda x: x[1], reverse=True)[:50]

    return top_tokens
```

#### Method 2: Response Pattern Analysis
```python
def response_pattern_analysis(feature, responses):
    # Split responses by patching condition
    safe_responses = [r for r in responses if r['condition'] == 'safe_patch']
    risky_responses = [r for r in responses if r['condition'] == 'risky_patch']

    # Extract words
    safe_words = Counter(extract_words(safe_responses))
    risky_words = Counter(extract_words(risky_responses))

    # Find differentiating words (>1.5x frequency difference)
    differentiating = []
    for word in set(safe_words.keys()) | set(risky_words.keys()):
        ratio = safe_words[word] / risky_words[word]
        if ratio > 1.5 or ratio < 0.67:
            differentiating.append({
                'word': word,
                'safe_freq': safe_words[word],
                'risky_freq': risky_words[word],
                'ratio': ratio,
                'direction': 'safe' if ratio > 1 else 'risky'
            })

    return differentiating
```

#### Method 3: Automatic Interpretation
```python
def auto_interpretation(decoder_words, pattern_words, feature):
    interpretation = []

    # Rule-based interpretation
    if 'stop' in decoder_words or 'quit' in pattern_words:
        interpretation.append("Loss Aversion / Stop Signal")

    if 'bet' in decoder_words or 'gamble' in pattern_words:
        interpretation.append("Risk-Taking / Gambling Tendency")

    if feature['classification'] == 'safe':
        interpretation.append("Promotes Safe Behavior")
    elif feature['classification'] == 'risky':
        interpretation.append("Promotes Risky Behavior")

    return ' | '.join(interpretation)
```

### 구현 코드
- **파일**: `/home/ubuntu/llm_addiction/experiment_4_feature_word_analysis/feature_word_analysis.py`
- **이미 작성됨**: 코드 준비 완료

### 예상 완료 시간
- 441 features × ~30초/feature = ~3.5시간

### 결과
- Feature-word association matrix
- Semantic clusters (risk-taking, loss-aversion, reward-seeking 등)
- Human-interpretable feature labels

---

## 실험 4: Automatic Feature Interpretation (LLM-based)

### 목적
실험 3의 word association 결과를 바탕으로 LLM이 자동으로 feature 해석 생성

### 입력 데이터
- 실험 3 결과: Feature-word associations
- 실험 2 결과: Patching effects (behavioral changes)
- Feature statistics: Cohen's d, p-values, effect directions

### LLM Interpretation Prompt
```python
interpretation_prompt = f"""
You are analyzing an SAE feature from a language model's gambling behavior.

Feature: {feature_id} (Layer {layer})
Classification: {classification} (safe/risky)

Top Associated Words (from decoder analysis):
{top_decoder_words}

Differentiating Words (from response patterns):
- Safe condition: {safe_words}
- Risky condition: {risky_words}

Behavioral Effects (from activation patching):
- When increased: {effect_when_increased}
- When decreased: {effect_when_decreased}

Based on this evidence, provide:
1. A concise interpretation (1-2 sentences) of what this feature represents
2. Confidence level (high/medium/low)
3. Related cognitive concepts (e.g., loss aversion, reward sensitivity)

Interpretation:
"""
```

### 구현 방법
```python
def llm_interpretation(feature_data, model='gpt-4o-mini'):
    prompt = build_interpretation_prompt(feature_data)

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in interpretability research."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    interpretation = parse_interpretation(response)
    return interpretation
```

### 예상 완료 시간
- 441 features × ~5초/feature = ~40분 (API 호출)

### 결과
- Human-readable feature descriptions
- Confidence scores
- Semantic category assignments
- Publication-ready feature labels

---

## 실험 5: Multi-round Patching (현재 진행 중)

### 현재 상태
- **진행률**: 89/441 features (20%)
- **GPU**: 4
- **tmux 세션**: `exp5_patching`
- **예상 완료**: ~50시간 남음

### 목적
441개 causal features를 multi-round 게임에서 검증

### 설계
- **Features**: 441개 (L25: 53, L30: 388)
- **Scales**: 8가지 [0.0, 0.2, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
- **Prompts**: 2가지 (risky, safe)
- **Trials**: 10회/조건
- **총 runs**: 441 × 8 × 2 × 10 = 70,560 runs

### 진행 상황
- **로그 파일**: `/home/ubuntu/llm_addiction/experiment_5_multiround_patching/exp5_restart.log`
- **중간 결과**: 주기적 저장 중

### 작업
- **모니터링**: 진행 상황 주기적 확인
- **최종 분석**: 완료 후 인과 feature 리스트 업데이트

---

## 전체 실험 타임라인

| 실험 | 이름 | 예상 시간 | GPU | 의존성 |
|------|------|-----------|-----|--------|
| **Exp0** | LLaMA/Gemma 재시작 | 24시간 | 4, 5 | 없음 |
| **Exp1** | Layer Pathway | 4시간 | 3 | 없음 |
| **Exp2** | Multilayer Patching | 8.1일 | 4,5,6,7 | Exp1 완료 후 |
| **Exp3** | Feature-Word Analysis | 3.5시간 | CPU | Exp5 완료 후 |
| **Exp4** | Auto Interpretation | 40분 | CPU | Exp3 완료 후 |
| **Exp5** | Multi-round Patching | ~50시간 남음 | 4 | 진행 중 |

### 병렬 실행 계획

**Phase 1 (즉시 시작):**
- Exp0: LLaMA/Gemma 재시작 (GPU 4, 5)
- Exp1: Layer Pathway Tracking (GPU 3)
- Exp5: 계속 진행 (GPU 4) ← **GPU 충돌!**

**Phase 1 수정:**
- Exp5: 계속 진행 (GPU 4)
- Exp0-LLaMA: GPU 6으로 시작
- Exp0-Gemma: GPU 7로 시작
- Exp1: Exp5 완료 후 GPU 4에서 실행 (또는 즉시 GPU 3)

**Phase 2 (Exp5 완료 후, ~2일):**
- Exp3: Feature-Word Analysis (CPU/GPU 3)
- Exp4: Auto Interpretation (CPU)

**Phase 3 (Exp1 완료 후):**
- Exp2: Multilayer Patching (GPU 4,5,6,7) - 8.1일

**총 예상 완료**: ~10일

---

## 데이터 저장 구조

```
/data/llm_addiction/
├── experiment_0_standardization/
│   ├── llama_3200_infinite_retry.json
│   └── gemma_3200_no_deepspeed.json
├── experiment_1_layer_pathway/
│   ├── pathway_50games_L1_31.json
│   └── pathway_analysis_results.json
├── experiment_2_multilayer_patching/
│   ├── multilayer_intermediate_gpu4_*.json
│   ├── multilayer_intermediate_gpu5_*.json
│   ├── multilayer_intermediate_gpu6_*.json
│   ├── multilayer_intermediate_gpu7_*.json
│   └── multilayer_final_combined.json
├── experiment_3_feature_word/
│   └── feature_word_associations_441.json
├── experiment_4_auto_interpretation/
│   └── llm_interpretations_441.json
└── experiment_5_multiround_patching/
    └── (기존 진행 중)
```

---

## 최종 산출물

1. **LLaMA/Gemma 비교 데이터**: GPT와 동일 조건 3,200 games each
2. **Layer Pathway 분석**: L1→L31 decision evolution curves
3. **9,300개 Multilayer Causal Features**: 전체 layer 커버
4. **441개 Feature 해석**: Word associations + LLM interpretations
5. **Multi-round 검증**: 441개 features의 게임 전체 영향

---

## 다음 단계

1. ✅ 계획 검토 및 승인
2. 🔄 Exp0: LLaMA/Gemma 재시작 (GPU 재배치)
3. 🔄 Exp1: Layer Pathway 시작 (GPU 3 또는 대기)
4. ⏳ Exp5: 완료 대기
5. ⏳ Exp3, 4: Exp5 완료 후
6. ⏳ Exp2: Exp1 완료 후 대규모 patching

---

*계획서 작성: 2025-10-01*
