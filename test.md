# SAE 분석 실험 전체 비교 및 인과관계 실험 제안

## 📌 핵심 질문들에 대한 답변

### Q1. "최종 결정 직전" 시점이 맞는 분석 방법인가?

**현재 방법의 문제점:**

```
[실제 게임 진행]
Round 1: Balance $100 → Bet $50 → Loss → Balance $50
Round 2: Balance $50 → Bet $10 → Loss → Balance $40
...
Round 5: Balance $20 → Bet $20 → Loss → Balance $0
[게임 종료]

[SAE Feature Extraction]
→ 게임이 끝난 후, 마지막 상태 (Balance $0)를 재구성한 프롬프트에서 feature 추출
→ 프롬프트: "Current balance: $0 ... Choose: 1) Bet 2) Stop"
→ 이 시점의 hidden state → SAE encoding
```

**문제:**
1. **시간적 역전**: 게임이 이미 끝난 후의 상태를 재구성
2. **인과관계 모호**: "이 feature가 파산을 야기했는가?" vs "파산 후 상태를 표현하는 feature인가?"
3. **대안 필요**: 매 라운드별 feature 추출 필요할 수도

---

### Q2. Fig1/Fig3는 왜 상관관계만 보여주는가?

**현재 분석 (additional_experiments/sae_condition_comparison):**

```python
# 단순히 그룹 간 feature activation 평균 비교
variable_features = features[bet_types == 'variable']  # (1600, 32768)
fixed_features = features[bet_types == 'fixed']        # (1600, 32768)

# Feature 26280의 평균 비교
t_stat, p_value = ttest_ind(variable_features[:, 26280],
                             fixed_features[:, 26280])
# p < 0.05 → "Variable 게임에서 이 feature가 높다"
```

**이것은 상관관계일 뿐:**
- ✅ "Variable 조건과 이 feature가 연관되어 있다"
- ❌ "이 feature를 조작하면 Variable-like 행동을 한다" (인과)

---

## 🔬 기존 인과관계 실험들 비교

### 1️⃣ Phase 4 Causal Pilot v2 (llama_sae_analysis)

**실험 설계:**

```python
# Input: Phase 2에서 나온 Safe/Risky features (Outcome-based)
# - Safe features: Cohen's d < 0 (Safe 게임에서 높음)
# - Risky features: Cohen's d > 0 (Bankrupt 게임에서 높음)

# Method: Residual Stream Patching
def generate_with_patching(prompt, layer, feature_id, alpha):
    # Forward pass with hook
    def hook(module, input, output):
        h = output.clone()
        h = h + alpha * W_dec[feature_id]  # Feature direction 더하기
        return h

    model.layers[layer].register_forward_hook(hook)
    response = model.generate(prompt)
    return response

# Test prompts (고정된 3개)
prompts = ['losing_streak', 'winning_streak', 'low_balance']

# Alpha values (양방향)
alphas = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]

# Measurement: Stop rate 변화
```

**실험 결과:**
```
총 features 테스트: 10개 (Safe 5 + Risky 5)
총 조건: 60개
Significant effects (p<0.05): 0개 ❌
Parse success rate: 11.8% ❌❌

→ 완전 실패!
```

**실패 원인:**
1. **Response parsing 실패**: 모델 출력을 "Bet" vs "Stop"으로 파싱 못함
2. **프롬프트 고정**: 테스트 프롬프트 3개만 사용 (실제 게임과 다름)
3. **Alpha 값 과다**: 0.5는 너무 큰 개입일 수 있음

---

### 2️⃣ Pathway Token Analysis (pathway_token_analysis)

**다른 접근:**
- Token-level feature tracking
- Temporal analysis (라운드별 변화)
- 별도 연구 목적

---

### 3️⃣ 현재 Condition Comparison (우리 분석)

**차이점:**

| 측면 | Phase 4 Causal Pilot | Condition Comparison (우리) |
|------|---------------------|---------------------------|
| **Feature 구분 기준** | Outcome (Safe vs Risky) | **Bet Type (Variable vs Fixed)** |
| **분석 방법** | Causal intervention | Correlation only |
| **데이터** | 합성 테스트 프롬프트 | 실제 게임 데이터 3200개 |
| **결과** | 실패 (parsing issue) | 성공 (11,999개 유의 feature) |
| **한계** | - | **인과관계 검증 없음** |

---

## 🎯 제안: Variable/Fixed Feature의 인과관계 실험

### **왜 새로운 실험이 필요한가?**

**기존 실험의 맹점:**
```
Phase 4 실험은 "Outcome" 차원만 봄:
- Safe features: 안전하게 멈춘 게임의 feature
- Risky features: 파산한 게임의 feature

하지만 우리가 발견한 것:
- Variable-associated features: Variable betting 조건의 feature
- Fixed-associated features: Fixed betting 조건의 feature

→ 완전히 다른 차원!
```

**예시:**
```
Feature L12-26280 (Cohen's d = 3.34):
- Variable 게임: 평균 activation 0.35
- Fixed 게임: 평균 activation 0.08

이 feature가 정말로 "Variable-like 행동"을 유발하는가?
→ 인과관계 실험 필요!
```

---

## 🔧 구체적 실험 설계 (4가지 제안)

### **실험 1: Direct Feature Manipulation**

**가설:** "Variable-associated feature를 활성화 → Variable-like 행동 유도"

**방법:**
```python
# Control: Fixed betting 게임 100개
# - 프롬프트: "Choose: 1) Bet $10 (fixed) 2) Stop"
# - 기대 행동: 보수적 베팅

# Intervention: Variable feature 활성화
for game in fixed_games:
    # Top 10 Variable features (Cohen's d > 2.0)
    for feature in top_variable_features:
        # Patching
        patched_response = generate_with_patching(
            game.prompt,
            layer=feature.layer,
            feature_id=feature.id,
            alpha=0.3  # 작은 값 사용 (기존 0.5 → 0.3)
        )

        # Measure: 베팅 금액 변화 (실제로는 $10 고정이지만 심리적 변화)
```

**예상 결과:**
- Variable feature 활성화 → "Bet" 선택 증가? 또는 reasoning에서 "더 큰 금액" 언급?

**차별점:**
- ✅ 실제 게임 프롬프트 사용 (합성 X)
- ✅ Variable/Fixed 차원 (Outcome이 아님)
- ✅ 작은 alpha (0.3) + better parsing

---

### **실험 2: Cross-Condition Transfer**

**가설:** "Variable 게임에 Fixed feature 주입 → 보수적 행동"

**방법:**
```python
# Variable-Bankrupt 게임 선택
variable_bankrupt_games = games[
    (bet_types == 'variable') & (outcomes == 'bankruptcy')
]  # 108개

# Top 10 Fixed-Safe features (Cohen's d < -2.0) 주입
for game in variable_bankrupt_games:
    # Counterfactual: Fixed feature로 치환
    mean_activation_fixed = features[fixed_mask, feature_id].mean()

    # Patching
    response = generate_with_feature_value(
        game.prompt,
        layer=feature.layer,
        feature_id=feature.id,
        target_value=mean_activation_fixed  # Fixed 평균값으로
    )
```

**측정:**
- 파산율 감소? (예: 108개 → 50개?)
- 베팅 금액 감소?

---

### **실험 3: Multi-Feature Intervention**

**가설:** "여러 Variable features 동시 활성화 → 더 강한 효과"

**방법:**
```python
# Top 5 Variable features 동시 조작
top_5 = variable_features_sorted_by_cohens_d[:5]

for game in fixed_safe_games:
    # Multi-feature patching
    def multi_hook(module, input, output):
        h = output.clone()
        for feat in top_5:
            h = h + 0.2 * W_dec[feat.id]  # 각각 작은 값
        return h

    response = generate_with_hook(game.prompt, multi_hook)
```

**예상:**
- 단일 feature보다 강한 효과
- 베팅 성향 변화 명확

---

### **실험 4: Ablation (제거 실험)**

**가설:** "Variable feature 제거 → Variable 게임이 Fixed-like 행동"

**방법:**
```python
# Variable 게임에서 Variable features 제거
for game in variable_games:
    # SAE level intervention
    sae_output = sae.encode(hidden_state)

    # Top 10 Variable features → 0
    for feat in top_variable_features:
        sae_output[feat.id] = 0.0

    # Reconstruct
    modified_h = sae.decode(sae_output)

    # Forward pass
    response = model.forward_from(layer+1, modified_h)
```

**측정:**
- 파산율 감소? (6.8% → 3%?)
- 평균 베팅 금액 감소?

---

## 📊 기존 vs 제안 실험 비교표

| 측면 | Phase 4 (기존) | 제안 실험 |
|------|---------------|----------|
| **Feature 기준** | Outcome (Safe/Risky) | **Bet Type (Var/Fixed)** |
| **프롬프트** | 합성 3개 | **실제 게임 3200개** |
| **Alpha 범위** | -0.5 ~ 0.5 | **-0.3 ~ 0.3** (더 안전) |
| **Parsing** | 11.8% 성공 | **개선된 파서** 필요 |
| **측정 지표** | Stop rate만 | **Stop rate + 베팅 금액 + 파산율** |
| **통계 검정** | Fisher's exact | **Fisher + t-test + 효과 크기** |

---

## ✅ 실행 계획

### **Step 1: Parsing 개선 (필수)**
```python
# 기존 파서 문제: "Final Decision: Bet $X" 형식 강제
# → LLaMA는 이 형식을 잘 안 따름

# 개선안:
def improved_parser(response):
    # 1. Logits 직접 확인 (generation 전)
    logits = model(prompt).logits[:, -1, :]
    bet_token_id = tokenizer.encode("Bet")[0]
    stop_token_id = tokenizer.encode("Stop")[0]

    if logits[0, bet_token_id] > logits[0, stop_token_id]:
        return "BET"
    else:
        return "STOP"

    # 2. Multiple-choice 형식 강제
    prompt += "\nYour answer (type 1 or 2): "
```

### **Step 2: Pilot Experiment (2-3시간)**
- Feature 10개 선택 (Variable top 5 + Fixed top 5)
- Fixed 게임 50개만 테스트
- Alpha = [0.1, 0.2, 0.3]
- Parsing 검증

### **Step 3: Full Experiment (1-2일)**
- Feature 50개
- 게임 200개
- 전체 통계 분석

---

## 🔍 결론

**현재 상황:**
1. ✅ **상관관계 발견**: Variable/Fixed features 11,999개 (성공!)
2. ❌ **인과관계 검증**: 없음
3. ❌ **기존 인과 실험**: Outcome 차원만, parsing 실패

**해야 할 일:**
1. **Parsing 먼저 고치기** (이게 핵심!)
2. Variable/Fixed feature의 **인과관계 실험** (새로운 차원)
3. 실제 게임 프롬프트 사용 (합성 프롬프트 X)

**질문:**
1. Parsing 개선부터 시작할까요? (Logits 직접 확인 방식)
2. 아니면 간단한 Pilot (10 features, 50 games)부터?
3. 기존 Phase 4 코드를 수정해서 재실행?
