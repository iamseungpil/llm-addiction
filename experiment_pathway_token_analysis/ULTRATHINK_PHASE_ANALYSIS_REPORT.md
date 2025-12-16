# Ultrathink: Phase 1-5 Pathway Analysis 코드 검증

**Date**: 2025-11-25
**Status**: 🚨 **심각한 문제 발견**

---

## 1. 2,510개가 재파싱 결과인지 확인 ✅

### 확인됨
```
/data/llm_addiction/experiment_2_multilayer_patching/reparsed/reparsed_causal_features_20251125_043558.json
└── total_causal_features: 2510
└── timestamp: 20251125_043558
```

### 비교
| 항목 | 원본 파싱 | 재파싱 |
|------|----------|--------|
| Causal features | 922개 | **2,510개** |
| 합의율 | - | 66.1% |
| 새로 발견 | - | +2,260개 |
| 제외됨 | - | -672개 |

**✅ 2,510개는 확실히 재파싱 결과입니다.**

---

## 2. Phase 1-5 코드 분석

### Phase 1: Patching + Multi-Feature Extraction

#### SAE Feature 사용 ✅
```python
# Line 110-117: SAE 로딩 올바름
def load_sae(self, layer: int):
    self.sae_cache[layer] = LlamaScopeDirect(layer=layer, device=str(self.device))
    return self.sae_cache[layer]

# Line 163-184: SAE를 통한 feature activation 추출 올바름
def extract_all_features(self, hidden_states_dict: dict):
    for layer, hidden_states in hidden_states_dict.items():
        sae = self.load_sae(layer)
        feature_acts = sae.encode(hidden_states.float())  # ✅ 올바른 SAE 인코딩
        final_acts = feature_acts[0, -1, :]  # ✅ 마지막 토큰 사용
```

#### Token 저장 ✅ (수정됨)
```python
# Line 262-265: 실제 BPE 토큰 저장 (내가 수정함)
prompt_len = inputs['input_ids'].shape[1]
generated_token_ids = full_sequence[prompt_len:].tolist()  # ✅ 실제 토큰 ID
generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]  # ✅ 디코딩된 토큰
```

#### 🚨 **문제 1: 잘못된 Feature Means 파일**

```python
# Line 420-423: 기본값이 잘못된 파일을 가리킴
parser.add_argument('--causal-features', type=str,
    default=".../causal_features_list.json")  # 구버전 (2,787개)
parser.add_argument('--feature-means', type=str,
    default=".../feature_means_lookup.json")  # 구버전
```

**내가 만든 launch script도 잘못됨:**
```bash
# launch_phase1_REPARSED_gpu4567.sh
FEATURE_MEANS=".../feature_means_lookup_REPARSED.json"  # ❌ 284개만 있음!
```

**올바른 파일:**
```
/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json
└── 13,434개 features with safe_mean/bankrupt_mean ✅
```

#### 🚨 **문제 2: Feature Means 형식 불일치**

**현재 코드가 예상하는 형식:**
```python
# Line 131, 360-362
self.feature_means = means_data['feature_means']
patch_values = {
    'safe_mean': self.feature_means[feature_name]['safe_mean'],
    'risky_mean': self.feature_means[feature_name]['risky_mean'],  # ❌ 'risky_mean' 없음!
}
```

**실제 CONVERTED 파일 형식:**
```json
{
  "layer_results": {
    "1": {
      "significant_features": [
        {
          "feature_idx": 5489,
          "safe_mean": 0.693,
          "bankrupt_mean": 0.946,  // ← 'risky_mean'이 아님!
          "cohen_d": 1.196
        }
      ]
    }
  }
}
```

---

### Phase 2: Feature-Feature Correlation

#### SAE Feature 사용 ✅
```python
# Line 82: Phase 1의 all_features 사용
all_features = record.get('all_features', {})  # ✅ Phase 1에서 추출한 SAE activations
```

#### Token 사용 ❌
**토큰을 사용하지 않음** - Phase 2는 feature-feature 상관관계만 분석

---

### Phase 3: Causal Validation

#### SAE Feature 사용 ✅
```python
# Line 61-64: Phase 1의 all_features 사용
all_features = record.get('all_features', {})  # ✅
if feature in all_features:
    trials[trial] = all_features[feature]
```

#### Token 사용 ❌
**토큰을 사용하지 않음** - causal direction 분석만 수행

---

### Phase 4: Word-Feature Correlation

#### SAE Feature 사용 ✅
```python
# Line 60: Phase 1의 all_features 사용
all_features = record['all_features']  # ✅ SAE activations
```

#### 🚨 **문제 3: Regex 토큰화 사용**
```python
# Line 42-46: ❌ 실제 BPE 토큰이 아닌 regex 사용!
def tokenize_response(self, response: str) -> List[str]:
    response = ' '.join(response.split())
    tokens = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', response.lower())  # ❌ REGEX!
    return tokens
```

**Phase 1이 이제 `generated_token_ids`와 `generated_tokens`를 저장하지만, Phase 4는 이것을 사용하지 않음!**

---

### Phase 5: Prompt-Feature Correlation

#### SAE Feature 사용 ✅
```python
# Line 49: Phase 1의 all_features 사용
all_features = record['all_features']  # ✅
```

#### Token 사용 ❌
**토큰을 사용하지 않음** - prompt type별 feature 분석만 수행

---

## 3. 문제 요약

| Phase | SAE Feature | Token 사용 | 문제 |
|-------|-------------|-----------|------|
| Phase 1 | ✅ 올바름 | ✅ 수정됨 | 🚨 잘못된 means 파일, 형식 불일치 |
| Phase 2 | ✅ 올바름 | N/A | - |
| Phase 3 | ✅ 올바름 | N/A | - |
| Phase 4 | ✅ 올바름 | ❌ Regex 사용 | 🚨 실제 토큰 미사용 |
| Phase 5 | ✅ 올바름 | N/A | - |

---

## 4. 수정 필요 사항

### 4.1 Feature Means 파일 변환 스크립트 필요
**문제**: `L1_31_features_CONVERTED_20251111.json` 형식이 Phase 1이 예상하는 형식과 다름

**현재 형식:**
```json
{
  "layer_results": {
    "1": {
      "significant_features": [
        {"feature_idx": 5489, "safe_mean": 0.693, "bankrupt_mean": 0.946}
      ]
    }
  }
}
```

**필요한 형식:**
```json
{
  "feature_means": {
    "L1-5489": {"safe_mean": 0.693, "risky_mean": 0.946}
  }
}
```

**수정 필요:**
1. 변환 스크립트 작성
2. `bankrupt_mean` → `risky_mean` 이름 변경
3. Feature name 형식 변환: `feature_idx` → `L{layer}-{feature_idx}`

### 4.2 Phase 4 토큰 사용 수정 필요
**현재:**
```python
tokens = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', response.lower())  # ❌ Regex
```

**수정 필요:**
```python
# Phase 1 출력에서 실제 토큰 사용
generated_tokens = record.get('generated_tokens', [])  # ✅ 실제 BPE 토큰
if not generated_tokens:
    # Fallback to regex for backward compatibility
    generated_tokens = self.tokenize_response(record['response'])
```

### 4.3 Launch Script 수정 필요
```bash
# 현재 (❌):
FEATURE_MEANS=".../feature_means_lookup_REPARSED.json"  # 284개만!

# 수정 필요 (✅):
FEATURE_MEANS=".../feature_means_lookup_REPARSED_FULL.json"  # 변환된 13,434개
```

---

## 5. 실행 전 필수 작업

1. **Feature means 변환 스크립트 작성 및 실행**
   - `L1_31_features_CONVERTED_20251111.json` → `feature_means_lookup_REPARSED_FULL.json`
   - 2,510개 reparsed features만 추출
   - `bankrupt_mean` → `risky_mean` 변환

2. **Phase 4 코드 수정**
   - `generated_tokens` 필드 사용하도록 수정
   - Fallback으로 regex 유지

3. **Launch script 수정**
   - 올바른 feature means 파일 경로 지정

---

## 6. 결론

| 항목 | 상태 |
|------|------|
| 2,510개 재파싱 확인 | ✅ 확인됨 |
| SAE feature 사용 | ✅ 모든 Phase에서 올바름 |
| 토큰 저장 (Phase 1) | ✅ 수정됨 |
| 토큰 사용 (Phase 4) | 🚨 **수정 필요** (Regex → BPE) |
| Feature means 파일 | 🚨 **수정 필요** (형식 변환) |
| Launch script | 🚨 **수정 필요** (파일 경로) |

**현재 상태로는 Phase 1 실행 불가!**
Feature means 파일 형식 변환이 먼저 필요합니다.
