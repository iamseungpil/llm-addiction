# 최종 수정된 4개 실험 계획 (승인 대기)

## 이해한 요구사항 정리

### 1. **실험 0: LLaMA/Gemma 재시작**
- ✅ 3,200 games each
- ✅ 무한 retry (빈 응답 방지)
- ✅ GPU 0 (LLaMA), GPU 1 (Gemma)

### 2. **실험 1: Layer Pathway Tracking**
- ✅ L1-31 전체 추적
- ✅ 50 games
- ✅ GPU 3

### 3. **실험 3: Feature-Word Analysis** ⚠️ **수정 필요**

**기존 계획**:
- 441개 causal features만 분석
- Exp2 response logs만 사용

**수정된 계획** (사용자 요구):
1. **데이터 소스 확대**:
   - ✅ 441개 causal feature patching 응답 (Exp5 multiround patching 응답)
   - ✅ 전체 6,400개 Exp1 응답 (파산/비파산 그룹)

2. **Feature 분석 방식**:
   - `/home/ubuntu/llm_addiction/causal_feature_discovery/src/extract_statistically_valid_features_all_layers.py` 방식 사용
   - 즉, **L25-31 layers에서 추출된 features**를 대상으로
   - 각 feature의 activation 값에 따라 응답을 그룹화
   - 파산 그룹 vs 비파산 그룹의 응답 패턴 비교

3. **분석 목표**:
   - Feature activation이 높을 때 vs 낮을 때의 단어 사용 차이
   - 파산 그룹과 비파산 그룹에서 해당 feature의 영향

### 4. **실험 2: Multilayer Patching** ⚠️ **수정 필요**

**기존 계획**:
- L1-31 전체 layer
- 각 layer당 top 300개 (총 9,300 features)
- 새로운 코드 작성 필요

**수정된 계획** (사용자 요구):
1. **코드 기반**:
   - ✅ `/home/ubuntu/llm_addiction/causal_feature_discovery/src/experiment_2_multilayer_population_mean.py` 방식 사용
   - Population mean patching (safe_mean, baseline, bankrupt_mean)
   - 3-condition design

2. **Feature 선택**:
   - ✅ L1-31까지 확장
   - ✅ 각 layer당 상위 300개 features (by |Cohen's d|)
   - ✅ 총 9,300 features

3. **데이터 소스**:
   - `/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json`
   - 각 layer의 bankrupt_mean, safe_mean 사용

4. **GPU 분산**:
   - ✅ GPU 2,4,6,7 병렬 실행
   - GPU 2: L1-8 (2,400 features)
   - GPU 4: L9-16 (2,400 features)
   - GPU 6: L17-24 (2,400 features)
   - GPU 7: L25-31 (2,100 features)

---

## 수정된 실험 상세 계획

---

## 실험 0: LLaMA/Gemma 재시작 (변경 없음)

### GPU 할당
- **GPU 0**: LLaMA (20GB, 24시간)
- **GPU 1**: Gemma (22GB, 24시간)

### 파일 경로
- LLaMA 코드: `/home/ubuntu/llm_addiction/experiment_0_standardization/llama_3200_infinite_retry.py`
- Gemma 코드: `/home/ubuntu/llm_addiction/experiment_0_standardization/gemma_3200_no_deepspeed.py`

### 실행 명령
```bash
# LLaMA
cd /home/ubuntu/llm_addiction/experiment_0_standardization
CUDA_VISIBLE_DEVICES=0 nohup conda run -n llama_sae_env python llama_3200_infinite_retry.py > llama_restart.log 2>&1 &

# Gemma
CUDA_VISIBLE_DEVICES=1 nohup conda run -n llama_sae_env python gemma_3200_no_deepspeed.py > gemma_restart.log 2>&1 &
```

---

## 실험 1: Layer Pathway Tracking (변경 없음)

### GPU 할당
- **GPU 3**: Pathway (30GB, 4시간)

### 파일 경로
- 코드: `/home/ubuntu/llm_addiction/experiment_1_layer_pathway/layer_pathway_tracking.py`

### 실행 명령
```bash
cd /home/ubuntu/llm_addiction/experiment_1_layer_pathway
CUDA_VISIBLE_DEVICES=3 nohup conda run -n llama_sae_env python layer_pathway_tracking.py > pathway.log 2>&1 &
```

---

## 실험 3: Feature-Word Analysis (수정됨 ⚠️)

### GPU 할당
- **GPU 3**: Exp1과 **병렬 실행** (즉시 시작)
- 총 메모리: 30GB (Exp1) + 26GB (Exp3) = 56GB < 81GB ✅

### 수정된 분석 방식

#### 데이터 소스
1. **Exp5 Multi-round Patching 응답** (441 features)
   - 파일: `/data/llm_addiction/results/exp5_multiround_*.json`
   - Feature patching 시 생성된 응답들

2. **Exp1 전체 6,400개 응답** (파산/비파산 그룹)
   - 파일:
     - `/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json`
     - `/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json`
   - 각 experiment의 `round_features`에서 마지막 round 응답 추출
   - 파산 그룹 vs 비파산 그룹 분리

#### 분석 방법 (extract_statistically_valid_features_all_layers.py 방식)

```python
# 1. Feature 추출 방식
for experiment in all_6400_experiments:
    last_round = experiment['round_features'][-1]
    prompt = last_round['prompt']
    response = last_round['response']

    # SAE로 feature activation 추출
    layer_features = extract_features(prompt)  # L25-31

    # 그룹 분류
    if experiment['is_bankrupt']:
        bankrupt_group.append({
            'features': layer_features,
            'response': response
        })
    else:
        safe_group.append({
            'features': layer_features,
            'response': response
        })

# 2. 441개 causal features에 대해 분석
for feature in causal_441_features:
    layer = feature['layer']
    feature_id = feature['feature_id']

    # 해당 feature activation으로 그룹 분할
    high_activation_responses = []
    low_activation_responses = []

    for sample in all_samples:
        activation = sample['features'][layer][feature_id]

        if activation > median:
            high_activation_responses.append(sample['response'])
        else:
            low_activation_responses.append(sample['response'])

    # 단어 빈도 분석
    high_words = extract_words(high_activation_responses)
    low_words = extract_words(low_activation_responses)

    # 차이 분석
    differentiating_words = find_significant_differences(high_words, low_words)
```

#### 분석 출력
- Feature activation이 높을 때 자주 나오는 단어들
- Feature activation이 낮을 때 자주 나오는 단어들
- 파산 그룹 vs 비파산 그룹에서의 단어 차이
- Feature별 semantic interpretation

### 파일 경로
- 코드: `/home/ubuntu/llm_addiction/experiment_3_feature_word_corrected/feature_word_analysis_corrected.py` (새로 작성)
- 로그: `/home/ubuntu/llm_addiction/experiment_3_feature_word_corrected/analysis.log`
- 결과: `/data/llm_addiction/experiment_3_feature_word_corrected/feature_word_associations.json`

### 실행 명령
```bash
cd /home/ubuntu/llm_addiction/experiment_3_feature_word_corrected
CUDA_VISIBLE_DEVICES=3 nohup conda run -n llama_sae_env python feature_word_analysis_corrected.py > analysis.log 2>&1 &
```

---

## 실험 2: Multilayer Patching (수정됨 ⚠️)

### GPU 할당 (즉시 병렬 실행)
- **GPU 2**: L1-8 (2,400 features, 8.1일)
- **GPU 4**: L9-16 (2,400 features, 8.1일)
- **GPU 6**: L17-24 (2,400 features, 8.1일)
- **GPU 7**: L25-31 (2,100 features, 8.1일)

### 코드 기반
- **참고 코드**: `/home/ubuntu/llm_addiction/causal_feature_discovery/src/experiment_2_multilayer_population_mean.py`
- **방법**: Population mean patching
- **조건**: 3-condition (patch_to_safe_mean, baseline, patch_to_bankrupt_mean)

### Feature 데이터 소스
- **파일**: `/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json`
- **구조**:
```json
{
  "layer_results": {
    "1": {
      "significant_features": [
        {
          "feature_idx": 123,
          "cohen_d": 0.567,
          "p_value": 0.001,
          "bankrupt_mean": 1.23,
          "safe_mean": 0.45,
          "bankrupt_std": 0.12,
          "safe_std": 0.08
        },
        ...
      ]
    },
    ...
  }
}
```

### Feature 선택 로직
```python
def load_top300_per_layer():
    """Load top 300 features per layer (L1-31)"""
    data = json.load('L1_31_features_FINAL_20250930_220003.json')

    all_features = []

    for layer in range(1, 32):
        layer_key = str(layer)
        features = data['layer_results'][layer_key]['significant_features']

        # Sort by |Cohen's d| descending
        features.sort(key=lambda x: abs(x['cohen_d']), reverse=True)

        # Take top 300
        top_300 = features[:300]

        for feat in top_300:
            all_features.append({
                'layer': layer,
                'feature_id': feat['feature_idx'],
                'cohen_d': feat['cohen_d'],
                'p_value': feat['p_value'],
                'bankrupt_mean': feat['bankrupt_mean'],
                'safe_mean': feat['safe_mean'],
                'bankrupt_std': feat['bankrupt_std'],
                'safe_std': feat['safe_std']
            })

    return all_features  # Total: 9,300 features
```

### Patching 방법 (experiment_2_multilayer_population_mean.py 방식)

```python
def test_single_feature(feature):
    """Test one feature with 3-condition population mean patching"""
    layer = feature['layer']
    feature_id = feature['feature_id']
    safe_mean = feature['safe_mean']
    bankrupt_mean = feature['bankrupt_mean']

    results = []

    # Condition 1: Patch to safe_mean
    for trial in range(50):
        response = generate_with_patching(
            prompt=risky_prompt,
            layer=layer,
            feature_id=feature_id,
            patched_value=safe_mean
        )
        bet = parse_bet(response)
        results.append({
            'condition': 'patch_to_safe_mean',
            'trial': trial,
            'bet': bet,
            'response': response
        })

    # Condition 2: Baseline (no patching)
    for trial in range(50):
        response = generate_with_patching(
            prompt=risky_prompt,
            layer=layer,
            feature_id=feature_id,
            patched_value=None  # No patching
        )
        bet = parse_bet(response)
        results.append({
            'condition': 'baseline',
            'trial': trial,
            'bet': bet,
            'response': response
        })

    # Condition 3: Patch to bankrupt_mean
    for trial in range(50):
        response = generate_with_patching(
            prompt=risky_prompt,
            layer=layer,
            feature_id=feature_id,
            patched_value=bankrupt_mean
        )
        bet = parse_bet(response)
        results.append({
            'condition': 'patch_to_bankrupt_mean',
            'trial': trial,
            'bet': bet,
            'response': response
        })

    # Statistical analysis
    safe_bets = [r['bet'] for r in results if r['condition'] == 'patch_to_safe_mean']
    baseline_bets = [r['bet'] for r in results if r['condition'] == 'baseline']
    bankrupt_bets = [r['bet'] for r in results if r['condition'] == 'patch_to_bankrupt_mean']

    t_safe, p_safe = ttest_ind(safe_bets, baseline_bets)
    t_bankrupt, p_bankrupt = ttest_ind(bankrupt_bets, baseline_bets)

    return {
        'feature': feature,
        'results': results,
        'statistics': {
            'safe_mean': np.mean(safe_bets),
            'baseline_mean': np.mean(baseline_bets),
            'bankrupt_mean': np.mean(bankrupt_bets),
            'safe_effect': np.mean(safe_bets) - np.mean(baseline_bets),
            'bankrupt_effect': np.mean(bankrupt_bets) - np.mean(baseline_bets),
            'p_safe': p_safe,
            'p_bankrupt': p_bankrupt
        }
    }
```

### GPU 분산 실행

각 GPU가 담당하는 layer 범위의 top 300 features만 처리:

```python
# GPU 2: L1-8
python multilayer_patching_corrected.py --gpu_id 2 --layer_start 1 --layer_end 8

# GPU 4: L9-16
python multilayer_patching_corrected.py --gpu_id 4 --layer_start 9 --layer_end 16

# GPU 6: L17-24
python multilayer_patching_corrected.py --gpu_id 6 --layer_start 17 --layer_end 24

# GPU 7: L25-31
python multilayer_patching_corrected.py --gpu_id 7 --layer_start 25 --layer_end 31
```

### 파일 경로
- 코드: `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_patching_corrected.py` (새로 작성)
- 로그:
  - `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_gpu2.log`
  - `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_gpu4.log`
  - `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_gpu6.log`
  - `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_gpu7.log`
- 결과:
  - `/data/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_final_gpu2.json`
  - `/data/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_final_gpu4.json`
  - `/data/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_final_gpu6.json`
  - `/data/llm_addiction/experiment_2_multilayer_patching_corrected/multilayer_final_gpu7.json`

### 실행 명령
```bash
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_corrected

# GPU 2: L1-8
CUDA_VISIBLE_DEVICES=2 nohup conda run -n llama_sae_env python multilayer_patching_corrected.py --gpu_id 2 --layer_start 1 --layer_end 8 > multilayer_gpu2.log 2>&1 &

# GPU 4: L9-16
CUDA_VISIBLE_DEVICES=4 nohup conda run -n llama_sae_env python multilayer_patching_corrected.py --gpu_id 4 --layer_start 9 --layer_end 16 > multilayer_gpu4.log 2>&1 &

# GPU 6: L17-24
CUDA_VISIBLE_DEVICES=6 nohup conda run -n llama_sae_env python multilayer_patching_corrected.py --gpu_id 6 --layer_start 17 --layer_end 24 > multilayer_gpu6.log 2>&1 &

# GPU 7: L25-31
CUDA_VISIBLE_DEVICES=7 nohup conda run -n llama_sae_env python multilayer_patching_corrected.py --gpu_id 7 --layer_start 25 --layer_end 31 > multilayer_gpu7.log 2>&1 &
```

---

## 실험 타임라인

### 즉시 실행 (Phase 1)

| 실험 | GPU | 시간 | 병렬 |
|------|-----|------|------|
| Exp0-LLaMA | GPU 0 | 24시간 | ✅ |
| Exp0-Gemma | GPU 1 | 24시간 | ✅ |
| Exp1-Pathway | GPU 3 | 4시간 | ✅ |
| **Exp3-Feature-Word** | **GPU 3** | **3.5시간** | **✅ (Exp1과 병렬)** |
| Exp2-L1-8 | GPU 2 | 8.1일 | ✅ |
| Exp2-L9-16 | GPU 4 | 8.1일 | ✅ |
| Exp2-L17-24 | GPU 6 | 8.1일 | ✅ |
| Exp2-L25-31 | GPU 7 | 8.1일 | ✅ |

### 완료 예상
- **Exp0, Exp1, Exp3**: ~1일 (24시간)
- **Exp2**: 8.1일
- **전체**: ~8.1일

---

## GPU 메모리 최종 확인

| GPU | 실험 | 메모리 사용 | 총 메모리 | 여유 | 상태 |
|-----|------|------------|----------|------|------|
| GPU 0 | LLaMA | 20 GB | 81 GB | 61 GB | ✅ 안전 |
| GPU 1 | Gemma | 22 GB | 81 GB | 59 GB | ✅ 안전 |
| GPU 2 | Exp2 L1-8 | 46 GB (21+25) | 81 GB | 35 GB | ✅ 안전 |
| GPU 3 | Exp1 + Exp3 | 56 GB (30+26) | 81 GB | 25 GB | ✅ 안전 |
| GPU 4 | Exp2 L9-16 + Exp5 | 49 GB (24+25) | 81 GB | 32 GB | ✅ 안전 |
| GPU 6 | Exp2 L17-24 | 65 GB (40+25) | 81 GB | 16 GB | ⚠️ 경계 |
| GPU 7 | Exp2 L25-31 | 71 GB (46+25) | 81 GB | 10 GB | ⚠️ 경계 |

---

## 수정 사항 요약

### ✅ 확인된 사항
1. Exp0 LLaMA/Gemma: 무한 retry ✅
2. Exp1 Pathway: L1-31 추적 ✅
3. Exp2 GPU 배치: GPU 2,4,6,7 병렬 ✅
4. Exp3 병렬: GPU 3에서 Exp1과 동시 실행 ✅

### ⚠️ 수정 필요
1. **Exp3 Feature-Word**:
   - ❌ 기존: 441 features, Exp2 response logs만
   - ✅ 수정: 441 features, **Exp5 응답 + 전체 6,400 Exp1 응답** 분석
   - ✅ 방식: `extract_statistically_valid_features_all_layers.py` 참고
   - ✅ Feature activation에 따른 응답 그룹화 및 단어 분석

2. **Exp2 Multilayer Patching**:
   - ❌ 기존: 새 코드 작성 필요
   - ✅ 수정: `experiment_2_multilayer_population_mean.py` 방식 사용
   - ✅ L1-31 각 layer 상위 300개 (총 9,300 features)
   - ✅ Population mean patching (safe_mean, baseline, bankrupt_mean)
   - ✅ 데이터: `L1_31_features_FINAL_20250930_220003.json`

---

## 승인 질문

1. ✅ Exp3 수정 사항 (6,400 Exp1 응답 + Exp5 응답 분석) 이해 맞나요?
2. ✅ Exp2 방식 (`experiment_2_multilayer_population_mean.py` 기반) 이해 맞나요?
3. ✅ GPU 3에서 Exp1 + Exp3 병렬 실행 괜찮으신가요?
4. ✅ GPU 2,4,6,7에서 Exp2 병렬 실행 (m-soar 공유) 괜찮으신가요?

승인하시면 모든 코드를 작성하고 실험을 시작하겠습니다.

---

*최종 수정: 2025-10-01 15:30*
