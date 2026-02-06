# Quick Start Guide

Investment Choice SAE Analysis를 빠르게 시작하는 가이드입니다.

## 1. 환경 설정

```bash
# Conda 환경 활성화
conda activate llama_sae_env

# 프로젝트 디렉토리로 이동
cd /mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/investment_choice_sae_analysis
```

## 2. Config 확인

`configs/experiment_config.yaml` 파일을 열어서 다음을 확인:

```yaml
# 데이터 경로가 맞는지 확인
data:
  data_dir: /mnt/c/Users/oollccddss/git/data/llm-addiction/investment_choice

# 분석할 모델 선택
# Gemma 또는 LLaMA 중 선택
```

## 3. 실행

### 옵션 A: 전체 파이프라인 실행 (Phase 1 + 2)

```bash
# Gemma 모델, GPU 0 사용
bash scripts/run_full_pipeline.sh gemma 0

# LLaMA 모델, GPU 1 사용
bash scripts/run_full_pipeline.sh llama 1
```

### 옵션 B: Phase별 개별 실행

```bash
# Phase 1: Feature Extraction (GPU 필요, ~3-5시간)
bash scripts/run_phase1.sh gemma 0

# Phase 2: Correlation Analysis (CPU, ~30분)
bash scripts/run_phase2.sh gemma
```

### 옵션 C: Python으로 직접 실행

```bash
# Phase 1
python src/phase1_feature_extraction.py --model gemma --gpu 0

# Phase 2
python src/phase2_correlation_analysis.py --model gemma
```

## 4. 결과 확인

### Phase 1 결과 (NPZ files)

```bash
# 생성된 NPZ 파일 확인
ls -lh results/features/

# 예시 출력:
# layer_20_features.npz  (200 MB)
# layer_21_features.npz  (200 MB)
# ...
# layer_41_features.npz  (200 MB)
```

### Phase 2 결과 (JSON)

```bash
# 분석 결과 JSON 확인
ls -lh results/correlations/

# JSON 내용 미리보기
head -50 results/correlations/correlation_analysis_gemma_*.json
```

### 시각화

```bash
# 결과 시각화 생성
python scripts/visualize_results.py \
    --results results/correlations/correlation_analysis_gemma_20260201_*.json \
    --output_dir results/visualizations/

# 생성된 그래프 확인
ls results/visualizations/
# - significant_features_by_layer.png
# - top_features_heatmap_binary.png
# - choice_prediction_by_layer.png
```

## 5. 주요 출력 해석

### NPZ 파일 구조

```python
import numpy as np

data = np.load('results/features/layer_30_features.npz', allow_pickle=True)

print("Keys:", data.files)
# ['features', 'choices', 'game_ids', 'rounds',
#  'prompt_conditions', 'bet_types', 'models']

print("Features shape:", data['features'].shape)
# (30000, 16384)  # [n_decisions, n_sae_features]

print("Choices:", np.unique(data['choices']))
# [1 2 3 4]  # Four investment options

print("Prompt conditions:", np.unique(data['prompt_conditions']))
# ['BASE' 'G' 'M' 'GM']
```

### JSON 결과 구조

```json
{
  "layer_30": {
    "layer": 30,
    "binary": {
      "analysis_type": "binary_safe_vs_risky",
      "n_significant_features": 4532,
      "safe_features": [
        [1234, 0.85],  // [feature_id, cohens_d]
        [5678, 0.73],
        ...
      ],
      "risky_features": [
        [9012, -0.92],
        ...
      ]
    },
    "multiclass": {
      "analysis_type": "multiclass_4way",
      "n_significant_features": 3421,
      "top_features": [
        [2345, 0.12],  // [feature_id, eta_squared]
        ...
      ]
    }
  }
}
```

## 6. 트러블슈팅

### GPU 메모리 부족

```yaml
# configs/experiment_config.yaml에서 batch_size 줄이기
phase1:
  batch_size: 4  # 8 → 4로 감소
```

### 데이터 파일을 찾을 수 없음

```bash
# 데이터 경로 확인
ls /mnt/c/Users/oollccddss/git/data/llm-addiction/investment_choice/

# Config 파일에서 data_dir 경로 수정
```

### Checkpoint에서 재시작

Phase 1이 중단되었을 경우, checkpoint에서 자동으로 재시작됩니다:

```bash
# Checkpoint 확인
ls results/features/checkpoints/

# 동일한 명령으로 재실행하면 checkpoint부터 이어서 진행
bash scripts/run_phase1.sh gemma 0
```

## 7. 다음 단계

Phase 1-2 완료 후:

1. **결과 분석**: JSON 파일에서 top features 확인
2. **Phase 3**: Semantic analysis (feature interpretation)
3. **Phase 4**: Causal validation (activation patching)
4. **Cross-model 비교**: Gemma vs LLaMA 피처 비교
5. **논문 작성**: 발견된 피처를 바탕으로 결과 정리

## 8. 예상 소요 시간

| Phase | GPU | 시간 | 저장 용량 |
|-------|-----|------|----------|
| Phase 1 (Gemma) | 필수 (20GB) | 3-5시간 | 4.4 GB |
| Phase 2 (Gemma) | 불필요 | 30분 | 10 MB |
| **합계** | | **4-6시간** | **~4.5 GB** |

## 9. 도움말

추가 질문이나 문제가 있으면:

1. README.md 전체 문서 참조
2. 코드 내 docstring 확인
3. GitHub Issues에 보고

---

**Good luck with your analysis!** 🚀
