# Investment Choice SAE Analysis

SAE (Sparse Autoencoder) 기반 interpretability 분석을 investment choice experiment에 적용하여 **의사결정 메커니즘**을 신경 피처 수준에서 분석합니다.

## 목적

Investment choice experiment의 4가지 옵션 선택(안전 → 고위험)을 예측하는 SAE 피처를 발견하고, 그 인과적 역할을 검증합니다.

### 배경

Investment choice experiment는 매 라운드마다 4가지 투자 옵션을 제공합니다:

| 옵션 | 승률 | 기댓값 | 특징 |
|------|------|--------|------|
| Option 1 | 100% | $10 | 안전, **게임 종료** |
| Option 2 | 50% | $9 | 저위험, 게임 계속 |
| Option 3 | 25% | $8 | 중위험, 게임 계속 |
| Option 4 | 10% | $9 | 고위험, 게임 계속 |

**핵심 연구 질문**:
- 어떤 SAE 피처가 안전/위험 선택을 예측하는가?
- 프롬프트 조건(BASE, G, M, GM)에 따라 활성화 패턴이 달라지는가?
- 이 피처들을 조작하면 실제 선택이 바뀌는가? (인과성)

## 파이프라인 구조

```
Phase 1: Feature Extraction (GPU 필요, ~3-5시간)
  Input:  Investment choice JSON files (6,400+ games, 30,000+ decisions)
  Output: layer_{L}_features.npz [n_decisions, d_sae]

Phase 2: Correlation Analysis (CPU, ~30분)
  Input:  NPZ files from Phase 1
  Output: Significant features predicting choice (1/2/3/4)

Phase 3: Semantic Analysis (GPU, ~1시간)
  Input:  Top features from Phase 2
  Output: Feature interpretations (what they encode)

Phase 4: Causal Validation (GPU, ~2-4시간)
  Input:  Significant features
  Output: Behavioral changes from activation patching
```

## 데이터 요구사항

### Input Data
- **Location**: `/mnt/c/Users/oollccddss/git/data/llm-addiction/investment_choice/`
- **Format**: JSON files with prompts, responses, choices, outcomes
- **Structure**:
  ```json
  {
    "results": [
      {
        "game_id": 1,
        "model": "claude_haiku",
        "bet_type": "fixed",
        "prompt_condition": "BASE",
        "decisions": [
          {
            "round": 1,
            "prompt": "...",      // Full prompt text
            "response": "...",    // Full model response
            "choice": 2,          // 1, 2, 3, or 4
            "balance_before": 100,
            "balance_after": 108
          }
        ]
      }
    ]
  }
  ```

### SAE Models
- **LLaMA**: `fnlp/Llama3_1-8B-Base-LXR-8x` (layers 25-31)
- **Gemma**: `google/gemma-scope-9b-pt-res-canonical` (layers 20-41)

## 실행 방법

### 1. 환경 설정
```bash
conda activate llama_sae_env
cd /mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/investment_choice_sae_analysis
```

### 2. Config 수정
`configs/experiment_config.yaml` 파일에서 설정 확인:
- `data_dir`: Investment choice JSON 파일 경로
- `model_name`: 'gemma' 또는 'llama'
- `target_layers`: 분석할 레이어 범위

### 3. Phase 1: Feature Extraction
**가장 시간이 많이 걸리는 단계** (GPU 필요)

```bash
# Gemma 모델로 Phase 1 실행
python src/phase1_feature_extraction.py --model gemma --gpu 0

# LLaMA 모델로 Phase 1 실행
python src/phase1_feature_extraction.py --model llama --gpu 0

# 또는 shell script 사용
bash scripts/run_phase1.sh gemma 0
```

**출력**: `results/features/layer_{L}_features.npz` (22 files for Gemma)

### 4. Phase 2: Correlation Analysis
**CPU 전용, GPU 불필요**

```bash
# Gemma 분석
python src/phase2_correlation_analysis.py --model gemma

# LLaMA 분석
python src/phase2_correlation_analysis.py --model llama

# 또는 shell script 사용
bash scripts/run_phase2.sh gemma
```

**출력**: `results/correlations/significant_features_{model}.json`

### 5. 전체 파이프라인 실행 (Phase 1-2)
```bash
bash scripts/run_full_pipeline.sh gemma 0
```

## 출력 파일

### Phase 1: NPZ Files
```
results/features/
├── layer_20_features.npz
├── layer_21_features.npz
...
└── layer_41_features.npz

# Each NPZ contains:
- features: [n_decisions, d_sae]  # SAE feature activations
- choices: [n_decisions]           # 1, 2, 3, or 4
- game_ids: [n_decisions]
- rounds: [n_decisions]
- prompt_conditions: [n_decisions] # BASE, G, M, GM
- bet_types: [n_decisions]         # fixed or variable
```

### Phase 2: Correlation Results
```json
{
  "layer_25": {
    "safe_features": [
      {
        "feature_id": 1234,
        "direction": "option1_vs_others",
        "cohens_d": 0.85,
        "p_value": 1.2e-45,
        "mean_option1": 0.92,
        "mean_option234": 0.23
      }
    ],
    "risky_features": [
      {
        "feature_id": 5678,
        "direction": "option4_vs_others",
        "cohens_d": -0.73,
        "p_value": 3.4e-32
      }
    ]
  }
}
```

## 분석 종류

### 1. Binary Classification
- **Safe (Option 1) vs Risky (Options 2/3/4)**
- 기존 slot machine bankruptcy 분석과 유사
- 가장 강력한 신호 예상

### 2. Multi-class Classification
- **Option 1 vs 2 vs 3 vs 4** (4-way ANOVA)
- 위험 수준에 따른 연속적 피처 변화 분석
- 더 세밀한 feature 발견 가능

### 3. Prompt Condition Effects
- **BASE vs G vs M vs GM**
- 목표 설정(G)이 어떤 피처를 활성화시키는가?
- 최대화 프레이밍(M)의 신경 메커니즘

### 4. Betting Type Effects
- **Fixed vs Variable**
- Variable betting에서 더 활성화되는 위험 피처
- Slot machine 분석과 교차 검증

## 예상 발견

### Safe-Seeking Features (Option 1)
- EV 계산 피처 (수학적 추론)
- Loss aversion 피처
- "Maximize" 프레이밍 활성화 피처

### Risk-Seeking Features (Options 3/4)
- Goal-oriented 피처 (G/GM 조건)
- Pattern-seeking 피처
- Variable betting 활성화 피처

## 컴퓨팅 자원

### Phase 1 (Feature Extraction)
- **GPU**: 1x ~20-24GB VRAM (Gemma-2-9B 또는 LLaMA-3.1-8B)
- **RAM**: 32GB+
- **시간**: 3-5시간 (30,000 decisions × 22 layers)
- **저장 용량**: ~4-5GB (22 layers × 200MB/layer)

### Phase 2-4 (Analysis)
- **GPU**: Phase 3, 4에서만 필요
- **RAM**: 8-16GB
- **시간**: 1-2시간 (통계 분석)

## 의존성

```yaml
# Core dependencies (from llama_sae_env)
- torch >= 2.0
- transformers >= 4.40
- sae_lens >= 6.5.1
- numpy
- scipy
- scikit-learn
- tqdm
- pyyaml
```

## Investment Choice vs Slot Machine 차이점

| 측면 | Slot Machine | Investment Choice |
|------|--------------|-------------------|
| **분석 단위** | Game-level (마지막 결정) | **Decision-level** (모든 라운드) |
| **데이터 포인트** | 3,200-6,400 games | **30,000+ decisions** |
| **Outcome 타입** | Binary (bankruptcy/stop) | **Multi-class (choice 1/2/3/4)** |
| **Prompt 저장** | 재구성 필요 | ✅ **이미 저장됨** |
| **시간적 변화** | 불가능 | ✅ **라운드별 추적 가능** |
| **분석 난이도** | 단순 | 복잡 (더 풍부한 정보) |

**Investment choice의 장점**:
- 10배 더 많은 데이터 포인트
- 4-way classification → 더 세밀한 신호 탐지
- 라운드별 dynamics 분석 가능 (시간에 따른 피처 변화)
- Prompt 조건 효과를 직접 비교 가능

## 참고 코드

이 파이프라인은 다음 기존 코드를 참고하여 새로 설계되었습니다:

- `paper_experiments/llama_sae_analysis/`: Slot machine SAE 분석
- `gemma_sae_experiment/`: Gemma 전용 파이프라인
- `lr_classification_experiment/src/hidden_state_extractor.py`: Hidden state 추출
- `additional_experiments/sae_condition_comparison/`: 조건별 비교 분석

## 향후 확장

- **Phase 3**: Semantic analysis (top features 해석)
- **Phase 4**: Causal validation (activation patching)
- **Phase 5**: Cross-experiment validation (slot machine vs investment choice)
- **Cross-model comparison**: Gemma vs LLaMA feature overlap
- **Temporal analysis**: 라운드별 피처 활성화 변화

## 주의사항

1. **GPU 메모리**: Gemma-2-9B는 ~22GB 필요, batch processing 사용
2. **Checkpoint**: Phase 1은 100 decisions마다 checkpoint 저장
3. **Resume 가능**: Checkpoint에서 재시작 가능
4. **FDR correction**: 다중 비교 보정 필수 (16,384 features × 22 layers)

## 라이선스

ICLR 2026 submission의 일부로, 기존 llm-addiction 프로젝트와 동일한 라이선스를 따릅니다.
