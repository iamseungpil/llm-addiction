# SAE Feature Activation Analysis

NPZ feature activation 분석 코드 모음 (neuron_sae 브랜치에서 가져옴).

## 구조

```
sae_feature_analysis/
├── slot_machine_condition_comparison/   # Slot Machine: Variable vs Fixed 조건 비교
│   ├── src/                             # 분석 코드 (condition_comparison, ANOVA, utils)
│   ├── scripts/                         # 시각화 & 실행 스크립트
│   └── configs/                         # YAML 설정 파일
│
├── investment_choice_sae/               # Investment Choice: SAE feature extraction + correlation
│   ├── src/                             # Phase1 extraction, Phase2 correlation, prompt utils
│   ├── scripts/                         # 파이프라인 실행 & 시각화
│   └── configs/                         # experiment_config.yaml
│
└── common_paradigm_sae/                 # 공통 모듈: 다양한 paradigm용 SAE 추출/분석
    ├── phase1_feature_extraction.py     # Lootbox/Blackjack 등 SAE feature 추출
    └── phase2_correlation_analysis.py   # Feature-behavior correlation 분석
```

## 모델별 SAE 리소스

| Model | SAE Repository | Layers | Features/Layer |
|-------|---------------|--------|----------------|
| LLaMA-3.1-8B | `fnlp/Llama3_1-8B-Base-LXR-8x` | 25-31 | 131K |
| Gemma-2-9B | `google/gemma-scope` | All 42 | 131K |

## 출처

`neuron_sae` 브랜치의 다음 디렉토리에서 가져옴:
- `exploratory_experiments/additional_experiments/sae_condition_comparison/`
- `exploratory_experiments/additional_experiments/investment_choice_sae_analysis/`
- `exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py`
- `exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py`
