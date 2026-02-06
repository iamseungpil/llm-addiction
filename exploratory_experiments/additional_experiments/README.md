# Additional Experiments

ICLR 2026 논문 제출 이후 추가로 진행하는 실험들입니다.

## 실험 목록

### 1. SAE Condition Comparison (`sae_condition_comparison/`)

**목적**: 기존 SAE 분석(파산 vs 비파산)을 확장하여 **베팅 조건(Variable vs Fixed)** 간 SAE 피처 차이 분석

**배경**:
- 논문에서는 파산/비파산 간 피처 차이만 분석
- Variable 조건에서 파산율이 2배 이상 높음 (LLaMA: 2.6% → 6.8%, Gemma: 12.8% → 29.1%)
- 이 차이의 원인을 SAE 피처 수준에서 규명

**분석 내용**:
1. **Variable vs Fixed 주효과**: t-test + Cohen's d + FDR 보정
2. **4-Way Comparison**: variable-bankrupt, variable-safe, fixed-bankrupt, fixed-safe (ANOVA)
3. **Interaction Analysis**: bet_type × outcome 상호작용

**실행**:
```bash
conda activate llama_sae_env
python -m sae_condition_comparison.src.condition_comparison --model llama
```

---

## 논문에 포함된 실험들

논문에 포함된 실험들은 `paper_experiments/` 폴더에 있습니다:

| 폴더 | 논문 섹션 | 설명 |
|------|----------|------|
| `slot_machine_6models/` | Section 3a | 6모델 슬롯머신 실험 |
| `investment_choice_experiment/` | Section 3b | Investment Choice 실험 |
| `llama_sae_analysis/` | Section 4 | SAE + Activation Patching |
| `pathway_token_analysis/` | Section 5 | 시계열/언어적 토큰 분석 |

---

## 폴더 구조

```
additional_experiments/
└── sae_condition_comparison/
    ├── configs/
    │   └── analysis_config.yaml
    ├── src/
    │   ├── condition_comparison.py
    │   └── utils.py
    ├── scripts/
    │   └── run_analysis.sh
    ├── results/
    ├── logs/
    └── README.md
```
