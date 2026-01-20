# Skills: LLM Addiction Research Data Reference

## Data Location

실험 데이터는 `/mnt/c/Users/oollccddss/git/data/llm-addiction/`에 저장되어 있음.

---

## 1. Slot Machine Experiment Data

각 모델별 gambling 실험 원본 데이터:

| Model | Path | Size |
|-------|------|------|
| **Gemma** | `slot_machine/gemma/final_gemma_20251004_172426.json` | ~2MB |
| **LLaMA** | `slot_machine/llama/final_llama_20251004_021106.json` | ~1.1MB |
| **Claude** | `slot_machine/claude/claude_experiment_20250920_003210.json` | ~2.7MB |
| **Gemini** | `slot_machine/gemini/gemini_experiment_20250922_003406.json` | ~9.8MB |
| **GPT** | `slot_machine/gpt/` | (archived) |

### 데이터 형식 (JSON)
```json
{
  "results": [
    {
      "game_id": 1,
      "outcome": "bankruptcy" | "voluntary_stop",
      "final_balance": 0,
      "config": {"initial_funds": 100, ...},
      "history": [
        {"round": 1, "bet": 10, "win": false, "balance_after": 90},
        ...
      ]
    },
    ...
  ]
}
```

---

## 2. Investment Choice Experiment Data

구조화된 선택 패러다임 실험:

```
investment_choice/
├── initial/           # 초기 실험
├── bet_constraint/    # 베팅 제한 조건
├── bet_constraint_cot/ # CoT 추가
└── extended_cot/      # 확장 CoT
```

각 폴더 내: `checkpoints/`, `logs/`, `results/`

---

## 3. SAE Patching Results

Sparse Autoencoder 기반 causal patching 실험:

```
sae_patching/
├── exp1_slot_machine_sae/    # 슬롯머신 SAE 분석
├── l1_31_extraction/         # L1-31 feature 추출
├── corrected_sae_analysis/   # 보정된 분석
├── multilayer_patching/      # 다중 레이어 패칭
└── patching_265_fdr/         # 265 features FDR 패칭
    └── patching_265_200_gpu*.jsonl  # 패칭 결과
```

---

## 4. Analysis Results

분석 및 비교 결과:

```
analysis/
├── experiment_1_pathway_L1_31/        # L1-31 pathway 분석
├── experiment_2_llama_standardization/ # LLaMA 표준화
├── experiment_3_gemma_addition/        # Gemma 추가
├── experiment_3_L1_31_word_analysis/   # 단어 수준 분석
├── fixed_variable_comparison/          # Fixed vs Variable bet 비교
│   ├── gpt_fixed_bet_size_results/
│   └── gpt_variable_max_bet_results/
├── gpt_results_fixed_parsing/          # GPT 파싱 수정 결과
├── pathway_token_analysis/             # 토큰별 pathway 분석
│   ├── phase1_*/  # Phase 1 결과
│   ├── phase4_*/  # Phase 4 결과
│   └── phase5_*/  # Phase 5 결과
└── steering_vector_experiment/         # Steering vector 실험
```

---

## 5. Quick Reference

### Gemma Metacognitive Experiment용 데이터 경로
```python
# config에서 사용하는 경로
experiment_data = "/mnt/c/Users/oollccddss/git/data/llm-addiction/slot_machine/gemma/final_gemma_20251004_172426.json"
```

### 주요 파일 크기
- Gemma slot machine: ~2MB (가장 정제된 데이터)
- LLaMA slot machine: ~1.1MB
- Claude slot machine: ~2.7MB
- Gemini slot machine: ~9.8MB (가장 큼)

### 데이터 로드 예시
```python
import json

DATA_ROOT = "/mnt/c/Users/oollccddss/git/data/llm-addiction"

# Gemma 데이터 로드
with open(f"{DATA_ROOT}/slot_machine/gemma/final_gemma_20251004_172426.json") as f:
    gemma_data = json.load(f)

games = gemma_data['results']
bankrupt = [g for g in games if g['outcome'] == 'bankruptcy']
safe = [g for g in games if g['outcome'] == 'voluntary_stop']
```

---

## 6. Config 업데이트

`gemma_metacognitive_experiment/configs/experiment_config.yaml`의 데이터 경로를 로컬로 수정:

```yaml
data:
  experiment_data: "/mnt/c/Users/oollccddss/git/data/llm-addiction/slot_machine/gemma/final_gemma_20251004_172426.json"
  output_dir: "/mnt/c/Users/oollccddss/git/data/llm-addiction/gemma_metacognitive_experiment"
```
