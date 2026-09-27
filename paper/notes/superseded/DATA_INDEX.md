# Paper Data Index

현재 논문에서 실제로 쓰는 canonical 데이터만 빠르게 찾기 위한 인덱스다.  
원칙은 단순하다. **행동은 6모델 전체를, 신경은 2개 오픈웨이트 모델을 canonical source 하나씩으로 고정**한다.

## 1. 행동 데이터

### Slot machine

6모델 모두를 사용한다.

| 모델 | canonical source | 비고 |
|---|---|---|
| GPT-4o-mini | HF `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json` (당시 snapshot `5b5ce148`) | corrected parsing archive. round outcome은 `game_history`를 함께 읽어야 함 |
| GPT-4.1-mini | `/tmp/llmadd_hf/slot_machine/gpt/gpt5_experiment_20250921_174509.json` | legacy filename이지만 실제 모델은 GPT-4.1-mini |
| Gemini-2.5-Flash | `/tmp/llmadd_hf/slot_machine/gemini/gemini_experiment_20250920_042809.json` | HF raw |
| Claude-3.5-Haiku | `/tmp/llmadd_hf/slot_machine/claude/claude_experiment_corrected_20250925.json` | HF corrected raw |
| LLaMA-3.1-8B | HF `behavioral/slot_machine/llama_v4_role/` | local canonical v4 |
| Gemma-2-9B | HF `behavioral/slot_machine/gemma_v4_role/` | local canonical v4 |

논문 기준 총 게임 수는 `19,200`이다.

현재 논문에서 사용하는 6모델 slot aggregate:

| bet type | I_BA | I_LC | I_EC |
|---|---:|---:|---:|
| fixed | 0.104 | 0.111 | 0.001 |
| variable | 0.294 | 0.672 | 0.196 |

### Investment choice

6모델 모두를 사용한다. API 4개는 HF raw, 오픈웨이트 2개는 local canonical을 사용한다.

| 모델군 | canonical source |
|---|---|
| API 4개 | `/tmp/llmadd_hf/investment_choice/bet_constraint/results/*.json` |
| LLaMA-3.1-8B | HF `behavioral/investment_choice/v2_role_llama/` |
| Gemma-2-9B | HF `behavioral/investment_choice/v2_role_gemma/` |

논문 기준 각 모델은 `1,600`게임, 전체는 `9,600`게임이다.  
중복 재실행 파일은 최신 canonical 파일만 남기고 집계한다.

현재 논문에서 사용하는 prompt-level aggregate:

| prompt | bankruptcy | goal escalation | high-variance choice |
|---|---:|---:|---:|
| BASE | 18.8 | 17.0 | 25.7 |
| G | 35.8 | 49.8 | 38.0 |
| M | 18.6 | 11.0 | 32.1 |
| GM | 35.7 | 47.8 | 42.2 |

## 2. 신경 데이터

신경 분석은 Gemma-2-9B와 LLaMA-3.1-8B 두 모델만 사용한다.

| 용도 | canonical source |
|---|---|
| RQ1 / RQ3 paper-facing audit | HF `sae_v3_analysis/results/paper_neural_audit.json` |
| RQ2 shared-subspace audit | HF `sae_v3_analysis/results/shared_subspace_hidden_audit_20260410.json` |
| hidden states | HF `sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/{gemma,llama}/hidden_states_dp.npz` |

논문 기준 신경 행동 표본은 `16,000`게임이다.

## 3. Figure regeneration

| figure | script |
|---|---|
| slot_machine_analysis2.pdf | repo `regenerate_fig2_fig4.py` |
| investment_choice2.pdf | repo `generate_paper_figures.py` |
| streak_analysis_1x2_comparison.pdf | repo `generate_paper_figures.py` |
| neural figures | 분리된 분석 트리 `sae_v3_analysis/src/plot_neural_figures.py` (`$LLM_ADDICTION_ANALYSIS` 기준; 이 저장소에도 HF에도 없음) |

## 4. Current paper outputs

| version | file |
|---|---|
| NMI | repo `nmi.pdf` (현재 트리에는 없음 — superseded) |
| NeurIPS | repo `neurips.pdf` |

## 5. Consistency note

이전 NMI 원고와 비교할 때, **slot bankruptcy와 open-weight neural 수치는 유지**되고, **slot round-level aggregate만 6모델 corrected source 기준으로 갱신**되었다.  
차이가 나는 이유는 GPT-4o-mini slot raw를 이제 `gpt_fixed_parsing_complete_20250919_151240.json`으로 복원했기 때문이다. 이전의 4모델/5모델 평균은 현재 논문 기준 canonical 집계가 아니다.
