# NMT 논문 Element → Data/Code Manifest

본 논문의 현재 빌드 엔트리(`nmi.tex`, `neurips.tex`)와 연결된 모든 표·그림·수치가 어떤 원천 데이터와 스크립트에서 생성되었는지를 정리한다.
외부 독자가 각 결과를 재현하려 할 때 이 표 하나로 접근 경로를 찾을 수 있게 하는 것이 목적이다.

빠른 경로 안내는 [DATA_INDEX.md](DATA_INDEX.md)에 따로 정리했다. 그 문서는 현재 논문 기준 canonical behavioral/neural source만 추려서 바로 찾게 하는 용도이고, 본 manifest는 element-level provenance를 남기는 용도다.

## 1. 논문 메타

| 파일 | 위치 |
|---|---|
| 공용 코어 | repo `shared/paper_core.tex` |
| NeurIPS wrapper | repo `neurips.tex` |
| NMI wrapper | repo `nmi.tex` (현재 트리에는 없음 — superseded) |
| 섹션 원고 | `content/0.abstract.tex` ~ `content/5.methods.tex`, `content/appendix_sae.tex` |
| NeurIPS 섹션 원고 | `neurips_content/*.tex` |
| 그림 | `images/*.pdf` |
| 빌드 | `latexmk -xelatex nmi.tex` 또는 `latexmk -xelatex neurips.tex` |

## 2. 행동 실험 원천 데이터

| 패러다임 / 모델 | 경로 | 게임 수 | BK 수 |
|---|---|---|---|
| SM Gemma | HF `behavioral/slot_machine/gemma_v4_role/` | 3,200 | 87 |
| SM LLaMA | HF `behavioral/slot_machine/llama_v4_role/` | 3,200 | 1,164 |
| IC Gemma | HF `behavioral/investment_choice/v2_role_gemma/` | 1,600 | 172 |
| IC LLaMA | HF `behavioral/investment_choice/v2_role_llama/` | 1,600 | 142 |
| MW Gemma | HF `behavioral/mystery_wheel/gemma_v2_role/` | 3,200 | 54 |
| MW LLaMA | HF `behavioral/mystery_wheel/llama_v2_role/` | 3,200 | 2,426 |
| **합계 (오픈웨이트)** | — | **16,000** | **4,045** |

각 디렉토리는 **1 canonical 버전**만 보유 (v4 for SM, v2 for IC/MW). 대체 프롬프트로 생성된 탐색적 데이터는 별도 아카이브되어 본 연구에 사용되지 않았다.

API 모델 투자 선택 결과와 오픈웨이트 투자 선택 결과를 합친 6모델 행동 집계는 HF dataset `llm-addiction-research/llm-addiction`과 로컬 canonical v2/v4 원본을 함께 사용한다. 투자 선택은 중복 재실행 파일을 제거한 뒤 모델당 1,600게임, 총 9,600게임으로 고정한다.

## 3. 신경 분석 원천 데이터

| 자료 | 경로 |
|---|---|
| 결정 시점 은닉 상태 (RQ2 shared-subspace) | HF `sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/{gemma,llama}/hidden_states_dp.npz` — 6 파일 |
| 라운드별 SAE 희소 특성 (RQ1, RQ3) | HF dataset (GemmaScope 131K, LlamaScope 32K 경유 생성) |
| Paper-facing 집계 | HF `sae_v3_analysis/results/paper_neural_audit.json` |
| Shared-subspace 감사 | HF `sae_v3_analysis/results/shared_subspace_hidden_audit_20260410.json` |

## 4. 논문 Element 추적표

### Abstract 주장

| 주장 | 원천 |
|---|---|
| 28,800 게임, 파산율 0–13%→6–48% | Table 5 sample-sizes, Figure 3a |
| 6 조합 I_LC $R^2$ = 0.24–0.78 | Table 1 (`tab:sae-results`) ← `paper_neural_audit.json.rq1_ilc` |
| 4 조합 I_BA $R^2$ = 0.06–0.16 | Table 1 ← `paper_neural_audit.json.rq1_direct` |
| rank 1 공통 축 | Section 3 RQ2 ← `shared_subspace_hidden_audit_20260410.json` |

### Section 3 (Results) 표/그림

| 번호 | 논문 위치 | 원천 |
|---|---|---|
| Fig 1 (representative_flow_diagram.pdf) | Intro | `images/representative_flow_diagram.pdf` |
| Fig 2 (slot_machine_analysis2.pdf) | Section 3.1.1 | (a) 6모델 BK는 canonical raw data 재집계, (b) I 지표도 6모델 round-level raw 재집계(GPT-4o-mini corrected parsing archive 포함) → `generate_paper_figures.py` / `regenerate_fig2_fig4.py` |
| Fig 3 (streak_analysis_1x2_comparison.pdf) | Section 3.1.2 | SM 19,200 games → streak 분석 script |
| Fig 4 (investment_choice2.pdf) | Section 3.1.3 | IC 9,600 games (6 models × 1,600) |
| Fig 5 (neural_analysis_combined.pdf) | Section 3.2 RQ1 | `paper_neural_audit.json` + layer sweep |
| Fig 6 (cross_paradigm_transfer.pdf) | Section 3.2 RQ2 | `paper_neural_audit.json` cross-transfer entries |
| Fig 7 (condition_modulation_iba.pdf) | Section 3.2 RQ3 | `paper_neural_audit.json.rq3_condition_i_ba` |
| Table 1 (tab:sae-results) | RQ1 | `paper_neural_audit.json.rq1_ilc` + `rq1_direct` |
| Table 2 (tab:selectivity-controls) | RQ1 | paper_neural_audit (selectivity entries) |
| Table 3 (tab:behavior-convergence) | RQ2 | Behavioral aggregation across 6 조합 |
| Table 4 (tab:condition-modulation) | RQ3 | `paper_neural_audit.json.rq3_condition_i_ba` |

### Section 5 (Methods) Tables

| Table | 원천 |
|---|---|
| tab:slot-machine-conditions | 실험 설계 상수 |
| tab:option-variance | 투자 선택 EV 계산 |
| tab:investment-choice-conditions | 실험 설계 상수 |
| tab:models | 상수 (모델 리스트) |
| tab:sample-sizes | 집계 상수 |

### Appendix 증거

| Table/Fig | 원천 |
|---|---|
| tab:appendix-slot-comprehensive | 6 모델 SM behavioral 집계 |
| tab:appendix-investment-comprehensive | 6 모델 IC behavioral 집계 |
| tab:hidden-subspace-audit | `shared_subspace_hidden_audit_20260410.json` |
| fig:appendix-complexity ~ fig:appendix-choice-distribution | behavioral aggregation scripts |
| fig:escalation | LLaMA SM 3,041 games trajectory |
| fig:temperature-robustness | LLaMA SM 1,600 games, temperature sweep |

## 5. 생성 스크립트 경로

모든 스크립트는 분리된 분석 트리의 `sae_v3_analysis/src/` 아래에 위치 (`$LLM_ADDICTION_ANALYSIS` 기준; 이 저장소에도 HF에도 없음).

| 용도 | 스크립트 | 출력 |
|---|---|---|
| Paper manifest 빌드 | `src/build_paper_neural_audit.py` | `results/paper_neural_audit.json` |
| 라운드별 I_BA 계산 | `src/run_comprehensive_robustness.py` | 라운드 레벨 집계 |
| I_LC 라벨 구축 | `src/run_perm_null_ilc.py` | 라벨 + permutation 결과 |
| 선택성 CV 검증 | `src/run_probe_selectivity_controls.py` | `tab:selectivity-controls` 값 |
| Hidden-state subspace 감사 | `src/build_paper_neural_audit.py` (hidden section) | `shared_subspace_hidden_audit_20260410.json` |
| 신경 figure 생성 | `src/plot_neural_figures.py` | `images/*.pdf` (일부) |

## 6. 접근 및 재현

1. **HF dataset** `llm-addiction-research/llm-addiction`: 행동 원자료, 파싱된 결정 테이블, 신경 분석 집계, 체크포인트 전부 보존.
2. **GitHub** `iamseungpil/llm-addiction`: 분석 스크립트, 논문 원고, figure 생성 코드.
3. 재현 순서: behavioral 원자료 로드 → `build_paper_neural_audit.py` → `plot_neural_figures.py` → `latexmk -xelatex nmi.tex` 또는 `latexmk -xelatex neurips.tex`.

## 7. 데이터 일관성 검증 (2026-04-16)

본 manifest는 다음 검증을 통과한 데이터에 대해서만 논문 element를 연결한다:

- ✅ Behavioral Table 1 (28,800 games)의 모든 셀이 실측 JSON과 일치
- ✅ Table 3 (behavior-convergence) I_BA / I_EC가 실측과 소수점 3자리 일치
- ✅ RQ1 Table 1 (SAE readout $R^2$)이 `paper_neural_audit.json`과 소수점 3자리 일치
- ✅ RQ3 Table 4 (condition modulation)의 12개 셀 전부 audit와 일치
- ✅ Hidden-state 감사 결과가 `shared_subspace_hidden_audit_20260410.json`과 일치

각 경로에는 **1 canonical 버전**만 존재하며, 대체 프롬프트나 이전 분석 단계에서 생성된 탐색 산출물은 본 논문 집계에서 제외한다.
