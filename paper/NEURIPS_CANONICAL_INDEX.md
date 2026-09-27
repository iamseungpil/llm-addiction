# NeurIPS Canonical Index — "Can LLMs Develop Gambling Addiction?"

NeurIPS 2026 제출본(`neurips.tex` / `neurips_en.tex`, content dir = `neurips_content/`·`neurips_content_en/`)을
**유일한 기준**으로 삼아, 본문 표·그림 각각이 어떤 데이터 파일과 스크립트에서 나오는지 1:1로 고정한다.

이 문서가 기존 `DATA_INDEX.md` / `PAPER_ELEMENT_MANIFEST.md`보다 우선한다(그 둘은 NMI/leaky 시절 참조가 섞여 있음).

## 0. Canonical 원칙 (NeurIPS)

| 항목 | 기준 |
|---|---|
| 신경 분석 파이프라인 | **strict-CV GroupKFold (game-id 단위)** 만 사용 |
| 대표 레이어 | **고정 L22** (peak-layer L24/L16 노선은 폐기 — §5 참조) |
| 행동 원자료 | open-weight = **V4role(SM) / V2role(IC·MW)** clean 데이터, API 4종 = corrected raw |
| 검증 시점 | 본 문서의 ✅는 HF snapshot `21bdaa32904abf9f54c257ccdd455be51ab7d1f7`과 셀 단위 대조 통과. 카메라레디 생성기·사이드카·그림 전체는 `7f3036d8` 로 업로드(`paper_neurips_2026/camera_ready/`, 논문 그림 18개 전부 존재). 논문 float ↔ HF 파일 대응은 `PAPER_ASSET_MAP.md`. §4 인과 배터리와 rebuttal audit는 해당 snapshot의 wave·audit 원파일과 대조 통과 |
| 인과 배터리 | **wave 로그 canonical = llm-addiction repo `multilayer_causal/experiments/sec4_causal/INDEX.md`** (W1–W14); 데이터는 HF `experiments/sec4_causal/` |

데이터셋: HF `llm-addiction-research/llm-addiction` (public) · 코드: GitHub `iamseungpil/llm-addiction`

> **float 단위 상세 provenance는 [`paper_index/`](paper_index/README.md)에 있다.**
> 이 문서는 "어느 파일인가"를 고정한다. "그 파일이 어떤 **형태**인가", "어느 **부분집합**을 읽었는가",
> "인쇄된 값이 실제로 재현되는가"는 고정하지 않는다. `paper_index/`는 본문·부록 49개 float 각각에
> 매니페스트 하나씩을 두고 그 세 가지를 기록하며, 라운드 단위 결과에 의존하는 float에는
> **schema map**(코퍼스별 JSON 경로와 타입)과 실행 가능한 **load invariant**(코퍼스별 라운드 승률
> 0.25–0.35, 과제 사양 0.30)를 함께 싣는다. 인쇄된 논문과 셀 단위로 대조된 문서는 그쪽이다.
>
> 이 문서와 `PAPER_ASSET_MAP.md`가 경로에 대한 기준이고, `paper_index/`가 재현 상태에 대한 기준이다.
>
> 인쇄 논문과 어긋난 채 print에 도달한 결함이 하나 있었고, 그 결함은 위 세 문서가 모두 **경로는 맞게**
> 적어 두었는데도 통과했다. 경로만으로는 막을 수 없다는 것이 `paper_index/`의 schema map과
> load invariant가 존재하는 이유다.

---

## 1. 표 (Tables) — 데이터 일치 검증 완료

> `neurips_en.aux`와 대조해 갱신. **본문에 있는 표는 Table 1 하나뿐**이다(`4.neural.tex`, p8).
> 아래 나머지 세 표는 전부 부록에 있고 번호도 부록 번호다 — sharing-transfer = **Table 20**,
> condition-modulation = **Table 22**, causal-battery-suffnec = **Table 23**. 이전 판은 이 셋을
> "본문 Table 2·3·4"로 적고 있었다. 표는 전부 33개이며, float별 provenance와 재현 상태는
> `paper_index/`에 표 하나당 매니페스트 하나로 있다.

| 표 | LaTeX label | Canonical HF 파일 | 생성 스크립트 | 검증 |
|---|---|---|---|---|
| **Table 1** (본문) SAE→indicator $R^2$ (L22) | `tab:neurips-sae-results` | `sae_v3_analysis/results/table1_groupkfold_L22.json` | `sae_v3_analysis/src/run_groupkfold_recompute.py` | ✅ 13/13 셀 정확 일치 |
| **Table 23** (부록) causal battery — steering 충분성 + removal 필요성 (Gemma·LLaMA) | `tab:causal-battery-suffnec` | 충분성: `experiments/sec4_causal/checkpoints/sec4_w2*`(Gemma)·`sec4_w10*`(LLaMA) summary들 · 필요성: `experiments/sec4_causal/checkpoints/sec4_w13/*_summary.json` | (표 수치는 wave summary에서 수기 전사; wave 로그 = llm-addiction repo `multilayer_causal/experiments/sec4_causal/INDEX.md` W10/W13) | ✅ 2026-07-10 3-agent 감사에서 본문 수치 원파일 대조 통과 |
| **Table 20** (부록) cross-task read sharing 요약 (Gemma L22) | `tab:sharing-transfer` | rank-1/rank-2 AUC: `paper_neurips_2026/tables/appendix/tableA05_hidden_subspace/data/rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r{1,2}_e8g1_L22_r{1,2}.json` · feature-transfer 실패: `sae_v3_analysis/results/iba_cross_task_transfer.json` | `tables/body/table2_rq2_audits/code/cross_domain.py` | ✅ rank-1 0.80/0.74/0.52 · rank-2 0.84/0.60/0.91 + transfer<0 확인 |
| **Table 22** (부록) condition modulation (SM I_BA, L22) | `tab:condition-modulation` | `sae_v3_analysis/results/condition_modulation_groupkfold_L22.json` | `tables/body/table3_condition_modulation/code/condition_analysis_v2.py` | ✅ 0.063→0.153 (Δ_G +143%) 일치 |
| (부록) RQ2 상세 감사표 | `tab:rq2-sharing` | Table 3와 동일 원천(부록은 rank별 전체 행 + null 포함) | 동일 | ✅ |

> ⚠️ causal battery 표 주의: HF 매니페스트가 대표 파일로 적은 `rq2_audit_consistent_layer.json`은 **에러 스텁**(`layer 23 not in...`)이라 실값이 아니다. 실제 값은 위 `rq2_aligned_hidden_transfer_*L22_r1*.json`에 있다.

## 2. 본문 그림 (Body Figures)

아래 표는 `neurips_en.aux`와 대조해 갱신했다. **본문 그림은 4개**다(그림 번호 5–16은 전부 부록).
이전 판은 본문 그림을 5개로 적고 Fig 3·4·5를 어긋나게 매겼는데, matched-cap은 별도 float이 아니라
Figure 3의 (d) 패널로 들어갔다.

| 그림 | LaTeX label | 본문이 include하는 파일 | 생성 스크립트 | 데이터 원천 |
|---|---|---|---|---|
| **Fig 1** overview | `fig:experimental-overview` | `images/representative_flow_diagram.pdf` | `scripts/figures/fig01_overview_flow.py` | 수작업 다이어그램 |
| **Fig 2** slot machine | `fig:slot-machine` | `images/fig02_slot_machine.pdf` | `scripts/figures/fig02_slot_machine.py` | §3 SM 6모델 raw, `paper_data/fig02_slot_machine.json` |
| **Fig 3** investment choice (a–d) | `fig:investment-choice` | `images/investment_choice3.pdf` | `scripts/figures/fig03_investment_choice_1x4.py` | (a)–(c) §3 IC 6모델 raw; **(d) matched-cap ablation** — `paper_data/fig05_matched_cap.json` |
| **Fig 4** causal battery | `fig:causal-battery` | `images/fig04_causal_battery.pdf` | `scripts/figures/fig04_causal_battery.py` | `experiments/sec4_causal/checkpoints/` steering·removal summaries |

부록 그림 12개(번호 5–16)의 파일·생성기 대응은 `PAPER_ASSET_MAP.md`에, float별 재현 상태는
`paper_index/`에 있다. 그중 6개는 오랫동안 "생성기 없음"으로 기록돼 있었으나 실제로는 코드 repo에서
**삭제**된 것이었고(`legacy/writing/table_figure/`, 커밋 `16acccf`), 지금은
`scripts/figures/recovered/`에 원본 그대로 복원돼 있다.

> 참고: `fig05_matched_cap.pdf`와 `fig2_combined.pdf`는 **본문이 include하지 않는다**. 이전 자산이다.
> 행동 그림의 로컬 파일명과 HF archive 파일명은 생성 시점에 따라 다르므로, 본문 include 경로가 기준이다.

## 2.5 §4 인과 배터리 (causal battery) 데이터 영역 — 2026-07-12 추가

`tab:causal-battery-suffnec`·`fig:causal-battery`와 부록 causal 절(transfer matrix·condition writability)의 원천은 전부
HF `experiments/sec4_causal/` 아래에 wave 단위로 격리돼 있다. wave별 서사·판정 로그는
llm-addiction repo `multilayer_causal/experiments/sec4_causal/INDEX.md`가 canonical이다.

| 하위 폴더 | 내용 | 논문 대응 |
|---|---|---|
| `checkpoints/sec4_p0, sec4_w2–w6` | Gemma 충분성(steering) 사다리 + null·프로토콜 정착 | `tab:causal-battery-suffnec` Gemma steering, `fig:causal-battery` |
| `checkpoints/sec4_w7` + `analysis/sec4_w7_adjudication.json` | 사전등록 12셀 cross-task 전이 행렬 (7/10, 부호 11/12, MW 천장 0.823) | §4.2 write, `fig:cross-context-write` 왼쪽 패널, 부록 `tab:causal-transfer-matrix` |
| `checkpoints/sec4_w8scan–w9` | LLaMA 윈도 스캔·초기 전이(약함 = 윈도 아티팩트, W11서 해소) | (논문 미인용; 부정결과 로그) |
| `checkpoints/sec4_w10–w11` | LLaMA 충분성 + task-own 윈도 (SM L14–19 / IC L12–17 / MW L16–21) | `tab:causal-battery-suffnec` LLaMA steering, §4.2 LLaMA 문장 |
| `checkpoints/sec4_w13` | 필요성(project-out removal) 양모델 + purity 해소 | `tab:causal-battery-suffnec` removal 행 전부 |
| `checkpoints/sec4_w14` + `analysis/sec4_w14_analysis.json` | 매칭 twin-graft 사다리 (±G/+M) 공통격자 | §4.3 수치(+0.0469/+0.0358/+0.0218, twin +0.0237), `fig:cross-context-write` 오른쪽 패널, 부록 `tab:causal-condition-writability` |
| `assets/` | 축 npz (behavioural/readout/confound, 과제별·공유) | steering에 쓰인 방향 벡터 원본 |

## 3. 행동 원자료 (§3 Behavior) — Canonical 6모델

### Slot machine (19,200 games) — Fig 2/3
| 모델 | Canonical 경로 (HF) |
|---|---|
| GPT-4o-mini | `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json` (corrected parsing; round outcome은 `game_history` 병행) |
| GPT-4.1-mini | `slot_machine/gpt/gpt5_experiment_20250921_174509.json` (파일명은 legacy지만 실모델 GPT-4.1-mini) |
| Gemini-2.5-Flash | `slot_machine/gemini/gemini_experiment_20250920_042809.json` |
| Claude-3.5-Haiku | `slot_machine/claude/claude_experiment_corrected_20250925.json` |
| LLaMA-3.1-8B | `behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json` |
| Gemma-2-9B | `behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json` |

### Investment choice (9,600 games = 6×1,600) — Fig 3
| 모델군 | Canonical 경로 (HF) |
|---|---|
| API 4종 | `investment_choice/bet_constraint/` (+ `bet_constraint_cot/`) |
| LLaMA-3.1-8B | `behavioral/investment_choice/v2_role_llama/llama_investment_c{10,30,50,70}_*.json` |
| Gemma-2-9B | `behavioral/investment_choice/v2_role_gemma/gemma_investment_c{10,30,50,70}_*.json` |

### Mystery wheel (신경 분석 전용, §4) — body Fig 미사용
`behavioral/mystery_wheel/{gemma_v2_role,llama_v2_role}/*_mysterywheel_c30_*.json`

## 4. 신경 데이터 (§4 Neural) — Gemma-2-9B, LLaMA-3.1-8B

| 용도 | Canonical 경로 (HF) |
|---|---|
| 결정 시점 hidden states (6 파일) | `sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/{gemma,llama}/hidden_states_dp.npz` |
| SAE 희소 특성 (Gemma, 42층) | `sae_features_v3/{task}/gemma/sae_features_L{0..41}.{json,npz}` (GemmaScope 131K) |
| SAE 희소 특성 (LLaMA IC, 32층) | `sae_features_v3/investment_choice/llama/sae_features_L{0..31}.{json,npz}` (LlamaScope 32K) |
| LLaMA SM·MW | **SAE npz 없음 — `hidden_states_dp.npz` 기반 Ridge readout 사용** (LlamaScope 공개층 L25–31 제약 때문, §4 본문과 정합) |

## 5. DEPRECATED / DO-NOT-CITE (NeurIPS 기준 제외) — ⚠️ 인용 금지

| 항목 | 위치 | 폐기 사유 |
|---|---|---|
| `paper_neural_audit.json` | `legacy/v17_leaky_pipeline/` | CV split 전 RF fit → **라벨 누수**, R² 부풀림(LLaMA/MW 0.293→0.779). NMI 헤드라인 출처 |
| 42-layer peak sweep | `legacy/pre_groupkfold_sweep/` | random-KFold 누수, body와 최대 3배 괴리. L24/L16 피크 선택 출처 |
| V1 슬롯 원자료 | `slot_machine/gemma/`, `slot_machine/llama/` | 토큰 잘림 + 고정베팅 24.6% 오류 + 가짜 파산. → 대체: `*_v4_role/` |
| SAE patching | `sae_patching/` | 위 V1 손상 데이터 기반. → 대체: `sae_features_v3/` |
| GPT-5-mini run | `slot_machine/gpt/archived_gpt5mini_20250921/` | API 오류 60%로 9.4%에서 중단. → 대체: GPT-4.1-mini |
| RQ2 smoke stub | `rq2_audit_consistent_layer.json` | 에러 스텁(실값 아님). → 대체: `rq2_aligned_hidden_transfer_*L22_r1*.json` |

## 6. 아직 남은 참조 수정 항목 (문서/매니페스트 텍스트만; 데이터는 정상)

1. **`paper_neural_audit.json`을 master로 적은 문서 8곳** → `table1_groupkfold_L22.json`으로 교체
   (`llm-addiction/MANIFEST.md`, `sae_v3_analysis/README.md`·`docs/{EXPERIMENT_INDEX,PAPER_CANONICAL,PAPER_MANIFEST,PUBLIC_RELEASE_INDEX_20260410,WORKSPACE_RUNBOOK}.md`, `results/README.md`, `results/reports/paper_asset_manifest.md`, + HF top `README.md`)
2. **HF `paper_neurips_2026/MANIFEST.md`** causal battery 표 포인터 → `rq2_aligned_hidden_transfer_*L22_r1*.json`
3. **HF `behavioural_data_used_by_paper/README.md`** 깨진 경로 정정:
   `behavioral/slot_machine/{claude,gemini,gpt}` → `slot_machine/{claude,gemini,gpt}`; `slot_machine/gpt5|gpt41` → `slot_machine/gpt`; 모델명 `GPT-5-mini` → `GPT-4o-mini`
4. **HF `sae_features_used_by_paper/README.md`** LLaMA SM/MW `sae_features_L22.npz` 경로(미존재) → `hidden_states_dp.npz` 명시
5. **NMI판(`content/`)** 신경 표는 leaky 값 — NeurIPS L22로 갱신할지 / NMI 노선 보류할지 결정 필요(별도 사안)

---
### 7. paper_index/ — float 단위 매니페스트 (이 문서의 하위 문서)

`paper_index/`는 인쇄된 논문과 셀 단위로 대조된 유일한 문서 집합이며, 이 문서를 대체하지 않고 보완한다.
분업은 다음과 같다.

| 질문 | 기준 문서 |
|---|---|
| 어느 HF 파일인가 / 무엇이 deprecated인가 | **이 문서** |
| 논문 float ↔ 자산·생성기 경로 | `PAPER_ASSET_MAP.md` |
| float 하나가 어느 **부분집합**을 어떤 **형태**로 읽었고, 재현되는가 | **`paper_index/<장>/<float>.md`** |

`paper_index/`가 추가로 싣는 두 필드가 이 문서에는 없다.

- **schema map** — 코퍼스별로 라운드 결과가 놓인 JSON 경로와 그 **타입**. 여섯 슬롯머신 코퍼스가
  서로 다르다: open-weight는 `history[i].win`(bool), GPT-4.1/Gemini/Claude는
  `round_details[i].game_result.result`(**dict**), GPT-4o-mini는 `game_history[i].result`이며
  `game_result`가 아예 없다. dict를 `str()`로 감싸면 예외 없이 승률 0.000이 나온다.
  프롬프트 알파벳도 갈린다 — 다섯 번째 모듈을 LLaMA V4role은 `H`, 나머지 다섯은 `R`로 쓴다.
- **load invariant** — 실행 가능한 단언과 기대값. 슬롯머신 코퍼스는 결과가 기록된 라운드 기준
  코퍼스별 승률이 `[0.25, 0.35]` 안에 있어야 한다(과제 사양 0.30). 공개 릴리스에 대해 실행했고
  여섯 코퍼스 모두 0.30에서 0.004 이내로 통과한다. **invariant를 돌리지 않은 매니페스트는
  VERIFIED를 주장할 수 없다.**

이 두 필드가 있는 이유는 단순하다. 인쇄에 도달한 결함 하나는 이 문서와 `PAPER_ASSET_MAP.md`가
**경로를 모두 맞게** 적어 둔 상태에서 통과했다. 경로는 형태를 말해 주지 않는다.

---

_생성: NeurIPS 기준 정리. 기존 파일·HF 미변경(비파괴). 위 §6은 승인 후 실제 수정 대상._
_2026-07-12: §4 인과 업그레이드 반영 — 본문 float 재번호(Table 2=causal battery, Table 3=sharing 요약, Table 4=condition), Fig 4·Fig 5 신규 매핑, §2.5 인과 배터리 데이터 영역 추가, `fig:sharing` 부록 이동 반영._
_이후 갱신: 위 2026-07-12 줄의 float 번호는 그 뒤 조판에서 바뀌었다. `neurips_en.aux` 기준으로 본문은
그림 4개와 표 1개뿐이고, causal battery·sharing·condition 표는 전부 부록(Table 23·20·22)이다.
§1·§2 표는 그에 맞춰 고쳐 두었으며, float 단위 재현 상태는 §7의 `paper_index/`가 기준이다._
