# All-Layer §4 Read/Write Spine — Design Spec (2026-06-14)

## 1. 목적 (one paragraph)
논문 §4의 세 read 감사(§4.1 지표 readout, §4.2 과제 공유 BK 기하, §4.3 자율성 조건변조)를
단일 대표층(L22)/격자({8,12,22,25,30})가 아니라 **모든 레이어**에서 재적합하고, 우리가
인과적으로 찾은 **쓰기 창(write window)** 의 층 프로파일과 **같은 좌표 위에 겹쳐** 그린다.
산출물은 세 절 × 두 모델 각각 `read(layer)` vs `write(layer)` 오버레이로, "어느 층에서
읽히고 어느 층에서 쓰이는가"를 한 그림에 담아 §4를 하나의 척추로 묶는다.

**불변 원칙**: 읽기 *방법*을 새로 만들지 않는다(기존 GroupKFold/SAE-feature·hidden-state
파이프라인을 더 많은 층에서 돌릴 뿐). 논문 본문 표·검증 수치 불변. 코드/산출물 격리.

## 2. 범위 (확정: 옵션 b)
- **모델**: Gemma-2-9B (42층) + LLaMA-3.1-8B (32층)
- **과제**: SM, IC, MW (세 과제)
- **절**: §4.1 + §4.2 + §4.3 전부, 각 절 read(전층) + write(층 프로파일)

### 데이터 가용성 (탐색 완료)
| 셀 | SAE 특징 | 전층 hidden state | read 비용 |
|---|---|---|---|
| Gemma SM/IC/MW | 42층 ✓ | phase_a ✓ | CPU |
| LLaMA IC | 32층 ✓ | phase_a ✓ | CPU |
| **LLaMA SM/MW** | 5층(dp)만 | 없음 | **GPU 추출** |

LLaMA-IC가 32층 SAE 특징을 이미 가지므로 추출 파이프라인 재현성은 증명됨 →
SM/MW도 동일 추출 가능(`extract_llama_*` 계열).

## 3. 산출물 (deliverable)
`experiments/multilayer_causal/results/spine/` 에:
- `read_profile_{model}_{task}_{section}.json` — 층별 read 지표
  (§4.1: 3지표 R²+perm-p / §4.2: LOTO AUC+transfer / §4.3: ±G·±M 선예화 ΔR²)
- `write_profile_{model}_{section}.json` — 층별 write 효과 (기존 W1–W3 + 신규 빈칸)
- `spine_{model}_{section}.{json,pdf}` — read vs write 오버레이 곡선

## 4. 컴포넌트 (각 단위: 무엇을/어떻게 쓰나/의존)

### C1. read 스윕 러너 (논문 파이프라인 확장, 격리 신규 스크립트)
- **무엇**: 한 (model, task, section, layer)의 read 지표를 계산.
- **어떻게**: 기존 `sae_v3_analysis/src` 함수 재사용 —
  §4.1 `fit_one_subset`/`fit_groupkfold`(SAE-feature Ridge, GroupKFold, RF 잔차화),
  §4.2 `cross_domain.py`(v_BK·LOTO PCA), §4.3 `condition_analysis_v2.py`(±조건 재적합).
  **신규 래퍼** `sae_v3_analysis/src/run_spine_layer.py` 가 `--model --task --section --layer`
  를 받아 위 함수를 호출(논문 스크립트 자체는 **수정 금지**, `VALID_LAYERS` 가드만 래퍼에서 우회).
- **의존**: HF의 `sae_features_L{n}.npz`(층별) + `phase_a_hidden_states.npz`.
- **방법 선택**: 1차 = **SAE-feature readout**(Table 1과 동일, SAE 있는 층). 보조 =
  **hidden-state readout**(모든 층 균일, 양 모델) — 두 곡선을 함께 보고해 격자에서 교차검증.

### C2. LLaMA SM/MW 전층 추출 (GPU, 노드)
- **무엇**: LLaMA SM/MW 결정-시점 전층(32) hidden state + SAE 특징 추출 → HF 업로드.
- **어떻게**: `extract_llama_*` 계열 재사용(IC가 선례). 3200 게임/과제, 후크로 전층 캡처.
- **의존**: meta-llama/Llama-3.1-8B-Instruct, LLaMA SAE 사전(IC 추출이 쓴 것과 동일).
- **게이트**: 추출본의 격자층 read R²가 논문 LLaMA 행(L22 I_BA 0.109 등) ±0.05 재현 →
  실패 시 추출 파이프라인 불일치로 보고, LLaMA 전층 보류(Gemma 단독 진행).

### C3. write 빈칸 채우기 (인과 층 프로파일 완성)
- §4.1 write: **이미 전층**(W1 e1 폭6 타일링 42층 + w1e 16–25). 재사용.
- §4.2 write: **신규** BK 통제축 층-스윕 — BK축을 폭6 윈도 7개에 steering(Gemma).
- §4.3 write: +G=e1(전층, 재사용) + **신규** +M 폭6 층-스윕(Gemma w3m 확장).
- LLaMA write: 단일-결정 ±G 갭 null이므로 행동축·BK축 steering으로만(추출 후, 5층→전층).
- **어떻게**: 기존 `multilayer_causal` runner(steer/patch) + arms.yaml 추가. GPU 노드.

### C4. 통합 분석/플롯
- **무엇**: read_profile + write_profile → 오버레이 곡선 + 핵심 수치표.
- **어떻게**: `multilayer_causal/src/spine_stats.py`(신규, CPU). 정규화: read=R²/AUC,
  write=−G 갭 대비 회복률. 층축 정렬, 두 곡선 교차/해리 지점 표시.

## 5. 데이터 흐름
HF(SAE feat/hidden) → [C2 LLaMA 추출→HF] → C1 read 스윕(층별 병렬) → read_profile;
기존 W1–W3 + C3 신규 write arms → write_profile; → C4 → spine 곡선.

## 6. 컴퓨트 전략
- **CPU 팬아웃 (Gemma + LLaMA-IC read)**: 42·42·42·32 층 적합을 노드 CPU에서 병렬
  (층당 수 분, 독립). 노드 1대로 충분, GPU는 C2/C3에만.
- **GPU (노드)**: C2 LLaMA SM/MW 추출 + C3 write arms. 4×H100, 기존 amlt 패턴.
- **순서**: ① Gemma read 스윕(즉시, CPU) + LLaMA 추출(GPU) 병렬 → ② LLaMA read 스윕
  → ③ write 빈칸 → ④ 통합. 단계별 HF 동기화·resume(기존 ArmCheckpoint 패턴).

## 7. 격리 / 검증
- 신규 코드: `sae_v3_analysis/src/run_spine_layer.py`(논문 함수 호출만, 원본 불변),
  `multilayer_causal/src/spine_stats.py`, write arms는 `multilayer_causal` 안.
- 논문 검증 스크립트·표 불변(격자층 재현으로 자기검증: 새 러너의 L22 출력 = `table1_groupkfold_L22.json` ±tol).
- 테스트: 격자층 재현 게이트(L8/12/22/25/30이 기존 sweep과 일치), 합성 단위테스트(잔차화·LOTO·정규화).

## 8. 사전등록 해석 (read vs write 곡선)
| 패턴 | 해석 |
|---|---|
| read 넓음(L8–30) · write 좁음(L16–21), read가 16–21에서도 ≥0 | "read ⊃ write": 디코딩 가능 ≠ 통제, 통제는 국소 |
| read가 L16–21에서 L22보다 **약함** | "모델은 디코딩 약한 층에서 결정을 쓴다"(강한 해리) |
| §4.2 공유/§4.3 선예화가 L16–21에 집중 | 상관 구조가 인과 층에 정박 — §4 통합 |
| 위가 read 층(L22)에 집중, write와 분리 | read-층 현상 vs write-층 — 명확한 이중 분리 |
| LLaMA read 피크 층 ≠ Gemma | 기능적 국소화 모델-특이 / 해리는 공통 |

## 9. 본문 반영 (불변 보장)
- 본문 Table 1/2/3·검증 수치 **불변**. 추가는 부록 layer-sweep 그림 확장 + §4 마감
  단락(읽기/쓰기 척추) + 기존 인과 결과(W1–W3)와의 연결.
- read가 L16–21에서 더 강하게 나와도 L22 대표 선택(최약셀 동률 근거)은 불변; L16–21은
  부록/인과 절 한정.

## 10. 비범위 (YAGNI)
- 새 readout *방법* 개발 ✗ (기존 파이프라인만).
- W4 인과 정밀화(3지표 디코더·30-random 특이성)는 **이 wave 후** read 결과 보고 설계.
- 본문 표 재작성 ✗.
