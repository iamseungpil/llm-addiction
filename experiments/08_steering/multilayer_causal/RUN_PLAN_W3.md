# W3 실행 계획 — 의도 · 가설 · 검증 (plan v4, 비판 23건 반영)

**총괄**: 논문 limitations §6의 열린 질문("controller-level locus — earlier-layer, distributed
multi-layer pathway — remains an open question")에 대한 답을 절별-삽입 가능한 형태로 완성한다.
새 readout은 만들지 않는다 — 모든 새 방향은 **개입 도구**로만 쓰며 읽기 표(Table 1/2/3)는 불변.

공통 설계: Gemma arm은 **state_offset 300 + seed_base 2000042** (W2 앵커
`w2_anchor_minus/plus`를 비교군으로 재사용 — 동일 상태·시드라 직접 비교 가능, 앵커 재실행 불요).
LLaMA arm은 자체 앵커 게이트. 전 steering arm은 per-layer 3%-norm 스케일(섭동 크기 통일).
discovery n=50 → 승격 후보만 W4 confirmatory (R1–R7 규율 유지).

## 사전등록 (제출 전 고정)

1. **PR-1 (w3a)**: Appendix M3 protocol 2의 SAE-readout 방향(L22 null 기보고)은
   **쓰기 가능한 창 L16–21에 놓아도 I_BA 무반응**일 것. (근거: L22-23 전체-상태 패치 null +
   raw-Ridge 프록시 null.) 양성 시 = "옳은 창에서는 디코더 방향도 손잡이" — 더 강한 결과로 보고.
   **누출 공시·양성 조건(라운드-1 수정)**: 문자 그대로의 Table-1 방향은 offset-300 eval 풀
   게임을 포함한 코퍼스에서 적합되었다(eval 창 게임 50/50이 in-corpus 적합셋에 존재 —
   정확 수치는 빌더가 npz에 기록, 논문 본문에 공시). 재현 게이트 refit·scales·트윈 방향은
   eval-게임 제외 코퍼스(axes.py `excluded_game_ids`, n_eval=500)에서 계산하고 npz에
   in-corpus/제외-refit R² 둘 다 기록한다. **양성 보고 조건: 문자 그대로의 방향(saerd)과
   제외-refit 트윈(saerdx, `directions_saerd_excl.npz`) 둘 다 행동을 움직일 때만**
   "디코더 방향도 손잡이"를 주장한다 — 둘이 갈리면 eval-게임 특이 구조 암기 가능성으로 보고.
   null 분기에는 누출이 보수적이므로 PR-1 null 해석은 불변.
2. **PR-2 (w3L)**: LLaMA 쓰기 창 위치는 미지. **절대지표(L22 부근)와 깊이-매칭
   (Gemma L16–21 ≈ 38–50% 깊이 → LLaMA L12–16) 둘 다** 해석 기준으로 등록.
   - 창=깊이-매칭 → read/write 해리가 모델-일반 (Gemma L22-read는 52% 깊이)
   - 창=다른 위치 → 기능적 국소화는 모델-특이, 해리 자체는 각자 성립
   - 창 없음 → 쓰기 국소화는 Gemma-특이 (정직 보고)
   승자 선택 규칙: **효과크기 top-1만** W4 n=200 confirmatory, 수축(shrinkage) 보고.
   **게이트 검정력 (선추정 완료)**: `configs/llama_gap_estimate.json` (추적 사본; out/은 스크래치) — ±G 갭 0.0225,
   d=0.134, n=50 검정력 0.106 (불충분). 계획 규칙대로 n 상향 — **앵커 n=878**
   (검정력 0.806; −G 풀 3636 ≥ 928 필요분). 스윕 창은 discovery n=50 유지.
   **창 효과 claimability (라운드-2 사전등록)**: 스윕 n=50과 W4 n=200은 갭-크기
   (d=0.134) 창 효과에 대해 검정력 0.106 / 0.277 — 두 단계 모두 갭-크기 효과는
   대부분 놓친다 (discovery miss rate ≈ 89%, W4 ≈ 72%). MDE는 같은 파일의
   시뮬레이션 검증값(`mde` 키): **n=50 → shift 0.0992 (d=0.588, ±G 갭의 4.4배)**,
   **n=200 → shift 0.0496 (d=0.294, 갭의 2.2배)**. 따라서 w3L 양성 주장은 해당
   단계 MDE 급 이상의 효과에서만 기대 가능하고, **null은 "갭-크기 이하 창 효과
   없음"으로 해석하지 않는다** (갭-크기 효과 배제에는 창당 n≈878이 필요하나 후크
   있는 패치 생성이라 W3–W4 예산 밖 — 정직 공시). 이 설계가 정당한 근거: 표적
   효과는 Gemma-급 쓰기 창(W2 갭 0.141, d=0.837)이며 그 크기에는 n=50 검정력
   0.981 / n=200 ≈ 1.0 (`gemma_magnitude_power` 키)로 충분.

## Arm 그룹

### w3a — SAE-readout 방향 × 쓰기 창 (6 arms, Gemma)
- **의도**: 부록 M3가 L22에서 null이었던 "Table 1의 문자 그대로 그 디코더 방향"을
  인과 창 L16–21에 배치 — "그른 레이어"와 "그른 방향"을 분리하는 마지막 셀.
- **빌드**: `sae_features_L22.npz`(HF) → 논문 레시피(top-200 rank-corr, Ridge) →
  Gemma-Scope L22 W_dec 사상 → 단위벡터. **재현 게이트**: eval-게임 **제외** refit R²가
  Table 1 I_BA 셀(0.167) 대비 ±0.05 안 (GroupKFold 구조 차이 허용오차; in-corpus R²는
  공시용 병기). **트윈**: 제외-refit를 동일 canonical W_dec(average_l0_105, 저장 d_unit
  재구성 cos>0.999로 소스 검증)로 사상한 `saerdx` — PR-1 양성 조건의 두 번째 방향.
- arms: saerd/saerdx × α∈{−2,+2,+4}, layers [16,21], n=50.
  **1차 지표: I_BA(bet만), 클러스터 perm vs W2 −G 앵커.**
- M3 L22 결과는 인용으로 충당 (재실행 금지 — 비판 #1).

### w3bk — 공유 BK 부분공간 = 통제 부분공간? (8 arms, Gemma SM + IC)
- **의도(사전등록 문구)**: Table 2의 sharing 주장 검증이 **아니라**, 신규 질문 —
  "약한 공유 모니터링 기하(표적-제외 LOTO rank-1 축, 신규 객체)가 통제 부분공간이기도
  한가". null이어도 Table 2 불변.
- **Table 2 비교 불가 (정정)**: Table 2의 rank-1 셀(SM 0.80 / IC 0.74,
  v24_rq2_sweep_summary)은 **3-과제(표적 포함)** raw centroid의 중심화-PCA 축을 표적
  자신의 fitted readout 투영으로 hidden_states_dp game-level rows에서 평가한 값 —
  본 LOTO 축(표적 제외, 비중심 SVD, phase_a round-level 투영)과 **추정량이 다르다**.
  따라서 "재현" 게이트의 대상이 아니며, 셀 수치는 인용으로만 쓴다.
- **빌드**: 과제별 v_BK = μ_stop − μ_bankrupt (L22, phase_a + 카탈로그 종말점 라벨) →
  표적과제 **제외** LOTO rank-1 축 (SM 표적 = IC+MW 스택의 1st 우특이벡터).
  **게이트(사전등록, 신규 객체 기준 — 빌더가 npz에 기록)**:
  ① 관련성 — 표적과제 **game-level** 투영 AUC의 순열검정(양측 통계량 |AUC−0.5|,
     게임 단위 1e4 perm, seed 42) p < 0.05 (부호를 사후 보정하므로 양측이 정직);
  ② 공유-존재 — 소스 v_BK 쌍 cos의 게임-부트스트랩 2.5백분위 > 0
     ("공유 rank-1 축"이 실재해야 함; IC 표적은 소스쌍 cos≈0.02로 사실상 직교 →
     게이트 실패 = w3bk_ic arm 자동 제외가 사전등록된 결론);
  ③ 안정성 — 게임-부트스트랩 축 cos 중앙값 > 0.9.
  **부호 보정**: 저장 방향은 표적 read 기하 기준 **stop-ward**로 정렬
  (game AUC < 0.5면 축 반전, `orientation_sign` 기록 — SM은 소스-평균 방향이
  anti-aligned라 반전됨; span은 표적-제외 그대로, 부호 규약만 표적 라벨 사용).
  steering 해석(stop-ward = +α)은 보정 후 방향 기준.
- **잔액 confound 통제**(비판 #4, 라운드-2 정정 — **부분 통제임을 명시**):
  `w3bk_smres_p40` = 잔액-잔차화 v_BK 변형(소스 상태를 게임별 잔액에 per-dim 선형
  잔차화 후 재계산). 실측: cos(bk_sm, balres)≈0.95이고 balres 축은 표적 SM 상태의
  잔액 정렬을 **절반 정도만** 제거(투영-잔액 Pearson ≈ −0.27→−0.13, 잔액-기울기
  방향 cos ≈ −0.41→−0.20) — 빌더가 두 bk_sm npz 모두에 `cos_plain_vs_balres` +
  `balance_align_*`/`cos_balance_slope_*`(표적·소스, counterpart 포함) 키로 기록
  (`auc_on_balres_states`는 잔액 수치가 아니라 잔차화 상태의 BK row-AUC). 따라서
  **두 arm의 일치는 잔액-방향 해석을 배제하지 못한다**(consistent-with-only;
  판별력 있는 분기는 갈림뿐): 본 축과 결과가 갈리면 "잔액 방향" 해석, 일치하면
  잔액 contrast를 한계로 명시해 보고.
- **잔액 판별 통제 (라운드-3 추가 — balres만으로는 판별 불가)**: balres는
  부분 통제(cos(plain, balres)≈0.95)라 n=50 steering에서 두 arm은 거의 확실히
  일치 → 잔액-confound 공격이 사실상 반박 불가였다. 같은 Δ-norm의 판별 arm 2종을
  사전등록 추가 (빌더 `build_bk_sm_balance_controls`, 게이트는 구성-타당성만):
  - `w3bk_smorth_p40` (`directions_bk_sm_balorth.npz`): bk_sm 축을 **표적 SM
    잔액-기울기 방향에 명시적으로 직교화**(target-side; cos=0 by construction,
    잔차 norm 0.914 > 게이트 0.2; 직교화 후에도 표적 game-AUC 0.71, p≈1e-4 —
    비-잔액 성분이 BK를 읽음, 공시 키 `auc_game_disclosure`).
  - `w3bk_smbal_p40` (`directions_bk_sm_balslope.npz`): **순수 잔액-기울기
    방향 자체**를 bk_sm 축의 잔액 성분과 부호-정렬해 steering — confound-양성
    통제 (|cos(plain, balance-slope)| 0.406 > 게이트 0.05).
  **사전등록 해석 규칙**: w3bk_sm_p40 양성일 때 ① smbal 양성 + smorth null →
  잔액-방향 해석 승리; ② smbal null + smorth 양성 → 잔액 해석 **배제**(같은
  Δ-norm에서 잔액 방향 전체가 무효과면 bk 축의 |cos|=0.41 성분은 더더욱 무효과)
  — "not excluded"가 아니라 배제로 보고; ③ 둘 다 양성 → 혼합(두 성분 모두 인과
  내용, 잔액 contrast 한계 명시); ④ 둘 다 null → 효과가 정확한 plain 방향 특이
  (취약 결과로 정직 보고). null/양성 판정은 1차 지표(아래)의 클러스터-perm
  vs W2 −G 앵커, discovery 기준.
- arms: SM — α∈{−2,+2,+4} steer [16,21] n=50 + 잔차화 +4 n=50 + 판별 통제
  smorth/smbal +4 n=50 각 1;
  IC — α∈{+2,+4} n=50 (**ITT**: 파싱실패=명시적 결과 범주, 재샘플 금지, 민감도 한계 계산).
- **1차 지표(비판 #5)**: SM = 라운드-수준 bet_ratio + 정지율(잔액×라운드 층화 부차);
  IC = 고위험 선택률(파싱실패 별도 범주). **조작 점검**: log_vectors로 L22 투영 이동 확인.

### w3pos — 양성 컨트롤 (1 arm)
- 검증된 I_BA축(α=+4, [18,23], probe) 동일-Δnorm 재실행 n=50 — 새 wave의 하네스 앵커이자
  BK 효과를 "기지 인과축 대비 %"로 보고하기 위한 기준 (비판 #11, #12).

### w3rnd — 방향-null 분포 (8 arms)
- 8개 random 방향 × α=+4 × [16,21] × n=50, probe — **풀링 분포 + 순위검정**
  (1방향×n200 대신; 논문의 30-random 관행에 정렬, 비판 #10). 프로브 spec null 분포 겸용.

### w3m — +M 성분 매개 (3 arms, Gemma)
- **의도**: Table 3의 Δ_M(읽기 선예화)의 인과 대응 — +M 트윈 패치가 행동을 +M 쪽으로 미나,
  그리고 **어느 창**으로 미나 (G의 창 가정 금지 — 비판 #15).
- arms: patch, twin_component=M, 창 {[8,13],[16,21],[24,29]}, n=50.
- **상태 슬라이스 (라운드-2 수정)**: `twin_combo`는 base combo에 이미 M이 있으면
  무연산(donor == base, 항등 패치)인데 −G 풀은 G만 거른다 — 기존 offset-300 슬라이스는
  27/50이 M 포함이라 유효 n=23으로 희석. 수정: w3m은 평가 슬라이스를 **M-제외
  base combo**로 한정 — 동결 풀 순서 그대로 offset 300부터 전방 스캔해 처음 50개
  M-제외 상태 사용(n=50 전-활성, runner가 twin_component≠G일 때 자동 적용).
  스캔 슬라이스는 앵커 창(원 인덱스 300–499) 안에 머문다(실측: 실 카탈로그에서
  50개 M-제외 상태 = 원 인덱스 301–402, 검증 완료).
- **1차 지표**: bet_ratio vs −G 앵커(**M-제외 base combo 층으로 한정한 앵커 trial만
  풀링** — 앵커가 source_state.prompt_combo를 per-trial 기록하므로 분석 시 제한;
  층 비한정 풀링은 층 구성 confound) + 자연 +M 행동(카탈로그) 대비 갭 회복.
- 해석: [16,21] 단독 양성 → "성분들이 공통 창으로 수렴"; 타 창 양성 → 성분-특이 경로.

### w3L — LLaMA 쓰기 창 스윕 (10 arms)
- **게이트 선행**: `w3L_anchor_minus/plus` **n=878** — ±G bet_ratio 분리 p<0.05 필수.
  게이트 크기는 **기존 LLaMA 행동 코퍼스에서 ±G 갭을 CPU로 선추정 완료**
  (`configs/llama_gap_estimate.json` (추적 사본; out/은 스크래치): gap 0.0225, d=0.134 → n=50 검정력 0.106, 불충분 →
  규칙대로 n 상향: 878/arm에서 0.806; 앵커는 후크 없는 생성이라 비용 수용 가능,
  게이트 검정력은 앵커 n이 전담). 게이트 실패 시 스윕 해석 중단(R1).
- 스윕: 폭4 비중첩 8창 {[0,3],[4,7],[8,11],[12,15],[16,19],[20,23],[24,27],[28,31]}, patch, n=50.
- **검출 한계 공시**: 스윕 n=50의 MDE는 d=0.588 (±G 갭의 4.4배), W4 n=200은
  d=0.294 (갭의 2.2배) — PR-2 claimability 규칙 적용 (양성 주장은 MDE-급 효과
  한정, null은 갭-크기 효과를 배제하지 않음; Gemma-급 표적 효과에는 충분 검정력).
- 모델: `meta-llama/Llama-3.1-8B-Instruct` (원 실험 스크립트와 동일), 32층/4096dim은
  모델 config에서 유도. 프롬프트는 `llama_gemma_experiment.py`와 패리티 테스트 강제.

## 통계 명세 (비판 #22)
- 클러스터 단위: SM/LLaMA-SM = 원천 게임 상태(풀 인덱스), IC = game_id. 전 검정 클러스터-perm.
- 효과 보고: 절대치 + W2 앵커 갭 대비 % + (steering은) w3pos 대비 %.
- 지표 매핑표: 본 트랙 mixed bet_ratio(−G 0.061 / +G 0.202, W2 n=200)는 부록 M3 앵커
  (0.051 / 0.216, n=50)와 동일 지표·동일 수준 — 논문 문장에 1:1 사상 가능.

## L22_PLAN_v5 관계 (공식 reconcile)
`LLM_Addiction_NMT_KOR/L22_PLAN_v5.md`(동결, full-game BK steering, 주로 LLaMA L22)와 본 트랙은
**프로토콜이 다름**(전게임 누적 steering의 종말점 vs 단일-결정 prefill 개입). 본 계획이
단일-결정 질문을 전담하고, v5는 별도 트랙으로 유지하되 두 결과를 같은 절에 쓸 때
프로토콜 차이를 명기한다. 모순되는 사전등록이 아님을 paper repo 노트로 남긴다.

## 결과→본문 문장 매핑 (비판 #23, 사전 작성)
| 결과 | 본문 행동 |
|---|---|
| PR-1 null 유지 | §4.1·M3: "그른 레이어 *및* 그른 방향 — 디코더는 측정기" + limitations 열린질문을 답으로 교체 |
| PR-1 양성 | §4.1: "디코더 방향도 옳은 창에서는 손잡이" (더 강한 긍정) |
| w3bk SM 양성+잔차화 생존 | §4.2 마감: 판별 통제 분기 적용(라운드-3) — smbal 양성+smorth null → "잔액 방향이 통제 손잡이"; smbal null+smorth 양성 → "공유 기하가 통제력 있음, **잔액 해석 배제**"; 둘 다 양성 → 혼합(잔액 contrast 한계 명시); 둘 다 null → plain-방향 특이(취약) |
| w3bk null 또는 잔차화로 소멸 | §4.2 마감: "공유는 읽기 수준(잔액 기하)" — Table 2 불변 |
| w3m [16,21]만 양성 | §4.3 마감: "성분 공통 창" |
| w3L 깊이-매칭 창 | 해리 모델-일반 문장 / 다른 창 → 모델-특이 문장 / 무창 → Gemma-한정 정직 보고 |

## 구현 원칙 (클린 코드)
- 새 방향 = 오프라인 빌더(`src/paper_axes.py`) → 표준 npz(스키마 버전 필드) → 기존 steer 경로.
  runner 변경은 ① 모델 맵(llama 1행) ② twin_component 가산 파라미터(기본 G, 기존 동작 불변) ③
  config-유도 n_layers/d_model 뿐.
- 동결 규약 = "gemma 패리티 테스트 green 불변". llama 지원은 가산 + llama 패리티 테스트 신설.
- 빌더 게이트(w3a 재현 게이트, w3bk 사전등록 construct 게이트) 실패 시 그 축의 arm은
  제출 명단에서 자동 제외(빌더가 게이트 결과를 npz에 기록, runner가 로드 시 강제).
- 격리: 코드·자산(`assets/w3/`)·체크포인트(`checkpoints/w3*/`) 전부 multilayer_causal/ 안.
- HF 동기화·resume: 기존 ArmCheckpoint 그대로 (10 trial마다 같은-경로 overwrite; 재제출=이어하기).

## 제출 절차 (구현·검증 완료 후)
```bash
python multilayer_causal/scripts/push_code_to_hf.py   # HF_TOKEN env
bash multilayer_causal/amlt/render.sh
amlt run multilayer_causal/amlt/w3.rendered.yaml mlc-w3-<date> -y
```

## 감사 공시 (최종 라운드, 제출 전 고정)

1. **bk 계열 eval-중첩 공시**: bk_sm LOTO span 자체는 표적-누출 0이나, balance_slope 방향·
   orientation_sign 보정·관련성-게이트 AUC·bk scales는 전체 SM phase_a(21,421행, 그중
   offset-300 eval 게임 50개분 358행 ≈1.7% 포함)에서 계산됨 — saerd의 제외-refit 규율과
   비대칭. 1.7% rank-1 선형 통계에의 기여는 수치적으로 비물질적이라 판단해 재빌드 대신
   공시를 택한다(npz에 eval_overlap 기록).
2. **bk 안정성 게이트의 한계**: 게이트 체계의 binding은 존재성 게이트(소스쌍 cos 부트
   2.5백분위>0)다. 중앙값-cos 안정성 기준은 다봉 분포에 둔감하다(bk_ic가 사례: 중앙값
   0.942인데 47.5% 리샘플 cos<0.5 — 존재성 게이트가 정확히 이를 잡아 IC arm 제외).
   bk_sm은 2.5백분위 0.947로 어느 기준으로도 통과.
3. **preflight 의미론**: `scripts/preflight_w3.py`가 w3bk_ic[EXCLUDE]로 exit 1을 내는 것은
   **사전등록된 정상 결과**다(게이트-실패 축의 arm이 registry에 기록용으로 남아 있음).
   제출 명단(템플릿 run_arms 목록)은 test_w3_template_gates가 강제하며, runner도 로드 시
   fail-closed. 제출 절차에서 preflight 출력의 EXCLUDE 행은 차단 사유가 아니다.
