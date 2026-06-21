# W4 실행 계획 — "다른 축" + LLaMA 구제 (plan v1)

**총괄**: W3가 연 두 공백을 닫는다 — (1) §4.1 인과 검증이 I_BA 디코더뿐(I_LC·I_EC 미검증),
(2) saerd가 L16–21에서 움직였으나 random도 움직여(rank 1/9=p0.11) **방향-특이성 미확정**,
(3) LLaMA가 G-트윈 게이트 실패로 null. W4는 셋 다 G-독립 축으로 해결한다.

공통: Gemma arm = offset 300 / seed_base 2000042 (W2 앵커 w2_anchor_minus/plus 재사용).
LLaMA arm = **W3의 n=878 앵커(w3L_anchor_minus/plus, offset 0·seed 42) 재사용** — 비싼
앵커를 다시 굽지 않고 steering arm만 같은 풀에서 n=50.

## 핵심 통찰 (W3 데이터가 강제한 설계 수정)
W3에서 누출-클린 디코더(saerdx)는 **α=4에서만** 효과(betΔ +0.169, parse 0.94); α=2는 무효
(+0.019, p=0.25). 그런데 α=4는 random도 +0.135까지 미는 "민감 창". 따라서 특이성 검정은
**α=4에서, random을 8개가 아닌 30개**로 깔아 tight null을 만들어야 한다(논문 30-random 관행).
saerdx가 30-random 95백분위 밖이면 = "디코더 방향도 특이 손잡이"; 안이면 = "L16–21은
용량만 키우면 방향 무관하게 미는 창 — 결정자는 레이어"(정직, M3 L22 null과 합치).

## Arm 그룹

### w4dec — 세 지표 디코더 × L16–21 (Gemma, "다른 축" 핵심)
- **의도**: Table 1이 보고하는 I_BA·I_LC·I_EC 세 readout을 **모두** 인과 창에서 검증.
  W3는 I_BA만. saerd 빌더의 타깃 열만 바꿔 ilc/iec 방향 생성(누출-제외 refit, saerdx 규약).
- arms: w4dec_iba_p20/p40, w4dec_iec_p20/p40 (단일생성), w4dec_ilc_p20/p40 (**probe** — I_LC
  손실특이성), 전부 steer [16,21] n=50. 디코더는 `assets/w4/directions_saerd_{iba,ilc,iec}.npz`.
- **재현 게이트**: 각 디코더 제외-refit R²가 Table 1 해당 셀(±0.05) — iba 0.167 / ilc 0.059 /
  iec 0.051. 게이트 실패 축은 제출 자동 제외(빌더가 npz에 기록).
- **1차 지표**: 해당 지표(클러스터 perm vs −G), **+ 30-random 대비 순위**(아래).

### w4rnd — 30-방향 특이성 null (Gemma)
- **의도**: 비판 #10 정식 해결 — α=4 [16,21] tight null. w4dec/w4bk의 방향-특이성을
  rank-검정(real이 30개 중 최상위면 one-sided p≈1/31<0.05).
- arms: w4rnd_s0..s29, steer [16,21] α=4 n=50, direction random, dir_seed 2026080000+i,
  단일생성(I_BA·I_EC·stop null). (I_LC probe-null은 W3 w3rnd 8개 재사용 — 약하나 공시.)

### w4bk — BK 통제축 특이성 마감 (Gemma)
- **의도**: W3 BK는 부호-특이성(random은 stop↓, BK는 stop↑)은 있으나 30-random 크기-특이성
  미정. w4bk_sm_p40 = bk_sm α=4 [16,21] n=50 재실행 → 30-random 순위.

### w4L — LLaMA G-독립 축 (PR-2 깊이-매칭 + 통제축 모델-일반성)
- **의도**: ±G 갭 없이도 LLaMA 인과를 검증. **L12 = 깊이-매칭** 가설의 sharp test.
  데이터: `sae_features_v3/{slot_machine,...}/llama/hidden_states_dp.npz` (5층 {8,12,22,25,30},
  3200행, outcome 라벨 有). I_BA축은 llama 행동 카탈로그와 game_id 조인.
- **빌드**: LLaMA I_BA 행동축(5층) + LLaMA BK-LOTO 통제축(5층, outcome=bankruptcy/stop) →
  `assets/w4/directions_llama_{iba,bk}.npz`.
- arms (전부 reuse w3L 앵커, offset 0 seed 42 n=50): w4L_iba_grid5(5층 동시, 읽기격자 쓰기
  테스트=Gemma w1b 대응), w4L_iba_l12(깊이-매칭 단독), w4L_iba_l22(읽기-대표 단독),
  w4L_bk_grid5, w4L_bk_l12.
- **해석(PR-2 사전등록)**: L12 양성 → read/write 해리 *및* 통제축이 모델-일반(깊이-매칭);
  5층 전부 null → "읽기격자≠쓰기격자"가 LLaMA서도 성립(단일-결정 상태가 통제면 아님);
  grid5만 양성 → 분산 다층 필요. **한계 공시**: 5층 밖 쓰기창은 탐지 불가(추출 필요, W5).

## 통계 (W1–W3와 동일)
클러스터: SM/LLaMA = 게임상태 풀 인덱스, IC 해당없음. 효과 = 절대치 + −G 갭 대비% +
w3pos(I_BA축) 대비% + 30-random 순위. 디코더 dose-monotonicity 보고.

## 사전등록 결과→본문 매핑
| 결과 | 본문 |
|---|---|
| saerdx 3지표 중 ≥1이 30-rnd 밖 | §4.1: "디코더 방향(들)도 옳은 창에서 특이 손잡이" |
| 전부 30-rnd 안 | §4.1: "디코더는 측정기; 인과는 레이어-국소 상태(행동축)" — M3+W3 종합 |
| w4L_iba_l12 양성 | §4 model-general: "해리·통제축 깊이-매칭으로 LLaMA 일반화" |
| w4L 전부 null | §4: "Gemma 단일-결정 통제면, LLaMA는 누적(L22_PLAN 트랙) 필요" |

## W5 (이번 미포함 — 사전 명기)
specificity 해결 후 confirmatory: BK_sm n=200 + 승격 디코더 n=200 (offset **500+**,
seed 제4셋, 평가-게임 제외 재추정 축). + LLaMA 신호 발견 시 전층 추출→정밀 창 스윕.

## 구현/제출 (W3와 동일 파이프라인)
새 축 = `paper_axes.py` 확장(타깃 열 파라미터화 + llama 빌더) → `assets/w4/` →
기존 steer 경로. preflight 게이트, HF 체크포인트 resume 동일. push→render→amlt run.
