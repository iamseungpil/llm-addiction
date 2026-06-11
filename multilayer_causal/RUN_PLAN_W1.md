# W1 실행 계획 — 의도 · 가설 · 검증 방법

**상태**: 구현·검증 완료, **제출 대기** (quota 보존을 위해 사용자 지시 시 제출)
**스펙**: `docs/superpowers/specs/2026-06-10-w1-section-interleaved-causal-design.md`
**총괄 질문**: 논문 §4가 *읽은* 상관(지표 readout · 공유축 · 조건변조)이 같은 축의 *인과*로 연결되는가.

이미 확정된 인과 결과 (W1 이전, n=200 confirmatory 통과):
- **L18–23 전체-상태 패치 = +G 행동 재현** (갭 회복 103.5%, CI [75%, 138%], vs −G perm p=1e-4) + **I_EC 4배 증폭** (Fisher p=5e-4).
- rank≤128 부분공간·G-프롬프트축(저용량)으로는 불가 — "읽기 넓고(L8–L30 R²>0) 쓰기 좁다(L18–23)".

---

## §4.1 묶음 — "Table 1이 읽는 지표 축이 컨트롤 손잡이인가"

### w1ro (readout-축 steering, 4 arms)
- **의도**: Table 1의 선형 독자(Ridge)가 사용하는 **바로 그 가중치 방향**으로 행동을 쓸 수 있는지 — "읽은 축에 대한 인과성"의 가장 문자적 검증.
- **가설** H1: α↑ ⇒ I_BA↑ (단조 용량-반응). H0: readout 축은 행동의 그림자(부산물 투영)라 무반응.
- **검증**: α∈{−2,0,+2,+4}×n=50, L18–23, per-layer 3%-norm 스케일. **1차 지표: Spearman ρ(α, I_BA[논문정의, bet만])**, 게임-클러스터 permutation p<0.05. 특이성: 동일 스케일 random-방향 8개 분포(기존) 95분위 밖. 파싱율<0.9 arm 무효(R5).
- **해석**: 성공 → "읽는 축=쓰는 축" (§4.1 최강 보강). 실패 → "readout은 그림자, 행동-대비 축(E3i, ρ=0.297 p=2e-7 기확보)이 실제 손잡이" — 어느 쪽이든 §4.1 마감문이 성립.

### w1lc (I_LC 손실 프로브, 7 arms, 페어드 2분기)
- **의도**: I_LC(손실 추격)를 **논문 공식 그대로** — max(0,(r_{t+1}−r_t)/r_t) — 측정하되 손실을 외생 주입해 교란 차단; 손실 추격이 인과 조작 가능한지.
- **가설** H1: I_LC축(+α)이 손실-특이적으로 r_{t+1}을 끌어올린다 (WIN 분기 대비 차분>0). H0: 손실 반응은 축 개입에 불변.
- **검증**: trial = 1차 결정 → 같은 결정에서 LOSS/WIN 두 분기(페어드, 시드 +1/+2) → 분기별 2차 결정. **1차 지표: 손실 특이성 = (r^L−r_t)/r_t − (r^W−r_t)/r_t 의 I_LC축 arm vs 자연 −G 차이**, 단측 클러스터 permutation. **게이트(R7): 자연 +G의 I_LC가 −G보다 높아야 프로브 유효** — 게이트 실패 시 개입 arm 해석 중단. all-in 1차 베팅은 LC 제외하고 "프로브-파산"으로 기록. arms: anchor −G/+G, patch L18–23, I_BA축+4, I_LC축+2/+4, random+4.
- **해석**: H1 → "세 지표 중 가장 임상적인 손실 추격까지 인과 손잡이 존재". 게이트 실패 → 프로브 무감도 보고, full-game으로만 측정 가능함을 한계에 명시.

### w1s (선택성 행렬, 4 arms)
- **의도**: 축이 지표별 **독립** 손잡이인지, 하나의 일반 위험축인지 — 3축(BA/LC/EC)×3지표 행렬의 비대각 채우기.
- **가설** H1(선택성): 대각 효과 ≥ 2× 비대각. H0(공통축): 전 지표 동반 이동.
- **검증**: I_LC축·I_EC축 {+2,+4} 단일-결정 steering, I_BA·I_EC 측정(LC 열은 w1lc에서). 판정은 random-방향 분포 기준 표준화 효과로 비교. 탐색적(R2) — 행렬 자체가 산출물.
- **해석**: 어느 쪽이든 §4.1 마감 단락의 결론 문장("indicator-specific handles" vs "one shared risk axis").

## §4.2 묶음 — "공유축의 인과 전이" (w1ic, 6 arms)

- **의도**: Table 2의 "SM→IC 예측 전이"의 인과 버전 — SM에서 만든 I_BA 행동축이 IC 결정을 움직이는가.
- **가설** H1: SM축(+α)이 IC 고위험 선택률을 올린다. H0: 전이는 읽기 수준에만 존재(E2의 rank-r 실패와 정합).
- **검증**: IC는 카탈로그(`v2_role_gemma`, §4 신경분석과 동일 코퍼스)의 **저장된 full_prompt 그대로** 사용, 파서는 v2_role 생성 러너에서 byte-동결(`parse_choice_fixed`, parity 테스트 강제). arms: 자연 anchor / SM-I_BA축 {+2,+4} / **IC-자체축 +4 (양성 대조)** / random ×2. **1차 지표: 고위험 선택률(choice∈{3,4} 비율, parse_ok만)**, 클러스터 permutation.
- **사전등록 해석 행렬**: SM✓·IC✓ → 공유 인과축 (Table 2 승격) / SM✗·IC✓ → 쓰기축은 과제-특이 (Table 2 정밀화) / 둘 다✗ → IC는 단일-결정 축 개입에 둔감 — §4.2엔 E2 결과만 서술.

## §4.3 묶음 — 국소화 마감 (w1b 2 + w1e 7 arms)

- **의도**: ① 논문이 *읽은* 5층 격자 {8,12,22,25,30}에 *쓰면* 어떻게 되나(통일성 브리지) ② S\* 경계를 ±1로.
- **가설**: ① H1(브리지): 5층 분산 패치는 연속 윈도보다 약하다 → "개수가 아니라 위치". ② H1(경계): 폭2 윈도 단독은 불충분하고 폭4 L18–21 또는 L20–23이 갭의 ≥50%를 회복.
- **검증**: w1b_patch5/w1b_steer5(I_BA축), w1e 폭2 ×5 + 폭4 ×2 (전부 patch). **1차 지표: +G 갭 회복률 + 클러스터-부트스트랩 CI** (R3 — "동등 p>0.05" 사용 금지).
- **해석**: §4.3 마감 단락 — "읽기 격자 ≠ 쓰기 격자; 인과 핵은 L_a–L_b".

---

## W2 — confirmatory wave (확정·제출: arms.yaml `phase: w2`, 8 arms)

W1 결과로 확정된 구성. 공통: **state_offset 300** (0–99 discovery / 100–299 e1c와 분리), **seed_base 2000042** (제3셋), 축은 전부 `assets/w2/` — **--n-eval 500 재추정판** (평가게임 363개 제외; W1 축과 cos: 행동축 ~0.99, readout ~0.80).

| arm | n | 의도 | 가설 | 1차 지표 |
|---|---|---|---|---|
| w2_anchor_minus/plus (probe) | 200×2 | 신규 오프셋 ±G 기준선 + R7 게이트 재검 | +G I_LC(프로브) > −G 재현 | 단측 클러스터 perm |
| w2e_1617 | 200 | **W1 헤드라인 확정**: L16–17 폭2가 갭 ~98% 회복 | 회복≥50%, vs −G p<0.05 | 갭회복+클러스터부트스트랩 CI |
| w2e_1821 | 200 | 핵 윈도 L18–21 확정 | 동상 | 동상 |
| w2lc_iba_p40 (probe) | 200 | **"한 축이 세 지표 전부" 마감**: W1에서 probe-n 23으로 미달했던 I_BA축의 I_LC 효과 | 손실 특이성 > −G (단측) | spec 차이, 단측 클러스터 perm |
| w2lc_ilc_p20 (probe) | 200 | W1 p=0.086 방향-일관 ILC축 확정 | 동상 | 동상 |
| w2rnd_p40 (probe) | 50 | 프로브 특이성 컨트롤 | random은 무효과 | — |
| w2ro5_p20 | 50 | **마지막 조합 마감**: 논문 읽기격자 {8,12,22,25,30} × readout축 | H0: 무반응 (read/write 비대칭 완결) | I_BA vs anchor |

**본문 승격 = confirmatory p<0.05 + random 대비 분리 + 파싱 ≥0.9 + 갭회복·CI 보고.** TOST는 클러스터 버전으로 교체 예정(현 retro_stats의 tost_p는 비클러스터 — 표기됨). 제출: `amlt run multilayer_causal/amlt/w2.rendered.yaml mlc-w2-<date> -y` (tarball push → render 선행, W1과 동일).

## 제출 절차 (지시 시 그대로 실행, 3단계)

```bash
# 1) 자산 포함 코드 tarball을 HF 최신으로
HF_TOKEN=$(python -c "from huggingface_hub import get_token; print(get_token())") \
  python multilayer_causal/scripts/push_code_to_hf.py
# 2) 토큰 렌더 (rendered는 gitignore)
bash multilayer_causal/amlt/render.sh
# 3) 제출 — 30 arms, 4×H100, ~반나절, preempt시 재제출만 하면 resume
amlt run multilayer_causal/amlt/w1.rendered.yaml mlc-w1-<date> -y
```

운영: HF `experiments/multilayer_causal/checkpoints/{phase}/{arm}.jsonl` 같은-경로 overwrite = 최신 resumable 상태(기존과 동일). 모니터링: amlt status + 체크포인트 크기. 완료 후 분석: `analyze summary` + `retro_stats` 패턴.

## 격리 확인

코드·자산·결과 전부 `multilayer_causal/`(+`docs/superpowers/`)에만 존재. 동결 원본(`prompts/states/hooks/checkpoint` 및 기존 실험 전체) 무수정 — 적대적 검증 에이전트가 git-status로 재확인(54 tests green). 브랜치 `feat/multilayer-causal`, HF 전용 prefix `experiments/multilayer_causal/`.
