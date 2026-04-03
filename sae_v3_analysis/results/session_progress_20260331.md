# 세션 진행 기록 및 후속 계획

> **날짜**: 2026-03-31  
> **프로젝트**: LLM Gambling Addiction — Korean Paper (LLM_Addiction_NMT_KOR)  
> **실험 환경**: Azure A100 80GB, conda env `llm-addiction`

---

## 1. 세션 개요

이번 세션의 핵심 목표는 한국어 논문(`LLM_Addiction_NMT_KOR`)의 Results 섹션을 RQ 중심으로 재구성하고, 그 과정에서 발견된 **overclaim 문제**를 해결하기 위해 V14 인과성 검증 실험을 설계·실행하는 것이었다.

### 주요 작업 흐름
```
논문 재구성 → 수치 검증(hallucination audit) → overclaim 발견 → V14 실험 설계 → 실험 실행 중
```

---

## 2. 완료된 작업

### 2.1 논문 구조 재편 (3.results.tex)

**Before**: 행동 결과 + SAE 분석이 분리된 구조  
**After**: 하나의 `\subsection` 아래 RQ1/RQ2/RQ3을 `\subsubsection`으로 통합

```
3.results.tex (245줄, 23페이지)
├── \subsection{행동 실험은 자율성 기반 위험 증가를 드러낸다}
│   ├── 발견 1: 가변 베팅 → 파산율 증폭
│   ├── 발견 2: 연속 추구 행동 증폭
│   ├── 발견 3: 목표 프롬프트 → 위험 선호 재구성
│   ├── 발견 4: 베팅 유연성의 독립적 효과
│   └── 발견 5: 언어적 인지 왜곡
└── \subsection{도박 행동의 신경 기반: 분류, 일반화, 인과적 제어} ← NEW (blue)
    ├── RQ1: 파산 예측 신경 패턴이 존재하는가?
    ├── RQ2: 이 패턴은 도박 도메인 간에 보편적인가?
    └── RQ3: 실험 조건은 이 패턴을 어떻게 조절하는가?
```

수정된 부분은 모두 `\textcolor{blue}{}`로 표시됨.

### 2.2 수치 검증 (Hallucination Audit)

JSON 원본 데이터와 논문 수치를 ~80개 항목에 대해 교차 검증.

| 항목 | 문제 | 조치 |
|------|------|------|
| LLaMA IC baseline BK | 0.000 → 0.020 | **수정 완료** |
| LLaMA MW baseline BK | 0.758 → 0.410 | **수정 완료** |
| Gemma MW baseline BK | 0.017 → 0.430 | **수정 완료** |
| Fig 6 L22 rho | 0.847 (n=100) vs 0.964 (n=200) | 불일치 확인, 미수정 |
| Table 2 Diff 열 산술 | IC 0.964-0.954=0.010인데 0.006 표기 | 확인됨, 미수정 |

### 2.3 Overclaim 발견 (핵심 이슈)

논문은 "4/6 model×task에서 유의한 steering"을 주장했으나, **random control 비교 시 direction specificity가 확인된 것은 1/6뿐**이었다.

| Model | Task | BK rho | BK p | Random도 유의? | Verdict |
|-------|------|--------|------|----------------|---------|
| LLaMA | SM | 0.964 | 0.00045 | 아니오 (0/3) | **BK_SPECIFIC_CONFIRMED** ✓ |
| LLaMA | IC | -0.991 | 0.00001 | 확인 불가 (n 부족) | NOT_SIGNIFICANT |
| LLaMA | MW | -0.955 | 0.00081 | **예** (1/3, p=0.014) | NOT_SIGNIFICANT |
| Gemma | SM | 0.512 | 0.240 | — | NS |
| Gemma | IC | NaN | NaN | — | NS |
| Gemma | MW | -1.000 | 0.000 | **예** (2/3 유의) | NOT_SIGNIFICANT |

**핵심 문제**: BK direction이 유의하더라도 random direction도 유의하면 **방향 특이성(direction specificity)**을 주장할 수 없다. 이것이 V14 실험의 동기.

### 2.4 기타 수정

- `0.abstract.tex`: SAE → activation-level direction steering 용어 변경
- `1.introduction.tex`: 4번째 기여 업데이트 (rho=0.964, p=0.00045)
- `4.discussion.tex`: activation steering 기반으로 업데이트
- `appendix_sae.tex`: 기존 SAE 112 features 내용을 appendix로 이동

---

## 3. V14 인과성 검증 실험

### 3.1 실험 설계 배경

V12에서 random control이 3개뿐이어서 permutation test의 검정력이 부족했다. V14는 이를 보완:

| 실험 | 모델 | 태스크 | n_games | n_random | 목적 |
|------|------|--------|---------|----------|------|
| Exp1 | LLaMA | SM | 100 | **20** | Gold standard 재검증 (perm_p = 1/21 가능) |
| Exp2a | LLaMA | IC | 100 | 10 | 음의 기울기 방향 특이성 검증 |
| Exp2b | LLaMA | MW | 100 | 10 | Random 유의성 문제 재검증 |
| Exp3 | LLaMA | Cross | 50 | 5×3 | MW→IC, MW→SM, IC→SM + 최초 random control |
| Exp4 | Gemma | MW | 100 | 10 | |rho|=1.0의 random confound 검증 |

### 3.2 실험 구현

- 메인 스크립트: `run_v14_experiments.py` (Exp1 순차 실행)
- 병렬 워커: `run_v14_parallel.py` (Exp2a, 2b, 3, 4 동시 실행)
- 래퍼: `run_v14_validation.sh`
- 게임 로직: `run_v12_all_steering.py`에서 `play_game`, `run_condition`, `compute_bk_direction` 재사용
- Steering 방식: `alpha × direction_tensor`를 L22 residual stream에 더하는 hook
- Alpha 범위: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] (7단계)

### 3.3 현재 진행 상황 (2026-03-31 19:10 UTC 기준)

6개 프로세스 병렬 실행 중 (GPU 79GB/80GB, 99% utilization):

#### Exp1 (PID 676138) — LLaMA SM, 20 randoms
```
BK Direction 완료: rho=0.919, p=0.003 ✓
  α=-2.0: 0.380 | α=-1.0: 0.380 | α=-0.5: 0.420 | α=0.0: 0.550
  α=+0.5: 0.480 | α=+1.0: 0.600 | α=+2.0: 0.570
Random 1/20 진행 중: α=-2.0: 0.590, α=-1.0: 0.470, α=-0.5 진행 중
```
→ BK Direction 유의 확인. 20개 random에서 permutation p-value 도출 대기.

#### Exp2a (PID 735769) — LLaMA IC, 10 randoms
```
BK Direction 진행 중: α=-2.0: BK=21/100=0.210 완료
```
→ α=-2.0에서 BK 21%로 baseline 대비 차이 관찰. 나머지 alpha 진행 중.

#### Exp2b (PID 735770) — LLaMA MW, 10 randoms
```
BK Direction 진행 중:
  α=-2.0: 0.600 | α=-1.0: 0.550 | α=-0.5: 0.400 | α=0.0: 진행 중 (50/100)
```
→ 음의 기울기 확인 (α 감소 → BK 증가). 나머지 alpha 진행 중.

#### Exp3 (PID 736618) — Cross-domain (MW→IC 먼저)
```
BK Direction (MW dir → IC task):
  α=-2.0: 0.040 | α=-1.0: 0.040
```
→ IC task에 MW direction 적용 시 BK rate 매우 낮음 (4%). 나머지 alpha 진행 중.

#### Exp4 (PID 736617) — Gemma MW, 10 randoms
```
모델 로딩 완료, BK Direction sweep 시작됨 (로그 미갱신)
```
→ Gemma는 LLaMA보다 느림. 로그 업데이트 대기.

### 3.4 예상 완료 시간

각 alpha sweep = ~12분 (LLaMA 100 games) ~ 20분 (Gemma 100 games)

| 실험 | 총 sweep 수 | 예상 소요 | 예상 완료 |
|------|------------|-----------|-----------|
| Exp1 | 7 + 20×7 = 147 | ~30시간 | 4/1 21:00 UTC |
| Exp2a | 7 + 10×7 = 77 | ~16시간 | 4/1 08:00 UTC |
| Exp2b | 7 + 10×7 = 77 | ~16시간 | 4/1 08:00 UTC |
| Exp3 | 3×(7+5×7) = 126 | ~26시간 | 4/1 18:00 UTC |
| Exp4 | 7 + 10×7 = 77 | ~25시간 | 4/1 17:00 UTC |

**가장 늦은 완료: Exp1 (4/1 21:00 UTC 추정)**

---

## 4. 이전 분석 보고서 체인 (V3→V13)

| 버전 | 날짜 | 핵심 내용 | 파일 |
|------|------|-----------|------|
| V3-V8 | 3/19 | SAE 초기 분석 (Gemma 중심) | results/sae_v3~v8_*.pdf |
| V9 | 3/19 | Cross-model 비교 시작 | results/sae_v9_*.pdf |
| V10 | 3/24 | Symmetric analysis — 1,334 universal BK neurons | results/sae_v10_*.pdf |
| V11 | 3/25 | Cross-model 정제 | results/sae_v11_*.pdf |
| **V12** | **3/30** | **6 within-domain + 6 cross-domain steering 완료** | **results/sae_v12_*.pdf (1.9MB)** |
| **V13** | **3/31** | **RQ1-RQ3 종합 보고서 (EN/KO)** | **results/sae_v13_*.pdf (1.1MB)** |

### V12에서 확인된 6가지 주요 결론
1. Direction specificity (LLaMA SM rho=0.964, 0/3 random significant)
2. Multi-layer distribution (L22+L25+L30 combined = 2.5× amplification)
3. Cross-task generalization (IC/MW |rho| ≥ 0.955)
4. Cross-model generalization (Gemma MW |rho| = 1.0)
5. Boundary conditions (Gemma SM/IC fail — BK variation 부족)
6. Cross-domain causal transfer (3/6 significant, MW hub paradigm)

### V14에서 재검증 대상
- 결론 1: ✓ (Exp1 — 20 random으로 강화)
- 결론 3, 4: Exp2a, 2b, 4 — random control로 direction specificity 확인
- 결론 6: Exp3 — 최초로 cross-domain에 random control 추가

---

## 5. 현재 논문 파일 상태

| 파일 | 줄 수 | 상태 |
|------|-------|------|
| `0.abstract.tex` | 3 | ✓ steering 키워드 반영 |
| `1.introduction.tex` | 17 | ⚠ overclaim 수정 필요 ("도메인 불변 특성") |
| `2.defining_addiction.tex` | 24 | ✓ |
| `3.results.tex` | 245 | ⚠ V14 결과 반영 후 수정 필요 |
| `4.discussion.tex` | 12 | ⚠ overclaim 수정 필요 |
| `5.methods.tex` | 222 | ✓ |
| `appendix_sae.tex` | 31 | ✓ |

### 알려진 논문 이슈 (V14 완료 후 수정 예정)
1. **"4/6 유의한 steering" 주장** → V14 결과에 따라 수정
2. **방향 특이성 주장** → permutation p-value 기반으로 재작성
3. **Fig 6 L22 rho 불일치** (0.847 vs 0.964) → 그림 재생성 필요
4. **Table 2 Diff 산술 오류** → 수정 필요
5. **6-panel dose-response figure** → 6개 model×task 모두에 대해 생성 필요
6. **Cross-domain steering random control 부재** → V14 Exp3로 해결
7. **RQ2/RQ3 콘텐츠 복원** → V3-V13 보고서에서 누락된 유효 내용 복원

---

## 6. 데이터 파일 위치

### 실험 결과 JSON
```
/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json/
├── v12_n200_20260327_030745.json          # LLaMA SM gold standard
├── v12_llama_ic_L22_20260329_022313.json   # LLaMA IC
├── v12_llama_mw_L22_20260329_072818.json   # LLaMA MW
├── v12_gemma_mw_L22_20260328_205618.json   # Gemma MW
├── v12_crossdomain_steering.json           # Cross-domain (random control 없음)
├── v12_sign_reversal_analysis.json         # Sign reversal model
└── v14_*.json                              # ← V14 결과 (실험 완료 시 생성)
```

### Hidden States (NPZ)
```
/home/v-seungplee/data/llm-addiction/sae_features_v3/
├── llama_ic_hidden_states.npz   # (1600, 5, 4096)
├── llama_sm_hidden_states.npz   # (3200, 5, 4096)
├── llama_mw_hidden_states.npz   # (3200, 5, 4096)
├── gemma_ic_hidden_states.npz   # (1600, 5, 3584)
├── gemma_sm_hidden_states.npz   # (3200, 5, 3584)
└── gemma_mw_hidden_states.npz   # (3200, 5, 3584)
```

### 실험 로그
```
/home/v-seungplee/llm-addiction/sae_v3_analysis/results/
├── v14_log.txt          # Exp1 (main sequential)
├── v14_exp2a_log.txt    # LLaMA IC parallel
├── v14_exp2b_log.txt    # LLaMA MW parallel
├── v14_exp3_log.txt     # Cross-domain parallel
└── v14_exp4_log.txt     # Gemma MW parallel
```

### 실험 코드
```
/home/v-seungplee/llm-addiction/sae_v3_analysis/src/
├── run_v14_experiments.py    # Exp1 메인 (순차)
├── run_v14_parallel.py       # Exp2a/2b/3/4 병렬 워커
├── run_v14_validation.sh     # 래퍼 스크립트
└── run_v12_all_steering.py   # 게임 로직 원본 (play_game 등)
```

---

## 7. 후속 계획 (V14 완료 후)

### Phase 1: V14 결과 분석 및 레포트 (실험 완료 직후)

1. **결과 수집**: `results/json/v14_*.json` 전체 읽기
2. **Verdict 분류**: 각 실험별 BK_SPECIFIC_CONFIRMED / NOT_SIGNIFICANT 판정
3. **V14 레포트 작성**: 
   - 각 실험의 BK rho, permutation p-value, random 유의 비율
   - Overclaim이 없는 honest assessment
   - 논문 반영 계획 포함
4. **사용자 승인 대기**

### Phase 2: 논문 수정 (승인 후)

5. **3.results.tex RQ1 수정**:
   - V14 결과 기반으로 steering 테이블 업데이트
   - Direction specificity 주장을 permutation p-value 기반으로 재작성
   - 유의하지 않은 경우: "steering 효과는 관측되었으나 방향 특이성은 확인되지 않았다" 등 정직한 서술
6. **6-panel dose-response figure 생성**:
   - 6개 model×task 모두 (SM, IC, MW × LLaMA, Gemma)
   - V14 n=100 데이터 사용
   - Random control과 함께 표시
7. **1.introduction.tex / 4.discussion.tex overclaim 수정**:
   - "도메인 불변 특성" → V14 결과에 따라 조건부 서술로 변경
   - Direction specificity가 확인된 case만 강한 주장
8. **Fig 6 L22 rho 불일치 수정** (0.847 → 0.964 또는 그림 재생성)
9. **Table 2 Diff 산술 오류 수정**

### Phase 3: RQ2/RQ3 콘텐츠 보강

10. **V3-V13 보고서 전수 검토**: RQ2 (cross-domain)와 RQ3 (실험 조건)에서 누락된 유효 분석 확인
    - MW hub 설명의 충분성
    - G-prompt 패러다임 의존성 분석
    - Autonomy paradox 상세화
11. **Appendix 이동 또는 본문 복원 결정**

### Phase 4: 최종 검수

12. **Critic loop**: 전체 논문 재검수 (hallucination, overclaim, 논리 일관성)
13. **LaTeX 컴파일 확인**: 페이지 수, 그림 배치, 참고문헌
14. **GitHub push**: https://github.com/iamseungpil/LLM_Addiction_NMT_KOR

---

## 8. 핵심 판단 포인트 (V14 결과에 따른 분기)

### 시나리오 A: 대부분 BK_SPECIFIC_CONFIRMED
→ 현재 논문 주장을 유지하되, permutation p-value로 보강. 가장 낙관적.

### 시나리오 B: LLaMA SM만 CONFIRMED, 나머지 NOT_SPECIFIC
→ 논문 톤을 대폭 조정. "SM에서만 방향 특이적 인과성 확인, 나머지는 관측적 증거"로 서술 변경. Contribution을 "방법론 제안 + 1개 강한 증거 + 5개 탐색적 결과"로 재구성.

### 시나리오 C: 모두 NOT_SIGNIFICANT
→ Steering 섹션을 탐색적 분석으로 완전 재작성. Contribution 4를 삭제하거나 "방법론적 기여"로 축소.

**현재 예상**: 시나리오 B가 가장 유력 (Exp1 BK Direction rho=0.919로 유의, random은 확인 중).

---

## 9. 프로세스 모니터링 명령어

```bash
# 프로세스 상태
ps aux | grep run_v14 | grep -v grep

# GPU 상태
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# 각 실험 최신 로그
tail -5 /home/v-seungplee/llm-addiction/sae_v3_analysis/results/v14_log.txt
tail -5 /home/v-seungplee/llm-addiction/sae_v3_analysis/results/v14_exp{2a,2b,3,4}_log.txt

# 완료된 결과 확인
ls -lt /home/v-seungplee/llm-addiction/sae_v3_analysis/results/json/v14_*.json
```
