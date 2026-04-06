# V16 Neural Analysis Plan: Per-Turn Irrationality Probe

**날짜:** 2026-04-06 (updated)  
**상태:** Codex 토론 수렴 완료, 실행 중

---

## 배경

Decision-point hidden state의 BK 분류(AUC 0.97)는 대부분 trivial features(잔액, 라운드, bet_type) 인코딩이었음.
Residual AUC 0.654(p=0.005)만 genuine 신호. Early-round BK 예측은 prompt-combo leakage(p=0.99).

**핵심 전환:** BK(게임 결과) 대신 **per-turn 비합리성 지표**를 타깃으로 변경.
"이 턴의 선택이 비합리적인가?"를 hidden state에서 예측.

---

## 데이터

### Gemma (핵심 분석 대상 — 전체 layer, 전체 round 존재)

| 패러다임 | shape | metadata | 상태 |
|----------|-------|----------|------|
| SM | (21,421 × 42 × 3,584) | ✅ 복원 완료 (21,421 매칭) | 분석 진행 중 |
| IC | (9,119 × 42 × 3,584) | ❌ 복원 필요 | 대기 |
| MW | (15,749 × 42 × 3,584) | ❌ 복원 필요 | 대기 |

### LLaMA (보조 — 부분적 데이터)

| 패러다임 | shape | 상태 |
|----------|-------|------|
| SM DP | (3,200 × 32 × 4,096) all-layers | DP만, Round 1-3도 있음 |
| IC | (3,291 × 32 × 4,096) phase_a on HF | 미다운로드 |
| IC/MW DP | (1,600/3,200 × 5 × 4,096) | 5 layers만 |

---

## Per-Turn 비합리성 지표 (Section 2의 I_BA, I_LC, I_EC에 대응)

| 지표 | 타입 | 정의 | Section 2 대응 |
|------|------|------|---------------|
| **bet_ratio** | 연속 [0,1] | bet/balance | I_BA의 per-turn 버전 |
| **is_extreme** | 이진 | bet_ratio ≥ 0.5 | I_EC의 per-turn 버전 |
| **is_loss_chasing** | 이진 | 손실 후 bet_ratio 증가 | I_LC의 per-turn 버전 |

핵심: 음의 기대값 게임에서 합리적 행동 = 즉시 중단. 
따라서 "계속 베팅" 자체가 비합리적이며, bet_ratio가 높을수록 더 비합리적.

---

## RQ별 분석 계획

### RQ1: 비합리적 의사결정의 신경 패턴이 존재하는가?

**의도:** 각 턴에서 모델이 비합리적 선택을 할 때 hidden state가 체계적으로 다른지 확인.

**가설:**
- H1a: Hidden state가 per-turn is_extreme을 trivial features(잔액, 라운드) 이상으로 예측한다 (residual AUC > 0.5).
- H1b: 이 신호는 특정 layer 범위에서 피크를 보인다 (middle-to-late layers).
- H1c: Within-round 분석(같은 라운드만 비교)에서도 유의하다.

**검증 방법:**
1. **42-layer sweep (Gemma SM):** 각 layer에서 logistic probe (is_extreme 예측).
   - AUC_full: hidden state만
   - AUC_trivial: balance + round만 (baseline)
   - AUC_residual: trivial 제거 후 hidden state (핵심 지표)
   - 3-fold CV, Variable betting 라운드만
2. **Within-round probe:** Round 1, 2, 3 각각에서 separately probe (context length 통제).
3. **Permutation test:** 최적 layer에서 100회 within-round-bin 순열 (p < 0.05 기준).

**성공 기준:** Residual AUC > 0.55 at 최적 layer, permutation p < 0.05.

### RQ2: 이 패턴이 도박 도메인 간에 공유되는가?

**의도:** SM에서 학습한 irrationality probe가 IC/MW에서도 작동하는지.

**가설:**
- H2: SM의 최적 layer에서 학습한 residual probe가 IC/MW에서 AUC > 0.5 달성.

**검증 방법:**
1. IC, MW metadata 복원 (SM과 동일한 방법).
2. SM에서 학습한 probe를 IC/MW에 적용 (zero-shot transfer).
3. Trivial-feature transfer baseline과 비교.
4. 전체 6방향 transfer matrix (SM↔IC↔MW).

**성공 기준:** Residual transfer AUC > trivial transfer AUC in ≥ 4/6 방향.

### RQ3: 자율성 조건이 이 패턴을 조절하는가?

**의도:** Variable vs Fixed, G-prompt vs non-G에서 probe 성능이 다른지.

**가설:**
- H3a: Variable 조건에서 irrationality probe AUC가 Fixed보다 높다 (더 다양한 행동 → 더 많은 신호).
- H3b: G-prompt 유무에 따라 irrationality direction의 mean projection이 이동한다.

**검증 방법:**
1. Variable vs Fixed 각각에서 within-condition probe AUC 비교.
2. Irrationality direction (RQ1에서 추출)에 대한 mean projection: 조건별 t-test.
3. Cross-condition transfer: Variable에서 학습 → Fixed에서 테스트 (vice versa).

**성공 기준:** 조건 간 AUC 차이 또는 projection 차이가 유의 (p < 0.05).

---

## 실행 순서

| 단계 | 작업 | 소요 | 상태 |
|------|------|------|------|
| 1 | Gemma SM metadata 복원 | 5분 | ✅ 완료 |
| **2** | **Gemma SM 42-layer sweep (is_extreme)** | ~30분 | 🔄 실행 중 |
| 3 | Gemma IC/MW metadata 복원 | 10분 | 대기 |
| 4 | Within-round probe (SM) | 20분 | 대기 |
| 5 | Permutation test (최적 layer) | 30분 | 대기 |
| 6 | Cross-paradigm transfer (Gemma) | 30분 | 대기 |
| 7 | Condition modulation (RQ3) | 20분 | 대기 |
| 8 | LLaMA 보조 분석 (DP all-layers) | 30분 | 대기 |
| 9 | Figure 생성 + 논문 반영 | 1시간 | 대기 |

---

## V12 Steering은 독립적으로 유지

V12/V14의 activation steering (LLaMA SM rho=0.964, perm p=0.048)은 
probe 분석과 독립적인 인과 증거. Decision-point 방향 조작이 행동을 바꾸는 것은 
trivial feature 문제와 무관. 부록에서 유지.
