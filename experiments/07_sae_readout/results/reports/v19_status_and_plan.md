# V19: Status Report — What Worked, What Failed, and What's Next

**Date**: 2026-04-07
**Models**: Gemma-2-9B-IT | LLaMA-3.1-8B-Instruct
**Data**: SM 3200 × 2, IC 1600 (4 constraints × 400) × 2, MW 3200 × 2 = 19,200 games (open-source)
**Per-Round SAE**: Gemma 42L × 131K | LLaMA 32L × 32K

---

## 1. Why V3 Paper Was Invalid

V3의 신경 섹션은 세 가지 주장을 했으나, baseline 검증에서 모두 문제가 발견됨.

| V3 주장 | 수치 | 실패 이유 |
|---------|------|----------|
| DP Classification AUC 0.97 | Trivial baseline (잔액+라운드+bet type) = **0.950** | AUC의 98%가 프롬프트 정보 인코딩 |
| Balance-Matched AUC 0.998 | $90-110에서도 잔액만으로 0.933 | 라운드 수 차이(BK: R1.6, VS: R12.5)가 confound |
| Cross-Domain Transfer 0.58-0.93 | Trivial features가 hidden state와 **동등 또는 우월** | 잔액 패턴이 과제 간에 유사해서 전이되는 것 |
| Feature Overlap 270x chance | Empirical null (라벨 순열) **p=0.58** | SAE features 간 구조적 상관이 높음 |
| Linear Residual R²=0.34 | Balance features와 **동등** (R²=0.53 vs 0.54) | 비선형 잔액 인코딩을 linear로 못 잡음 |

**근본 원인**: Decision Point hidden state는 프롬프트의 잔액·라운드·베팅 타입을 인코딩하며, V3는 이 trivial baseline을 보고하지 않았음.

---

## 2. What IS Valid (Current Findings)

### RQ1: 비합리성을 인코딩하는 SAE Features가 존재하는가?

**방법**: 잔액+라운드의 비선형 효과를 Random Forest로 제거 → 잔여 분산을 SAE features로 예측 → 무작위 features와 비교

**결과**: I_LC (손실 추구)가 **6개 모델×패러다임 조합 모두에서 유의**

| Model | Task | n | R² | Random | p |
|-------|------|---|-----|--------|---|
| Gemma | SM | 12,246 | **0.248** | 0.000 | <0.05 |
| Gemma | IC | 1,492 | **0.476** | -0.010 | <0.05 |
| Gemma | MW | 10,544 | **0.553** | 0.001 | <0.05 |
| LLaMA | SM | 45,551 | **0.345** | 0.042 | <0.05 |
| LLaMA | IC | 592 | **0.381** | -2.46 | <0.05 |
| LLaMA | MW | 57,361 | **0.779** | 0.130 | <0.05 |

I_BA(SM only R²=0.12-0.16), I_EC(SM only R²=0.04-0.05)도 SM에서 유의.

**검증**: Pipeline permutation p=0.019, feature ablation k=20→500 단조 증가, random baseline 200회 p=0.005.

**Layer profile**: I_LC 중간-후반 layer에서 피크 (Gemma L24, LLaMA L16). 초기 layer < 0.03.

### RQ3: 실험 조건이 이 신호를 조절하는가?

| 조건 | R² | 비교 | 해석 |
|------|-----|------|------|
| Gemma SM +G | **0.278** | vs -G: 0.161 | **1.73x 증폭** |
| Gemma SM +M | **0.264** | vs -M: 0.215 | 1.23x 증폭 |
| 양 모델 Fixed | **≈ 0** | vs Variable | 선택 자유 없으면 소멸 |
| LLaMA SM steering | **rho=0.919** | perm p=0.048 | 탐색적 인과 증거 |

---

## 3. Complete Behavioral Matrix (Corrected)

| Model | Task | Var BK% | Fix BK% | ΔBK | Direction |
|-------|------|---------|---------|-----|-----------|
| Gemma | SM | 5.4% | 0.0% | +5.4% | **Var > Fix** ✅ |
| Gemma | IC (all c) | 1.8% | 19.8% | -18.0% | **Fix >> Var** (autonomy paradox) |
| Gemma | MW | 0.2% | 3.1% | -2.9% | Fix > Var |
| LLaMA | SM | 72.3% | 0.4% | +71.9% | **Var >>> Fix** ✅✅ |
| LLaMA | IC (all c) | 9.6% | 8.1% | +1.5% | Var ≈ Fix |
| LLaMA | MW | 79.4% | 72.2% | +7.2% | **Var > Fix** ✅ |

**패턴**: SM에서 양 모델 일관(Var > Fix). IC에서 Gemma는 역방향(autonomy paradox). MW에서 LLaMA는 양 방향 BK가 매우 높음(72-79%).

**Gemma Autonomy Paradox**: IC/MW에서 Variable이 더 보수적으로 행동 → Fixed가 강제 베팅으로 더 많이 파산. V8에서 이미 보고됨: "BK feature가 인코딩하는 것은 선택의 위험성이 아니라 재정적 궤적의 위험성."

---

## 4. What Failed for RQ2 (Cross-Task)

| 접근 | 결과 | 왜 실패했나 |
|------|------|------------|
| SAE feature overlap (top-k) | Empirical null p=0.58 | SAE 구조적 상관이 높음 |
| SAE feature overlap (threshold) | 87개, p=0.08 borderline | 비슷한 이유 |
| Hidden state direction cosine | 모든 layer cos≈0, p>0.08 | 방향이 과제 특이적 |
| Cross-paradigm projection | r=-0.03, random과 차이 없음 | SM direction이 MW에서 무의미 |
| Mediation (split-half) | CI includes 0 | Feature set이 split에 민감 |

**해석**: 행동적으로 비슷한 결과(손실 추구)가 나오지만, 모델 내부에서는 **다른 computational pathway로 구현**됨. 이것은 "수렴 진화"에 비유할 수 있음 — 같은 기능이 다른 구조로 달성.

---

## 5. RQ2를 어떻게 채울 것인가?

### 현재 가능한 증거:

**A. 행동적 Cross-Task 일관성 (6모델, 본문)**
- 원래 논문의 25,600 게임에서 6모델 모두 Variable > Fixed (SM)
- 이것은 이미 행동 섹션에 있음

**B. I_LC가 6/6 독립적으로 유의 (신경, 2모델)**
- SM, IC, MW 각각에서 I_LC가 잔액 통제 후 유의
- 같은 features/direction은 아니지만 **같은 종류의 신호**가 독립적으로 존재
- "수렴적 증거" — 과제마다 독립적으로 loss-chasing representation이 형성됨

**C. 과제 구조에 따른 체계적 차이 (새로 분석 가능)**
- SM (이항 결정): Variable > Fixed 일관
- IC (다지선다 + constraint): Gemma autonomy paradox
- MW (연속 스핀): LLaMA 양방향 높은 BK
- 이 차이 자체가 "과제 구조가 비합리성 표상을 조절한다"는 RQ3 확장

### 제안하는 RQ2 프레이밍:

> "비합리적 행동의 공통 특성이 과제를 관통하여 관찰되는가?"
> → **행동적으로 Yes**: I_LC(손실 추구)가 모든 과제에서 Variable에서 증폭되는 경향.
>    다만 IC에서는 과제 구조에 의해 방향이 반전될 수 있음(autonomy paradox).
> → **신경적으로 Partial**: I_LC 관련 SAE features가 6개 조합 모두에서 독립적으로 존재하지만,
>    features 자체나 hidden state direction은 과제마다 다름.

---

## 6. Next Steps

| 우선순위 | 작업 | 내용 |
|----------|------|------|
| **P0** | RQ2 행동 분석 formalize | I_LC per-turn을 모든 model×paradigm에서 condition별 비교 |
| **P0** | 논문 신경 섹션 업데이트 | V19 결과 반영 (IC 전체 데이터 포함) |
| **P1** | LLaMA IC steering (20 randoms) | 추가 인과 검증 (V14 Exp2a가 borderline이었음) |
| **P1** | I_BA/I_EC layer sweep | 세 지표의 완전한 layer profile |
| **P2** | Gemma autonomy paradox 분석 | IC/MW에서 Fixed > Variable인 신경적 이유 |
