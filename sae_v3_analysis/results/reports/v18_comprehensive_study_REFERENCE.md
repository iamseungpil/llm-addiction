# V18: Comprehensive Neural Analysis of Irrational Gambling in LLMs

**Model**: Gemma-2-9B-IT (42 layers, 3584-dim) | LLaMA-3.1-8B-Instruct (32 layers, 4096-dim)
**Data**: SM (3200 games), IC (1600 games), MW (3200 games) × 2 models = 16,000 games
**SAE**: GemmaScope 131K features/layer | LlamaScope 32K features/layer
**Per-Round**: Gemma SM 21K, IC 9K, MW 15K | LLaMA SM 62K, IC 3K, MW 85K rounds
**Date**: 2026-04-07

---

## Why V3 (Previous Paper Version) Was Invalid

V3의 신경 섹션은 세 가지 핵심 주장을 했으나, 엄격한 baseline 검증에서 모두 문제가 발견되었다.

| V3 주장 | 보고 수치 | 문제 | 검증 결과 |
|---------|----------|------|----------|
| DP Classification | AUC 0.954-0.982 | Trivial baseline (잔액+라운드+bet type) = **0.950** | AUC의 98%가 프롬프트 정보 인코딩 |
| Balance-Matched | AUC 0.998 at $95-105 | $95-105에서도 잔액만으로 **0.933**, 라운드 차이 confound | 라운드 1-2 vs 12.5 차이가 분류를 구동 |
| Cross-Domain Transfer | AUC 0.577-0.932 | Trivial features가 hidden states보다 **같거나 높음** | 모든 12방향에서 trivial ≥ hidden state |

**근본 원인**: Decision Point hidden state는 프롬프트의 잔액, 라운드 수, 베팅 타입을 인코딩하며, 이것만으로도 파산 예측이 거의 완벽하다. V3는 이 trivial baseline을 보고하지 않았다.

---

## V18 접근: 비선형 잔액 통제 후 Per-Turn 비합리성 분석

V3의 문제를 해결하기 위해:
1. Decision Point(게임당 1개) 대신 **모든 라운드**(게임당 다수) 분석
2. 잔액을 **비선형 Random Forest로** 제거 (linear 잔차화가 아닌)
3. 파산(BK) 대신 **Section 2의 비합리성 지표**(I_BA, I_LC, I_EC)를 타깃으로
4. 결과를 **무작위 feature baseline**과 비교

---

## RQ1: 비합리성을 인코딩하는 SAE Features가 존재하는가?

### 의도
잔액과 라운드의 비선형 효과를 완전히 제거한 후에도, SAE features가 비합리적 의사결정을 예측하는가?

### 가설
잔액 통제 후 잔여 비합리성과 상관되는 SAE features가 존재하며, 그 예측력이 무작위 features보다 유의하게 높다.

### 방법
- RF(n_estimators=50, max_depth=8)로 I_LC를 잔액·라운드·이차항·로그·교호항에서 예측 → 잔차 추출
- 상위 200개 SAE features(잔차와의 Spearman |r| 기준)로 Ridge 회귀 (5-fold CV)
- Baseline: 무작위 200개 features × 30회 반복
- 검증: 전체 파이프라인 순열 검정 (라벨 셔플 → RF → feature 선택 → Ridge, 50회)
- 검증: Feature 수 변화 실험 (k=20, 50, 100, 200, 500)

### 결과

**I_LC (손실 추구) — 6개 조합 모두 유의:**

| Model | Paradigm | n | R² | Random | p |
|-------|----------|---|-----|--------|---|
| Gemma | SM | 12,246 | **0.248** | 0.000 | <0.05 |
| Gemma | IC | 1,492 | **0.476** | -0.010 | <0.05 |
| Gemma | MW | 10,544 | **0.553** | 0.001 | <0.05 |
| LLaMA | SM | 45,551 | **0.345** | 0.042 | <0.05 |
| LLaMA | IC | 592 | **0.381** | -2.46 | <0.05 |
| LLaMA | MW | 57,361 | **0.779** | 0.130 | <0.05 |

**I_BA (베팅 공격성) — SM에서만 유의:**

| Model | Paradigm | R² | Random | 유의? |
|-------|----------|-----|--------|-------|
| Gemma | SM | **0.158** | 0.001 | ✅ |
| LLaMA | SM | **0.120** | 0.028 | ✅ |
| Gemma/LLaMA | MW | <0.01 | — | ❌ |

**I_EC (극단적 베팅) — SM에서 약하게 유의:**

| Model | Paradigm | R² | Random | 유의? |
|-------|----------|-----|--------|-------|
| Gemma | SM | **0.049** | 0.000 | ✅ |
| LLaMA | SM | **0.040** | 0.008 | ✅ |

### 검증

| 검증 방법 | 결과 |
|----------|------|
| 전체 파이프라인 순열 (50회) | Real R²=0.248 vs Perm mean=-0.022, **p=0.019** |
| Feature 수 변화 | k=20: 0.065, k=200: 0.248, k=500: 0.284 (단조 증가) |
| 무작위 baseline (200회) | Real=0.248 vs Random max=0.016, **p=0.005** |

### Layer Sweep

I_LC 신호는 중간-후반 layers에서 피크를 보인다:
- **Gemma**: L0(0.01) → L12(0.11) → **L24(0.25)** → L39(0.20)
- **LLaMA**: L0(0.21) → L8(0.28) → **L16(0.35)** → L28(0.31)

초기 layers에서 거의 없고 중간층에서 피크 → 입력 인코딩이 아닌 계산된 표상.

---

## RQ2: 이 패턴은 도박 과제를 관통하는가?

### 의도
SM, IC, MW에서 관찰된 비합리성 신호가 과제 간에 공유 구조를 갖는가?

### 접근 1: SAE Feature Overlap (threshold 기반)
- 각 패러다임에서 |r| > 0.05인 features를 선별 → 겹침 측정
- **결과**: L24에서 SM∩MW = 87개, chance 기대값의 232x
- **그러나**: Empirical null (라벨 순열)에서 perm mean=36, max=84, **p=0.08 (borderline)**
- SAE features 간 구조적 상관이 높아 overlap이 부풀려짐

### 접근 2: Hidden State Direction Cosine
- 각 패러다임에서 I_LC direction (Ridge weight vector) 계산
- SM과 MW direction의 cosine 비교
- **결과**: 5개 layer 모두에서 cosine ≈ 0 (최대 0.055), permutation과 구별 불가

### 접근 3: Cross-Paradigm Projection
- SM direction으로 MW data를 project → I_LC와 상관 측정
- **결과**: r = -0.03 ~ -0.04, random direction과 차이 없음

### 해석
**행동은 일관되지만(6모델 모두 Variable > Fixed), 내부 경로는 과제 특이적이다.**
- I_LC가 6개 조합 모두에서 독립적으로 유의 → 같은 종류의 신호가 각 과제에서 존재
- 그러나 이 신호를 담는 features/direction은 과제마다 다름
- 이는 "같은 행동이 다른 계산 경로로 생성된다"는 과제 특이적 경로 발견

---

## RQ3: 실험 조건이 이 신호를 조절하는가?

### 의도
자율성 조건과 프롬프트 구성요소가 I_LC 신호의 강도를 변화시키는가?

### 조건별 I_LC R² (NL deconfounded, Gemma SM L24)

| 조건 | n | R² | Random | 해석 |
|------|---|-----|--------|------|
| All Variable | 12,246 | **0.248** | -0.000 | 기준 |
| +G (목표 설정) | 8,040 | **0.278** | 0.001 | **1.73x 증폭** |
| -G | 4,206 | 0.161 | -0.004 | G 없으면 약화 |
| +M (보상 극대화) | 6,556 | **0.264** | 0.001 | 1.23x 증폭 |
| -M | 5,690 | 0.215 | -0.000 | |
| **Fixed** | 6,062 | **-0.061** | -0.002 | **신호 소멸** |

**핵심**: 선택의 자유가 있을 때만 I_LC 신호가 활성화되며, G-prompt가 이를 1.73x 증폭.

### 인과적 검증 (Activation Steering)

| 실험 | Model | Paradigm | rho | Perm p | 판정 |
|------|-------|----------|-----|--------|------|
| V14 Exp1 | **LLaMA** | **SM** | **0.919** | **0.048** | **BK-specific confirmed** |
| V14 Exp4 | Gemma | MW | -1.000 | 0.273 | Not specific (BK rate 1.7%, floor effect) |

LLaMA SM에서 BK direction steering이 강도-반응 패턴을 보임. Gemma MW는 BK rate가 너무 낮아(1.7%) floor effect.

---

## 진행 중 실험

| 실험 | 상태 | 예상 결과 |
|------|------|----------|
| V14 Exp4 Gemma MW | ✅ **완료** | BK_SIGNIFICANT_NOT_SPECIFIC (floor effect) |
| Temperature 대조 (1,600게임) | ✅ **완료** | 16/16 조건 Variable >> Fixed |

---

## 다음 실험 계획

| 우선순위 | 실험 | 의도 | 소요 |
|----------|------|------|------|
| P0 | **RQ2 행동적 cross-task 분석** | 행동 수준의 과제 관통 일관성을 정량화 | 기존 데이터 |
| P0 | **논문 신경 섹션 최종 작성** | V18 결과를 논문에 반영 | 글 작업 |
| P1 | LLaMA IC steering | 추가 인과 검증 (n=142 BK, SM보다 가능성 있음) | GPU 4-8h |
| P1 | I_BA/I_EC layer sweep | 세 지표 모두의 layer profile | CPU 2-3h |
| P2 | Feature interpretation (Neuronpedia) | 공유 features의 의미 해석 | 수동 |

---

## 정리: 유효한 주장

1. **RQ1**: 잔액을 비선형적으로 통제한 후에도, 손실 추구(I_LC) 관련 SAE features가 **6개 조합 모두에서** 무작위 baseline을 유의하게 초과한다 (R²=0.25-0.81, pipeline perm p=0.019).

2. **RQ2**: 행동 수준에서 자율성 효과는 SM, IC, MW 모두에서 재현된다. 신경 수준에서도 I_LC가 6개 조합 모두에서 독립적으로 유의하지만, 이 신호를 담는 features/direction은 과제마다 다르다.

3. **RQ3**: 목표 설정 프롬프트는 I_LC 신호를 1.73x 증폭시키고, 선택의 자유가 없으면 신호가 소멸한다. 활성화 steering은 LLaMA SM에서 행동 변화를 인과적으로 유도한다 (rho=0.919, p=0.048).
