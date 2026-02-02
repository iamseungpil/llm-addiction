# Figure 1 히트맵 해석 가이드
## SAE Feature Activation Patterns 이해하기

**작성일**: 2026-02-02
**대상 Figure**: `fig1_improved_heatmap_llama.png`, `fig1_improved_heatmap_gemma.png`
**목적**: 히트맵을 보고 LLM의 신경 표상 전략을 이해하기

---

## 목차

1. [Figure 1이 보여주는 것](#1-figure-1이-보여주는-것)
2. [핵심 개념](#2-핵심-개념)
3. [통계 분석 방법](#3-통계-분석-방법)
4. [Z-score 정규화](#4-z-score-정규화)
5. [히트맵 읽는 법](#5-히트맵-읽는-법)
6. [LLaMA vs Gemma 패턴](#6-llama-vs-gemma-패턴)
7. [주의사항](#7-주의사항)
8. [실전 해석 예시](#8-실전-해석-예시)

---

## 1. Figure 1이 보여주는 것

### 1.1 한 문장 요약

> **"가장 중요한 SAE features 20개가 4가지 조건에서 어떻게 다르게 반응하는지 보여주는 히트맵"**

### 1.2 Figure 구조

```
┌─────────────────────────────────────────────────┐
│  SAE Feature Activation Patterns (LLaMA-3.1-8B) │
│                                                  │
│     Variable Betting  |  Fixed Betting          │
│     Bankrupt  Safe    |  Bankrupt  Safe         │
│     ──────────────────┼──────────────────       │
│ L14-12265  🔵  🔵    |  🔴  🔴   η²=0.850      │
│ L15-23456  🔵  🔵    |  🔴  🔴   η²=0.723      │
│ L16-34567  🔵  🔵    |  🔴  🔴   η²=0.689      │
│     ⋮                                            │
│ (20개 feature)                                   │
│                                                  │
│ 색상: 빨간색=높음, 파란색=낮음, 흰색=평균       │
└─────────────────────────────────────────────────┘
```

### 1.3 연구 질문

- **Q1**: LLM의 "뇌세포"(SAE features)가 무엇을 인코딩하는가?
- **Q2**: **베팅 조건**(Variable vs Fixed)을 인코딩하는가?
- **Q3**: **게임 결과**(Bankrupt vs Safe)를 인코딩하는가?
- **Q4**: 모델마다 다른가?

**패턴 판별**:
- 점선(Variable|Fixed 경계) 기준 색 변화 → **Bet Type 인코딩**
- 열(Bankrupt vs Safe) 기준 색 변화 → **Outcome 인코딩**

---

## 2. 핵심 개념

### 2.1 SAE Feature

**정의**: Sparse Autoencoder가 학습한 신경망의 해석 가능한 방향

**비유**:
```
신경망 = 4096차원 공간
SAE Feature = 의미 있는 축 (32,768개)
Feature 활성화 = 특정 축으로 얼마나 투영되는가
```

**예시**:
```
Feature L14-12265:
- Variable 게임: 0.0083 (낮음)
- Fixed 게임: 0.2172 (높음)
→ "Fixed Betting 조건"을 감지
```

### 2.2 4가지 조건

**실험 설계**:
```
              | Outcome
              | Bankrupt  Safe
──────────────┼─────────────────
Bet    Variable│   VB      VS
Type   Fixed   │   FB      FS
```

**샘플 크기 (LLaMA)**:
```
VB: 108 games (3.4%)   ← 작음
VS: 1,492 games (46.6%)
FB: 42 games (1.3%)    ← 매우 작음
FS: 1,558 games (48.7%)
Total: 3,200 games
```

### 2.3 Feature 활성화

**정의**: 특정 게임에서 SAE feature가 얼마나 "켜지는가"

**데이터 예시**:
```
Layer 14, Feature 12265:
- Game 1 (VB): 0.012
- Game 2 (VS): 0.000
- Game 3 (FB): 0.245
- ...
- Game 3200 (FS): 0.289

그룹 평균:
VB: 0.0083
VS: 0.0020
FB: 0.2172
FS: 0.2562
```

### 2.4 Top 20 선택 기준

**왜 20개인가?**
- 전체: 7 layers × 32,768 features = 229,376개
- 너무 많으면 패턴 파악 어려움
- **η² (효과 크기) 상위 20개 = 가장 중요한 features**

**선택 기준**:
- p-value (존재 여부) ❌ → 샘플 3200개면 거의 다 유의
- **η² (중요도)** ✅ → 실질적으로 중요한 feature

---

## 3. 통계 분석 방법

### 3.1 One-Way ANOVA

**목적**: 4개 그룹의 평균이 통계적으로 다른가?

**모형**:
```
Y_ij = μ + α_i + ε_ij

여기서:
- Y_ij: i번째 그룹의 j번째 관측치
- μ: 전체 평균 (grand mean)
- α_i: i번째 그룹의 효과 (Σα_i = 0)
- ε_ij: 잔차
```

**귀무가설**:
```
H₀: μ_VB = μ_VS = μ_FB = μ_FS
   (4개 그룹의 평균이 모두 같다)
```

**검정 통계량**:
```
F = MS_between / MS_within
  = (그룹 간 분산) / (그룹 내 분산)
```

### 3.2 제곱합 (Sum of Squares)

**총 제곱합 (SS_total)**:
```
SS_total = Σ (Y_ij - Ȳ)²

Ȳ = 전체 평균
```

**그룹 간 제곱합 (SS_between)**:
```
SS_between = Σ n_i (Ȳ_i - Ȳ)²

예시 (Feature L14-12265):
Ȳ = (108×0.0083 + 1492×0.0020 + 42×0.2172 + 1558×0.2562) / 3200
  = 0.1264

SS_between = 108×(0.0083-0.1264)² + 1492×(0.0020-0.1264)² +
             42×(0.2172-0.1264)² + 1558×(0.2562-0.1264)²
           = 1.504 + 23.023 + 0.347 + 26.326
           = 51.200
```

**그룹 내 제곱합 (SS_within)**:
```
SS_within = SS_total - SS_between
```

### 3.3 Eta-squared (η²)

**정의**:
```
η² = SS_between / SS_total
   = "그룹 간 차이가 총 분산의 몇 %를 설명하는가"
```

**계산 예시**:
```
Feature L14-12265:
SS_between = 51.200
SS_total = 60.235

η² = 51.200 / 60.235 = 0.850

→ 그룹 차이가 총 분산의 85%를 설명!
```

**해석 기준**:
```
η² < 0.01  : Negligible (무시 가능)
η² = 0.01  : Small (작음)
η² = 0.06  : Medium (중간)
η² = 0.14  : Large (큼)
η² ≥ 0.50  : Very large (매우 큼) ← L14-12265
```

### 3.4 FDR 보정

**문제**: 229,376번의 검정 → Type I 오류율 폭증

**Benjamini-Hochberg 절차**:
```
1. p-values를 오름차순 정렬: p₁ ≤ p₂ ≤ ... ≤ p_m

2. 각 i에 대해 확인:
   p_i ≤ (i/m) × α
   (α = 0.05, m = 229,376)

3. 조건을 만족하는 최대 i를 찾음

4. p₁부터 p_i까지 유의하다고 판정
```

**효과**:
```
보정 전: 95%가 유의 (p < 0.05)
보정 후: 15%가 유의 (q < 0.05)
→ False positives 대폭 감소
```

---

## 4. Z-score 정규화

### 4.1 왜 필요한가?

**문제 1: 절댓값 차이**
```
Feature A: [0.01, 0.00, 0.20, 0.25]  (범위: 0.25)
Feature B: [0.50, 0.30, 2.00, 2.50]  (범위: 2.20)

→ Feature B만 눈에 띔
→ 하지만 상대적 패턴은 같을 수 있음!
```

**문제 2: SAE의 Sparse 특성**
```
대부분 features:
- 90% 게임에서 0
- 10% 게임에서만 활성화
- 활성화 값도 천차만별 (0.01~10.0)
```

**해결책**: 각 feature 내에서 **상대적 차이**만 표현

### 4.2 Z-score 계산

**공식**:
```
z_ij = (x_ij - μ_i) / σ_i

여기서:
- x_ij: i번째 feature의 j번째 조건 평균
- μ_i: i번째 feature의 4개 조건 평균
- σ_i: i번째 feature의 4개 조건 표준편차
```

**실제 계산 예시** (Feature L14-12265):
```
Raw values:
VB: 0.0083, VS: 0.0020, FB: 0.2172, FS: 0.2562

Step 1: 평균
μ = (0.0083 + 0.0020 + 0.2172 + 0.2562) / 4 = 0.1209

Step 2: 표준편차
σ = sqrt(Σ(x - μ)² / 4) = 0.1167

Step 3: Z-score
z_VB = (0.0083 - 0.1209) / 0.1167 = -0.96 (파란색)
z_VS = (0.0020 - 0.1209) / 0.1167 = -1.02 (파란색)
z_FB = (0.2172 - 0.1209) / 0.1167 = +0.83 (빨간색)
z_FS = (0.2562 - 0.1209) / 0.1167 = +1.16 (빨간색)
```

**결과**:
```
L14-12265:  🔵  🔵  |  🔴  🔴
            VB  VS  |  FB  FS

→ "Variable에서 낮고, Fixed에서 높다"
```

### 4.3 Z-score 속성

**속성 1**: 각 feature의 평균 = 0
```
z_VB + z_VS + z_FB + z_FS ≈ 0
```

**속성 2**: 각 feature의 표준편차 = 1

**속성 3**: 해석의 용이성
```
z = 0    : 평균 수준 (흰색)
z = ±1   : 평균 ± 1 표준편차
z = ±2   : 평균 ± 2 표준편차
```

### 4.4 Z-score의 한계

**한계 1: 절댓값 정보 손실**
```
Feature A: [0.01, 0.00, 0.20, 0.25]  → z = [-1.0, -1.1, +0.8, +1.3]
Feature B: [0.50, 0.30, 2.00, 2.50]  → z = [-1.0, -1.1, +0.8, +1.3]

Z-score는 같지만:
- Feature A: 최대 0.25 (약함)
- Feature B: 최대 2.50 (강함)
```

**한계 2: Feature 간 비교 불가**
```
같은 빨간색(z=+1.0)이어도:
- Feature A의 +1.0 = 절댓값 0.25
- Feature B의 +1.0 = 절댓값 2.50

→ "Feature B가 더 중요"라고 할 수 없음!
```

**대응**: η² 값을 Y축에 병기 → Feature 중요도 판단

---

## 5. 히트맵 읽는 법

### 5.1 색상 의미

**컬러맵**: RdBu_r (Red-Blue reversed)

```
진한 파란색 (-2.5) ← 평균보다 매우 낮음
  ↓
중간 파란색 (-1.0)
  ↓
흰색 (0.0) ← 평균 수준
  ↓
중간 빨간색 (+1.0)
  ↓
진한 빨간색 (+2.5) ← 평균보다 매우 높음
```

**주의**: 빨간색 = "높다"가 아니라 "**이 feature 기준으로** 평균보다 높다"

### 5.2 구조

**X축**:
```
     Variable Betting  |  Fixed Betting
     ─────────────────┼─────────────────
     Bankrupt  Safe    |  Bankrupt  Safe
        0       1       |     2       3
                   점선 →
```

**Y축**: η² 기준 상위 20개 features (큰 순서대로)

### 5.3 패턴 유형

#### 패턴 A: Bet Type Encoding

```
        VB  VS  |  FB  FS
Row i:  🔵  🔵 |  🔴  🔴
```

**특징**:
- 점선(Variable|Fixed 경계) 기준 색 변화
- VB와 VS 비슷 (둘 다 Variable)
- FB와 FS 비슷 (둘 다 Fixed)
- Bankrupt vs Safe는 별 차이 없음

**해석**: "어떤 베팅 조건인가"를 인코딩

**주로 발견**: LLaMA

#### 패턴 B: Outcome Encoding

```
        VB  VS  |  FB  FS
Row j:  🔴  🔵 |  🔴  🔵
```

**특징**:
- Bankrupt(0, 2열)과 Safe(1, 3열)로 색 구분
- VB와 FB 비슷 (둘 다 Bankrupt)
- VS와 FS 비슷 (둘 다 Safe)
- Variable vs Fixed는 별 차이 없음

**해석**: "파산했는가 안전한가"를 인코딩

**주로 발견**: Gemma

#### 패턴 C: Mixed Encoding

```
        VB  VS  |  FB  FS
Row k:  🔴  🔵 |  🔵  🔴
```

**특징**: 4개 조건이 모두 다른 색 (대각선/체커보드)

**해석**: Bet Type과 Outcome 모두 영향 (교호작용)

**발견**: 드물게

### 5.4 η² 값 읽기

**표시 위치**: Y축 오른쪽

**의미**:
```
L14-12265  η²=0.850
→ 4개 그룹 차이가 총 분산의 85% 설명
→ 매우 강한 효과 (η² ≥ 0.50)
→ 이 feature는 매우 중요
```

**비교**:
```
L14-12265  η²=0.850  ← 최고 중요
L28-45678  η²=0.301  ← 덜 중요 (하지만 Top 20)
```

### 5.5 점선의 의미

**위치**: x=1.5 (Variable|Fixed 경계)

**활용**:
- 점선 기준 색 변화 → Bet Type 인코딩
- 점선 무관 색 변화 → Outcome 인코딩

---

## 6. LLaMA vs Gemma 패턴

### 6.1 LLaMA-3.1-8B

**전형적 패턴**:
```
        Variable  |  Fixed
        Bank Safe | Bank Safe   η²
Row 1:  🔵  🔵  |  🔴  🔴    0.850
Row 2:  🔵  🔵  |  🔴  🔴    0.723
Row 3:  🔵  🔵  |  🔴  🔴    0.689
⋮
Row 20: 🔵  🔵  |  🔴  🔴    0.301

Top 20 중 18-19개가 이 패턴
```

**특징**:
- **압도적으로 패턴 A** (Bet Type Encoding)
- 점선 기준 색 변화
- Bankrupt vs Safe는 거의 구분 안 됨

**통계**:
```
Bet Type dominant: 74.6% (1617/2168 features)
Outcome dominant: 25.4% (551/2168 features)

Mean η²(Bet Type): 0.0412
Mean η²(Outcome): 0.0080

Bet/Outcome Ratio: 5.2×
```

**해석**: **환경 표상 우선 전략** (Environment-First)
- "어떤 게임 규칙인가" 먼저 파악
- Layer 12-15에서 베팅 조건 명시적 표상

### 6.2 Gemma-2-9B-IT

**전형적 패턴**:
```
        Variable  |  Fixed
        Bank Safe | Bank Safe   η²
Row 1:  🔴  🔵  |  🔴  🔵    0.923
Row 2:  🔴  🔵  |  🔴  🔵    0.887
Row 3:  🔴  🔵  |  🔴  🔵    0.856
⋮
Row 20: 🔴  🔵  |  🔴  🔵    0.645

Top 20 중 18-19개가 이 패턴
```

**특징**:
- **압도적으로 패턴 B** (Outcome Encoding)
- Bankrupt/Safe로 색 구분
- Variable vs Fixed는 거의 구분 안 됨
- **활성화 차이가 극단적** (50~100배)

**통계**:
```
Outcome dominant: 약 80% 이상
Mean η²(Outcome) >> Mean η²(Bet Type)

활성화 차이:
- Bankrupt: 50.0
- Safe: 0.5
- 100배 차이!
```

**해석**: **결과 표상 우선 전략** (Outcome-First)
- "무슨 일이 일어났나" 먼저 기록
- Layer 26-40에서 파산 여부 명시적 표상

### 6.3 왜 다른가?

**가설 1: 학습 데이터 차이**
```
LLaMA: 일반 웹 텍스트
→ 맥락 의존적 표상

Gemma: Instruction-tuned
→ 결과 중심 표상
```

**가설 2: 아키텍처 차이**
```
LLaMA: 8B params, 32 layers
Gemma: 9B params, 42 layers
→ 표상 전략이 다를 수 있음
```

**가설 3: 알고리즘적 다양성**
```
같은 태스크, 같은 결과
하지만 내부 계산 방법이 다름

→ Multiple Realizability (다중 실현)
```

**의미**:
- 블랙박스 내부의 이질성
- 모델마다 다른 모니터링 전략 필요
- AI Safety 함의

---

## 7. 주의사항

### 7.1 Z-score 해석 함정

❌ **잘못된 해석**:
```
"Feature A의 빨간색(z=+1.5)이 Feature B의 빨간색(z=+1.2)보다 진하니까
 Feature A가 더 중요하다"
```

✅ **올바른 해석**:
```
"Feature A와 B의 절댓값 스케일이 다르므로
 중요도는 η² 값으로 판단해야 한다"
```

**핵심**: 같은 색 ≠ 같은 활성화

### 7.2 샘플 크기 불균형

**문제**:
```
VB: 108 (3.4%)   ← 작음
FB: 42 (1.3%)    ← 매우 작음
VS: 1492 (46.6%)
FS: 1558 (48.7%)
```

**영향**: 작은 그룹의 평균 불안정

**대응**:
- Welch's ANOVA 사용 (등분산 가정 불필요)
- 효과 크기(η²) 중심 해석

### 7.3 Sparse Features

**문제**:
```
95% 게임에서 0
5% 게임에서만 활성화
→ 표준편차 ≈ 0
→ Z-score 불안정
```

**대응**: 최소 분산/활성화율 필터링

### 7.4 다중 비교

**문제**: 229,376번 검정 → False positives

**대응**: FDR 보정 + η² 기준 선택

### 7.5 인과성 vs 상관성

❌ **할 수 없는 주장**:
```
"Feature가 행동을 유발한다"
```

✅ **할 수 있는 주장**:
```
"Feature가 조건과 연관되어 있다"
"인과성은 별도 실험에서 검증"
```

**인과성 검증**: Activation Patching, Steering Vectors

---

## 8. 실전 해석 예시

### 8.1 LLaMA Figure 해석

**관찰**:
```
Top 20 중 18개가 패턴 A:
        VB  VS  |  FB  FS
        🔵  🔵 |  🔴  🔴
```

**5단계 해석**:

**Step 1: 패턴 인식**
- 점선 기준 색 변화
- → Bet Type Encoding

**Step 2: 방향 확인**
- Variable: 파란색 (낮음)
- Fixed: 빨간색 (높음)

**Step 3: 효과 크기**
- η² = 0.3~0.85 (중간~매우 큼)

**Step 4: 일반화**
- 18/20 = 90%가 같은 패턴

**Step 5: 통계 지원**
- 74.6% features가 Bet Type dominant
- η² ratio = 5.2×

**결론**:
```
"LLaMA-3.1-8B는 슬롯머신 게임에서
 '어떤 베팅 조건인가'(Variable vs Fixed)를 우선적으로 인코딩한다.
 상위 20개 feature 중 90%가 베팅 조건에 따라
 명확히 구분되는 활성화 패턴을 보였다 (η²=0.30~0.85).
 게임 결과(파산 여부)는 상대적으로 약하게 표상된다."
```

### 8.2 Gemma Figure 해석

**관찰**:
```
Top 20 중 19개가 패턴 B:
        VB  VS  |  FB  FS
        🔴  🔵 |  🔴  🔵
```

**해석**:

**패턴**: Outcome Encoding
**방향**: Bankrupt 높음, Safe 낮음
**효과**: η² = 0.6~0.92 (매우 큼)
**일반화**: 19/20 = 95%

**결론**:
```
"Gemma-2-9B-IT는 슬롯머신 게임에서
 '무슨 일이 일어났는가'(Bankrupt vs Safe)를 우선적으로 인코딩한다.
 상위 20개 feature 중 95%가 게임 결과에 따라
 극단적으로(50~100배) 다른 활성화를 보였다 (η²=0.60~0.92).
 베팅 조건은 거의 표상되지 않았다.
 이는 LLaMA와 정반대의 표상 전략이다."
```

### 8.3 연구 의의

**발견 1: 알고리즘적 다양성**
```
같은 태스크, 같은 행동 결과
하지만 완전히 다른 내부 표상

→ Multiple Realizability
```

**발견 2: 모델별 맞춤 해석**
```
LLaMA: Bet Type features 추적 (L12-15)
Gemma: Outcome features 추적 (L26-40)

→ 일률적 접근 불가
```

**발견 3: AI Safety 함의**
```
블랙박스 내부가 이질적
→ 모델별 모니터링 전략 필요
```

---

## 부록 A: 핵심 공식 요약

### ANOVA
```
F = MS_between / MS_within

MS_between = SS_between / (k-1)
MS_within = SS_within / (N-k)

k = 4 (그룹 수)
N = 3200 (총 샘플 수)
```

### Eta-squared
```
η² = SS_between / SS_total

해석:
< 0.01: Negligible
= 0.01: Small
= 0.06: Medium
= 0.14: Large
≥ 0.50: Very large
```

### Z-score
```
z = (x - μ) / σ

각 feature(행) 내에서:
- μ: 4개 조건의 평균
- σ: 4개 조건의 표준편차
```

### FDR 보정
```
p_i ≤ (i/m) × α

i: 순위 (오름차순)
m: 총 검정 횟수
α: 0.05
```

---

## 부록 B: 체크리스트

### Figure 1을 처음 볼 때
- [ ] 모델 이름 확인 (LLaMA vs Gemma)
- [ ] η² 값 범위 확인
- [ ] 점선 위치 확인
- [ ] 전체 색 패턴 파악

### 패턴 분석 시
- [ ] 패턴 A (Bet Type): 점선 기준?
- [ ] 패턴 B (Outcome): Bankrupt/Safe 기준?
- [ ] 20개 중 몇 개가 각 패턴?

### 통계 검증 시
- [ ] η² ≥ 0.01 확인
- [ ] FDR 보정 적용 여부
- [ ] 샘플 크기 불균형 고려
- [ ] Sparse features 필터링

### 해석 작성 시
- [ ] "인코딩" (관찰) vs "유발" (인과) 구분
- [ ] 정량적 증거 포함 (η², 비율)
- [ ] 모델 이름 명시
- [ ] 한계 언급

---

## 부록 C: 용어 사전

| 용어 | 정의 |
|------|------|
| **SAE Feature** | 신경망의 해석 가능한 방향 |
| **활성화** | Feature의 반응 강도 (스칼라) |
| **그룹 평균** | 특정 조건에서 활성화 평균 |
| **One-Way ANOVA** | 3개 이상 그룹 평균 차이 검정 |
| **η² (Eta-squared)** | 그룹 차이가 설명하는 분산 비율 |
| **Z-score** | 평균으로부터의 표준편차 거리 |
| **FDR** | 다중 비교 false positive 제어 |
| **Bet Type** | 베팅 조건 (Variable vs Fixed) |
| **Outcome** | 게임 결과 (Bankrupt vs Safe) |

---

**문서 작성**: 2026-02-02
**분석 대상**: LLaMA-3.1-8B, Gemma-2-9B-IT
**실험 데이터**: 3,200 games, 32,768 features/layer
