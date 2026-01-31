# SAE Condition Comparison — Figure Analysis Guide

본 문서는 SAE 조건 비교 분석에서 생성된 4개의 Figure에 대한 해석 가이드이다. 분석 대상은 LLaMA-3.1-8B(31 layers, 32,768 features/layer)과 Gemma-2-9B-IT(42 layers, 131,072 features/layer)이며, 각 모델에서 Variable Betting과 Fixed Betting 조건(각 1,600 games, 총 3,200 games/model)의 SAE feature 활성화 패턴을 비교한다. 모든 분석은 게임 결과에 따라 4가지 그룹으로 나뉜다: Variable-Bankrupt(VB), Variable-Safe(VS), Fixed-Bankrupt(FB), Fixed-Safe(FS).

---

## Figure 1: Four-Way SAE Feature Activation Heatmap

**파일**: `fig1_four_way_heatmap.png` (LLaMA), `fig1_four_way_heatmap_gemma.png` (Gemma)

### 보는 법

이 그래프는 Four-Way ANOVA에서 eta^2가 가장 높은 상위 20개 SAE feature가 4가지 조건에서 어떻게 활성화되는지를 시각화한 것이다. 세로축(Y축)에 feature 이름이 나열되어 있고, 가로축(X축)에는 4가지 조건(VB, VS, FB, FS)이 배치되어 있다. 가로축의 중앙에 점선이 있는데, 이 점선을 기준으로 왼쪽 두 열이 Variable 조건, 오른쪽 두 열이 Fixed 조건이다.

그래프에는 두 개의 패널이 있다. 패널 (A)는 실제 활성화 수치(Raw Activation Values)로, 색이 진할수록 해당 조건에서 그 feature의 활성화가 높다는 의미이다. 패널 (B)는 feature별로 Z-score 정규화를 적용한 것으로, 빨간색은 해당 feature의 평균 이상, 파란색은 평균 이하를 나타낸다. 핵심적으로 봐야 할 것은 패널 (B)에서 빨강과 파랑이 어떤 기준으로 나뉘는지이다. 점선(좌/우)을 기준으로 색이 나뉘면 베팅 조건(Bet Type)이 지배적이라는 뜻이고, 1열·3열 vs 2열·4열로 나뉘면 결과(Outcome)가 지배적이라는 뜻이다.

### LLaMA 특징

LLaMA의 heatmap에서 가장 눈에 띄는 특징은 점선을 기준으로 색이 명확하게 나뉜다는 점이다. 패널 (B)를 보면, 왼쪽 두 열(Variable Bankrupt, Variable Safe)은 대부분 파란색이고, 오른쪽 두 열(Fixed Bankrupt, Fixed Safe)은 대부분 빨간색이다. 즉, 상위 feature들(L14-12265, L13-32317, L15-27263 등)은 파산 여부와 관계없이 Fixed 조건에서 일관되게 높은 활성화를 보인다. VB와 VS의 색이 거의 같고, FB와 FS의 색도 거의 같다는 점이 이를 잘 보여준다.

패널 (A)에서는 전체 활성화 범위가 0~2.5로 비교적 완만한 그라데이션을 보인다. 다만 L12-26280, L18-31208 같은 일부 feature는 반대 패턴을 보이는데, 이들은 Variable 조건에서 더 높게 활성화되는 feature이다. 종합하면, LLaMA의 상위 discriminative feature들은 "이것이 Variable 게임인지 Fixed 게임인지"를 인코딩하고 있다.

### Gemma 특징

Gemma의 heatmap은 LLaMA와 완전히 다른 패턴을 보인다. 패널 (B)에서 색이 나뉘는 기준이 점선(Variable vs Fixed)이 아니라, 1열·3열(Bankrupt) vs 2열·4열(Safe)이다. 모든 상위 feature가 동일한 패턴을 보이는데, Bankrupt 조건(VB, FB)에서는 빨간색, Safe 조건(VS, FS)에서는 파란색이다. 베팅 조건(Variable vs Fixed)에 따른 차이는 거의 보이지 않는다.

더욱 주목할 점은 패널 (A)의 극단적인 이진 패턴이다. Bankrupt 열의 활성화 값이 10~50 범위인 반면, Safe 열은 0~1 범위에 불과하다. 활성화 차이가 50~100배에 달하며, 특히 L40-60350은 VB와 FB 모두에서 약 50의 활성화를 보이는 반면, VS/FS에서는 거의 0에 가깝다. 이는 Gemma의 상위 feature들이 "이 게임이 파산으로 끝나는 게임인지"를 극도로 강하게 인코딩하고 있음을 의미한다.

### 분석 포인트

두 모델의 heatmap은 같은 분석(Four-Way ANOVA)의 상위 feature임에도 완전히 다른 패턴을 보인다는 점에서 가장 근본적인 차이를 드러낸다. LLaMA는 베팅 조건을, Gemma는 결과를 인코딩한다. 이는 동일한 행동적 결과(Variable에서 더 높은 파산율)가 서로 다른 내부 표상에서 발생할 수 있음을 시사한다.

Gemma의 극단성도 주목할 만하다. Raw 값에서 Bankrupt vs Safe의 차이가 50~100배라는 것은 Gemma가 파산 게임에서 특정 feature를 극도로 강하게 활성화한다는 의미이며, 일종의 "파산 감지기(bankruptcy detector)" feature가 존재함을 시사한다. 논문에서 LLaMA와 Gemma의 heatmap을 나란히 배치하면, "동일한 행동적 결과에서 다른 신경적 전략이 관찰된다"는 핵심 메시지를 시각적으로 전달할 수 있다.

---

## Figure 2: Layer-wise Effect Size Distribution

**파일**: `fig2_layer_effect_size.png` (LLaMA), `fig2_layer_effect_size_gemma.png` (Gemma)

### 보는 법

이 그래프는 모델의 각 레이어에서 Variable과 Fixed 조건의 구분이 얼마나 강한지를 보여준다. 가로축(X축)은 레이어 번호로, 왼쪽이 초반 레이어(입력에 가까움), 오른쪽이 후반 레이어(출력에 가까움)이다. 세로축(Y축)은 레이어별 상위 50개 feature의 평균 |Cohen's d|로, 효과 크기를 나타낸다.

각 레이어에 두 개의 막대가 있다. 빨간 막대는 Variable 조건에서 더 높게 활성화되는 feature들의 평균 효과 크기이고, 파란 막대는 Fixed 조건에서 더 높게 활성화되는 feature들의 평균 효과 크기이다. LLaMA 그래프에는 피크 구간을 표시하는 노란 하이라이트가 있으며, 점선은 |d| = 2.0(Very Large effect size) 기준선이다. 핵심적으로, 막대가 높은 레이어가 Variable과 Fixed를 가장 강하게 구분하는 레이어이다.

### LLaMA 특징

LLaMA에서 가장 뚜렷한 특징은 중간 레이어(L12-L15)에 노란 하이라이트가 집중되어 있다는 점이다. 이 구간에서 빨간 막대와 파란 막대 모두 높은 값을 보이며, 양방향 구분이 가장 활발하게 일어난다. L9에서는 Fixed 쪽 feature의 효과 크기가 약 3.5로 급격히 높아지는 것도 눈에 띈다.

전체적으로 대부분의 레이어에서 |d|가 2.0(Very Large 기준)을 초과하며, 파란 막대가 빨간 막대보다 전반적으로 더 높다. 이는 Fixed 조건에서 활성화되는 feature들이 Variable 쪽 feature보다 더 선명한 구분력을 가진다는 의미이다. 이 결과는 LLaMA가 중간 레이어에서 베팅 조건을 처리하며, 중간 수준의 추상화 단계에서 "자유로운 베팅 환경"이라는 개념을 표상하고 있음을 시사한다.

### Gemma 특징

Gemma의 레이어 분포는 LLaMA와 대조적으로 분산된 패턴을 보인다. 피크가 L4(파란 막대 약 2.0)와 L16(파란 막대 약 2.1)에서 나타나지만, 특정 구간에 집중되지 않아 하이라이트 표시가 없다. 전체 효과 크기는 대부분 |d| = 1.0~1.6 수준으로, LLaMA(2.0~3.5)의 절반에 불과하다.

L2와 L18-L19에서는 Variable 쪽 feature(빨간 막대)가 상대적으로 높게 나타나며, L5-L6, L8-L12 등 일부 레이어는 그래프에서 아예 누락되어 있다. 이 레이어들에서는 상위 50개 feature의 효과 크기가 표시 기준에 미치지 못했다는 의미이다. 이러한 결과는 Gemma가 Variable vs Fixed 구분을 LLaMA만큼 강하게 수행하지 않는다는 것을 보여주며, Figure 1에서 확인된 발견과 일치한다. Gemma의 상위 feature들은 베팅 조건이 아닌 결과를 인코딩하기 때문이다.

### 분석 포인트

두 모델의 처리 위치가 다르다는 점이 가장 중요한 발견이다. LLaMA는 중간 레이어(L12-15)에 집중되어 있고, Gemma는 레이어 전반에 걸쳐 분산되어 있다. 이는 두 모델의 아키텍처와 학습 방식의 차이를 반영할 수 있다.

효과 크기의 차이도 의미가 있다. LLaMA가 Gemma보다 Variable vs Fixed를 더 강하게 구분한다는 것은, LLaMA가 베팅 조건 자체에 더 민감하다는 Figure 1의 발견과 일치한다. 또한 논문 Section 3.2의 기존 Finding 2에서 "risky features는 후반 레이어, safe features는 초반~중간 레이어에 분포한다"는 causal patching 결과와 이 Figure의 레이어 분포를 직접 비교할 수 있다는 점도 활용 가치가 높다.

---

## Figure 3: Bet Type vs Outcome Effect Scatter

**파일**: `fig3_bet_vs_outcome_scatter.png` (LLaMA), `fig3_bet_vs_outcome_scatter_gemma.png` (Gemma)

### 보는 법

이 그래프는 4개의 Figure 중 가장 핵심적인 것으로, 각 SAE feature가 "베팅 조건"과 "결과" 중 어느 쪽을 더 강하게 인코딩하는지를 한눈에 보여준다. 그래프의 각 점이 하나의 SAE feature를 나타낸다.

가로축(X축)은 Outcome Effect, 즉 |Bankrupt 평균 - Safe 평균|이다. 이 값이 클수록 해당 feature가 파산 게임과 안전 게임을 강하게 구분한다는 의미이다. 세로축(Y축)은 Bet Type Effect, 즉 |Variable 평균 - Fixed 평균|이다. 이 값이 클수록 해당 feature가 Variable과 Fixed 조건을 강하게 구분한다는 의미이다.

그래프에 대각선(점선)이 그려져 있는데, 이 선은 두 효과가 동일한 지점을 나타낸다. 대각선 위쪽(보라색 영역)에 있는 점은 베팅 조건이 결과보다 더 중요한 feature이고, 대각선 아래쪽(흰색 영역)에 있는 점은 결과가 베팅 조건보다 더 중요한 feature이다. 점의 색상은 eta^2 값으로, 노란색일수록 전체 효과 크기가 크고, 보라색일수록 상대적으로 작다.

### LLaMA 특징

LLaMA의 scatter plot에서 거의 모든 점이 대각선 위, 즉 보라색 영역에 위치한다. Bet Type Effect(Y축)의 범위가 0.07~0.40인 반면, Outcome Effect(X축)의 범위는 0.00~0.14에 불과하여, Y축 범위가 X축 범위보다 약 3배 넓다. 이는 상위 feature들에게 있어 베팅 조건이 결과보다 훨씬 중요한 구분자라는 뜻이다.

특히 eta^2가 높은 최상위 feature들(노란색 점)인 L14-12265, L13-32317, L12-30147 등은 모두 대각선 위에 있으며, Outcome Effect가 0.01~0.03 수준으로 매우 작다. 유일한 예외는 L13-7256으로, X축 약 0.11, Y축 약 0.25에 위치하여 Outcome Effect도 어느 정도 있지만, 여전히 Bet Type이 더 크다. 결론적으로, LLaMA의 상위 feature들은 압도적으로 "Variable인지 Fixed인지"를 인코딩하고 있다.

### Gemma 특징

Gemma의 scatter plot은 LLaMA와 정반대의 패턴을 보인다. 거의 모든 점이 대각선 아래, 즉 흰색 영역에 위치한다. Outcome Effect(X축)의 범위가 5~50인 반면, Bet Type Effect(Y축)의 범위는 0~5에 불과하여, X축 범위가 Y축 범위보다 약 10배 넓다.

eta^2가 높은 최상위 feature들(노란색 점)인 L40-108098, L26-33483 등은 Outcome Effect가 20~30인 반면 Bet Type Effect는 1~2에 불과하다. 즉, 결과(파산 여부)가 베팅 조건보다 10~20배 더 강한 구분자로 작용한다. 축의 스케일 자체가 LLaMA와 크게 다르다는 점도 주목해야 한다. LLaMA는 X축이 0~0.14인 반면 Gemma는 0~50으로, 이는 GemmaScope과 LlamaScope의 SAE 학습 설정 차이에 따른 raw 활성화 값의 차이를 반영한다. Gemma의 상위 feature들은 압도적으로 "파산했는지 안 했는지"를 인코딩하고 있다.

### 분석 포인트

Figure 3는 논문의 핵심 Figure로 가장 적합하다. 한 장의 그래프로 "LLaMA는 베팅 조건을, Gemma는 결과를 인코딩한다"는 핵심 메시지를 명확하게 전달할 수 있으며, 두 모델을 (A), (B) 패널로 나란히 배치하면 대비 효과가 극대화된다.

대각선의 위치가 갖는 의미도 강력하다. LLaMA는 모든 점이 대각선 위에, Gemma는 모든 점이 대각선 아래에 있어서, 두 모델이 완전히 반대 방향의 인코딩 전략을 채택하고 있음을 보여준다. 다만 논문에서 두 모델을 직접 비교할 때는 축 스케일의 차이(LLaMA 0~0.4 vs Gemma 0~50)를 고려하여, 정규화된 비율(Bet Type Effect / Outcome Effect)로 비교하는 것이 더 적절할 수 있다.

eta^2 색상도 의미 있는 정보를 담고 있다. 노란 점(높은 eta^2)이 대부분 대각선의 같은 쪽에 몰려 있다는 것은, 가장 중요한 feature일수록 각 모델의 인코딩 경향이 더욱 뚜렷해진다는 의미이다.

---

## Figure 4: Top Differential SAE Features Bar Chart

**파일**: `fig4_top_features_bar.png` (LLaMA), `fig4_top_features_bar_gemma.png` (Gemma)

### 보는 법

이 그래프는 Analysis 1(독립 t-test)의 결과를 시각화한 것으로, Variable과 Fixed 조건 간에 가장 큰 활성화 차이를 보이는 개별 SAE feature들을 보여준다. 4개의 Figure 중 가장 직관적으로 이해할 수 있는 그래프이다.

세로축(Y축)에 개별 SAE feature 이름이 나열되어 있으며, 각 이름은 "L[레이어 번호]-[feature ID]" 형식이다. 가로축(X축)은 Cohen's d, 즉 효과 크기이다. 양수 방향(오른쪽)의 빨간 막대는 Variable 조건에서 더 높게 활성화되는 feature를, 음수 방향(왼쪽)의 파란 막대는 Fixed 조건에서 더 높게 활성화되는 feature를 나타낸다. 그래프에는 두 개의 기준 점선이 있는데, |d| = 0.8은 Large effect size, |d| = 2.0은 Very Large effect size를 의미한다. 막대가 2.0 점선을 넘으면 매우 신뢰할 수 있는 차이라고 판단할 수 있다.

### LLaMA 특징

LLaMA에서 가장 강한 feature는 L14-12265로, Cohen's d가 약 -4.75로 Fixed 조건에서 압도적으로 높은 활성화를 보인다. 이어서 L13-32317(d 약 -4.56), L15-27263(d 약 -4.21)이 뒤따르며, 상위 3개 feature가 모두 Fixed 쪽(파란 막대)이라는 점이 주목할 만하다.

그러나 전체적으로 보면 빨간 막대 약 15개, 파란 막대 약 15개로 양쪽이 대략 균등하게 분포되어 있다. Variable 쪽 최강 feature는 L12-26280(d 약 3.34)이다. 모든 feature의 효과 크기가 |d| = 2.0~4.75 범위로, Very Large 기준을 초과한다. 레이어 분포를 보면, Fixed 쪽 feature는 L13, L14, L15에 집중되어 있고, Variable 쪽 feature는 L7, L12, L14, L18 등으로 상대적으로 분산되어 있다. 이는 LLaMA가 Variable과 Fixed를 모두 강하게 인코딩하는 feature를 보유하고 있으며, 특히 "고정 베팅 환경"에 대한 표상이 더 선명하다는 것을 보여준다.

### Gemma 특징

Gemma에서 가장 강한 feature는 L4-43019으로, Cohen's d가 약 -3.67이다. 이 feature만이 유일하게 |d| > 3.0을 넘으며, 나머지 feature들은 대부분 |d| = 1.0~2.0 범위에 있어 LLaMA보다 효과 크기가 전반적으로 작다.

Variable 쪽 상위 feature로는 L18-108762(d 약 2.07)과 L19-118800(d 약 2.00)이 있다. 레이어 분포를 보면 Variable 쪽 feature(빨간 막대)가 초반 레이어 L2, L3과 중후반 레이어 L17-L19, L22-L24에 넓게 분산되어 있고, Fixed 쪽 feature(파란 막대)는 L4-43019이 압도적이며 나머지는 L16-L21에 분포한다. LLaMA의 최대 효과 크기가 4.75인 데 비해 Gemma는 3.67로, Variable vs Fixed 구분력이 상대적으로 약하다. 이는 Gemma가 베팅 조건보다 결과(파산 여부)에 더 민감하기 때문이며, Figure 1과 Figure 3의 발견과 일관된다.

### 분석 포인트

Figure 4는 구체적인 feature를 식별해야 할 때 가장 유용하다. 여기에 나열된 feature 이름을 Neuronpedia에서 조회하면 해당 feature가 어떤 토큰이나 개념에 반응하는지 의미적 해석이 가능하다.

LLaMA가 양쪽(Variable, Fixed) 모두에서 강한 feature를 균등하게 보유한다는 점도 중요하다. 이는 모델이 단순히 한 조건을 인식하는 것이 아니라, 두 조건의 차이를 적극적으로 인코딩하고 있음을 시사한다. 실용적으로는 |d| > 2.0인 feature들이 매우 신뢰할 수 있는 차이를 보이므로, 이 feature들을 causal patching의 후보로 사용하면 인과적 검증의 성공률을 높일 수 있다.

---

## 종합: 논문에서의 Figure 활용 전략

### 추천 구성

4개의 Figure 중 논문 본문에 가장 적합한 것은 Figure 3(Scatter)이다. 한 장의 그래프로 "LLaMA는 베팅 조건을, Gemma는 결과를 인코딩한다"는 핵심 메시지를 전달할 수 있으며, 두 모델의 완전히 반대되는 경향이 대각선을 기준으로 명확하게 드러난다. Figure 1(Heatmap)은 본문의 보조 Figure로 적합한데, 구체적인 활성화 패턴을 보여주면서 특히 Gemma의 극단적인 이진 패턴이 시각적으로 인상적이기 때문이다. Figure 2(Layer Distribution)와 Figure 4(Bar Chart)는 Supplementary에 배치하는 것을 추천한다. Figure 2는 레이어 분포의 상세 정보를 담고 있어 기존 causal patching의 layer 분석과 연결할 수 있고, Figure 4는 개별 feature 목록으로 Neuronpedia 해석 시 참조용으로 활용할 수 있다.

논문 Figure로 구성할 때는, (A) LLaMA Scatter, (B) Gemma Scatter, (C) LLaMA Heatmap Z-score 패널, (D) Gemma Heatmap Z-score 패널의 4패널 구성이 가장 효과적이다.

### 핵심 결론

4개의 Figure를 종합하면 다음과 같은 일관된 결론에 도달한다. LLaMA의 상위 discriminative feature들은 베팅 조건(Variable vs Fixed)을 인코딩하고 있으며, 이 처리가 중간 레이어(L12-L15)에 집중되어 있다. Gemma의 상위 discriminative feature들은 게임 결과(Bankrupt vs Safe)를 인코딩하고 있으며, 이 처리가 후반 레이어(L26-L40)에서 극단적인 활성화 차이와 함께 나타난다. 이 발견은 두 모델이 동일한 행동적 결과(Variable 조건에서 약 2.4배 높은 파산율)를 보이면서도, 신경망 수준에서는 서로 다른 계산 전략을 채택하고 있음을 의미한다.

---

## 통계적 기준

본 분석에서는 Benjamini-Hochberg 방법으로 FDR 보정(alpha = 0.05)을 적용하였다. Analysis 1(Variable vs Fixed t-test)에서는 |Cohen's d| >= 0.3을 분석 포함 기준으로 사용하였고, Analysis 2와 3(Four-Way ANOVA, Interaction)에서는 eta^2 >= 0.01을 기준으로 하였다. Figure 4에 표시된 점선 기준은 |d| = 0.8(Large effect size)과 |d| = 2.0(Very Large effect size)이다.

---

*문서 생성일: 2026-01-29*
*분석 데이터: sae_condition_comparison/results/*
