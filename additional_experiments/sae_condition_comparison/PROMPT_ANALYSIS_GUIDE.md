# Prompt Component Analysis - 결과 해석 가이드

생성일: 2026-02-01
분석 유형: Prompt Component × Outcome Interaction

---

## 분석 개요

### 연구 질문
**각 prompt component(G/M/R/W/P)가 bankruptcy vs safe outcome에서 SAE feature 활성화에 어떤 영향을 주는가?**

### 방법론
- **Two-way ANOVA**: Component (포함 vs 미포함) × Outcome (bankruptcy vs safe)
- **샘플 크기**: 각 그룹 1,600 games (충분히 큼)
- **통계 보정**: FDR correction (Benjamini-Hochberg)
- **효과 크기**: η² (eta-squared) ≥ 0.01

---

## Prompt Components 정의

| Component | 설명 | 예시 프롬프트 |
|-----------|------|--------------|
| **G** | Goal-setting | "set a target amount before starting" |
| **M** | Maximize reward | "maximize the reward as much as possible" |
| **R** | Hidden patterns | "may have hidden patterns you can discover" |
| **W** | Win multiplier | "Win multiplier: 3.0 times your bet" |
| **P** | Win rate (Probability) | "Win rate: 30%" |

### 조합 예시
- `BASE`: 아무 component 없음
- `G`: Goal-setting만
- `GM`: Goal-setting + Maximize
- `GMRWP`: 모든 component 포함

---

## 결과 파일 구조

```
results/
├── prompt_component/
│   ├── G_llama_20260201_230709.json    # Goal-setting 결과
│   ├── M_llama_20260201_230709.json    # Maximize 결과
│   ├── R_llama_20260201_230709.json    # Hidden patterns 결과
│   ├── W_llama_20260201_230709.json    # Win multiplier 결과
│   ├── P_llama_20260201_230709.json    # Win rate 결과
│   ├── (gemma 버전들...)
│   └── ...
└── figures/
    ├── component_layer_heatmap_llama.png    # Component × Layer heatmap
    ├── component_barplot_llama.png          # Top features per component
    ├── component_summary_table_llama.png    # 요약 테이블
    ├── model_comparison.png                 # LLaMA vs Gemma 비교
    └── component_summary_llama.csv          # CSV 요약
```

---

## JSON 결과 파일 구조

### 각 component의 JSON 파일 (예: `G_llama_*.json`)

```json
{
  "summary": {
    "component": "G",
    "component_name": "Goal-setting",
    "total_features_analyzed": 21194,
    "fdr_significant_count": 1027,
    "significant_with_min_eta": 1027,
    "max_interaction_eta": 1.0,
    "timestamp": "20260201_230709",
    "model_type": "llama"
  },
  "top_features": [
    {
      "layer": 21,
      "feature_id": 3084,
      "component": "G",
      "component_f": 123.45,
      "component_p": 1.23e-10,
      "component_eta": 0.45,
      "outcome_f": 234.56,
      "outcome_p": 2.34e-20,
      "outcome_eta": 0.67,
      "interaction_f": 345.67,
      "interaction_p": 3.45e-30,
      "interaction_eta": 0.89,
      "group_means": {
        "False_bankruptcy": 0.0,
        "False_voluntary_stop": 0.0,
        "True_bankruptcy": 0.0030,
        "True_voluntary_stop": 0.0030
      },
      "n_with_component": 1600,
      "n_without_component": 1600,
      "interaction_p_fdr": 0.0,
      "interaction_fdr_significant": true
    }
  ],
  "all_results": [...]  // 전체 결과 (save_all_results=true인 경우)
}
```

---

## 결과 해석 가이드

### 1. Interaction Effect (핵심 지표)

**Interaction η²**: Component 포함 여부가 bankruptcy vs safe 차이에 미치는 영향

- **η² ≥ 0.14**: Large effect (매우 강한 상호작용)
- **η² ≥ 0.06**: Medium effect (중간 상호작용)
- **η² ≥ 0.01**: Small effect (작은 상호작용, 최소 threshold)

**해석 예시**:
```json
"interaction_eta": 0.85,
"group_means": {
  "False_bankruptcy": 0.005,  // Component 없을 때, Bankruptcy
  "False_voluntary_stop": 0.005,  // Component 없을 때, Safe
  "True_bankruptcy": 0.250,  // Component 있을 때, Bankruptcy
  "True_voluntary_stop": 0.001   // Component 있을 때, Safe
}
```

**해석**:
- Goal-setting(G)이 있을 때만 bankruptcy에서 feature가 강하게 활성화됨 (0.250 vs 0.001)
- Goal-setting이 없으면 bankruptcy와 safe의 차이가 없음 (0.005 vs 0.005)
- **→ Goal-setting이 bankruptcy encoding을 촉발하는 trigger 역할**

---

### 2. Main Effects

#### Component Main Effect (component_eta)
- Component 포함 vs 미포함의 평균 차이
- Outcome(bankruptcy/safe)과 무관한 전반적 차이

**해석 예시**:
- `component_eta = 0.30`: Component가 있으면 해당 feature가 전반적으로 더 활성화됨
- **→ 이 component가 특정 feature의 활성화를 유도**

#### Outcome Main Effect (outcome_eta)
- Bankruptcy vs Safe의 평균 차이
- Component 포함 여부와 무관한 결과 차이

**해석 예시**:
- `outcome_eta = 0.50`: Bankruptcy에서 feature가 더 활성화됨 (component 무관)
- **→ 이 feature는 원래 bankruptcy를 encoding**

---

### 3. Group Means 패턴 분류

#### 패턴 A: Component-specific Bankruptcy Encoding
```json
{
  "False_bankruptcy": 0.001,
  "False_voluntary_stop": 0.001,
  "True_bankruptcy": 0.500,  // 높음
  "True_voluntary_stop": 0.002
}
```
**해석**: Component가 있을 때만 bankruptcy에서 활성화
- **의미**: 이 component가 bankruptcy risk encoding을 유도

#### 패턴 B: Component-specific Safe Encoding
```json
{
  "False_bankruptcy": 0.001,
  "False_voluntary_stop": 0.001,
  "True_bankruptcy": 0.002,
  "True_voluntary_stop": 0.500  // 높음
}
```
**해석**: Component가 있을 때만 safe에서 활성화
- **의미**: 이 component가 safety/risk-aversion encoding을 유도

#### 패턴 C: Component-suppressed Outcome Encoding
```json
{
  "False_bankruptcy": 0.500,  // 높음
  "False_voluntary_stop": 0.002,
  "True_bankruptcy": 0.001,
  "True_voluntary_stop": 0.001
}
```
**해석**: Component가 없을 때만 bankruptcy에서 활성화
- **의미**: 이 component가 기존 bankruptcy encoding을 억제

#### 패턴 D: Opposite Effects
```json
{
  "False_bankruptcy": 0.500,  // Without: bankruptcy 높음
  "False_voluntary_stop": 0.002,
  "True_bankruptcy": 0.002,
  "True_voluntary_stop": 0.500  // With: safe 높음
}
```
**해석**: Component가 bankruptcy → safe encoding을 반전
- **의미**: 이 component가 모델의 outcome representation을 근본적으로 변경

---

### 4. Layer 분포 해석

**LLaMA Layer 분포 예상**:
- **L1-10**: 입력 토큰 처리, 초기 표현
- **L11-20**: 중간 추상화, 의사결정 encoding 시작
- **L21-31**: 최종 의사결정, 출력 생성

**Gemma Layer 분포 예상**:
- **L0-15**: 입력 처리
- **L16-30**: 중간 표현
- **L31-41**: 최종 의사결정

**해석 예시**:
- Component G의 significant features가 L12-15에 집중
  - **→ Goal-setting이 중간 layer에서 의사결정 encoding을 촉발**
- Component W의 significant features가 L25-31에 집중
  - **→ Win multiplier 정보가 최종 의사결정 직전에 통합됨**

---

## Sparse Feature 문제 및 필터링

### 문제
- SAE features는 본질적으로 sparse (L1 penalty 설계)
- Activation rate < 1%인 features는 interaction η² ≈ 1.0 artifact 발생
- ANALYSIS_ISSUES_REPORT.md 참조

### 현재 필터링 기준
```yaml
sparse_filter:
  enabled: true
  min_activation_rate: 0.01  # 1% 이상 활성화
  min_mean_activation: 0.001  # 평균 활성화 0.001 이상
```

### 해석 시 주의사항
1. **η² = 1.0 features는 의심**: 대부분 sparse artifact
2. **Group means 확인 필수**: 한 그룹만 비정상적으로 높으면 sparse
3. **신뢰 기준**:
   - ✅ 0.01 < η² < 0.90 + 모든 그룹에 값 분포
   - ⚠️ η² > 0.95 + 한 그룹만 활성화 → Sparse artifact
   - ❌ η² ≈ 1.0 → 거의 항상 artifact

---

## 시각화 해석

### 1. Component × Layer Heatmap
- **X축**: Layer (1-31 for LLaMA, 0-41 for Gemma)
- **Y축**: Component (G/M/R/W/P)
- **색상**: Mean interaction η² (top 20 features per component)

**해석**:
- 어두운 빨강: 해당 component가 해당 layer에서 강한 interaction
- 파랑/흰색: 약한 interaction
- **패턴 예시**: G가 L12-15에서 빨강 → Goal-setting의 bankruptcy encoding이 중간 layer에 집중

### 2. Component Barplot
- 각 component의 top 10 features
- 색상: Red (η² > 0.5), Orange (0.2-0.5), Blue (< 0.2)

**해석**:
- 많은 red bars: 해당 component가 강한 interaction 유도
- 대부분 blue: 약한 효과
- **비교**: G vs W 중 어느 component가 더 강한 interaction을 유도하는지

### 3. Summary Table
- Total Features: 분석된 feature 수 (sparse filter 후)
- FDR Significant: p_fdr < 0.05인 feature 수
- With Min η²: η² ≥ 0.01인 feature 수
- Max η²: 최대 interaction effect size

**해석 예시**:
```
Component | Total | FDR Sig | With Min η² | Max η²
G         | 21194 | 1027    | 1027        | 1.0000
M         | 21000 | 50      | 50          | 0.3500
```
- G가 M보다 훨씬 많은 significant features (1027 vs 50)
- **→ Goal-setting이 Maximize보다 강한 outcome interaction 유도**

### 4. Model Comparison
- LLaMA vs Gemma의 significant feature 수 비교

**해석 예시**:
- LLaMA: G=1027, M=50, R=100, W=200, P=150
- Gemma: G=500, M=800, R=600, W=1200, P=700
- **→ LLaMA는 Goal-setting에 민감, Gemma는 Maximize/Win multiplier에 민감**

---

## 논문 활용 가이드

### Main Text (핵심 발견)
1. **Component × Outcome Interaction 발견**
   - "Goal-setting component가 bankruptcy outcome에서 특정 SAE features를 차별적으로 활성화 (η² = 0.XX, p_fdr < 0.001)"
   - Layer 분포 + Top feature 예시

2. **Model 비교**
   - "LLaMA와 Gemma는 다른 prompt components에 민감함"
   - LLaMA: Goal-driven, Gemma: Reward-driven

### Supplementary Material
1. **전체 Component 결과** (5개 components)
2. **Sparse feature 필터링 방법론**
3. **Layer-by-layer 분석**

### Discussion
1. **Prompt engineering implications**
   - 어떤 component가 bankruptcy risk를 증가시키는지
   - 안전한 prompt 설계 가이드라인

2. **Neural mechanisms**
   - Layer 분포 → 의사결정 과정 추론
   - Component-specific pathways

---

## 분석 확장 방향

### 1. Three-way ANOVA (선택적)
- Component × Outcome × Bet_type
- Variable betting 조건에서 component 효과가 다른지 분석

### 2. Component Combination Analysis
- 특정 조합(예: GM, GR)의 효과
- Synergistic vs Antagonistic interactions

### 3. Complexity Analysis
- Prompt 복잡도(0-5)에 따른 feature 활성화 변화
- Linear trend vs Optimal complexity

---

## 체크리스트

분석 완료 후 확인사항:

- [ ] 모든 5개 components의 결과 파일 생성됨 (G, M, R, W, P)
- [ ] Sparse filtering이 올바르게 적용됨 (activation rate 확인)
- [ ] Top features의 group means가 reasonable (한 그룹만 비정상적으로 높지 않음)
- [ ] FDR correction이 적용됨 (p_fdr 필드 존재)
- [ ] Layer 분포가 reasonable (모든 layer에 골고루 분포 또는 특정 구간 집중)
- [ ] 시각화 생성됨 (heatmap, barplot, table, comparison)
- [ ] Summary statistics 계산됨
- [ ] LLaMA와 Gemma 결과 모두 생성됨

---

## 문의 및 추가 분석

이 분석 결과에 대한 추가 질문이나 후속 분석이 필요한 경우:
1. ANALYSIS_ISSUES_REPORT.md 참조 (sparse feature 문제)
2. INTERACTION_ETA_PROBLEM_EXPLAINED.md 참조 (interaction artifact 설명)
3. Additional analysis 요청: prompt_complexity_analysis.py 실행

---

**Last Updated**: 2026-02-01
**Analyst**: Claude Code
**Status**: Analysis in progress
