# SAE Condition Comparison: 실험 정리

**생성일**: 2026-02-02
**실험 경로**: `exploratory_experiments/additional_experiments/sae_condition_comparison/`

## 📋 실험 개요

이 실험은 **Variable vs Fixed 베팅 조건**이 SAE 피처 활성화에 미치는 영향을 분석합니다. 기존 SAE 분석(파산 vs 비파산)을 확장하여, 베팅 조건이라는 외생 변수가 신경 표현에 어떻게 인코딩되는지 탐구합니다.

### 연구 동기

| Model | Fixed 파산율 | Variable 파산율 | 증가율 |
|-------|-------------|----------------|--------|
| LLaMA | 2.6% (42/1600) | 6.8% (108/1600) | **2.6배** |
| Gemma | 12.8% (205/1600) | 29.1% (465/1600) | **2.3배** |

**핵심 질문**: Variable 조건에서 파산율이 2배 이상 높은 이유는 무엇인가? SAE 피처가 이 차이를 설명할 수 있는가?

---

## 🗂️ 디렉토리 구조

```
sae_condition_comparison/
├── src/                              # 소스 코드 (2,215 lines)
│   ├── condition_comparison.py       # 메인 분석 (3가지 분석)
│   ├── two_way_anova_analysis.py     # 2-way ANOVA (bet_type × outcome)
│   ├── prompt_component_analysis.py  # 프롬프트 구성 요소 분석
│   ├── prompt_complexity_analysis.py # 프롬프트 복잡도 분석
│   ├── prompt_combo_explorer.py      # 프롬프트 조합 탐색
│   └── utils.py                      # 공통 유틸리티 (490 lines)
│
├── configs/                          # 설정 파일
│   ├── analysis_config.yaml          # 메인 분석 설정
│   └── prompt_analysis_config.yaml   # 프롬프트 분석 설정
│
├── scripts/                          # 실행 및 시각화 스크립트
│   ├── run_analysis.sh               # 메인 분석 실행
│   ├── run_all_analyses.sh           # 전체 분석 파이프라인
│   ├── visualize_*.py                # 다양한 시각화 스크립트 (7개)
│   └── comprehensive_distribution_analysis.py
│
├── results/                          # 분석 결과
│   ├── condition_comparison_summary_*.json   # 요약 (125KB)
│   ├── four_way_*.json                       # 4-way ANOVA (LLaMA: 619MB, Gemma: 3.3GB)
│   ├── interaction_*.json                    # 상호작용 분석 (LLaMA: 534MB, Gemma: 2.9GB)
│   ├── variable_vs_fixed_*.json              # t-test 결과
│   ├── two_way_anova_*.json                  # 2-way ANOVA 결과
│   ├── prompt_component/, prompt_complexity/, prompt_combo/  # 프롬프트 분석
│   └── figures/                              # 생성된 그래프 (18개 PNG)
│
├── logs/                             # 실행 로그 (13개 로그 파일)
│
└── *.md                              # 문서 (12개 마크다운 파일)
    ├── README.md                     # 기본 사용법
    ├── ANALYSIS_ISSUES_REPORT.md     # ⚠️ 통계적 이슈 보고서
    ├── INTERACTION_ETA_PROBLEM_EXPLAINED.md  # Sparse feature 문제
    ├── TWO_WAY_ANOVA_GUIDE.md        # 2-way ANOVA 가이드
    ├── PROMPT_*_*.md                 # 프롬프트 분석 문서 (4개)
    └── SAE_Condition_Comparison_Results.md, ...
```

---

## 🔬 분석 파이프라인

### **Phase 1: 메인 조건 비교** (`condition_comparison.py`)

3가지 독립적인 분석을 수행:

#### Analysis 1: Variable vs Fixed (주효과)
- **방법**: Welch's t-test + Cohen's d
- **샘플**: Variable 1,600개 vs Fixed 1,600개
- **출력**: `variable_vs_fixed_*.json`
- **신뢰도**: ✅ **높음** (충분한 샘플, 올바른 통계)

```python
# 각 SAE 피처에 대해
t_stat, p_value = welch_ttest(variable_features, fixed_features)
cohens_d = (mean_variable - mean_fixed) / pooled_std
# FDR 보정 적용
```

#### Analysis 2: Four-Way ANOVA
- **방법**: One-way ANOVA + eta-squared
- **4개 그룹**:
  - Variable-Bankrupt (LLaMA: 108, Gemma: 465)
  - Variable-Safe (LLaMA: 1,492, Gemma: 1,135)
  - Fixed-Bankrupt (LLaMA: 42, Gemma: 205)
  - Fixed-Safe (LLaMA: 1,558, Gemma: 1,395)
- **출력**: `four_way_*.json` (3.3GB for Gemma!)
- **신뢰도**: ⚠️ **중간** (샘플 불균형, FDR 보정됨)

```python
# 4개 그룹의 평균 차이 검정
f_stat, p_value = f_oneway(VB, VS, FB, FS)
eta_squared = SS_between / SS_total
```

#### Analysis 3: Interaction (bet_type × outcome)
- **방법**: 2×2 교차 테이블 + 잔차 분석
- **목적**: 베팅 조건의 효과가 결과(파산/비파산)에 따라 다른가?
- **출력**: `interaction_*.json` (2.9GB)
- **신뢰도**: ❌ **낮음** (Sparse feature artifact - 아래 참조)

**⚠️ CRITICAL ISSUE**: 92%의 피처가 `interaction_eta ≈ 1.0`을 보임
- **원인**: 극도로 sparse한 피처 (99.88%가 0)
- **예시**: L1-3679는 3,200개 게임 중 단 4개에서만 활성화
- **해결**: `activation_rate >= 1%` 필터링 필요 (아직 미적용)

---

### **Phase 2: Two-Way ANOVA** (`two_way_anova_analysis.py`)

정통적인 2-way ANOVA 분석:
- **독립변수**: bet_type (Variable/Fixed) × outcome (Bankrupt/Safe)
- **종속변수**: 각 SAE 피처 활성화
- **주효과**: bet_type, outcome
- **상호작용**: bet_type × outcome

**⚠️ 구현 주의사항**:
- 현재 `utils.py:294-391`의 `two_way_anova_simple()`은 **근사 계산**
- Main effects는 separate one-way ANOVA로 계산 (정확함)
- Interaction은 "difference of differences"로 추정 (근사치)
- 계산 효율성 위해 선택 (1M+ features × statsmodels는 너무 느림)
- **권장**: 상위 100개 피처는 `statsmodels.ols()` + `anova_lm()`로 재검증

---

### **Phase 3: 프롬프트 분석** (3개 스크립트)

베팅 조건 차이가 프롬프트 구성에서 기인하는지 분석:

#### 3.1 Component Analysis (`prompt_component_analysis.py`)
프롬프트를 5개 구성 요소로 분해:
- **G** (Goal): 목표 제시 부분
- **M** (Money): 현재 잔고 정보
- **P** (Progress): 진행 상황
- **R** (Reminder): 규칙 상기
- **W** (Win rate): 승률 정보

각 구성 요소를 제거했을 때 SAE 피처 변화 측정.

#### 3.2 Complexity Analysis (`prompt_complexity_analysis.py`)
프롬프트 길이/복잡도와 SAE 피처의 관계:
- Token 수
- 문장 수
- 정보량 (엔트로피)

#### 3.3 Combo Explorer (`prompt_combo_explorer.py`)
구성 요소 조합의 효과 탐색:
- 2^5 = 32개 가능한 조합
- 각 조합에서 피처 활성화 패턴 분석

---

## 📊 주요 발견 (예비 결과)

### 1. 모델별 인코딩 차이

**LLaMA (L12-15 집중)**:
- **베팅 조건 인코딩**: L14-12265 (eta² = 0.850)
  - Fixed-Bankrupt: 0.217, Fixed-Safe: 0.256
  - Variable-Bankrupt: 0.008, Variable-Safe: 0.002
  - → **Fixed 조건에서 명확히 높은 활성화**

**Gemma (L26-40 집중)**:
- **결과 인코딩**: L40-108098
  - Bankrupt 평균: 30.7, Safe 평균: 0.5
  - → **Bankrupt에서 50배 높은 활성화**

⚠️ **주의**: 절대값 비교는 무의미 (SAE 스케일이 다름). 상대적 패턴만 해석 가능.

### 2. 레이어 분포

```
LLaMA:
  Layer 1:  299 features analyzed (dead features 제외)
  Layer 12: 979 features
  Layer 31: 1,171 features

Gemma:
  Layer당 131K features (전체)
  분석된 수는 로그에서 확인 필요
```

---

## ⚠️ 통계적 이슈 및 주의사항

### Issue 1: Sparse Feature Artifact (가장 심각)
- **영향 범위**: Interaction 분석 (Analysis 3)
- **증상**: 92%의 피처가 interaction_eta ≈ 1.0
- **원인**: 활성화율 < 0.12%인 극도로 sparse한 피처
- **해결**:
  ```python
  min_activation_rate = 0.01  # 1% 이상
  min_mean_activation = 0.001
  ```
- **상태**: 🔴 **미해결** (필터링 코드 존재하지만 아직 적용 안 됨)

### Issue 2: 샘플 크기 불균형
- **Fixed-Bankrupt: 42개** (LLaMA) - 매우 작음
- **영향**: Four-Way ANOVA의 통계적 검정력 감소
- **완화**:
  - Analysis 1은 영향 없음 (전체 Variable vs Fixed)
  - FDR 보정으로 다중 비교 보정됨
  - Bootstrap CI로 안정성 검증 고려

### Issue 3: Two-Way ANOVA 근사 계산
- **현재**: Separate one-way ANOVAs + difference-of-differences
- **이유**: 계산 효율성 (1M+ features)
- **대안**: 상위 100개 피처는 statsmodels로 정확히 재계산

### Issue 4: Dead Feature 처리
- **방법**: `if np.std(v_vals) == 0 and np.std(f_vals) == 0: continue`
- **결과**: 레이어마다 분석된 피처 수가 다름
- **평가**: ✅ 올바른 처리

---

## 📈 생성된 시각화

### 메인 Figure (4개)
1. **fig1_four_way_heatmap.png**: 4-way ANOVA 히트맵
2. **fig2_layer_effect_size.png**: 레이어별 효과 크기
3. **fig3_bet_vs_outcome_scatter.png**: 베팅 vs 결과 효과 산점도
4. **fig4_top_features_bar.png**: 상위 피처 막대그래프

### Two-Way ANOVA Figure (4개)
- `two_way_anova_heatmap_bet_type_*.png`: 베팅 타입 주효과
- `two_way_anova_heatmap_outcome_*.png`: 결과 주효과
- `two_way_anova_heatmap_interaction_*.png`: 상호작용
- `two_way_anova_heatmap_total_*.png`: 전체 효과

### 프롬프트 분석 Figure (results/figures/, 8개)
- `component_*.png`: 구성 요소 분석
- `complexity_*.png`: 복잡도 분석
- `combo_*.png`: 조합 탐색
- `comprehensive_summary_*.png`: 종합 요약

---

## 🚀 실행 방법

### 1. 환경 설정
```bash
conda activate llama_sae_env
cd /mnt/c/Users/oollccddss/git/llm-addiction/exploratory_experiments/additional_experiments/sae_condition_comparison
```

### 2. 메인 분석 실행
```bash
# LLaMA 분석 (CPU-only, ~15-30분)
python -m src.condition_comparison --model llama

# Gemma 분석 (CPU-only, ~1-2시간, 131K features/layer)
python -m src.condition_comparison --model gemma

# 또는 스크립트 사용
bash scripts/run_analysis.sh llama
```

### 3. Two-Way ANOVA 실행
```bash
python -m src.two_way_anova_analysis --model llama
```

### 4. 프롬프트 분석 실행
```bash
# Component analysis
python -m src.prompt_component_analysis --model llama

# Complexity analysis
python -m src.prompt_complexity_analysis --model llama

# Combo explorer
python -m src.prompt_combo_explorer --model llama
```

### 5. 시각화
```bash
# 메인 결과 시각화
python scripts/visualize_results_improved.py

# Two-way ANOVA 히트맵
python scripts/visualize_two_way_anova_heatmap.py

# 프롬프트 분석 시각화
python scripts/visualize_prompt_results.py

# 전체 파이프라인 (분석 + 시각화)
bash scripts/run_all_analyses.sh
```

---

## 📦 데이터 의존성

### 입력 데이터
- **SAE 피처**: `paper_experiments/llama_sae_analysis/results/layer_{N}_features.npz`
  - `features`: (n_games, n_features) 배열
  - `outcomes`: (n_games,) - 'bankrupt' 또는 'voluntary_stop'
  - `game_ids`: (n_games,) - JSON 인덱스와 매핑

- **원본 실험 JSON**: `/mnt/c/Users/oollccddss/git/data/llm-addiction/slot_machine/{model}/final_{model}_*.json`
  - `bet_type`: 'variable' 또는 'fixed'
  - 기타 게임 메타데이터

### 출력 데이터
- **Summary**: 125KB (상위 피처 요약)
- **Full results**:
  - LLaMA: 619MB (four_way) + 534MB (interaction)
  - Gemma: 3.3GB (four_way) + 2.9GB (interaction)

---

## 🔍 코드 구조

### `src/condition_comparison.py` (433 lines)
```python
class ConditionComparisonAnalyzer:
    def analyze_variable_vs_fixed_layer(layer) -> List[dict]
        # Analysis 1: t-test + Cohen's d

    def analyze_four_way_layer(layer) -> List[dict]
        # Analysis 2: One-way ANOVA + eta-squared

    def analyze_interaction_layer(layer) -> List[dict]
        # Analysis 3: 2×2 interaction + eta
```

### `src/utils.py` (490 lines)
```python
class DataLoader:
    def load_layer_features_grouped(layer) -> dict
        # NPZ 로드 + JSON 매핑 + 그룹화

class StatisticalAnalyzer:
    def welch_ttest(x, y) -> (t_stat, p_value)
    def compute_cohens_d(x, y) -> float
    def fdr_correction(p_values) -> (reject, adjusted_p)
    def two_way_anova_simple(data, factor1, factor2) -> dict
        # ⚠️ 근사 계산 (주석 참조)
```

---

## 📚 중요 문서

### 필수 읽기
1. **ANALYSIS_ISSUES_REPORT.md** (274 lines) ⚠️
   - 5가지 통계적 이슈 상세 설명
   - Sparse feature artifact 원인 분석
   - 신뢰도 평가 및 권장 조치

2. **INTERACTION_ETA_PROBLEM_EXPLAINED.md**
   - 왜 92%의 피처가 eta=1.0인가?
   - 수치적 예시와 시각화

3. **TWO_WAY_ANOVA_GUIDE.md**
   - 2-way ANOVA 구현 설명
   - 근사 vs 정확 계산 비교

### 참고 자료
4. **PROMPT_COMPONENT_README.md**: 프롬프트 분석 설계
5. **FIGURE1_HEATMAP_GUIDE.md**: 히트맵 해석 가이드
6. **SAE_Figure_Analysis_Guide.md**: 전체 Figure 해석

---

## 🎯 분석 결과 신뢰도 평가

| 분석 | 샘플 크기 | 통계 방법 | 신뢰도 | 논문 사용 권장 |
|------|----------|-----------|--------|---------------|
| **Analysis 1: Variable vs Fixed** | 1600 vs 1600 | Welch's t-test | ✅ **높음** | Main Figure |
| **Analysis 2: Four-Way ANOVA** | VB:108, VS:1492, FB:42, FS:1558 | One-way ANOVA | ⚠️ **중간** | Main/Supplementary |
| **Analysis 3: Interaction** | 동일 | 2×2 잔차 분석 | ❌ **낮음** | 🔴 **재분석 필요** |
| **Two-Way ANOVA** | 동일 | 근사 계산 | ⚠️ **중간** | 상위 100개 검증 후 |
| **프롬프트 분석** | Component별 다름 | t-test | ✅ **높음** | Supplementary |

---

## ✅ TODO / 개선 사항

### 즉시 필요 (논문 제출 전)
- [ ] **Sparse feature 필터링 적용**
  - `activation_rate >= 0.01` 조건 추가
  - Interaction 분석 재실행
  - 결과 비교 (before/after)

- [ ] **상위 100개 피처 statsmodels 검증**
  ```python
  import statsmodels.api as sm
  from statsmodels.formula.api import ols
  # Two-way ANOVA 정확히 재계산
  ```

- [ ] **Bootstrap CI 추가** (Fixed-Bankrupt n=42 안정성)

### 문서화
- [ ] **Limitations 섹션 작성**
  - 샘플 불균형 명시
  - SAE 스케일 차이 설명
  - Dead feature 제외 기준

- [ ] **Methods 섹션**
  - Two-way ANOVA 근사 계산 설명
  - 왜 이 방법을 선택했는지 (계산 효율성)

### 선택적 개선
- [ ] **Neuronpedia 링크 생성** (results/neuronpedia_links.txt)
- [ ] **Interactive 시각화** (Plotly/Dash)
- [ ] **Cross-model 비교** (LLaMA vs Gemma 직접 비교 figure)

---

## 💡 핵심 Insight

### 1. 모델별 인코딩 전략 차이
- **LLaMA**: 베팅 조건을 중간 레이어(L12-15)에 인코딩
- **Gemma**: 최종 결과를 후기 레이어(L26-40)에 인코딩
- → **Architectural difference**: LLaMA는 조건 민감, Gemma는 결과 민감

### 2. Variable 조건의 위험 증폭 메커니즘
- Variable 조건에서 특정 피처가 더 활성화
- 이 피처들이 위험 감수 의사결정과 연관
- → **Autonomy effect**: 선택의 자유 → 위험 증가

### 3. Sparse Feature의 중요성
- 대부분의 SAE 피처는 극도로 sparse (>99% zeros)
- 하지만 일부 dense feature가 큰 효과 크기 (Cohen's d > 4)
- → **Sparsity vs Impact**: 희소성과 중요도는 독립적

---

## 🔗 관련 실험

- **Upstream**: `paper_experiments/llama_sae_analysis/` (Phase 1 feature extraction)
- **Parallel**: `paper_experiments/slot_machine_6models/` (원본 행동 데이터)
- **Related**: `exploratory_experiments/lr_classification_experiment/` (Hidden state 분류)

---

## 📞 문의 및 이슈

실험 관련 질문은 다음 문서 참조:
1. **통계 이슈**: `ANALYSIS_ISSUES_REPORT.md`
2. **코드 사용법**: `README.md`
3. **결과 해석**: `SAE_Condition_Comparison_Results.md`
4. **Figure 설명**: `results/SAE_Figure_Analysis_Guide.md`

---

## 📝 업데이트 로그

- **2026-01-27**: 초기 분석 완료 (LLaMA, Gemma)
- **2026-02-01**: 프롬프트 분석 추가 (3개 스크립트)
- **2026-02-01**: ANALYSIS_ISSUES_REPORT 작성 (통계적 이슈 발견)
- **2026-02-02**: Two-Way ANOVA 분석 추가
- **2026-02-02**: 실험 정리 문서 작성 (이 파일)

---

**마지막 실행**: 2026-02-02 16:27
**상태**: ⚠️ Sparse feature 필터링 필요
**다음 단계**: 재분석 → 논문 Figure 확정
