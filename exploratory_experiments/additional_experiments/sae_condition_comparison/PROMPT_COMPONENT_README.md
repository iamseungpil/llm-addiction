# Prompt Component Analysis - README

## 개요

이 분석은 prompt의 개별 구성 요소(G/M/R/W/P)가 SAE feature 활성화 패턴에 미치는 영향을 조사합니다.

- **분석 유형**: Prompt Component × Outcome Two-way ANOVA
- **연구 질문**: 각 prompt component가 bankruptcy vs safe outcome에서 어떤 feature를 차별적으로 활성화하는가?
- **모델**: LLaMA-3.1-8B, Gemma-2-9B

---

## 실행 방법

### 1. 전체 분석 실행 (권장)

```bash
# LLaMA 분석 (모든 5개 components)
python3 -m src.prompt_component_analysis --model llama

# Gemma 분석 (모든 5개 components)
python3 -m src.prompt_component_analysis --model gemma
```

### 2. 특정 component만 분석

```bash
# Goal-setting component만
python3 -m src.prompt_component_analysis --model llama --components G

# Goal-setting과 Maximize만
python3 -m src.prompt_component_analysis --model llama --components G M
```

### 3. 시각화 생성

```bash
# LLaMA 결과 시각화
python3 scripts/visualize_prompt_results.py --model llama

# Gemma 결과 시각화
python3 scripts/visualize_prompt_results.py --model gemma

# 양쪽 모두 + 비교 시각화
python3 scripts/visualize_prompt_results.py --model both
```

---

## 출력 파일

### 분석 결과 (JSON)

```
results/prompt_component/
├── G_llama_20260201_HHMMSS.json    # Goal-setting
├── M_llama_20260201_HHMMSS.json    # Maximize
├── R_llama_20260201_HHMMSS.json    # Hidden patterns
├── W_llama_20260201_HHMMSS.json    # Win multiplier
├── P_llama_20260201_HHMMSS.json    # Win rate
└── (gemma versions...)
```

### 시각화 (PNG/CSV)

```
results/figures/
├── component_layer_heatmap_llama.png   # Component × Layer heatmap
├── component_barplot_llama.png          # Top features per component
├── component_summary_table_llama.png    # Summary table
├── component_summary_llama.csv          # Summary CSV
├── (gemma versions...)
└── model_comparison.png                 # LLaMA vs Gemma comparison
```

### 로그 파일

```
logs/
├── prompt_component_llama_20260201_HHMMSS.log
└── prompt_component_gemma_20260201_HHMMSS.log
```

---

## Prompt Components 정의

| ID | 이름 | 설명 | 프롬프트 예시 |
|----|------|------|--------------|
| **G** | Goal-setting | 목표 설정 유도 | "set a target amount before starting" |
| **M** | Maximize | 보상 최대화 강조 | "maximize the reward as much as possible" |
| **R** | Risk/Patterns | 숨겨진 패턴 암시 | "may have hidden patterns you can discover" |
| **W** | Win multiplier | 승리 배율 제시 | "Win multiplier: 3.0 times your bet" |
| **P** | Probability | 승률 정보 제공 | "Win rate: 30%" |

### 샘플 크기
- **Component 포함**: 16개 조합 × 100 games = 1,600 games
- **Component 미포함**: 16개 조합 × 100 games = 1,600 games
- **총 3,200 games** (충분한 통계적 검정력)

---

## 주요 통계 지표

### Interaction Effect (핵심)
- **interaction_eta**: Component × Outcome 상호작용 효과 크기
  - η² ≥ 0.14: Large effect
  - η² ≥ 0.06: Medium effect
  - η² ≥ 0.01: Small effect (minimum threshold)

### Main Effects
- **component_eta**: Component 유무의 평균 효과
- **outcome_eta**: Outcome(bankruptcy/safe)의 평균 효과

### Group Means
- `False_bankruptcy`: Component 없을 때, Bankruptcy
- `False_voluntary_stop`: Component 없을 때, Safe
- `True_bankruptcy`: Component 있을 때, Bankruptcy
- `True_voluntary_stop`: Component 있을 때, Safe

---

## 결과 해석 예시

### 예시 1: Component-Specific Bankruptcy Encoding

```json
{
  "layer": 14,
  "feature_id": 12265,
  "component": "G",
  "interaction_eta": 0.85,
  "group_means": {
    "False_bankruptcy": 0.005,
    "False_voluntary_stop": 0.005,
    "True_bankruptcy": 0.250,  // 매우 높음!
    "True_voluntary_stop": 0.001
  }
}
```

**해석**:
- Goal-setting(G)이 **있을 때만** bankruptcy에서 feature가 강하게 활성화됨
- Goal-setting이 없으면 bankruptcy와 safe의 차이가 거의 없음
- **결론**: Goal-setting이 bankruptcy risk encoding을 촉발하는 trigger 역할

### 예시 2: Weak Interaction

```json
{
  "layer": 25,
  "feature_id": 5432,
  "component": "P",
  "interaction_eta": 0.02,
  "group_means": {
    "False_bankruptcy": 0.100,
    "False_voluntary_stop": 0.095,
    "True_bankruptcy": 0.105,
    "True_voluntary_stop": 0.098
  }
}
```

**해석**:
- Probability component의 영향이 매우 약함
- 모든 그룹에서 activation이 비슷함
- **결론**: Win rate 정보는 이 feature의 outcome encoding에 거의 영향 없음

---

## Sparse Feature 문제 및 해결

### 문제
- SAE features는 inherently sparse (L1 penalty)
- Activation rate < 1%인 features는 interaction η² ≈ 1.0 artifact 발생
- 상세 내용: `ANALYSIS_ISSUES_REPORT.md` 참조

### 현재 필터링
```yaml
sparse_filter:
  enabled: true
  min_activation_rate: 0.01  # 1% 이상 활성화 필요
  min_mean_activation: 0.001  # 평균 활성화 0.001 이상
```

### 해석 시 주의
- ⚠️ **η² = 1.0 features는 의심**: 대부분 sparse artifact
- ✅ **0.01 < η² < 0.90**: Reliable interaction
- ❌ **η² ≈ 1.0 + 한 그룹만 활성화**: Sparse artifact

---

## 시각화 해석

### 1. Component × Layer Heatmap
- **색상**: Mean interaction η² (top 20 features)
- **X축**: Layer (1-31 for LLaMA)
- **Y축**: Component (G/M/R/W/P)

**해석 팁**:
- 빨간색 영역: 해당 component가 해당 layer에서 강한 interaction
- 파란색 영역: 약한 interaction
- **Layer 분포 패턴**으로 component의 작동 시점 추론

### 2. Component Barplot
- 각 component의 top 10 features
- 색상 코드:
  - Red (η² > 0.5): 매우 강한 효과
  - Orange (0.2-0.5): 중간 효과
  - Blue (< 0.2): 약한 효과

**해석 팁**:
- Red bars가 많음 → 해당 component의 강한 영향력
- 대부분 blue → 약한 효과 또는 sparse artifacts

### 3. Model Comparison
- LLaMA vs Gemma의 significant features 수 비교

**해석 예시**:
- LLaMA: G에서 많은 features → Goal-setting에 민감
- Gemma: W에서 많은 features → Win multiplier에 민감
- **→ 모델 간 prompt sensitivity 차이**

---

## 실행 시간

- **LLaMA (31 layers)**: ~15분 per component → 총 ~75분
- **Gemma (42 layers)**: ~25분 per component → 총 ~125분
- **시각화**: ~1분

**권장**: 백그라운드에서 실행

```bash
# Background 실행
nohup python3 -m src.prompt_component_analysis --model llama > llama_analysis.log 2>&1 &
nohup python3 -m src.prompt_component_analysis --model gemma > gemma_analysis.log 2>&1 &

# 진행 상황 확인
tail -f llama_analysis.log
```

---

## 문제 해결

### Q: "No module named 'utils'" 에러
```bash
# src/ 디렉토리에서 실행하지 말고, 프로젝트 루트에서 실행
cd /path/to/sae_condition_comparison
python3 -m src.prompt_component_analysis --model llama
```

### Q: 결과 파일이 비어있음
```bash
# NPZ 파일 경로 확인
ls /mnt/c/Users/oollccddss/git/data/llm-addiction/sae_patching/corrected_sae_analysis/llama/

# JSON 파일 경로 확인
ls /mnt/c/Users/oollccddss/git/data/llm-addiction/slot_machine/llama/
```

### Q: Sparse filtering이 너무 많은 features를 제거함
```yaml
# configs/prompt_analysis_config.yaml 수정
sparse_filter:
  enabled: true
  min_activation_rate: 0.005  # 0.01에서 0.005로 완화
  min_mean_activation: 0.0005  # 0.001에서 0.0005로 완화
```

### Q: 시각화에서 "No data" 표시
```bash
# 결과 파일 존재 확인
ls results/prompt_component/*_llama_*.json

# 최신 파일인지 확인 (timestamp 비교)
```

---

## 추가 분석

### 1. Three-way ANOVA (Component × Outcome × Bet_type)
```python
# utils.py의 two_way_anova_simple을 three_way로 확장 필요
# 구현 예정
```

### 2. Prompt Complexity Analysis
```bash
# Complexity level (0-5)별 분석
python3 -m src.prompt_complexity_analysis --model llama
```

### 3. Individual Combo Analysis
```bash
# 32개 조합 각각에 대한 탐색적 분석
python3 -m src.prompt_combo_explorer --model llama
```

---

## 참고 문서

1. **PROMPT_ANALYSIS_GUIDE.md**: 상세 결과 해석 가이드
2. **ANALYSIS_ISSUES_REPORT.md**: Sparse feature 문제 설명
3. **INTERACTION_ETA_PROBLEM_EXPLAINED.md**: Interaction artifact 설명
4. **CLAUDE.md**: 전체 프로젝트 구조 및 실험 설명

---

## Citation

이 분석 방법을 사용하는 경우:

```bibtex
@misc{prompt_component_analysis_2026,
  title={Prompt Component Analysis for SAE Feature Activation},
  author={LLM Addiction Research Team},
  year={2026},
  note={Supplementary analysis for ICLR 2026 submission}
}
```

---

## 라이선스

이 프로젝트의 일부로서 동일한 라이선스 적용

---

**Last Updated**: 2026-02-01
**Status**: Analysis in progress
**Contact**: See main project README
