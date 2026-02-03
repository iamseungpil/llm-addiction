# LLM 중독 연구: SAE 기반 도메인 일반화 실험

**작성일**: 2026-02-03
**연구 목표**: LLM의 중독 현상이 다양한 도메인과 모델에서 동일한 신경 메커니즘(SAE features)으로 발현되는지 검증

---

## 📋 연구 배경

### 기존 연구 성과 (ICLR 2026 제출)

1. **행동 수준 중독 현상 발견**
   - 6개 모델에서 도박 중독 패턴 확인 (LLaMA, Gemma, GPT-4o-mini, Claude, Gemini)
   - 자기조절 실패: 베팅 공격성, 극단적 선택, 손실 추격
   - 목표 조절 장애: 목표 달성 후 상향 조정 (20% → 50%)

2. **자율성 효과 발견 (Finding 3)**
   - Variable betting → +3.3% 파산율 증가 vs Fixed betting
   - "선택의 자유 → 위험 감수 증가" 메커니즘

3. **신경 메커니즘 식별 (LLaMA, Slot Machine)**
   - **Phase2 Correlation**: 6,641 risky features, 5,979 safe features (Cohen's d 기반)
   - **Phase4 Causal Validation**: 112개 features가 행동에 인과적 영향 (activation patching)

### 현재 연구 질문

**Q1**: 중독 현상이 도메인에 걸쳐 일반화되는가?
- Slot Machine 외에 다른 도박 과제(Loot Box, Blackjack)에서도 동일한 SAE features 활성화?

**Q2**: 세팅 변화가 feature 활성화를 어떻게 조절하는가?
- Variable vs Fixed betting이 feature 수준에서 어떻게 다른가?
- Prompt components (Goal, Warning, Practice)가 feature 활성화에 영향을 주는가?

**Q3**: 현재 통계 분석이 유효한가?
- Condition comparison 분석의 92% artifact 문제 해결
- Two-way ANOVA 근사 계산의 정확성 검증

---

## 🎯 구현된 3-Part 실험 파이프라인

### Part 1: Condition Comparison 통계 이슈 수정 (~10시간, CPU-only)

**문제점:**
- Analysis 3 (Interaction): 92%가 통계적 artifact (eta² ≈ 1.0)
- 원인: 극도로 sparse한 features (활성화율 <0.12%, 3,200 게임 중 4개만 활성)
- Two-way ANOVA가 근사 계산 사용 (separate one-way ANOVAs)

**해결책:**
1. **Sparse Feature 필터링**: `filter_sparse_features()` 추가
   - 최소 활성화율: 1% (32/3,200 게임)
   - 최소 평균 활성화: 0.001

2. **Statsmodels 검증**: 상위 100개 features를 정확한 2-way ANOVA로 재검증
   - `statsmodels.formula.api.ols()` + `anova_lm()`

3. **중독 Features 교차 검증**: Condition features ↔ Addiction features 연결
   - **Risk Amplification**: Variable-higher ∩ Risky → 중독 증가 메커니즘
   - **Protective**: Fixed-higher ∩ Safe → 보호 메커니즘

**구현 파일:**
- `exploratory_experiments/additional_experiments/sae_condition_comparison/src/utils.py` (filter_sparse_features)
- `exploratory_experiments/additional_experiments/sae_condition_comparison/src/condition_comparison.py` (수정)
- `exploratory_experiments/additional_experiments/sae_condition_comparison/src/cross_reference_addiction_features.py` (신규)

**실행:**
```bash
cd /mnt/c/Users/oollccddss/git/llm-addiction

# 1. LLaMA 재분석 (1-2시간)
python -m exploratory_experiments.additional_experiments.sae_condition_comparison.src.condition_comparison --model llama

# 2. Gemma 재분석 (1-2시간)
python -m exploratory_experiments.additional_experiments.sae_condition_comparison.src.condition_comparison --model gemma

# 3. Cross-reference 분석 (30분)
python exploratory_experiments/additional_experiments/sae_condition_comparison/src/cross_reference_addiction_features.py --model llama
python exploratory_experiments/additional_experiments/sae_condition_comparison/src/cross_reference_addiction_features.py --model gemma
```

**예상 결과:**
- Artifact rate: 92% → <10%
- Risk Amplification features: 500-1000개
- Protective features: 400-800개

---

### Part 2: Alternative Paradigms SAE 인프라 구축 (~14시간 개발 + 82시간 GPU)

**목표**: Loot Box, Blackjack 실험에 Phase 1-2 SAE 파이프라인 구축

#### 2-1. 실험 재실행 (hidden states + full_prompt 저장)

**문제**: 기존 실험 데이터에 prompts가 저장 안 됨 → SAE 분석 불가

**해결**: `full_prompt` 필드 추가 후 재실행

**수정 파일:**
- `exploratory_experiments/alternative_paradigms/src/common/model_loader.py`
  - `generate_with_hidden_states()` 메서드 추가

- `exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py`
  - `trials` 리스트에 `full_prompt` 필드 추가

- `exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py` (신규 구현)
  - Blackjack 게임 로직 + 실험 runner
  - `rounds` 리스트에 `full_prompt` 필드 추가

**실행:**
```bash
# Loot Box 재실행 (2 models × 4h = 8h GPU)
python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py --model llama --gpu 0
python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py --model gemma --gpu 0

# Blackjack 재실행 (2 models × 4h = 8h GPU)
python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py --model llama --gpu 0 --bet-type variable
python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py --model gemma --gpu 0 --bet-type variable

# 총 GPU 시간: 16h
```

**출력:**
- `/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms/lootbox/*.json`
- `/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms/blackjack/*.json`

#### 2-2. Phase 1: SAE Feature Extraction

**파이프라인**: JSON → prompts → hidden states → SAE encoding → NPZ

**구현 파일:**
- `exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py`

**실행:**
```bash
# Loot Box
python exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py --paradigm lootbox --model llama --gpu 0
# LLaMA: 7 layers × 1h = 7h GPU

python exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py --paradigm lootbox --model gemma --gpu 0
# Gemma: 26 layers × 1h = 26h GPU

# Blackjack
python exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py --paradigm blackjack --model llama --gpu 0
# LLaMA: 7h GPU

python exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py --paradigm blackjack --model gemma --gpu 0
# Gemma: 26h GPU

# 총 GPU 시간: (7 + 26) × 2 paradigms = 66h
```

**출력:**
- `/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms/{paradigm}/sae_features/layer_*_features.npz`

**NPZ 포맷:**
```python
{
    'features': (n_samples, n_features),  # SAE activations
    'outcomes': (n_samples,),  # 'bankrupt' or 'voluntary_stop'
    'game_ids': (n_samples,),
    'layer': int,
    'model_type': str
}
```

#### 2-3. Phase 2: Correlation Analysis

**파이프라인**: NPZ → Welch's t-test + Cohen's d → FDR correction → Risky/Safe features

**구현 파일:**
- `exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py`

**실행:**
```bash
# Loot Box
python exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py --paradigm lootbox --model llama
# ~1h CPU

python exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py --paradigm lootbox --model gemma
# ~2h CPU (more layers)

# Blackjack
python exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py --paradigm blackjack --model llama
# ~1h CPU

python exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py --paradigm blackjack --model gemma
# ~2h CPU

# 총 CPU 시간: ~6h
```

**출력:**
- `correlation_all_features_{timestamp}.json`: 모든 features 통계
- `correlation_significant_{timestamp}.json`: 유의미한 features (top 100)
- `correlation_summary_{timestamp}.json`: 요약 통계

---

### Part 3: Cross-Domain 비교 프레임워크 (~12시간, CPU-only)

**목표**: 3개 도메인(Slot Machine, Loot Box, Blackjack) 간 feature overlap 분석

**구현 파일:**
- `exploratory_experiments/additional_experiments/cross_domain_sae_comparison/src/cross_domain_analysis.py`

**분석 내용:**
1. **Pairwise Jaccard Similarity**: 도메인 간 feature 중복도
2. **Core Features**: 2개 이상 도메인에 나타나는 features
3. **Universal Features**: 3개 도메인 모두에 나타나는 features

**실행:**
```bash
cd exploratory_experiments/additional_experiments/cross_domain_sae_comparison

# LLaMA 분석 (~3h CPU)
python src/cross_domain_analysis.py --model llama

# Gemma 분석 (~3h CPU)
python src/cross_domain_analysis.py --model gemma
```

**출력:**
```json
{
  "pairwise_overlaps": [
    {"domain1": "slot_machine", "domain2": "lootbox", "risky_jaccard": 0.28, ...},
    {"domain1": "slot_machine", "domain2": "blackjack", "risky_jaccard": 0.25, ...},
    {"domain1": "lootbox", "domain2": "blackjack", "risky_jaccard": 0.32, ...}
  ],
  "core_features": {
    "core_risky_count": 150,  // 2+ domains
    "core_safe_count": 120,
    "universal_risky_count": 25,  // All 3 domains
    "universal_safe_count": 18
  }
}
```

---

## 📊 성공 기준

| 지표 | 강한 일반화 | 중간 일반화 (예상) | 약한 일반화 (최소) |
|------|------------|-------------------|-------------------|
| **Core features** | ≥150 risky + ≥150 safe | ≥80 (2+ domains) | ≥40 (2 domains) |
| **Jaccard similarity** | > 0.40 | > 0.25 | > 0.10 |
| **Universal features** | ≥30 (all 3) | ≥15 (all 3) | ≥5 (all 3) |

**Random baseline**: Jaccard ≈ 0.01 (1% FDR × 1% FDR)

---

## 🗓️ 실행 순서

### Week 1: Part 1 완료 (✅ 코드 구현 완료)
```bash
# Day 1-2: LLaMA/Gemma 재분석 (4h CPU)
cd /mnt/c/Users/oollccddss/git/llm-addiction
python -m exploratory_experiments.additional_experiments.sae_condition_comparison.src.condition_comparison --model llama
python -m exploratory_experiments.additional_experiments.sae_condition_comparison.src.condition_comparison --model gemma

# Day 2-3: Cross-reference 분석 (1h CPU)
python exploratory_experiments/additional_experiments/sae_condition_comparison/src/cross_reference_addiction_features.py --model llama
python exploratory_experiments/additional_experiments/sae_condition_comparison/src/cross_reference_addiction_features.py --model gemma
```

### Week 2: Part 2 실험 재실행 (16h GPU)
```bash
# Loot Box
python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py --model llama --gpu 0
python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py --model gemma --gpu 0

# Blackjack
python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py --model llama --gpu 0 --bet-type variable
python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py --model gemma --gpu 0 --bet-type variable
```

### Week 3: Part 2 SAE 추출 (66h GPU)
```bash
# Phase 1: LLaMA (14h GPU)
python exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py --paradigm lootbox --model llama --gpu 0
python exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py --paradigm blackjack --model llama --gpu 0

# Phase 1: Gemma (52h GPU)
python exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py --paradigm lootbox --model gemma --gpu 0
python exploratory_experiments/alternative_paradigms/src/common/phase1_feature_extraction.py --paradigm blackjack --model gemma --gpu 0

# 병렬 실행 시 2 GPU 사용으로 단축 가능
```

### Week 4: Part 2-3 분석 (12h CPU)
```bash
# Phase 2: Correlation (6h CPU)
python exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py --paradigm lootbox --model llama
python exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py --paradigm lootbox --model gemma
python exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py --paradigm blackjack --model llama
python exploratory_experiments/alternative_paradigms/src/common/phase2_correlation_analysis.py --paradigm blackjack --model gemma

# Cross-domain 비교 (6h CPU)
cd exploratory_experiments/additional_experiments/cross_domain_sae_comparison
python src/cross_domain_analysis.py --model llama
python src/cross_domain_analysis.py --model gemma
```

---

## 🖥️ GPU 요구사항

| 단계 | LLaMA | Gemma | 총합 |
|------|-------|-------|------|
| Part 1 (재분석) | 0h | 0h | **0h** (CPU-only) |
| Part 2-1 (실험 재실행) | 8h | 8h | **16h** |
| Part 2-2 (Phase 1) | 14h | 52h | **66h** |
| Part 2-3 (Phase 2) | 0h | 0h | **0h** (CPU-only) |
| Part 3 (Cross-domain) | 0h | 0h | **0h** (CPU-only) |
| **총 GPU 시간** | 22h | 60h | **82h** |

**병렬화**: 2 GPU 사용 시 ~41h (동시 LLaMA + Gemma)

**VRAM 요구**:
- LLaMA-3.1-8B: 19GB
- Gemma-2-9B: 22GB

---

## 📁 최종 출력 파일 구조

```
/mnt/c/Users/oollccddss/git/data/llm-addiction/
├── sae_patching/corrected_sae_analysis/
│   ├── llama/
│   │   ├── correlation_all_features_*.json  (기존 Slot Machine)
│   │   └── correlation_summary_*.json
│   └── gemma/
│       └── ...
│
├── alternative_paradigms/
│   ├── lootbox/
│   │   ├── llama_lootbox_*.json  (재실행 결과, full_prompt 포함)
│   │   ├── gemma_lootbox_*.json
│   │   └── sae_features/
│   │       ├── layer_*_features.npz  (Phase 1)
│   │       ├── correlation_all_features_*.json  (Phase 2)
│   │       └── correlation_summary_*.json
│   │
│   └── blackjack/
│       ├── llama_blackjack_*.json
│       ├── gemma_blackjack_*.json
│       └── sae_features/
│           ├── layer_*_features.npz
│           └── ...
│
└── exploratory_experiments/additional_experiments/
    ├── sae_condition_comparison/results/
    │   ├── variable_vs_fixed_llama_*.json  (Part 1 재분석)
    │   ├── interaction_llama_*.json
    │   └── cross_reference_llama_*.json  (Part 1 교차 검증)
    │
    └── cross_domain_sae_comparison/results/
        ├── cross_domain_overlap_llama_*.json  (Part 3)
        └── cross_domain_overlap_gemma_*.json
```

---

## 📊 논문 기여

### 1. Setting Modulation (Part 1)
"Variable betting conditions selectively activate risk-amplifying features (N=XXX), explaining 2.6× bankruptcy rate increase. Fixed conditions activate protective features (N=XXX)."

### 2. Domain Generalization (Part 2-3)
"XXX core addiction features generalize across slot machine, loot box, and blackjack paradigms (Jaccard=0.XX, Cohen's h=X.XX), demonstrating domain-agnostic neural substrates of gambling addiction."

### 3. Methodological Rigor (Part 1)
"After sparse feature filtering (activation rate >= 1%), interaction analysis artifact rate reduced from 92% to <10%, validated with exact statsmodels 2-way ANOVA."

---

## ⚠️ 주의사항

1. **GPU 메모리**: 각 모델별로 별도 GPU 사용 권장 (LLaMA 19GB, Gemma 22GB)
2. **Conda 환경**: `conda activate llama_sae_env` 필수
3. **데이터 검증**: 실험 재실행 시 `full_prompt` 필드 존재 확인
4. **백업**: Phase 1 완료 후 NPZ 파일 백업 (재생성 시간 길음)
5. **Sparse 필터링**: Part 1 전에 실행 필수 (이미 추출된 features에 적용)

---

## 🔧 환경 설정

```bash
# 1. Repository clone (새 서버)
cd /path/to/workspace
git clone https://github.com/iamseungpil/llm-addiction.git
cd llm-addiction
git checkout neuron_sae  # 또는 sae (둘 다 동일)

# 2. Conda 환경 활성화
conda activate llama_sae_env

# 3. 데이터 디렉토리 확인
ls /mnt/c/Users/oollccddss/git/data/llm-addiction/

# 4. GPU 확인
nvidia-smi
```

---

## 📖 참고 문서

- **CLAUDE.md**: Repository 전체 구조 및 규칙
- **STRUCTURE.md**: 파일 구조 가이드
- **exploratory_experiments/EXPERIMENT_PIPELINE.md**: 이 실험의 상세 파이프라인
- **exploratory_experiments/additional_experiments/sae_condition_comparison/ANALYSIS_ISSUES_REPORT.md**: 통계 분석 이슈 상세 설명

---

**작성자**: Claude Code
**최종 업데이트**: 2026-02-03
**상태**: ✅ 코드 구현 100% 완료, 실행 대기 중
