# 전체 실험 파이프라인

**목표**: 중독 현상이 다양한 도메인과 모델에서 동일한 신경 메커니즘(SAE features)으로 발현되는지 검증

**기간**: 2-4주 (개발 36h + GPU 78h + CPU 15h)
**날짜**: 2026-02-03

---

## 📊 실험 구조

### 도메인 (3개)
1. **Slot Machine** (기존) - 기본 도박 메커니즘
2. **Loot Box** (신규) - 비화폐 보상, 변동 강화 스케줄
3. **Blackjack** (신규) - 전략적 도박, 복잡한 의사결정

### 모델 (2개)
- **LLaMA-3.1-8B** (~19GB VRAM)
- **Gemma-2-9B** (~22GB VRAM)

### 조작 변수
- **Variable vs Fixed betting** (autonomy effect 검증)
- **Prompt components** (G, W, P)

---

## 🔄 파이프라인 (3 Parts)

```
Part 1: Condition Comparison 수정 (CPU-only, ~10h)
  └─> 통계 이슈 해결 + Addiction features 교차 검증

Part 2: Alternative Paradigms SAE 인프라 (~14h 개발 + 78h GPU)
  └─> Loot Box, Blackjack 실험 → SAE features 추출

Part 3: Cross-Domain 비교 (~12h CPU)
  └─> 3개 도메인 feature overlap 분석 (Jaccard, Core features)
```

---

## Part 1: Condition Comparison 수정 (✅ 완료)

### 목적
- **문제**: Analysis 3 (Interaction)의 92%가 통계적 artifact
- **해결**: Sparse feature 필터링 + statsmodels 검증
- **추가**: Condition features ↔ Addiction features 교차 검증

### 구현된 파일
1. `sae_condition_comparison/src/utils.py`
   - `filter_sparse_features()`: 활성화율 < 1% 제거

2. `sae_condition_comparison/src/condition_comparison.py`
   - `analyze_interaction_layer()`: 필터링 적용
   - `validate_top_features_with_statsmodels()`: 정확한 2-way ANOVA

3. `sae_condition_comparison/src/cross_reference_addiction_features.py` (신규)
   - Risk Amplification: Variable-higher ∩ Risky
   - Protective: Fixed-higher ∩ Safe

### 실행 방법 (CPU-only, GPU 불필요)

```bash
# 1. LLaMA 재분석 (1-2시간)
cd exploratory_experiments/additional_experiments/sae_condition_comparison
python -m src.condition_comparison --model llama

# 2. Gemma 재분석 (1-2시간)
python -m src.condition_comparison --model gemma

# 3. Cross-reference 분석 (30분)
python src/cross_reference_addiction_features.py --model llama
python src/cross_reference_addiction_features.py --model gemma
```

### 예상 결과
- Artifact rate: 92% → <10%
- Risk Amplification features: 500-1000개 (Variable ∩ Risky)
- Protective features: 400-800개 (Fixed ∩ Safe)

---

## Part 2: Alternative Paradigms SAE 인프라

### 2-1. 실험 재실행 (hidden states + full_prompt 저장)

**수정된 파일**:
- `alternative_paradigms/src/common/model_loader.py`
  - `generate_with_hidden_states()` 메서드 추가

- `alternative_paradigms/src/lootbox/run_experiment.py`
  - `trials` 리스트에 `full_prompt` 필드 추가

- `alternative_paradigms/src/blackjack/run_experiment.py`
  - `rounds` 리스트에 `full_prompt` 필드 추가

**실행 방법** (GPU 필요):

```bash
# Loot Box 재실행 (2 models × 4h = 8h GPU)
python src/lootbox/run_experiment.py --model llama --gpu 0
python src/lootbox/run_experiment.py --model gemma --gpu 0

# Blackjack 재실행 (2 models × 4h = 8h GPU)
python src/blackjack/run_experiment.py --model llama --gpu 0 --bet-type variable
python src/blackjack/run_experiment.py --model gemma --gpu 0 --bet-type variable

# 총 GPU 시간: 16h
```

**출력**:
- `/data/llm-addiction/alternative_paradigms/lootbox/*.json`
- `/data/llm-addiction/alternative_paradigms/blackjack/*.json`
- 각 파일에 `trials[].full_prompt` 또는 `rounds[].full_prompt` 포함

---

### 2-2. Phase 1: SAE Feature Extraction

**신규 파일**: `alternative_paradigms/src/common/phase1_feature_extraction.py`

**기능**:
1. JSON에서 prompts 추출
2. Model forward pass → hidden states
3. SAE encoding → features (NPZ 저장)

**실행 방법** (GPU 필요):

```bash
# Loot Box
python src/common/phase1_feature_extraction.py --paradigm lootbox --model llama --gpu 0
# LLaMA: 7 layers × 1h = 7h GPU

python src/common/phase1_feature_extraction.py --paradigm lootbox --model gemma --gpu 0
# Gemma: 26 layers × 1h = 26h GPU

# Blackjack
python src/common/phase1_feature_extraction.py --paradigm blackjack --model llama --gpu 0
# LLaMA: 7h GPU

python src/common/phase1_feature_extraction.py --paradigm blackjack --model gemma --gpu 0
# Gemma: 26h GPU

# 총 GPU 시간: (7 + 26) × 2 paradigms = 66h
```

**출력**:
- `/data/llm-addiction/alternative_paradigms/lootbox/sae_features/layer_*_features.npz`
- `/data/llm-addiction/alternative_paradigms/blackjack/sae_features/layer_*_features.npz`

**NPZ 포맷**:
```python
{
    'features': (n_samples, n_features),  # SAE activations
    'outcomes': (n_samples,),  # 'bankrupt' or 'voluntary_stop'
    'game_ids': (n_samples,),
    'layer': int,
    'model_type': str
}
```

---

### 2-3. Phase 2: Correlation Analysis

**신규 파일**: `alternative_paradigms/src/common/phase2_correlation_analysis.py`

**기능**:
1. NPZ 로드
2. Bankrupt vs Safe 그룹으로 Welch's t-test
3. Cohen's d 계산
4. FDR correction (Benjamini-Hochberg)
5. Risky/Safe features 식별 (|d| >= 0.3, FDR < 0.05)

**실행 방법** (CPU-only):

```bash
# Loot Box
python src/common/phase2_correlation_analysis.py --paradigm lootbox --model llama
# ~1h CPU

python src/common/phase2_correlation_analysis.py --paradigm lootbox --model gemma
# ~2h CPU (more layers)

# Blackjack
python src/common/phase2_correlation_analysis.py --paradigm blackjack --model llama
# ~1h CPU

python src/common/phase2_correlation_analysis.py --paradigm blackjack --model gemma
# ~2h CPU

# 총 CPU 시간: ~6h
```

**출력**:
- `correlation_all_features_{timestamp}.json`: 모든 features 통계
- `correlation_significant_{timestamp}.json`: 유의미한 features (top 100)
- `correlation_summary_{timestamp}.json`: 요약 통계

---

## Part 3: Cross-Domain 비교 프레임워크

### 3-1. Cross-Domain Overlap Analysis

**신규 파일**: `cross_domain_sae_comparison/src/cross_domain_analysis.py`

**기능**:
1. 3개 도메인 Phase2 결과 로드
2. Pairwise Jaccard similarity 계산
3. Core features 식별 (2+ domains)
4. Universal features 식별 (3 domains)

**실행 방법** (CPU-only):

```bash
cd exploratory_experiments/additional_experiments/cross_domain_sae_comparison

# LLaMA 분석 (~3h CPU)
python src/cross_domain_analysis.py --model llama

# Gemma 분석 (~3h CPU)
python src/cross_domain_analysis.py --model gemma
```

**출력**:
- `results/cross_domain_overlap_llama_{timestamp}.json`
- `results/cross_domain_overlap_gemma_{timestamp}.json`

**분석 내용**:
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

## 📈 성공 기준 (3 domains)

| 지표 | 강한 일반화 | 중간 일반화 (예상) | 약한 일반화 (최소) |
|------|------------|-------------------|-------------------|
| **Core features** | ≥150 risky + ≥150 safe | ≥80 (2+ domains) | ≥40 (2 domains) |
| **Jaccard similarity** | > 0.40 | > 0.25 | > 0.10 |
| **Universal features** | ≥30 (all 3) | ≥15 (all 3) | ≥5 (all 3) |

**Random baseline**: Jaccard ≈ 0.01 (1% FDR × 1% FDR)

---

## 🗓️ 실행 순서 및 시간

### Week 1: Part 1 완료 (✅ 코드 구현 완료)
- [ ] Day 1-2: LLaMA/Gemma 재분석 (CPU, 4h)
- [ ] Day 2-3: Cross-reference 분석 (CPU, 1h)
- [ ] Day 3-5: 결과 검증 및 문서화

### Week 2: Part 2 실험 재실행
- [ ] Day 1-2: Loot Box + Blackjack 재실행 (16h GPU)
- [ ] Day 3-5: Phase 1 feature extraction 준비

### Week 3: Part 2 SAE 추출
- [ ] Day 1-3: Phase 1 LLaMA (14h GPU)
- [ ] Day 3-5: Phase 1 Gemma (52h GPU)
- [ ] 병렬 실행 시 3-4일 (2 GPU 사용)

### Week 4: Part 2-3 분석 및 논문
- [ ] Day 1-2: Phase 2 correlation (6h CPU)
- [ ] Day 3-4: Cross-domain 비교 (6h CPU)
- [ ] Day 4-5: 논문 Figure 생성 및 보고서 작성

**총 시간**: 개발 완료 + **실행 시간 (GPU 82h + CPU 17h)**

---

## 🖥️ GPU 요구사항 요약

| 단계 | LLaMA | Gemma | 총합 |
|------|-------|-------|------|
| **Part 1** (재분석) | 0h | 0h | **0h** (CPU-only) |
| **Part 2-1** (실험 재실행) | 8h | 8h | **16h** |
| **Part 2-2** (Phase 1) | 14h | 52h | **66h** |
| **Part 2-3** (Phase 2) | 0h | 0h | **0h** (CPU-only) |
| **Part 3** (Cross-domain) | 0h | 0h | **0h** (CPU-only) |
| **총 GPU 시간** | 22h | 60h | **82h** |

**병렬화**: 2 GPU 사용 시 ~41h (동시 LLaMA + Gemma)

**VRAM 요구**:
- LLaMA-3.1-8B: 19GB
- Gemma-2-9B: 22GB

---

## 📁 최종 출력 파일 구조

```
/data/llm-addiction/
├── sae_patching/corrected_sae_analysis/
│   ├── llama/
│   │   ├── correlation_all_features_*.json  (기존)
│   │   └── ... (Phase2 결과)
│   └── gemma/
│       └── ...
│
├── alternative_paradigms/
│   ├── lootbox/
│   │   ├── llama_lootbox_*.json  (재실행 결과)
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
    │   ├── cross_reference_llama_*.json  (Part 1 교차 검증)
    │   └── ...
    │
    └── cross_domain_sae_comparison/results/
        ├── cross_domain_overlap_llama_*.json  (Part 3)
        └── cross_domain_overlap_gemma_*.json
```

---

## 📊 논문 기여

완료 시 주장 가능:

### 1. Setting Modulation (Part 1)
"Variable betting conditions selectively activate risk-amplifying features (N=XXX), explaining 2.6× bankruptcy rate increase. Fixed conditions activate protective features (N=XXX)."

### 2. Domain Generalization (Part 2-3)
"XXX core addiction features generalize across slot machine, loot box, and blackjack paradigms (Jaccard=0.XX, Cohen's h=X.XX), demonstrating domain-agnostic neural substrates of gambling addiction."

### 3. Methodological Rigor (Part 1)
"After sparse feature filtering (activation rate >= 1%), interaction analysis artifact rate reduced from 92% to <10%, validated with exact statsmodels 2-way ANOVA."

---

## ⚠️ 주의사항

1. **GPU 메모리**: 각 모델별로 별도 GPU 사용 권장
2. **Sparse Filtering**: Phase 1 전에 실행 불가 (이미 추출된 features에 적용)
3. **데이터 검증**: 실험 재실행 시 `full_prompt` 필드 존재 확인 필수
4. **백업**: Phase 1 완료 후 NPZ 파일 백업 (재생성 시간 길음)

---

**작성일**: 2026-02-03
**상태**: ✅ 코드 구현 완료, 실행 대기 중
**다음 단계**: Part 1 재분석 실행 (CPU-only)
