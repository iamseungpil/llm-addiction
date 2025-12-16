# LLM Addiction Research Project Status

## 실험 개요 (2025-12-16 업데이트)

### 현재 상태 요약

| 모델 | 실험 수 | 파산율 | 데이터 위치 |
|------|--------|--------|-------------|
| LLaMA 3.1-8B | 3,200 | 4.69% | `/data/llm_addiction/experiment_0_llama_corrected/` |
| Gemma 2-9B | 3,200 | 20.94% | `/data/llm_addiction/experiment_0_gemma_corrected/` |
| GPT-4o-mini | 3,200 | Variable별 상이 | `/data/llm_addiction/gpt_results_fixed_parsing/` |

---

## 🆕 새 실험 계획: Steering Vector Analysis

### 목표
CAA(Contrastive Activation Addition) 기반 steering vector로 gambling behavior 조작 및 SAE 해석

### Phase 1: Steering Vector 추출 (데이터 준비 완료)

**데이터 소스**:
- LLaMA: 3,200 games, 150 bankruptcy (4.69%), 3,050 voluntary_stop
- Gemma: 3,200 games, 670 bankruptcy (20.94%), 2,530 voluntary_stop

**Steering Vector 계산**:
```python
# 각 모델별로 계산
steering_vector[layer] = mean(bankrupt_hidden_states) - mean(safe_hidden_states)
```

**Target Layers**: 10, 15, 20, 25, 30 (중간~후반 layers)

### Phase 2: Steering 실험

**조건**:
- Steering 강도: [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0]
- 양방향: safe→risky, risky→safe
- 각 조건당 50 trials

**측정 변수**:
- 파산율 변화
- 평균 베팅 금액
- Stop 결정 비율

### Phase 3: SAE 해석

**목표**: Steering vector가 어떤 SAE features를 활성화하는지 분석

```python
# Steering vector를 SAE feature space로 변환
feature_contributions = sae.encode(steering_vector)
top_features = argsort(abs(feature_contributions))[-50:]
```

**SAE 모델**:
- LLaMA: LlamaScope (L1-31, 32K features/layer)
- Gemma: GemmaScope (sae_lens 6.5.1 설치됨)

---

## 주요 데이터 파일

### 실험 0 (LLaMA/Gemma 3,200 games)
```
/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json
/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json
```

### 실험 코드
```
/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/experiment_0_restart_corrected.py
```

### 논문/분석
```
/home/ubuntu/llm_addiction/writing/
/home/ubuntu/llm_addiction/rebuttal_analysis/
```

---

## 데이터 무결성 검증 (2025-12-16 완료)

### LLaMA 데이터
- ✅ 3,200 실험 (64 conditions × 50 reps)
- ✅ 중복 없음
- ✅ 잔고 계산 정확
- ✅ 승률 30.87% (예상 30%)
- ⚠️ 47% empty history (즉시 stop - 정상 동작)

### Gemma 데이터
- ✅ 3,200 실험 (64 conditions × 50 reps)
- ✅ 중복 없음
- ✅ 잔고 계산 정확
- ✅ 승률 29.36% (예상 30%)
- ⚠️ 17% empty history

---

## 환경 설정

### Conda Environment
```bash
# LLaMA/Gemma SAE 분석용
conda activate llama_sae_env

# 설치된 패키지
- sae_lens 6.5.1 (GemmaScope 지원)
- torch 2.7.1
- transformers 4.53.3
```

### SAE 모델 경로
```
LlamaScope: /data/.cache/huggingface/hub/models--fnlp--Llama3_1-8B-Base-LXR-8x/
GemmaScope: huggingface (google/gemma-scope)
```

---

## 파일 구조

```
/home/ubuntu/llm_addiction/
├── CLAUDE.md                           # 이 파일
├── AGENTS.md                           # 코드 스타일 가이드
├── experiment_0_llama_gemma_restart/   # 실험 0 코드
├── experiment_2_multilayer_patching_L1_31/  # SAE patching
├── experiment_pathway_token_analysis/  # Pathway 분석
├── writing/                            # 논문
├── rebuttal_analysis/                  # Rebuttal figures
├── 1216_legacy_code/                   # 정리된 레거시 파일
└── ARCHIVE_*/                          # 아카이브

/data/llm_addiction/
├── experiment_0_llama_corrected/       # LLaMA 3,200 games
├── experiment_0_gemma_corrected/       # Gemma 3,200 games
├── gpt_results_fixed_parsing/          # GPT 실험
└── 1216_legacy_data/                   # 정리된 레거시 데이터
```

---

## 다음 단계

1. **Steering Vector 구현**: LLaMA/Gemma hidden state 추출 및 steering vector 계산
2. **Steering 실험 실행**: 7개 강도 × 2방향 × 50 trials
3. **SAE 해석**: Steering vector의 feature-level 분해
4. **비교 분석**: LLaMA vs Gemma steering 효과 차이

---

*마지막 업데이트: 2025-12-16*
