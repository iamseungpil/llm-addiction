# 최종 4개 실험 계획 (승인 대기)

## 현재 GPU 상태 확인

| GPU | 메모리 사용 | 메모리 여유 | 현재 용도 | 실험 가능? |
|-----|------------|------------|----------|-----------|
| GPU 0 | 0 MB | 81 GB | 비어있음 | ✅ Yes |
| GPU 1 | 0 MB | 81 GB | 비어있음 | ✅ Yes |
| GPU 2 | 20.8 GB | 60 GB | m-soar (Qwen 7B) | ⚠️ Limited |
| GPU 3 | 0 MB | 81 GB | 비어있음 | ✅ Yes |
| GPU 4 | 23.6 GB | 57 GB | **Exp5 진행 중** | ⚠️ Limited |
| GPU 5 | 80.5 GB | 0.5 GB | m-soar (Qwen 32B) | ❌ No |
| GPU 6 | 40.5 GB | 40 GB | m-soar (Qwen 7B) | ⚠️ Limited |
| GPU 7 | 45.8 GB | 35 GB | m-soar (Qwen 7B) | ⚠️ Limited |

### GPU 사용 가능성 분석

**안전하게 사용 가능 (메모리 충분)**:
- GPU 0, 1, 3: 완전히 비어있음 (81GB)

**제한적 사용 가능 (다른 프로세스와 공유)**:
- GPU 2: 60GB 여유 → 실험 메모리 25GB 필요 → **가능하지만 m-soar와 공유**
- GPU 4: 57GB 여유 → 실험 메모리 25GB 필요 → **가능하지만 Exp5와 공유**
- GPU 6: 40GB 여유 → 실험 메모리 25GB 필요 → **가능하지만 m-soar와 공유**
- GPU 7: 35GB 여유 → 실험 메모리 25GB 필요 → **가능하지만 m-soar와 공유**

**위험도 평가**:
- m-soar 서버는 API 서비스이므로 **갑작스런 메모리 증가 가능**
- Exp5는 안정적으로 23GB 사용 중 (예측 가능)

---

## 실험 2: Multilayer Patching 상세 확인

### Feature 선택 확인 ✅

**각 layer당 정확히 300개 features 선택**:
- L1-31 각각: 300 features
- 총 31 layers × 300 = **9,300 features**

**Layer 그룹별 분포**:
- **L1-8**: 2,400 features (8 layers × 300)
- **L9-16**: 2,400 features (8 layers × 300)
- **L17-24**: 2,400 features (8 layers × 300)
- **L25-31**: 2,100 features (7 layers × 300)

**Cohen's d 범위**:
- L1-8: 0.30 ~ 1.24
- L9-16: 0.30 ~ 1.44
- L17-24: 0.30 ~ 1.58
- L25-31: 0.30 ~ 1.63

---

## 최종 4개 실험 계획

---

## 실험 0: LLaMA/Gemma 재시작 (3,200 games each)

### 목적
GPT 실험과 동일한 조건으로 LLaMA/Gemma 비교 데이터 수집

### 설계
- **조건**: 64개 (5개 component 조합 32가지 × 2 bet types)
- **반복**: 50회/조건
- **총 게임**: 3,200 games (LLaMA), 3,200 games (Gemma)
- **무한 retry**: 빈 응답 나올 때까지 재시도

### GPU 할당

#### LLaMA
- **GPU**: GPU 0
- **메모리**: ~20GB 필요, 81GB 여유 ✅
- **환경**: conda llama_sae_env
- **예상 시간**: ~24시간

#### Gemma
- **GPU**: GPU 1
- **메모리**: ~22GB 필요, 81GB 여유 ✅
- **환경**: conda llama_sae_env (DeepSpeed 제거)
- **예상 시간**: ~24시간

### 파일 경로
- **LLaMA 코드**: `/home/ubuntu/llm_addiction/experiment_0_standardization/llama_3200_infinite_retry.py`
- **Gemma 코드**: `/home/ubuntu/llm_addiction/experiment_0_standardization/gemma_3200_no_deepspeed.py`
- **LLaMA 로그**: `/home/ubuntu/llm_addiction/experiment_0_standardization/llama_restart.log`
- **Gemma 로그**: `/home/ubuntu/llm_addiction/experiment_0_standardization/gemma_restart.log`
- **LLaMA 결과**: `/data/llm_addiction/experiment_0_standardization/llama_3200_complete.json`
- **Gemma 결과**: `/data/llm_addiction/experiment_0_standardization/gemma_3200_complete.json`

### 실행 명령
```bash
# LLaMA
cd /home/ubuntu/llm_addiction/experiment_0_standardization
CUDA_VISIBLE_DEVICES=0 nohup conda run -n llama_sae_env python llama_3200_infinite_retry.py > llama_restart.log 2>&1 &

# Gemma
CUDA_VISIBLE_DEVICES=1 nohup conda run -n llama_sae_env python gemma_3200_no_deepspeed.py > gemma_restart.log 2>&1 &
```

---

## 실험 1: Layer Pathway Tracking (L1-31)

### 목적
도박 결정이 L1→L31에서 어떻게 진화하는지 추적

### 설계
- **게임 수**: 50 games (25 bankruptcy + 25 voluntary stop)
- **추적**: 매 round마다 L1-31 전체 activations 저장
- **SAE 추출**: 각 layer의 top features 추출 (on-demand)

### GPU 할당
- **GPU**: GPU 3
- **메모리**: ~30GB 필요, 81GB 여유 ✅
- **환경**: conda llama_sae_env
- **예상 시간**: ~4시간

### 파일 경로
- **코드**: `/home/ubuntu/llm_addiction/experiment_1_layer_pathway/layer_pathway_tracking.py`
- **로그**: `/home/ubuntu/llm_addiction/experiment_1_layer_pathway/pathway.log`
- **결과**: `/data/llm_addiction/experiment_1_layer_pathway/pathway_50games.json`

### 실행 명령
```bash
cd /home/ubuntu/llm_addiction/experiment_1_layer_pathway
CUDA_VISIBLE_DEVICES=3 nohup conda run -n llama_sae_env python layer_pathway_tracking.py > pathway.log 2>&1 &
```

---

## 실험 2: Multilayer Patching (9,300 features, 4 GPUs 병렬)

### 목적
L1-31의 9,300개 features (각 layer당 top 300개)의 인과성 검증

### 설계
- **Features**: 9,300개 (31 layers × 300 features/layer)
- **Patching 방법**: Population mean patching (safe_mean, baseline, risky_mean)
- **Prompts**: 2가지 (safe_prompt, risky_prompt)
- **Trials**: 20회/조건
- **총 runs**: 9,300 × 3 scales × 2 prompts × 20 trials = 1,116,000 runs

### GPU 병렬 분산 (4 GPUs)

| GPU | Layer 범위 | Features | 메모리 필요 | 메모리 여유 | 안전성 | 예상 시간 |
|-----|-----------|----------|------------|------------|--------|----------|
| **GPU 2** | L1-8 | 2,400 | 25GB | 60GB ✅ | ⚠️ m-soar 공유 | 8.1일 |
| **GPU 4** | L9-16 | 2,400 | 25GB | 57GB ✅ | ⚠️ Exp5 공유 | 8.1일 |
| **GPU 6** | L17-24 | 2,400 | 25GB | 40GB ⚠️ | ⚠️ m-soar 공유 | 8.1일 |
| **GPU 7** | L25-31 | 2,100 | 25GB | 35GB ⚠️ | ⚠️ m-soar 공유 | 7.1일 |

**위험도**:
- GPU 2: 60GB 여유 → 25GB 사용 → 35GB 남음 (m-soar 20GB) → **안전**
- GPU 4: 57GB 여유 → 25GB 사용 → 32GB 남음 (Exp5는 안정적) → **안전**
- GPU 6: 40GB 여유 → 25GB 사용 → 15GB 남음 (m-soar 40GB) → **경계**
- GPU 7: 35GB 여유 → 25GB 사용 → 10GB 남음 (m-soar 45GB) → **경계**

**대안 (더 안전한 선택)**:
- **Option A**: GPU 0, 1, 3만 사용 (Exp0, Exp1 완료 후)
  - GPU 0: L1-8 (2,400 features)
  - GPU 1: L9-16 (2,400 features)
  - GPU 3: L17-31 (4,500 features) → 12.2일
  - 총 시간: 12.2일 (GPU 3가 느림)

- **Option B**: GPU 0, 1, 3, 4 사용 (Exp0, Exp1 완료, Exp5 완료 후)
  - GPU 0: L1-8 (2,400 features)
  - GPU 1: L9-16 (2,400 features)
  - GPU 3: L17-24 (2,400 features)
  - GPU 4: L25-31 (2,100 features)
  - 총 시간: 8.1일

- **Option C (추천)**: GPU 2, 4, 6, 7 사용 (즉시 시작)
  - m-soar와 공유하지만 메모리 충분
  - 즉시 시작 가능
  - 총 시간: 8.1일

### 파일 경로
- **코드**: `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching/multilayer_patching.py`
- **로그**:
  - `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching/multilayer_gpu2.log`
  - `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching/multilayer_gpu4.log`
  - `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching/multilayer_gpu6.log`
  - `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching/multilayer_gpu7.log`
- **결과**:
  - `/data/llm_addiction/experiment_2_multilayer_patching/multilayer_final_gpu2.json`
  - `/data/llm_addiction/experiment_2_multilayer_patching/multilayer_final_gpu4.json`
  - `/data/llm_addiction/experiment_2_multilayer_patching/multilayer_final_gpu6.json`
  - `/data/llm_addiction/experiment_2_multilayer_patching/multilayer_final_gpu7.json`

### 실행 명령 (Option C: 즉시 시작)
```bash
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching

# GPU 2: L1-8
CUDA_VISIBLE_DEVICES=2 nohup conda run -n llama_sae_env python multilayer_patching.py --gpu_id 2 --layer_start 1 --layer_end 8 > multilayer_gpu2.log 2>&1 &

# GPU 4: L9-16 (Exp5와 공유)
CUDA_VISIBLE_DEVICES=4 nohup conda run -n llama_sae_env python multilayer_patching.py --gpu_id 4 --layer_start 9 --layer_end 16 > multilayer_gpu4.log 2>&1 &

# GPU 6: L17-24
CUDA_VISIBLE_DEVICES=6 nohup conda run -n llama_sae_env python multilayer_patching.py --gpu_id 6 --layer_start 17 --layer_end 24 > multilayer_gpu6.log 2>&1 &

# GPU 7: L25-31
CUDA_VISIBLE_DEVICES=7 nohup conda run -n llama_sae_env python multilayer_patching.py --gpu_id 7 --layer_start 25 --layer_end 31 > multilayer_gpu7.log 2>&1 &
```

### 중간 저장
- 매 100 features마다 저장
- 파일명: `multilayer_intermediate_gpu{id}_{timestamp}.json`

---

## 실험 3: Feature-Word Analysis (441 features)

### 목적
441개 causal features가 어떤 단어/개념과 연관되는지 분석

### 설계
- **Features**: 441개 (현재 Exp5에서 검증 중)
- **방법**:
  1. SAE Decoder Weight Analysis (top 50 tokens)
  2. Response Pattern Analysis (safe vs risky 단어 빈도 차이)

### GPU 할당
- **GPU**: GPU 3 (Exp1 완료 후, ~4시간 후)
- **메모리**: ~26GB 필요, 81GB 여유 ✅
- **환경**: conda llama_sae_env
- **예상 시간**: ~3.5시간

### 파일 경로
- **코드**: `/home/ubuntu/llm_addiction/experiment_4_feature_word_analysis/feature_word_analysis.py` (이미 존재)
- **로그**: `/home/ubuntu/llm_addiction/experiment_4_feature_word_analysis/analysis.log`
- **결과**: `/data/llm_addiction/experiment_4_feature_word_analysis/feature_word_associations.json`

### 실행 명령
```bash
cd /home/ubuntu/llm_addiction/experiment_4_feature_word_analysis
CUDA_VISIBLE_DEVICES=3 nohup conda run -n llama_sae_env python feature_word_analysis.py > analysis.log 2>&1 &
```

---

## 실험 타임라인 (Option C: GPU 2,4,6,7 즉시 사용)

### Phase 1: 즉시 시작

| 실험 | GPU | 시간 | 상태 |
|------|-----|------|------|
| Exp0-LLaMA | GPU 0 | 24시간 | 즉시 |
| Exp0-Gemma | GPU 1 | 24시간 | 즉시 |
| Exp1-Pathway | GPU 3 | 4시간 | 즉시 |
| **Exp2-Part1 (L1-8)** | **GPU 2** | **8.1일** | **즉시** |
| **Exp2-Part2 (L9-16)** | **GPU 4** | **8.1일** | **즉시** |
| **Exp2-Part3 (L17-24)** | **GPU 6** | **8.1일** | **즉시** |
| **Exp2-Part4 (L25-31)** | **GPU 7** | **8.1일** | **즉시** |
| Exp5 (계속) | GPU 4 | ~50시간 | 진행 중 |

### Phase 2: 4시간 후 (Exp1 완료)

| 실험 | GPU | 시간 |
|------|-----|------|
| Exp3-Feature-Word | GPU 3 | 3.5시간 |

### 총 완료 시간
- **Exp0, Exp1, Exp3**: ~1.5일
- **Exp2**: 8.1일 (병렬)
- **Exp5**: ~2일 남음

**전체 완료**: ~8.1일 (Exp2가 가장 김)

---

## GPU 메모리 안전성 체크

### 현재 사용량 + 실험 추가 시

| GPU | 현재 | 실험 추가 | 합계 | 총 메모리 | 안전 여유 |
|-----|------|----------|------|----------|----------|
| GPU 0 | 0 GB | 20 GB (LLaMA) | 20 GB | 81 GB | 61 GB ✅ |
| GPU 1 | 0 GB | 22 GB (Gemma) | 22 GB | 81 GB | 59 GB ✅ |
| GPU 2 | 21 GB | 25 GB (Exp2-L1-8) | 46 GB | 81 GB | 35 GB ✅ |
| GPU 3 | 0 GB | 30 GB (Exp1) | 30 GB | 81 GB | 51 GB ✅ |
| GPU 4 | 24 GB | 25 GB (Exp2-L9-16) | 49 GB | 81 GB | 32 GB ✅ |
| GPU 6 | 40 GB | 25 GB (Exp2-L17-24) | 65 GB | 81 GB | 16 GB ⚠️ |
| GPU 7 | 46 GB | 25 GB (Exp2-L25-31) | 71 GB | 81 GB | 10 GB ⚠️ |

**위험 평가**:
- GPU 0, 1, 2, 3, 4: **안전** (30GB 이상 여유)
- GPU 6, 7: **경계** (10-16GB 여유, m-soar 사용량 증가 시 위험)

**권장사항**:
- GPU 6, 7 모니터링 필요
- 메모리 부족 시 m-soar 서버 일시 중단 고려

---

## 대안 선택지 정리

### Option A: 안전 우선 (GPU 0,1,3만 사용, Exp0/Exp1 완료 후)
- **장점**: 메모리 안전, m-soar 방해 없음
- **단점**: 12.2일 소요 (GPU 3가 L17-31 담당)
- **시작 시기**: 1.5일 후

### Option B: 균형 (GPU 0,1,3,4 사용, Exp0/Exp1/Exp5 완료 후)
- **장점**: 8.1일 소요, 메모리 안전
- **단점**: 2일 대기 필요 (Exp5 완료)
- **시작 시기**: ~2일 후

### Option C: 속도 우선 (GPU 2,4,6,7 즉시 사용) ⭐ **추천**
- **장점**: 즉시 시작, 8.1일 소요
- **단점**: GPU 6,7 메모리 경계, m-soar와 공유
- **시작 시기**: 즉시

---

## 최종 권장 계획 (승인 필요)

### 즉시 실행:
1. **Exp0-LLaMA** (GPU 0, 24시간)
2. **Exp0-Gemma** (GPU 1, 24시간)
3. **Exp1-Pathway** (GPU 3, 4시간)
4. **Exp2-Multilayer** (GPU 2,4,6,7, 8.1일) - **Option C**

### 4시간 후:
5. **Exp3-Feature-Word** (GPU 3, 3.5시간)

### 모니터링:
- GPU 6, 7 메모리 사용량 주기적 체크
- m-soar 서버 영향 확인

---

## 승인 질문

1. **Exp2 GPU 선택**: Option A/B/C 중 어느 것을 선택하시겠습니까?
   - **Option C (추천)**: GPU 2,4,6,7 즉시 시작 (메모리 경계, 8.1일)
   - Option B: GPU 0,1,3,4 사용 (2일 대기, 8.1일)
   - Option A: GPU 0,1,3만 사용 (1.5일 대기, 12.2일)

2. **Exp5 처리**: GPU 4에서 Exp2와 병렬 실행 괜찮으신가요?
   - Exp5 (23GB) + Exp2 (25GB) = 48GB < 57GB 여유 ✅

3. **m-soar 서버**: GPU 2,6,7 공유 괜찮으신가요?
   - 메모리 모니터링 필요

---

*계획서 작성: 2025-10-01 15:05*
