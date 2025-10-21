# LLM Addiction - 5개 실험 상태 보고서
**날짜**: 2025-10-02 04:50 KST

## 📊 전체 요약

| 실험 | 상태 | 진행률 | GPU | 비고 |
|------|------|--------|-----|------|
| **Exp0 LLaMA** | ✅ 실행 중 | 6/128 (4.7%) | GPU 0 | 정상 작동 |
| **Exp0 Gemma** | ❌ 멈춤 | 0/128 (0%) | GPU 1 | 게임 생성 미시작 |
| **Exp2 Patching** | ⚠️ 로딩 중 | 0% | GPU 2,5,6,7 | SAE 로딩 단계 정지 |
| **Exp1 Pathway** | ✅ 완료 | 100% | - | 2.7GB 결과 파일 |
| **Exp3 Word** | ✅ 완료 | 100% | - | 441 features 분석 완료 |

---

## 🔍 상세 분석

### 1. ✅ Exp0 LLaMA (정상 실행 중)

**위치**: `/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/`
**코드**: `experiment_0_restart.py`
**로그**: `logs/exp0_llama.log`
**tmux 세션**: `exp0_llama`

**진행 상황**:
- 6/128 조건 완료 (4.7%)
- 예상 완료 시간: ~4시간 (125초/조건)
- GPU 0: 15.9GB 사용
- CPU: 정상

**예상 결과 파일**: `/data/llm_addiction/experiment_0_llama_restart/`

---

### 2. ❌ Exp0 Gemma (멈춤 - 진단 필요)

**위치**: `/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/`
**코드**: `experiment_0_restart.py --model gemma`
**로그**: `logs/exp0_gemma.log`
**tmux 세션**: `exp0_gemma`

**문제 상황**:
- 모델 로딩 성공 (✅ GEMMA loaded successfully)
- 0/128 조건에서 멈춤 (게임 시작 안 됨)
- GPU 1: 59GB 사용 (매우 높음)
- JAX 제거했으나 여전히 문제

**가능한 원인**:
1. Gemma-2-9b-it의 메모리 사용량이 너무 높음 (59GB)
2. Chat template 또는 generation 설정 문제
3. Empty response 무한 루프 가능성

**해결 방안**:
- Gemma 프로세스 재시작 필요
- 메모리 설정 최적화 필요
- Generation 파라미터 조정 필요

---

### 3. ⚠️ Exp2 Patching (SAE 로딩 단계 정지)

**위치**: `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/`
**코드**: `experiment_2_L1_31_top300.py`
**로그**: `logs/exp2_L1_8_gpu2.log` (등 4개)
**tmux 세션**: `exp2_p1`, `exp2_p2`, `exp2_p3`, `exp2_p4`

**문제 상황**:
```
   Loading W_E (encoder.weight)... converted... ✅ (torch.Size([4096, 32768]))
   Loading b_E (encoder.bias)... converted... ✅ (torch.Size([32768]))
   Loading W_D (decoder.weight)... converted... ✅ (torch.Size([32768, 4096]))
   Loading b_D (decoder.bias)... converted... ← 여기서 멈춤
```

- 모든 4개 프로세스가 동일한 위치에서 정지
- CPU 사용률 102-105% (실행은 되고 있음)
- GPU 메모리 할당 완료 (16-17GB per GPU)
- "converted..." 출력 후 "✅ (torch.Size...)" 미출력

**기술적 원인**:
- llama_scope_working.py line 228의 print 버퍼링
- line 227 `new_state_dict[target_name] = weight` 매우 느림
- 또는 line 235+ 단계로 넘어갔으나 출력 없음

**시도한 해결책**:
1. ✅ `.float()` → `.to(torch.float32)` 변경
2. ✅ Staggered launch (30초 지연)
3. ✅ 프로세스 수 감소 (12개 → 4개)
4. ❌ 여전히 동일 지점에서 정지

**다음 단계**:
1. Python unbuffered mode로 실행 (`python -u`)
2. 모든 print에 `flush=True` 추가
3. SAE 로딩 코드 더 단순화
4. Checkpoint를 미리 메모리에 로드

---

### 4. ✅ Exp1 Pathway (완료)

**위치**: `/data/llm_addiction/experiment_1_pathway_L1_31/`
**결과 파일**: `final_pathway_L1_31_20251001_165207.json` (2.7GB)

**완료 내역**:
- 50 games 분석
- L1-31 (전체 31 layers) tracking 완료
- Voluntary stop vs Bankruptcy 그룹 비교

---

### 5. ✅ Exp3 Word Analysis (완료)

**위치**: `/data/llm_addiction/experiment_4_feature_word_analysis/`
**결과 파일**: `feature_word_analysis_20251001_000025.json`

**완료 내역**:
- 441 features (Layer 25) 분석 완료
- Decoder weight analysis로 영향받는 단어 분석
- Bankrupt vs Safe group delta 계산

---

## 🛠️ 해결한 문제들

### 1. ✅ device_map 오류
- **문제**: `device_map={'': self.gpu_id}` → GPU ID 충돌
- **해결**: `device_map={'': 0}` (CUDA_VISIBLE_DEVICES 사용 시)
- **파일**: `experiment_2_L1_31_top300.py:119`

### 2. ✅ JAX 의존성 제거
- **문제**: Gemma 로딩 시 `AttributeError: _ARRAY_API not found`
- **해결**: `pip uninstall -y jax jaxlib`
- **결과**: Gemma 모델 로딩 성공

### 3. ✅ CUDA_VISIBLE_DEVICES 전파
- **문제**: tmux 명령어에서 환경변수 미전달
- **해결**: `env CUDA_VISIBLE_DEVICES=X python ...` 형식 사용
- **적용**: 모든 launcher 스크립트

---

## 📁 실험 코드 및 결과 경로

### 실험 코드
```
/home/ubuntu/llm_addiction/
├── experiment_0_llama_gemma_restart/
│   ├── experiment_0_restart.py ← Exp0 코드
│   └── logs/
│       ├── exp0_llama.log ← LLaMA 로그
│       └── exp0_gemma.log ← Gemma 로그
├── experiment_2_multilayer_patching_L1_31/
│   ├── experiment_2_L1_31_top300.py ← Exp2 코드
│   ├── launch_safe.sh ← 실행 스크립트
│   └── logs/
│       ├── exp2_L1_8_gpu2.log
│       ├── exp2_L9_16_gpu5.log
│       ├── exp2_L17_24_gpu6.log
│       └── exp2_L25_31_gpu7.log
├── experiment_1_layer_pathway_L1_31/
│   └── experiment_1_pathway.py ← Exp1 코드 (완료)
└── experiment_3_feature_word_6400/
    └── experiment_3_feature_word.py ← Exp3 코드 (완료)
```

### 결과 데이터
```
/data/llm_addiction/
├── experiment_0_llama_restart/ ← Exp0 LLaMA 결과 (진행 중)
├── experiment_0_gemma_restart/ ← Exp0 Gemma 결과 (멈춤)
├── experiment_2_multilayer_patching/ ← Exp2 결과 (진행 중)
├── experiment_1_pathway_L1_31/
│   └── final_pathway_L1_31_20251001_165207.json ← Exp1 완료 (2.7GB)
└── experiment_4_feature_word_analysis/
    └── feature_word_analysis_20251001_000025.json ← Exp3 완료
```

### Feature 데이터
```
/data/llm_addiction/experiment_1_L1_31_extraction/
└── L1_31_features_FINAL_20250930_220003.json ← 87,012 features (29MB)
```

---

## 🎯 다음 단계 권장사항

### 즉시 조치 필요:
1. **Exp0 Gemma**: 재시작 필요 (메모리 최적화)
2. **Exp2 Patching**: Python unbuffered mode로 재실행

### 장기 해결책:
1. SAE loader를 shared memory 방식으로 변경
2. Gemma generation 파라미터 최적화
3. Checkpoint pre-loading 메커니즘 추가

---

## 📊 예상 완료 시간

| 실험 | 현재 상태 | 예상 완료 | 소요 시간 |
|------|-----------|-----------|-----------|
| Exp0 LLaMA | 6/128 (4.7%) | ~4시간 | 125s/조건 |
| Exp0 Gemma | 멈춤 | 재시작 후 4시간 | - |
| Exp2 Patching | 로딩 중 | 해결 후 12-15시간 | - |
| Exp1 Pathway | 완료 | - | - |
| Exp3 Word | 완료 | - | - |

**전체 완료 예상**: 문제 해결 후 약 15-20시간

---

*보고서 생성: 2025-10-02 04:50 KST*
