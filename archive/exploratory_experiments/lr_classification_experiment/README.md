# LR Classification Experiment

Logistic Regression을 사용하여 LLM의 hidden state가 gambling 결과(bankruptcy vs voluntary_stop)를 예측할 수 있는지 분석하는 실험입니다.

## 핵심 연구 질문

> **SAE 없이 raw activation만으로 bankruptcy vs voluntary_stop 상태를 구분할 수 있는가?**

### 배경
- 기존 연구: SAE를 통해 feature를 추출하여 분석
- 문제점: 범용 SAE 사용 → task-specific하지 않음, 시간/비용 높음
- 새 접근: Raw hidden state를 직접 사용 → 더 직접적인 분석

---

## 데이터

### 소스
기존 slot machine 실험 결과를 재활용합니다 (새 대화 생성 없음).

| Model | 파일 | Games | Bankruptcy Rate |
|-------|------|-------|-----------------|
| Gemma-2-9B | `slot_machine/gemma/final_gemma_20251004_172426.json` | 3,200 | ~20.9% |
| LLaMA-3.1-8B | `slot_machine/llama/final_llama_20251004_021106.json` | 3,200 | TBD |

### 실험 조건 (64 conditions)
- **bet_type**: fixed, variable (2)
- **prompt_combo**: BASE + 31 combinations (32)
  - G: Goal setting ("set a target amount")
  - M: Maximize reward ("maximize the reward")
  - R: Hidden patterns ("may have hidden patterns")
  - W: Win multiplier ("3.0 times your bet")
  - P: Win rate ("Win rate: 30%")

---

## 분석 옵션

### Option A: 시작 시점 (Start Point)
- 게임 시작 전 상태 (history 없음, balance=$100)
- 동일 조건 내에서 동일한 프롬프트 → 모델이 같은 입력에 다른 내부 상태를 가지는지 테스트
- 샘플 수: ~3,200

### Option B: 종료 직전 (End Point) - **핵심 실험**
- 게임의 마지막 결정 직전 상태
- **Bankruptcy**: 최종 패배 베팅 직전 상태
- **Voluntary Stop**: STOP 결정 직전 상태
- **Immediate Stop**: 초기 상태 (Round 0에서 바로 중단)
- 샘플 수: ~3,200

### Option C: 전체 라운드 (All Rounds)
- 게임의 모든 결정 시점
- Trajectory 분석에 유용
- 샘플 수: ~32,000+

---

## 파이프라인

```
JSON 데이터 → 프롬프트 재구성 → 모델 Forward Pass → Hidden State → LR 분류 → 평가
                                      ↓
                              Baseline 비교
```

---

## 사용법

### 기본 실행 (GPU 필요)

```bash
# Option B (핵심 실험) - Gemma
python run_experiment.py --model gemma --option B --gpu 0

# Quick 모드 (적은 레이어만)
python run_experiment.py --model gemma --option B --gpu 0 --quick

# Full 모드 (모든 레이어)
python run_experiment.py --model gemma --option B --gpu 0 --full

# 특정 레이어만
python run_experiment.py --model gemma --option B --gpu 0 --layers 15,20,25,30,35

# 캐시된 hidden state 사용 (재실행 시)
python run_experiment.py --model gemma --option B --skip-extraction

# 모든 옵션 실행
python run_experiment.py --model gemma --option all --gpu 0

# 두 모델 모두
python run_experiment.py --model all --option B --gpu 0
```

### Baseline만 실행 (GPU 불필요)

```bash
# GPU 없이 baseline만 실행
python run_experiment.py --model gemma --option B --baselines-only
```

### 옵션 설명

| 옵션 | 설명 |
|------|------|
| `--model` | gemma, llama, all |
| `--option` | A, B, C, all (B=핵심) |
| `--gpu` | GPU ID |
| `--quick` | 적은 레이어만 (layers 15,20,25,30,35) |
| `--full` | 모든 레이어 (2개씩 건너뛰며) |
| `--layers` | 특정 레이어 지정 (예: 15,20,25) |
| `--skip-extraction` | 캐시된 hidden state 사용 |
| `--no-baselines` | baseline 비교 건너뛰기 |
| `--no-groups` | 그룹 분석 건너뛰기 |
| `--baselines-only` | baseline만 실행 (GPU 불필요) |

---

## 평가 지표

### 1. Accuracy (정확도)
```
Accuracy = (TP + TN) / Total
```
- 전체 예측 중 맞춘 비율
- 클래스 불균형 시 misleading 가능

### 2. AUC-ROC
- ROC 곡선 아래 면적
- 0.5 = random, 1.0 = perfect
- 클래스 불균형에 robust

### 3. Precision (정밀도)
```
Precision = TP / (TP + FP)
```
- "bankruptcy"로 예측한 것 중 실제 bankruptcy 비율

### 4. Recall (재현율)
```
Recall = TP / (TP + FN)
```
- 실제 bankruptcy 중 맞춘 비율

### 5. F1 Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Precision과 Recall의 조화 평균

### 6. Cross-Validation
- 5-Fold Stratified CV
- mean ± std로 보고

---

## Baseline 비교

### 1. Chance Level
- 다수 클래스 예측 (항상 safe 예측)
- Gemma 기준: ~79.1%
- Hidden state가 이보다 나아야 의미 있음

### 2. Text-Only (TF-IDF)
- 프롬프트 텍스트만 사용 (hidden state 없이)
- TF-IDF로 feature 추출 후 LR
- Hidden state가 텍스트 이상의 정보를 담는지 확인

### 3. Metadata-Only
- 게임 상태 메타데이터만 사용
- Features: balance, n_rounds, n_wins, n_losses, consecutive_losses 등
- Hidden state가 명시적 상태 이상의 정보를 담는지 확인

### 4. Random Projection
- Hidden state를 랜덤 매트릭스로 차원 축소
- LR 성능이 structured information 때문인지 확인

---

## 분석 그룹

### bet_type별 분석
| Group | N | 설명 |
|-------|---|------|
| fixed | ~1,600 | 고정 베팅 ($10) |
| variable | ~1,600 | 가변 베팅 ($5~$balance) |

### prompt_combo별 분석
각 조합에 대해 개별 분석 (32 groups × 50 repetitions)

---

## 폴더 구조

```
lr_classification_experiment/
├── README.md                      # 이 문서
├── EXPERIMENT_DESIGN.md           # 상세 실험 설계
├── config.yaml                    # 설정 파일
├── run_experiment.py              # 메인 실행 스크립트
└── src/
    ├── __init__.py
    ├── prompt_reconstruction.py   # 프롬프트 재구성
    ├── hidden_state_extractor.py  # Hidden state 추출
    ├── lr_classifier.py           # LR 분류 및 평가
    └── baselines.py               # Baseline 비교
```

---

## 출력 파일

실행 후 생성되는 파일:

```
{output_dir}/
├── hidden_states_{model}_option{opt}.npz    # 캐시된 hidden states
├── lr_result_{model}_option{opt}_{ts}.json  # LR 분류 결과
└── baselines_{model}_option{opt}_{ts}.json  # Baseline 결과
```

---

## GPU 요구사항

| Model | bf16 Memory | 권장 GPU |
|-------|-------------|----------|
| Gemma-2-9B | ~22GB | RTX 3090, A100, etc. |
| LLaMA-3.1-8B | ~19GB | RTX 3090, A100, etc. |

**중요**: bf16 precision을 사용해야 원본 실험과 동일한 activation을 얻을 수 있습니다.
Quantization(4-bit, 8-bit)은 activation을 변형시키므로 사용하지 마세요.

---

## 예상 결과 해석

### 시나리오 1: LR >> Chance Level
```
예: LR Accuracy = 75%, Chance = 50%
```
- Hidden state가 결과 예측 정보를 담고 있음
- 모델이 "위험 상태"를 내부적으로 인코딩

### 시나리오 2: LR ≈ Chance Level
```
예: LR Accuracy = 52%, Chance = 50%
```
- Hidden state만으로는 구분 불가
- 결과가 random하거나 다른 표현 공간 필요

### 시나리오 3: LR >> Metadata Baseline
```
예: LR = 75%, Metadata = 60%
```
- Hidden state가 메타데이터 이상의 정보 포함
- 모델이 "잠재적 상태"를 인코딩

### 시나리오 4: Later Layers > Earlier Layers
```
예: Layer 35 = 75%, Layer 10 = 55%
```
- 후반 레이어에서 추상적 표현 형성
- 결정 관련 정보가 깊은 레이어에 집중

---

## 관련 참고문헌

- Ji-An et al. (2025). "Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations" (arXiv:2505.13763)
- LR을 사용하여 activation space에서 의미있는 방향을 식별

---

## 체크리스트

- [x] Option A/B/C 정의 및 구현
- [x] 프롬프트 재구성 (원본 실험과 정확히 일치)
- [x] Hidden state 추출
- [x] LR 분류 및 모든 평가 지표
- [x] bet_type별 그룹 분석
- [x] prompt_combo별 그룹 분석
- [x] Baseline 비교 (4가지)
- [x] 결과 저장 및 캐싱
- [ ] GPU 환경에서 실험 실행
- [ ] 결과 분석 및 시각화

---

## Author

LLM Addiction Research Team
Last Updated: 2025-01
