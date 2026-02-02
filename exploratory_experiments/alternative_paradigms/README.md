# Alternative Paradigms for LLM Addiction Research

3가지 대안적 실험 패러다임을 통해 슬롯머신 실험 결과의 도메인 일반화를 검증합니다.

## 실험 개요

### 1. Iowa Gambling Task (IGT)
**도메인**: 카드 게임
**측정**: 학습 메커니즘, 불확실성 하 의사결정, 처벌 vs 보상 학습
**게임 수**: 3,200 (64 조건 × 50 반복)
**시행/게임**: 100회 (고정)

### 2. Loot Box Mechanics
**도메인**: 비금전적 게임 소액 거래
**측정**: 가변 강화 일정, 희귀 아이템 추적, near-miss 효과
**게임 수**: 3,200 (64 조건 × 50 반복)
**시행/게임**: 변동 (평균 25-40회)

### 3. Near-Miss Enhancement
**도메인**: 슬롯머신 (기존 실험 확장)
**측정**: Near-miss 효과, 통제 환상, 인지 왜곡
**게임 수**: 6,400 (슬롯머신의 2배)
**시행/게임**: 변동 (평균 32회)

## 폴더 구조

```
alternative_paradigms/
├── README.md                    # 이 문서
├── configs/
│   ├── igt_config.yaml          # IGT 실험 설정
│   ├── lootbox_config.yaml      # Loot Box 실험 설정
│   └── nearmiss_config.yaml     # Near-Miss 실험 설정
├── src/
│   ├── common/                  # 공통 유틸리티
│   │   ├── __init__.py
│   │   ├── model_loader.py      # 모델 로딩 (LLaMA, Gemma, Qwen)
│   │   ├── prompt_builder.py    # 프롬프트 생성
│   │   └── utils.py             # 유틸리티 함수
│   ├── igt/                     # Iowa Gambling Task
│   │   ├── __init__.py
│   │   ├── game_logic.py        # 게임 로직
│   │   └── run_experiment.py    # 실험 실행
│   ├── lootbox/                 # Loot Box Mechanics
│   │   ├── __init__.py
│   │   ├── game_logic.py
│   │   └── run_experiment.py
│   └── nearmiss/                # Near-Miss Enhancement
│       ├── __init__.py
│       ├── game_logic.py
│       └── run_experiment.py
├── scripts/
│   ├── run_igt.sh               # IGT 실행 스크립트
│   ├── run_lootbox.sh           # Loot Box 실행 스크립트
│   └── run_nearmiss.sh          # Near-Miss 실행 스크립트
└── docs/
    ├── IGT_DESIGN.md            # IGT 실험 설계 문서
    ├── LOOTBOX_DESIGN.md        # Loot Box 실험 설계 문서
    └── NEARMISS_DESIGN.md       # Near-Miss 실험 설계 문서
```

## 사용법

### 환경 설정

```bash
# Conda 환경 활성화
conda activate llama_sae_env

# CUDA GPU 설정
export CUDA_VISIBLE_DEVICES=0
```

### 실험 실행

#### 1. Iowa Gambling Task

```bash
# LLaMA 모델
python src/igt/run_experiment.py --model llama --gpu 0

# Gemma 모델
python src/igt/run_experiment.py --model gemma --gpu 0

# Qwen 모델
python src/igt/run_experiment.py --model qwen --gpu 0

# 또는 스크립트 사용
bash scripts/run_igt.sh llama 0
```

#### 2. Loot Box Mechanics

```bash
# LLaMA 모델
python src/lootbox/run_experiment.py --model llama --gpu 0

# Gemma 모델
python src/lootbox/run_experiment.py --model gemma --gpu 0

# Qwen 모델
python src/lootbox/run_experiment.py --model qwen --gpu 0

# 또는 스크립트 사용
bash scripts/run_lootbox.sh llama 0
```

#### 3. Near-Miss Enhancement

```bash
# LLaMA 모델
python src/nearmiss/run_experiment.py --model llama --gpu 0

# Gemma 모델
python src/nearmiss/run_experiment.py --model gemma --gpu 0

# Qwen 모델
python src/nearmiss/run_experiment.py --model qwen --gpu 0

# 또는 스크립트 사용
bash scripts/run_nearmiss.sh llama 0
```

## 출력 파일

결과는 `/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms/`에 저장됩니다:

```
alternative_paradigms/
├── igt/
│   ├── llama_igt_YYYYMMDD_HHMMSS.json
│   ├── gemma_igt_YYYYMMDD_HHMMSS.json
│   └── qwen_igt_YYYYMMDD_HHMMSS.json
├── lootbox/
│   ├── llama_lootbox_YYYYMMDD_HHMMSS.json
│   ├── gemma_lootbox_YYYYMMDD_HHMMSS.json
│   └── qwen_lootbox_YYYYMMDD_HHMMSS.json
└── nearmiss/
    ├── llama_nearmiss_YYYYMMDD_HHMMSS.json
    ├── gemma_nearmiss_YYYYMMDD_HHMMSS.json
    └── qwen_nearmiss_YYYYMMDD_HHMMSS.json
```

## GPU 요구사항

| 모델 | bf16 메모리 | 권장 GPU |
|------|------------|---------|
| LLaMA-3.1-8B | ~19GB | RTX 3090, A100 |
| Gemma-2-9B | ~22GB | RTX 3090, A100 |
| Qwen2.5-7B | ~16GB | RTX 3090, A100 |

## 예상 실험 시간

| 실험 | 게임 수 | 예상 시간 (GPU) |
|------|---------|----------------|
| IGT | 3,200 | 30-40시간 |
| Loot Box | 3,200 | 20-30시간 |
| Near-Miss | 6,400 | 40-60시간 |

**총 예상 시간**: 90-130시간 (모델당)

## 아키텍처 패턴

모든 실험은 기존 슬롯머신 실험과 동일한 패턴을 따릅니다:

1. **게임 로직 클래스**: `GameLogic` - 게임 상태, 보상 구조, 히스토리 관리
2. **실험 클래스**: `Experiment` - 모델 로딩, 프롬프트 생성, 실험 실행
3. **Config 기반**: YAML 파일로 모든 하이퍼파라미터 관리
4. **재현 가능성**: Random seed 고정, 체크포인트 저장

## 참고 문헌

- **IGT**: Bechara et al. (1994), [Frontiers 2025](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1492471/full)
- **Loot Box**: [PLOS One 2023](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263177)
- **Near-Miss**: [PubMed 2024](https://pubmed.ncbi.nlm.nih.gov/38709628/)

## 문의

LLM Addiction Research Team
Last Updated: 2026-01-30
