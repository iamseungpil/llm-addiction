# Alternative Paradigms for LLM Addiction Research

3가지 대안적 실험 패러다임을 통해 슬롯머신 실험 결과의 도메인 일반화를 검증합니다.

## 실험 개요

### 1. Investment Choice Task
**도메인**: 투자 선택
**측정**: 리스크 선호도, 손실 회피, 의사결정 패턴
**게임 수**: 160 (Quick mode) / 3,200 (Full mode)
**시행/게임**: 변동 (평균 25-40회)

### 2. Blackjack (Near-Miss)
**도메인**: 블랙잭 카드 게임
**측정**: Near-miss 효과, 통제 환상, 인지 왜곡
**게임 수**: 320 (Quick mode)
**시행/게임**: 변동 (평균 20-30회)

## 폴더 구조

```
alternative_paradigms/
├── README.md                    # 이 문서
├── src/
│   ├── common/                  # 공통 유틸리티
│   │   ├── __init__.py
│   │   ├── model_loader.py      # 모델 로딩 (LLaMA, Gemma, Qwen)
│   │   ├── prompt_builder.py    # 프롬프트 생성
│   │   └── utils.py             # 유틸리티 함수
│   ├── investment_choice/       # Investment Choice Task
│   │   ├── __init__.py
│   │   ├── game_logic.py        # 게임 로직
│   │   └── run_experiment.py    # 실험 실행
│   └── blackjack/               # Blackjack (Near-Miss)
│       ├── __init__.py
│       ├── game_logic.py
│       └── run_experiment.py
└── scripts/
    ├── run_quick_all.sh         # 모든 실험 실행 (Quick mode)
    ├── test_mini.sh             # 미니 테스트
    └── test_quick.sh            # 빠른 테스트
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

#### 1. Investment Choice

```bash
# LLaMA 모델
python src/investment_choice/run_experiment.py --model llama --gpu 0

# Gemma 모델
python src/investment_choice/run_experiment.py --model gemma --gpu 0

# Quick mode (160 games)
python src/investment_choice/run_experiment.py --model gemma --gpu 0 --quick
```

#### 2. Blackjack

```bash
# LLaMA 모델
python src/blackjack/run_experiment.py --model llama --gpu 0

# Gemma 모델
python src/blackjack/run_experiment.py --model gemma --gpu 0

# Quick mode (320 games)
python src/blackjack/run_experiment.py --model gemma --gpu 0 --quick
```

#### 모든 실험 실행

```bash
# Quick mode로 모든 실험 실행
bash scripts/run_quick_all.sh gemma
```

## 출력 파일

결과는 `/scratch/x3415a02/data/llm-addiction/`에 저장됩니다:

```
data/llm-addiction/
├── investment_choice/
│   ├── llama_investment_*.json
│   └── gemma_investment_*.json
└── blackjack/
    ├── llama_blackjack_*.json
    └── gemma_blackjack_*.json
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

- **Investment Choice**: Classical risk-taking paradigms in behavioral economics
- **Blackjack Near-Miss**: [PubMed 2024](https://pubmed.ncbi.nlm.nih.gov/38709628/)

## 문의

LLM Addiction Research Team
Last Updated: 2026-01-30
