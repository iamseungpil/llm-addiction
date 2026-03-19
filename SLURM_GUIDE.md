# Experiment Execution Guide

이 프로젝트의 실험 실행 가이드입니다.

## 현재 환경: OpenHPC (Kubernetes/JupyterHub)

> **Note**: 이전에는 SLURM HPC 클러스터를 사용했으나, 현재는 OpenHPC 환경으로 이전했습니다.
> GPU가 직접 할당되어 있어 `sbatch`/`srun` 없이 바로 실행 가능합니다.

### GPU 정보

| GPU | 모델 | VRAM | CUDA |
|-----|------|------|------|
| GPU 0 | NVIDIA A100-SXM4-40GB | 39.5GB | 12.9 |
| GPU 1 | NVIDIA A100-SXM4-40GB | 39.5GB | 12.9 |

### 모델별 GPU 사용량

| 모델 | VRAM 요구량 | A100 40GB 적합 |
|------|-------------|----------------|
| LLaMA-3.1-8B | ~19GB (bf16) | O (2개 동시 가능) |
| Gemma-2-9B | ~22GB (bf16) | O |
| Qwen models | ~19GB (bf16) | O (2개 동시 가능) |

### 시스템 사양

- **CPU**: 100 cores
- **RAM**: 1TB
- **Python**: 3.13.11 (Anaconda)
- **PyTorch**: 2.8.0+cu128

## 실험 실행 방법

### 기본 실행

```bash
# 리포지토리로 이동
cd /home/jovyan/llm-addiction

# GPU 0에서 실행
python your_experiment.py --gpu 0

# GPU 1에서 실행
CUDA_VISIBLE_DEVICES=1 python your_experiment.py --gpu 0
```

### 백그라운드 실행

```bash
# nohup으로 백그라운드 실행
nohup python your_experiment.py --gpu 0 \
  > /home/jovyan/beomi/llm-addiction-data/logs/experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# PID 확인
echo $!

# 로그 실시간 확인
tail -f /home/jovyan/beomi/llm-addiction-data/logs/experiment_*.log
```

### 병렬 실행 (2개 GPU 동시 사용)

```bash
# GPU 0: LLaMA 실험
CUDA_VISIBLE_DEVICES=0 nohup python experiment.py --model llama --gpu 0 \
  > /home/jovyan/beomi/llm-addiction-data/logs/llama_exp.log 2>&1 &

# GPU 1: Gemma 실험
CUDA_VISIBLE_DEVICES=1 nohup python experiment.py --model gemma --gpu 0 \
  > /home/jovyan/beomi/llm-addiction-data/logs/gemma_exp.log 2>&1 &
```

## 실험별 실행 예시

### Blackjack 실험

```bash
python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
    --model llama \
    --gpu 0
```

### Investment Choice 실험

```bash
python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py \
    --model gemma \
    --gpu 0
```

### SAE 분석 파이프라인

```bash
# Phase 1: Feature extraction
python paper_experiments/llama_sae_analysis/src/phase1_feature_extraction.py

# Phase 2: Correlation analysis
python paper_experiments/llama_sae_analysis/src/phase2_correlation_analysis.py

# Phase 4: Causal validation
python paper_experiments/llama_sae_analysis/src/phase4_causal_pilot_v2.py
```

## 모니터링

```bash
# GPU 상태
nvidia-smi
watch -n 1 nvidia-smi

# 실행 중인 Python 프로세스
ps aux | grep python

# 로그 확인
tail -f /home/jovyan/beomi/llm-addiction-data/logs/*.log

# 백그라운드 잡 확인
jobs -l
```

## 프로세스 관리

```bash
# 특정 프로세스 종료
kill <PID>

# GPU를 사용하는 프로세스 확인
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

## 디렉토리 구조

```
/home/jovyan/beomi/llm-addiction-data/
├── logs/                  # 실험 로그
├── investment_choice/     # Investment choice 실험 결과
├── blackjack/             # Blackjack 실험 결과
└── slot_machine/          # Slot machine 실험 결과
```

## 주의사항

1. **bf16 필수**: 모델은 항상 bf16으로 로딩 (float16/quantized 사용 금지)
2. **GPU 메모리**: 실험 간 `clear_gpu_memory()` 호출 필수
3. **데이터 경로**: 결과는 반드시 `/home/jovyan/beomi/llm-addiction-data/`에 저장
4. **프로젝트 경로**: 코드는 `/home/jovyan/llm-addiction/`

## 레거시 SLURM 스크립트

이전 SLURM 환경의 쉘 스크립트들은 `scripts/` 디렉토리에 남아있으며, `#SBATCH` 라인은 `[SLURM-DISABLED]`로 주석 처리되었습니다. Python 실행 명령어는 여전히 유효하므로 참고용으로 활용 가능합니다.
