# SLURM Job Submission Guide

이 프로젝트의 HPC 클러스터 사용 가이드입니다.

## 클러스터 정보

### GPU 파티션

| 파티션 | GPU 종류 | GPU 수/노드 | VRAM | 용도 |
|--------|----------|-------------|------|------|
| `cas_v100nv_8` | V100 | 8 | 32GB | 대규모 실험 |
| `cas_v100nv_4` | V100 | 4 | 32GB | 중규모 실험 |
| `cas_v100_4` | V100 | 4 | 32GB | 일반 실험 |
| `cas_v100_2` | V100 | 2 | 32GB | 소규모 실험 |
| `amd_a100nv_8` | A100 | 8 | 80GB | 대형 모델 |
| `amd_a100_4` | A100 | 4 | 80GB | 대형 모델 |
| `amd_h200nv_8` | H200 | 8 | 141GB | 초대형 모델 |

### 모델별 권장 파티션

| 모델 | VRAM 요구량 | 권장 파티션 |
|------|-------------|-------------|
| LLaMA-3.1-8B | ~19GB (bf16) | `cas_v100_4` |
| Gemma-2-9B | ~22GB (bf16) | `cas_v100_4` |
| 대형 모델 (70B+) | 80GB+ | `amd_a100nv_8` |

## 기본 명령어

### 클러스터 상태 확인

```bash
# 파티션 상태
sinfo

# 내 잡 확인
squeue -u $USER

# 특정 파티션의 가용 노드
sinfo -p cas_v100_4
```

### Interactive 세션

```bash
# V100 1개로 2시간 interactive 세션
srun -p cas_v100_4 --gres=gpu:1 --time=02:00:00 --pty bash

# A100 1개로 4시간 interactive 세션
srun -p amd_a100_4 --gres=gpu:1 --time=04:00:00 --pty bash
```

## Job Script 템플릿

### 기본 템플릿: `scripts/run_experiment.sh`

```bash
#!/bin/bash
#SBATCH --job-name=llm-exp
#SBATCH --partition=cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/%x_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/%x_%j.err

# 환경 설정
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# 작업 디렉토리
cd /scratch/x3415a02/projects/llm-addiction

# 실험 실행
python your_experiment.py --gpu 0 --output-dir /scratch/x3415a02/data/llm-addiction/
```

### Investment Choice 실험

```bash
#!/bin/bash
#SBATCH --job-name=invest-exp
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/investment_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/investment_%j.err

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction
cd /scratch/x3415a02/projects/llm-addiction

python paper_experiments/investment_choice_experiment/src/run_experiment.py \
    --model llama \
    --gpu 0 \
    --output-dir /scratch/x3415a02/data/llm-addiction/investment_choice/
```

### Loot Box 실험

```bash
#!/bin/bash
#SBATCH --job-name=lootbox-exp
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/lootbox_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/lootbox_%j.err

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction
cd /scratch/x3415a02/projects/llm-addiction

python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py \
    --model gemma \
    --gpu 0 \
    --output-dir /scratch/x3415a02/data/llm-addiction/lootbox/
```

### Blackjack 실험

```bash
#!/bin/bash
#SBATCH --job-name=blackjack-exp
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_%j.err

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction
cd /scratch/x3415a02/projects/llm-addiction

python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
    --model llama \
    --gpu 0 \
    --output-dir /scratch/x3415a02/data/llm-addiction/blackjack/
```

## Job 제출 및 관리

### 제출

```bash
# 단일 잡 제출
sbatch scripts/run_experiment.sh

# 여러 모델 동시 실행 (Array Job)
sbatch --array=0-2 scripts/run_multi_model.sh
```

### 모니터링

```bash
# 잡 상태 확인
squeue -u $USER

# 잡 상세 정보
scontrol show job <JOBID>

# 실시간 로그 확인
tail -f /scratch/x3415a02/data/llm-addiction/logs/llm-exp_<JOBID>.out
```

### 취소

```bash
# 특정 잡 취소
scancel <JOBID>

# 내 모든 잡 취소
scancel -u $USER
```

## Array Job (다중 실험)

여러 모델/조건을 병렬로 실행:

```bash
#!/bin/bash
#SBATCH --job-name=multi-model
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/%x_%A_%a.out

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction
cd /scratch/x3415a02/projects/llm-addiction

# 모델 배열
MODELS=("llama" "gemma" "qwen")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

python your_experiment.py --model $MODEL --gpu 0
```

## 디렉토리 구조

```
/scratch/x3415a02/data/llm-addiction/
├── logs/                  # SLURM 로그 (.out, .err)
├── investment_choice/     # 실험 결과
├── blackjack/
├── lootbox/
└── slot_machine/
```

로그 디렉토리 생성:
```bash
mkdir -p /scratch/x3415a02/data/llm-addiction/logs
```

## 주의사항

1. **시간 제한**: 대부분 파티션이 2일(48시간) 제한. 긴 실험은 체크포인트 사용
2. **메모리**: LLaMA/Gemma는 32GB면 충분, 여유 있게 설정
3. **GPU**: bf16 로딩 필수 (float16/quantized 사용 금지)
4. **출력 경로**: 항상 `/scratch/` 사용 (홈 디렉토리 용량 제한)

## Quick Reference

```bash
# 환경 활성화
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Interactive GPU 세션
srun -p cas_v100_4 --gres=gpu:1 --time=02:00:00 --pty bash

# 잡 제출
sbatch scripts/run_experiment.sh

# 상태 확인
squeue -u $USER
```
