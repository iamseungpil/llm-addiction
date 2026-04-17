#!/bin/bash
# D1 Gemma baseline inflation — 3-condition parallel launcher for run_c node
#
# GPU allocation (3 processes on 3 GPUs):
#   GPU 0: D1-test (Gemma SM, variable + G-prompts)
#   GPU 1: D1-neg  (Gemma SM, variable + no-G prompts)
#   GPU 2: D1-pos  (LLaMA SM, variable + G-prompts)
# GPU 3: reserved/idle
set -euo pipefail

cd /scratch/llm_addiction/sae_v3_analysis

ENV_COMMON=(
    LLM_ADDICTION_ANALYSIS_ROOT=/scratch/llm_addiction/sae_v3_analysis
    LLM_ADDICTION_DATA_ROOT=/scratch/llm_addiction/data/sae_features_v3
    LLM_ADDICTION_BEHAVIORAL_ROOT=/scratch/llm_addiction/data/behavioral
    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
    TORCHDYNAMO_DISABLE=1
    HF_TOKEN="${HF_TOKEN:-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}"
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-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}"
)

PYTHON=/opt/conda/envs/ptca/bin/python
LOGDIR=results/logs
mkdir -p "$LOGDIR"

NMAIN="${NMAIN:-200}"
NNULL_GAMES="${NNULL_GAMES:-50}"
NNULL_DIRS="${NNULL_DIRS:-100}"

echo "=== D1 launcher (run_c): n_main=$NMAIN null_games=$NNULL_GAMES null_dirs=$NNULL_DIRS ==="

# GPU 0: D1-test
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29600 \
    nohup $PYTHON -u src/run_d1_gemma_inflation.py \
        --condition D1-test --gpu 0 \
        --n-main "$NMAIN" --n-null-games "$NNULL_GAMES" --n-null-dirs "$NNULL_DIRS" \
    > "$LOGDIR/d1_test.log" 2>&1 &
echo "GPU 0 D1-test PID=$!"

# GPU 1: D1-neg
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29601 \
    nohup $PYTHON -u src/run_d1_gemma_inflation.py \
        --condition D1-neg --gpu 1 \
        --n-main "$NMAIN" --n-null-games "$NNULL_GAMES" --n-null-dirs "$NNULL_DIRS" \
    > "$LOGDIR/d1_neg.log" 2>&1 &
echo "GPU 1 D1-neg PID=$!"

# GPU 2: D1-pos
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29602 \
    nohup $PYTHON -u src/run_d1_gemma_inflation.py \
        --condition D1-pos --gpu 2 \
        --n-main "$NMAIN" --n-null-games "$NNULL_GAMES" --n-null-dirs "$NNULL_DIRS" \
    > "$LOGDIR/d1_pos.log" 2>&1 &
echo "GPU 2 D1-pos PID=$!"

echo ""
echo "=== 3 processes launched ==="
echo "Monitor: tail -f $LOGDIR/d1_*.log"
sleep 10
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
