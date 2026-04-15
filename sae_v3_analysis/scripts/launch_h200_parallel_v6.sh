#!/bin/bash
# =============================================================================
# H200 Parallel Launcher v6 — 3 processes per GPU for ~3x throughput
# =============================================================================
#
# H200: 143GB VRAM per GPU (vs A100 80GB)
#   - LLaMA 8B ≈ 16GB → 3 copies = 48GB (33% utilization)
#   - Gemma 9B ≈ 20GB → 3 copies = 60GB (42% utilization)
#   - Much higher memory bandwidth (4.8 TB/s) → faster inference
#
# GPU allocation (10 processes on 4 GPUs):
#   GPU 0: Exp C (LLaMA SM)  + Exp A SM (LLaMA) + V16 (LLaMA L22)  [3× LLaMA = 48GB]
#   GPU 1: Exp A IC (LLaMA)  + Exp A MW (LLaMA) + spare             [2× LLaMA = 32GB]
#   GPU 2: Exp B IC (Gemma)  + Exp B SM (Gemma)                     [2× Gemma = 40GB]
#   GPU 3: Exp B MW (Gemma)  + spare                                [1× Gemma = 20GB]
#
# Expected speedup vs single-process A100: ~3-4x
# Expected total time: ~1.5-2 days
# =============================================================================

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
NGAMES="${NGAMES:-200}"
CONCURRENT="${CONCURRENT:-2}"
LOGDIR=results/logs
mkdir -p "$LOGDIR"

echo "=== H200 Parallel Launcher v6 ==="
echo "  n_games=$NGAMES, concurrent=$CONCURRENT"
echo "  GPUs: 4 × H200 143GB, 10 processes"
echo ""

# ── GPU 0: 3× LLaMA (Exp C + Exp A SM + V16) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment c --model llama --gpu 0 \
        --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v6_expc.log" 2>&1 &
echo "GPU 0 [1/3] Exp C: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29510 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 0 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v6_expa_sm.log" 2>&1 &
echo "GPU 0 [2/3] Exp A (SM): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29520 \
    nohup $PYTHON -u src/run_v16_multilayer_steering.py \
        --model llama --layers 2 \
        --alpha-mode absolute --alpha-absolute-base 1.0 \
        --n-bk-games "$NGAMES" --n-rand-games 100 --n-random-dirs 20 \
        --tag v16_h200 \
    > "$LOGDIR/h200_v6_v16.log" 2>&1 &
echo "GPU 0 [3/3] V16: PID $!"

# ── GPU 1: 2× LLaMA (Exp A IC + Exp A MW) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v6_expa_ic.log" 2>&1 &
echo "GPU 1 [1/2] Exp A (IC): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29511 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v6_expa_mw.log" 2>&1 &
echo "GPU 1 [2/2] Exp A (MW): PID $!"

# ── GPU 2: 2× Gemma (Exp B IC + Exp B SM) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29502 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v6_expb_ic.log" 2>&1 &
echo "GPU 2 [1/2] Exp B (IC): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29512 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v6_expb_sm.log" 2>&1 &
echo "GPU 2 [2/2] Exp B (SM): PID $!"

# ── GPU 3: 1× Gemma (Exp B MW) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29503 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 3 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v6_expb_mw.log" 2>&1 &
echo "GPU 3 [1/1] Exp B (MW): PID $!"

echo ""
echo "=== 8 experiment processes launched ==="
echo "Monitor: tail -f $LOGDIR/h200_v6_*.log"
echo "Check:   nvidia-smi"
sleep 10
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
