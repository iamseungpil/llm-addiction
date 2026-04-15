#!/bin/bash
# =============================================================================
# E8 Parallel Launcher v6 — 2-3 processes per GPU for ~2x throughput
# =============================================================================
#
# Strategy:
#   - LLaMA 8B ≈ 16GB VRAM. Two copies on A100 80GB = 32GB (48% utilization).
#   - Gemma 9B ≈ 20GB VRAM. Two copies = 40GB (49% utilization).
#   - Experiments sharing the same model type are co-located on the same GPU.
#   - --task-filter splits multi-task experiments across GPUs.
#   - --concurrent-games=2 adds CPU-GPU overlap within each process.
#
# GPU allocation (8 processes on 4 GPUs):
#   GPU 0: Exp C (LLaMA SM)          + Exp A SM-only (LLaMA)
#   GPU 1: V16  (LLaMA SM L22)       + Exp A IC-only (LLaMA)
#   GPU 2: Exp B IC-only (Gemma)     + Exp B SM-only (Gemma)
#   GPU 3: Exp A MW-only (LLaMA)     + Exp B MW-only (Gemma) ← mixed but fits
#
# Expected speedup: ~2x vs single-process-per-GPU
# Expected total time: ~2-3 days (down from 5-6 days)
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
NGAMES="${NGAMES:-200}"        # override with NGAMES=100 for faster run
CONCURRENT="${CONCURRENT:-2}"  # games in parallel within each process
LOGDIR=results/logs
mkdir -p "$LOGDIR"

echo "=== E8 Parallel Launcher v6 ==="
echo "  n_games=$NGAMES, concurrent=$CONCURRENT"
echo "  GPUs: 4 × A100 80GB, 8 processes"
echo ""

# ── GPU 0: Exp C + Exp A (SM only) — both LLaMA ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment c --model llama --gpu 0 \
        --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/e8_v6_expc.log" 2>&1 &
echo "GPU 0 [1/2] Exp C: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29510 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 0 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/e8_v6_expa_sm.log" 2>&1 &
echo "GPU 0 [2/2] Exp A (SM): PID $!"

# ── GPU 1: V16 + Exp A (IC only) — both LLaMA ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 \
    nohup $PYTHON -u src/run_v16_multilayer_steering.py \
        --model llama --layers 2 \
        --alpha-mode absolute --alpha-absolute-base 1.0 \
        --n-bk-games "$NGAMES" --n-rand-games 100 --n-random-dirs 20 \
        --tag v16_v6 \
    > "$LOGDIR/e8_v6_v16.log" 2>&1 &
echo "GPU 1 [1/2] V16: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29511 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/e8_v6_expa_ic.log" 2>&1 &
echo "GPU 1 [2/2] Exp A (IC): PID $!"

# ── GPU 2: Exp B (IC) + Exp B (SM) — both Gemma ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29502 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/e8_v6_expb_ic.log" 2>&1 &
echo "GPU 2 [1/2] Exp B (IC): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29512 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/e8_v6_expb_sm.log" 2>&1 &
echo "GPU 2 [2/2] Exp B (SM): PID $!"

# ── GPU 3: Exp A (MW) + Exp B (MW) — LLaMA + Gemma ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29503 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 3 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/e8_v6_expa_mw.log" 2>&1 &
echo "GPU 3 [1/2] Exp A (MW): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29513 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 3 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/e8_v6_expb_mw.log" 2>&1 &
echo "GPU 3 [2/2] Exp B (MW): PID $!"

echo ""
echo "=== 8 processes launched ==="
echo "Monitor: tail -f $LOGDIR/e8_v6_*.log"
echo "Check:   nvidia-smi"
sleep 5
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
