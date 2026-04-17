#!/bin/bash
# =============================================================================
# H200 Parallel Launcher v7 — v6 + RQ2 shared-axis causal validation
# =============================================================================
#
# H200: 141GB VRAM per GPU
#   - LLaMA 8B ≈ 16GB, Gemma 9B ≈ 20GB
#   - All process counts below fit <= ~80GB per GPU (safe margin)
#
# What's new in v7 vs v6:
#   - +6 shared_axis processes (2 models × 3 tasks) for RQ2 causal validation
#   - Uses same checkpoint mechanism (alpha-level resume) via --checkpoint-key shared_axis_{model}
#
# GPU allocation (14 processes on 4 GPUs):
#   GPU 0: Exp C (LLaMA)    + Exp A SM       + V16               + shared_axis LLaMA SM   [4× LLaMA = 64GB]
#   GPU 1: Exp A IC (LLaMA) + Exp A MW       + shared_axis LL IC + shared_axis LL MW      [4× LLaMA = 64GB]
#   GPU 2: Exp B IC (Gemma) + Exp B SM       + shared_axis Gemma SM                       [3× Gemma = 60GB]
#   GPU 3: Exp B MW (Gemma) + shared_axis Gemma IC + shared_axis Gemma MW                 [3× Gemma = 60GB]
#
# Expected total time: ~2-3 days with H200 bandwidth + alpha checkpoint resume
# All runs resume from checkpoints/ckpt_* if present (BSC preempt-safe)
# =============================================================================

set -euo pipefail

cd /scratch/llm_addiction/sae_v3_analysis

ENV_COMMON=(
    LLM_ADDICTION_ANALYSIS_ROOT=/scratch/llm_addiction/sae_v3_analysis
    LLM_ADDICTION_DATA_ROOT=/scratch/llm_addiction/data/sae_features_v3
    LLM_ADDICTION_BEHAVIORAL_ROOT=/scratch/llm_addiction/data/behavioral
    PYTHONPATH=/scratch/llm_addiction/paper_experiments/slot_machine_6models/src:/scratch/llm_addiction/exploratory_experiments/alternative_paradigms/src
    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
    TORCHDYNAMO_DISABLE=1
    HF_TOKEN="${HF_TOKEN:-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}"
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-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}"
)

PYTHON=/opt/conda/envs/ptca/bin/python
NGAMES="${NGAMES:-200}"
CONCURRENT="${CONCURRENT:-2}"
SA_NULL_GAMES="${SA_NULL_GAMES:-50}"
SA_NULL_DIRS="${SA_NULL_DIRS:-100}"
LOGDIR=results/logs
mkdir -p "$LOGDIR"

echo "=== H200 Parallel Launcher v7 (aligned + V16 + shared-axis) ==="
echo "  n_games=$NGAMES, concurrent=$CONCURRENT"
echo "  shared_axis: null_games=$SA_NULL_GAMES, null_dirs=$SA_NULL_DIRS"
echo "  GPUs: 4 × H200 141GB, 14 processes"
echo ""

# ── GPU 0: 4× LLaMA ─────────────────────────────────────────────────────────
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment c --model llama --gpu 0 \
        --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_expc.log" 2>&1 &
echo "GPU 0 [1/4] Exp C (LLaMA SM): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29510 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 0 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_expa_sm.log" 2>&1 &
echo "GPU 0 [2/4] Exp A (SM): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29520 \
    nohup $PYTHON -u src/run_v16_multilayer_steering.py \
        --model llama --layers 2 \
        --alpha-mode absolute --alpha-absolute-base 1.0 \
        --n-bk-games "$NGAMES" --n-rand-games 100 --n-random-dirs 20 \
        --tag v16_h200 \
    > "$LOGDIR/h200_v7_v16.log" 2>&1 &
echo "GPU 0 [3/4] V16 multilayer: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29530 \
    nohup $PYTHON -u src/run_shared_axis_steering.py \
        --model llama --task sm --gpu 0 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_sa_llama_sm.log" 2>&1 &
echo "GPU 0 [4/4] SharedAxis LLaMA SM: PID $!"

# ── GPU 1: 4× LLaMA ─────────────────────────────────────────────────────────
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_expa_ic.log" 2>&1 &
echo "GPU 1 [1/4] Exp A (IC): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29511 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_expa_mw.log" 2>&1 &
echo "GPU 1 [2/4] Exp A (MW): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29521 \
    nohup $PYTHON -u src/run_shared_axis_steering.py \
        --model llama --task ic --gpu 1 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_sa_llama_ic.log" 2>&1 &
echo "GPU 1 [3/4] SharedAxis LLaMA IC: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29531 \
    nohup $PYTHON -u src/run_shared_axis_steering.py \
        --model llama --task mw --gpu 1 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_sa_llama_mw.log" 2>&1 &
echo "GPU 1 [4/4] SharedAxis LLaMA MW: PID $!"

# ── GPU 2: 3× Gemma ─────────────────────────────────────────────────────────
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29502 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_expb_ic.log" 2>&1 &
echo "GPU 2 [1/3] Exp B (IC): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29512 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_expb_sm.log" 2>&1 &
echo "GPU 2 [2/3] Exp B (SM): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29522 \
    nohup $PYTHON -u src/run_shared_axis_steering.py \
        --model gemma --task sm --gpu 2 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_sa_gemma_sm.log" 2>&1 &
echo "GPU 2 [3/3] SharedAxis Gemma SM: PID $!"

# ── GPU 3: 3× Gemma ─────────────────────────────────────────────────────────
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29503 \
    nohup $PYTHON -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 3 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_expb_mw.log" 2>&1 &
echo "GPU 3 [1/3] Exp B (MW): PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29513 \
    nohup $PYTHON -u src/run_shared_axis_steering.py \
        --model gemma --task ic --gpu 3 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_sa_gemma_ic.log" 2>&1 &
echo "GPU 3 [2/3] SharedAxis Gemma IC: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29523 \
    nohup $PYTHON -u src/run_shared_axis_steering.py \
        --model gemma --task mw --gpu 3 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/h200_v7_sa_gemma_mw.log" 2>&1 &
echo "GPU 3 [3/3] SharedAxis Gemma MW: PID $!"

echo ""
echo "=== 14 processes launched ==="
echo "Monitor: tail -f $LOGDIR/h200_v7_*.log"
echo "Check:   nvidia-smi"
sleep 12
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
