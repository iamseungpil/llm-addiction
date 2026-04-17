#!/bin/bash
# run_c P0 + P1 launcher (Plan v7)
# P0: Repair LLaMA Exp A SM/MW + Exp C SM (Phase 1 α≥0 rerun + Phase 2 null 100 dirs)
# P1: Finalize Exp A IC + Exp B IC (Phase 2 null 100 dirs, Phase 1 auto-skip via ckpt)
#
# GPU layout (5 new procs + 2 existing D1):
#   GPU 0: D1-test(G, cont.) + Exp A LLaMA SM rerun (L)
#   GPU 1: Exp A LLaMA MW rerun (L)  + Exp A LLaMA IC Phase 2 (L)
#   GPU 2: D1-pos(L, cont.)  + Exp C LLaMA SM rerun (L)
#   GPU 3: Exp B Gemma IC Phase 2 (G)  + placeholder for P2 wave

set -euo pipefail

cd /scratch/llm_addiction/sae_v3_analysis

ENV_COMMON=(
    LLM_ADDICTION_ANALYSIS_ROOT=/scratch/llm_addiction/sae_v3_analysis
    LLM_ADDICTION_DATA_ROOT=/scratch/llm_addiction/data/sae_features_v3
    LLM_ADDICTION_BEHAVIORAL_ROOT=/scratch/llm_addiction/data/behavioral
    PYTHONPATH=/scratch/llm_addiction/paper_experiments/slot_machine_6models/src:/scratch/llm_addiction/exploratory_experiments/alternative_paradigms/src
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
    TORCHDYNAMO_DISABLE=1
    HF_TOKEN="${HF_TOKEN:-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}"
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-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}"
)

PY=/opt/conda/envs/ptca/bin/python
NGAMES="${NGAMES:-200}"
CONCURRENT="${CONCURRENT:-2}"
LOGDIR=results/logs
mkdir -p "$LOGDIR" results/checkpoints results/json

echo "=== run_c v7 P0+P1 launcher ==="
echo "  n_games=$NGAMES, concurrent=$CONCURRENT"
echo ""

# ── GPU 0: Exp A LLaMA SM rerun (D1-test continues) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29601 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 0 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expa_sm.log" 2>&1 &
echo "GPU 0 Exp A LLaMA SM rerun: PID $!"

# ── GPU 1: Exp A LLaMA MW rerun + Exp A LLaMA IC Phase 2 ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29602 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expa_mw.log" 2>&1 &
echo "GPU 1 Exp A LLaMA MW rerun: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29612 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expa_ic.log" 2>&1 &
echo "GPU 1 Exp A LLaMA IC Phase 2 only: PID $!"

# ── GPU 2: Exp C LLaMA SM rerun (D1-pos continues) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29603 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment c --model llama --gpu 2 \
        --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expc.log" 2>&1 &
echo "GPU 2 Exp C LLaMA SM rerun: PID $!"

# ── GPU 3: Exp B Gemma IC Phase 2 ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29604 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 3 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expb_ic.log" 2>&1 &
echo "GPU 3 Exp B Gemma IC Phase 2 only: PID $!"

echo ""
echo "=== 5 processes launched ==="
echo "Monitor: tail -f $LOGDIR/runc_v7_*.log"
sleep 15
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
