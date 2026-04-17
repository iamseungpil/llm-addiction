#!/bin/bash
# run_c P2 (optional): Exp B Gemma SM/MW Phase 2 null (Gemma floor conditions)
# Launch after P1 canonical if time permits.
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
NGAMES=200
CONCURRENT=2
LOGDIR=results/logs

echo "=== run_c v7 P2 (Gemma floor null) ==="

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29610 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 3 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expb_sm.log" 2>&1 &
echo "GPU 3 Exp B Gemma SM Phase 2: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29611 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 3 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expb_mw.log" 2>&1 &
echo "GPU 3 Exp B Gemma MW Phase 2: PID $!"

echo "=== P2 launched (2 procs) ==="
sleep 10
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
