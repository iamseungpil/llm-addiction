#!/bin/bash
# run_c Wave 2: SA_LLaMA_MW + SA_Gemma_MW (auto-dispatched when Wave 1 IC conditions finish)
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
SA_NULL_GAMES="${SA_NULL_GAMES:-50}"
SA_NULL_DIRS="${SA_NULL_DIRS:-100}"
LOGDIR=results/logs

echo "=== run_c Wave 2 launcher ==="

# Pick GPU with freest memory via nvidia-smi parse (free > 60GB preferred)
# Simple: put SA LLaMA MW on GPU 1 (2L slot), SA Gemma MW on GPU 3 (Gemma slot)
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29532 \
    nohup $PY -u src/run_shared_axis_steering.py \
        --model llama --task mw --gpu 1 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_sa_llama_mw.log" 2>&1 &
echo "GPU 1 SA LLaMA MW: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29523 \
    nohup $PY -u src/run_shared_axis_steering.py \
        --model gemma --task mw --gpu 3 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_sa_gemma_mw.log" 2>&1 &
echo "GPU 3 SA Gemma MW: PID $!"

echo "=== Wave 2 launched (2 procs) ==="
sleep 10
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
