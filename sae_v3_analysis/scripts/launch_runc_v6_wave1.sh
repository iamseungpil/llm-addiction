#!/bin/bash
# run_c 4-GPU Wave 1 launcher (Plan v6)
# 12 new procs on top of existing D1-test (GPU 0) + D1-pos (GPU 2) — D1-neg already killed.
#
# GPU layout:
#   GPU 0: D1-test(G)       + SA_LLaMA_SM(L)   + Exp B Gemma IC(G)    [1L+2G]
#   GPU 1: Exp A LLaMA IC(L) + Exp A LLaMA MW(L) + SA_Gemma_SM(G)      [2L+1G]
#   GPU 2: D1-pos(L)        + SA_LLaMA_IC(L)   + Exp B Gemma SM(G)    [2L+1G]
#   GPU 3: Exp B Gemma MW(G) + SA_Gemma_IC(G) + V16_multilayer(L)      [1L+2G]
#
# Wave 2 (deferred): SA_LLaMA_MW, SA_Gemma_MW (launched by local monitor auto-dispatch)

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
mkdir -p "$LOGDIR" results/checkpoints results/json

echo "=== run_c Wave 1 launcher v6 ==="
echo "  n_games=$NGAMES, concurrent=$CONCURRENT, sa_null_dirs=$SA_NULL_DIRS"
echo ""

# ── GPU 0: SA LLaMA SM + Exp B Gemma IC (D1-test continues on same GPU) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29530 \
    nohup $PY -u src/run_shared_axis_steering.py \
        --model llama --task sm --gpu 0 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_sa_llama_sm.log" 2>&1 &
echo "GPU 0 SA LLaMA SM: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29540 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 0 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_expb_ic.log" 2>&1 &
echo "GPU 0 Exp B Gemma IC: PID $!"

# ── GPU 1: Exp A LLaMA IC + Exp A LLaMA MW + SA Gemma SM ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_expa_ic.log" 2>&1 &
echo "GPU 1 Exp A LLaMA IC: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29511 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 1 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_expa_mw.log" 2>&1 &
echo "GPU 1 Exp A LLaMA MW: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29531 \
    nohup $PY -u src/run_shared_axis_steering.py \
        --model gemma --task sm --gpu 1 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_sa_gemma_sm.log" 2>&1 &
echo "GPU 1 SA Gemma SM: PID $!"

# ── GPU 2: SA LLaMA IC + Exp B Gemma SM (D1-pos continues on same GPU) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29521 \
    nohup $PY -u src/run_shared_axis_steering.py \
        --model llama --task ic --gpu 2 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_sa_llama_ic.log" 2>&1 &
echo "GPU 2 SA LLaMA IC: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29550 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_expb_sm.log" 2>&1 &
echo "GPU 2 Exp B Gemma SM: PID $!"

# ── GPU 3: Exp B Gemma MW + SA Gemma IC + V16 multilayer ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29560 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 3 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_expb_mw.log" 2>&1 &
echo "GPU 3 Exp B Gemma MW: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29513 \
    nohup $PY -u src/run_shared_axis_steering.py \
        --model gemma --task ic --gpu 3 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v6_sa_gemma_ic.log" 2>&1 &
echo "GPU 3 SA Gemma IC: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29520 \
    nohup $PY -u src/run_v16_multilayer_steering.py \
        --model llama --layers 5 \
        --alpha-mode absolute --alpha-absolute-base 1.0 \
        --n-bk-games "$NGAMES" --n-rand-games 100 --n-random-dirs 20 \
        --tag v16_runc \
    > "$LOGDIR/runc_v6_v16.log" 2>&1 &
echo "GPU 3 V16 multilayer: PID $!"

echo ""
echo "=== 10 Wave 1 processes launched ==="
echo "Monitor: tail -f $LOGDIR/runc_v6_*.log"
sleep 18
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
