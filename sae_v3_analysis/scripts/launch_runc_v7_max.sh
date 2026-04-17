#!/bin/bash
# run_c v7 MAX — all 4 GPUs saturated (3 procs per GPU, 11 new procs + 2 D1)
# Priority order per codex: P0 (repair) > P1 (IC finalize) > P2 (Gemma floor null) > SA/V16 (fill slots)
#
# GPU layout (11 new + 2 D1 = 13 procs on 4 GPUs):
#   GPU 0 (1G+2L): D1-test(G) + Exp A LLaMA SM rerun + Exp C LLaMA SM rerun
#   GPU 1 (3L):    Exp A LLaMA MW rerun + Exp A LLaMA IC Phase 2 + V16 multilayer
#   GPU 2 (1L+2G): D1-pos(L) + Exp B Gemma IC Phase 2 + Exp B Gemma SM Phase 2
#   GPU 3 (1L+2G): Exp B Gemma MW Phase 2 + SA LLaMA SM + SA Gemma SM
#
# Memory: each GPU ~56-64GB base, ~100GB peak with H200 141GB → safe margin.

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

echo "=== run_c v7 MAX launcher (11 new procs, all 4 GPUs) ==="
echo "  n_games=$NGAMES, concurrent=$CONCURRENT, sa_null_dirs=$SA_NULL_DIRS"
echo ""

# ── GPU 0: Exp A LLaMA SM rerun + Exp C LLaMA SM rerun (D1-test continues) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29601 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment a --model llama --gpu 0 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expa_sm.log" 2>&1 &
echo "GPU 0 Exp A LLaMA SM rerun: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29605 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment c --model llama --gpu 0 \
        --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expc.log" 2>&1 &
echo "GPU 0 Exp C LLaMA SM rerun: PID $!"

# ── GPU 1: Exp A LLaMA MW rerun + Exp A LLaMA IC Phase 2 + V16 multilayer ──
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
echo "GPU 1 Exp A LLaMA IC Phase 2: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29620 \
    nohup $PY -u src/run_v16_multilayer_steering.py \
        --model llama --layers 5 \
        --alpha-mode absolute --alpha-absolute-base 1.0 \
        --n-bk-games "$NGAMES" --n-rand-games 100 --n-random-dirs 20 \
        --tag v16_runc_v7 \
    > "$LOGDIR/runc_v7_v16.log" 2>&1 &
echo "GPU 1 V16 multilayer LLaMA: PID $!"

# ── GPU 2: Exp B Gemma IC Phase 2 + Exp B Gemma SM Phase 2 (D1-pos continues) ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29604 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter ic --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expb_ic.log" 2>&1 &
echo "GPU 2 Exp B Gemma IC Phase 2: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=2 MASTER_PORT=29614 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 2 \
        --task-filter sm --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expb_sm.log" 2>&1 &
echo "GPU 2 Exp B Gemma SM Phase 2: PID $!"

# ── GPU 3: Exp B Gemma MW Phase 2 + SA LLaMA SM + SA Gemma SM ──
env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29624 \
    nohup $PY -u src/run_aligned_factor_steering.py \
        --experiment b --model gemma --gpu 3 \
        --task-filter mw --n-games "$NGAMES" --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_expb_mw.log" 2>&1 &
echo "GPU 3 Exp B Gemma MW Phase 2: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29630 \
    nohup $PY -u src/run_shared_axis_steering.py \
        --model llama --task sm --gpu 3 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_sa_llama_sm.log" 2>&1 &
echo "GPU 3 SA LLaMA SM: PID $!"

env "${ENV_COMMON[@]}" CUDA_VISIBLE_DEVICES=3 MASTER_PORT=29631 \
    nohup $PY -u src/run_shared_axis_steering.py \
        --model gemma --task sm --gpu 3 \
        --n-main "$NGAMES" --n-null-games "$SA_NULL_GAMES" --n-null-dirs "$SA_NULL_DIRS" \
        --concurrent-games "$CONCURRENT" \
    > "$LOGDIR/runc_v7_sa_gemma_sm.log" 2>&1 &
echo "GPU 3 SA Gemma SM: PID $!"

echo ""
echo "=== 10 new procs launched + 2 D1 = 12 procs active ==="
echo "Monitor: tail -f $LOGDIR/runc_v7_*.log"
sleep 20
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
