#!/usr/bin/env bash
set -uo pipefail

# Track D top-K removal: 2 models × 3 K values × (1 top + 50 random baselines).
# Each (model, K) cell does both top and random in one invocation; the 50
# random replicates are managed inside the script via --n_random.

OUTPUT_DIR="/scratch/x3415a02/data/llm-addiction/d_robustness"
mkdir -p "${OUTPUT_DIR}" /scratch/logs

for model in gemma llama; do
    for K in 10 50 100; do
        echo "[d] (${model}, K=${K}) — top + 50 random baselines"
        python paper_experiments/d_distributed_effect/src/run_d_topk_removal.py \
            --model "${model}" --layer 22 --task sm --indicator i_ba \
            --K "${K}" --only both --n_random 50 \
            --output_dir "${OUTPUT_DIR}" \
            2>&1 | tee -a "/scratch/logs/d_${model}_K${K}.log" || \
            echo "[d] cell ${model}/${K} FAILED, continuing"
    done
done

python paper_experiments/d_distributed_effect/src/analyze_d.py \
    --per_run_dir "${OUTPUT_DIR}/per_run" \
    --output_dir "${OUTPUT_DIR}" --model all || \
    echo "[d] analyze FAILED"

echo "[d] grid done"
