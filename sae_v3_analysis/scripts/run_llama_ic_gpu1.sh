#!/bin/bash
# Run LLaMA IC c50+c70 on GPU 1 (parallel with GPU 0 running c30)
set -e
cd /home/jovyan/llm-addiction
OUTPUT_DIR="/home/jovyan/beomi/llm-addiction-data/investment_choice_v2_role_llama_gpu1"
MAIN_DIR="/home/jovyan/beomi/llm-addiction-data/investment_choice_v2_role_llama"

echo "=========================================="
echo "LLaMA IC V2role — GPU 1 (c50, c70)"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"
echo "=========================================="

for CONSTRAINT in 50 70; do
    echo ""
    echo ">>> Constraint: c${CONSTRAINT} — $(date)"
    CUDA_VISIBLE_DEVICES=1 python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py \
        --model llama \
        --gpu 0 \
        --constraint ${CONSTRAINT} \
        --output-dir ${OUTPUT_DIR}
    echo ">>> c${CONSTRAINT} done — $(date)"
    # Copy final file to main directory
    cp ${OUTPUT_DIR}/llama_investment_c${CONSTRAINT}_*.json ${MAIN_DIR}/ 2>/dev/null || true
    echo ">>> Copied c${CONSTRAINT} results to main dir"
done

echo ""
echo "=========================================="
echo "GPU 1 ALL DONE — $(date)"
echo "=========================================="
