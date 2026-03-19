#!/bin/bash
# Run LLaMA IC experiment with Gemma V2role design (symmetric)
# 4 constraints × 2 bet types × 4 conditions × 50 reps = 1600 games
# GPU 0, estimated ~3-4 hours

set -e

cd /home/jovyan/llm-addiction
OUTPUT_DIR="/home/jovyan/beomi/llm-addiction-data/investment_choice_v2_role_llama"

echo "=========================================="
echo "LLaMA IC V2role Experiment"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"
echo "=========================================="

for CONSTRAINT in 10 30 50 70; do
    echo ""
    echo ">>> Constraint: c${CONSTRAINT} — $(date)"
    python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py \
        --model llama \
        --gpu 0 \
        --constraint ${CONSTRAINT} \
        --output-dir ${OUTPUT_DIR}
    echo ">>> c${CONSTRAINT} done — $(date)"
done

echo ""
echo "=========================================="
echo "ALL DONE — $(date)"
echo "=========================================="
