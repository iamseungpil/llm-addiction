#!/bin/bash
# Gemma-IT 50 trials full experiment
# Run 4 bet constraints sequentially on specified GPU
# Each run: 2 bet_types × 4 conditions × 50 reps = 400 games per constraint
# Total: 1,600 games

set -e
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN environment variable}"

GPU_ID=${1:-0}
SCRIPT_DIR="/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src"
OUTPUT_DIR="/home/jovyan/beomi/llm-addiction-data/investment_choice/gemma_50trials"
LOG_DIR="/home/jovyan/beomi/llm-addiction-data/investment_choice/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "$(date): Starting Gemma-IT 50-trial experiments on GPU $GPU_ID"
echo "Output: $OUTPUT_DIR"

for CONSTRAINT in 10 30 50 70; do
    echo ""
    echo "$(date): ===== Constraint $CONSTRAINT on GPU $GPU_ID ====="

    cd "$SCRIPT_DIR"
    python investment_choice/run_experiment.py \
        --model gemma \
        --gpu "$GPU_ID" \
        --constraint "$CONSTRAINT" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_DIR/gemma_50t_c${CONSTRAINT}_gpu${GPU_ID}.log"

    echo "$(date): Constraint $CONSTRAINT DONE"
done

echo ""
echo "$(date): ALL GEMMA 50-TRIAL EXPERIMENTS COMPLETED"
