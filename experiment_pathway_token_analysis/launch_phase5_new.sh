#!/bin/bash

# Phase 5 (Redesigned): Input Prompt-Feature Correlation Analysis
# Analyzes which features are differentially activated by prompt type

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase5_prompt_feature"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 5 (NEW): Input Prompt-Feature Correlation Analysis Launch ==="

# Process each GPU's Phase 1 patching file
for gpu in 4 5 6 7; do
    PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu${gpu}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/prompt_feature_correlation_gpu${gpu}.json"
    LOG_FILE="${LOG_DIR}/phase5_new_gpu${gpu}.log"

    if [ ! -f "$PATCHING_FILE" ]; then
        echo "⚠️  Patching file not found for GPU $gpu: $PATCHING_FILE"
        continue
    fi

    # Skip if already processed
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✅ Already processed: GPU $gpu"
        continue
    fi

    echo "Processing GPU $gpu..."

    python3 src/phase5_prompt_feature_correlation.py \
        --patching-file "$PATCHING_FILE" \
        --output "$OUTPUT_FILE" \
        > "$LOG_FILE" 2>&1 &

    echo "  Launched: GPU $gpu (PID: $!)"
done

# Wait for all background jobs
wait

echo ""
echo "✅ Phase 5 (NEW) complete! All GPU prompt-feature correlations finished"
echo "Results in: $OUTPUT_DIR"
