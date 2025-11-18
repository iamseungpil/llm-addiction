#!/bin/bash

# Phase 4 (Redesigned): Output Word-Feature Correlation Analysis
# Analyzes which features are activated when specific words appear

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_feature"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 4 (NEW): Output Word-Feature Correlation Analysis Launch ==="

# Process each GPU's Phase 1 patching file
for gpu in 4 5 6 7; do
    PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu${gpu}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/word_feature_correlation_gpu${gpu}.json"
    LOG_FILE="${LOG_DIR}/phase4_new_gpu${gpu}.log"

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

    python3 src/phase4_word_feature_correlation.py \
        --patching-file "$PATCHING_FILE" \
        --output "$OUTPUT_FILE" \
        --min-word-count 1 \
        > "$LOG_FILE" 2>&1 &

    echo "  Launched: GPU $gpu (PID: $!)"
done

# Wait for all background jobs
wait

echo ""
echo "✅ Phase 4 (NEW) complete! All GPU word-feature correlations finished"
echo "Results in: $OUTPUT_DIR"
