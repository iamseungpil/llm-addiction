#!/bin/bash

# Phase 4: Word Analysis for All Phase 1 Patching Data
# Analyzes word patterns in responses for each GPU's patching data

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_words"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 4: Word Analysis Launch ==="

# Process each GPU's Phase 1 patching file
for gpu in 4 5 6 7; do
    PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu${gpu}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/word_analysis_gpu${gpu}.json"
    LOG_FILE="${LOG_DIR}/phase4_gpu${gpu}.log"

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

    python3 src/phase4_patching_word_analysis.py \
        --patching-file "$PATCHING_FILE" \
        --output "$OUTPUT_FILE" \
        > "$LOG_FILE" 2>&1 &

    echo "  Launched: GPU $gpu (PID: $!)"
done

# Wait for all background jobs
wait

echo ""
echo "✅ Phase 4 complete! All GPU word analyses finished"
echo "Results in: $OUTPUT_DIR"
