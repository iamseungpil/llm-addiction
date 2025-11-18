#!/bin/bash

# Phase 3: Causal Direction Analysis (Full 2787 features)
# Runs AFTER Phase 2 completes

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase3_causal_full"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 3 (FULL 2787 features): Causal Direction Analysis Launch ==="
echo "Analyzing correlation pairs from Phase 2"
echo ""

# Process each GPU's Phase 2 output
for gpu in 5 6 7; do
    CORRELATION_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations/high_correlation_pairs_gpu${gpu}.json"
    PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full/phase1_patching_multifeature_gpu${gpu}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/causal_directions_gpu${gpu}.json"
    LOG_FILE="${LOG_DIR}/phase3_full_gpu${gpu}.log"

    if [ ! -f "$CORRELATION_FILE" ]; then
        echo "⚠️  Correlation file not found for GPU $gpu: $CORRELATION_FILE"
        echo "   Phase 2 may not be complete yet"
        continue
    fi

    if [ ! -f "$PATCHING_FILE" ]; then
        echo "⚠️  Patching file not found for GPU $gpu: $PATCHING_FILE"
        continue
    fi

    echo "Processing GPU $gpu..."

    python3 src/phase3_causal_validation.py \
        --correlation-file "$CORRELATION_FILE" \
        --patching-file "$PATCHING_FILE" \
        --output "$OUTPUT_FILE" \
        > "$LOG_FILE" 2>&1 &

    echo "  Launched: GPU $gpu (PID: $!)"
done

wait

echo ""
echo "✅ Phase 3 (FULL) complete! All GPU causal analyses finished"
echo "Results in: $OUTPUT_DIR"
