#!/bin/bash

# Phase 3: Causal Direction Validation (REPARSED 2510 features)
# Runs AFTER Phase 2 REPARSED completes
# Uses GPU 4, 5, 6, 7 outputs

PHASE1_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_REPARSED"
PHASE2_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations_REPARSED"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase3_causal_REPARSED"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 3 (REPARSED 2510 features): Causal Direction Validation ==="
echo "Validating causal directions from Phase 2 correlations"
echo "Processing GPU 4, 5, 6, 7 outputs"
echo ""

for gpu in 4 5 6 7; do
    CORRELATION_FILE="${PHASE2_DIR}/high_correlation_pairs_gpu${gpu}.json"
    PATCHING_FILE="${PHASE1_DIR}/phase1_patching_multifeature_gpu${gpu}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/causal_directions_gpu${gpu}.json"
    LOG_FILE="${LOG_DIR}/phase3_REPARSED_gpu${gpu}.log"

    if [ ! -f "$CORRELATION_FILE" ]; then
        echo "⚠️  Correlation file not found for GPU $gpu: $CORRELATION_FILE"
        continue
    fi

    if [ ! -f "$PATCHING_FILE" ]; then
        echo "⚠️  Patching file not found for GPU $gpu: $PATCHING_FILE"
        continue
    fi

    echo "Processing GPU $gpu..."

    python3 src/phase3_patching_causal_validation.py \
        --correlation-file "$CORRELATION_FILE" \
        --patching-file "$PATCHING_FILE" \
        --output "$OUTPUT_FILE" \
        > "$LOG_FILE" 2>&1 &

    echo "  Launched: GPU $gpu (PID: $!)"
done

wait

echo ""
echo "✅ Phase 3 (REPARSED) complete! All causal direction validations finished"
echo "Results in: $OUTPUT_DIR"
