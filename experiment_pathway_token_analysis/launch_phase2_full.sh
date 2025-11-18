#!/bin/bash

# Phase 2: High Correlation Pair Discovery (Full 2787 features)
# Runs AFTER Phase 1 completes

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 2 (FULL 2787 features): Correlation Analysis Launch ==="
echo "Analyzing all Phase 1 patching results"
echo ""

# Process each GPU's Phase 1 output
for gpu in 5 6 7; do
    PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full/phase1_patching_multifeature_gpu${gpu}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/high_correlation_pairs_gpu${gpu}.json"
    LOG_FILE="${LOG_DIR}/phase2_full_gpu${gpu}.log"

    if [ ! -f "$PATCHING_FILE" ]; then
        echo "⚠️  Patching file not found for GPU $gpu: $PATCHING_FILE"
        echo "   Phase 1 may not be complete yet"
        continue
    fi

    # Check file size (should be >100MB if complete)
    file_size=$(stat -c%s "$PATCHING_FILE" 2>/dev/null || echo "0")
    if [ "$file_size" -lt 100000000 ]; then
        echo "⚠️  Patching file too small for GPU $gpu: $(numfmt --to=iec $file_size)"
        echo "   Phase 1 may still be running"
        continue
    fi

    echo "Processing GPU $gpu..."

    python3 src/phase2_high_correlation_pairs.py \
        --patching-file "$PATCHING_FILE" \
        --output "$OUTPUT_FILE" \
        --min-correlation 0.7 \
        > "$LOG_FILE" 2>&1 &

    echo "  Launched: GPU $gpu (PID: $!)"
done

wait

echo ""
echo "✅ Phase 2 (FULL) complete! All GPU correlation analyses finished"
echo "Results in: $OUTPUT_DIR"
