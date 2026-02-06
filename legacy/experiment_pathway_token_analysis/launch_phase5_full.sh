#!/bin/bash

# Phase 5: Prompt-Feature Correlation Analysis (Full 2787 features)
# Runs AFTER Phase 1 completes (uses Phase 1 data directly)

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase5_prompt_feature_full"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 5 (FULL 2787 features): Prompt-Feature Correlation Launch ==="
echo "Analyzing safe vs risky feature differences from Phase 1"
echo ""

# Process each GPU's Phase 1 patching file
for gpu in 5 6 7; do
    PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full/phase1_patching_multifeature_gpu${gpu}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/prompt_feature_correlation_gpu${gpu}.json"
    LOG_FILE="${LOG_DIR}/phase5_full_gpu${gpu}.log"

    if [ ! -f "$PATCHING_FILE" ]; then
        echo "⚠️  Patching file not found for GPU $gpu: $PATCHING_FILE"
        echo "   Phase 1 may not be complete yet"
        continue
    fi

    # Check file size
    file_size=$(stat -c%s "$PATCHING_FILE" 2>/dev/null || echo "0")
    if [ "$file_size" -lt 100000000 ]; then
        echo "⚠️  Patching file too small for GPU $gpu: $(numfmt --to=iec $file_size)"
        echo "   Phase 1 may still be running"
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
        --min-effect-size 0.3 \
        > "$LOG_FILE" 2>&1 &

    echo "  Launched: GPU $gpu (PID: $!)"
done

wait

echo ""
echo "✅ Phase 5 (FULL) complete! All GPU prompt-feature correlations finished"
echo "Results in: $OUTPUT_DIR"
