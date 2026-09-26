#!/bin/bash

# Phase 4: Word-Feature Correlation (REPARSED 2510 features)
# NOW USES ACTUAL BPE TOKENS (not regex!)
# Uses GPU 4, 5, 6, 7 outputs

PHASE1_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_REPARSED"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_feature_REPARSED"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 4 (REPARSED 2510 features): Word-Feature Correlation ==="
echo "Using ACTUAL BPE tokens from Phase 1 (generated_tokens field)"
echo "Processing GPU 4, 5, 6, 7 outputs"
echo ""

for gpu in 4 5 6 7; do
    PATCHING_FILE="${PHASE1_DIR}/phase1_patching_multifeature_gpu${gpu}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/word_feature_correlation_gpu${gpu}.json"
    LOG_FILE="${LOG_DIR}/phase4_REPARSED_gpu${gpu}.log"

    if [ ! -f "$PATCHING_FILE" ]; then
        echo "⚠️  Patching file not found for GPU $gpu: $PATCHING_FILE"
        continue
    fi

    # Check file size
    file_size=$(stat -c%s "$PATCHING_FILE" 2>/dev/null || echo "0")
    if [ "$file_size" -lt 100000000 ]; then
        echo "⚠️  Patching file too small for GPU $gpu: $(numfmt --to=iec $file_size)"
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

wait

echo ""
echo "✅ Phase 4 (REPARSED) complete! Word-feature correlations finished"
echo "Results in: $OUTPUT_DIR"
