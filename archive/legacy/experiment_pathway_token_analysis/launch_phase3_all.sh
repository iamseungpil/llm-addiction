#!/bin/bash

# Phase 3: Causal Validation for All Correlation Files
# Processes each Phase 2 correlation file with its corresponding Phase 1 patching data

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase3_causal"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"
CORR_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Map target features to GPU files (based on Phase 1 execution)
# GPU 4: features 0-49, GPU 5: features 50-99, GPU 6: features 100-149, GPU 7: features 150-199

# Create GPU mapping for each target feature
declare -A FEATURE_TO_GPU

# Extract feature-to-GPU mapping from Phase 1 files
for gpu in 4 5 6 7; do
    PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu${gpu}.jsonl"

    if [ -f "$PATCHING_FILE" ]; then
        # Extract unique target features from this GPU's file
        python3 << PYTHON_EOF
import json
filepath = "$PATCHING_FILE"
target_features = set()
with open(filepath, 'r') as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            target_features.add(record['target_feature'])

for feat in target_features:
    print(f"{feat}\t${gpu}")
PYTHON_EOF
    fi
done > /tmp/phase3_feature_gpu_mapping.txt

# Process each correlation file
CORR_FILES=("$CORR_DIR"/*.jsonl)
TOTAL_FILES=${#CORR_FILES[@]}
echo "Processing $TOTAL_FILES correlation files..."

PROCESSED=0
for CORR_FILE in "${CORR_FILES[@]}"; do
    BASENAME=$(basename "$CORR_FILE")

    # Extract target_feature and condition from filename
    # Format: correlations_<feature>_<condition>.jsonl
    # Example: correlations_L1-1272_safe_mean_safe.jsonl
    FEATURE=$(echo "$BASENAME" | sed -E 's/correlations_(.*)_(safe_mean_safe|safe_mean_risky|risky_mean_safe|risky_mean_risky)\.jsonl/\1/')
    CONDITION=$(echo "$BASENAME" | sed -E 's/correlations_.*_(safe_mean_safe|safe_mean_risky|risky_mean_safe|risky_mean_risky)\.jsonl/\1/')

    # Find which GPU has this feature
    GPU=$(grep "^${FEATURE}\s" /tmp/phase3_feature_gpu_mapping.txt | awk '{print $2}')

    if [ -z "$GPU" ]; then
        echo "⚠️  Could not find GPU for feature $FEATURE, skipping..."
        continue
    fi

    PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu${GPU}.jsonl"
    OUTPUT_FILE="${OUTPUT_DIR}/causal_${FEATURE}_${CONDITION}.jsonl"
    LOG_FILE="${LOG_DIR}/phase3_${FEATURE}_${CONDITION}.log"

    # Skip if already processed
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✅ Already processed: ${FEATURE}_${CONDITION}"
        ((PROCESSED++))
        continue
    fi

    echo "Processing [$PROCESSED/$TOTAL_FILES]: ${FEATURE}_${CONDITION} (GPU $GPU)"

    python3 src/phase3_patching_causal_validation.py \
        --correlation-file "$CORR_FILE" \
        --patching-file "$PATCHING_FILE" \
        --output "$OUTPUT_FILE" \
        > "$LOG_FILE" 2>&1 &

    # Limit concurrent processes to avoid overload
    while [ $(jobs -r | wc -l) -ge 20 ]; do
        sleep 1
    done

    ((PROCESSED++))
done

# Wait for all background jobs
wait

echo "✅ Phase 3 complete! Processed $TOTAL_FILES correlation files"
echo "Results in: $OUTPUT_DIR"
