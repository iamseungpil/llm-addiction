#!/bin/bash

# Phase 2: Correlation Analysis for GPU 4
# 50 features × 4 conditions = 200 analyses

PATCHING_FILE="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu4.jsonl"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Extract unique target features from GPU 4's Phase 1 output
python3 << 'PYTHON_EOF' > /tmp/phase2_gpu4_features.txt
import json

filepath = "/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu4.jsonl"
target_features = set()

with open(filepath, 'r') as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            target_features.add(record['target_feature'])

for feat in sorted(target_features):
    print(feat)
PYTHON_EOF

# Run Phase 2 for each (target_feature, condition) combination
CONDITIONS=("safe_mean_safe" "safe_mean_risky" "risky_mean_safe" "risky_mean_risky")

while IFS= read -r FEATURE; do
    for CONDITION in "${CONDITIONS[@]}"; do
        OUTPUT_FILE="${OUTPUT_DIR}/correlations_${FEATURE}_${CONDITION}.jsonl"
        LOG_FILE="${LOG_DIR}/phase2_${FEATURE}_${CONDITION}.log"

        echo "Processing: $FEATURE / $CONDITION"

        python3 src/phase2_patching_correlations.py \
            --patching-file "$PATCHING_FILE" \
            --target-feature "$FEATURE" \
            --condition "$CONDITION" \
            --output "$OUTPUT_FILE" \
            > "$LOG_FILE" 2>&1 &

        # Limit concurrent processes to avoid overload
        while [ $(jobs -r | wc -l) -ge 10 ]; do
            sleep 1
        done
    done
done < /tmp/phase2_gpu4_features.txt

# Wait for all background jobs
wait

echo "GPU 4 Phase 2 complete!"
