#!/bin/bash
# Launch L1-31 GLOBAL FDR Feature Extraction
# Uses CORRECT global FDR correction across ALL layers

set -e

GPU_ID=${1:-5}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/ubuntu/llm_addiction/experiment_L1_31_CORRECT/logs"
SCRIPT_DIR="/home/ubuntu/llm_addiction/experiment_L1_31_CORRECT"

mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/L1_31_GLOBAL_FDR_gpu${GPU_ID}_${TIMESTAMP}.log"

echo "=================================="
echo "L1-31 GLOBAL FDR Feature Extraction"
echo "=================================="
echo "GPU: $GPU_ID"
echo "Log: $LOG_FILE"
echo "Script: $SCRIPT_DIR/extract_L1_31_GLOBAL_FDR.py"
echo ""
echo "Starting extraction..."

cd "$SCRIPT_DIR"

python3 extract_L1_31_GLOBAL_FDR.py --gpu "$GPU_ID" 2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Extraction complete!"
echo "Check results in: /data/llm_addiction/results/L1_31_GLOBAL_FDR_*"
