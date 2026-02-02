#!/bin/bash

# Full Pipeline: Phase 1 + Phase 2
# Usage: bash scripts/run_full_pipeline.sh [model] [gpu_id]
#   model: gemma or llama
#   gpu_id: GPU device ID (default: 0)

set -e

# Parse arguments
MODEL=${1:-gemma}
GPU_ID=${2:-0}

echo "======================================"
echo "Investment Choice SAE Analysis"
echo "Full Pipeline: Phase 1 + Phase 2"
echo "======================================"
echo "Model: $MODEL"
echo "GPU: $GPU_ID"
echo "======================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run Phase 1
echo ""
echo "STEP 1/2: Feature Extraction"
echo ""
bash "$SCRIPT_DIR/run_phase1.sh" "$MODEL" "$GPU_ID"

# Run Phase 2
echo ""
echo "STEP 2/2: Correlation Analysis"
echo ""
bash "$SCRIPT_DIR/run_phase2.sh" "$MODEL"

echo ""
echo "======================================"
echo "Full pipeline complete!"
echo "======================================"
echo ""
echo "Results saved to:"
echo "  Features: results/features/"
echo "  Correlations: results/correlations/"
echo ""
