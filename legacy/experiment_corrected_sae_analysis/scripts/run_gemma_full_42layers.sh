#!/bin/bash
# Run Gemma full 42-layer analysis with llama_sae_env
# Usage: bash run_gemma_full_42layers.sh [GPU_ID]

GPU_ID=${1:-6}
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
CONFIG="$SCRIPT_DIR/../configs/analysis_config.yaml"

# Use llama_sae_env conda environment
CONDA_ENV="llama_sae_env"

echo "========================================"
echo "Gemma FULL 42-Layer SAE Analysis"
echo "========================================"
echo "GPU: $GPU_ID"
echo "Conda env: $CONDA_ENV"
echo "Config: $CONFIG"
echo "Width: 131K features per layer"
echo "Total: 42 layers × 131K = 5.5M features"
echo "Estimated time: 10-15 hours"
echo "========================================"

# Activate conda environment and run
source /data/miniforge3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Phase 1: Feature Extraction (longest phase)
echo ""
echo "[Phase 1] SAE Feature Extraction..."
echo "Processing 42 layers with 131K features each"
echo ""

python3 "$SRC_DIR/phase1_feature_extraction.py" \
    --model gemma \
    --config "$CONFIG" \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "Phase 1 failed!"
    exit 1
fi

echo "Phase 1 completed successfully!"
echo "Results saved to: /data/llm_addiction/experiment_corrected_sae_analysis/gemma_full_42layers/"
