#!/bin/bash
# Run complete analysis pipeline for LLaMA
# Usage: bash run_llama_pipeline.sh [GPU_ID]

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
CONFIG="$SCRIPT_DIR/../configs/analysis_config.yaml"

echo "========================================"
echo "LLaMA SAE Analysis Pipeline"
echo "========================================"
echo "GPU: $GPU_ID"
echo "Config: $CONFIG"
echo "========================================"

# Phase 1: Feature Extraction (longest phase)
echo ""
echo "[Phase 1] SAE Feature Extraction..."
echo "Estimated time: ~2-3 hours for all 31 layers"
echo ""

python3 "$SRC_DIR/phase1_feature_extraction.py" \
    --model llama \
    --config "$CONFIG" \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "Phase 1 failed!"
    exit 1
fi

# Phase 2: Correlation Analysis
echo ""
echo "[Phase 2] Correlation Analysis + FDR..."
echo ""

python3 "$SRC_DIR/phase2_correlation_analysis.py" \
    --model llama \
    --config "$CONFIG"

if [ $? -ne 0 ]; then
    echo "Phase 2 failed!"
    exit 1
fi

# Phase 3: Semantic Analysis
echo ""
echo "[Phase 3] Semantic Analysis..."
echo ""

python3 "$SRC_DIR/phase3_semantic_analysis.py" \
    --model llama \
    --config "$CONFIG" \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "Phase 3 failed!"
    exit 1
fi

# Phase 4: Causal Pilot (optional, exploratory)
echo ""
echo "[Phase 4] Causal Pilot (Exploratory)..."
echo ""

python3 "$SRC_DIR/phase4_causal_pilot.py" \
    --model llama \
    --config "$CONFIG" \
    --device cuda:0

echo ""
echo "========================================"
echo "LLaMA Pipeline Complete!"
echo "========================================"
echo "Results saved to: /data/llm_addiction/experiment_corrected_sae_analysis/llama/"
