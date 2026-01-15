#!/bin/bash
# =============================================================================
# Full Layer Causal Analysis Pipeline
# =============================================================================
#
# Phases:
#   1. Steering Vector Extraction (all 32/42 layers)
#   2. SAE Feature Projection (all layers)
#   3. Soft Interpolation Patching (8 conditions, real prompts)
#   5. Gambling-Context Interpretation
#
# Phase 4 (Head Patching) is SKIPPED
#
# GPU Assignment:
#   GPU 4: LLaMA
#   GPU 5: Gemma
#
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG="${SCRIPT_DIR}/configs/experiment_config_full_layers.yaml"
OUTPUT_DIR="/data/llm_addiction/steering_vector_experiment_full"
LOG_DIR="${OUTPUT_DIR}/logs"
CONDA_ENV="/data/miniforge3/envs/llama_sae_env/bin/python"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " FULL LAYER CAUSAL ANALYSIS PIPELINE"
echo "============================================================"
echo " Config: ${CONFIG}"
echo " Output: ${OUTPUT_DIR}"
echo " Timestamp: ${TIMESTAMP}"
echo " Phases: 1, 2, 3, 5 (Phase 4 skipped)"
echo "============================================================"

# Step 0: Extract 8-condition prompts
echo ""
echo "[Step 0] Extracting 8-condition prompts from real experimental data..."
${CONDA_ENV} "${SRC_DIR}/real_prompt_extractor.py" \
    --output "${OUTPUT_DIR}/condition_prompts_${TIMESTAMP}.json" \
    2>&1 | tee "${LOG_DIR}/prompt_extraction_${TIMESTAMP}.log"

echo ""
echo "============================================================"
echo " Starting parallel model analysis"
echo "============================================================"

# Run LLaMA on GPU 4
echo ""
echo "[LLaMA] Starting on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup ${CONDA_ENV} "${SRC_DIR}/run_full_pipeline.py" \
    --model llama \
    --gpu 4 \
    --config "${CONFIG}" \
    --phases 1,2,3,5 \
    > "${LOG_DIR}/llama_full_${TIMESTAMP}.log" 2>&1 &
LLAMA_PID=$!
echo "  PID: ${LLAMA_PID}"
echo "  Log: ${LOG_DIR}/llama_full_${TIMESTAMP}.log"

# Run Gemma on GPU 5
echo ""
echo "[Gemma] Starting on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup ${CONDA_ENV} "${SRC_DIR}/run_full_pipeline.py" \
    --model gemma \
    --gpu 5 \
    --config "${CONFIG}" \
    --phases 1,2,3,5 \
    > "${LOG_DIR}/gemma_full_${TIMESTAMP}.log" 2>&1 &
GEMMA_PID=$!
echo "  PID: ${GEMMA_PID}"
echo "  Log: ${LOG_DIR}/gemma_full_${TIMESTAMP}.log"

echo ""
echo "============================================================"
echo " Experiments running in background"
echo "============================================================"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_DIR}/llama_full_${TIMESTAMP}.log"
echo "  tail -f ${LOG_DIR}/gemma_full_${TIMESTAMP}.log"
echo ""
echo "To check status:"
echo "  ps aux | grep run_full_pipeline"
echo ""
echo "Expected duration: ~8 hours"
echo "============================================================"
