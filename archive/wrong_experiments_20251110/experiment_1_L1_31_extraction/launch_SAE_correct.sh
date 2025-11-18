#!/bin/bash
# Launch SAE Feature Extraction with correct conda environment

GPU_ID=${1:-5}

echo "=================================="
echo "SAE Feature Extraction L1-31"
echo "=================================="
echo "GPU: $GPU_ID"
echo "Conda env: llama_sae_env"
echo "Started: $(date)"
echo "=================================="

# Activate conda
source /data/miniforge3/etc/profile.d/conda.sh
conda activate llama_sae_env

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create log directory
mkdir -p /home/ubuntu/llm_addiction/experiment_1_L1_31_extraction/logs

# Run extraction
cd /home/ubuntu/llm_addiction/experiment_1_L1_31_extraction

python -u extract_L1_31_SAE_CORRECTED.py --gpu 0 2>&1 | tee logs/sae_extraction_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=================================="
echo "Completed: $(date)"
echo "=================================="
