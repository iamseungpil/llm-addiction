#!/bin/bash
# Launch Circuit Discovery Pipeline
# Usage: ./launch_circuit_pipeline.sh <model> <gpu>
# Example: ./launch_circuit_pipeline.sh llama 0
#          ./launch_circuit_pipeline.sh gemma 1

set -e

MODEL=${1:-llama}
GPU=${2:-0}

echo "=============================================="
echo "Circuit Discovery Pipeline"
echo "=============================================="
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "=============================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sae_env

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

# Navigate to src directory
cd "$(dirname "$0")/src"

# Run full pipeline
python run_circuit_pipeline.py \
    --model $MODEL \
    --gpu 0 \
    --node-threshold 0.1 \
    --top-k 50

echo ""
echo "Pipeline complete for $MODEL!"
echo "Results saved to: /data/llm_addiction/steering_vector_experiment/"
