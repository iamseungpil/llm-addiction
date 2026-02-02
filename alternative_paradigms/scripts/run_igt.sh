#!/bin/bash
# Run Iowa Gambling Task experiment
#
# Usage:
#   bash scripts/run_igt.sh llama 0       # Full experiment
#   bash scripts/run_igt.sh gemma 0 --quick  # Quick test

MODEL=$1
GPU=$2
EXTRA_ARGS="${@:3}"

if [ -z "$MODEL" ] || [ -z "$GPU" ]; then
    echo "Usage: bash scripts/run_igt.sh <model> <gpu> [--quick]"
    echo "  model: llama, gemma, or qwen"
    echo "  gpu: GPU ID (0, 1, etc.)"
    echo "  --quick: Optional quick mode (80 games instead of 1,600)"
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_sae_env

# Run experiment
cd /mnt/c/Users/oollccddss/git/llm-addiction/alternative_paradigms

echo "=========================================="
echo "Iowa Gambling Task Experiment"
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="

python src/igt/run_experiment.py --model $MODEL --gpu 0 $EXTRA_ARGS

echo "Experiment completed!"
