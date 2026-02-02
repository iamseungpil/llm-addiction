#!/bin/bash
# Run Loot Box Mechanics experiment
#
# Usage:
#   bash scripts/run_lootbox.sh llama 0       # Full experiment
#   bash scripts/run_lootbox.sh gemma 0 --quick  # Quick test

MODEL=$1
GPU=$2
EXTRA_ARGS="${@:3}"

if [ -z "$MODEL" ] || [ -z "$GPU" ]; then
    echo "Usage: bash scripts/run_lootbox.sh <model> <gpu> [--quick]"
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
echo "Loot Box Mechanics Experiment"
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="

python src/lootbox/run_experiment.py --model $MODEL --gpu 0 $EXTRA_ARGS

echo "Experiment completed!"
