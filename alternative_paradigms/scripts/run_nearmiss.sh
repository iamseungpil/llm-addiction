#!/bin/bash
# Run Near-Miss Enhancement experiment
#
# Usage:
#   bash scripts/run_nearmiss.sh llama 0 variable       # Full experiment, variable betting
#   bash scripts/run_nearmiss.sh gemma 0 fixed --quick  # Quick test, fixed betting

MODEL=$1
GPU=$2
BET_TYPE=$3
EXTRA_ARGS="${@:4}"

if [ -z "$MODEL" ] || [ -z "$GPU" ] || [ -z "$BET_TYPE" ]; then
    echo "Usage: bash scripts/run_nearmiss.sh <model> <gpu> <bet_type> [--quick]"
    echo "  model: llama, gemma, or qwen"
    echo "  gpu: GPU ID (0, 1, etc.)"
    echo "  bet_type: fixed or variable"
    echo "  --quick: Optional quick mode (80 games instead of 3,200)"
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
echo "Near-Miss Enhancement Experiment"
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "Bet Type: $BET_TYPE"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="

python src/nearmiss/run_experiment.py --model $MODEL --gpu 0 --bet-type $BET_TYPE $EXTRA_ARGS

echo "Experiment completed!"
