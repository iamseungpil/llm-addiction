#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=alt_paradigm_test
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/test_%j.log
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/test_%j.err
# [SLURM-DISABLED] #SBATCH --time=00:30:00
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --comment pytorch

# Quick test script for alternative paradigms experiments
# Usage: sbatch scripts/test_quick.sh [model] [experiment]
#   model: llama, gemma, qwen (default: gemma)
#   experiment: blackjack, investment, all (default: all)

MODEL=${1:-gemma}
EXPERIMENT=${2:-all}

echo "=========================================="
echo "Alternative Paradigms Quick Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Experiment: $EXPERIMENT"
echo "Start time: $(date)"
echo "=========================================="

# Create log directory
mkdir -p /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs

# Activate conda environment
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

# HuggingFace token for gated models (Gemma, LLaMA)
# Set HF_TOKEN in ~/.bashrc or pass as environment variable
# export HF_TOKEN="your_token_here"

# Set working directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Run quick test
python test_quick.py --model $MODEL --experiment $EXPERIMENT --gpu 0

echo "=========================================="
echo "Test completed at: $(date)"
echo "=========================================="
