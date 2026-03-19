#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=mini_exp_test
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/mini_%j.log
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/mini_%j.err
# [SLURM-DISABLED] #SBATCH --time=00:30:00
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --comment pytorch

# Mini experiment test - full pipeline verification
# Usage: sbatch scripts/test_mini.sh [experiment] [model] [n_games]
#   experiment: investment, blackjack, all (default: all)
#   model: gemma, llama, qwen (default: gemma)
#   n_games: games per condition (default: 2)

EXPERIMENT=${1:-all}
MODEL=${2:-gemma}
N_GAMES=${3:-2}

echo "=========================================="
echo "Mini Experiment Pipeline Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Experiment: $EXPERIMENT"
echo "Model: $MODEL"
echo "Games per condition: $N_GAMES"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

# HuggingFace token (set in ~/.bashrc or pass as environment variable)
# export HF_TOKEN="your_token_here"

# Set working directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Run mini experiment
python test_mini_experiment.py --experiment $EXPERIMENT --model $MODEL --n-games $N_GAMES --gpu 0

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
