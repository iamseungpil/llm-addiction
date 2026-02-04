#!/bin/bash
#SBATCH --job-name=mini_exp_test
#SBATCH --partition=cas_v100_4
#SBATCH --output=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/mini_%j.log
#SBATCH --error=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/mini_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --comment pytorch

# Mini experiment test - full pipeline verification
# Usage: sbatch scripts/test_mini.sh [experiment] [model] [n_games]
#   experiment: investment, lootbox, blackjack, all (default: all)
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
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# HuggingFace token (set in ~/.bashrc or pass as environment variable)
# export HF_TOKEN="your_token_here"

# Set working directory
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Run mini experiment
python test_mini_experiment.py --experiment $EXPERIMENT --model $MODEL --n-games $N_GAMES --gpu 0

echo "=========================================="
echo "Completed at: $(date)"
echo "=========================================="
