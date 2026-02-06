#!/bin/bash
#SBATCH --job-name=completion_test
#SBATCH --partition=cas_v100_4
#SBATCH --output=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/completion_test_%j.log
#SBATCH --error=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/completion_test_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --comment pytorch

# Test completion-style prompts for both Blackjack and Lootbox
# Using LLaMA Base model with new prompt format

MODEL=${1:-llama}

echo "=========================================="
echo "Completion-Style Prompt Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Set working directory
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Create log directory
mkdir -p logs

echo ""
echo "=========================================="
echo "1. BLACKJACK TEST (Quick: 320 games)"
echo "   New: Completion-style prompts"
echo "=========================================="
python -m src.blackjack.run_experiment --model $MODEL --gpu 0 --quick

echo ""
echo "=========================================="
echo "2. LOOTBOX TEST (Quick: 320 games)"
echo "   New: Completion-style prompts"
echo "=========================================="
python -m src.lootbox.run_experiment --model $MODEL --gpu 0 --quick

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "End time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Blackjack output files:"
ls -la /scratch/x3415a02/data/llm-addiction/blackjack/ | tail -5

echo ""
echo "Lootbox output files:"
ls -la /scratch/x3415a02/data/llm-addiction/lootbox/ | tail -5
