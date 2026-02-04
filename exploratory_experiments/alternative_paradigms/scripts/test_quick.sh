#!/bin/bash
#SBATCH --job-name=alt_paradigm_test
#SBATCH --partition=cas_v100_4
#SBATCH --output=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/test_%j.log
#SBATCH --error=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/test_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Quick test script for alternative paradigms experiments
# Usage: sbatch scripts/test_quick.sh [model] [experiment]
#   model: llama, gemma, qwen (default: gemma)
#   experiment: lootbox, blackjack, investment, all (default: all)

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
mkdir -p /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs

# Activate conda environment
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Set working directory
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Run quick test
python test_quick.py --model $MODEL --experiment $EXPERIMENT --gpu 0

echo "=========================================="
echo "Test completed at: $(date)"
echo "=========================================="
