#!/bin/bash
#SBATCH --job-name=quick_all_exp
#SBATCH --partition=cas_v100_4
#SBATCH --output=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/quick_all_%j.log
#SBATCH --error=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/quick_all_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --comment pytorch

# Quick mode experiment - all experiments
# Total games: 160 + 320 = 480 games

MODEL=${1:-gemma}

echo "=========================================="
echo "Quick Mode - All Experiments"
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
echo "1. INVESTMENT CHOICE (Quick: 160 games)"
echo "=========================================="
python -m src.investment_choice.run_experiment --model $MODEL --gpu 0 --quick

echo ""
echo "=========================================="
echo "2. BLACKJACK (Quick: 320 games)"
echo "=========================================="
python -m src.blackjack.run_experiment --model $MODEL --gpu 0 --quick

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "End time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
ls -la /scratch/x3415a02/data/llm-addiction/*/
