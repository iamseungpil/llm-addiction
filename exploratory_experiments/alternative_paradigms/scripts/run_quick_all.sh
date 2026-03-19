#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=quick_all_exp
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/quick_all_%j.log
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/quick_all_%j.err
# [SLURM-DISABLED] #SBATCH --time=04:00:00
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --comment pytorch

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
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

# Set working directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

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
ls -la /home/jovyan/beomi/llm-addiction-data/*/
