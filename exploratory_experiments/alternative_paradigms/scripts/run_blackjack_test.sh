#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack_test
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/blackjack_test_%j.log
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/blackjack_test_%j.err
# [SLURM-DISABLED] #SBATCH --time=02:00:00
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --comment pytorch

# Blackjack experiment test with simplified rules (like Slot Machine)
# Changes: $100 initial, $5-$100 bets, Hit/Stand only (no Double)

MODEL=${1:-llama}

echo "=========================================="
echo "Blackjack Test - Simplified Rules"
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
echo "BLACKJACK (Quick: 320 games)"
echo "Settings: $100 initial, $5-$100 bets, Hit/Stand only"
echo "=========================================="
python -m src.blackjack.run_experiment --model $MODEL --gpu 0 --quick

echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETED"
echo "End time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
ls -la /home/jovyan/beomi/llm-addiction-data/blackjack/
