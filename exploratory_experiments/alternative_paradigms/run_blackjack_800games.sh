#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack_800
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --comment=pytorch
# [SLURM-DISABLED] #SBATCH --time=24:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_800_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_800_%j.err

# Blackjack experiment with 800 games per model
# 2 bet types × 8 conditions × 50 reps = 800 games

# Initialize conda
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Usage: sbatch run_blackjack_800games.sh <model>"
    echo "  model: llama, gemma"
    exit 1
fi

echo "========================================="
echo "Blackjack 800 Games Experiment"
echo "========================================="
echo "Model: $MODEL"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================="

# Navigate to experiment directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Run experiment (modified to use 50 reps instead of 20)
python src/blackjack/run_experiment_800.py \
    --model $MODEL \
    --gpu 0

echo ""
echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "========================================="
