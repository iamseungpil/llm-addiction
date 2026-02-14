#!/bin/bash
#SBATCH --job-name=blackjack_800
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --comment=pytorch
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_800_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_800_%j.err

# Blackjack experiment with 800 games per model
# 2 bet types × 8 conditions × 50 reps = 800 games

# Initialize conda
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

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
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Run experiment (modified to use 50 reps instead of 20)
python src/blackjack/run_experiment_800.py \
    --model $MODEL \
    --gpu 0

echo ""
echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "========================================="
