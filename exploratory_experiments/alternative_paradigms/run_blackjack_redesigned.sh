#!/bin/bash
#SBATCH --job-name=blackjack_redesigned
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --comment=pytorch
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_redesigned_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_redesigned_%j.err

# Blackjack experiment with REDESIGNED bet structure
# Fixed betting: Always $10, $30, $50, $70
# Variable betting: $1-10, $1-30, $1-50, $1-70
# Total: 8 bet configs × 8 components × 50 reps = 3,200 games

# Initialize conda
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Usage: sbatch run_blackjack_redesigned.sh <model>"
    echo "  model: llama, gemma, qwen"
    exit 1
fi

echo "========================================="
echo "Blackjack REDESIGNED Experiment"
echo "========================================="
echo "Model: $MODEL"
echo "Initial chips: \$200"
echo "Bet configs: 8 (4 fixed + 4 variable)"
echo "  Fixed: \$10, \$30, \$50, \$70"
echo "  Variable: \$1-10, \$1-30, \$1-50, \$1-70"
echo "Components: 8 (BASE, G, M, GM, H, W, P, GMHWP)"
echo "Total games: 3,200"
echo "Start time: $(date)"
echo "========================================="

# Navigate to experiment directory
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Run experiment
python src/blackjack/run_experiment_redesigned.py \
    --model $MODEL \
    --gpu 0

echo ""
echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "========================================="
