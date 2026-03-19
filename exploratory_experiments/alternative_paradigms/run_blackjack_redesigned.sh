#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack_redesigned
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --comment=pytorch
# [SLURM-DISABLED] #SBATCH --time=48:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_redesigned_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_redesigned_%j.err

# Blackjack experiment with REDESIGNED bet structure
# Fixed betting: Always $10, $30, $50, $70
# Variable betting: $1-10, $1-30, $1-50, $1-70
# Total: 8 bet configs × 8 components × 50 reps = 3,200 games

# Initialize conda
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

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
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Run experiment
python src/blackjack/run_experiment_redesigned.py \
    --model $MODEL \
    --gpu 0

echo ""
echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "========================================="
