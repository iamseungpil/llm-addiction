#!/bin/bash
#SBATCH --job-name=dice-pilot
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/dice_pilot_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/dice_pilot_%j.err

# REQUIRED: Conda initialization on HPC cluster
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to repository
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Print environment info
echo "=========================================="
echo "Dice Rolling Pilot Test"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# Run pilot test (5 games, BASE condition, variable betting)
python src/dice_rolling/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --components BASE \
  --bet-type variable \
  --num-games 5 \
  --seed 42

echo ""
echo "=========================================="
echo "Pilot test completed at $(date)"
echo "=========================================="
