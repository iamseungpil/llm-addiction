#!/bin/bash
#SBATCH --job-name=dice-v2
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/dice_pilot_v2_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/dice_pilot_v2_%j.err

# Conda initialization
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

echo "=========================================="
echo "Dice Rolling Pilot Test V2 (Improved)"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Changes:"
echo "  - Neutral language (Number Selection Game)"
echo "  - Concise prompt (no gambling terms)"
echo "  - Robust parsing (3-stage fallback)"
echo "  - Optimized generation (100 tokens, temp=0.5)"
echo "=========================================="
echo ""

# Run test (3 games for quick validation)
python src/dice_rolling/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --components BASE \
  --bet-type variable \
  --num-games 3 \
  --seed 42

echo ""
echo "=========================================="
echo "Test completed at $(date)"
echo "=========================================="
