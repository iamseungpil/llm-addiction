#!/bin/bash
#SBATCH --job-name=dice-full
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/dice_full_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/dice_full_%j.err

# Conda initialization
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

echo "=========================================="
echo "Dice Rolling - Full Experiment"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""
echo "Experimental Design:"
echo "  Model: Gemma-2-9b-it"
echo "  Conditions: 4 (BASE-Fixed, BASE-Variable, GM-Fixed, GM-Variable)"
echo "  Games per condition: 50"
echo "  Total games: 200"
echo "  Expected time: ~100 minutes"
echo "=========================================="
echo ""

# Run all 4 conditions
echo "[1/4] Running BASE-Fixed condition..."
python src/dice_rolling/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --components BASE \
  --bet-type fixed \
  --bet-constraint 10 \
  --num-games 50 \
  --seed 42

echo ""
echo "[2/4] Running BASE-Variable condition..."
python src/dice_rolling/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --components BASE \
  --bet-type variable \
  --bet-constraint 50 \
  --num-games 50 \
  --seed 43

echo ""
echo "[3/4] Running GM-Fixed condition..."
python src/dice_rolling/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --components GM \
  --bet-type fixed \
  --bet-constraint 10 \
  --num-games 50 \
  --seed 44

echo ""
echo "[4/4] Running GM-Variable condition..."
python src/dice_rolling/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --components GM \
  --bet-type variable \
  --bet-constraint 50 \
  --num-games 50 \
  --seed 45

echo ""
echo "=========================================="
echo "All conditions completed at $(date)"
echo "=========================================="
echo ""
echo "Results saved to: /scratch/x3415a02/data/llm-addiction/dice_rolling/"
echo ""
echo "Summary:"
ls -lh /scratch/x3415a02/data/llm-addiction/dice_rolling/dice_gemma_*.json | tail -4
