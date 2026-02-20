#!/bin/bash
#SBATCH --job-name=blackjack_gemma_var
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --comment=python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_gemma_var_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_gemma_var_%j.err

# Conda initialization (required on HPC cluster)
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to source directory
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/src

# Run Gemma Blackjack experiment with variable betting
# Quick mode: 8 components × 20 reps = 160 games (same as LLaMA)
python blackjack/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --bet-type variable \
  --quick

echo "Gemma variable betting experiment completed!"
