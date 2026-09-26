#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack_gemma_var
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=8
# [SLURM-DISABLED] #SBATCH --mem=40G
# [SLURM-DISABLED] #SBATCH --time=04:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_gemma_var_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_gemma_var_%j.err

# Conda initialization (required on HPC cluster)
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to source directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src

# Run Gemma Blackjack experiment with variable betting
# Quick mode: 8 components × 20 reps = 160 games (same as LLaMA)
python blackjack/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --bet-type variable \
  --quick

echo "Gemma variable betting experiment completed!"
