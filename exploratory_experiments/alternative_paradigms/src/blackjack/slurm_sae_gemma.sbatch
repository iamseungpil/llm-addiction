#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack_sae_gemma
# [SLURM-DISABLED] #SBATCH --partition=amd_a100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=8
# [SLURM-DISABLED] #SBATCH --mem=80G
# [SLURM-DISABLED] #SBATCH --time=08:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_sae_gemma_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_sae_gemma_%j.err

# Conda initialization
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src

# Find the most recent Gemma blackjack results (fixed betting)
GEMMA_RESULTS=$(ls -t /home/jovyan/beomi/llm-addiction-data/blackjack/blackjack_gemma_*.json | head -1)

echo "Processing Gemma results: $GEMMA_RESULTS"

# Run Phase 1: SAE feature extraction for Gemma
# Focus on layers 26-40 (mid-to-late layers where behavior is likely encoded)
# Using fewer layers to save time (GemmaScope is large)
python blackjack/phase1_feature_extraction.py \
  --model gemma \
  --gpu 0 \
  --input "$GEMMA_RESULTS" \
  --layers 26,28,30,32,35,38,40 \
  --output-dir /home/jovyan/beomi/llm-addiction-data/blackjack/sae_features

echo "Gemma SAE feature extraction completed!"
