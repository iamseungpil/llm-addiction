#!/bin/bash
#SBATCH --job-name=blackjack_sae_gemma
#SBATCH --partition=amd_a100_4
#SBATCH --gres=gpu:1
#SBATCH --comment=python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_sae_gemma_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_sae_gemma_%j.err

# Conda initialization
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/src

# Find the most recent Gemma blackjack results (fixed betting)
GEMMA_RESULTS=$(ls -t /scratch/x3415a02/data/llm-addiction/blackjack/blackjack_gemma_*.json | head -1)

echo "Processing Gemma results: $GEMMA_RESULTS"

# Run Phase 1: SAE feature extraction for Gemma
# Focus on layers 26-40 (mid-to-late layers where behavior is likely encoded)
# Using fewer layers to save time (GemmaScope is large)
python blackjack/phase1_feature_extraction.py \
  --model gemma \
  --gpu 0 \
  --input "$GEMMA_RESULTS" \
  --layers 26,28,30,32,35,38,40 \
  --output-dir /scratch/x3415a02/data/llm-addiction/blackjack/sae_features

echo "Gemma SAE feature extraction completed!"
