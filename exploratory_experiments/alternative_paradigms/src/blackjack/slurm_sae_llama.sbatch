#!/bin/bash
#SBATCH --job-name=blackjack_sae_llama
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --comment=python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_sae_llama_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_sae_llama_%j.err

# Conda initialization
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/src

# Find the most recent LLaMA blackjack results
LLAMA_RESULTS=$(ls -t /scratch/x3415a02/data/llm-addiction/blackjack/blackjack_llama_*.json | head -1)

echo "Processing LLaMA results: $LLAMA_RESULTS"

# Run Phase 1: SAE feature extraction for LLaMA
# Focus on layers 25-31 (LlamaScope coverage)
python blackjack/phase1_feature_extraction.py \
  --model llama \
  --gpu 0 \
  --input "$LLAMA_RESULTS" \
  --layers 25,26,27,28,29,30,31 \
  --output-dir /scratch/x3415a02/data/llm-addiction/blackjack/sae_features

echo "LLaMA SAE feature extraction completed!"
