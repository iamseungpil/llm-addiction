#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack_sae_llama
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=8
# [SLURM-DISABLED] #SBATCH --mem=64G
# [SLURM-DISABLED] #SBATCH --time=06:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_sae_llama_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_sae_llama_%j.err

# Conda initialization
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src

# Find the most recent LLaMA blackjack results
LLAMA_RESULTS=$(ls -t /home/jovyan/beomi/llm-addiction-data/blackjack/blackjack_llama_*.json | head -1)

echo "Processing LLaMA results: $LLAMA_RESULTS"

# Run Phase 1: SAE feature extraction for LLaMA
# Focus on layers 25-31 (LlamaScope coverage)
python blackjack/phase1_feature_extraction.py \
  --model llama \
  --gpu 0 \
  --input "$LLAMA_RESULTS" \
  --layers 25,26,27,28,29,30,31 \
  --output-dir /home/jovyan/beomi/llm-addiction-data/blackjack/sae_features

echo "LLaMA SAE feature extraction completed!"
