#!/bin/bash
#SBATCH --job-name=inv_llama_c30_new
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --comment=python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/inv_llama_c30_new_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/inv_llama_c30_new_%j.err

# IMPORTANT: Only submit this AFTER verifying new prompt test succeeds!

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/src

echo "================================"
echo "LLaMA Investment Choice - c30"
echo "WITH IMPROVED PROMPT FORMAT"
echo "================================"
echo "Improvements applied:"
echo "  - max_new_tokens: 250 → 50"
echo "  - temperature: 0.7 → 0.5"
echo "  - Format hints for rounds > 0"
echo "  - Prefix-completion optimization"
echo "================================"

# Full experiment: 2 bet types × 4 conditions × 50 reps = 400 games
# Using improved prompt format to reduce hallucination and improve parsing

python investment_choice/run_experiment.py \
  --model llama \
  --gpu 0 \
  --constraint 30

echo "LLaMA c30 (new prompt) completed!"
