#!/bin/bash
#SBATCH --job-name=debug_prompt
#SBATCH --partition=cas_v100_4
#SBATCH --output=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/debug_prompt_%j.log
#SBATCH --error=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/debug_prompt_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --comment pytorch

echo "Debug Completion-Style Prompts"
echo "=============================="

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

python scripts/debug_completion_prompt.py --test blackjack

echo ""
echo "Done!"
