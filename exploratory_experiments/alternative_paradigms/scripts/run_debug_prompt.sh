#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=debug_prompt
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/debug_prompt_%j.log
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/logs/debug_prompt_%j.err
# [SLURM-DISABLED] #SBATCH --time=00:30:00
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --comment pytorch

echo "Debug Completion-Style Prompts"
echo "=============================="

# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

python scripts/debug_completion_prompt.py --test blackjack

echo ""
echo "Done!"
