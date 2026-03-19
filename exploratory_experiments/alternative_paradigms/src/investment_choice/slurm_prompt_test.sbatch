#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=inv_prompt_test
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --comment python
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=8
# [SLURM-DISABLED] #SBATCH --mem=40G
# [SLURM-DISABLED] #SBATCH --time=02:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/inv_prompt_test_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/inv_prompt_test_%j.err

# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src

python investment_choice/test_new_prompt.py --gpu 0 --n-reps 10 --constraint 30
