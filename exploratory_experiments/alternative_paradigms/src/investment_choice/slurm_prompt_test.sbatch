#!/bin/bash
#SBATCH --job-name=inv_prompt_test
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --comment python
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/inv_prompt_test_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/inv_prompt_test_%j.err

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/src

python investment_choice/test_new_prompt.py --gpu 0 --n-reps 10 --constraint 30
