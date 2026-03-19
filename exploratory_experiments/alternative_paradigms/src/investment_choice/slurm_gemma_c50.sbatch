#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=gemma-inv-c50
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --time=03:00:00
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/gemma_investment_c50_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/gemma_investment_c50_%j.err

# REQUIRED: Conda initialization on HPC cluster
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to repository
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Run experiment
echo "Starting Gemma Investment Choice c50 experiment..."
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

python src/investment_choice/run_experiment.py \
    --model gemma \
    --gpu 0 \
    --constraint 50

echo "End time: $(date)"
echo "Experiment completed!"
