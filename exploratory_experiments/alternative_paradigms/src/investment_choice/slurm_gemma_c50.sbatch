#!/bin/bash
#SBATCH --job-name=gemma-inv-c50
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/gemma_investment_c50_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/gemma_investment_c50_%j.err

# REQUIRED: Conda initialization on HPC cluster
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to repository
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

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
