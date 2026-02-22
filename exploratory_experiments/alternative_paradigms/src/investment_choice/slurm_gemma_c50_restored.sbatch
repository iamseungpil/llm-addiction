#!/bin/bash
#SBATCH --job-name=gemma-inv-c50-restored
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/gemma_investment_c50_restored_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/gemma_investment_c50_restored_%j.err

# REQUIRED: Conda initialization on HPC cluster
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to repository
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Run experiment with RESTORED format (Investment Game header + full history + CoT)
echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "=========================================="
echo "Starting Gemma Investment Choice c50 experiment (RESTORED format)..."
echo "Restored changes:"
echo "  - Investment Game (Round X/Y) header"
echo "  - Full history (max_rounds=5)"
echo "  - max_new_tokens: 250 (was 20)"
echo "  - temperature: 0.7 (was 0.3)"
echo "  - CoT: 'Explain your reasoning...'"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

python src/investment_choice/run_experiment.py \
    --model gemma \
    --gpu 0 \
    --constraint 50

echo "End time: $(date)"
echo "Experiment completed!"

