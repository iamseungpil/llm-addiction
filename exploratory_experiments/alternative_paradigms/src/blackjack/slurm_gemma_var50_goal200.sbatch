#!/bin/bash
#SBATCH --job-name=gemma-v50-g200
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_gemma_var50_goal200_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_gemma_var50_goal200_%j.err

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOBID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "=========================================="

# Conda initialization
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to project directory
cd /scratch/x3415a02/projects/llm-addiction

# Run experiment - Variable max $50 with Goal $200
echo "Starting Variable max \$50 with Goal \$200..."

python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
    --model gemma \
    --gpu 0 \
    --bet-type variable \
    --constraint 50 \
    --quick \
    --no-fewshot \
    --initial-goal 200

echo "Experiment completed!"
