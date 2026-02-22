#!/bin/bash
#SBATCH --job-name=gemma-v30-g300-cot
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_gemma_var30_goal300_cot_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_gemma_var30_goal300_cot_%j.err

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOBID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "=========================================="

# Conda initialization
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to project directory
cd /scratch/x3415a02/projects/llm-addiction

# Run experiment - Variable max $30 with Goal $300 + Chain-of-Thought
echo "Starting Variable max \$30 with Goal \$300 + CoT..."

python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
    --model gemma \
    --gpu 0 \
    --bet-type variable \
    --constraint 30 \
    --quick \
    --no-fewshot \
    --initial-goal 300 \
    --cot

echo "Experiment completed!"
