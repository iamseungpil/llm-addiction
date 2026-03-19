#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=gemma-v30-pzs
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --time=08:00:00
# [SLURM-DISABLED] #SBATCH --comment=pytorch
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_gemma_var30_pure_zs_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_gemma_var30_pure_zs_%j.err

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOBID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "=========================================="

# Conda initialization
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to project directory
cd /home/jovyan/llm-addiction

# Run experiment - Gemma Variable (max $30) PURE ZERO-SHOT (no examples)
echo "Starting Gemma Variable (max \$30) PURE ZERO-SHOT (no examples)..."

python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
    --model gemma \
    --gpu 0 \
    --bet-type variable \
    --constraint 30 \
    --quick \
    --no-fewshot \
    --no-examples

echo "Experiment completed!"
