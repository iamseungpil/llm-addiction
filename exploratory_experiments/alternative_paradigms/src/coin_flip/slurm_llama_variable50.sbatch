#!/bin/bash
#SBATCH --job-name=coin-llama-var50
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --comment=pytorch
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/coin_flip_llama_var50_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/coin_flip_llama_var50_%j.err

# REQUIRED: Conda initialization on HPC cluster
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to repository
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Log GPU info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
nvidia-smi

# Run experiment: LLaMA, Variable betting, $50 constraint
python src/coin_flip/run_experiment.py \
    --model llama \
    --gpu 0 \
    --bet-type variable \
    --bet-constraint 50 \
    --num-games 50

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
