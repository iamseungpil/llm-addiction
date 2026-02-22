#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=coin-llama-fix10
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=8
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --time=02:00:00
# [SLURM-DISABLED] #SBATCH --comment=pytorch
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/coin_flip_llama_fixed10_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/coin_flip_llama_fixed10_%j.err

# REQUIRED: Conda initialization on HPC cluster
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to repository
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Log GPU info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
nvidia-smi

# Run experiment: LLaMA, Fixed betting, $10
python src/coin_flip/run_experiment.py \
    --model llama \
    --gpu 0 \
    --bet-type fixed \
    --bet-constraint 10 \
    --num-games 50

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
