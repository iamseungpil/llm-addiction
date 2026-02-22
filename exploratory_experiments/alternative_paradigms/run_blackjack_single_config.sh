#!/bin/bash
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --comment=pytorch
# [SLURM-DISABLED] #SBATCH --time=08:00:00

# Blackjack experiment - Single configuration
# 8 components × 50 reps = 400 games per config

# Initialize conda
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

MODEL=$1
BET_TYPE=$2
CONSTRAINT=$3

if [ -z "$MODEL" ] || [ -z "$BET_TYPE" ] || [ -z "$CONSTRAINT" ]; then
    echo "Usage: sbatch run_blackjack_single_config.sh <model> <bet_type> <constraint>"
    echo "  model: llama, gemma, qwen"
    echo "  bet_type: fixed, variable"
    echo "  constraint: 10, 30, 50, 70"
    exit 1
fi

# Set job name dynamically
CONFIG_NAME="${BET_TYPE}_${CONSTRAINT}"

# Update SLURM job name and output files
# [SLURM-DISABLED] #SBATCH --job-name=bj_${MODEL}_${CONFIG_NAME}
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_${MODEL}_${CONFIG_NAME}_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_${MODEL}_${CONFIG_NAME}_%j.err

echo "========================================="
echo "Blackjack Experiment - Single Config"
echo "========================================="
echo "Model: $MODEL"
echo "Bet Type: $BET_TYPE"
echo "Constraint: \$$CONSTRAINT"
echo "Config: $CONFIG_NAME"
echo "Games: 400 (8 components × 50 reps)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Navigate to experiment directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Run experiment
python src/blackjack/run_experiment_single_config.py \
    --model $MODEL \
    --gpu 0 \
    --bet-type $BET_TYPE \
    --constraint $CONSTRAINT

echo ""
echo "========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "========================================="
