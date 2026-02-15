#!/bin/bash
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --comment=pytorch
#SBATCH --time=08:00:00

# Blackjack experiment - Single configuration
# 8 components × 50 reps = 400 games per config

# Initialize conda
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

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
#SBATCH --job-name=bj_${MODEL}_${CONFIG_NAME}
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_${MODEL}_${CONFIG_NAME}_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_${MODEL}_${CONFIG_NAME}_%j.err

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
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

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
