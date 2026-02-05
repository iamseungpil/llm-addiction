#!/bin/bash
#SBATCH --job-name=lootbox_test
#SBATCH --partition=cas_v100_4
#SBATCH --output=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/lootbox_test_%j.log
#SBATCH --error=/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/logs/lootbox_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --comment pytorch

# Lootbox experiment test with new total assets prompt

MODEL=${1:-gemma}

echo "=========================================="
echo "Lootbox Test - Total Assets Prompt"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Set working directory
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Create log directory
mkdir -p logs

echo ""
echo "=========================================="
echo "LOOT BOX (Quick: 320 games)"
echo "=========================================="
python -m src.lootbox.run_experiment --model $MODEL --gpu 0 --quick

echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETED"
echo "End time: $(date)"
echo "=========================================="

# List output files
echo ""
echo "Output files:"
ls -la /scratch/x3415a02/data/llm-addiction/lootbox/
