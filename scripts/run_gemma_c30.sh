#!/bin/bash
#SBATCH --job-name=gemma_c30_investment
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/gemma_c30_investment_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/gemma_c30_investment_%j.err

echo "================================================================"
echo "Gemma c30 Investment Choice Experiment - Retry"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================================"

# Initialize conda
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to repository
cd /scratch/x3415a02/projects/llm-addiction

# Check GPU
nvidia-smi

echo ""
echo "Starting Gemma c30 experiment..."
echo "Time limit: 5 hours"
echo ""

# Run experiment
python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py \
    --model gemma \
    --gpu 0 \
    --constraint 30 \
    --quick

exitcode=$?

echo ""
echo "================================================================"
echo "Experiment completed with exit code: $exitcode"
echo "End time: $(date)"
echo "================================================================"

exit $exitcode
