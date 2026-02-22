#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=gemma_c30_investment
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --time=05:00:00
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/gemma_c30_investment_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/gemma_c30_investment_%j.err

echo "================================================================"
echo "Gemma c30 Investment Choice Experiment - Retry"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================================"

# Initialize conda
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

# Navigate to repository
cd /home/jovyan/llm-addiction

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
