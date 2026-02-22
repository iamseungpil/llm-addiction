#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=llama-investment-full
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --time=20:00:00
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/%x_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/%x_%j.err

# REQUIRED: Conda initialization on HPC cluster
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

# Navigate to repository
cd /home/jovyan/llm-addiction

echo "================================================================"
echo "LLaMA Full Mode Investment Choice Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================================"

# GPU check
nvidia-smi

# Experiment 1: LLaMA c30 Full Mode (400 games)
echo ""
echo "================================================================"
echo "Starting LLaMA c30 Full Mode (400 games)..."
echo "Expected time: ~7-8 hours"
echo "================================================================"

python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py \
  --model llama \
  --gpu 0 \
  --constraint 30

LLAMA_C30_EXIT=$?

if [ $LLAMA_C30_EXIT -eq 0 ]; then
    echo "LLaMA c30 completed successfully!"
else
    echo "LLaMA c30 failed with exit code: $LLAMA_C30_EXIT"
    exit $LLAMA_C30_EXIT
fi

# Clear GPU memory between experiments
sleep 10

# Experiment 2: LLaMA c50 Full Mode (400 games)
echo ""
echo "================================================================"
echo "Starting LLaMA c50 Full Mode (400 games)..."
echo "Expected time: ~7-8 hours"
echo "================================================================"

python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py \
  --model llama \
  --gpu 0 \
  --constraint 50

LLAMA_C50_EXIT=$?

echo ""
echo "================================================================"
echo "All LLaMA experiments completed!"
echo "c30 exit code: $LLAMA_C30_EXIT"
echo "c50 exit code: $LLAMA_C50_EXIT"
echo "End time: $(date)"
echo "================================================================"

exit $LLAMA_C50_EXIT
