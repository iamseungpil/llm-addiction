#!/bin/bash
#SBATCH --job-name=blackjack-all
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_all_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_all_%j.err

# REQUIRED: Conda initialization on HPC cluster
source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

# Navigate to repository
cd /scratch/x3415a02/projects/llm-addiction

echo "======================================================================="
echo "BLACKJACK EXPERIMENT BATCH - 14 EXPERIMENTS"
echo "======================================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Create log directory
mkdir -p /scratch/x3415a02/data/llm-addiction/logs

# Function to run experiment with error handling
run_exp() {
    local model=$1
    local constraint=$2
    local bet_type=$3
    local gpu=$4

    local name="${model}_${bet_type}_${constraint}"
    echo "----------------------------------------"
    echo "Starting: $name on GPU $gpu"
    echo "Time: $(date)"

    if [ "$constraint" == "unconstrained" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
            --model $model --gpu 0 --bet-type $bet_type --quick \
            2>&1 | tee /scratch/x3415a02/data/llm-addiction/logs/${name}_${SLURM_JOB_ID}.log
    else
        CUDA_VISIBLE_DEVICES=$gpu python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
            --model $model --gpu 0 --bet-type $bet_type --constraint $constraint --quick \
            2>&1 | tee /scratch/x3415a02/data/llm-addiction/logs/${name}_${SLURM_JOB_ID}.log
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $name"
    else
        echo "✗ FAILED: $name"
    fi
    echo ""
}

# GPU 0: All Fixed betting experiments (7 total)
(
    echo "=== GPU 0: Fixed Betting Experiments ==="
    run_exp llama 10 fixed 0
    run_exp llama 30 fixed 0
    run_exp llama 50 fixed 0
    run_exp gemma 10 fixed 0
    run_exp gemma 30 fixed 0
    run_exp gemma 50 fixed 0
    run_exp gemma unconstrained variable 0
) &

# GPU 1: All Variable betting experiments (7 total)
(
    echo "=== GPU 1: Variable Betting Experiments ==="
    run_exp llama 10 variable 1
    run_exp llama 30 variable 1
    run_exp llama 50 variable 1
    run_exp llama unconstrained variable 1
    run_exp gemma 10 variable 1
    run_exp gemma 30 variable 1
    run_exp gemma 50 variable 1
) &

# Wait for both GPU processes to complete
wait

echo ""
echo "======================================================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "======================================================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: /scratch/x3415a02/data/llm-addiction/blackjack/"
echo "Logs saved to: /scratch/x3415a02/data/llm-addiction/logs/"
