#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack-all
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:2
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=8
# [SLURM-DISABLED] #SBATCH --mem=64G
# [SLURM-DISABLED] #SBATCH --time=06:00:00
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_all_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_all_%j.err

# REQUIRED: Conda initialization on HPC cluster
# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

# Navigate to repository
cd /home/jovyan/llm-addiction

echo "======================================================================="
echo "BLACKJACK EXPERIMENT BATCH - 14 EXPERIMENTS"
echo "======================================================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Create log directory
mkdir -p /home/jovyan/beomi/llm-addiction-data/logs

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
            2>&1 | tee /home/jovyan/beomi/llm-addiction-data/logs/${name}_${SLURM_JOB_ID}.log
    else
        CUDA_VISIBLE_DEVICES=$gpu python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
            --model $model --gpu 0 --bet-type $bet_type --constraint $constraint --quick \
            2>&1 | tee /home/jovyan/beomi/llm-addiction-data/logs/${name}_${SLURM_JOB_ID}.log
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
echo "Results saved to: /home/jovyan/beomi/llm-addiction-data/blackjack/"
echo "Logs saved to: /home/jovyan/beomi/llm-addiction-data/logs/"
