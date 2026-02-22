#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack-gemma
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:2
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=16
# [SLURM-DISABLED] #SBATCH --mem=80G
# [SLURM-DISABLED] #SBATCH --time=1-12:00:00
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_gemma_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_gemma_%j.err

# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

cd /home/jovyan/llm-addiction

echo "======================================================="
echo "BLACKJACK - GEMMA ALL EXPERIMENTS (7 conditions)"
echo "======================================================="
echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

run_exp() {
    local constraint=$1
    local bet_type=$2
    local gpu=$3
    local name="gemma_${bet_type}_${constraint}"
    echo "--- Starting: $name on GPU $gpu | $(date) ---"

    if [ "$constraint" == "unconstrained" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
            --model gemma --gpu 0 --bet-type $bet_type --quick \
            2>&1
    else
        CUDA_VISIBLE_DEVICES=$gpu python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
            --model gemma --gpu 0 --bet-type $bet_type --constraint $constraint --quick \
            2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $name"
    else
        echo "✗ FAILED: $name"
    fi
    echo ""
}

# GPU 0: Fixed betting (3) + variable_unconstrained (1)
(
    echo "=== GPU 0: Gemma Fixed ==="
    run_exp 10 fixed 0
    run_exp 30 fixed 0
    run_exp 50 fixed 0
    run_exp unconstrained variable 0
) &

# GPU 1: Variable betting (3)
(
    echo "=== GPU 1: Gemma Variable ==="
    run_exp 10 variable 1
    run_exp 30 variable 1
    run_exp 50 variable 1
) &

wait

echo "======================================================="
echo "ALL DONE | End: $(date)"
echo "======================================================="
