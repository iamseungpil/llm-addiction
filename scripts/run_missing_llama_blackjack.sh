#!/bin/bash
# [SLURM-DISABLED] #SBATCH --job-name=blackjack-llama-missing
# [SLURM-DISABLED] #SBATCH --partition=cas_v100_4
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --ntasks=1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=8
# [SLURM-DISABLED] #SBATCH --mem=40G
# [SLURM-DISABLED] #SBATCH --time=06:00:00
# [SLURM-DISABLED] #SBATCH --comment=python
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_llama_missing_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/blackjack_llama_missing_%j.err

# [SLURM-DISABLED] source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
# [OpenHPC] conda already activated
# conda activate llm-addiction

cd /home/jovyan/llm-addiction

echo "======================================================="
echo "BLACKJACK - LLAMA MISSING EXPERIMENTS"
echo "======================================================="
echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

run_exp() {
    local constraint=$1
    local bet_type=$2
    local name="llama_${bet_type}_${constraint}"
    echo "--- Starting: $name | $(date) ---"

    if [ "$constraint" == "unconstrained" ]; then
        CUDA_VISIBLE_DEVICES=0 python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
            --model llama --gpu 0 --bet-type $bet_type --quick \
            2>&1
    else
        CUDA_VISIBLE_DEVICES=0 python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
            --model llama --gpu 0 --bet-type $bet_type --constraint $constraint --quick \
            2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Completed: $name"
    else
        echo "✗ FAILED: $name"
    fi
    echo ""
}

# Missing LLaMA experiments
run_exp 50 variable
run_exp unconstrained variable

echo "======================================================="
echo "ALL DONE | End: $(date)"
echo "======================================================="
