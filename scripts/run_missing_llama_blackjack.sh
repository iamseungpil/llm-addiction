#!/bin/bash
#SBATCH --job-name=blackjack-llama-missing
#SBATCH --partition=cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --comment=python
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/blackjack_llama_missing_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/blackjack_llama_missing_%j.err

source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
conda activate llm-addiction

cd /scratch/x3415a02/projects/llm-addiction

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
