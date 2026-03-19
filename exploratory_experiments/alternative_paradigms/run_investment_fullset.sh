#!/bin/bash
# [SLURM-DISABLED] #SBATCH -J investment_fullset
# [SLURM-DISABLED] #SBATCH -p cas_v100_4
# [SLURM-DISABLED] #SBATCH --comment pytorch
# [SLURM-DISABLED] #SBATCH --gres=gpu:1
# [SLURM-DISABLED] #SBATCH --cpus-per-task=4
# [SLURM-DISABLED] #SBATCH --mem=32G
# [SLURM-DISABLED] #SBATCH --time=06:00:00
# [SLURM-DISABLED] #SBATCH --output=/home/jovyan/beomi/llm-addiction-data/logs/investment_fullset_%j.out
# [SLURM-DISABLED] #SBATCH --error=/home/jovyan/beomi/llm-addiction-data/logs/investment_fullset_%j.err

echo "========================================================================"
echo "Investment Choice Full-Set Experiment"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================================"

# Activate conda environment
source ~/.bashrc
conda activate llama_sae_env

# Navigate to project directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Check GPU
echo ""
echo "GPU Status:"
nvidia-smi
echo ""

# Run LLaMA full experiment
echo "========================================================================"
echo "Starting LLaMA-3.1-8B Full Experiment (400 games)"
echo "========================================================================"
echo "Start time: $(date)"

python src/investment_choice/run_experiment.py \
  --model llama \
  --gpu 0 \
  --constraint unlimited

echo ""
echo "LLaMA experiment completed at: $(date)"
echo ""

# Clear GPU memory and wait
sleep 30

# Check GPU again
echo "GPU Status after LLaMA:"
nvidia-smi
echo ""

# Run Gemma full experiment
echo "========================================================================"
echo "Starting Gemma-2-9B Full Experiment (400 games)"
echo "========================================================================"
echo "Start time: $(date)"

python src/investment_choice/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --constraint unlimited

echo ""
echo "Gemma experiment completed at: $(date)"
echo ""

# List output files
echo "========================================================================"
echo "Output Files:"
echo "========================================================================"
ls -lh /home/jovyan/beomi/llm-addiction-data/investment_choice/*$(date +%Y%m%d)*.json

echo ""
echo "========================================================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "========================================================================"
