#!/bin/bash
#SBATCH -J investment_fullset
#SBATCH -p cas_v100_4
#SBATCH --comment pytorch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/investment_fullset_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/investment_fullset_%j.err

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
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

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
ls -lh /scratch/x3415a02/data/llm-addiction/investment_choice/*$(date +%Y%m%d)*.json

echo ""
echo "========================================================================"
echo "All experiments completed!"
echo "End time: $(date)"
echo "========================================================================"
