#!/bin/bash

# Activate conda environment
source /data/miniforge3/etc/profile.d/conda.sh
conda activate llama_sae_env

# Set your OpenAI API key here
export OPENAI_API_KEY="your_api_key_here"

# Run the experiment
echo "Starting GPT-4o-mini experiment with 50 repetitions..."
echo "Total experiments: 6,400 (128 conditions × 50 repetitions)"
echo "Please update OPENAI_API_KEY in this script before running!"
echo ""

python gpt_multiround_50reps.py