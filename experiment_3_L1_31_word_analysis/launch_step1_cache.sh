#!/bin/bash

# Step 1: Cache Hidden States for L1-31 Analysis
# This step extracts hidden states ONCE for all 6,400 responses
# Time: ~1.1 days
# Output: ~465MB cache files

cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis

python3 step1_cache_hidden_states.py \
    --gpu 0 \
    --layer-start 1 \
    --layer-end 31 \
    2>&1 | tee logs/step1_cache_L1_31_$(date +%Y%m%d_%H%M%S).log
