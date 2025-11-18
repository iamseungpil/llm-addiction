#!/bin/bash

# Step 2: Analyze Feature-Word Patterns using Cached States
# This step analyzes all 87,012 features using pre-computed hidden states
# Time: ~15 hours
# Requires: Step 1 cache files must exist

cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis

python3 step2_analyze_with_cache.py \
    --gpu 0 \
    --layer-start 1 \
    --layer-end 31 \
    2>&1 | tee logs/step2_analyze_L1_31_$(date +%Y%m%d_%H%M%S).log
