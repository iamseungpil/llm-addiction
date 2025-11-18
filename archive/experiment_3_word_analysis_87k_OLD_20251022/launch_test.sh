#!/bin/bash

# Test script - analyze only Layer 25 first
# This will analyze ~3,478 features from Layer 25

cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis

python3 experiment_3_L1_31_extended.py \
    --gpu 0 \
    --layer-start 25 \
    --layer-end 25 \
    2>&1 | tee logs/test_L25_$(date +%Y%m%d_%H%M%S).log
