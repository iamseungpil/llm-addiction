#!/bin/bash

# L1-31 Token Tracking - ATTENTION ONLY (lightweight)
# File size: ~300MB
# Time: ~3-4 hours

cd /home/ubuntu/llm_addiction/experiment_6_L1_31_token_tracking

python3 experiment_6_L1_31_optimized.py \
    --gpu 0 \
    --layer-start 1 \
    --layer-end 31 \
    --no-features \
    2>&1 | tee logs/attention_only_L1_31_$(date +%Y%m%d_%H%M%S).log
