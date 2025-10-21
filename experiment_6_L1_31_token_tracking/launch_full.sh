#!/bin/bash

# Full L1-31 Token Tracking - Features + Attention
# File size: ~10GB
# Time: ~5-6 hours

cd /home/ubuntu/llm_addiction/experiment_6_L1_31_token_tracking

python3 experiment_6_L1_31_extended.py \
    --gpu 0 \
    --layer-start 1 \
    --layer-end 31 \
    2>&1 | tee logs/full_L1_31_$(date +%Y%m%d_%H%M%S).log
