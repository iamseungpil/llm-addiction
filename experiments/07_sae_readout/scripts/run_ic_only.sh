#!/bin/bash
# Run analysis on IC only (already extracted)
# Can run immediately without waiting for SM/MW extraction

set -e
cd /home/jovyan/llm-addiction/sae_v3_analysis/src

echo "============================================"
echo "IC-Only V3 SAE Analysis"
echo "============================================"

# 1. BK Classification
echo "[1/2] BK Classification (IC)..."
python classify_bk.py --paradigm ic

# 2. Round Trajectory
echo "[2/2] Round Trajectory (IC, L18 L26 L30)..."
python round_trajectory.py --paradigm ic --layer 18 26 30

echo ""
echo "COMPLETE. Results: ../results/"
