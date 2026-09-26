#!/bin/bash
# Run all V3 SAE analyses
# Prerequisites: extraction complete for all 3 paradigms
#   sae_features_v3/{investment_choice,slot_machine,mystery_wheel}/gemma/

set -e
cd /home/jovyan/llm-addiction/sae_v3_analysis/src

echo "============================================"
echo "V3 SAE Analysis Pipeline"
echo "============================================"

# 1. BK Classification (per paradigm, 3 modes)
echo ""
echo "[1/3] BK Classification..."
python classify_bk.py --paradigm ic sm mw

# 2. Cross-Domain Transfer
echo ""
echo "[2/3] Cross-Domain Transfer..."
python cross_domain.py --mode decision_point
python cross_domain.py --mode game_mean

# 3. Round-Level Trajectory (best layers from IC V2: L18, L26, L30)
echo ""
echo "[3/3] Round Trajectory Analysis..."
python round_trajectory.py --paradigm ic sm mw --layer 18 26 30

echo ""
echo "============================================"
echo "ALL COMPLETE"
echo "Results: /home/jovyan/llm-addiction/sae_v3_analysis/results/"
echo "============================================"
