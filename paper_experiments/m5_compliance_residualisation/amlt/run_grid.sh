#!/usr/bin/env bash
set -uo pipefail

# M5 compliance residualisation. The underlying scripts handle the
# (model × direction × mode) sweep internally via the config file, so this
# launcher is just an ordered chain of the four phases.

OUTPUT_DIR="/scratch/x3415a02/data/llm-addiction/m5_residualisation"
mkdir -p "${OUTPUT_DIR}/directions" "${OUTPUT_DIR}/residualised" /scratch/logs

echo "[m5] phase 1: extract directions (gemma + llama × 3 directions)"
python paper_experiments/m5_compliance_residualisation/src/extract_compliance_directions.py \
    --model all --output "${OUTPUT_DIR}" \
    2>&1 | tee -a /scratch/logs/m5_extract.log || \
    echo "[m5] extract FAILED, continuing"

echo "[m5] phase 2: baseline ΔG·dp (gemma + llama)"
python paper_experiments/m5_compliance_residualisation/src/compute_baseline_dp.py \
    --model all --output "${OUTPUT_DIR}/delta_g_dp_baseline.json" \
    2>&1 | tee -a /scratch/logs/m5_baseline.log || \
    echo "[m5] baseline FAILED, continuing"

echo "[m5] phase 3: residualise SAE features (gemma + llama × individual+joint)"
python paper_experiments/m5_compliance_residualisation/src/residualise_sae_features.py \
    --model all --directions-dir "${OUTPUT_DIR}/directions" --output "${OUTPUT_DIR}" \
    2>&1 | tee -a /scratch/logs/m5_residualise.log || \
    echo "[m5] residualise FAILED, continuing"

echo "[m5] phase 4: refit Table 3 on residualised features"
python paper_experiments/m5_compliance_residualisation/src/refit_table3_residualised.py \
    --model all --input "${OUTPUT_DIR}/residualised" --output "${OUTPUT_DIR}" \
    2>&1 | tee -a /scratch/logs/m5_refit.log || \
    echo "[m5] refit FAILED, continuing"

echo "[m5] phase 5: aggregate analysis"
python paper_experiments/m5_compliance_residualisation/src/analyze_m5.py \
    --output "${OUTPUT_DIR}/_analysis.json" \
    2>&1 | tee -a /scratch/logs/m5_analyze.log || \
    echo "[m5] analyze FAILED"

echo "[m5] grid done"
