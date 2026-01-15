#!/bin/bash
# =============================================================================
# Full Layer Causal Analysis Pipeline - 4 GPU Version
# =============================================================================
#
# GPU Assignment:
#   Phase 1 (Extraction): LLaMA on GPU 4, Gemma on GPU 5
#   Phase 2 (SAE Projection): 4-way parallel
#     - LLaMA L0-15 on GPU 4
#     - LLaMA L16-31 on GPU 6
#     - Gemma L0-20 on GPU 5
#     - Gemma L21-41 on GPU 7
#   Phase 3, 5: LLaMA on GPU 4, Gemma on GPU 5
#
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG="${SCRIPT_DIR}/configs/experiment_config_full_layers.yaml"
OUTPUT_DIR="/data/llm_addiction/steering_vector_experiment_full"
LOG_DIR="${OUTPUT_DIR}/logs"
CONDA_ENV="/data/miniforge3/envs/llama_sae_env/bin/python"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo " FULL LAYER CAUSAL ANALYSIS PIPELINE (4 GPU)"
echo "============================================================"
echo " Config: ${CONFIG}"
echo " Output: ${OUTPUT_DIR}"
echo " Timestamp: ${TIMESTAMP}"
echo " Phases: 1, 2, 3, 5 (Phase 4 skipped)"
echo " GPUs: 4, 5, 6, 7"
echo "============================================================"

# Step 0: Extract 8-condition prompts
echo ""
echo "[Step 0] Extracting 8-condition prompts..."
${CONDA_ENV} "${SRC_DIR}/real_prompt_extractor.py" \
    --output "${OUTPUT_DIR}/condition_prompts_${TIMESTAMP}.json" \
    2>&1 | tee "${LOG_DIR}/prompt_extraction_${TIMESTAMP}.log"

# ============================================================
# PHASE 1: Steering Vector Extraction (2 GPUs)
# ============================================================
echo ""
echo "============================================================"
echo " PHASE 1: Steering Vector Extraction"
echo "============================================================"

# Check if LLaMA vectors already exist
LLAMA_VECTORS=$(ls -t ${OUTPUT_DIR}/steering_vectors_llama_*.npz 2>/dev/null | head -1)
if [ -n "$LLAMA_VECTORS" ]; then
    echo "[LLaMA] Using existing vectors: $LLAMA_VECTORS"
else
    echo "[LLaMA] Starting extraction on GPU 4..."
    CUDA_VISIBLE_DEVICES=4 ${CONDA_ENV} "${SRC_DIR}/extract_steering_vectors.py" \
        --model llama \
        --gpu 4 \
        --config "${CONFIG}" \
        2>&1 | tee "${LOG_DIR}/phase1_llama_${TIMESTAMP}.log" &
    LLAMA_P1_PID=$!
fi

# Check if Gemma vectors already exist
GEMMA_VECTORS=$(ls -t ${OUTPUT_DIR}/steering_vectors_gemma_*.npz 2>/dev/null | head -1)
if [ -n "$GEMMA_VECTORS" ]; then
    echo "[Gemma] Using existing vectors: $GEMMA_VECTORS"
else
    echo "[Gemma] Starting extraction on GPU 5..."
    CUDA_VISIBLE_DEVICES=5 ${CONDA_ENV} "${SRC_DIR}/extract_steering_vectors.py" \
        --model gemma \
        --gpu 5 \
        --config "${CONFIG}" \
        2>&1 | tee "${LOG_DIR}/phase1_gemma_${TIMESTAMP}.log" &
    GEMMA_P1_PID=$!
fi

# Wait for Phase 1 to complete
if [ -n "$LLAMA_P1_PID" ]; then
    echo "Waiting for LLaMA extraction (PID: $LLAMA_P1_PID)..."
    wait $LLAMA_P1_PID
fi
if [ -n "$GEMMA_P1_PID" ]; then
    echo "Waiting for Gemma extraction (PID: $GEMMA_P1_PID)..."
    wait $GEMMA_P1_PID
fi

echo "Phase 1 complete!"

# Get the latest vector files
LLAMA_VECTORS=$(ls -t ${OUTPUT_DIR}/steering_vectors_llama_*.npz 2>/dev/null | head -1)
GEMMA_VECTORS=$(ls -t ${OUTPUT_DIR}/steering_vectors_gemma_*.npz 2>/dev/null | head -1)

if [ -z "$LLAMA_VECTORS" ] || [ -z "$GEMMA_VECTORS" ]; then
    echo "ERROR: Missing steering vectors!"
    exit 1
fi

echo "LLaMA vectors: $LLAMA_VECTORS"
echo "Gemma vectors: $GEMMA_VECTORS"

# ============================================================
# PHASE 2: SAE Feature Projection (4 GPUs parallel)
# ============================================================
echo ""
echo "============================================================"
echo " PHASE 2: SAE Feature Projection (4-way parallel)"
echo "============================================================"

# LLaMA layers 0-15 on GPU 4
echo "[LLaMA L0-15] Starting on GPU 4..."
CUDA_VISIBLE_DEVICES=4 ${CONDA_ENV} "${SRC_DIR}/sae_feature_projection.py" \
    --model llama \
    --gpu 4 \
    --config "${CONFIG}" \
    --vectors "${LLAMA_VECTORS}" \
    --layers "0-15" \
    2>&1 | tee "${LOG_DIR}/phase2_llama_L0-15_${TIMESTAMP}.log" &
LLAMA_P2A_PID=$!

# LLaMA layers 16-31 on GPU 6
echo "[LLaMA L16-31] Starting on GPU 6..."
CUDA_VISIBLE_DEVICES=6 ${CONDA_ENV} "${SRC_DIR}/sae_feature_projection.py" \
    --model llama \
    --gpu 6 \
    --config "${CONFIG}" \
    --vectors "${LLAMA_VECTORS}" \
    --layers "16-31" \
    2>&1 | tee "${LOG_DIR}/phase2_llama_L16-31_${TIMESTAMP}.log" &
LLAMA_P2B_PID=$!

# Gemma layers 0-20 on GPU 5
echo "[Gemma L0-20] Starting on GPU 5..."
CUDA_VISIBLE_DEVICES=5 ${CONDA_ENV} "${SRC_DIR}/sae_feature_projection.py" \
    --model gemma \
    --gpu 5 \
    --config "${CONFIG}" \
    --vectors "${GEMMA_VECTORS}" \
    --layers "0-20" \
    2>&1 | tee "${LOG_DIR}/phase2_gemma_L0-20_${TIMESTAMP}.log" &
GEMMA_P2A_PID=$!

# Gemma layers 21-41 on GPU 7
echo "[Gemma L21-41] Starting on GPU 7..."
CUDA_VISIBLE_DEVICES=7 ${CONDA_ENV} "${SRC_DIR}/sae_feature_projection.py" \
    --model gemma \
    --gpu 7 \
    --config "${CONFIG}" \
    --vectors "${GEMMA_VECTORS}" \
    --layers "21-41" \
    2>&1 | tee "${LOG_DIR}/phase2_gemma_L21-41_${TIMESTAMP}.log" &
GEMMA_P2B_PID=$!

echo ""
echo "Phase 2 running on 4 GPUs..."
echo "  LLaMA L0-15:  PID $LLAMA_P2A_PID (GPU 4)"
echo "  LLaMA L16-31: PID $LLAMA_P2B_PID (GPU 6)"
echo "  Gemma L0-20:  PID $GEMMA_P2A_PID (GPU 5)"
echo "  Gemma L21-41: PID $GEMMA_P2B_PID (GPU 7)"

# Wait for all Phase 2 to complete
wait $LLAMA_P2A_PID $LLAMA_P2B_PID $GEMMA_P2A_PID $GEMMA_P2B_PID
echo "Phase 2 complete!"

# Merge candidate features from parallel runs
echo ""
echo "Merging candidate features..."
${CONDA_ENV} - << 'MERGE_SCRIPT'
import json
from pathlib import Path
from datetime import datetime

output_dir = Path("/data/llm_addiction/steering_vector_experiment_full")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

for model in ['llama', 'gemma']:
    # Find all candidate files for this model (with layer suffix like _L0-15_)
    pattern = f"candidate_features_{model}_L*_*.json"
    files = sorted(output_dir.glob(pattern))

    if not files:
        # Try without layer suffix
        pattern = f"candidate_features_{model}_*.json"
        files = [f for f in output_dir.glob(pattern) if 'merged' not in f.name]

    if not files:
        print(f"No candidate files found for {model}")
        continue

    # Merge all layers from all files
    all_layers = {}
    all_candidates = []

    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)

            # Handle 'layers' format (from sae_feature_projection.py)
            if 'layers' in data:
                for layer_str, layer_data in data['layers'].items():
                    layer_num = int(layer_str)
                    if layer_num not in all_layers:
                        all_layers[layer_num] = layer_data
                    # Extract candidates from top_features
                    if 'top_features' in layer_data:
                        for feat in layer_data['top_features']:
                            all_candidates.append({
                                'layer': layer_num,
                                'feature_id': feat['feature_id'],
                                'contribution': feat.get('magnitude', feat.get('value', 0.0)),
                                'direction': feat.get('direction', 'risky')
                            })

    # Sort by contribution magnitude
    all_candidates.sort(key=lambda x: abs(x.get('contribution', 0)), reverse=True)

    # Save merged file
    merged_path = output_dir / f"candidate_features_{model}_merged_{timestamp}.json"
    with open(merged_path, 'w') as fp:
        json.dump({
            'model': model,
            'total_candidates': len(all_candidates),
            'total_layers': len(all_layers),
            'source_files': [str(f) for f in files],
            'candidates': all_candidates,
            'layers': {str(k): v for k, v in sorted(all_layers.items())}
        }, fp, indent=2)

    print(f"{model}: Merged {len(all_candidates)} candidates from {len(all_layers)} layers ({len(files)} files) -> {merged_path}")
MERGE_SCRIPT

# ============================================================
# PHASE 3: Soft Interpolation Patching (2 GPUs)
# ============================================================
echo ""
echo "============================================================"
echo " PHASE 3: Soft Interpolation Patching"
echo "============================================================"

LLAMA_CANDIDATES=$(ls -t ${OUTPUT_DIR}/candidate_features_llama_merged_*.json 2>/dev/null | head -1)
GEMMA_CANDIDATES=$(ls -t ${OUTPUT_DIR}/candidate_features_gemma_merged_*.json 2>/dev/null | head -1)
PROMPTS_FILE=$(ls -t ${OUTPUT_DIR}/condition_prompts_*.json 2>/dev/null | head -1)

echo "[LLaMA] Starting on GPU 4..."
CUDA_VISIBLE_DEVICES=4 ${CONDA_ENV} "${SRC_DIR}/soft_interpolation_patching.py" \
    --model llama \
    --gpu 4 \
    --config "${CONFIG}" \
    --candidates "${LLAMA_CANDIDATES}" \
    --prompts "${PROMPTS_FILE}" \
    2>&1 | tee "${LOG_DIR}/phase3_llama_${TIMESTAMP}.log" &
LLAMA_P3_PID=$!

echo "[Gemma] Starting on GPU 5..."
CUDA_VISIBLE_DEVICES=5 ${CONDA_ENV} "${SRC_DIR}/soft_interpolation_patching.py" \
    --model gemma \
    --gpu 5 \
    --config "${CONFIG}" \
    --candidates "${GEMMA_CANDIDATES}" \
    --prompts "${PROMPTS_FILE}" \
    2>&1 | tee "${LOG_DIR}/phase3_gemma_${TIMESTAMP}.log" &
GEMMA_P3_PID=$!

echo "Waiting for Phase 3..."
wait $LLAMA_P3_PID $GEMMA_P3_PID
echo "Phase 3 complete!"

# ============================================================
# PHASE 5: Gambling-Context Interpretation (2 GPUs)
# ============================================================
echo ""
echo "============================================================"
echo " PHASE 5: Gambling-Context Interpretation"
echo "============================================================"

LLAMA_VALIDATED=$(ls -t ${OUTPUT_DIR}/validated_features_llama_*.json 2>/dev/null | head -1)
GEMMA_VALIDATED=$(ls -t ${OUTPUT_DIR}/validated_features_gemma_*.json 2>/dev/null | head -1)

echo "[LLaMA] Starting on GPU 4..."
CUDA_VISIBLE_DEVICES=4 ${CONDA_ENV} "${SRC_DIR}/gambling_interpretation.py" \
    --model llama \
    --gpu 4 \
    --config "${CONFIG}" \
    --validated "${LLAMA_VALIDATED}" \
    2>&1 | tee "${LOG_DIR}/phase5_llama_${TIMESTAMP}.log" &
LLAMA_P5_PID=$!

echo "[Gemma] Starting on GPU 5..."
CUDA_VISIBLE_DEVICES=5 ${CONDA_ENV} "${SRC_DIR}/gambling_interpretation.py" \
    --model gemma \
    --gpu 5 \
    --config "${CONFIG}" \
    --validated "${GEMMA_VALIDATED}" \
    2>&1 | tee "${LOG_DIR}/phase5_gemma_${TIMESTAMP}.log" &
GEMMA_P5_PID=$!

echo "Waiting for Phase 5..."
wait $LLAMA_P5_PID $GEMMA_P5_PID
echo "Phase 5 complete!"

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results in: ${OUTPUT_DIR}"
echo "Logs in: ${LOG_DIR}"
echo "============================================================"
