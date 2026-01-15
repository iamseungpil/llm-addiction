#!/bin/bash

# Sequential execution of all phases for REPARSED experiment
# Waits for each phase to complete before starting the next
# Uses GPU 4, 5, 6, 7 outputs

PHASE1_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_REPARSED"

echo "=== REPARSED Pathway Analysis: All Phases Sequential ==="
echo "Phase 1 must be running or completed before this script"
echo ""

# Function to check if Phase 1 is complete
check_phase1_complete() {
    local all_complete=true

    for gpu in 4 5 6 7; do
        local patching_file="${PHASE1_DIR}/phase1_patching_multifeature_gpu${gpu}.jsonl"

        if [ ! -f "$patching_file" ]; then
            all_complete=false
            break
        fi

        # Check if file size is large enough (>500MB expected per GPU)
        local file_size=$(stat -c%s "$patching_file" 2>/dev/null || echo "0")
        if [ "$file_size" -lt 500000000 ]; then
            all_complete=false
            break
        fi
    done

    if $all_complete; then
        return 0
    else
        return 1
    fi
}

# Wait for Phase 1 to complete
echo "⏳ Waiting for Phase 1 REPARSED to complete..."
echo "   Expected: GPU 4, 5, 6, 7 files >500MB each"
echo ""

while ! check_phase1_complete; do
    sleep 1800  # Check every 30 minutes

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Phase 1 still running..."

    for gpu in 4 5 6 7; do
        local patching_file="${PHASE1_DIR}/phase1_patching_multifeature_gpu${gpu}.jsonl"
        if [ -f "$patching_file" ]; then
            local size=$(stat -c%s "$patching_file" 2>/dev/null || echo "0")
            echo "  GPU $gpu: $(numfmt --to=iec $size)"
        else
            echo "  GPU $gpu: not started"
        fi
    done
    echo ""
done

echo "✅ Phase 1 complete!"
echo ""

# Phase 2
echo "=== Starting Phase 2: Correlation Analysis ==="
bash launch_phase2_REPARSED.sh
echo ""

# Phase 3
echo "=== Starting Phase 3: Causal Validation ==="
bash launch_phase3_REPARSED.sh
echo ""

# Phase 4
echo "=== Starting Phase 4: Word-Feature Correlation ==="
bash launch_phase4_REPARSED.sh
echo ""

# Phase 5
echo "=== Starting Phase 5: Prompt-Feature Correlation ==="
bash launch_phase5_REPARSED.sh
echo ""

echo "🎉 All phases complete!"
echo ""
echo "Results:"
echo "  Phase 1: ${PHASE1_DIR}"
echo "  Phase 2: /data/.../results/phase2_correlations_REPARSED"
echo "  Phase 3: /data/.../results/phase3_causal_REPARSED"
echo "  Phase 4: /data/.../results/phase4_word_feature_REPARSED"
echo "  Phase 5: /data/.../results/phase5_prompt_feature_REPARSED"
