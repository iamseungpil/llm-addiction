#!/bin/bash

# Sequential launcher for ALL phases (Phase 1-5)
# Use this to automatically run Phase 2-5 after Phase 1 completes

echo "=== Sequential Phase Launcher (Full 2787 features) ==="
echo "This will monitor Phase 1 and auto-launch Phase 2-5 when ready"
echo ""

# Function to check if Phase 1 is complete
check_phase1_complete() {
    local all_complete=true

    for gpu in 5 6 7; do
        local patching_file="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full/phase1_patching_multifeature_gpu${gpu}.jsonl"

        # Check if file exists
        if [ ! -f "$patching_file" ]; then
            all_complete=false
            break
        fi

        # Check if file size is large enough (>1GB expected)
        local file_size=$(stat -c%s "$patching_file" 2>/dev/null || echo "0")
        if [ "$file_size" -lt 1000000000 ]; then
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
echo "⏳ Waiting for Phase 1 to complete..."
echo "   Expected time: ~117 hours (4.9 days)"
echo ""

while ! check_phase1_complete; do
    sleep 3600  # Check every hour

    # Show progress
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Phase 1 still running..."

    # Show file sizes
    for gpu in 5 6 7; do
        local patching_file="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full/phase1_patching_multifeature_gpu${gpu}.jsonl"
        if [ -f "$patching_file" ]; then
            local size=$(stat -c%s "$patching_file" 2>/dev/null || echo "0")
            echo "  GPU $gpu: $(numfmt --to=iec $size)"
        fi
    done
    echo ""
done

echo ""
echo "✅ Phase 1 COMPLETE! Starting Phase 2-5..."
echo ""

# Launch Phase 2
echo "=== Starting Phase 2: Correlation Analysis ==="
bash launch_phase2_full.sh
echo ""

# Launch Phase 3
echo "=== Starting Phase 3: Causal Direction Analysis ==="
bash launch_phase3_full.sh
echo ""

# Launch Phase 4
echo "=== Starting Phase 4: Word-Feature Correlation ==="
bash launch_phase4_full.sh
echo ""

# Launch Phase 5
echo "=== Starting Phase 5: Prompt-Feature Correlation ==="
bash launch_phase5_full.sh
echo ""

echo "🎉 ALL PHASES (1-5) COMPLETE!"
echo "Results in: /data/llm_addiction/experiment_pathway_token_analysis/results/"
