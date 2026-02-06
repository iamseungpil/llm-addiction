#!/bin/bash

echo "=== Phase 2 Correlation Analysis Monitor ==="
echo "Date: $(date)"
echo ""

# Check running processes
echo "📊 Running Processes:"
ps aux | grep "phase2_patching_correlations.py" | grep -v grep | wc -l | awk '{print "  Active correlation jobs: "$1}'
echo ""

# Check output files
echo "📁 Output Files (by GPU):"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations"
if [ -d "$OUTPUT_DIR" ]; then
    for gpu in 4 5 6 7; do
        files=$(ls "$OUTPUT_DIR"/correlations_L*_*.jsonl 2>/dev/null | grep -c "")
        if [ "$files" -gt 0 ]; then
            # Count files per GPU by checking phase1 patching file
            gpu_files=$(find "$OUTPUT_DIR" -name "*.jsonl" -newer "/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching/phase1_patching_multifeature_gpu${gpu}.jsonl" 2>/dev/null | wc -l)
            echo "  GPU $gpu: $gpu_files correlation files completed"
        else
            echo "  GPU $gpu: No output yet"
        fi
    done

    total_files=$(ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo ""
    echo "  Total correlation files: $total_files / 800 expected"
else
    echo "  Output directory not yet created"
fi
echo ""

# Check log files for recent activity
echo "⚡ Recent Activity (last 5 log entries per GPU):"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"
for gpu in 4 5 6 7; do
    master_log="${LOG_DIR}/phase2_master_gpu${gpu}.log"
    if [ -f "$master_log" ]; then
        recent=$(tail -5 "$master_log" 2>/dev/null | grep "Processing:" | tail -3)
        if [ -n "$recent" ]; then
            echo "  GPU $gpu:"
            echo "$recent" | sed 's/^/    /'
        fi
    fi
done
echo ""

# Check for errors
echo "⚠️  Recent Errors:"
has_errors=0
for gpu in 4 5 6 7; do
    # Check individual correlation logs for errors
    error_count=$(find "$LOG_DIR" -name "phase2_L*_*.log" -type f -exec grep -l "Error\|Exception\|Traceback" {} \; 2>/dev/null | wc -l)
    if [ "$error_count" -gt 0 ]; then
        echo "  GPU $gpu: $error_count logs with errors"
        has_errors=1
    fi
done

if [ "$has_errors" -eq 0 ]; then
    echo "  No errors detected ✅"
fi
echo ""

# Estimate progress
echo "📈 Estimated Progress:"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations"
if [ -d "$OUTPUT_DIR" ]; then
    total_files=$(ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | wc -l)
    expected_total=800  # 200 features × 4 conditions
    progress=$(echo "scale=1; $total_files / $expected_total * 100" | bc 2>/dev/null || echo "0.0")
    echo "  Overall: ${progress}% ($total_files / $expected_total analyses)"

    # Estimate per GPU (50 features × 4 conditions = 200 each)
    for gpu in 4 5 6 7; do
        # Extract features from that GPU's Phase 1 output and count completed correlations
        gpu_expected=200
        # This is approximate - we'd need to parse actual feature lists for exact count
        echo "  GPU $gpu: ~25% expected share (50/200 features)"
    done
else
    echo "  Overall: 0.0% (0 / 800 analyses)"
fi

echo ""
echo "==============================================="
