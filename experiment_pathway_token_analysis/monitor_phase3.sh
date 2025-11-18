#!/bin/bash

echo "=== Phase 3 Causal Validation Monitor ==="
echo "Date: $(date)"
echo ""

# Check running processes
echo "📊 Running Processes:"
ps aux | grep "phase3_patching_causal_validation.py" | grep -v grep | wc -l | awk '{print "  Active validation jobs: "$1}'
echo ""

# Check output files
echo "📁 Output Files:"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase3_causal"
if [ -d "$OUTPUT_DIR" ]; then
    total_files=$(ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | wc -l)
    echo "  Total causal validation files: $total_files / 800 expected"

    # Check file sizes
    if [ "$total_files" -gt 0 ]; then
        total_size=$(du -sh "$OUTPUT_DIR" | awk '{print $1}')
        echo "  Total size: $total_size"

        # Sample file
        sample=$(ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | head -1)
        if [ -n "$sample" ]; then
            lines=$(wc -l < "$sample")
            echo "  Sample file: $(basename $sample)"
            echo "  Sample records: $lines"
        fi
    fi
else
    echo "  Output directory not yet created"
fi
echo ""

# Check log files for errors
echo "⚠️  Recent Errors:"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"
has_errors=0
if [ -d "$LOG_DIR" ]; then
    error_count=$(find "$LOG_DIR" -name "phase3_*.log" -type f -exec grep -l "Error\|Exception\|Traceback" {} \; 2>/dev/null | wc -l)
    if [ "$error_count" -gt 0 ]; then
        echo "  $error_count logs with errors"
        has_errors=1

        # Show first error
        first_error=$(find "$LOG_DIR" -name "phase3_*.log" -type f -exec grep -l "Error\|Exception\|Traceback" {} \; 2>/dev/null | head -1)
        if [ -n "$first_error" ]; then
            echo "  First error from: $(basename $first_error)"
            grep -A 3 "Error\|Exception\|Traceback" "$first_error" 2>/dev/null | head -5
        fi
    fi
fi

if [ "$has_errors" -eq 0 ]; then
    echo "  No errors detected ✅"
fi
echo ""

# Estimate progress
echo "📈 Estimated Progress:"
if [ -d "$OUTPUT_DIR" ]; then
    total_files=$(ls "$OUTPUT_DIR"/*.jsonl 2>/dev/null | wc -l)
    expected_total=800  # 200 features × 4 conditions
    progress=$(echo "scale=1; $total_files / $expected_total * 100" | bc 2>/dev/null || echo "0.0")
    echo "  Overall: ${progress}% ($total_files / $expected_total analyses)"
else
    echo "  Overall: 0.0% (0 / 800 analyses)"
fi

echo ""
echo "==============================================="
