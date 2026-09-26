#!/bin/bash

echo "=== Phase 4 Word Analysis Monitor ==="
echo "Date: $(date)"
echo ""

# Check running processes
echo "📊 Running Processes:"
ps aux | grep "phase4_patching_word_analysis.py" | grep -v grep | wc -l | awk '{print "  Active word analysis jobs: "$1}'
echo ""

# Check output files
echo "📁 Output Files:"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_words"
if [ -d "$OUTPUT_DIR" ]; then
    for gpu in 4 5 6 7; do
        file="${OUTPUT_DIR}/word_analysis_gpu${gpu}.json"
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            echo "  GPU $gpu: ✅ Complete ($size)"
        else
            echo "  GPU $gpu: ⏳ Not yet complete"
        fi
    done

    total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
    echo ""
    echo "  Total size: $total_size"
else
    echo "  Output directory not yet created"
fi
echo ""

# Check log files for errors
echo "⚠️  Recent Errors:"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"
has_errors=0
for gpu in 4 5 6 7; do
    logfile="${LOG_DIR}/phase4_gpu${gpu}.log"
    if [ -f "$logfile" ]; then
        errors=$(grep -i "error\|exception\|traceback" "$logfile" 2>/dev/null | tail -5)
        if [ -n "$errors" ]; then
            echo "  GPU $gpu:"
            echo "$errors" | sed 's/^/    /'
            has_errors=1
        fi
    fi
done

if [ "$has_errors" -eq 0 ]; then
    echo "  No errors detected ✅"
fi

echo ""
echo "==============================================="
