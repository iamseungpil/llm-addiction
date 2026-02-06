#!/bin/bash
#
# Complete Reparsing and Analysis Pipeline
#
# This script runs the complete reparsing pipeline:
# 1. Reparse all response logs with improved parsing
# 2. Analyze reparsed results to identify causal features
# 3. Compare original vs reparsed results
#
# Usage:
#   bash run_reparsing_analysis.sh
#
# Estimated time: 25-30 minutes
#

set -e  # Exit on error

echo "================================================================================"
echo "REPARSING AND ANALYSIS PIPELINE"
echo "================================================================================"
echo ""
echo "This pipeline will:"
echo "  1. Reparse all response logs (2 GB, ~3.3M responses)"
echo "  2. Recalculate causal features with improved parsing"
echo "  3. Compare original vs reparsed results"
echo "  4. Classify features as SAFE or RISKY"
echo ""
echo "Estimated time: 25-30 minutes"
echo "================================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Reparse response logs
echo "STEP 1/3: Reparsing response logs..."
echo "--------------------------------------------------------------------------------"
python3 reparse_response_logs.py

if [ $? -ne 0 ]; then
    echo "❌ Reparsing failed!"
    exit 1
fi

echo ""
echo "✅ Step 1 complete"
echo ""

# Step 2: Analyze reparsed results
echo "STEP 2/3: Analyzing reparsed results..."
echo "--------------------------------------------------------------------------------"
python3 analyze_reparsed_results.py

if [ $? -ne 0 ]; then
    echo "❌ Analysis failed!"
    exit 1
fi

echo ""
echo "✅ Step 2 complete"
echo ""

# Step 3: Compare results
echo "STEP 3/4: Comparing original vs reparsed results..."
echo "--------------------------------------------------------------------------------"
python3 compare_parsing_methods.py

if [ $? -ne 0 ]; then
    echo "❌ Comparison failed!"
    exit 1
fi

echo ""
echo "✅ Step 3 complete"
echo ""

# Step 4: Classify features
echo "STEP 4/4: Classifying features as SAFE or RISKY..."
echo "--------------------------------------------------------------------------------"
python3 classify_safe_risky_features.py

if [ $? -ne 0 ]; then
    echo "❌ Classification failed!"
    exit 1
fi

echo ""
echo "✅ Step 4 complete"
echo ""

# Summary
echo "================================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: /data/llm_addiction/experiment_2_multilayer_patching/reparsed/"
echo ""
echo "Output files:"
echo "  - reparsed_responses_*.json (reparsed bet amounts)"
echo "  - reparsing_summary_*.json (parsing comparison stats)"
echo "  - reparsed_all_features_*.json (all feature results)"
echo "  - reparsed_causal_features_*.json (causal features only)"
echo "  - parsing_comparison_report_*.json (comparison data)"
echo "  - parsing_comparison_report_*.txt (human-readable report)"
echo "  - classified_features_*.json (all features with SAFE/RISKY labels)"
echo "  - safe_features_*.json (only safe features)"
echo "  - risky_features_*.json (only risky features)"
echo ""
echo "View the text report for detailed comparison:"
echo "  cat /data/llm_addiction/experiment_2_multilayer_patching/reparsed/parsing_comparison_report_*.txt | less"
echo ""
echo "================================================================================"
