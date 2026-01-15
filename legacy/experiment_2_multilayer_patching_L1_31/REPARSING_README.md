# Response Log Reparsing and Analysis

This directory contains scripts for reparsing experiment 2 response logs with improved parsing logic and recalculating causal features.

## Background

During the original experiment, we discovered that the parsing logic may have affected the results:

- **Original parsing**: Uses `amounts[-1]` (last dollar amount found)
- **Improved parsing**: Uses first valid amount with more comprehensive patterns

Analysis of sample responses showed:
- **34.4% disagreement** between the two parsing methods
- Original parsing failed on many responses where improved parsing succeeded
- This may have led to **undercounting causal features**

## Files

### Scripts

1. **`reparse_response_logs.py`**
   - Loads all response logs (2 GB, 3.3M responses)
   - Applies improved parsing logic
   - Saves reparsed results
   - Generates parsing comparison statistics

2. **`analyze_reparsed_results.py`**
   - Loads reparsed responses
   - Recalculates causal effects using Chi-square and Mann-Whitney tests
   - Identifies causal features with p < 0.05
   - Saves causal feature results

3. **`compare_parsing_methods.py`**
   - Compares original vs reparsed causal feature identification
   - Generates detailed comparison report
   - Identifies which features differ between methods

4. **`run_reparsing_analysis.sh`**
   - Runs complete pipeline (all 3 steps)
   - Estimated time: 25-30 minutes

### Usage

#### Quick Start (Recommended)

Run the complete pipeline:

```bash
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31
bash run_reparsing_analysis.sh
```

This will:
1. Reparse all response logs (~10 min)
2. Analyze reparsed results (~10 min)
3. Compare with original results (~5 min)

#### Step by Step

If you want to run steps individually:

```bash
# Step 1: Reparse response logs
python3 reparse_response_logs.py

# Step 2: Analyze reparsed results
python3 analyze_reparsed_results.py

# Step 3: Compare results
python3 compare_parsing_methods.py --reparsed /data/.../reparsed_causal_features_*.json
```

## Output Files

All outputs are saved to: `/data/llm_addiction/experiment_2_multilayer_patching/reparsed/`

### From Step 1 (Reparsing)

- **`reparsed_responses_YYYYMMDD_HHMMSS.json`**
  - All responses with both original and reparsed bet amounts
  - Grouped by feature and condition
  - Format: `{feature: {condition: [trials]}}`

- **`reparsing_summary_YYYYMMDD_HHMMSS.json`**
  - Parsing statistics
  - Agreement/disagreement rates
  - Success rates for each method

### From Step 2 (Analysis)

- **`reparsed_all_features_YYYYMMDD_HHMMSS.json`**
  - Causal analysis results for all features
  - Includes p-values, effect sizes, interpretation

- **`reparsed_causal_features_YYYYMMDD_HHMMSS.json`**
  - Only features identified as causal (p < 0.05)
  - Sorted by effect size

### From Step 3 (Comparison)

- **`parsing_comparison_report_YYYYMMDD_HHMMSS.json`**
  - Machine-readable comparison data
  - Categorized disagreements

- **`parsing_comparison_report_YYYYMMDD_HHMMSS.txt`**
  - Human-readable report
  - Summary statistics
  - Top disagreements with p-values
  - Interpretation

## Expected Results

Based on preliminary analysis:

### Parsing Improvement

- Original valid parse rate: ~40%
- Improved valid parse rate: ~75%
- Additional responses successfully parsed: ~35%

### Causal Feature Changes

- More features may be identified as causal
- Effect sizes may change
- Statistical significance may improve

### Agreement Rate

Expected scenarios:

- **High agreement (>90%)**: Parsing differences have minimal impact
- **Moderate agreement (70-90%)**: Some differences, both should be reported
- **Low agreement (<70%)**: Significant impact, parsing choice matters

## Interpretation for Paper

Include both results in the paper:

1. **Primary results**: Original parsing (what was actually run)
2. **Sensitivity analysis**: Reparsed results (robustness check)

If agreement is high:
- "Results are robust to parsing method choice"

If agreement is moderate/low:
- "Parsing method affects feature identification"
- Report range of causal features identified
- Discuss which parsing is more accurate

## Data Integrity

**Important**: This reparsing does NOT affect the original experiment:

✅ Original response logs preserved
✅ Original checkpoints untouched
✅ Original results remain valid
✅ Reparsing is completely separate

You can always:
- Re-run reparsing with different logic
- Compare multiple parsing methods
- Use original results for publication

## Disk Space

Required disk space: ~3.5 GB

- Original logs: 2.0 GB (already exists)
- Reparsed results: ~1.5 GB (new)
- Reports: ~100 MB (new)

## Troubleshooting

### "No reparsed files found"

Run the scripts in order:
1. `reparse_response_logs.py` first
2. Then `analyze_reparsed_results.py`
3. Finally `compare_parsing_methods.py`

### Memory errors

The scripts process data in chunks and should handle 2 GB of logs fine. If you get memory errors:
- Close other programs
- Run on a machine with more RAM
- Process logs in batches (modify scripts)

### Missing dependencies

Required packages:
```bash
pip install numpy scipy tqdm
```

## Contact

For questions or issues with these scripts, check:
1. The text report for detailed comparison
2. Original response logs for raw data
3. Checkpoint files for original results

## Citation

If you use these reparsing scripts in your research, please note that:
- Original experiment used `amounts[-1]` parsing
- Sensitivity analysis used improved parsing with multiple patterns
- Both results should be reported for transparency
